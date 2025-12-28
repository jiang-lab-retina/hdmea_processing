"""
AP-STA Soma Geometry Extraction Module

Extracts soma geometry (center location, size, axis lengths) from eimage_sta data
using the MaxMin method and saves results to HDF5.

Compatible with both:
- Session mode: Works with PipelineSession for deferred saving
- Immediate mode: Writes directly to HDF5 file

Author: Generated for experimental analysis
Date: 2024-12
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
from scipy.signal import savgol_filter

# Try to import PipelineSession for session mode
try:
    from hdmea.pipeline import PipelineSession
    HAS_SESSION = True
except ImportError:
    HAS_SESSION = False


# =============================================================================
# Configuration
# =============================================================================

# Test file path (same as sta_quantification.py)
HDF5_PATH = Path(__file__).parent.parent / "pipeline_test" / "data" / "2024.03.01-14.40.14-Rec.h5"
OUTPUT_DIR = Path(__file__).parent / "results"


@dataclass
class TemporalConfig:
    """Configuration for temporal preprocessing."""
    sigma_t: float = 2.0  # Temporal Gaussian smoothing sigma (frames)
    sigma_x: float = 3.0  # Spatial smoothing sigma in x
    sigma_y: float = 3.0  # Spatial smoothing sigma in y
    baseline_frames: int = 5  # Number of initial frames for baseline
    savgol_window: int = 7  # Savitzky-Golay window size
    savgol_order: int = 3   # Polynomial order


TEMPORAL_CONFIG = TemporalConfig()


@dataclass
class SomaGeometry:
    """
    Container for soma geometry results.
    
    Attributes:
        center_row: Row coordinate of soma center
        center_col: Column coordinate of soma center
        size_x: Width (short axis) in electrodes
        size_y: Height (long axis) in electrodes
        area: Area in electrodes^2
        equivalent_diameter: Equivalent circular diameter
        diff_map: 2D max-min difference map (optional, for visualization)
    """
    center_row: float
    center_col: float
    size_x: float
    size_y: float
    area: float
    equivalent_diameter: float
    diff_map: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HDF5 storage (excludes diff_map)."""
        return {
            'center_row': self.center_row,
            'center_col': self.center_col,
            'size_x': self.size_x,
            'size_y': self.size_y,
            'area': self.area,
            'equivalent_diameter': self.equivalent_diameter,
        }


# =============================================================================
# Preprocessing Utilities
# =============================================================================

def baseline_subtract(data: np.ndarray, n_baseline_frames: int = None) -> np.ndarray:
    """
    Subtract temporal baseline using mean of initial frames.
    
    Args:
        data: 3D array (time, rows, cols)
        n_baseline_frames: Number of frames for baseline
        
    Returns:
        Baseline-subtracted array
    """
    if n_baseline_frames is None:
        n_baseline_frames = TEMPORAL_CONFIG.baseline_frames
    
    n_baseline_frames = min(n_baseline_frames, data.shape[0])
    baseline = np.nanmean(data[:n_baseline_frames], axis=0, keepdims=True)
    return data - baseline


def temporal_smooth(data: np.ndarray, sigma_t: float = None) -> np.ndarray:
    """Apply 1D Gaussian smoothing along the temporal axis."""
    if sigma_t is None:
        sigma_t = TEMPORAL_CONFIG.sigma_t
    if sigma_t <= 0:
        return data.copy()
    return gaussian_filter1d(data, sigma=sigma_t, axis=0, mode='nearest')


def preprocess_temporal(
    eimage_sta: np.ndarray,
    smooth_temporal: bool = True,
    subtract_baseline: bool = True,
) -> np.ndarray:
    """
    Preprocess eimage_sta data for analysis.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        smooth_temporal: Apply temporal smoothing
        subtract_baseline: Subtract baseline from initial frames
        
    Returns:
        Preprocessed 3D array
    """
    # Handle NaN values
    data = np.nan_to_num(eimage_sta, nan=0.0).astype(np.float64)
    
    # Baseline subtraction
    if subtract_baseline:
        data = baseline_subtract(data, TEMPORAL_CONFIG.baseline_frames)
    
    # Temporal smoothing
    if smooth_temporal:
        data = temporal_smooth(data, TEMPORAL_CONFIG.sigma_t)
    
    return data


def robust_peak_trough_detection(
    signal: np.ndarray,
    window: int = None,
    order: int = None,
) -> Tuple[float, float, int, int]:
    """
    Detect peak and trough in a temporal signal using Savitzky-Golay filtering.
    
    Args:
        signal: 1D temporal signal
        window: Savitzky-Golay window size
        order: Savitzky-Golay polynomial order
        
    Returns:
        Tuple of (peak_value, trough_value, peak_idx, trough_idx)
    """
    if window is None:
        window = TEMPORAL_CONFIG.savgol_window
    if order is None:
        order = TEMPORAL_CONFIG.savgol_order
    
    # Ensure window is valid
    if len(signal) < window:
        window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    if window < order + 2:
        window = order + 2 if (order + 2) % 2 == 1 else order + 3
    
    # Apply Savitzky-Golay filter for smoothing
    smoothed = savgol_filter(signal, window_length=window, polyorder=order)
    
    peak_idx = int(np.argmax(smoothed))
    trough_idx = int(np.argmin(smoothed))
    
    return smoothed[peak_idx], smoothed[trough_idx], peak_idx, trough_idx


# =============================================================================
# Core Geometry Extraction Functions
# =============================================================================

def find_center_maxmin(eimage_sta: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Find cell center using max-min difference along time axis.
    
    Uses Savitzky-Golay filtering for robust peak/trough detection,
    considering temporal waveform shape rather than simple max/min.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (center_row, center_col, diff_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, subtract_baseline=True)
    
    # Compute robust peak-to-peak for each electrode
    diff_map = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            signal = data[:, r, c]
            peak_val, trough_val, _, _ = robust_peak_trough_detection(signal)
            diff_map[r, c] = peak_val - trough_val
    
    # Find location of maximum difference
    center_idx = np.nanargmax(diff_map)
    center_row, center_col = np.unravel_index(center_idx, diff_map.shape)
    
    return int(center_row), int(center_col), diff_map


def estimate_soma_size_maxmin(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Estimate soma size using max-min difference map.
    
    The soma should have high max-min difference.
    Size is estimated from the extent of high-activity region.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        threshold_fraction: Fraction of max for threshold
        
    Returns:
        Tuple of (size_x, size_y, area, equivalent_diameter, diff_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Compute max - min along time axis
    max_vals = np.nanmax(data_subset, axis=0)
    min_vals = np.nanmin(data_subset, axis=0)
    diff_map = max_vals - min_vals
    diff_map = np.nan_to_num(diff_map, nan=0.0)
    
    # Smooth
    diff_smooth = gaussian_filter(diff_map, sigma=1.0)
    
    # Threshold
    max_diff = np.max(diff_smooth)
    threshold = max_diff * threshold_fraction
    high_activity_mask = diff_smooth > threshold
    
    # Connected component containing center
    labeled, n_labels = label(high_activity_mask)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    center_label = labeled[center_row, center_col]
    
    if center_label > 0:
        soma_mask = labeled == center_label
    else:
        soma_mask = high_activity_mask
    
    area = float(np.sum(soma_mask))
    
    if area > 0:
        y_coords, x_coords = np.where(soma_mask)
        size_y = float(np.max(y_coords) - np.min(y_coords) + 1)
        size_x = float(np.max(x_coords) - np.min(x_coords) + 1)
    else:
        size_x = size_y = 0.0
    
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0.0
    
    return size_x, size_y, area, equivalent_diameter, diff_map


def extract_soma_geometry(
    eimage_sta: np.ndarray,
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
) -> SomaGeometry:
    """
    Extract soma geometry using the MaxMin method.
    
    This is the main function that combines center detection and size estimation.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        frame_range: Range of frames to use for size estimation (start, end inclusive)
        threshold_fraction: Fraction of max for threshold in size estimation
        
    Returns:
        SomaGeometry object with all geometry attributes
    """
    # Step 1: Find center using MaxMin method
    center_row, center_col, diff_map_full = find_center_maxmin(eimage_sta)
    
    # Step 2: Estimate size using specified frame range
    size_x, size_y, area, equiv_diam, diff_map = estimate_soma_size_maxmin(
        eimage_sta,
        center=(center_row, center_col),
        frame_range=frame_range,
        threshold_fraction=threshold_fraction,
    )
    
    return SomaGeometry(
        center_row=float(center_row),
        center_col=float(center_col),
        size_x=size_x,
        size_y=size_y,
        area=area,
        equivalent_diameter=equiv_diam,
        diff_map=diff_map,
    )


# =============================================================================
# Session-Compatible Pipeline Function
# =============================================================================

def extract_eimage_sta_geometry(
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
    session: Optional[Any] = None,
    hdf5_path: Optional[Path] = None,
) -> Optional[Any]:
    """
    Extract soma geometry for all units with eimage_sta data.
    
    Pipeline-compatible function following hdmea session workflow pattern.
    
    Works in two modes:
    - Session mode (deferred): Pass session=session, returns updated session
    - Immediate mode: Pass hdf5_path=path, writes directly to file
    
    Usage (Session mode - recommended):
        session = create_session(dataset_id="my_dataset")
        session = load_recording_with_eimage_sta(..., session=session)
        session = extract_eimage_sta_geometry(session=session)
        session.save()
    
    Usage (Immediate mode):
        extract_eimage_sta_geometry(hdf5_path="path/to/file.h5")
    
    Args:
        frame_range: Range of frames for size estimation (default: 10-14)
        threshold_fraction: Threshold fraction for size estimation (default: 0.5)
        session: PipelineSession object (for deferred mode)
        hdf5_path: Path to HDF5 file (for immediate mode)
        
    Returns:
        Updated session (in session mode) or None (in immediate mode)
    """
    if session is not None and hdf5_path is not None:
        raise ValueError("Provide either session or hdf5_path, not both")
    
    if session is None and hdf5_path is None:
        raise ValueError("Must provide either session or hdf5_path")
    
    # Session mode
    if session is not None:
        return _compute_geometry_session(session, frame_range, threshold_fraction)
    
    # Immediate mode
    else:
        _compute_geometry_hdf5(Path(hdf5_path), frame_range, threshold_fraction)
        return None


# Alias for backward compatibility
compute_eimage_sta_geometry = extract_eimage_sta_geometry


def _compute_geometry_session(
    session: Any,
    frame_range: Tuple[int, int],
    threshold_fraction: float,
) -> Any:
    """Compute geometry in session mode (deferred saving)."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # Fallback if tqdm not available
    
    processed_count = 0
    units_with_data = [(uid, ud) for uid, ud in session.units.items() 
                       if 'data' in ud.get('features', {}).get('eimage_sta', {})]
    
    for unit_id, unit_data in tqdm(units_with_data, desc="Extracting soma geometry"):
        # Get eimage_sta data
        features = unit_data.get('features', {})
        eimage_sta_data = features.get('eimage_sta', {})
        eimage_sta = eimage_sta_data['data']
        
        # Extract geometry
        geometry = extract_soma_geometry(eimage_sta, frame_range, threshold_fraction)
        
        # Store in session under features/eimage_sta/geometry/
        if 'features' not in session.units[unit_id]:
            session.units[unit_id]['features'] = {}
        if 'eimage_sta' not in session.units[unit_id]['features']:
            session.units[unit_id]['features']['eimage_sta'] = {}
        
        session.units[unit_id]['features']['eimage_sta']['geometry'] = {
            'center_row': geometry.center_row,
            'center_col': geometry.center_col,
            'size_x': geometry.size_x,
            'size_y': geometry.size_y,
            'area': geometry.area,
            'equivalent_diameter': geometry.equivalent_diameter,
            'diff_map': geometry.diff_map,
        }
        
        processed_count += 1
    
    # Mark step as completed
    session.completed_steps.add('eimage_sta_geometry')
    
    return session


def _compute_geometry_hdf5(
    hdf5_path: Path,
    frame_range: Tuple[int, int],
    threshold_fraction: float,
) -> Dict[str, SomaGeometry]:
    """Compute geometry in immediate mode (write directly to HDF5)."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # Fallback if tqdm not available
    
    all_geometries = {}
    
    with h5py.File(hdf5_path, 'r+') as f:
        if 'units' not in f:
            return all_geometries
        
        unit_ids = list(f['units'].keys())
        
        for unit_id in tqdm(unit_ids, desc="Extracting soma geometry"):
            unit_group = f[f'units/{unit_id}']
            
            # Check for eimage_sta data
            eimage_sta_path = 'features/eimage_sta/data'
            if eimage_sta_path not in unit_group:
                continue
            
            # Load eimage_sta
            eimage_sta = unit_group[eimage_sta_path][:]
            
            # Extract geometry
            geometry = extract_soma_geometry(eimage_sta, frame_range, threshold_fraction)
            all_geometries[unit_id] = geometry
            
            # Create geometry group if not exists
            geometry_path = 'features/eimage_sta/geometry'
            if geometry_path in unit_group:
                del unit_group[geometry_path]
            
            geom_group = unit_group.create_group(geometry_path)
            
            # Store geometry attributes
            geom_group.create_dataset('center_row', data=geometry.center_row)
            geom_group.create_dataset('center_col', data=geometry.center_col)
            geom_group.create_dataset('size_x', data=geometry.size_x)
            geom_group.create_dataset('size_y', data=geometry.size_y)
            geom_group.create_dataset('area', data=geometry.area)
            geom_group.create_dataset('equivalent_diameter', data=geometry.equivalent_diameter)
            geom_group.create_dataset('diff_map', data=geometry.diff_map, compression='gzip')
    
    return all_geometries


def load_geometries_from_hdf5(hdf5_path: Path) -> Dict[str, SomaGeometry]:
    """
    Load previously computed geometries from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        Dictionary mapping unit_id to SomaGeometry
    """
    geometries = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'units' not in f:
            return geometries
        
        for unit_id in f['units'].keys():
            geom_path = f'units/{unit_id}/features/eimage_sta/geometry'
            if geom_path not in f:
                continue
            
            geom_group = f[geom_path]
            
            # Load diff_map if available
            diff_map = None
            if 'diff_map' in geom_group:
                diff_map = geom_group['diff_map'][:]
            
            geometries[unit_id] = SomaGeometry(
                center_row=float(geom_group['center_row'][()]),
                center_col=float(geom_group['center_col'][()]),
                size_x=float(geom_group['size_x'][()]),
                size_y=float(geom_group['size_y'][()]),
                area=float(geom_group['area'][()]),
                equivalent_diameter=float(geom_group['equivalent_diameter'][()]),
                diff_map=diff_map,
            )
    
    return geometries


# =============================================================================
# Visualization Functions (Separate from Pipeline)
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_unit_geometry(
    unit_id: str,
    eimage_sta: np.ndarray,
    geometry: SomaGeometry,
    output_path: Path,
) -> None:
    """
    Plot soma geometry visualization for a single unit.
    
    Creates a 2x2 subplot with:
    - Top-left: Peak activity frame with soma contour
    - Top-right: Diff map with center marker
    - Bottom-left: Cross-section profiles through center
    - Bottom-right: Size annotation box
    
    Args:
        unit_id: Unit identifier
        eimage_sta: Original eimage_sta data (time, rows, cols)
        geometry: SomaGeometry object with extracted geometry
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    n_frames, rows, cols = eimage_sta.shape
    center_row = int(geometry.center_row)
    center_col = int(geometry.center_col)
    
    # Find peak frame
    frame_activity = np.nanmax(np.abs(eimage_sta), axis=(1, 2))
    peak_frame_idx = int(np.argmax(frame_activity))
    peak_frame = eimage_sta[peak_frame_idx]
    
    vmin = np.nanpercentile(peak_frame, 1)
    vmax = np.nanpercentile(peak_frame, 99)
    
    # --- Top-left: Peak frame with soma contour ---
    ax = axes[0, 0]
    im = ax.imshow(peak_frame, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # Draw ellipse representing soma size
    if geometry.size_x > 0 and geometry.size_y > 0:
        ellipse = Ellipse(
            (geometry.center_col, geometry.center_row),
            width=geometry.size_x,
            height=geometry.size_y,
            fill=False,
            edgecolor='lime',
            linewidth=2,
            linestyle='--',
        )
        ax.add_patch(ellipse)
    
    # Mark center
    ax.plot(geometry.center_col, geometry.center_row, 'g+', markersize=15, markeredgewidth=3)
    
    ax.set_title(f'Peak Frame (t={peak_frame_idx})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # --- Top-right: Diff map with center ---
    ax = axes[0, 1]
    if geometry.diff_map is not None:
        im = ax.imshow(geometry.diff_map, cmap='hot')
        ax.plot(geometry.center_col, geometry.center_row, 'c+', markersize=15, markeredgewidth=3)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Max-Min Diff')
    else:
        ax.text(0.5, 0.5, 'No diff map', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('Max-Min Difference Map', fontsize=11, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # --- Bottom-left: Cross-section profiles ---
    ax = axes[1, 0]
    
    # Horizontal and vertical profiles through center
    if geometry.diff_map is not None:
        h_profile = geometry.diff_map[center_row, :]
        v_profile = geometry.diff_map[:, center_col]
        
        ax.plot(range(cols), h_profile, 'b-', linewidth=2, label='Horizontal')
        ax.plot(range(rows), v_profile, 'r-', linewidth=2, label='Vertical')
        
        # Mark center position
        ax.axvline(x=center_col, color='b', linestyle='--', alpha=0.5)
        ax.axvline(x=center_row, color='r', linestyle='--', alpha=0.5)
        
        # Add size indicators
        half_x = geometry.size_x / 2
        half_y = geometry.size_y / 2
        ax.axvspan(center_col - half_x, center_col + half_x, alpha=0.2, color='blue')
        ax.axvspan(center_row - half_y, center_row + half_y, alpha=0.2, color='red')
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Max-Min Difference')
    ax.set_title('Cross-section Profiles', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Bottom-right: Size annotation ---
    ax = axes[1, 1]
    ax.axis('off')
    
    info_text = f"""
    Soma Geometry Summary
    {'='*30}
    
    Center Location:
      Row: {geometry.center_row:.2f}
      Col: {geometry.center_col:.2f}
    
    Size:
      Width (X):  {geometry.size_x:.2f} electrodes
      Height (Y): {geometry.size_y:.2f} electrodes
      
    Area: {geometry.area:.2f} electrodes²
    
    Equivalent Diameter: {geometry.equivalent_diameter:.2f}
    
    Long Axis:  {max(geometry.size_x, geometry.size_y):.2f}
    Short Axis: {min(geometry.size_x, geometry.size_y):.2f}
    Aspect Ratio: {max(geometry.size_x, geometry.size_y) / max(min(geometry.size_x, geometry.size_y), 0.1):.2f}
    """
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'{unit_id} - Soma Geometry (MaxMin Method)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {output_path.name}")


def plot_geometry_summary(
    all_geometries: Dict[str, SomaGeometry],
    output_path: Path,
) -> None:
    """
    Create summary plot comparing soma geometries across all units.
    
    Creates a 2x2 subplot with:
    - Top-left: Scatter plot of soma centers
    - Top-right: Histogram of equivalent diameters
    - Bottom-left: Size distribution (box plot)
    - Bottom-right: Summary table
    
    Args:
        all_geometries: Dict mapping unit_id to SomaGeometry
        output_path: Path to save the figure
    """
    if not all_geometries:
        print("No geometries to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    unit_ids = list(all_geometries.keys())
    centers_row = [g.center_row for g in all_geometries.values()]
    centers_col = [g.center_col for g in all_geometries.values()]
    sizes_x = [g.size_x for g in all_geometries.values()]
    sizes_y = [g.size_y for g in all_geometries.values()]
    areas = [g.area for g in all_geometries.values()]
    diameters = [g.equivalent_diameter for g in all_geometries.values()]
    
    # --- Top-left: Scatter plot of soma centers ---
    ax = axes[0, 0]
    scatter = ax.scatter(centers_col, centers_row, c=diameters, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    
    # Add unit labels
    for i, uid in enumerate(unit_ids):
        ax.annotate(uid, (centers_col[i], centers_row[i]), fontsize=8,
                   xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel('Column', fontsize=11)
    ax.set_ylabel('Row', fontsize=11)
    ax.set_title('Soma Center Locations', fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # Match image coordinates
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Equivalent Diameter')
    ax.grid(True, alpha=0.3)
    
    # --- Top-right: Histogram of equivalent diameters ---
    ax = axes[0, 1]
    ax.hist(diameters, bins=max(5, len(diameters)//2), color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(diameters), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(diameters):.2f}')
    ax.axvline(x=np.median(diameters), color='orange', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(diameters):.2f}')
    
    ax.set_xlabel('Equivalent Diameter (electrodes)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Soma Size Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom-left: Box plot of sizes ---
    ax = axes[1, 0]
    box_data = [sizes_x, sizes_y, diameters]
    bp = ax.boxplot(box_data, tick_labels=['Width (X)', 'Height (Y)', 'Eq. Diameter'],
                    patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Size (electrodes)', fontsize=11)
    ax.set_title('Size Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom-right: Summary table ---
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build table data
    table_data = [
        ['Unit ID', 'Center (r,c)', 'Size X', 'Size Y', 'Area', 'Ø'],
    ]
    
    for uid in unit_ids:
        g = all_geometries[uid]
        table_data.append([
            uid,
            f'({g.center_row:.1f}, {g.center_col:.1f})',
            f'{g.size_x:.1f}',
            f'{g.size_y:.1f}',
            f'{g.area:.1f}',
            f'{g.equivalent_diameter:.1f}',
        ])
    
    # Add summary row
    table_data.append([
        'Mean',
        '-',
        f'{np.mean(sizes_x):.1f}',
        f'{np.mean(sizes_y):.1f}',
        f'{np.mean(areas):.1f}',
        f'{np.mean(diameters):.1f}',
    ])
    table_data.append([
        'Std',
        '-',
        f'{np.std(sizes_x):.1f}',
        f'{np.std(sizes_y):.1f}',
        f'{np.std(areas):.1f}',
        f'{np.std(diameters):.1f}',
    ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.2, 0.12, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('lightgray')
        table[(0, j)].set_text_props(fontweight='bold')
    
    # Style summary rows
    for j in range(len(table_data[0])):
        table[(len(table_data)-2, j)].set_facecolor('lightyellow')
        table[(len(table_data)-1, j)].set_facecolor('lightyellow')
    
    ax.set_title('Geometry Summary Table', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle('Soma Geometry Summary (MaxMin Method)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved summary: {output_path.name}")


def plot_geometry_results(
    hdf5_path: Optional[Path] = None,
    geometries: Optional[Dict[str, SomaGeometry]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Generate all visualization plots for soma geometry results.
    
    Can work with either:
    - An HDF5 file path (loads geometries and eimage_sta from file)
    - Pre-computed geometries dictionary
    
    Args:
        hdf5_path: Path to HDF5 file with computed geometries
        geometries: Pre-computed geometries dictionary
        output_dir: Output directory for plots
    """
    if hdf5_path is None and geometries is None:
        raise ValueError("Must provide either hdf5_path or geometries")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load geometries from HDF5 if not provided
    if geometries is None:
        geometries = load_geometries_from_hdf5(Path(hdf5_path))
    
    if not geometries:
        print("No geometries found to plot")
        return
    
    print(f"\nGenerating visualization plots for {len(geometries)} units...")
    
    # Generate individual unit plots
    if hdf5_path is not None:
        with h5py.File(hdf5_path, 'r') as f:
            for unit_id, geometry in geometries.items():
                eimage_sta_path = f'units/{unit_id}/features/eimage_sta/data'
                if eimage_sta_path in f:
                    eimage_sta = f[eimage_sta_path][:]
                    output_path = output_dir / f'{unit_id}_geometry.png'
                    plot_unit_geometry(unit_id, eimage_sta, geometry, output_path)
    
    # Generate summary plot
    summary_path = output_dir / 'geometry_summary.png'
    plot_geometry_summary(geometries, summary_path)
    
    print(f"\nPlots saved to: {output_dir}")


# =============================================================================
# Main Test Function
# =============================================================================

if __name__ == "__main__":
    # Test with example HDF5 file
    print("=" * 60)
    print("AP-STA Soma Geometry Extraction")
    print("=" * 60)
    print(f"Input file: {HDF5_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    if not HDF5_PATH.exists():
        print(f"Error: HDF5 file not found: {HDF5_PATH}")
        exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Run geometry extraction in immediate mode
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 1: Computing Soma Geometry")
    print("-" * 60)
    
    geometries = _compute_geometry_hdf5(
        HDF5_PATH, 
        frame_range=(10, 14),  # Use frames 10-14 as specified
        threshold_fraction=0.5
    )
    
    # =========================================================================
    # Step 2: Generate visualization plots
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Generating Visualization Plots")
    print("-" * 60)
    
    plot_geometry_results(
        hdf5_path=HDF5_PATH,
        geometries=geometries,
        output_dir=OUTPUT_DIR,
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("GEOMETRY EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Units processed: {len(geometries)}")
    print(f"Frame range used: 10-14")
    print(f"\nOutput files:")
    print(f"  - Individual plots: {{unit_id}}_geometry.png")
    print(f"  - Summary plot: geometry_summary.png")
    print(f"\nResults saved to HDF5: units/{{unit_id}}/features/eimage_sta/geometry/")
    print(f"  - center_row, center_col (soma center)")
    print(f"  - size_x, size_y (axis sizes)")
    print(f"  - area, equivalent_diameter")
    print(f"  - diff_map (2D max-min difference)")
    print(f"\nOutput directory: {OUTPUT_DIR}")

