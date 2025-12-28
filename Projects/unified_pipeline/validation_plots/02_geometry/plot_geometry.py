#!/usr/bin/env python
"""
Geometry Validation Plots - Comprehensive RF Analysis.

Generates detailed 3x3 visualizations of RF geometry with Gaussian and DoG fitting:
- Row 1: Peak frame, Diff map, Extreme map
- Row 2: Gaussian fit, DoG fit, ON/OFF comparison
- Row 3: Temporal profile, Summary statistics

Usage:
    python plot_geometry.py <hdf5_path> [--output-dir <path>] [--unit <unit_id>]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Ellipse, Circle
from tqdm import tqdm

# Add src to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# =============================================================================
# Configuration
# =============================================================================

# Analysis parameters
FRAME_RANGE = (40, 60)  # Frames to use for RF analysis
BASELINE_FRAMES = 10    # Frames 0-10 for baseline

# Spatial preprocessing
PADDING = 5             # Padding in pixels
GAUSSIAN_SIGMA = 1.5    # Blur sigma

# =============================================================================
# Data Loading
# =============================================================================

def load_unit_geometry(hdf5_path: Path, unit_id: str) -> Dict[str, Any]:
    """Load geometry data for a unit."""
    data = {}
    
    with h5py.File(hdf5_path, "r") as f:
        unit_path = f"units/{unit_id}"
        if unit_path not in f:
            return data
        
        # Load eimage_sta data
        sta_path = f"{unit_path}/features/eimage_sta/data"
        if sta_path in f:
            data["eimage_sta"] = f[sta_path][()]
        
        # Load eimage_sta/geometry (basic soma geometry)
        eimage_geom_path = f"{unit_path}/features/eimage_sta/geometry"
        if eimage_geom_path in f:
            geom = {}
            for key in f[eimage_geom_path]:
                val = f[f"{eimage_geom_path}/{key}"][()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="ignore")
                geom[key] = val
            data["eimage_sta_geometry"] = geom
        
        # Load sta_perfect_dense_noise data and its sta_geometry (full RF geometry)
        if f"{unit_path}/features" in f:
            for key in f[f"{unit_path}/features"]:
                if "sta_perfect" in key or "dense_noise" in key.lower():
                    # Load the STA data
                    noise_path = f"{unit_path}/features/{key}/data"
                    if noise_path in f:
                        data["sta_dense_noise"] = f[noise_path][()]
                        data["sta_dense_noise_name"] = key
                    
                    # Load the sta_geometry with Gaussian/DoG/ONOFF fits
                    sta_geom_path = f"{unit_path}/features/{key}/sta_geometry"
                    if sta_geom_path in f:
                        rf = {}
                        for gkey in f[sta_geom_path]:
                            item = f[f"{sta_geom_path}/{gkey}"]
                            if isinstance(item, h5py.Group):
                                # Handle subgroups like gaussian_fit, DoG, ONOFF_model
                                rf[gkey] = {}
                                for subkey in item:
                                    val = item[subkey][()]
                                    if isinstance(val, bytes):
                                        val = val.decode("utf-8", errors="ignore")
                                    rf[gkey][subkey] = val
                            else:
                                val = item[()]
                                if isinstance(val, bytes):
                                    val = val.decode("utf-8", errors="ignore")
                                rf[gkey] = val
                        data["rf_sta_geometry"] = rf
                        
                        # Load diff_map if stored
                        if "diff_map" in rf and isinstance(rf["diff_map"], np.ndarray):
                            data["diff_map"] = rf["diff_map"]
                    break
    
    return data


def get_unit_ids_with_geometry(hdf5_path: Path) -> list:
    """Get list of unit IDs that have geometry or STA data."""
    unit_ids = []
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            return []
        for unit_id in f["units"]:
            # Check for either geometry data or any STA data
            egeom = f"units/{unit_id}/features/eimage_sta_geometry"
            rfgeom = f"units/{unit_id}/features/rf_sta_geometry"
            esta = f"units/{unit_id}/features/eimage_sta/data"
            
            if egeom in f or rfgeom in f or esta in f:
                unit_ids.append(unit_id)
    return sorted(unit_ids)


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_sta(sta_data: np.ndarray) -> tuple:
    """
    Full preprocessing pipeline for STA data.
    
    Order:
    1. Baseline subtraction (frames 0-10)
    2. Add padding with zero value
    3. Apply 2D Gaussian blur
    
    Returns:
        Tuple of (preprocessed_data, padding_used)
    """
    # Handle NaN values
    data = np.nan_to_num(sta_data, nan=0.0).astype(np.float64)
    
    # Step 1: Baseline subtraction
    n_baseline = min(BASELINE_FRAMES, data.shape[0])
    baseline = np.nanmean(data[:n_baseline], axis=0, keepdims=True)
    data = data - baseline
    
    # Step 2: Add padding
    n_frames, rows, cols = data.shape
    padded = np.zeros((n_frames, rows + 2*PADDING, cols + 2*PADDING), dtype=data.dtype)
    padded[:, PADDING:PADDING+rows, PADDING:PADDING+cols] = data
    
    # Step 3: Apply 2D Gaussian blur
    for t in range(n_frames):
        padded[t] = gaussian_filter(padded[t], sigma=GAUSSIAN_SIGMA)
    
    return padded, PADDING


def find_center_extreme(data: np.ndarray, frame_range: tuple = None) -> tuple:
    """Find RF center using extreme absolute value method."""
    if frame_range is not None:
        start, end = frame_range
        start = max(0, start)
        end = min(data.shape[0], end)
        data_subset = data[start:end+1]
    else:
        data_subset = data
    
    # For each electrode, find the extreme value (max absolute)
    max_pos = np.nanmax(data_subset, axis=0)
    min_neg = np.nanmin(data_subset, axis=0)
    
    # Use the value with larger absolute magnitude
    extreme_map = np.where(np.abs(max_pos) > np.abs(min_neg), max_pos, min_neg)
    
    # Find center at maximum absolute value
    abs_extreme = np.abs(extreme_map)
    center_idx = np.nanargmax(abs_extreme)
    center_row, center_col = np.unravel_index(center_idx, extreme_map.shape)
    
    return int(center_row), int(center_col), extreme_map


def compute_diff_map(data: np.ndarray) -> np.ndarray:
    """Compute max - min difference map along time axis."""
    max_vals = np.nanmax(data, axis=0)
    min_vals = np.nanmin(data, axis=0)
    diff_map = max_vals - min_vals
    return np.nan_to_num(diff_map, nan=0.0)


# =============================================================================
# 2D Gaussian Functions for Fitted Overlays
# =============================================================================

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function with rotation."""
    x, y = xy
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    
    g = offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return g.ravel()


def dog_2d(xy, amp_exc, amp_inh, x0, y0, sigma_exc, sigma_inh, offset):
    """Difference of Gaussians (center-surround) function."""
    x, y = xy
    r_sq = (x - x0)**2 + (y - y0)**2
    
    exc = amp_exc * np.exp(-r_sq / (2 * sigma_exc**2))
    inh = amp_inh * np.exp(-r_sq / (2 * sigma_inh**2))
    
    dog = offset + exc - inh
    return dog.ravel()


# =============================================================================
# Comprehensive Plotting Function
# =============================================================================

def plot_unit_rf_geometry(
    unit_id: str,
    sta_data: np.ndarray,
    geometry: Dict[str, Any],
    rf_geometry: Dict[str, Any],
    output_path: Path,
    sta_name: str = "sta_perfect_dense_noise",
    stored_diff_map: Optional[np.ndarray] = None,
) -> None:
    """
    Plot RF geometry visualization for a single unit.
    
    Creates a 3x3 subplot with:
    - Row 1: Peak frame, Diff map, Extreme map
    - Row 2: Gaussian fit, DoG fit, ON/OFF comparison
    - Row 3: Temporal profile, Fit statistics (2 panels)
    
    Uses stored values from HDF5 when available.
    """
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    
    n_frames, rows, cols = sta_data.shape
    
    # Get geometry info from stored HDF5 values
    center_row = float(geometry.get('center_row', rows // 2))
    center_col = float(geometry.get('center_col', cols // 2))
    size_x = float(geometry.get('size_x', 5))
    size_y = float(geometry.get('size_y', 5))
    equiv_diam = float(geometry.get('equivalent_diameter', 5))
    peak_frame = int(geometry.get('peak_frame', n_frames // 2))
    
    # Use stored diff_map if available, otherwise compute for visualization
    if stored_diff_map is not None:
        diff_map_crop = stored_diff_map
    else:
        # Fallback: compute on the fly for visualization only
        preprocessed, pad = preprocess_sta(sta_data)
        diff_map = compute_diff_map(preprocessed)
        if pad > 0:
            diff_map_crop = diff_map[pad:-pad, pad:-pad] if diff_map.shape[0] > 2*pad else diff_map
        else:
            diff_map_crop = diff_map
    
    # Compute extreme map for visualization (not stored in HDF5)
    preprocessed, pad = preprocess_sta(sta_data)
    _, _, extreme_map = find_center_extreme(preprocessed, frame_range=FRAME_RANGE)
    if pad > 0:
        extreme_map_crop = extreme_map[pad:-pad, pad:-pad] if extreme_map.shape[0] > 2*pad else extreme_map
    else:
        extreme_map_crop = extreme_map
    
    peak_frame_idx = min(peak_frame, n_frames - 1)
    peak_frame_data = sta_data[peak_frame_idx]
    
    vmin = np.nanpercentile(peak_frame_data, 1)
    vmax = np.nanpercentile(peak_frame_data, 99)
    vmax_sym = max(abs(vmin), abs(vmax)) if vmax != vmin else 1
    
    # =========================================================================
    # Row 1, Col 1: Peak frame
    # =========================================================================
    ax = fig.add_subplot(3, 3, 1)
    im = ax.imshow(peak_frame_data, cmap='RdBu_r', vmin=-vmax_sym, vmax=vmax_sym, origin='lower')
    ax.plot(center_col, center_row, 'g+', markersize=15, markeredgewidth=3)
    
    if size_x > 0 and size_y > 0:
        ellipse = Ellipse(
            (center_col, center_row),
            width=size_x * 2, height=size_y * 2,
            fill=False, edgecolor='lime', linewidth=2, linestyle='--',
        )
        ax.add_patch(ellipse)
    
    ax.set_title(f'Peak Frame (t={peak_frame_idx})', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # =========================================================================
    # Row 1, Col 2: Diff map
    # =========================================================================
    ax = fig.add_subplot(3, 3, 2)
    im = ax.imshow(diff_map_crop, cmap='hot', origin='lower')
    ax.plot(center_col, center_row, 'c+', markersize=15, markeredgewidth=3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Max-Min Diff')
    ax.set_title('Difference Map', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # =========================================================================
    # Row 1, Col 3: Extreme map
    # =========================================================================
    ax = fig.add_subplot(3, 3, 3)
    ext_max = np.max(np.abs(extreme_map_crop)) if np.max(np.abs(extreme_map_crop)) > 0 else 1
    im = ax.imshow(extreme_map_crop, cmap='RdBu_r', vmin=-ext_max, vmax=ext_max, origin='lower')
    ax.plot(center_col, center_row, 'g+', markersize=15, markeredgewidth=3)
    ax.set_title('Extreme Value Map', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Extreme Value')
    
    # =========================================================================
    # Row 2, Col 1: Gaussian fit contours
    # =========================================================================
    ax = fig.add_subplot(3, 3, 4)
    im = ax.imshow(diff_map_crop, cmap='hot', origin='lower', alpha=0.7)
    
    # Check both key names: 'gaussian_fit' or older format
    gauss_fit = rf_geometry.get('gaussian_fit', rf_geometry.get('Gaussian', {}))
    if gauss_fit and gauss_fit.get('center_x') is not None:
        gf_cx = float(gauss_fit.get('center_x', center_col))
        gf_cy = float(gauss_fit.get('center_y', center_row))
        gf_sx = float(gauss_fit.get('sigma_x', 2))
        gf_sy = float(gauss_fit.get('sigma_y', 2))
        gf_theta = float(gauss_fit.get('theta', 0))
        gf_r2 = float(gauss_fit.get('r_squared', 0))
        gf_amp = float(gauss_fit.get('amplitude', 1))
        gf_offset = float(gauss_fit.get('offset', 0))
        
        # Draw Gaussian contours
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        try:
            fitted = gaussian_2d(
                (x_grid, y_grid), gf_amp, gf_cx, gf_cy,
                gf_sx, gf_sy, gf_theta, gf_offset
            ).reshape(rows, cols)
            
            level1 = gf_offset + gf_amp * np.exp(-0.5)
            level2 = gf_offset + gf_amp * np.exp(-2)
            levels = sorted([level1, level2])
            ax.contour(fitted, levels=levels, colors=['cyan', 'lime'], linewidths=2)
        except Exception:
            pass
        
        ax.plot(gf_cx, gf_cy, 'g*', markersize=12)
        ax.set_title(f'Gaussian Fit (R2={gf_r2:.3f})', fontsize=11, fontweight='bold')
    else:
        ax.set_title('Gaussian Fit (No Data)', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # =========================================================================
    # Row 2, Col 2: DoG fit
    # =========================================================================
    ax = fig.add_subplot(3, 3, 5)
    
    # Check both key names: 'dog_fit' or 'DoG'
    dog_fit = rf_geometry.get('dog_fit', rf_geometry.get('DoG', {}))
    if dog_fit and dog_fit.get('center_x') is not None:
        df_cx = float(dog_fit.get('center_x', center_col))
        df_cy = float(dog_fit.get('center_y', center_row))
        df_s_exc = float(dog_fit.get('sigma_exc', 2))
        df_s_inh = float(dog_fit.get('sigma_inh', 4))
        df_a_exc = float(dog_fit.get('amp_exc', 1))
        df_a_inh = float(dog_fit.get('amp_inh', 0.5))
        df_offset = float(dog_fit.get('offset', 0))
        df_r2 = float(dog_fit.get('r_squared', 0))
        
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        try:
            fitted_dog = dog_2d(
                (x_grid, y_grid), df_a_exc, df_a_inh, df_cx, df_cy,
                df_s_exc, df_s_inh, df_offset
            ).reshape(rows, cols)
            im = ax.imshow(fitted_dog, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, ax=ax, shrink=0.8)
        except Exception:
            ax.imshow(diff_map_crop, cmap='hot', origin='lower')
        
        ax.plot(df_cx, df_cy, 'g*', markersize=12)
        
        # Draw sigma circles
        exc_circle = Circle((df_cx, df_cy), df_s_exc, 
                            fill=False, edgecolor='lime', linewidth=2, label='Excitatory')
        inh_circle = Circle((df_cx, df_cy), df_s_inh,
                            fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Inhibitory')
        ax.add_patch(exc_circle)
        ax.add_patch(inh_circle)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'DoG Fit (R2={df_r2:.3f})', fontsize=11, fontweight='bold')
    else:
        ax.imshow(diff_map_crop, cmap='hot', origin='lower', alpha=0.5)
        ax.text(0.5, 0.5, 'DoG Fit: No Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('DoG Fit (No Data)', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # =========================================================================
    # Row 2, Col 3: ON/OFF comparison
    # =========================================================================
    ax = fig.add_subplot(3, 3, 6)
    im = ax.imshow(extreme_map_crop, cmap='RdBu_r', vmin=-ext_max, vmax=ext_max, 
                  origin='lower', alpha=0.6)
    
    # Check both key names: 'on_off_fit' or 'ONOFF_model'
    on_off_fit = rf_geometry.get('on_off_fit', rf_geometry.get('ONOFF_model', {}))
    if on_off_fit:
        on_amp = float(on_off_fit.get('on_amplitude', 0))
        off_amp = float(on_off_fit.get('off_amplitude', 0))
        
        if on_amp > 0:
            on_cx = float(on_off_fit.get('on_center_x', center_col))
            on_cy = float(on_off_fit.get('on_center_y', center_row))
            on_sx = float(on_off_fit.get('on_sigma_x', 2))
            on_r2 = float(on_off_fit.get('on_r_squared', 0))
            
            on_circle = Circle((on_cx, on_cy), max(on_sx, 1),
                              fill=False, edgecolor='red', linewidth=2, 
                              label=f'ON (s={on_sx:.1f})')
            ax.add_patch(on_circle)
            ax.plot(on_cx, on_cy, 'r+', markersize=12, markeredgewidth=2)
        
        if off_amp > 0:
            off_cx = float(on_off_fit.get('off_center_x', center_col))
            off_cy = float(on_off_fit.get('off_center_y', center_row))
            off_sx = float(on_off_fit.get('off_sigma_x', 2))
            off_r2 = float(on_off_fit.get('off_r_squared', 0))
            
            off_circle = Circle((off_cx, off_cy), max(off_sx, 1),
                               fill=False, edgecolor='blue', linewidth=2, 
                               label=f'OFF (s={off_sx:.1f})')
            ax.add_patch(off_circle)
            ax.plot(off_cx, off_cy, 'b+', markersize=12, markeredgewidth=2)
        
        on_r2_str = on_off_fit.get('on_r_squared', 0)
        off_r2_str = on_off_fit.get('off_r_squared', 0)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'ON/OFF (ON R2={float(on_r2_str):.2f}, OFF R2={float(off_r2_str):.2f})', 
                    fontsize=10, fontweight='bold')
    else:
        ax.set_title('ON/OFF Fit (No Data)', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(-0.5, rows-0.5)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # =========================================================================
    # Row 3, Col 1: Temporal profile
    # =========================================================================
    ax = fig.add_subplot(3, 3, 7)
    
    center_r = int(np.clip(center_row, 0, rows-1))
    center_c = int(np.clip(center_col, 0, cols-1))
    
    if 0 <= center_r < rows and 0 <= center_c < cols:
        center_signal = sta_data[:, center_r, center_c]
        frames = np.arange(n_frames)
        ax.plot(frames, center_signal, 'b-', linewidth=2, label='Center pixel')
        ax.axvspan(FRAME_RANGE[0], FRAME_RANGE[1], alpha=0.2, color='yellow', label='Analysis window')
        ax.axvspan(0, BASELINE_FRAMES, alpha=0.2, color='gray', label='Baseline (0-10)')
        ax.axvline(x=peak_frame_idx, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('STA Value')
    ax.set_title('Temporal Profile at RF Center', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 3, Col 2: Summary statistics
    # =========================================================================
    ax = fig.add_subplot(3, 3, 8)
    ax.axis('off')
    
    info_lines = [
        "RF GEOMETRY SUMMARY",
        "=" * 35,
        "",
        f"Center (x, y): ({center_col:.2f}, {center_row:.2f})",
        f"Size: {size_x:.2f} x {size_y:.2f} pixels",
        f"Area: {geometry.get('area', 0):.2f} pixels2",
        f"Equiv. Diameter: {equiv_diam:.2f} pixels",
        f"Peak Frame: {peak_frame}",
        "",
        "PREPROCESSING (order)",
        "-" * 35,
        f"1. Baseline: frames 0-{BASELINE_FRAMES}",
        f"2. Padding: {PADDING} pixels",
        f"3. Blur sigma: {GAUSSIAN_SIGMA}",
        "",
        "ANALYSIS WINDOW",
        "-" * 35,
        f"Frame range: {FRAME_RANGE}",
    ]
    
    ax.text(0.05, 0.95, '\n'.join(info_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # =========================================================================
    # Row 3, Col 3: Fit results
    # =========================================================================
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    
    fit_lines = ["FIT RESULTS", "=" * 35, ""]
    
    # Re-fetch fit data with both key naming conventions for summary
    gauss_fit_summary = rf_geometry.get('gaussian_fit', rf_geometry.get('Gaussian', {}))
    dog_fit_summary = rf_geometry.get('dog_fit', rf_geometry.get('DoG', {}))
    on_off_fit_summary = rf_geometry.get('on_off_fit', rf_geometry.get('ONOFF_model', {}))
    
    if gauss_fit_summary and gauss_fit_summary.get('center_x') is not None:
        fit_lines.extend([
            "2D GAUSSIAN FIT",
            f"  Center: ({gauss_fit_summary.get('center_x', 0):.2f}, {gauss_fit_summary.get('center_y', 0):.2f})",
            f"  sigma_x, sigma_y: {gauss_fit_summary.get('sigma_x', 0):.2f}, {gauss_fit_summary.get('sigma_y', 0):.2f}",
            f"  R2: {gauss_fit_summary.get('r_squared', 0):.4f}",
            "",
        ])
    else:
        fit_lines.extend(["2D GAUSSIAN FIT: No data", ""])
    
    if dog_fit_summary and dog_fit_summary.get('center_x') is not None:
        fit_lines.extend([
            "DoG (CENTER-SURROUND)",
            f"  sigma_exc: {dog_fit_summary.get('sigma_exc', 0):.2f}, sigma_inh: {dog_fit_summary.get('sigma_inh', 0):.2f}",
            f"  A_exc: {dog_fit_summary.get('amp_exc', 0):.3f}, A_inh: {dog_fit_summary.get('amp_inh', 0):.3f}",
            f"  R2: {dog_fit_summary.get('r_squared', 0):.4f}",
            "",
        ])
    else:
        fit_lines.extend(["DoG FIT: No data", ""])
    
    if on_off_fit_summary:
        fit_lines.extend([
            "ON/OFF MODEL",
            f"  ON: ({on_off_fit_summary.get('on_center_x', 0):.1f}, {on_off_fit_summary.get('on_center_y', 0):.1f}), "
            f"s={on_off_fit_summary.get('on_sigma_x', 0):.2f}",
            f"  OFF: ({on_off_fit_summary.get('off_center_x', 0):.1f}, {on_off_fit_summary.get('off_center_y', 0):.1f}), "
            f"s={on_off_fit_summary.get('off_sigma_x', 0):.2f}",
        ])
    else:
        fit_lines.extend(["ON/OFF MODEL: No data"])
    
    ax.text(0.05, 0.95, '\n'.join(fit_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # =========================================================================
    # Title and save
    # =========================================================================
    fig.suptitle(f'{unit_id} - RF Analysis (Gaussian + DoG Fitting)\n{sta_name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_unit_geometry(data: Dict[str, Any], unit_id: str, output_dir: Path):
    """Generate complete geometry validation plot for one unit."""
    
    # Prioritize dense noise STA (15x15) for RF geometry plots
    # since the Gaussian/DoG/ON-OFF fits are computed from this data
    sta_data = data.get("sta_dense_noise")
    sta_name = data.get("sta_dense_noise_name", "sta_perfect_dense_noise")
    
    # Fall back to eimage_sta if dense noise not available
    if sta_data is None:
        sta_data = data.get("eimage_sta")
        sta_name = "eimage_sta"
    
    if sta_data is None:
        # Cannot plot without data
        return None
    
    # Get geometry info - use rf_sta_geometry when using dense noise STA
    rf_geometry = data.get("rf_sta_geometry", {})
    
    # Use rf_sta_geometry for base geometry if available (for dense noise STA)
    if rf_geometry and sta_name != "eimage_sta":
        geometry = {
            'center_row': rf_geometry.get('center_row', sta_data.shape[1] // 2),
            'center_col': rf_geometry.get('center_col', sta_data.shape[2] // 2),
            'size_x': rf_geometry.get('size_x', 5),
            'size_y': rf_geometry.get('size_y', 5),
            'area': rf_geometry.get('area', 25),
            'equivalent_diameter': rf_geometry.get('equivalent_diameter', 5.6),
            'peak_frame': rf_geometry.get('peak_frame', sta_data.shape[0] // 2),
        }
    else:
        geometry = data.get("eimage_sta_geometry", {})
    
    # Fill in defaults if geometry is empty
    if not geometry:
        n_frames, rows, cols = sta_data.shape
        geometry = {
            'center_row': rows // 2,
            'center_col': cols // 2,
            'size_x': 5,
            'size_y': 5,
            'area': 25,
            'equivalent_diameter': 5.6,
            'peak_frame': n_frames // 2,
        }
    
    # Get stored diff_map if available
    stored_diff_map = data.get("diff_map")
    if stored_diff_map is None and rf_geometry:
        stored_diff_map = rf_geometry.get("diff_map")
    
    output_path = output_dir / f"{unit_id}_geometry.png"
    plot_unit_rf_geometry(unit_id, sta_data, geometry, rf_geometry, output_path, sta_name, stored_diff_map)
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def run_validation(
    hdf5_path: Path,
    output_dir: Optional[Path] = None,
    unit_id: Optional[str] = None,
):
    """Run geometry validation plots."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / hdf5_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get units to process
    if unit_id:
        unit_ids = [unit_id]
    else:
        unit_ids = get_unit_ids_with_geometry(hdf5_path)
    
    print(f"Processing {len(unit_ids)} units for geometry validation...")
    
    success = 0
    for uid in tqdm(unit_ids, desc="Geometry plots"):
        data = load_unit_geometry(hdf5_path, uid)
        if not data:
            continue
        
        try:
            result = plot_unit_geometry(data, uid, output_dir)
            if result:
                success += 1
        except Exception as e:
            print(f"  Error for {uid}: {e}")
    
    print(f"Generated {success}/{len(unit_ids)} plots in {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Geometry validation plots")
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--unit", type=str, help="Single unit ID to process")
    
    args = parser.parse_args()
    
    if not args.hdf5_path.exists():
        print(f"Error: File not found: {args.hdf5_path}")
        return 1
    
    run_validation(args.hdf5_path, args.output_dir, args.unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
