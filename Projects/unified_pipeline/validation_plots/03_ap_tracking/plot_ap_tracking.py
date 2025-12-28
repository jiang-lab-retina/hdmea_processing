#!/usr/bin/env python
"""
AP Tracking Validation Plots.

Generates visualizations of AP tracking results:
- Per-unit: STA with soma/AIS markers, prediction, centroids + pathway
- Summary: ONH intersection with all pathways, polar coordinate distribution

Usage:
    python plot_ap_tracking.py <hdf5_path> [--output-dir <path>] [--unit <unit_id>]
    
Example:
    python plot_ap_tracking.py ../test_output/2024.08.08-10.40.20-Rec_final.h5
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from tqdm import tqdm

# Add src to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# =============================================================================
# Styling
# =============================================================================

plt.style.use("dark_background")

NEURAL_CMAP = LinearSegmentedColormap.from_list(
    "neural",
    ["#0d1117", "#1a1f29", "#0e4429", "#006d32", "#26a641", "#39d353", "#7ee787"],
)

PRED_CMAP = LinearSegmentedColormap.from_list(
    "prediction",
    ["#0d1117", "#21262d", "#3d1a78", "#6e40c9", "#8b5cf6", "#a78bfa", "#c4b5fd"],
)

BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
SOMA_COLOR = "#ff6b6b"
AIS_COLOR = "#4ecdc4"
PATHWAY_COLOR = "#ffd93d"
ONH_COLOR = "#ff00ff"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnitAPData:
    """Container for AP tracking data."""
    unit_id: str
    cell_type: Optional[str] = None
    sta_data: Optional[np.ndarray] = None
    
    # Soma/AIS
    soma_t: Optional[int] = None
    soma_x: Optional[int] = None
    soma_y: Optional[int] = None
    ais_t: Optional[int] = None
    ais_x: Optional[int] = None
    ais_y: Optional[int] = None
    
    # Prediction
    prediction_data: Optional[np.ndarray] = None
    filtered_prediction: Optional[np.ndarray] = None
    axon_centroids: Optional[np.ndarray] = None
    
    # Pathway
    pathway_slope: Optional[float] = None
    pathway_intercept: Optional[float] = None
    pathway_r_value: Optional[float] = None
    
    # Polar coordinates
    polar_radius: Optional[float] = None
    polar_angle: Optional[float] = None


@dataclass
class RecordingData:
    """Container for recording-level data."""
    dataset_id: str = ""
    onh_x: Optional[float] = None
    onh_y: Optional[float] = None
    units: Dict[str, UnitAPData] = field(default_factory=dict)


# =============================================================================
# Data Loading
# =============================================================================

def load_recording_data(hdf5_path: Path) -> RecordingData:
    """Load all AP tracking data from HDF5."""
    recording = RecordingData(dataset_id=hdf5_path.stem)
    
    with h5py.File(hdf5_path, "r") as f:
        # Load ONH intersection from metadata
        onh_path = "metadata/ap_tracking/all_ap_intersection"
        if onh_path in f:
            onh_grp = f[onh_path]
            if "x" in onh_grp:
                recording.onh_x = float(onh_grp["x"][()])
            if "y" in onh_grp:
                recording.onh_y = float(onh_grp["y"][()])
        
        # Load units
        if "units" not in f:
            return recording
        
        for unit_id in f["units"]:
            unit_data = UnitAPData(unit_id=unit_id)
            unit_path = f"units/{unit_id}"
            
            # Cell type
            ct_path = f"{unit_path}/auto_label/axon_type"
            if ct_path in f:
                ct = f[ct_path][()]
                if isinstance(ct, bytes):
                    unit_data.cell_type = ct.decode("utf-8")
                elif isinstance(ct, np.ndarray):
                    unit_data.cell_type = str(ct.flat[0])
                else:
                    unit_data.cell_type = str(ct)
            
            # STA data
            sta_path = f"{unit_path}/features/eimage_sta/data"
            if sta_path in f:
                unit_data.sta_data = f[sta_path][()]
            
            # AP tracking
            ap_path = f"{unit_path}/features/ap_tracking"
            if ap_path not in f:
                recording.units[unit_id] = unit_data
                continue
            
            ap_grp = f[ap_path]
            
            # Refined soma
            if "refined_soma" in ap_grp:
                soma = ap_grp["refined_soma"]
                for key in ["t", "x", "y"]:
                    if key in soma:
                        val = soma[key][()]
                        if not (isinstance(val, float) and np.isnan(val)):
                            setattr(unit_data, f"soma_{key}", int(val))
            
            # AIS
            if "axon_initial_segment" in ap_grp:
                ais = ap_grp["axon_initial_segment"]
                for key in ["t", "x", "y"]:
                    if key in ais:
                        val = ais[key][()]
                        if not (isinstance(val, float) and np.isnan(val)):
                            setattr(unit_data, f"ais_{key}", int(val))
            
            # Prediction
            if "prediction_sta_data" in ap_grp:
                unit_data.prediction_data = ap_grp["prediction_sta_data"][()]
            
            if "post_processed_data" in ap_grp:
                pp = ap_grp["post_processed_data"]
                if "filtered_prediction" in pp:
                    unit_data.filtered_prediction = pp["filtered_prediction"][()]
                if "axon_centroids" in pp:
                    unit_data.axon_centroids = pp["axon_centroids"][()]
            
            # Pathway
            if "ap_pathway" in ap_grp:
                pw = ap_grp["ap_pathway"]
                for key in ["slope", "intercept", "r_value"]:
                    if key in pw:
                        val = pw[key][()]
                        if not (isinstance(val, float) and np.isnan(val)):
                            setattr(unit_data, f"pathway_{key}", float(val))
            
            # Polar
            if "soma_polar_coordinates" in ap_grp:
                polar = ap_grp["soma_polar_coordinates"]
                if "radius" in polar:
                    val = polar["radius"][()]
                    if not (isinstance(val, float) and np.isnan(val)):
                        unit_data.polar_radius = float(val)
                if "angle" in polar:
                    val = polar["angle"][()]
                    if not (isinstance(val, float) and np.isnan(val)):
                        unit_data.polar_angle = float(val)
            
            recording.units[unit_id] = unit_data
    
    return recording


# =============================================================================
# Per-Unit Plots
# =============================================================================

def plot_unit_ap_tracking(unit: UnitAPData, output_dir: Path, onh: Tuple[float, float] = None):
    """Generate AP tracking validation plot for one unit."""
    fig = plt.figure(figsize=(16, 5), facecolor=BG_COLOR)
    
    gs = fig.add_gridspec(1, 4, wspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
    
    # Panel 1: STA at soma timepoint with soma/AIS markers
    # NOTE: soma_x is ROW, soma_y is COLUMN (from RefinedSoma class)
    # For imshow with origin="lower": x-axis=column, y-axis=row
    # So we plot: x=soma_y (col), y=soma_x (row)
    ax = axes[0]
    if unit.sta_data is not None:
        # Use soma timepoint if available, otherwise max projection
        if unit.soma_t is not None and 0 <= unit.soma_t < unit.sta_data.shape[0]:
            sta_frame = unit.sta_data[unit.soma_t]
            title_suffix = f"(t={unit.soma_t})"
        else:
            sta_frame = np.nanmax(np.abs(unit.sta_data), axis=0)
            title_suffix = "(max)"
        
        # Use diverging colormap for signed data
        vmax = np.nanmax(np.abs(sta_frame))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(sta_frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        
        # Mark soma: x=col(soma_y), y=row(soma_x)
        if unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter([unit.soma_y], [unit.soma_x], marker="o", s=200, 
                      facecolors="none", edgecolors=SOMA_COLOR, linewidth=3, 
                      label=f"Soma (r={unit.soma_x}, c={unit.soma_y})")
            ax.scatter([unit.soma_y], [unit.soma_x], marker="+", s=100, 
                      c=SOMA_COLOR, linewidth=2)
        
        # Mark AIS: x=col(ais_y), y=row(ais_x)
        if unit.ais_x is not None and unit.ais_y is not None:
            ax.scatter([unit.ais_y], [unit.ais_x], marker="^", s=150,
                      facecolors="none", edgecolors=AIS_COLOR, linewidth=3,
                      label=f"AIS (r={unit.ais_x}, c={unit.ais_y})")
        
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"STA {title_suffix}", fontsize=10)
    else:
        ax.set_title("STA (no data)", fontsize=10)
    
    # Panel 2: Prediction max projection with soma overlay
    ax = axes[1]
    pred_data = unit.prediction_data if unit.prediction_data is not None else unit.filtered_prediction
    if pred_data is not None:
        pred_max = np.nanmax(pred_data, axis=0)
        ax.imshow(pred_max, cmap=PRED_CMAP, origin="lower")
        
        # Mark soma on prediction: x=col(soma_y), y=row(soma_x)
        if unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter([unit.soma_y], [unit.soma_x], marker="o", s=150, 
                      facecolors="none", edgecolors=SOMA_COLOR, linewidth=2)
            ax.scatter([unit.soma_y], [unit.soma_x], marker="+", s=80, 
                      c=SOMA_COLOR, linewidth=2)
    ax.set_title("CNN Prediction (max)", fontsize=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Panel 3: Centroids + pathway
    # Centroids are stored as [t, row, col] (shape N x 3)
    # Pathway fit uses x=col, y=row: row = slope * col + intercept
    # So we plot: x=col (centroids[:, 2]), y=row (centroids[:, 1])
    ax = axes[2]
    if unit.axon_centroids is not None and len(unit.axon_centroids) > 0:
        centroids = unit.axon_centroids
        
        # Color by time (first column is time)
        colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))
        
        # Plot: x=col (index 2), y=row (index 1)
        if centroids.shape[1] >= 3:
            ax.scatter(centroids[:, 2], centroids[:, 1], c=colors, s=30, alpha=0.8)
        else:
            # Fallback for 2D centroids [row, col]
            ax.scatter(centroids[:, 1], centroids[:, 0], c=colors, s=30, alpha=0.8)
        
        # Draw pathway line: y(row) = slope * x(col) + intercept
        if unit.pathway_slope is not None and unit.pathway_intercept is not None:
            x_range = np.array([0, 64])  # column values
            y_line = unit.pathway_slope * x_range + unit.pathway_intercept  # row values
            r2 = unit.pathway_r_value ** 2 if unit.pathway_r_value else 0
            ax.plot(x_range, y_line, color=PATHWAY_COLOR, linewidth=2, 
                   linestyle="--", label=f"R2={r2:.2f}")
        
        # Mark soma: x=col(soma_y), y=row(soma_x)
        if unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter([unit.soma_y], [unit.soma_x], marker="o", s=150,
                      facecolors=SOMA_COLOR, edgecolors="white", linewidth=2)
        
        ax.set_xlim(-5, 70)
        ax.set_ylim(-5, 70)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Centroids + Pathway", fontsize=10)
    ax.set_aspect("equal")
    
    # Panel 4: Summary text
    ax = axes[3]
    ax.axis("off")
    
    lines = [
        f"Unit: {unit.unit_id}",
        f"Cell Type: {unit.cell_type or 'N/A'}",
        "",
        "=== Soma ===",
        f"  T: {unit.soma_t}",
        f"  Row: {unit.soma_x}, Col: {unit.soma_y}",
        "",
        "=== AIS ===",
        f"  T: {unit.ais_t}",
        f"  Row: {unit.ais_x}, Col: {unit.ais_y}",
        "",
        "=== Pathway ===",
        f"  Slope: {unit.pathway_slope:.3f}" if unit.pathway_slope else "  Slope: N/A",
        f"  R: {unit.pathway_r_value:.3f}" if unit.pathway_r_value else "  R: N/A",
        "",
        "=== Polar ===",
        f"  Radius: {unit.polar_radius:.2f}" if unit.polar_radius else "  Radius: N/A",
        f"  Angle: {unit.polar_angle:.2f}" if unit.polar_angle else "  Angle: N/A",
    ]
    
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor=PANEL_COLOR, alpha=0.8))
    
    # Title
    fig.suptitle(f"{unit.unit_id} - AP Tracking Validation", fontsize=12, fontweight="bold")
    
    # Save
    output_path = output_dir / f"{unit.unit_id}_ap_tracking.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    
    return output_path


# =============================================================================
# Summary Plots (Enhanced from visualize_intersection.py)
# =============================================================================

def calculate_direction_from_centroids(
    centroids: np.ndarray,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Calculate AP propagation direction from temporal order of axon centroids.
    Returns (direction_angle, start_point) where start_point is (row, col).
    """
    if centroids is None or len(centroids) < 2:
        return None, None
    
    if centroids.ndim == 1:
        return None, None
    
    if centroids.shape[1] == 3:
        sorted_idx = np.argsort(centroids[:, 0])
        sorted_centroids = centroids[sorted_idx]
        row_coords = sorted_centroids[:, 1]
        col_coords = sorted_centroids[:, 2]
    elif centroids.shape[1] == 2:
        row_coords = centroids[:, 0]
        col_coords = centroids[:, 1]
    else:
        return None, None
    
    valid_mask = ~(np.isnan(row_coords) | np.isnan(col_coords))
    row_coords = row_coords[valid_mask]
    col_coords = col_coords[valid_mask]
    
    if len(row_coords) < 2:
        return None, None
    
    start_point = (float(row_coords[0]), float(col_coords[0]))
    
    n = len(row_coords)
    weights = np.arange(1, n + 1)
    
    row_mean = np.average(row_coords, weights=weights)
    col_mean = np.average(col_coords, weights=weights)
    
    d_row = row_mean - row_coords[0]
    d_col = col_mean - col_coords[0]
    
    if abs(d_row) < 1e-10 and abs(d_col) < 1e-10:
        d_row = row_coords[-1] - row_coords[0]
        d_col = col_coords[-1] - col_coords[0]
    
    if abs(d_row) < 1e-10 and abs(d_col) < 1e-10:
        return None, start_point
    
    angle = np.degrees(np.arctan2(d_row, d_col))
    if angle < 0:
        angle += 360
    
    return float(angle), start_point


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles."""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def weighted_circular_mean(angles: List[float], weights: List[float]) -> float:
    """Calculate weighted circular mean of angles."""
    if not angles or not weights:
        return 0.0
    
    angles_arr = np.array(angles)
    weights_arr = np.array(weights) / np.sum(weights)
    radians = np.radians(angles_arr)
    
    x = np.sum(weights_arr * np.cos(radians))
    y = np.sum(weights_arr * np.sin(radians))
    
    mean_deg = np.degrees(np.arctan2(y, x))
    if mean_deg < 0:
        mean_deg += 360
    
    return float(mean_deg)


@dataclass
class PathwayInfo:
    """Extended pathway info for visualization."""
    unit_id: str
    slope: float
    intercept: float
    r_value: float
    direction_angle: Optional[float] = None
    start_point: Optional[Tuple[float, float]] = None  # (row, col)
    direction_valid: bool = True
    centroids: Optional[np.ndarray] = None
    
    @property
    def r_squared(self) -> float:
        return self.r_value ** 2


def plot_rays_with_direction(
    ax: plt.Axes,
    pathways: Dict[str, PathwayInfo],
    onh: Optional[Tuple[float, float]],
    consensus_direction: Optional[float],
    r2_threshold: float = 0.8,
) -> None:
    """Plot rays with direction arrows extending to ONH."""
    from matplotlib.patches import Rectangle
    
    ax.set_facecolor(PANEL_COLOR)
    
    # Collect bounds
    all_x = [0, 64]
    all_y = [0, 64]
    
    if onh:
        all_x.append(onh[0])
        all_y.append(onh[1])
    
    margin = 15
    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin
    
    # MEA boundary
    ax.add_patch(Rectangle(
        (0, 0), 64, 64,
        fill=False, edgecolor="#8b949e", linestyle="--", linewidth=1.5,
        label="MEA boundary"
    ))
    
    valid_count = 0
    invalid_count = 0
    
    # Plot each ray
    for uid, pw in pathways.items():
        if pw.r_squared < r2_threshold:
            continue
        
        # Color based on direction validity
        if pw.direction_valid:
            color = "#39d353"  # Green
            alpha = 0.7
            linewidth = 0.8
            valid_count += 1
        else:
            color = "#f85149"  # Red
            alpha = 0.3
            linewidth = 0.5
            invalid_count += 1
        
        # Get start point (row, col)
        if pw.start_point is not None:
            start_row, start_col = pw.start_point
        else:
            start_col = 32
            start_row = pw.slope * start_col + pw.intercept
        
        # Calculate ray direction along the fitted line
        line_dcol = 1.0 / np.sqrt(1 + pw.slope**2)
        line_drow = pw.slope * line_dcol
        
        # Use direction_angle to determine sign
        if pw.direction_angle is not None:
            dir_rad = np.radians(pw.direction_angle)
            dir_dcol = np.cos(dir_rad)
            dir_drow = np.sin(dir_rad)
            
            dot = line_drow * dir_drow + line_dcol * dir_dcol
            if dot < 0:
                line_dcol = -line_dcol
                line_drow = -line_drow
        
        # Extend ray
        if onh is not None:
            dist_to_onh = np.sqrt((onh[0] - start_col)**2 + (onh[1] - start_row)**2)
            ray_length = dist_to_onh * 1.2
        else:
            ray_length = 100
        
        end_col = start_col + ray_length * line_dcol
        end_row = start_row + ray_length * line_drow
        
        # Draw ray (x=col, y=row)
        ax.plot([start_col, end_col], [start_row, end_row], 
               color=color, linewidth=linewidth, alpha=alpha, zorder=4)
        
        # Mark start point
        ax.scatter([start_col], [start_row], c=color, s=30, marker='o',
                  edgecolors='white', linewidths=0.3, alpha=alpha, zorder=5)
    
    # Plot ONH intersection
    if onh:
        ax.scatter([onh[0]], [onh[1]], 
                  c="#ff6b6b", s=400, marker="X", edgecolors="white",
                  linewidths=3, zorder=10, label=f"ONH ({onh[0]:.1f}, {onh[1]:.1f})")
    
    # Draw consensus direction indicator
    if consensus_direction is not None:
        compass_x, compass_y = x_min + 8, y_min + 8
        compass_len = 5
        dx = compass_len * np.cos(np.radians(consensus_direction))
        dy = compass_len * np.sin(np.radians(consensus_direction))
        
        ax.annotate('', xy=(compass_x + dx, compass_y + dy), 
                   xytext=(compass_x, compass_y),
                   arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=3))
        ax.text(compass_x, compass_y + 5, f"Consensus: {consensus_direction:.0f}deg",
               color='#58a6ff', fontsize=8, ha='center')
    
    # Legend entries
    ax.plot([], [], color="#39d353", linewidth=2, label=f"Valid ({valid_count})")
    if invalid_count > 0:
        ax.plot([], [], color="#f85149", linewidth=2, label=f"Invalid dir ({invalid_count})")
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Inverted for image coordinates
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Column", color="#c9d1d9")
    ax.set_ylabel("Row", color="#c9d1d9")
    ax.set_title(f"AP Pathway Rays", color="#c9d1d9", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.2, color="#8b949e")
    ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", 
             labelcolor="#c9d1d9", fontsize=7)


def plot_unit_centroids_summary(
    ax: plt.Axes,
    pathways: Dict[str, PathwayInfo],
    onh: Optional[Tuple[float, float]],
    r2_threshold: float = 0.8,
) -> None:
    """Plot centroids from each valid unit with different colors."""
    from matplotlib.patches import Rectangle
    
    ax.set_facecolor(PANEL_COLOR)
    
    # MEA boundary
    ax.add_patch(Rectangle(
        (0, 0), 64, 64,
        fill=True, facecolor="#21262d", edgecolor="#8b949e", 
        linestyle="-", linewidth=2, alpha=0.3, zorder=1,
    ))
    
    # Filter pathways
    valid_pathways = {
        k: v for k, v in pathways.items()
        if v.r_squared >= r2_threshold and v.direction_valid and v.centroids is not None
    }
    
    if not valid_pathways:
        ax.text(0.5, 0.5, "No valid units with centroids", 
               ha="center", va="center", color="#8b949e", fontsize=10,
               transform=ax.transAxes)
        ax.set_title("Unit Centroids", color="#c9d1d9", fontsize=11, fontweight="bold")
        return
    
    n_units = len(valid_pathways)
    cmap = plt.colormaps["tab20" if n_units <= 20 else "hsv"]
    colors = [cmap(i / max(1, n_units)) for i in range(n_units)]
    
    # Plot each unit's centroids
    for i, (unit_id, pw) in enumerate(valid_pathways.items()):
        centroids = pw.centroids
        if centroids is None or len(centroids) == 0:
            continue
        
        color = colors[i]
        
        if centroids.shape[1] == 3:
            sorted_idx = np.argsort(centroids[:, 0])
            sorted_centroids = centroids[sorted_idx]
            row_coords = sorted_centroids[:, 1]
            col_coords = sorted_centroids[:, 2]
            t_coords = sorted_centroids[:, 0]
        else:
            row_coords = centroids[:, 0]
            col_coords = centroids[:, 1]
            t_coords = np.arange(len(centroids))
        
        # Trajectory line (x=col, y=row)
        ax.plot(col_coords, row_coords, color=color, linewidth=1.5, alpha=0.7, zorder=3)
        
        # Centroids with temporal size
        sizes = 20 + (t_coords - t_coords.min()) / (t_coords.max() - t_coords.min() + 1e-6) * 40
        ax.scatter(col_coords, row_coords, c=[color], s=sizes, alpha=0.8, 
                  edgecolors='white', linewidths=0.5, zorder=4)
        
        # Start marker
        ax.scatter([col_coords[0]], [row_coords[0]], c=[color], s=100, marker='*',
                  edgecolors='white', linewidths=1, zorder=5)
    
    if onh:
        ax.scatter([onh[0]], [onh[1]], c="#ff6b6b", s=200, marker="X",
                  edgecolors="white", linewidths=3, zorder=10,
                  label=f"ONH ({onh[0]:.1f}, {onh[1]:.1f})")
    
    ax.set_xlim(-5, 70)
    ax.set_ylim(70, -5)  # Inverted
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Column", color="#c9d1d9")
    ax.set_ylabel("Row", color="#c9d1d9")
    ax.set_title(f"Unit Centroids ({n_units} units)", color="#c9d1d9", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    if onh:
        ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", 
                 labelcolor="#c9d1d9", fontsize=8)


def plot_direction_distribution(
    ax: plt.Axes,
    pathways: Dict[str, PathwayInfo],
    consensus_direction: Optional[float],
    r2_threshold: float = 0.8,
    tolerance: float = 45.0,
) -> Optional[plt.Axes]:
    """Polar plot showing direction distribution of pathways."""
    angles = []
    r2_values = []
    valid_mask = []
    
    for pw in pathways.values():
        if pw.r_squared >= r2_threshold and pw.direction_angle is not None:
            angles.append(np.radians(pw.direction_angle))
            r2_values.append(pw.r_squared)
            valid_mask.append(pw.direction_valid)
    
    if not angles:
        ax.set_facecolor(PANEL_COLOR)
        ax.text(0.5, 0.5, "No direction data", transform=ax.transAxes,
               ha='center', va='center', color='#c9d1d9')
        ax.axis('off')
        return None
    
    # Create polar subplot
    ax.clear()
    ax = ax.figure.add_subplot(2, 3, 2, projection='polar')
    ax.set_facecolor(PANEL_COLOR)
    
    angles_arr = np.array(angles)
    r2_arr = np.array(r2_values)
    valid_arr = np.array(valid_mask)
    
    colors = ['#39d353' if v else '#f85149' for v in valid_arr]
    sizes = r2_arr * 200 + 30
    
    ax.scatter(angles_arr, r2_arr, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Draw consensus direction
    if consensus_direction is not None:
        ax.axvline(np.radians(consensus_direction), color='#58a6ff', linewidth=2)
        
        # Draw tolerance wedge
        theta1 = np.radians(consensus_direction - tolerance)
        theta2 = np.radians(consensus_direction + tolerance)
        theta_range = np.linspace(theta1, theta2, 50)
        ax.fill_between(theta_range, 0, 1, alpha=0.2, color='#58a6ff')
    
    ax.set_ylim(0, 1)
    ax.set_title("Direction Distribution", color="#c9d1d9", fontsize=11, fontweight="bold", pad=15)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_r2_bar_chart(
    ax: plt.Axes,
    pathways: Dict[str, PathwayInfo],
    r2_threshold: float = 0.8,
) -> None:
    """Bar chart showing R² values with direction validity color-coding."""
    ax.set_facecolor(PANEL_COLOR)
    
    sorted_items = sorted(
        [(uid, pw) for uid, pw in pathways.items() if pw.direction_angle is not None],
        key=lambda x: x[1].r_squared,
        reverse=True
    )
    
    if not sorted_items:
        ax.text(0.5, 0.5, "No direction data", transform=ax.transAxes,
               ha='center', va='center', color='#c9d1d9')
        ax.axis('off')
        return
    
    unit_ids = [item[0] for item in sorted_items]
    r2_values = [item[1].r_squared for item in sorted_items]
    valid = [item[1].direction_valid for item in sorted_items]
    
    colors = []
    for r2, v in zip(r2_values, valid):
        if r2 >= r2_threshold:
            colors.append('#39d353' if v else '#f85149')
        else:
            colors.append('#484f58')
    
    y_pos = np.arange(len(unit_ids))
    ax.barh(y_pos, r2_values, color=colors, edgecolor="#30363d", linewidth=1)
    ax.axvline(x=r2_threshold, color="#ffa657", linestyle="--", linewidth=2)
    
    included_count = sum(1 for r2, v in zip(r2_values, valid) if r2 >= r2_threshold and v)
    total_above = sum(1 for r2 in r2_values if r2 >= r2_threshold)
    
    ax.text(0.95, 0.95, f"Valid: {included_count}/{total_above}",
            transform=ax.transAxes, ha="right", va="top",
            color="#39d353", fontsize=9, fontweight="bold")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(unit_ids, color="#c9d1d9", fontsize=7)
    ax.set_xlabel("R squared", color="#c9d1d9")
    ax.set_title("R squared Values", color="#c9d1d9", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def plot_polar_distribution(
    ax: plt.Axes,
    recording: RecordingData,
) -> None:
    """Polar plot of soma positions relative to ONH."""
    angles = []
    radii = []
    cell_types = []
    
    for unit in recording.units.values():
        if unit.polar_angle is not None and unit.polar_radius is not None:
            angles.append(unit.polar_angle)
            radii.append(unit.polar_radius)
            cell_types.append(unit.cell_type or "unknown")
    
    if not angles:
        ax.set_facecolor(PANEL_COLOR)
        ax.text(0.5, 0.5, "No polar data", transform=ax.transAxes,
               ha='center', va='center', color='#c9d1d9')
        return
    
    # Color by cell type
    unique_types = list(set(cell_types))
    type_colors = {t: plt.cm.Set2(i / len(unique_types)) for i, t in enumerate(unique_types)}
    colors = [type_colors[t] for t in cell_types]
    
    ax.scatter(angles, radii, c=colors, s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Add legend for cell types
    for ct, color in type_colors.items():
        count = sum(1 for t in cell_types if t == ct)
        ax.scatter([], [], c=[color], s=40, label=f"{ct} ({count})")
    
    ax.legend(loc="upper right", fontsize=7, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title("Soma Polar Distribution", color="#c9d1d9", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#8b949e")


def plot_summary_text(
    ax: plt.Axes,
    recording: RecordingData,
    pathways: Dict[str, PathwayInfo],
    consensus_direction: Optional[float],
    r2_threshold: float,
) -> None:
    """Summary statistics panel."""
    ax.set_facecolor(PANEL_COLOR)
    ax.axis("off")
    
    total = len(pathways)
    with_direction = sum(1 for pw in pathways.values() if pw.direction_angle is not None)
    above_r2 = sum(1 for pw in pathways.values() if pw.r_squared >= r2_threshold)
    valid_direction = sum(1 for pw in pathways.values() 
                         if pw.r_squared >= r2_threshold and pw.direction_valid)
    
    n_rgc = sum(1 for u in recording.units.values() 
                if u.cell_type and 'rgc' in u.cell_type.lower())
    n_with_polar = sum(1 for u in recording.units.values() 
                       if u.polar_radius is not None)
    
    summary = f"""
ONH INTERSECTION ANALYSIS
=============================================

File: {recording.dataset_id}

PATHWAY STATISTICS
---------------------------------------------
Total pathways:           {total}
With direction data:      {with_direction}
Above R2 threshold:       {above_r2}
Valid direction:          {valid_direction}

CELL STATISTICS
---------------------------------------------
Total units:              {len(recording.units)}
RGC units:                {n_rgc}
With polar coords:        {n_with_polar}
"""
    
    if recording.onh_x is not None and recording.onh_y is not None:
        dist = np.sqrt((recording.onh_x - 32)**2 + (recording.onh_y - 32)**2)
        summary += f"""
ONH RESULT
---------------------------------------------
Location: ({recording.onh_x:.2f}, {recording.onh_y:.2f})
Distance from center: {dist:.1f} px
"""
        if consensus_direction is not None:
            summary += f"Consensus direction: {consensus_direction:.1f} deg\n"
    
    ax.text(0.05, 0.95, summary, ha="left", va="top",
            color="#c9d1d9", fontsize=8, family="monospace",
            transform=ax.transAxes)


def plot_onh_summary(recording: RecordingData, output_dir: Path, direction_tolerance: float = 45.0):
    """Generate comprehensive ONH intersection summary plot."""
    r2_threshold = 0.8
    
    # Build pathway info
    pathways: Dict[str, PathwayInfo] = {}
    for unit_id, unit in recording.units.items():
        if unit.pathway_slope is not None and unit.pathway_r_value is not None:
            direction_angle, start_point = calculate_direction_from_centroids(unit.axon_centroids)
            
            # Fallback start point to soma
            if start_point is None and unit.soma_x is not None and unit.soma_y is not None:
                start_point = (unit.soma_x, unit.soma_y)
            
            pathways[unit_id] = PathwayInfo(
                unit_id=unit_id,
                slope=unit.pathway_slope,
                intercept=unit.pathway_intercept,
                r_value=unit.pathway_r_value,
                direction_angle=direction_angle,
                start_point=start_point,
                centroids=unit.axon_centroids,
            )
    
    # Calculate consensus direction
    angles = []
    weights = []
    for pw in pathways.values():
        if pw.r_squared >= r2_threshold and pw.direction_angle is not None:
            angles.append(pw.direction_angle)
            weights.append(pw.r_squared)
    consensus_direction = weighted_circular_mean(angles, weights) if angles else None
    
    # Mark direction validity
    for pw in pathways.values():
        if pw.direction_angle is not None and consensus_direction is not None:
            diff = angle_difference(pw.direction_angle, consensus_direction)
            pw.direction_valid = diff <= direction_tolerance
        else:
            pw.direction_valid = True
    
    onh = None
    if recording.onh_x is not None and recording.onh_y is not None:
        onh = (recording.onh_x, recording.onh_y)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12), facecolor=BG_COLOR)
    
    # Panel 1: Rays with direction
    ax1 = fig.add_subplot(2, 3, 1)
    plot_rays_with_direction(ax1, pathways, onh, consensus_direction, r2_threshold)
    
    # Panel 2: Direction distribution (polar)
    ax2 = fig.add_subplot(2, 3, 2)
    plot_direction_distribution(ax2, pathways, consensus_direction, r2_threshold, direction_tolerance)
    
    # Panel 3: R² values
    ax3 = fig.add_subplot(2, 3, 3)
    plot_r2_bar_chart(ax3, pathways, r2_threshold)
    
    # Panel 4: Unit centroids
    ax4 = fig.add_subplot(2, 3, 4)
    plot_unit_centroids_summary(ax4, pathways, onh, r2_threshold)
    
    # Panel 5: Polar distribution
    ax5 = fig.add_subplot(2, 3, 5, projection="polar")
    plot_polar_distribution(ax5, recording)
    
    # Panel 6: Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    plot_summary_text(ax6, recording, pathways, consensus_direction, r2_threshold)
    
    # Main title
    fig.suptitle(
        f"ONH Intersection Analysis - {recording.dataset_id}",
        color="#c9d1d9", fontsize=14, fontweight="bold", y=0.98
    )
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05, wspace=0.25, hspace=0.3)
    
    # Save
    output_path = output_dir / "summary_onh.png"
    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def run_validation(
    hdf5_path: Path,
    output_dir: Optional[Path] = None,
    unit_id: Optional[str] = None,
):
    """Run AP tracking validation plots."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / hdf5_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading AP tracking data...")
    recording = load_recording_data(hdf5_path)
    
    onh = None
    if recording.onh_x is not None and recording.onh_y is not None:
        onh = (recording.onh_x, recording.onh_y)
        print(f"  ONH position: ({recording.onh_x:.2f}, {recording.onh_y:.2f})")
    
    # Get units to process
    if unit_id:
        unit_ids = [unit_id]
    else:
        unit_ids = [uid for uid, u in recording.units.items() 
                   if u.sta_data is not None or u.soma_x is not None]
    
    print(f"Processing {len(unit_ids)} units for AP tracking validation...")
    
    # Generate per-unit plots
    success = 0
    for uid in tqdm(unit_ids, desc="AP tracking plots"):
        if uid not in recording.units:
            continue
        unit = recording.units[uid]
        
        try:
            plot_unit_ap_tracking(unit, output_dir, onh)
            success += 1
        except Exception as e:
            print(f"  Error for {uid}: {e}")
    
    # Generate summary plot
    print("Generating summary plot...")
    try:
        plot_onh_summary(recording, output_dir)
    except Exception as e:
        print(f"  Error generating summary: {e}")
    
    print(f"Generated {success}/{len(unit_ids)} unit plots + summary in {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="AP tracking validation plots")
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

