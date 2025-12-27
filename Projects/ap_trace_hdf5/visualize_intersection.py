#!/usr/bin/env python
"""
Optic Nerve Head (ONH) Intersection Visualization Script.

Generates comprehensive multi-panel figures visualizing the pre-computed
ONH intersection from AP pathway data stored in HDF5 files.

This script is visualization-only - all calculations are done in the
analysis code (src/hdmea/features/ap_tracking/pathway_analysis.py).

Usage:
    python visualize_intersection.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

# =============================================================================
# Configuration
# =============================================================================

# Input file
HDF5_FILE = Path(__file__).parent / "export" / "2024.03.25-13.44.04-Rec.h5"

# Output
OUTPUT_DIR = Path(__file__).parent / "plots" / "2024.03.25-13.44.04-Rec"

# Visualization parameters (for display only, not for recalculation)
R2_THRESHOLD = 0.8
DIRECTION_TOLERANCE = 45.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PathwayData:
    """Fitted line data for a single unit with direction information."""
    unit_id: str
    slope: float
    intercept: float
    r_value: float
    direction_angle: Optional[float] = None
    start_point: Optional[Tuple[float, float]] = None
    direction_valid: bool = True
    centroids: Optional[np.ndarray] = None
    
    @property
    def r_squared(self) -> float:
        return self.r_value ** 2


@dataclass
class ONHData:
    """Pre-computed ONH intersection data from HDF5."""
    x: float
    y: float
    mse: Optional[float] = None
    rmse: Optional[float] = None
    n_cluster_points: Optional[int] = None
    n_total_intersections: Optional[int] = None
    n_valid_after_direction: Optional[int] = None
    consensus_direction: Optional[float] = None
    r2_threshold: Optional[float] = None
    direction_tolerance: Optional[float] = None
    cluster_eps: Optional[float] = None
    cluster_min_samples: Optional[int] = None
    method: Optional[str] = None
    # Final valid intersection points (from main cluster)
    cluster_points: Optional[np.ndarray] = None


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_direction_from_centroids(
    centroids: np.ndarray,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Calculate AP propagation direction from temporal order of axon centroids.
    Used for visualization purposes.
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
    """Calculate weighted circular mean of angles for visualization."""
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


# =============================================================================
# Data Loading
# =============================================================================


def load_pathway_data(hdf5_path: Path) -> Tuple[Dict[str, PathwayData], Optional[ONHData]]:
    """
    Load pathway fit data, centroid direction, and pre-computed ONH from HDF5 file.
    
    Returns:
        Tuple of (pathways dict with direction info, ONHData or None)
    """
    pathways = {}
    onh_data = None
    
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            print("No 'units' group found in HDF5 file")
            return pathways, onh_data
        
        for unit_id in f["units"].keys():
            unit_path = f"units/{unit_id}"
            ap_path = f"{unit_path}/features/ap_tracking"
            
            if ap_path not in f:
                continue
            
            pathway_path = f"{ap_path}/ap_pathway"
            if pathway_path not in f:
                continue
            
            pw_grp = f[pathway_path]
            
            if "slope" in pw_grp and "intercept" in pw_grp and "r_value" in pw_grp:
                slope = float(pw_grp["slope"][()])
                intercept = float(pw_grp["intercept"][()])
                r_value = float(pw_grp["r_value"][()])
                
                # Check for NaN values
                if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
                    continue
                
                # Load centroid data for direction visualization
                direction_angle = None
                start_point = None
                centroids = None
                
                centroids_path = f"{ap_path}/post_processed_data/axon_centroids"
                if centroids_path in f:
                    centroids = f[centroids_path][:]
                    direction_angle, start_point = calculate_direction_from_centroids(centroids)
                
                # Fallback: use soma position as start point
                if start_point is None:
                    soma_path = f"{ap_path}/refined_soma"
                    if soma_path in f:
                        soma_grp = f[soma_path]
                        if "x" in soma_grp and "y" in soma_grp:
                            soma_x = soma_grp["x"][()]
                            soma_y = soma_grp["y"][()]
                            if soma_x is not None and not np.isnan(soma_x):
                                start_point = (float(soma_x), float(soma_y))
                
                pathways[unit_id] = PathwayData(
                    unit_id=unit_id,
                    slope=slope,
                    intercept=intercept,
                    r_value=r_value,
                    direction_angle=direction_angle,
                    start_point=start_point,
                    centroids=centroids,
                )
            
            # Load pre-computed ONH intersection (same for all units)
            if onh_data is None:
                int_path = f"{ap_path}/all_ap_intersection"
                if int_path in f:
                    int_grp = f[int_path]
                    if "x" in int_grp and "y" in int_grp:
                        x_val = int_grp["x"][()]
                        y_val = int_grp["y"][()]
                        
                        if not np.isnan(x_val) and not np.isnan(y_val):
                            # Read optional enhanced fields
                            onh_data = ONHData(
                                x=float(x_val),
                                y=float(y_val),
                                mse=_read_scalar(int_grp, "mse"),
                                rmse=_read_scalar(int_grp, "rmse"),
                                n_cluster_points=_read_scalar(int_grp, "n_cluster_points", int),
                                n_total_intersections=_read_scalar(int_grp, "n_total_intersections", int),
                                n_valid_after_direction=_read_scalar(int_grp, "n_valid_after_direction", int),
                                consensus_direction=_read_scalar(int_grp, "consensus_direction"),
                                r2_threshold=_read_scalar(int_grp, "r2_threshold"),
                                direction_tolerance=_read_scalar(int_grp, "direction_tolerance"),
                                cluster_eps=_read_scalar(int_grp, "cluster_eps"),
                                cluster_min_samples=_read_scalar(int_grp, "cluster_min_samples", int),
                                method=_read_string(int_grp, "method"),
                                cluster_points=int_grp["cluster_points"][:] if "cluster_points" in int_grp else None,
                            )
    
    return pathways, onh_data


def _read_scalar(group: h5py.Group, name: str, dtype=float) -> Optional:
    """Read a scalar dataset, returning None if not found or NaN."""
    if name not in group:
        return None
    val = group[name][()]
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return None
    return dtype(val) if val is not None else None


def _read_string(group: h5py.Group, name: str) -> Optional[str]:
    """Read a string dataset, returning None if not found."""
    if name not in group:
        return None
    val = group[name][()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val) if val is not None else None


def filter_pathways_for_display(
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float = 0.8,
    direction_tolerance: float = 45.0,
) -> Dict[str, PathwayData]:
    """
    Filter pathways for display based on R² and direction.
    
    Uses pre-computed consensus direction from ONH if available.
    """
    # Get consensus direction from ONH or calculate for display
    if onh_data and onh_data.consensus_direction is not None:
        consensus_direction = onh_data.consensus_direction
    else:
        # Calculate for visualization only
        angles = []
        weights = []
        for pw in pathways.values():
            if pw.r_squared >= r2_threshold and pw.direction_angle is not None:
                angles.append(pw.direction_angle)
                weights.append(pw.r_squared)
        consensus_direction = weighted_circular_mean(angles, weights) if angles else None
    
    # Mark direction validity for display
    for pw in pathways.values():
        if pw.direction_angle is not None and consensus_direction is not None:
            diff = angle_difference(pw.direction_angle, consensus_direction)
            pw.direction_valid = diff <= direction_tolerance
        else:
            pw.direction_valid = True
    
    return pathways


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_rays_with_direction(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float = 0.8,
) -> None:
    """Plot rays with direction arrows extending to ONH, with cluster points shown."""
    ax.set_facecolor("#161b22")
    
    # Get consensus direction and intersection for display
    consensus_direction = onh_data.consensus_direction if onh_data else None
    intersection = (onh_data.x, onh_data.y) if onh_data else None
    cluster_points = onh_data.cluster_points if onh_data else None
    
    # Collect bounds
    all_x = [0, 64]
    all_y = [0, 64]
    
    if intersection:
        all_x.append(intersection[0])
        all_y.append(intersection[1])
    
    if cluster_points is not None and len(cluster_points) > 0:
        all_x.extend(cluster_points[:, 0])
        all_y.extend(cluster_points[:, 1])
    
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
    
    # Plot cluster points (valid intersection points) first
    if cluster_points is not None and len(cluster_points) > 0:
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c="#ffa657", s=25, marker="o", alpha=0.5,
            edgecolors="none", zorder=3,
            label=f"Valid intersections ({len(cluster_points)})"
        )
    
    valid_count = 0
    # Plot each ray extending to all valid intersections
    for uid, pw in pathways.items():
        if pw.r_squared < r2_threshold:
            continue
        
        # Color based on direction validity
        if pw.direction_valid:
            color = "#39d353"
            alpha = 0.7
            linewidth = 0.8  # Thinner lines for clarity
            valid_count += 1
        else:
            color = "#f85149"
            alpha = 0.3
            linewidth = 0.5
        
        # Get start point (row, col)
        if pw.start_point is not None:
            start_row, start_col = pw.start_point
        else:
            start_col = 32
            start_row = pw.slope * start_col + pw.intercept
        
        # Calculate ray direction along the fitted line
        line_dcol = 1.0 / np.sqrt(1 + pw.slope**2)
        line_drow = pw.slope * line_dcol
        
        # Use direction_angle to determine sign (which direction along line)
        # Direction angle convention: atan2(d_row, d_col) -> 0° = +col, 90° = +row
        if pw.direction_angle is not None:
            dir_rad = np.radians(pw.direction_angle)
            dir_dcol = np.cos(dir_rad)  # 0° = +col direction
            dir_drow = np.sin(dir_rad)  # 90° = +row direction
            
            dot = line_drow * dir_drow + line_dcol * dir_dcol
            if dot < 0:
                line_dcol = -line_dcol
                line_drow = -line_drow
        
        # Extend ray to reach ALL valid intersections (not just ONH)
        # Calculate distance to farthest cluster point plus margin
        if cluster_points is not None and len(cluster_points) > 0 and pw.direction_valid:
            # Find max distance to any cluster point
            max_dist = 0
            for cp in cluster_points:
                dist = np.sqrt((cp[0] - start_col)**2 + (cp[1] - start_row)**2)
                max_dist = max(max_dist, dist)
            # Add 20% margin to extend past the farthest point
            ray_length = max_dist * 1.2
            # Minimum length to reach ONH
            if intersection is not None:
                dist_to_onh = np.sqrt(
                    (intersection[0] - start_col)**2 + (intersection[1] - start_row)**2
                )
                ray_length = max(ray_length, dist_to_onh * 1.2)
        elif intersection is not None:
            dist_to_onh = np.sqrt(
                (intersection[0] - start_col)**2 + (intersection[1] - start_row)**2
            )
            ray_length = dist_to_onh * 1.2
        else:
            ray_length = 100  # Long enough to see intersections
        
        end_col = start_col + ray_length * line_dcol
        end_row = start_row + ray_length * line_drow
        
        # Draw ray as thin line (no arrow for cleaner look)
        ax.plot([start_col, end_col], [start_row, end_row], 
               color=color, linewidth=linewidth, alpha=alpha, zorder=4)
        
        # Mark start point
        ax.scatter([start_col], [start_row], c=color, s=30, marker='o',
                  edgecolors='white', linewidths=0.3, alpha=alpha, zorder=5)
    
    # Plot ONH intersection
    if intersection:
        ax.scatter([intersection[0]], [intersection[1]], 
                  c="#ff6b6b", s=400, marker="X", edgecolors="white",
                  linewidths=3, zorder=10, label=f"ONH ({intersection[0]:.1f}, {intersection[1]:.1f})")
    
    # Draw consensus direction indicator
    if consensus_direction is not None:
        compass_x, compass_y = x_min + 8, y_min + 8
        compass_len = 5
        dx = compass_len * np.cos(np.radians(consensus_direction))
        dy = compass_len * np.sin(np.radians(consensus_direction))
        
        ax.annotate('', xy=(compass_x + dx, compass_y + dy), 
                   xytext=(compass_x, compass_y),
                   arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=3))
        ax.text(compass_x, compass_y + 5, f"Consensus: {consensus_direction:.0f} deg",
               color='#58a6ff', fontsize=9, ha='center')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Column", color="#c9d1d9")
    ax.set_ylabel("Row", color="#c9d1d9")
    ax.set_title(f"AP Pathway Rays ({valid_count} valid)", color="#c9d1d9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.2, color="#8b949e")
    ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", 
             labelcolor="#c9d1d9", fontsize=8)


def plot_direction_distribution(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float = 0.8,
    tolerance: float = 45.0,
) -> Optional[plt.Axes]:
    """Polar plot showing direction distribution of pathways."""
    ax.set_facecolor("#161b22")
    
    angles = []
    r2_values = []
    valid_mask = []
    
    for pw in pathways.values():
        if pw.r_squared >= r2_threshold and pw.direction_angle is not None:
            angles.append(np.radians(pw.direction_angle))
            r2_values.append(pw.r_squared)
            valid_mask.append(pw.direction_valid)
    
    if not angles:
        ax.text(0.5, 0.5, "No direction data", transform=ax.transAxes,
               ha='center', va='center', color='#c9d1d9')
        ax.axis('off')
        return None
    
    # Create polar subplot
    ax.clear()
    ax = ax.figure.add_subplot(2, 3, 2, projection='polar')
    ax.set_facecolor("#161b22")
    
    angles_arr = np.array(angles)
    r2_arr = np.array(r2_values)
    valid_arr = np.array(valid_mask)
    
    colors = ['#39d353' if v else '#f85149' for v in valid_arr]
    sizes = r2_arr * 200 + 30
    
    ax.scatter(angles_arr, r2_arr, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Draw consensus direction
    consensus_direction = onh_data.consensus_direction if onh_data else None
    if consensus_direction is not None:
        ax.axvline(np.radians(consensus_direction), color='#58a6ff', linewidth=2)
        
        # Draw tolerance wedge
        theta1 = np.radians(consensus_direction - tolerance)
        theta2 = np.radians(consensus_direction + tolerance)
        theta_range = np.linspace(theta1, theta2, 50)
        ax.fill_between(theta_range, 0, 1, alpha=0.2, color='#58a6ff')
    
    ax.set_ylim(0, 1)
    ax.set_title("Direction Distribution", color="#c9d1d9", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_r2_values(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    r2_threshold: float = 0.8,
) -> None:
    """Bar chart showing R² values with direction validity color-coding."""
    ax.set_facecolor("#161b22")
    
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
    
    ax.text(0.95, 0.95, f"Direction valid: {included_count}/{total_above}",
            transform=ax.transAxes, ha="right", va="top",
            color="#39d353", fontsize=10, fontweight="bold")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(unit_ids, color="#c9d1d9", fontsize=8)
    ax.set_xlabel("R squared", color="#c9d1d9")
    ax.set_title("R squared Values", color="#c9d1d9", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def plot_unit_centroids(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float = 0.8,
) -> None:
    """Plot centroids from each valid unit with different colors."""
    ax.set_facecolor("#161b22")
    
    # MEA boundary
    ax.add_patch(Rectangle(
        (0, 0), 64, 64,
        fill=True, facecolor="#21262d", edgecolor="#8b949e", 
        linestyle="-", linewidth=2, alpha=0.3, zorder=1,
    ))
    
    valid_pathways = {
        k: v for k, v in pathways.items()
        if v.r_squared >= r2_threshold and v.direction_valid and v.centroids is not None
    }
    
    if not valid_pathways:
        ax.text(0.5, 0.5, "No valid units with centroids", 
               ha="center", va="center", color="#8b949e", fontsize=12,
               transform=ax.transAxes)
        ax.set_title("Unit Centroids", color="#c9d1d9", fontsize=12, fontweight="bold")
        return
    
    n_units = len(valid_pathways)
    cmap = plt.colormaps["tab20" if n_units <= 20 else "hsv"]
    colors = [cmap(i / n_units) for i in range(n_units)]
    
    for i, (unit_id, pw) in enumerate(valid_pathways.items()):
        centroids = pw.centroids
        if centroids is None or len(centroids) == 0:
            continue
        
        color = colors[i]
        
        if centroids.shape[1] == 3:
            sorted_idx = np.argsort(centroids[:, 0])
            sorted_centroids = centroids[sorted_idx]
            x_coords = sorted_centroids[:, 1]
            y_coords = sorted_centroids[:, 2]
            t_coords = sorted_centroids[:, 0]
        else:
            x_coords = centroids[:, 0]
            y_coords = centroids[:, 1]
            t_coords = np.arange(len(centroids))
        
        ax.plot(y_coords, x_coords, color=color, linewidth=1.5, alpha=0.7, zorder=3)
        sizes = 20 + (t_coords - t_coords.min()) / (t_coords.max() - t_coords.min() + 1e-6) * 40
        ax.scatter(y_coords, x_coords, c=[color], s=sizes, alpha=0.8, 
                  edgecolors='white', linewidths=0.5, zorder=4)
        ax.scatter([y_coords[0]], [x_coords[0]], c=[color], s=100, marker='*',
                  edgecolors='white', linewidths=1, zorder=5)
    
    if onh_data:
        ax.scatter([onh_data.x], [onh_data.y], c="#ff6b6b", s=200, marker="X",
                  edgecolors="white", linewidths=3, zorder=10,
                  label=f"ONH ({onh_data.x:.1f}, {onh_data.y:.1f})")
    
    ax.set_xlim(-5, 70)
    ax.set_ylim(70, -5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Column", color="#c9d1d9")
    ax.set_ylabel("Row", color="#c9d1d9")
    ax.set_title(f"Unit Centroids ({len(valid_pathways)} units)", color="#c9d1d9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    if onh_data:
        ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", 
                 labelcolor="#c9d1d9", fontsize=9)


def plot_onh_zoomed(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float = 0.8,
) -> None:
    """Plot zoomed view of ONH location with cluster points."""
    ax.set_facecolor("#161b22")
    
    if onh_data is None:
        ax.text(0.5, 0.5, "No ONH data", ha="center", va="center",
               color="#8b949e", fontsize=12, transform=ax.transAxes)
        ax.set_title("ONH Location", color="#c9d1d9", fontsize=12, fontweight="bold")
        return
    
    # Determine bounds - include cluster points
    x_min, x_max = 0, 64
    y_min, y_max = 0, 64
    
    x_min = min(x_min, onh_data.x)
    x_max = max(x_max, onh_data.x)
    y_min = min(y_min, onh_data.y)
    y_max = max(y_max, onh_data.y)
    
    if onh_data.cluster_points is not None and len(onh_data.cluster_points) > 0:
        x_min = min(x_min, onh_data.cluster_points[:, 0].min())
        x_max = max(x_max, onh_data.cluster_points[:, 0].max())
        y_min = min(y_min, onh_data.cluster_points[:, 1].min())
        y_max = max(y_max, onh_data.cluster_points[:, 1].max())
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.1
    
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    # MEA boundary
    ax.add_patch(Rectangle(
        (0, 0), 64, 64,
        fill=True, facecolor="#21262d", edgecolor="#8b949e", 
        linestyle="-", linewidth=2, alpha=0.5, zorder=1,
    ))
    
    # Plot cluster points (valid intersection points)
    if onh_data.cluster_points is not None and len(onh_data.cluster_points) > 0:
        n_points = len(onh_data.cluster_points)
        ax.scatter(
            onh_data.cluster_points[:, 0],
            onh_data.cluster_points[:, 1],
            c="#ffa657", s=40, marker="o", alpha=0.6,
            edgecolors="white", linewidths=0.5, zorder=5,
            label=f"Cluster points ({n_points})"
        )
    
    # ONH point (weighted mean of cluster)
    ax.scatter([onh_data.x], [onh_data.y], c="#ff6b6b", s=300, marker="X",
              edgecolors="white", linewidths=3, zorder=10,
              label=f"ONH ({onh_data.x:.1f}, {onh_data.y:.1f})")
    
    # Crosshairs
    ax.axhline(y=onh_data.y, color="#ff6b6b", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=onh_data.x, color="#ff6b6b", linestyle=":", linewidth=1, alpha=0.5)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Column", color="#c9d1d9")
    ax.set_ylabel("Row", color="#c9d1d9")
    ax.set_title("ONH with Cluster Points", color="#c9d1d9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", 
             labelcolor="#c9d1d9", fontsize=8)


def plot_summary(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float,
    direction_tolerance: float,
    hdf5_path: Path,
) -> None:
    """Summary statistics panel."""
    ax.set_facecolor("#161b22")
    ax.axis("off")
    
    total = len(pathways)
    with_direction = sum(1 for pw in pathways.values() if pw.direction_angle is not None)
    above_r2 = sum(1 for pw in pathways.values() if pw.r_squared >= r2_threshold)
    valid_direction = sum(1 for pw in pathways.values() 
                         if pw.r_squared >= r2_threshold and pw.direction_valid)
    
    summary = f"""
    ONH INTERSECTION ANALYSIS (Pre-computed)
    =============================================
    
    File: {hdf5_path.name}
    
    PATHWAY STATISTICS
    ---------------------------------------------
    Total pathways:           {total}
    With direction data:      {with_direction}
    Above R2 threshold:       {above_r2}
    Valid direction:          {valid_direction}
    """
    
    if onh_data:
        summary += f"""
    ONH RESULT
    ---------------------------------------------
    Location: ({onh_data.x:.2f}, {onh_data.y:.2f})
    Method: {onh_data.method or 'unknown'}
    """
        if onh_data.rmse is not None:
            summary += f"    RMSE: {onh_data.rmse:.4f}\n"
        if onh_data.n_cluster_points is not None:
            summary += f"    Cluster points: {onh_data.n_cluster_points}\n"
        if onh_data.n_total_intersections is not None:
            summary += f"    Total intersections: {onh_data.n_total_intersections}\n"
        if onh_data.consensus_direction is not None:
            summary += f"    Consensus direction: {onh_data.consensus_direction:.1f} deg\n"
    
    ax.text(0.05, 0.95, summary, ha="left", va="top",
            color="#c9d1d9", fontsize=9, family="monospace",
            transform=ax.transAxes)


def plot_unit_legend(
    ax: plt.Axes,
    pathways: Dict[str, PathwayData],
    r2_threshold: float = 0.8,
) -> None:
    """Show legend/color key for unit centroids."""
    ax.set_facecolor("#161b22")
    ax.axis("off")
    
    valid_pathways = {
        k: v for k, v in pathways.items()
        if v.r_squared >= r2_threshold and v.direction_valid and v.centroids is not None
    }
    
    if not valid_pathways:
        ax.text(0.5, 0.5, "No valid units", 
               ha="center", va="center", color="#8b949e", fontsize=12,
               transform=ax.transAxes)
        return
    
    n_units = len(valid_pathways)
    cmap = plt.colormaps["tab20" if n_units <= 20 else "hsv"]
    colors = [cmap(i / n_units) for i in range(n_units)]
    
    y_pos = 0.95
    ax.text(0.05, y_pos, "UNIT COLORS", ha="left", va="top",
           color="#c9d1d9", fontsize=11, fontweight="bold",
           transform=ax.transAxes)
    y_pos -= 0.06
    
    ax.text(0.05, y_pos, "-" * 30, ha="left", va="top",
           color="#484f58", fontsize=9, family="monospace",
           transform=ax.transAxes)
    y_pos -= 0.05
    
    for i, (unit_id, pw) in enumerate(valid_pathways.items()):
        if y_pos < 0.05:
            ax.text(0.05, y_pos, f"... and {n_units - i} more",
                   color="#8b949e", fontsize=9, transform=ax.transAxes)
            break
        
        color = colors[i]
        n_centroids = len(pw.centroids) if pw.centroids is not None else 0
        r2 = pw.r_squared
        
        ax.scatter([0.08], [y_pos], c=[color], s=80, marker='s',
                  transform=ax.transAxes, zorder=10)
        
        info = f"{unit_id}: R2={r2:.2f}, {n_centroids} pts"
        ax.text(0.15, y_pos, info, ha="left", va="center",
               color="#c9d1d9", fontsize=8, family="monospace",
               transform=ax.transAxes)
        
        y_pos -= 0.045
    
    ax.set_title("Unit Legend", color="#c9d1d9", fontsize=12, fontweight="bold")


def create_individual_unit_plots(
    pathways: Dict[str, PathwayData],
    onh_data: Optional[ONHData],
    r2_threshold: float,
    output_dir: Path,
    hdf5_path: Path,
) -> Optional[Path]:
    """Create a figure with individual subplots for each valid unit."""
    valid_pathways = {
        k: v for k, v in pathways.items()
        if v.r_squared >= r2_threshold and v.direction_valid and v.centroids is not None
    }
    
    if not valid_pathways:
        print("No valid units for individual plots")
        return None
    
    n_units = len(valid_pathways)
    print(f"Creating individual plots for {n_units} units...")
    
    n_cols = min(5, n_units)
    n_rows = (n_units + n_cols - 1) // n_cols
    
    fig_width = n_cols * 4
    fig_height = n_rows * 4 + 0.8
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#0d1117")
    
    cmap = plt.colormaps["tab20" if n_units <= 20 else "hsv"]
    colors = [cmap(i / n_units) for i in range(n_units)]
    
    optimal = (onh_data.x, onh_data.y) if onh_data else None
    
    for i, (unit_id, pw) in enumerate(valid_pathways.items()):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_facecolor("#161b22")
        
        # MEA boundary
        ax.add_patch(Rectangle(
            (0, 0), 64, 64,
            fill=True, facecolor="#21262d", edgecolor="#8b949e", 
            linestyle="-", linewidth=1.5, alpha=0.3, zorder=1,
        ))
        
        centroids = pw.centroids
        if centroids is None or len(centroids) == 0:
            ax.text(0.5, 0.5, "No centroids", 
                   ha="center", va="center", color="#8b949e", fontsize=10,
                   transform=ax.transAxes)
            ax.set_title(f"{pw.unit_id}", color="#c9d1d9", fontsize=10, fontweight="bold")
            continue
        
        color = colors[i]
        
        if centroids.shape[1] == 3:
            sorted_idx = np.argsort(centroids[:, 0])
            sorted_centroids = centroids[sorted_idx]
            x_coords = sorted_centroids[:, 1]
            y_coords = sorted_centroids[:, 2]
            t_coords = sorted_centroids[:, 0]
        else:
            x_coords = centroids[:, 0]
            y_coords = centroids[:, 1]
            t_coords = np.arange(len(centroids))
        
        # Trajectory line
        ax.plot(y_coords, x_coords, color=color, linewidth=2, alpha=0.8, zorder=3)
        
        # Centroids with time coloring
        scatter = ax.scatter(y_coords, x_coords, c=t_coords, cmap='plasma',
                            s=60, alpha=0.9, edgecolors='white', linewidths=0.5, zorder=4)
        
        # Start and end markers
        ax.scatter([y_coords[0]], [x_coords[0]], c=[color], s=150, marker='*',
                  edgecolors='white', linewidths=1.5, zorder=5, label='Start')
        ax.scatter([y_coords[-1]], [x_coords[-1]], c=[color], s=100, marker='D',
                  edgecolors='white', linewidths=1.5, zorder=5, label='End')
        
        # Projection ray
        if pw.direction_angle is not None:
            if pw.start_point is not None:
                ray_start_x, ray_start_y = pw.start_point
            else:
                ray_start_x = x_coords[0]
                ray_start_y = y_coords[0]
            
            slope = pw.slope
            line_dx = 1.0 / np.sqrt(1 + slope**2)
            line_dy = slope * line_dx
            
            dir_rad = np.radians(pw.direction_angle)
            dir_dx = np.cos(dir_rad)
            dir_dy = np.sin(dir_rad)
            
            dot = line_dy * dir_dx + line_dx * dir_dy
            if dot < 0:
                line_dx = -line_dx
                line_dy = -line_dy
            
            ray_length = 100
            ray_end_y = ray_start_y + ray_length * line_dx
            ray_end_x = ray_start_x + ray_length * line_dy
            
            ax.annotate('', 
                       xy=(ray_end_y, ray_end_x),
                       xytext=(ray_start_y, ray_start_x),
                       arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=2.5,
                                       linestyle='--', alpha=0.8, shrinkA=0, shrinkB=0),
                       zorder=6)
        
        # ONH point
        if optimal is not None:
            ax.scatter([optimal[0]], [optimal[1]], c="#ff6b6b", s=150, marker="X",
                      edgecolors="white", linewidths=2, zorder=10, label='ONH')
        
        ax.set_xlim(-5, 70)
        ax.set_ylim(70, -5)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        
        r2 = pw.r_squared
        n_pts = len(centroids)
        dir_str = f"{pw.direction_angle:.0f} deg" if pw.direction_angle is not None else "N/A"
        ax.set_title(f"{pw.unit_id}\nR2={r2:.2f}, {n_pts} pts, dir={dir_str}", 
                    color="#c9d1d9", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", 
                 labelcolor="#c9d1d9", fontsize=7, markerscale=0.7)
    
    fig.suptitle(
        f"Individual Unit Centroids & Projections - {hdf5_path.name}\n"
        f"({n_units} units, R2 >= {r2_threshold})",
        color="#c9d1d9", fontsize=14, fontweight="bold", y=0.99
    )
    
    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03, wspace=0.3, hspace=0.4)
    
    output_path = output_dir / f"{hdf5_path.stem}_individual_units.png"
    fig.savefig(output_path, dpi=120, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# Main Visualization
# =============================================================================


def visualize_intersection(
    hdf5_path: Path,
    output_dir: Path,
    r2_threshold: float = 0.8,
    direction_tolerance: float = 45.0,
) -> Path:
    """
    Generate comprehensive ONH visualization from pre-computed data.
    
    All analysis is read from HDF5 - no calculations performed here.
    """
    print(f"Loading pathway data from {hdf5_path}...")
    pathways, onh_data = load_pathway_data(hdf5_path)
    
    if not pathways:
        print("No pathway data found!")
        return None
    
    print(f"Found {len(pathways)} pathways")
    
    if onh_data:
        print(f"ONH: ({onh_data.x:.2f}, {onh_data.y:.2f}), method={onh_data.method}")
        if onh_data.consensus_direction is not None:
            print(f"Consensus direction: {onh_data.consensus_direction:.1f} deg")
        if onh_data.n_cluster_points is not None:
            print(f"Cluster: {onh_data.n_cluster_points}/{onh_data.n_total_intersections} points")
    else:
        print("Warning: No ONH data found in HDF5")
    
    # Filter pathways for display
    pathways = filter_pathways_for_display(
        pathways, onh_data, r2_threshold, direction_tolerance
    )
    
    valid_count = sum(1 for pw in pathways.values() 
                     if pw.r_squared >= r2_threshold and pw.direction_valid)
    print(f"{valid_count} pathways pass display filters")
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    
    # Panel 1: Rays with direction
    ax1 = fig.add_subplot(2, 3, 1)
    plot_rays_with_direction(ax1, pathways, onh_data, r2_threshold)
    
    # Panel 2: Direction distribution (polar)
    ax2 = fig.add_subplot(2, 3, 2)
    plot_direction_distribution(ax2, pathways, onh_data, r2_threshold, direction_tolerance)
    
    # Panel 3: R² values
    ax3 = fig.add_subplot(2, 3, 3)
    plot_r2_values(ax3, pathways, r2_threshold)
    
    # Panel 4: Unit centroids
    ax4 = fig.add_subplot(2, 3, 4)
    plot_unit_centroids(ax4, pathways, onh_data, r2_threshold)
    
    # Panel 5: ONH zoomed
    ax5 = fig.add_subplot(2, 3, 5)
    plot_onh_zoomed(ax5, pathways, onh_data, r2_threshold)
    
    # Panel 6: Summary
    ax6 = fig.add_subplot(2, 3, 6)
    plot_summary(ax6, pathways, onh_data, r2_threshold, direction_tolerance, hdf5_path)
    
    # Main title
    fig.suptitle(
        f"ONH Intersection Analysis - {hdf5_path.name}",
        color="#c9d1d9", fontsize=16, fontweight="bold", y=0.98
    )
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05, wspace=0.25, hspace=0.3)
    
    # Save main figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hdf5_path.stem}_intersection_analysis.png"
    fig.savefig(output_path, dpi=150, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    
    # Create individual unit plots
    create_individual_unit_plots(pathways, onh_data, r2_threshold, output_dir, hdf5_path)
    
    return output_path


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("ONH Intersection Visualization")
    print("=" * 60)
    
    if not HDF5_FILE.exists():
        print(f"ERROR: File not found: {HDF5_FILE}")
        exit(1)
    
    output = visualize_intersection(
        HDF5_FILE,
        OUTPUT_DIR,
        R2_THRESHOLD,
        DIRECTION_TOLERANCE,
    )
    
    if output:
        print("\nVisualization complete!")
        print(f"Output: {output}")
    else:
        print("\nVisualization failed!")
