#!/usr/bin/env python
"""
AP Tracking Visualization Script.

Generates comprehensive visualizations of AP tracking results from HDF5 files.
Creates per-unit detailed plots and recording-level summary plots.

Can automatically run AP tracking if the file doesn't have it yet.

Usage:
    python visualize_ap_tracking.py <hdf5_path> [--output-dir <path>] [--run-tracking]
    
Example:
    # Visualize existing AP tracking data
    python visualize_ap_tracking.py export/2025.03.06-12.38.11-Rec.h5
    
    # Run AP tracking first, then visualize (copies to export folder)
    python visualize_ap_tracking.py path/to/file.h5 --run-tracking
"""

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# Add src to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# =============================================================================
# Custom colormap for neural data
# =============================================================================

# Dark-to-cyan colormap for STA
NEURAL_CMAP = LinearSegmentedColormap.from_list(
    "neural",
    ["#0d1117", "#1a1f29", "#0e4429", "#006d32", "#26a641", "#39d353", "#7ee787"],
)

# Prediction colormap (purple gradient)
PRED_CMAP = LinearSegmentedColormap.from_list(
    "prediction",
    ["#0d1117", "#21262d", "#3d1a78", "#6e40c9", "#8b5cf6", "#a78bfa", "#c4b5fd"],
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class UnitAPData:
    """Container for all AP tracking data of a single unit."""

    unit_id: str

    # Input data
    sta_data: Optional[np.ndarray] = None

    # DVNT position
    dv_position: Optional[float] = None
    nt_position: Optional[float] = None
    lr_position: Optional[str] = None

    # Refined soma
    soma_t: Optional[int] = None
    soma_x: Optional[int] = None
    soma_y: Optional[int] = None

    # AIS
    ais_t: Optional[int] = None
    ais_x: Optional[int] = None
    ais_y: Optional[int] = None

    # Prediction data
    prediction_data: Optional[np.ndarray] = None
    filtered_prediction: Optional[np.ndarray] = None
    axon_centroids: Optional[np.ndarray] = None

    # AP pathway
    pathway_slope: Optional[float] = None
    pathway_intercept: Optional[float] = None
    pathway_r_value: Optional[float] = None
    pathway_p_value: Optional[float] = None
    pathway_std_err: Optional[float] = None

    # Intersection
    intersection_x: Optional[float] = None
    intersection_y: Optional[float] = None

    # Polar coordinates (basic)
    polar_radius: Optional[float] = None
    polar_angle: Optional[float] = None
    polar_cartesian_x: Optional[float] = None
    polar_cartesian_y: Optional[float] = None
    polar_quadrant: Optional[str] = None
    polar_anatomical_quadrant: Optional[str] = None

    # Legacy polar coordinate fields
    polar_theta_deg: Optional[float] = None
    polar_theta_deg_raw: Optional[float] = None
    polar_theta_deg_corrected: Optional[float] = None
    polar_transformed_x: Optional[float] = None
    polar_transformed_y: Optional[float] = None
    polar_original_x: Optional[int] = None
    polar_original_y: Optional[int] = None
    polar_angle_correction: Optional[float] = None


@dataclass
class RecordingAPData:
    """Container for all AP tracking data of a recording."""

    recording_name: str
    units: Dict[str, UnitAPData] = field(default_factory=dict)

    # Common intersection point
    intersection_x: Optional[float] = None
    intersection_y: Optional[float] = None


# =============================================================================
# Data Loading
# =============================================================================


def load_ap_tracking_data(hdf5_path: Path) -> RecordingAPData:
    """
    Load all AP tracking data from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        RecordingAPData containing all unit data
    """
    recording = RecordingAPData(recording_name=hdf5_path.stem)

    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            return recording

        for unit_id in f["units"].keys():
            unit_data = UnitAPData(unit_id=unit_id)
            unit_path = f"units/{unit_id}"

            # Load STA data
            sta_path = f"{unit_path}/features/eimage_sta/data"
            if sta_path in f:
                unit_data.sta_data = f[sta_path][:]

            # Load AP tracking features
            ap_path = f"{unit_path}/features/ap_tracking"
            if ap_path not in f:
                recording.units[unit_id] = unit_data
                continue

            ap_grp = f[ap_path]

            # DVNT position
            if "DV_position" in ap_grp:
                unit_data.dv_position = float(ap_grp["DV_position"][()])
            if "NT_position" in ap_grp:
                unit_data.nt_position = float(ap_grp["NT_position"][()])
            if "LR_position" in ap_grp:
                lr = ap_grp["LR_position"][()]
                unit_data.lr_position = lr.decode("utf-8") if isinstance(lr, bytes) else str(lr)

            # Refined soma
            if "refined_soma" in ap_grp:
                soma_grp = ap_grp["refined_soma"]
                if "t" in soma_grp:
                    unit_data.soma_t = int(soma_grp["t"][()])
                if "x" in soma_grp:
                    unit_data.soma_x = int(soma_grp["x"][()])
                if "y" in soma_grp:
                    unit_data.soma_y = int(soma_grp["y"][()])

            # AIS
            if "axon_initial_segment" in ap_grp:
                ais_grp = ap_grp["axon_initial_segment"]
                if "t" in ais_grp:
                    t_val = ais_grp["t"][()]
                    if t_val is not None and not (isinstance(t_val, float) and np.isnan(t_val)):
                        unit_data.ais_t = int(t_val)
                if "x" in ais_grp:
                    unit_data.ais_x = int(ais_grp["x"][()])
                if "y" in ais_grp:
                    unit_data.ais_y = int(ais_grp["y"][()])

            # Prediction data
            if "prediction_sta_data" in ap_grp:
                unit_data.prediction_data = ap_grp["prediction_sta_data"][:]

            # Post-processed data
            if "post_processed_data" in ap_grp:
                pp_grp = ap_grp["post_processed_data"]
                if "filtered_prediction" in pp_grp:
                    unit_data.filtered_prediction = pp_grp["filtered_prediction"][:]
                if "axon_centroids" in pp_grp:
                    unit_data.axon_centroids = pp_grp["axon_centroids"][:]

            # AP pathway
            if "ap_pathway" in ap_grp:
                pw_grp = ap_grp["ap_pathway"]
                if "slope" in pw_grp:
                    unit_data.pathway_slope = float(pw_grp["slope"][()])
                if "intercept" in pw_grp:
                    unit_data.pathway_intercept = float(pw_grp["intercept"][()])
                if "r_value" in pw_grp:
                    unit_data.pathway_r_value = float(pw_grp["r_value"][()])
                if "p_value" in pw_grp:
                    unit_data.pathway_p_value = float(pw_grp["p_value"][()])
                if "std_err" in pw_grp:
                    unit_data.pathway_std_err = float(pw_grp["std_err"][()])

            # Intersection
            if "all_ap_intersection" in ap_grp:
                int_grp = ap_grp["all_ap_intersection"]
                if "x" in int_grp:
                    unit_data.intersection_x = float(int_grp["x"][()])
                if "y" in int_grp:
                    unit_data.intersection_y = float(int_grp["y"][()])

            # Polar coordinates
            if "soma_polar_coordinates" in ap_grp:
                polar_grp = ap_grp["soma_polar_coordinates"]
                if "radius" in polar_grp:
                    unit_data.polar_radius = float(polar_grp["radius"][()])
                if "angle" in polar_grp:
                    unit_data.polar_angle = float(polar_grp["angle"][()])
                if "cartesian_x" in polar_grp:
                    unit_data.polar_cartesian_x = float(polar_grp["cartesian_x"][()])
                if "cartesian_y" in polar_grp:
                    unit_data.polar_cartesian_y = float(polar_grp["cartesian_y"][()])
                if "quadrant" in polar_grp:
                    q = polar_grp["quadrant"][()]
                    unit_data.polar_quadrant = q.decode("utf-8") if isinstance(q, bytes) else str(q)
                if "anatomical_quadrant" in polar_grp:
                    aq = polar_grp["anatomical_quadrant"][()]
                    unit_data.polar_anatomical_quadrant = aq.decode("utf-8") if isinstance(aq, bytes) else str(aq)

                # Legacy polar coordinate fields
                if "theta_deg" in polar_grp:
                    unit_data.polar_theta_deg = float(polar_grp["theta_deg"][()])
                if "theta_deg_raw" in polar_grp:
                    unit_data.polar_theta_deg_raw = float(polar_grp["theta_deg_raw"][()])
                if "theta_deg_corrected" in polar_grp:
                    unit_data.polar_theta_deg_corrected = float(polar_grp["theta_deg_corrected"][()])
                if "transformed_x" in polar_grp:
                    unit_data.polar_transformed_x = float(polar_grp["transformed_x"][()])
                if "transformed_y" in polar_grp:
                    unit_data.polar_transformed_y = float(polar_grp["transformed_y"][()])
                if "original_x" in polar_grp:
                    unit_data.polar_original_x = int(polar_grp["original_x"][()])
                if "original_y" in polar_grp:
                    unit_data.polar_original_y = int(polar_grp["original_y"][()])
                if "angle_correction_applied" in polar_grp:
                    unit_data.polar_angle_correction = float(polar_grp["angle_correction_applied"][()])

            recording.units[unit_id] = unit_data

            # Track common intersection
            if unit_data.intersection_x is not None:
                recording.intersection_x = unit_data.intersection_x
                recording.intersection_y = unit_data.intersection_y

    return recording


# =============================================================================
# Per-Unit Visualization
# =============================================================================


def plot_unit_summary(unit: UnitAPData, output_path: Path) -> None:
    """
    Generate a 3x3 multi-panel figure for a single unit.

    Args:
        unit: UnitAPData containing all features
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.patch.set_facecolor("#0d1117")

    # Style all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

    # Title
    fig.suptitle(
        f"AP Tracking Analysis: {unit.unit_id}",
        fontsize=16,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    # Panel (0,0): STA max projection with soma and AIS markers
    ax = axes[0, 0]
    if unit.sta_data is not None:
        sta_max = np.max(np.abs(unit.sta_data), axis=0)
        im = ax.imshow(sta_max, cmap=NEURAL_CMAP, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark soma
        if unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter(unit.soma_y, unit.soma_x, c="#ff6b6b", s=120, marker="o", 
                      edgecolors="white", linewidths=2, label="Soma", zorder=10)

        # Mark AIS
        if unit.ais_x is not None and unit.ais_y is not None:
            ax.scatter(unit.ais_y, unit.ais_x, c="#4dabf7", s=100, marker="^",
                      edgecolors="white", linewidths=2, label="AIS", zorder=10)

        ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d",
                 labelcolor="#c9d1d9", fontsize=8)

    ax.set_title("STA Max Projection", color="#c9d1d9", fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", color="#8b949e", fontsize=9)
    ax.set_ylabel("Row", color="#8b949e", fontsize=9)

    # Panel (0,1): CNN prediction max projection
    ax = axes[0, 1]
    if unit.prediction_data is not None:
        pred_max = np.max(unit.prediction_data, axis=0)
        im = ax.imshow(pred_max, cmap=PRED_CMAP, aspect="equal", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("CNN Prediction Max", color="#c9d1d9", fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", color="#8b949e", fontsize=9)
    ax.set_ylabel("Row", color="#8b949e", fontsize=9)

    # Panel (0,2): Overlay - STA + prediction + centroids
    ax = axes[0, 2]
    if unit.sta_data is not None and unit.prediction_data is not None:
        sta_max = np.max(np.abs(unit.sta_data), axis=0)
        pred_max = np.max(unit.prediction_data, axis=0)

        # Normalize for overlay
        sta_norm = (sta_max - sta_max.min()) / (sta_max.max() - sta_max.min() + 1e-8)
        pred_norm = pred_max

        # Create RGB overlay (STA=green, pred=magenta)
        overlay = np.zeros((*sta_norm.shape, 3))
        overlay[:, :, 0] = pred_norm * 0.8  # Red channel
        overlay[:, :, 1] = sta_norm * 0.8   # Green channel
        overlay[:, :, 2] = pred_norm * 0.8  # Blue channel

        ax.imshow(overlay, aspect="equal")

        # Plot axon centroids
        if unit.axon_centroids is not None and len(unit.axon_centroids) > 0:
            centroids = unit.axon_centroids
            ax.scatter(centroids[:, 2], centroids[:, 1], c=centroids[:, 0],
                      cmap="plasma", s=20, alpha=0.7, edgecolors="white", linewidths=0.5)

    ax.set_title("Overlay + Centroids", color="#c9d1d9", fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", color="#8b949e", fontsize=9)
    ax.set_ylabel("Row", color="#8b949e", fontsize=9)

    # Panel (1,0): STA time slice at soma t
    ax = axes[1, 0]
    if unit.sta_data is not None and unit.soma_t is not None:
        t_slice = unit.soma_t
        sta_slice = unit.sta_data[t_slice]
        im = ax.imshow(sta_slice, cmap=NEURAL_CMAP, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark soma position
        if unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter(unit.soma_y, unit.soma_x, c="#ff6b6b", s=100, marker="o",
                      edgecolors="white", linewidths=2, zorder=10)

    ax.set_title(f"STA @ t={unit.soma_t}", color="#c9d1d9", fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", color="#8b949e", fontsize=9)
    ax.set_ylabel("Row", color="#8b949e", fontsize=9)

    # Panel (1,1): Prediction time slice at soma t
    ax = axes[1, 1]
    if unit.prediction_data is not None and unit.soma_t is not None:
        t_slice = unit.soma_t
        pred_slice = unit.prediction_data[t_slice]
        im = ax.imshow(pred_slice, cmap=PRED_CMAP, aspect="equal", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Prediction @ t={unit.soma_t}", color="#c9d1d9", fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", color="#8b949e", fontsize=9)
    ax.set_ylabel("Row", color="#8b949e", fontsize=9)

    # Panel (1,2): Axon centroids 3D scatter
    ax = axes[1, 2]
    ax.remove()
    ax = fig.add_subplot(3, 3, 6, projection="3d")
    ax.set_facecolor("#161b22")

    if unit.axon_centroids is not None and len(unit.axon_centroids) > 0:
        centroids = unit.axon_centroids
        scatter = ax.scatter(
            centroids[:, 2],  # Y (column)
            centroids[:, 1],  # X (row)
            centroids[:, 0],  # T (time)
            c=centroids[:, 0],
            cmap="plasma",
            s=30,
            alpha=0.8,
        )

        # Mark soma
        if unit.soma_t is not None and unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter([unit.soma_y], [unit.soma_x], [unit.soma_t],
                      c="#ff6b6b", s=150, marker="o", edgecolors="white", linewidths=2)

    ax.set_xlabel("Col", color="#8b949e", fontsize=8, labelpad=-2)
    ax.set_ylabel("Row", color="#8b949e", fontsize=8, labelpad=-2)
    ax.set_zlabel("Time", color="#8b949e", fontsize=8, labelpad=-2)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title("Axon Centroids 3D", color="#c9d1d9", fontsize=10, fontweight="bold", y=1.02)

    # Panel (2,0): AP pathway line fit
    # The analysis computes: row = slope * column + intercept
    # using linregress(column_indices, row_indices) on non-zero projection pixels.
    # We use the slope/intercept STORED in HDF5 (not recomputed).
    ax = axes[2, 0]
    has_projection_data = False

    if unit.filtered_prediction is not None:
        # Calculate projection the same way as fit_line_to_projections() for visualization
        projections = np.max(unit.filtered_prediction, axis=0) - np.min(unit.filtered_prediction, axis=0)

        # Get non-zero coordinates (same as analysis pipeline)
        non_zero = np.where(projections > 0)
        row_coords = non_zero[0]  # Row indices (what analysis calls y_coords)
        col_coords = non_zero[1]  # Column indices (what analysis calls x_coords)

        if len(col_coords) > 0:
            has_projection_data = True
            # Plot with column on x-axis, row on y-axis (matching image orientation)
            ax.scatter(col_coords, row_coords, c="#7ee787", s=10, alpha=0.5, 
                      label=f"Projection ({len(col_coords)} pts)")

            # Plot fitted line using STORED slope/intercept from HDF5
            # Analysis fit: row = slope * column + intercept
            if unit.pathway_slope is not None and unit.pathway_intercept is not None:
                col_line = np.linspace(col_coords.min() - 5, col_coords.max() + 5, 100)
                row_line = unit.pathway_slope * col_line + unit.pathway_intercept
                r_sq = unit.pathway_r_value ** 2 if unit.pathway_r_value else 0
                ax.plot(col_line, row_line, color="#ff6b6b", linewidth=2, linestyle="--",
                       label=f"HDF5 Fit (R²={r_sq:.3f})")

    # Mark soma position (soma_x=row, soma_y=col)
    if unit.soma_x is not None and unit.soma_y is not None:
        ax.scatter([unit.soma_y], [unit.soma_x], c="#4dabf7", s=150, marker="*",
                  edgecolors="white", linewidths=2, label="Soma", zorder=10)

    if has_projection_data or (unit.soma_x is not None):
        ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d",
                 labelcolor="#c9d1d9", fontsize=8)

    ax.set_title("AP Pathway Fit (from HDF5)", color="#c9d1d9", fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", color="#8b949e", fontsize=9)
    ax.set_ylabel("Row", color="#8b949e", fontsize=9)
    ax.invert_yaxis()  # Match image orientation: row 0 at top
    ax.set_aspect("equal")

    # Panel (2,1): Polar coordinate diagram
    ax = axes[2, 1]
    ax.remove()
    ax = fig.add_subplot(3, 3, 8, projection="polar")
    ax.set_facecolor("#161b22")

    if unit.polar_radius is not None and unit.polar_angle is not None:
        ax.scatter([unit.polar_angle], [unit.polar_radius], c="#ff6b6b", s=200,
                  marker="o", edgecolors="white", linewidths=2, zorder=10)

        # Draw radius line
        ax.plot([0, unit.polar_angle], [0, unit.polar_radius], color="#4dabf7",
               linewidth=2, alpha=0.7)

        # Add quadrant label
        if unit.polar_quadrant:
            ax.annotate(
                unit.polar_quadrant,
                xy=(unit.polar_angle, unit.polar_radius),
                xytext=(unit.polar_angle + 0.3, unit.polar_radius * 1.2),
                color="#c9d1d9",
                fontsize=10,
                fontweight="bold",
            )

    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title("Polar Coordinates", color="#c9d1d9", fontsize=10, fontweight="bold", y=1.08)

    # Panel (2,2): Text summary
    ax = axes[2, 2]
    ax.axis("off")

    summary_lines = [
        f"Unit: {unit.unit_id}",
        "",
        "━━━ DVNT Position ━━━",
        f"  DV: {unit.dv_position:.2f}" if unit.dv_position is not None and not np.isnan(unit.dv_position) else "  DV: N/A",
        f"  NT: {unit.nt_position:.2f}" if unit.nt_position is not None and not np.isnan(unit.nt_position) else "  NT: N/A",
        f"  LR: {unit.lr_position}" if unit.lr_position else "  LR: N/A",
        "",
        "━━━ Refined Soma ━━━",
        f"  t={unit.soma_t}, x={unit.soma_x}, y={unit.soma_y}" if unit.soma_t is not None else "  N/A",
        "",
        "━━━ AIS ━━━",
        f"  t={unit.ais_t}, x={unit.ais_x}, y={unit.ais_y}" if unit.ais_t is not None else "  N/A",
        "",
        "━━━ AP Pathway ━━━",
        f"  Slope: {unit.pathway_slope:.4f}" if unit.pathway_slope is not None else "  Slope: N/A",
        f"  R²: {unit.pathway_r_value**2:.4f}" if unit.pathway_r_value is not None else "  R²: N/A",
        f"  p-value: {unit.pathway_p_value:.2e}" if unit.pathway_p_value is not None else "  p-value: N/A",
        "",
        "━━━ Polar Coordinates ━━━",
        f"  Radius: {unit.polar_radius:.2f}" if unit.polar_radius is not None and not np.isnan(unit.polar_radius) else "  Radius: N/A",
        f"  θ raw: {unit.polar_theta_deg_raw:.1f}°" if unit.polar_theta_deg_raw is not None and not np.isnan(unit.polar_theta_deg_raw) else "  θ raw: N/A",
        f"  θ corrected: {unit.polar_theta_deg_corrected:.1f}°" if unit.polar_theta_deg_corrected is not None and not np.isnan(unit.polar_theta_deg_corrected) else "  θ corrected: N/A",
        f"  Correction: {unit.polar_angle_correction:.1f}°" if unit.polar_angle_correction is not None and not np.isnan(unit.polar_angle_correction) else "  Correction: N/A",
        f"  Quadrant: {unit.polar_quadrant}" if unit.polar_quadrant else "  Quadrant: N/A",
        f"  Anatomical: {unit.polar_anatomical_quadrant}" if unit.polar_anatomical_quadrant else "  Anatomical: N/A",
    ]

    summary_text = "\n".join(summary_lines)
    ax.text(
        0.05, 0.95, summary_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d"),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=150, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)


# =============================================================================
# Recording-Level Visualizations
# =============================================================================


def plot_recording_pathways(recording: RecordingAPData, output_path: Path) -> None:
    """
    Plot all AP pathways with their intersection point.

    Args:
        recording: RecordingAPData with all unit data
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Collect valid pathways
    valid_units = []
    for unit in recording.units.values():
        if unit.pathway_slope is not None and unit.pathway_r_value is not None:
            valid_units.append(unit)

    if not valid_units:
        ax.text(0.5, 0.5, "No valid AP pathways found",
               transform=ax.transAxes, ha="center", va="center",
               color="#c9d1d9", fontsize=14)
        plt.savefig(output_path, dpi=150, facecolor="#0d1117")
        plt.close(fig)
        return

    # Determine plot bounds
    all_x, all_y = [], []
    for unit in valid_units:
        if unit.soma_x is not None and unit.soma_y is not None:
            all_x.append(unit.soma_x)
            all_y.append(unit.soma_y)

    if all_x:
        x_min, x_max = min(all_x) - 10, max(all_x) + 10
        y_min, y_max = min(all_y) - 10, max(all_y) + 10
    else:
        x_min, x_max, y_min, y_max = 0, 65, 0, 65

    # Plot each pathway
    cmap = plt.cm.RdYlGn
    for unit in valid_units:
        r_squared = unit.pathway_r_value ** 2 if unit.pathway_r_value else 0
        color = cmap(r_squared)

        # Draw pathway line
        y_line = np.linspace(y_min, y_max, 100)
        x_line = unit.pathway_slope * y_line + unit.pathway_intercept
        ax.plot(y_line, x_line, color=color, linewidth=1.5, alpha=0.7)

        # Mark soma position
        if unit.soma_x is not None and unit.soma_y is not None:
            ax.scatter([unit.soma_y], [unit.soma_x], c=[color], s=80,
                      edgecolors="white", linewidths=1, zorder=5)

    # Plot intersection point
    if recording.intersection_x is not None and recording.intersection_y is not None:
        ax.scatter([recording.intersection_y], [recording.intersection_x],
                  c="#ff6b6b", s=300, marker="X", edgecolors="white",
                  linewidths=3, zorder=10, label="Intersection")

    # Colorbar for R²
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("R²", color="#c9d1d9", fontsize=12)
    cbar.ax.tick_params(colors="#8b949e")

    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal")

    ax.set_title(f"AP Pathways - {recording.recording_name}\n({len(valid_units)} units)",
                color="#f0f6fc", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column (Y)", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Row (X)", color="#c9d1d9", fontsize=12)
    ax.tick_params(colors="#8b949e")

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d",
             labelcolor="#c9d1d9", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)


def plot_polar_summary(recording: RecordingAPData, output_path: Path) -> None:
    """
    Create a polar plot showing all soma positions relative to intersection.

    Uses the corrected angle (theta_deg_corrected) for proper anatomical alignment.

    Args:
        recording: RecordingAPData with all unit data
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#0d1117")

    # Collect polar coordinates (raw and corrected)
    angles_raw, angles_corrected, radii, labels = [], [], [], []
    for unit in recording.units.values():
        if unit.polar_radius is not None and not np.isnan(unit.polar_radius):
            # Get raw and corrected angles
            if unit.polar_theta_deg_raw is not None and not np.isnan(unit.polar_theta_deg_raw):
                angles_raw.append(np.radians(unit.polar_theta_deg_raw))
            else:
                angles_raw.append(None)

            if unit.polar_theta_deg_corrected is not None and not np.isnan(unit.polar_theta_deg_corrected):
                angles_corrected.append(np.radians(unit.polar_theta_deg_corrected))
            else:
                angles_corrected.append(None)

            radii.append(unit.polar_radius)
            labels.append(unit.unit_id)

    if not radii:
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.text(0, 0, "No polar coordinates available",
                   ha="center", va="center", color="#c9d1d9", fontsize=14)
        plt.savefig(output_path, dpi=150, facecolor="#0d1117")
        plt.close(fig)
        return

    # Get angle correction (should be same for all units)
    angle_correction = None
    for unit in recording.units.values():
        if unit.polar_angle_correction is not None and not np.isnan(unit.polar_angle_correction):
            angle_correction = unit.polar_angle_correction
            break

    # Panel 1: Raw angles (uncorrected)
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")

    valid_raw = [(a, r) for a, r in zip(angles_raw, radii) if a is not None]
    if valid_raw:
        raw_a, raw_r = zip(*valid_raw)
        scatter1 = ax1.scatter(raw_a, raw_r, c=raw_r, cmap="viridis", s=100,
                              edgecolors="white", linewidths=1, alpha=0.8)
        cbar1 = plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.1)
        cbar1.set_label("Radius", color="#c9d1d9", fontsize=11)
        cbar1.ax.tick_params(colors="#8b949e")

    ax1.scatter([0], [0], c="#ff6b6b", s=200, marker="X",
               edgecolors="white", linewidths=2, zorder=10, label="Intersection")

    ax1.set_title(f"Raw θ (Uncorrected)\n{len(valid_raw)} units",
                 color="#f0f6fc", fontsize=13, fontweight="bold", y=1.08)
    ax1.tick_params(colors="#8b949e")

    # Panel 2: Corrected angles (anatomically aligned)
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")

    valid_corrected = [(a, r) for a, r in zip(angles_corrected, radii) if a is not None]
    if valid_corrected:
        corr_a, corr_r = zip(*valid_corrected)
        scatter2 = ax2.scatter(corr_a, corr_r, c=corr_r, cmap="plasma", s=100,
                              edgecolors="white", linewidths=1, alpha=0.8)
        cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.1)
        cbar2.set_label("Radius", color="#c9d1d9", fontsize=11)
        cbar2.ax.tick_params(colors="#8b949e")

    ax2.scatter([0], [0], c="#ff6b6b", s=200, marker="X",
               edgecolors="white", linewidths=2, zorder=10, label="Intersection")

    # Add quadrant labels for corrected plot (anatomically meaningful)
    max_r = max(radii) if radii else 1
    ax2.text(0, max_r * 0.5, "N/D", ha="center", va="center",
            color="#7ee787", fontsize=11, fontweight="bold")
    ax2.text(np.pi/2, max_r * 0.5, "T/D", ha="center", va="center",
            color="#7ee787", fontsize=11, fontweight="bold")
    ax2.text(np.pi, max_r * 0.5, "N/V", ha="center", va="center",
            color="#7ee787", fontsize=11, fontweight="bold")
    ax2.text(3*np.pi/2, max_r * 0.5, "T/V", ha="center", va="center",
            color="#7ee787", fontsize=11, fontweight="bold")

    corr_str = f" (Δ={angle_correction:.1f}°)" if angle_correction is not None else ""
    ax2.set_title(f"Corrected θ{corr_str}\n{len(valid_corrected)} units",
                 color="#f0f6fc", fontsize=13, fontweight="bold", y=1.08)
    ax2.tick_params(colors="#8b949e")
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=10)

    fig.suptitle(f"Soma Polar Positions - {recording.recording_name}",
                color="#f0f6fc", fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)


# =============================================================================
# AP Tracking Runner
# =============================================================================


def has_ap_tracking(hdf5_path: Path) -> bool:
    """Check if HDF5 file already has AP tracking data."""
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            return False
        for unit_id in f["units"].keys():
            ap_path = f"units/{unit_id}/features/ap_tracking"
            if ap_path in f:
                return True
    return False


def run_ap_tracking_on_file(
    source_path: Path,
    r2_threshold: float = 0.2,
) -> Path:
    """
    Run AP tracking on a file, copying to export folder first.
    
    Args:
        source_path: Path to source HDF5 file
        r2_threshold: R² threshold for pathway fitting
        
    Returns:
        Path to the processed file in export folder
    """
    from hdmea.features.ap_tracking import compute_ap_tracking
    
    # Define paths
    export_folder = Path(__file__).parent / "export"
    export_folder.mkdir(parents=True, exist_ok=True)
    target_path = export_folder / source_path.name
    model_path = Path(__file__).parent / "model" / "CNN_3d_with_velocity_model_from_all_process.pth"
    
    # Copy to export folder if source is not already there
    if source_path.resolve() != target_path.resolve():
        print(f"Copying {source_path.name} to export folder...")
        if target_path.exists():
            target_path.unlink()
        shutil.copy(source_path, target_path)
    
    # Run AP tracking
    print(f"Running AP tracking with r2_threshold={r2_threshold}...")
    compute_ap_tracking(
        target_path,
        model_path,
        r2_threshold=r2_threshold,
    )
    print("AP tracking complete!")
    
    return target_path


# =============================================================================
# Main Entry Point
# =============================================================================


def visualize_ap_tracking(
    hdf5_path: Path,
    output_dir: Optional[Path] = None,
    max_units: Optional[int] = None,
    run_tracking: bool = False,
    r2_threshold: float = 0.2,
) -> Path:
    """
    Generate all visualizations for an AP tracking result file.

    Args:
        hdf5_path: Path to the HDF5 file (source or processed)
        output_dir: Output directory (default: plots/ folder next to this script)
        max_units: Maximum number of units to visualize (None = all)
        run_tracking: If True, run AP tracking first (copies to export folder)
        r2_threshold: R² threshold for pathway fitting (if running tracking)

    Returns:
        Path to the output directory
    """
    hdf5_path = Path(hdf5_path)
    
    # Check if we need to run AP tracking
    if run_tracking or not has_ap_tracking(hdf5_path):
        if not has_ap_tracking(hdf5_path):
            print(f"No AP tracking data found in {hdf5_path.name}")
        hdf5_path = run_ap_tracking_on_file(hdf5_path, r2_threshold=r2_threshold)

    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "plots"
    output_dir = Path(output_dir)

    # Create recording-specific folder
    recording_dir = output_dir / hdf5_path.stem
    recording_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AP tracking data from: {hdf5_path}")
    recording = load_ap_tracking_data(hdf5_path)
    print(f"Found {len(recording.units)} units")

    # Generate per-unit plots
    unit_ids = list(recording.units.keys())
    if max_units is not None:
        unit_ids = unit_ids[:max_units]

    print(f"Generating per-unit plots for {len(unit_ids)} units...")
    for i, unit_id in enumerate(unit_ids):
        unit = recording.units[unit_id]
        output_path = recording_dir / f"unit_{unit_id}_ap_tracking.png"
        plot_unit_summary(unit, output_path)
        print(f"  [{i+1}/{len(unit_ids)}] {unit_id} -> {output_path.name}")

    # Generate recording summary plots
    print("Generating recording summary plots...")

    pathways_path = recording_dir / "recording_summary.png"
    plot_recording_pathways(recording, pathways_path)
    print(f"  -> {pathways_path.name}")

    polar_path = recording_dir / "polar_summary.png"
    plot_polar_summary(recording, polar_path)
    print(f"  -> {polar_path.name}")

    print(f"\nAll plots saved to: {recording_dir}")
    return recording_dir


# =============================================================================
# CLI
# =============================================================================


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize AP tracking results from HDF5 files"
    )
    parser.add_argument(
        "hdf5_path",
        type=Path,
        help="Path to the HDF5 file (will run AP tracking if needed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: plots/ folder)",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Maximum number of units to visualize",
    )
    parser.add_argument(
        "--run-tracking",
        action="store_true",
        help="Force re-run AP tracking even if data exists",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.2,
        help="R² threshold for pathway fitting (default: 0.2)",
    )

    args = parser.parse_args()

    visualize_ap_tracking(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        max_units=args.max_units,
        run_tracking=args.run_tracking,
        r2_threshold=args.r2_threshold,
    )


if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURE YOUR INPUT FILE HERE
    # ==========================================================================
    
    # Option 1: Direct path to HDF5 file (will auto-run AP tracking if needed)
    INPUT_FILE = "Projects/ap_trace_hdf5/export/2024.05.23-12.05.03-Rec.h5"
    
    # Option 2: Use file in export folder (already processed)
    # INPUT_FILE = "Projects/ap_trace_hdf5/export/2024.05.23-12.05.03-Rec.h5"
    
    # Settings
    MAX_UNITS = None        # None = all units, or set a number like 5
    R2_THRESHOLD = 0.2      # R² threshold for pathway fitting
    RUN_TRACKING = False    # True = force re-run AP tracking
    
    # ==========================================================================
    # RUN VISUALIZATION
    # ==========================================================================
    
    input_path = _PROJECT_ROOT / INPUT_FILE
    
    visualize_ap_tracking(
        hdf5_path=input_path,
        max_units=MAX_UNITS,
        run_tracking=RUN_TRACKING,
        r2_threshold=R2_THRESHOLD,
    )

