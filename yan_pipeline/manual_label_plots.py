#!/usr/bin/env python
"""
Manual Label Plot Generation.

Generates per-unit combined visualization images for manual labeling.
Each unit gets one image containing all selected plot types in a horizontal layout.

Plot types:
- eimage_sta: STA montage showing multiple frames
- geometry: Soma center + RF center
- ap_tracking: AP spread pattern
- dsgc: Direction tuning polar plot
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# =============================================================================
# Styling
# =============================================================================

plt.style.use("dark_background")

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"

# STA colormap (blue-white-red diverging)
STA_CMAP = LinearSegmentedColormap.from_list(
    "sta_diverging",
    ["#0d47a1", "#1976d2", "#64b5f6", "#ffffff", "#ef5350", "#c62828", "#7f0000"],
)

# AP tracking colormap
NEURAL_CMAP = LinearSegmentedColormap.from_list(
    "neural",
    ["#0d1117", "#1a1f29", "#0e4429", "#006d32", "#26a641", "#39d353", "#7ee787"],
)

SOMA_COLOR = "#ff6b6b"
RF_COLOR = "#4ecdc4"
AIS_COLOR = "#ffd93d"

# DSGC colors
DIRECTION_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]


# =============================================================================
# Data Loading
# =============================================================================

def get_unit_ids(hdf5_path: Path) -> List[str]:
    """Get list of all unit IDs in HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            return []
        return sorted(list(f["units"].keys()))


def load_unit_data(hdf5_path: Path, unit_id: str) -> Dict[str, Any]:
    """Load all relevant data for a unit."""
    data = {"unit_id": unit_id}
    
    with h5py.File(hdf5_path, "r") as f:
        unit_path = f"units/{unit_id}"
        if unit_path not in f:
            return data
        
        # === eimage_sta ===
        sta_path = f"{unit_path}/features/eimage_sta/data"
        if sta_path in f:
            data["eimage_sta"] = f[sta_path][()]
        
        # === Soma geometry ===
        soma_geom_path = f"{unit_path}/features/eimage_sta/geometry"
        if soma_geom_path in f:
            geom = f[soma_geom_path]
            if "center_t" in geom:
                data["soma_t"] = int(geom["center_t"][()])
            if "center_row" in geom:
                data["soma_row"] = float(geom["center_row"][()])
            if "center_col" in geom:
                data["soma_col"] = float(geom["center_col"][()])
        
        # === RF geometry (from STA dense noise) ===
        if f"{unit_path}/features" in f:
            for key in f[f"{unit_path}/features"]:
                if "sta_perfect" in key or "dense_noise" in key.lower():
                    sta_geom_path = f"{unit_path}/features/{key}/sta_geometry"
                    if sta_geom_path in f:
                        rf = f[sta_geom_path]
                        if "gaussian_fit" in rf:
                            gf = rf["gaussian_fit"]
                            if "x0" in gf:
                                data["rf_x0"] = float(gf["x0"][()])
                            if "y0" in gf:
                                data["rf_y0"] = float(gf["y0"][()])
                            if "sigma_x" in gf:
                                data["rf_sigma_x"] = float(gf["sigma_x"][()])
                            if "sigma_y" in gf:
                                data["rf_sigma_y"] = float(gf["sigma_y"][()])
                    
                    # Load STA data
                    sta_data_path = f"{unit_path}/features/{key}/data"
                    if sta_data_path in f:
                        data["sta_dense_noise"] = f[sta_data_path][()]
                    break
        
        # === AP tracking ===
        ap_path = f"{unit_path}/features/ap_tracking"
        if ap_path in f:
            ap = f[ap_path]
            if "prediction_data" in ap:
                data["ap_prediction"] = ap["prediction_data"][()]
            if "axon_centroids" in ap:
                data["ap_centroids"] = ap["axon_centroids"][()]
            if "pathway_slope" in ap:
                data["ap_slope"] = float(ap["pathway_slope"][()])
            if "pathway_intercept" in ap:
                data["ap_intercept"] = float(ap["pathway_intercept"][()])
        
        # === DSGC direction sectioning ===
        dsgc_path = f"{unit_path}/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section"
        if dsgc_path in f:
            section = f[dsgc_path]
            data["dsgc_directions"] = {}
            
            if "directions" in section:
                dir_grp = section["directions"]
                for dir_key in dir_grp:
                    try:
                        direction = int(dir_key)
                    except ValueError:
                        direction = int(dir_key.replace("dir_", ""))
                    
                    dir_data = dir_grp[dir_key]
                    total_spikes = 0
                    
                    if "trials" in dir_data:
                        trials_grp = dir_data["trials"]
                        for trial_key in trials_grp.keys():
                            spikes = trials_grp[trial_key][()]
                            if isinstance(spikes, np.ndarray):
                                total_spikes += len(spikes)
                    
                    data["dsgc_directions"][direction] = total_spikes
    
    return data


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_eimage_sta(ax, data: Dict[str, Any], n_frames: int = 6):
    """Plot STA montage."""
    ax.set_facecolor(PANEL_COLOR)
    
    sta = data.get("eimage_sta")
    if sta is None:
        ax.text(0.5, 0.5, "No STA", ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("eimage_sta", fontsize=9)
        ax.axis("off")
        return
    
    # Find peak frame
    abs_sta = np.abs(np.nan_to_num(sta))
    peak_t = np.unravel_index(np.argmax(abs_sta), sta.shape)[0]
    
    # Select frames around peak
    n_total = sta.shape[0]
    start_frame = max(0, peak_t - n_frames // 2)
    end_frame = min(n_total, start_frame + n_frames)
    if end_frame - start_frame < n_frames:
        start_frame = max(0, end_frame - n_frames)
    
    frame_indices = list(range(start_frame, end_frame))
    
    # Create montage
    n_cols = min(n_frames, len(frame_indices))
    n_rows = 1
    
    frames_to_show = [sta[i] for i in frame_indices]
    if frames_to_show:
        montage = np.hstack(frames_to_show)
        
        vmax = np.nanmax(np.abs(montage))
        if vmax == 0:
            vmax = 1
        
        ax.imshow(montage, cmap=STA_CMAP, vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    
    ax.set_title(f"STA (t={peak_t})", fontsize=9)
    ax.axis("off")


def plot_geometry(ax, data: Dict[str, Any]):
    """Plot soma + RF geometry centers."""
    ax.set_facecolor(PANEL_COLOR)
    
    has_soma = "soma_row" in data and "soma_col" in data
    has_rf = "rf_x0" in data and "rf_y0" in data
    
    if not has_soma and not has_rf:
        ax.text(0.5, 0.5, "No Geometry", ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("Geometry", fontsize=9)
        ax.axis("off")
        return
    
    # Plot background (use STA if available)
    sta = data.get("eimage_sta")
    if sta is not None:
        peak_t = data.get("soma_t", sta.shape[0] // 2)
        peak_t = min(peak_t, sta.shape[0] - 1)
        frame = sta[peak_t]
        vmax = np.nanmax(np.abs(frame))
        if vmax > 0:
            ax.imshow(frame, cmap="gray", vmin=-vmax, vmax=vmax, origin="lower", alpha=0.5)
    
    # Plot soma center
    if has_soma:
        ax.scatter([data["soma_col"]], [data["soma_row"]], 
                   c=SOMA_COLOR, s=100, marker="o", label="Soma", edgecolors="white", linewidths=1)
    
    # Plot RF center and ellipse
    if has_rf:
        ax.scatter([data["rf_x0"]], [data["rf_y0"]], 
                   c=RF_COLOR, s=100, marker="^", label="RF", edgecolors="white", linewidths=1)
        
        if "rf_sigma_x" in data and "rf_sigma_y" in data:
            ellipse = Ellipse(
                (data["rf_x0"], data["rf_y0"]),
                width=data["rf_sigma_x"] * 2,
                height=data["rf_sigma_y"] * 2,
                fill=False, edgecolor=RF_COLOR, linewidth=1.5, linestyle="--"
            )
            ax.add_patch(ellipse)
    
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax.set_title("Geometry", fontsize=9)
    ax.set_aspect("equal")


def plot_ap_tracking(ax, data: Dict[str, Any]):
    """Plot AP spread pattern."""
    ax.set_facecolor(PANEL_COLOR)
    
    pred = data.get("ap_prediction")
    centroids = data.get("ap_centroids")
    
    if pred is None and centroids is None:
        ax.text(0.5, 0.5, "No AP", ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("AP Tracking", fontsize=9)
        ax.axis("off")
        return
    
    # Plot prediction heatmap (max projection over time)
    if pred is not None:
        max_proj = np.nanmax(pred, axis=0)
        if max_proj.max() > 0:
            ax.imshow(max_proj, cmap=NEURAL_CMAP, origin="lower", aspect="auto")
    
    # Plot centroids and pathway
    if centroids is not None and len(centroids) > 0:
        # centroids shape: (n_frames, 2) or similar
        if centroids.ndim == 2 and centroids.shape[1] >= 2:
            valid_mask = ~np.isnan(centroids[:, 0]) & ~np.isnan(centroids[:, 1])
            valid_centroids = centroids[valid_mask]
            if len(valid_centroids) > 0:
                ax.scatter(valid_centroids[:, 1], valid_centroids[:, 0], 
                          c=AIS_COLOR, s=10, alpha=0.7)
                
                # Draw pathway line if available
                if "ap_slope" in data and "ap_intercept" in data:
                    slope = data["ap_slope"]
                    intercept = data["ap_intercept"]
                    x_range = np.array([0, pred.shape[2] if pred is not None else 64])
                    y_line = slope * x_range + intercept
                    ax.plot(x_range, y_line, color=AIS_COLOR, linestyle="--", linewidth=1)
    
    ax.set_title("AP Tracking", fontsize=9)
    ax.axis("off")


def plot_dsgc(ax, data: Dict[str, Any]):
    """Plot DSGC direction tuning polar plot."""
    directions = data.get("dsgc_directions", {})
    
    if not directions:
        ax.text(0.5, 0.5, "No DSGC", ha="center", va="center", fontsize=10, color="gray", 
                transform=ax.transAxes)
        ax.set_title("DSGC", fontsize=9)
        ax.axis("off")
        return
    
    # Convert to polar
    angles = np.array(sorted(directions.keys())) * np.pi / 180
    counts = np.array([directions[d] for d in sorted(directions.keys())])
    
    # Close the polar plot
    angles = np.append(angles, angles[0])
    counts = np.append(counts, counts[0])
    
    # Clear and recreate as polar
    ax.clear()
    ax.set_facecolor(PANEL_COLOR)
    
    # Plot as polar in regular axes
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'gray', linewidth=0.5, alpha=0.3)
    
    # Normalize counts
    max_count = max(counts) if max(counts) > 0 else 1
    norm_counts = counts / max_count
    
    # Plot direction responses
    x = norm_counts * np.cos(angles)
    y = norm_counts * np.sin(angles)
    ax.fill(x, y, alpha=0.3, color="#4ecdc4")
    ax.plot(x, y, color="#4ecdc4", linewidth=2)
    
    # Add direction labels
    for i, (ang, cnt) in enumerate(zip(angles[:-1], counts[:-1])):
        label_r = 1.15
        ax.text(label_r * np.cos(ang), label_r * np.sin(ang), 
                f"{int(ang * 180 / np.pi)}°", ha="center", va="center", fontsize=6)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DSGC", fontsize=9)


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_unit_image(
    hdf5_path: Path,
    unit_id: str,
    output_path: Path,
    plot_types: Dict[str, bool],
) -> Optional[Path]:
    """
    Generate combined image for a single unit.
    
    Args:
        hdf5_path: Path to HDF5 file
        unit_id: Unit ID to plot
        output_path: Path to save the image
        plot_types: Dict of plot type -> enabled
    
    Returns:
        Path to saved image, or None if failed
    """
    # Get enabled plots
    enabled_plots = [k for k, v in plot_types.items() if v]
    n_plots = len(enabled_plots)
    
    if n_plots == 0:
        return None
    
    # Load unit data
    data = load_unit_data(hdf5_path, unit_id)
    
    # Create figure
    fig_width = 4 * n_plots
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 4), facecolor=BG_COLOR)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot each enabled type
    plot_funcs = {
        "eimage_sta": plot_eimage_sta,
        "geometry": plot_geometry,
        "ap_tracking": plot_ap_tracking,
        "dsgc": plot_dsgc,
    }
    
    for ax, plot_type in zip(axes, enabled_plots):
        if plot_type in plot_funcs:
            plot_funcs[plot_type](ax, data)
    
    # Title
    fig.suptitle(f"{unit_id}", fontsize=12, fontweight="bold", color="white")
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    
    return output_path


def generate_manual_label_images(
    hdf5_path: Path,
    output_dir: Path,
    plot_types: Dict[str, bool] = None,
) -> List[Path]:
    """
    Generate manual label images for all units in an HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Directory to save images
        plot_types: Dict of plot type -> enabled (default: all True except dsgc)
    
    Returns:
        List of generated image paths
    """
    if plot_types is None:
        plot_types = {
            "eimage_sta": True,
            "geometry": True,
            "ap_tracking": True,
            "dsgc": False,
        }
    
    # Get all unit IDs
    unit_ids = get_unit_ids(hdf5_path)
    
    if not unit_ids:
        logger.warning(f"No units found in {hdf5_path}")
        return []
    
    logger.info(f"Generating manual label images for {len(unit_ids)} units...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    
    for unit_id in tqdm(unit_ids, desc="Generating plots", leave=False):
        output_path = output_dir / f"{unit_id}.png"
        
        try:
            result = generate_unit_image(hdf5_path, unit_id, output_path, plot_types)
            if result:
                generated.append(result)
        except Exception as e:
            logger.warning(f"Failed to generate image for {unit_id}: {e}")
    
    logger.info(f"Generated {len(generated)} images in {output_dir}")
    return generated


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate manual label images")
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 file")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--unit", type=str, help="Single unit ID to process")
    parser.add_argument("--no-sta", action="store_true", help="Disable STA plot")
    parser.add_argument("--no-geometry", action="store_true", help="Disable geometry plot")
    parser.add_argument("--no-ap", action="store_true", help="Disable AP tracking plot")
    parser.add_argument("--dsgc", action="store_true", help="Enable DSGC plot")
    
    args = parser.parse_args()
    
    if not args.hdf5_path.exists():
        print(f"Error: File not found: {args.hdf5_path}")
        return 1
    
    output_dir = args.output or args.hdf5_path.parent / f"{args.hdf5_path.stem}_manual_label"
    
    plot_types = {
        "eimage_sta": not args.no_sta,
        "geometry": not args.no_geometry,
        "ap_tracking": not args.no_ap,
        "dsgc": args.dsgc,
    }
    
    if args.unit:
        output_path = output_dir / f"{args.unit}.png"
        generate_unit_image(args.hdf5_path, args.unit, output_path, plot_types)
    else:
        generate_manual_label_images(args.hdf5_path, output_dir, plot_types)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
