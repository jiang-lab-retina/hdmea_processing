#!/usr/bin/env python
"""
DSGC Direction Sectioning Validation Plots.

Generates visualizations of direction sectioning results:
- Spike raster by direction
- Polar direction tuning curve
- Trial consistency heatmap
- PSTH per direction

Usage:
    python plot_dsgc.py <hdf5_path> [--output-dir <path>] [--unit <unit_id>]
    
Example:
    python plot_dsgc.py ../test_output/2024.08.08-10.40.20-Rec_final.h5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm

# Add src to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# =============================================================================
# Constants
# =============================================================================

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8
N_REPETITIONS = 3

DEFAULT_MOVIE_NAME = "moving_h_bar_s5_d8_3x"

# =============================================================================
# Styling
# =============================================================================

plt.style.use("dark_background")

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
SPIKE_COLOR = "#00FF88"
ON_COLOR = "#FF6B6B"
OFF_COLOR = "#4ECDC4"
CENTER_COLOR = "#FFD93D"


# =============================================================================
# Data Loading
# =============================================================================

def load_unit_dsgc_data(hdf5_path: Path, unit_id: str, 
                        movie_name: str = DEFAULT_MOVIE_NAME) -> Optional[Dict[str, Any]]:
    """Load DSGC direction sectioning data for a unit."""
    data = {}
    
    with h5py.File(hdf5_path, "r") as f:
        section_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/direction_section"
        
        if section_path not in f:
            return None
        
        section = f[section_path]
        
        # Get cell center
        data["center_row"] = None
        data["center_col"] = None
        
        # Try different key formats
        for key in ["center_row_300", "cell_center_row"]:
            if key in section:
                data["center_row"] = int(section[key][()])
                break
        for key in ["center_col_300", "cell_center_col"]:
            if key in section:
                data["center_col"] = int(section[key][()])
                break
        
        # Load direction data
        data["directions"] = {}
        
        # Check if directions group exists
        if "directions" in section:
            dir_grp = section["directions"]
            for dir_key in dir_grp:
                try:
                    direction = int(dir_key)
                except ValueError:
                    # Handle "dir_0" format
                    direction = int(dir_key.replace("dir_", ""))
                
                dir_data = dir_grp[dir_key]
                trials = []
                
                # Load trials
                if "trials" in dir_data:
                    trials_grp = dir_data["trials"]
                    # Sort trial keys numerically (they could be '0', '1', '2' or 'trial_0', etc.)
                    trial_keys = sorted(trials_grp.keys(), key=lambda x: int(x.replace('trial_', '')) if 'trial_' in x else int(x))
                    for trial_key in trial_keys:
                        trial_spikes = trials_grp[trial_key][()]
                        if isinstance(trial_spikes, np.ndarray):
                            trials.append(trial_spikes.tolist())
                        else:
                            trials.append([trial_spikes])
                
                # Load bounds if available
                bounds = []
                if "bounds" in dir_data or "section_bounds" in dir_data:
                    bounds_key = "bounds" if "bounds" in dir_data else "section_bounds"
                    bounds_data = dir_data[bounds_key][()]
                    if bounds_data is not None:
                        bounds = bounds_data.tolist() if hasattr(bounds_data, 'tolist') else []
                
                data["directions"][direction] = {
                    "trials": trials,
                    "bounds": bounds,
                }
        
        # Try loading directions at top level (keys like '0', '45', '90', etc.)
        if not data["directions"]:
            for dir_key in DIRECTION_LIST:
                str_key = str(dir_key)
                if str_key in section:
                    dir_data = section[str_key]
                    trials = []
                    bounds = []
                    
                    if "trials" in dir_data:
                        trials_grp = dir_data["trials"]
                        # Sort trial keys numerically (could be '0', '1', '2' or 'trial_0', etc.)
                        trial_keys = sorted(trials_grp.keys(), key=lambda x: int(x.replace('trial_', '')) if 'trial_' in x else int(x))
                        for trial_key in trial_keys:
                            trial_spikes = trials_grp[trial_key][()]
                            if isinstance(trial_spikes, np.ndarray):
                                trials.append(trial_spikes.tolist())
                            else:
                                trials.append([trial_spikes])
                    
                    # Load bounds if available
                    if "section_bounds" in dir_data:
                        bounds_data = dir_data["section_bounds"][()]
                        if bounds_data is not None:
                            bounds = bounds_data.tolist() if hasattr(bounds_data, 'tolist') else []
                    
                    data["directions"][dir_key] = {
                        "trials": trials,
                        "bounds": bounds,
                    }
    
    # Validate we have direction data
    if not data["directions"]:
        return None
    
    return data


def get_unit_ids_with_dsgc(hdf5_path: Path, movie_name: str = DEFAULT_MOVIE_NAME) -> List[str]:
    """Get list of unit IDs that have DSGC data."""
    unit_ids = []
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            return []
        for unit_id in f["units"]:
            section_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/direction_section"
            if section_path in f:
                unit_ids.append(unit_id)
    return sorted(unit_ids)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_polar_tuning(ax, direction_data: Dict[int, Dict], title: str = "Direction Tuning"):
    """Plot polar tuning curve."""
    spike_counts = []
    for direction in DIRECTION_LIST:
        if direction in direction_data:
            total = sum(len(t) for t in direction_data[direction].get("trials", []))
        else:
            total = 0
        spike_counts.append(total)
    
    angles = np.deg2rad(DIRECTION_LIST)
    angles_closed = np.append(angles, angles[0])
    counts_closed = spike_counts + [spike_counts[0]]
    
    ax.plot(angles_closed, counts_closed, "o-", color=SPIKE_COLOR, linewidth=2, markersize=8)
    ax.fill(angles_closed, counts_closed, color=SPIKE_COLOR, alpha=0.3)
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{d}" for d in DIRECTION_LIST], fontsize=8, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3)
    
    # DSI calculation
    max_count = max(spike_counts) if spike_counts else 0
    min_count = min(spike_counts) if spike_counts else 0
    if max_count > 0:
        dsi = (max_count - min_count) / (max_count + min_count)
        preferred_dir = DIRECTION_LIST[spike_counts.index(max_count)]
        ax.set_title(f"{title}\nDSI={dsi:.2f}, Pref={preferred_dir}deg", fontsize=9, color="white")
    else:
        ax.set_title(title, fontsize=9, color="white")


def plot_trial_heatmap(ax, direction_data: Dict[int, Dict]):
    """Plot trial consistency heatmap."""
    counts_matrix = np.zeros((N_DIRECTIONS, N_REPETITIONS))
    
    for i, direction in enumerate(DIRECTION_LIST):
        if direction in direction_data:
            trials = direction_data[direction].get("trials", [])
            for rep in range(min(N_REPETITIONS, len(trials))):
                counts_matrix[i, rep] = len(trials[rep])
    
    im = ax.imshow(counts_matrix, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spikes")
    
    ax.set_xticks(range(N_REPETITIONS))
    ax.set_xticklabels([f"R{i+1}" for i in range(N_REPETITIONS)], fontsize=8)
    ax.set_yticks(range(N_DIRECTIONS))
    ax.set_yticklabels([f"{d}" for d in DIRECTION_LIST], fontsize=8)
    ax.set_xlabel("Repetition", fontsize=9)
    ax.set_ylabel("Direction", fontsize=9)
    
    # Add count labels
    for i in range(N_DIRECTIONS):
        for j in range(N_REPETITIONS):
            count = int(counts_matrix[i, j])
            text_color = "white" if count < counts_matrix.max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=text_color, fontsize=7)
    
    mean_count = counts_matrix.mean()
    cv = counts_matrix.std() / mean_count if mean_count > 0 else 0
    ax.set_title(f"Trial Consistency (Mean={mean_count:.1f}, CV={cv:.2f})", fontsize=9)


def plot_raster_grid(axes, direction_data: Dict[int, Dict]):
    """Plot spike rasters for all directions, relative to stimulus onset."""
    for i, direction in enumerate(DIRECTION_LIST):
        ax = axes[i]
        ax.set_facecolor(PANEL_COLOR)
        
        if direction not in direction_data:
            ax.set_title(f"{direction}deg - No data", fontsize=9)
            continue
        
        trials = direction_data[direction].get("trials", [])
        bounds = direction_data[direction].get("bounds", [])
        
        for rep in range(min(N_REPETITIONS, len(trials))):
            spikes = np.array(trials[rep])
            if len(spikes) > 0:
                # Convert to relative frame numbers if bounds available
                if rep < len(bounds) and len(bounds[rep]) >= 2:
                    start_frame = bounds[rep][0]
                    spikes_rel = spikes - start_frame
                else:
                    spikes_rel = spikes
                
                ax.eventplot(
                    [spikes_rel],
                    lineoffsets=rep,
                    linelengths=0.8,
                    colors=SPIKE_COLOR,
                    linewidths=1.5,
                )
        
        # Mark stimulus window (typically ~40 frames for this stimulus)
        ax.axvline(0, color=ON_COLOR, linestyle="--", alpha=0.7, linewidth=1, label="ON")
        ax.axvline(40, color=OFF_COLOR, linestyle="--", alpha=0.7, linewidth=1, label="OFF")
        
        ax.set_xlim(-10, 60)
        ax.set_ylim(-0.5, N_REPETITIONS - 0.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["1", "2", "3"], fontsize=7)
        ax.set_title(f"{direction}deg", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)


def plot_psth_grid(axes, direction_data: Dict[int, Dict]):
    """Plot PSTH for all directions, relative to stimulus onset."""
    bin_size = 2
    
    # Find global max for consistent y-axis, using relative spike times
    all_counts = []
    
    for i, direction in enumerate(DIRECTION_LIST):
        ax = axes[i]
        ax.set_facecolor(PANEL_COLOR)
        
        if direction not in direction_data:
            continue
        
        trials = direction_data[direction].get("trials", [])
        bounds = direction_data[direction].get("bounds", [])
        all_spikes_rel = []
        
        for rep, trial in enumerate(trials):
            spikes = np.array(trial)
            if len(spikes) > 0:
                # Convert to relative frame numbers
                if rep < len(bounds) and len(bounds[rep]) >= 2:
                    start_frame = bounds[rep][0]
                    spikes_rel = spikes - start_frame
                else:
                    spikes_rel = spikes
                all_spikes_rel.extend(spikes_rel.tolist())
        
        if len(all_spikes_rel) > 0:
            # Compute histogram over fixed relative window
            bins = np.arange(-10, 60, bin_size)
            counts, edges = np.histogram(all_spikes_rel, bins=bins)
            all_counts.extend(counts)
    
    max_count = max(all_counts) if all_counts else 1
    
    for i, direction in enumerate(DIRECTION_LIST):
        ax = axes[i]
        
        if direction not in direction_data:
            continue
        
        trials = direction_data[direction].get("trials", [])
        bounds = direction_data[direction].get("bounds", [])
        all_spikes_rel = []
        
        for rep, trial in enumerate(trials):
            spikes = np.array(trial)
            if len(spikes) > 0:
                # Convert to relative frame numbers
                if rep < len(bounds) and len(bounds[rep]) >= 2:
                    start_frame = bounds[rep][0]
                    spikes_rel = spikes - start_frame
                else:
                    spikes_rel = spikes
                all_spikes_rel.extend(spikes_rel.tolist())
        
        if len(all_spikes_rel) > 0:
            bins = np.arange(-10, 60, bin_size)
            counts, edges = np.histogram(all_spikes_rel, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            
            ax.bar(centers, counts, width=bin_size * 0.9, color=SPIKE_COLOR, alpha=0.8)
            ax.axvline(0, color=ON_COLOR, linestyle="--", alpha=0.7, linewidth=1)
            ax.axvline(40, color=OFF_COLOR, linestyle="--", alpha=0.7, linewidth=1)
            ax.set_xlim(-10, 60)
            ax.set_ylim(0, max_count * 1.1)
        
        ax.set_title(f"{direction}deg", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)


def plot_unit_dsgc(data: Dict[str, Any], unit_id: str, output_dir: Path):
    """Generate complete DSGC validation plot for one unit."""
    fig = plt.figure(figsize=(16, 12), facecolor=BG_COLOR)
    
    # Create grid: 4 rows
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1],
                          hspace=0.35, wspace=0.3)
    
    direction_data = data["directions"]
    center = (data.get("center_row"), data.get("center_col"))
    
    # Row 1: Polar + Heatmap
    ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_polar.set_facecolor(PANEL_COLOR)
    plot_polar_tuning(ax_polar, direction_data, "Polar Direction Tuning")
    
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_heat.set_facecolor(PANEL_COLOR)
    plot_trial_heatmap(ax_heat, direction_data)
    
    # Row 2-3: Rasters (2x4 grid)
    gs_raster = gs[1, :].subgridspec(2, 4, hspace=0.3, wspace=0.25)
    raster_axes = []
    for r in range(2):
        for c in range(4):
            ax = fig.add_subplot(gs_raster[r, c])
            raster_axes.append(ax)
    plot_raster_grid(raster_axes, direction_data)
    
    # Row 4: PSTH (2x4 grid)
    gs_psth = gs[2, :].subgridspec(2, 4, hspace=0.3, wspace=0.25)
    psth_axes = []
    for r in range(2):
        for c in range(4):
            ax = fig.add_subplot(gs_psth[r, c])
            psth_axes.append(ax)
    plot_psth_grid(psth_axes, direction_data)
    
    # Row 5: Summary stats
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis("off")
    ax_summary.set_facecolor(PANEL_COLOR)
    
    # Calculate stats
    total_spikes = sum(
        sum(len(t) for t in d.get("trials", []))
        for d in direction_data.values()
    )
    spike_counts = [
        sum(len(t) for t in direction_data.get(d, {}).get("trials", []))
        for d in DIRECTION_LIST
    ]
    max_count = max(spike_counts) if spike_counts else 0
    min_count = min(spike_counts) if spike_counts else 0
    dsi = (max_count - min_count) / (max_count + min_count) if max_count > 0 else 0
    preferred = DIRECTION_LIST[spike_counts.index(max_count)] if max_count > 0 else "N/A"
    
    summary_text = (
        f"Summary Statistics\n"
        f"{'='*40}\n"
        f"Total Spikes: {total_spikes}\n"
        f"Preferred Direction: {preferred}deg\n"
        f"Direction Selectivity Index (DSI): {dsi:.3f}\n"
        f"Max Spike Count: {max_count} ({preferred}deg)\n"
        f"Min Spike Count: {min_count}\n"
        f"Cell Center: ({center[0]}, {center[1]})"
    )
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, ha="center", va="center", family="monospace",
                   bbox=dict(boxstyle="round", facecolor=PANEL_COLOR, alpha=0.8))
    
    # Title
    fig.suptitle(f"{unit_id} - DSGC Direction Sectioning Validation",
                fontsize=14, fontweight="bold", y=0.98)
    
    # Save
    output_path = output_dir / f"{unit_id}_dsgc.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def run_validation(
    hdf5_path: Path,
    output_dir: Optional[Path] = None,
    unit_id: Optional[str] = None,
    movie_name: str = DEFAULT_MOVIE_NAME,
):
    """Run DSGC validation plots."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / hdf5_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get units to process
    if unit_id:
        unit_ids = [unit_id]
    else:
        unit_ids = get_unit_ids_with_dsgc(hdf5_path, movie_name)
    
    print(f"Processing {len(unit_ids)} units for DSGC validation...")
    
    success = 0
    for uid in tqdm(unit_ids, desc="DSGC plots"):
        data = load_unit_dsgc_data(hdf5_path, uid, movie_name)
        if data is None:
            continue
        
        try:
            plot_unit_dsgc(data, uid, output_dir)
            success += 1
        except Exception as e:
            print(f"  Error for {uid}: {e}")
    
    print(f"Generated {success}/{len(unit_ids)} plots in {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="DSGC validation plots")
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--unit", type=str, help="Single unit ID to process")
    parser.add_argument("--movie", type=str, default=DEFAULT_MOVIE_NAME,
                       help="Movie name for direction sectioning")
    
    args = parser.parse_args()
    
    if not args.hdf5_path.exists():
        print(f"Error: File not found: {args.hdf5_path}")
        return 1
    
    run_validation(args.hdf5_path, args.output_dir, args.unit, args.movie)
    return 0


if __name__ == "__main__":
    sys.exit(main())

