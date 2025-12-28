"""
Validation plots for DSGC Direction Sectioning.

Generates a combined visualization figure to verify direction sectioning results:
- Movie frames at on/off times with cell center overlay
- Spike raster by direction
- Polar direction tuning curve
- PSTH (Peri-Stimulus Time Histogram)
- Trial consistency heatmap

Usage:
    python dsgc_validation_plots.py [unit_id]    # Single unit
    python dsgc_validation_plots.py --all        # All units
    
Example:
    python dsgc_validation_plots.py unit_001
    python dsgc_validation_plots.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for development
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import pickle
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Circle

from hdmea.features.dsgc_direction import (
    DIRECTION_LIST,
    N_DIRECTIONS,
    N_REPETITIONS,
    DEFAULT_STIMULI_DIR,
    DEFAULT_MOVIE_NAME,
    STA_GEOMETRY_FEATURE,
)
from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM


# =============================================================================
# Configuration
# =============================================================================

EXPORT_DIR = Path(__file__).parent / "export"
HDF5_FILE = EXPORT_DIR / "2024.08.08-10.40.20-Rec.h5"
MOVIE_PATH = DEFAULT_STIMULI_DIR / f"{DEFAULT_MOVIE_NAME}.npy"
# Use _hd version (single pixel) instead of _area_hd (25x25 area average)
ON_OFF_DICT_PATH = DEFAULT_STIMULI_DIR / f"{DEFAULT_MOVIE_NAME}_on_off_dict_hd.pkl"

# Plot styling
plt.style.use("dark_background")
COLORMAP = "viridis"
SPIKE_COLOR = "#00FF88"
ON_COLOR = "#FF6B6B"
OFF_COLOR = "#4ECDC4"
CENTER_COLOR = "#FFD93D"
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_movie() -> np.ndarray:
    """Load the moving bar movie."""
    return np.load(MOVIE_PATH)


def load_on_off_dict() -> Dict[Tuple[int, int], Dict[str, List[int]]]:
    """Load the per-pixel on/off timing dictionary."""
    with open(ON_OFF_DICT_PATH, "rb") as f:
        return pickle.load(f)


def load_unit_data(
    hdf5_path: Path,
    unit_id: str,
    movie_name: str = DEFAULT_MOVIE_NAME,
) -> Dict:
    """Load all data needed for validation plots for a single unit."""
    with h5py.File(hdf5_path, "r") as f:
        section_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/direction_section"
        
        if section_path not in f:
            raise ValueError(f"No direction_section found for {unit_id}")
        
        section_group = f[section_path]
        cell_center = (
            int(section_group.attrs["cell_center_row"]),
            int(section_group.attrs["cell_center_col"]),
        )
        
        # Load spike data for each direction
        direction_data = {}
        for direction in DIRECTION_LIST:
            dir_group = section_group[str(direction)]
            trials = []
            for rep in range(N_REPETITIONS):
                spikes = dir_group[f"trials/{rep}"][:]
                trials.append(spikes)
            
            bounds = dir_group["section_bounds"][:]
            direction_data[direction] = {
                "trials": trials,
                "bounds": bounds,
            }
        
        # Get frame_timestamps and movie start
        frame_timestamps = f["metadata/frame_timestamps"][:]
        section_time = f[f"stimulus/section_time/{movie_name}"][:]
        movie_start_sample = section_time[0, 0]
    
    movie_start_frame = int(convert_sample_index_to_frame(
        np.array([movie_start_sample]), frame_timestamps
    )[0]) + PRE_MARGIN_FRAME_NUM
    
    # Load on/off times from dictionary
    on_off_dict = load_on_off_dict()
    pixel_timing = on_off_dict[cell_center]
    
    on_off_times = {}
    for direction in DIRECTION_LIST:
        on_off_times[direction] = []
    
    for trial_idx in range(24):
        direction_idx = trial_idx % N_DIRECTIONS
        direction = DIRECTION_LIST[direction_idx]
        on_time = pixel_timing["on_peak_location"][trial_idx]
        off_time = pixel_timing["off_peak_location"][trial_idx]
        on_off_times[direction].append((on_time, off_time))
    
    return {
        "cell_center": cell_center,
        "direction_data": direction_data,
        "on_off_times": on_off_times,
        "frame_timestamps": frame_timestamps,
        "movie_start_frame": movie_start_frame,
    }


def get_all_units_with_direction_section(hdf5_path: Path) -> List[str]:
    """Get list of all units that have direction_section data."""
    units = []
    with h5py.File(hdf5_path, "r") as f:
        for unit_id in f["units"].keys():
            section_path = f"units/{unit_id}/spike_times_sectioned/{DEFAULT_MOVIE_NAME}/direction_section"
            if section_path in f:
                units.append(unit_id)
    return sorted(units)


# =============================================================================
# Combined Plot Function
# =============================================================================

def generate_combined_plot(
    unit_id: str,
    unit_data: Dict,
    movie: np.ndarray,
    output_path: Path,
) -> None:
    """
    Generate a single combined figure with all validation plots.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Movie Frames: 8 directions × 3 trials × ON/OFF                 │
    │  (8 columns for directions, 6 rows for trials×on/off)          │
    ├─────────────────────────┬───────────────────────────────────────┤
    │  Polar Tuning           │  Trial Consistency Heatmap            │
    ├─────────────────────────┴───────────────────────────────────────┤
    │  Spike Raster (8 directions)                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  PSTH (8 directions)                                            │
    └─────────────────────────────────────────────────────────────────┘
    """
    fig = plt.figure(figsize=(24, 36), facecolor=BG_COLOR)
    
    # Create grid spec - movie section now larger
    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[2.5, 1, 1, 1],  # More space for movie frames
        width_ratios=[1, 1.2],
        hspace=0.2,
        wspace=0.2,
    )
    
    cell_center = unit_data["cell_center"]
    direction_data = unit_data["direction_data"]
    on_off_times = unit_data["on_off_times"]
    frame_timestamps = unit_data["frame_timestamps"]
    movie_start_frame = unit_data["movie_start_frame"]
    
    # Title
    fig.suptitle(
        f"DSGC Direction Sectioning Validation - {unit_id}\n"
        f"Cell Center: ({cell_center[0]}, {cell_center[1]})",
        fontsize=18,
        fontweight="bold",
        color="white",
        y=0.98,
    )
    
    # =========================================================================
    # Panel 1: Movie Frames - ALL 8 directions × 3 trials × ON/OFF
    # Layout: 8 columns (directions), 6 rows (3 trials × 2 on/off)
    # =========================================================================
    gs_movie = gridspec.GridSpecFromSubplotSpec(
        6, 8,  # 6 rows (trial1-ON, trial1-OFF, trial2-ON, trial2-OFF, trial3-ON, trial3-OFF), 8 cols (directions)
        subplot_spec=gs[0, :],
        hspace=0.08,
        wspace=0.05,
    )
    
    for dir_idx, direction in enumerate(DIRECTION_LIST):
        for trial_idx in range(N_REPETITIONS):
            on_time, off_time = on_off_times[direction][trial_idx]
            
            # ON frame (even rows: 0, 2, 4)
            row_on = trial_idx * 2
            ax_on = fig.add_subplot(gs_movie[row_on, dir_idx])
            ax_on.imshow(movie[on_time], cmap="gray", vmin=0, vmax=255)
            circle = Circle((cell_center[1], cell_center[0]), radius=4, fill=False, color=CENTER_COLOR, linewidth=1.5)
            ax_on.add_patch(circle)
            ax_on.plot(cell_center[1], cell_center[0], ".", color=CENTER_COLOR, markersize=2)
            ax_on.axis("off")
            
            # Add labels
            if trial_idx == 0:  # Top row - add direction label
                ax_on.set_title(f"{direction}°", fontsize=9, fontweight="bold", color="white", pad=2)
            if dir_idx == 0:  # Left column - add trial/on label
                ax_on.text(-0.1, 0.5, f"T{trial_idx+1}\nON", transform=ax_on.transAxes,
                          fontsize=7, color=ON_COLOR, ha="right", va="center")
            
            # OFF frame (odd rows: 1, 3, 5)
            row_off = trial_idx * 2 + 1
            ax_off = fig.add_subplot(gs_movie[row_off, dir_idx])
            ax_off.imshow(movie[off_time], cmap="gray", vmin=0, vmax=255)
            circle = Circle((cell_center[1], cell_center[0]), radius=4, fill=False, color=CENTER_COLOR, linewidth=1.5)
            ax_off.add_patch(circle)
            ax_off.plot(cell_center[1], cell_center[0], ".", color=CENTER_COLOR, markersize=2)
            ax_off.axis("off")
            
            if dir_idx == 0:  # Left column - add trial/off label
                ax_off.text(-0.1, 0.5, f"OFF", transform=ax_off.transAxes,
                           fontsize=7, color=OFF_COLOR, ha="right", va="center")
    
    # =========================================================================
    # Panel 2: Polar Tuning (second row, left)
    # =========================================================================
    ax_polar = fig.add_subplot(gs[1, 0], projection="polar")
    ax_polar.set_facecolor(PANEL_COLOR)
    
    spike_counts = []
    for direction in DIRECTION_LIST:
        total = sum(len(t) for t in direction_data[direction]["trials"])
        spike_counts.append(total)
    
    angles = np.deg2rad(DIRECTION_LIST)
    angles_closed = np.append(angles, angles[0])
    counts_closed = spike_counts + [spike_counts[0]]
    
    ax_polar.plot(angles_closed, counts_closed, "o-", color=SPIKE_COLOR, linewidth=2, markersize=8)
    ax_polar.fill(angles_closed, counts_closed, color=SPIKE_COLOR, alpha=0.3)
    
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_xticks(angles)
    ax_polar.set_xticklabels([f"{d}°" for d in DIRECTION_LIST], fontsize=9, color="white")
    ax_polar.tick_params(colors="white")
    ax_polar.grid(True, alpha=0.3)
    
    # DSI annotation
    max_count = max(spike_counts) if spike_counts else 0
    min_count = min(spike_counts) if spike_counts else 0
    if max_count > 0:
        dsi = (max_count - min_count) / (max_count + min_count)
        preferred_dir = DIRECTION_LIST[spike_counts.index(max_count)]
        ax_polar.set_title(
            f"Direction Tuning (DSI={dsi:.2f}, Pref={preferred_dir}°)",
            fontsize=11,
            fontweight="bold",
            color="white",
            pad=15,
        )
    else:
        ax_polar.set_title("Direction Tuning", fontsize=11, fontweight="bold", color="white", pad=15)
    
    # =========================================================================
    # Panel 3: Trial Consistency Heatmap (second row, right)
    # =========================================================================
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.set_facecolor(PANEL_COLOR)
    
    counts_matrix = np.zeros((N_DIRECTIONS, N_REPETITIONS))
    for i, direction in enumerate(DIRECTION_LIST):
        for rep in range(N_REPETITIONS):
            counts_matrix[i, rep] = len(direction_data[direction]["trials"][rep])
    
    im = ax_heat.imshow(counts_matrix, cmap="YlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.set_label("Spike Count", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    
    ax_heat.set_xticks(range(N_REPETITIONS))
    ax_heat.set_xticklabels([f"Rep {i+1}" for i in range(N_REPETITIONS)], fontsize=9, color="white")
    ax_heat.set_yticks(range(N_DIRECTIONS))
    ax_heat.set_yticklabels([f"{d}°" for d in DIRECTION_LIST], fontsize=9, color="white")
    ax_heat.set_xlabel("Repetition", fontsize=10, color="white")
    ax_heat.set_ylabel("Direction", fontsize=10, color="white")
    
    for i in range(N_DIRECTIONS):
        for j in range(N_REPETITIONS):
            count = int(counts_matrix[i, j])
            text_color = "white" if count < counts_matrix.max() / 2 else "black"
            ax_heat.text(j, i, str(count), ha="center", va="center", color=text_color, fontsize=9)
    
    mean_count = counts_matrix.mean()
    std_count = counts_matrix.std()
    cv = std_count / mean_count if mean_count > 0 else 0
    ax_heat.set_title(
        f"Trial Consistency (Mean={mean_count:.1f}, CV={cv:.2f})",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    
    # =========================================================================
    # Panel 4: Spike Raster (third row, spans both columns)
    # =========================================================================
    gs_raster = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[2, :], hspace=0.3, wspace=0.25)
    
    for i, direction in enumerate(DIRECTION_LIST):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(gs_raster[row, col])
        ax.set_facecolor(PANEL_COLOR)
        
        dir_data = direction_data[direction]
        
        for rep in range(N_REPETITIONS):
            spike_samples = dir_data["trials"][rep]
            on_time, off_time = on_off_times[direction][rep]
            
            if len(spike_samples) > 0:
                spike_frames = convert_sample_index_to_frame(spike_samples, frame_timestamps)
                spike_frames_movie = spike_frames - movie_start_frame
                spike_frames_rel = spike_frames_movie - on_time
                
                ax.eventplot(
                    [spike_frames_rel],
                    lineoffsets=rep,
                    linelengths=0.8,
                    colors=SPIKE_COLOR,
                    linewidths=1.5,
                )
        
        # ON/OFF lines
        on_time_0, off_time_0 = on_off_times[direction][0]
        ax.axvline(0, color=ON_COLOR, linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(off_time_0 - on_time_0, color=OFF_COLOR, linestyle="--", alpha=0.7, linewidth=1)
        
        ax.set_xlim(-12, 32)
        ax.set_ylim(-0.5, N_REPETITIONS - 0.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["1", "2", "3"], fontsize=8, color="white")
        ax.set_title(f"{direction}°", fontsize=10, fontweight="bold", color="white")
        ax.tick_params(colors="white", labelsize=8)
        
        if row == 1:
            ax.set_xlabel("Frames", fontsize=8, color="white")
        if col == 0:
            ax.set_ylabel("Rep", fontsize=8, color="white")
    
    # =========================================================================
    # Panel 5: PSTH (fourth row, spans both columns)
    # =========================================================================
    gs_psth = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[3, :], hspace=0.3, wspace=0.25)
    
    bin_size = 2
    bin_edges = np.arange(-12, 35, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for i, direction in enumerate(DIRECTION_LIST):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(gs_psth[row, col])
        ax.set_facecolor(PANEL_COLOR)
        
        dir_data = direction_data[direction]
        all_spikes_rel = []
        
        for rep in range(N_REPETITIONS):
            spike_samples = dir_data["trials"][rep]
            on_time, off_time = on_off_times[direction][rep]
            
            if len(spike_samples) > 0:
                spike_frames = convert_sample_index_to_frame(spike_samples, frame_timestamps)
                spike_frames_movie = spike_frames - movie_start_frame
                spike_frames_rel = spike_frames_movie - on_time
                all_spikes_rel.extend(spike_frames_rel)
        
        if len(all_spikes_rel) > 0:
            counts, _ = np.histogram(all_spikes_rel, bins=bin_edges)
            rate = counts / N_REPETITIONS
        else:
            rate = np.zeros(len(bin_centers))
        
        ax.bar(bin_centers, rate, width=bin_size * 0.9, color=SPIKE_COLOR, alpha=0.8)
        
        on_time_0, off_time_0 = on_off_times[direction][0]
        ax.axvspan(0, off_time_0 - on_time_0, alpha=0.15, color=ON_COLOR)
        ax.axvline(0, color=ON_COLOR, linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(off_time_0 - on_time_0, color=OFF_COLOR, linestyle="--", alpha=0.7, linewidth=1)
        
        ax.set_xlim(-12, 32)
        ax.set_title(f"{direction}°", fontsize=10, fontweight="bold", color="white")
        ax.tick_params(colors="white", labelsize=8)
        
        if row == 1:
            ax.set_xlabel("Frames", fontsize=8, color="white")
        if col == 0:
            ax.set_ylabel("Spikes/trial", fontsize=8, color="white")
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


# =============================================================================
# Main Runner
# =============================================================================

def generate_plots_for_unit(
    unit_id: str,
    movie: np.ndarray,
    hdf5_path: Path,
    output_dir: Path,
) -> bool:
    """Generate combined validation plot for a single unit."""
    try:
        unit_data = load_unit_data(hdf5_path, unit_id)
        output_path = output_dir / f"{unit_id}_validation.png"
        generate_combined_plot(unit_id, unit_data, movie, output_path)
        return True
    except Exception as e:
        print(f"  ERROR for {unit_id}: {e}")
        return False


def main():
    """Main entry point."""
    print("=" * 70)
    print("DSGC Direction Sectioning Validation Plots")
    print("=" * 70)
    
    # Check if --all flag
    plot_all = "--all" in sys.argv or len(sys.argv) == 1
    
    # Get units to process
    if plot_all:
        unit_ids = get_all_units_with_direction_section(HDF5_FILE)
        print(f"\nProcessing ALL {len(unit_ids)} units with direction_section data")
    else:
        unit_ids = [sys.argv[1]]
        print(f"\nProcessing single unit: {unit_ids[0]}")
    
    # Create output directory
    output_dir = EXPORT_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load movie once (shared across all units)
    print("\nLoading movie...")
    movie = load_movie()
    print(f"  Movie shape: {movie.shape}")
    
    # Process each unit
    print(f"\nGenerating plots...")
    success_count = 0
    
    for i, unit_id in enumerate(unit_ids):
        print(f"  [{i+1}/{len(unit_ids)}] {unit_id}...", end=" ")
        if generate_plots_for_unit(unit_id, movie, HDF5_FILE, output_dir):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
    
    print(f"\n{'=' * 70}")
    print(f"Complete: {success_count}/{len(unit_ids)} units processed")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    return 0 if success_count == len(unit_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
