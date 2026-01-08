"""
Firing Rate Pipeline

Combined pipeline that:
1. Generates inter-frame firing rates from HDF5 files
2. Reshapes into movie-based format (2D arrays per movie)
3. Plots firing rates with trial traces and means

Usage:
    python pipeline_firing_rate.py                      # Process all files
    python pipeline_firing_rate.py --start 0 --end 10   # Process files 0-9
    python pipeline_firing_rate.py --start 5            # Process from file 5 to end
    python pipeline_firing_rate.py --list               # List all files with indices
    python pipeline_firing_rate.py --no-plots           # Skip plot generation
    python pipeline_firing_rate.py --strict             # Fail on validation errors
    python pipeline_firing_rate.py --incremental        # Skip processed files, append new
    python pipeline_firing_rate.py --start-step 2       # Start from step 2 (reshape)

Note: By default, units with inconsistent trial lengths are excluded with a warning.
      Use --strict to fail instead, or --skip-validation to skip validation entirely.
"""

import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Input/Output directories
HDF5_DIR = project_root / "Projects/unified_pipeline/export_dsgc_sta_updated"
#HDF5_DIR = project_root / "Projects/unified_pipeline/export_all_steps"

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Movies to process
MOVIES_FRAME_ALIGNED = [
    "baseline_127",
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x",
    "step_up_5s_5i_b0_30x",
    "step_up_5s_5i_b0_3x",
]
MOVIE_DIRECTION_SECTION = "moving_h_bar_s5_d8_3x"
MOVIE_SAMPLE_BASED = "iprgc_test"
IPRGC_TARGET_RATE_HZ = 60.0
IPRGC_EXPECTED_BINS = 7200  # Expected number of bins for valid iprgc_test trials
IPRGC_LENGTH_TOLERANCE = 0.10  # Allow ±10% variation in trial length

# Excluded columns (inconsistent data)
EXCLUDED_MOVIES = ["perfect_dense_noise_15x15_15hz_r42_3min"]
EXCLUDED_COLUMNS = []  # No hardcoded exclusions; iprgc_test trials filtered dynamically by length

# Moving bar prefix for parsing
MOVING_BAR_PREFIX = "moving_h_bar_s5_d8_3x"

# Plot settings
TRIAL_ALPHA = 0.4
TRIAL_COLOR = "steelblue"
MEAN_COLOR = "darkred"
MEAN_LINEWIDTH = 1.5
FIGSIZE_PER_SUBPLOT = (2.5, 1.5)
MAX_UNITS_PER_PLOT = 20


# =============================================================================
# STEP 1: GENERATE INTERFRAME FIRING RATES
# =============================================================================

def get_frame_aligned_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    all_frame_samples: np.ndarray,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, float]:
    """Convert spike times to frame-aligned firing rate."""
    mask = (all_frame_samples >= trial_start_sample) & (all_frame_samples < trial_end_sample)
    trial_frames = all_frame_samples[mask]
    
    if len(trial_frames) < 2:
        return np.array([], dtype=np.float32), 0.0
    
    relative_spikes = spike_times.astype(np.int64) - int(trial_start_sample)
    trial_duration = int(trial_end_sample) - int(trial_start_sample)
    spike_mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[spike_mask]
    
    relative_frames = trial_frames.astype(np.int64) - int(trial_start_sample)
    counts, _ = np.histogram(trial_spikes, bins=relative_frames)
    frame_intervals = np.diff(relative_frames) / acquisition_rate
    
    firing_rate = np.zeros_like(counts, dtype=np.float32)
    valid_intervals = frame_intervals > 0
    firing_rate[valid_intervals] = counts[valid_intervals] / frame_intervals[valid_intervals]
    
    mean_frame_rate = 1.0 / np.mean(frame_intervals[valid_intervals]) if np.any(valid_intervals) else 0.0
    return firing_rate, mean_frame_rate


def get_sample_based_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    target_rate_hz: float = 60.0,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, int]:
    """Convert spike times to sample-based firing rate (for iprgc_test)."""
    samples_per_bin = acquisition_rate / target_rate_hz
    relative_spikes = spike_times.astype(np.int64) - int(trial_start_sample)
    trial_duration = int(trial_end_sample) - int(trial_start_sample)
    mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[mask]
    
    n_bins = int(np.ceil(trial_duration / samples_per_bin))
    bin_edges = np.arange(0, n_bins + 1) * samples_per_bin
    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
    
    bin_duration_s = 1.0 / target_rate_hz
    firing_rate = counts.astype(np.float32) / bin_duration_s
    return firing_rate, n_bins


def determine_target_lengths(h5_paths: List[Path]) -> Dict[str, int]:
    """First pass: Determine minimum frame counts per trial type."""
    frame_counts: Dict[str, List[int]] = {}
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if "metadata/frame_timestamps" not in f:
                continue
            all_frames = f["metadata/frame_timestamps"][:]
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            sample_unit = unit_ids[0]
            
            # Regular movies
            for movie_name in MOVIES_FRAME_ALIGNED:
                trials_path = f"units/{sample_unit}/spike_times_sectioned/{movie_name}/trials_start_end"
                if trials_path not in f:
                    continue
                trials_start_end = f[trials_path][:]
                for trial_idx, (start, end) in enumerate(trials_start_end):
                    mask = (all_frames >= start) & (all_frames < end)
                    n_bins = np.sum(mask) - 1
                    col_key = f"{movie_name}_{trial_idx}"
                    frame_counts.setdefault(col_key, []).append(n_bins)
            
            # Direction sections
            dir_section_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/direction_section"
            if dir_section_path in f:
                dir_group = f[dir_section_path]
                for direction in dir_group.keys():
                    if direction == "_attrs":
                        continue
                    dir_data = dir_group[direction]
                    if "section_bounds" not in dir_data:
                        continue
                    bounds = dir_data["section_bounds"][:]
                    for rep_idx, (start_rel, end_rel) in enumerate(bounds):
                        n_frames = end_rel - start_rel
                        if n_frames <= 0:
                            continue
                        col_key = f"{MOVIE_DIRECTION_SECTION}_{direction}_{rep_idx}"
                        frame_counts.setdefault(col_key, []).append(n_frames - 1)
            
            # iprgc_test - only include trials within ±10% of expected length
            iprgc_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_start_end"
            if iprgc_path in f:
                trials_start_end = f[iprgc_path][:]
                samples_per_bin = acq_rate / IPRGC_TARGET_RATE_HZ
                min_expected = int(IPRGC_EXPECTED_BINS * (1 - IPRGC_LENGTH_TOLERANCE))
                max_expected = int(IPRGC_EXPECTED_BINS * (1 + IPRGC_LENGTH_TOLERANCE))
                for trial_idx, (start, end) in enumerate(trials_start_end):
                    n_bins = int(np.ceil((end - start) / samples_per_bin))
                    # Only include trials within tolerance
                    if min_expected <= n_bins <= max_expected:
                        col_key = f"{MOVIE_SAMPLE_BASED}_{trial_idx}"
                        frame_counts.setdefault(col_key, []).append(n_bins)
    
    return {k: min(v) for k, v in frame_counts.items()}


def process_all_data(h5_paths: List[Path], target_lengths: Dict[str, int]) -> pd.DataFrame:
    """Second pass: Compute firing rates and build DataFrame."""
    rows = []
    
    for h5_path in tqdm(h5_paths, desc="Processing HDF5 files"):
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            if "metadata/frame_timestamps" not in f:
                continue
            all_frames = f["metadata/frame_timestamps"][:]
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            for unit_id in f["units"].keys():
                row_data = {"row_index": f"{dataset_id}_{unit_id}"}
                
                # Regular movies
                for movie_name in MOVIES_FRAME_ALIGNED:
                    trials_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_spike_times"
                    starts_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_start_end"
                    if trials_path not in f or starts_path not in f:
                        continue
                    trials_start_end = f[starts_path][:]
                    trials_group = f[trials_path]
                    
                    for trial_idx_str in trials_group.keys():
                        trial_idx = int(trial_idx_str)
                        trial_ds = trials_group[trial_idx_str]
                        # Handle scalar (empty) vs array datasets
                        if trial_ds.shape == ():
                            spike_times = np.array([])
                        else:
                            spike_times = trial_ds[:]
                        trial_start, trial_end = trials_start_end[trial_idx]
                        
                        fr, _ = get_frame_aligned_firing_rate(
                            spike_times, trial_start, trial_end, all_frames, acq_rate
                        )
                        col_key = f"{movie_name}_{trial_idx}"
                        target_len = target_lengths.get(col_key, len(fr))
                        if len(fr) > target_len:
                            fr = fr[:target_len]
                        elif len(fr) < target_len:
                            fr = np.pad(fr, (0, target_len - len(fr)))
                        row_data[col_key] = fr.tolist()
                
                # Direction sections
                dir_section_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/direction_section"
                if dir_section_path in f:
                    dir_group = f[dir_section_path]
                    movie_trials_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/trials_start_end"
                    if movie_trials_path in f:
                        movie_start_end = f[movie_trials_path][:]
                        movie_start_sample = movie_start_end[0, 0]
                        movie_start_frame_idx = np.searchsorted(all_frames, movie_start_sample)
                        
                        for direction in dir_group.keys():
                            if direction == "_attrs":
                                continue
                            dir_data = dir_group[direction]
                            if "section_bounds" not in dir_data or "trials" not in dir_data:
                                continue
                            bounds = dir_data["section_bounds"][:]
                            trials_group = dir_data["trials"]
                            
                            for rep_idx_str in trials_group.keys():
                                rep_idx = int(rep_idx_str)
                                start_frame_rel, end_frame_rel = bounds[rep_idx]
                                if end_frame_rel <= start_frame_rel:
                                    continue
                                
                                trial_ds = trials_group[rep_idx_str]
                                if trial_ds.shape == ():
                                    spike_frames = np.array([])
                                else:
                                    spike_frames = trial_ds[:]
                                start_frame_abs = movie_start_frame_idx + start_frame_rel
                                end_frame_abs = movie_start_frame_idx + end_frame_rel
                                
                                if start_frame_abs >= len(all_frames) or end_frame_abs > len(all_frames):
                                    continue
                                section_frames = all_frames[int(start_frame_abs):int(end_frame_abs)]
                                if len(section_frames) < 2:
                                    continue
                                
                                n_frames = len(section_frames)
                                counts = np.zeros(n_frames - 1, dtype=np.float32)
                                for spike_frame in spike_frames:
                                    bin_idx = spike_frame - start_frame_rel
                                    if 0 <= bin_idx < (n_frames - 1):
                                        counts[int(bin_idx)] += 1
                                
                                frame_intervals = np.diff(section_frames) / acq_rate
                                fr = np.zeros_like(counts, dtype=np.float32)
                                valid = frame_intervals > 0
                                fr[valid] = counts[valid] / frame_intervals[valid]
                                
                                col_key = f"{MOVIE_DIRECTION_SECTION}_{direction}_{rep_idx}"
                                target_len = target_lengths.get(col_key, len(fr))
                                if len(fr) > target_len:
                                    fr = fr[:target_len]
                                elif len(fr) < target_len:
                                    fr = np.pad(fr, (0, target_len - len(fr)))
                                row_data[col_key] = fr.tolist()
                
                # iprgc_test - dynamically filter by expected length (±10% tolerance)
                iprgc_trials_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_spike_times"
                iprgc_starts_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_start_end"
                if iprgc_trials_path in f and iprgc_starts_path in f:
                    trials_start_end = f[iprgc_starts_path][:]
                    trials_group = f[iprgc_trials_path]
                    
                    min_expected = int(IPRGC_EXPECTED_BINS * (1 - IPRGC_LENGTH_TOLERANCE))
                    max_expected = int(IPRGC_EXPECTED_BINS * (1 + IPRGC_LENGTH_TOLERANCE))
                    
                    # First pass: collect valid trials and their firing rates
                    valid_trials = []  # List of (trial_idx, firing_rate_array)
                    excluded_iprgc_trials = []
                    
                    for trial_idx_str in trials_group.keys():
                        trial_idx = int(trial_idx_str)
                        trial_ds = trials_group[trial_idx_str]
                        if trial_ds.shape == ():
                            spike_times = np.array([])
                        else:
                            spike_times = trial_ds[:]
                        trial_start, trial_end = trials_start_end[trial_idx]
                        
                        fr, n_bins = get_sample_based_firing_rate(
                            spike_times, trial_start, trial_end, IPRGC_TARGET_RATE_HZ, acq_rate
                        )
                        
                        # Check if trial is within ±10% tolerance
                        if not (min_expected <= n_bins <= max_expected):
                            excluded_iprgc_trials.append((trial_idx, n_bins))
                            continue
                        
                        valid_trials.append((trial_idx, fr))
                    
                    # Find minimum length among valid trials for this unit
                    if valid_trials:
                        min_len_unit = min(len(fr) for _, fr in valid_trials)
                        
                        # Second pass: trim all valid trials to minimum length
                        for trial_idx, fr in valid_trials:
                            col_key = f"{MOVIE_SAMPLE_BASED}_{trial_idx}"
                            # Trim to unit's minimum valid length
                            fr_trimmed = fr[:min_len_unit]
                            row_data[col_key] = fr_trimmed.tolist()
                    
                    # Warn if less than 2 valid trials
                    if len(valid_trials) < 2:
                        print(f"\033[91mWARNING: {dataset_id}_{unit_id} has only {len(valid_trials)} valid iprgc_test trial(s) "
                              f"(excluded: {excluded_iprgc_trials})\033[0m")
                
                rows.append(row_data)
    
    df = pd.DataFrame(rows)
    return df.set_index("row_index")


# =============================================================================
# STEP 2: RESHAPE INTO MOVIE-BASED FORMAT
# =============================================================================

def parse_column_groups(columns: List[str]) -> Dict[str, List[str]]:
    """Group columns by movie name."""
    groups: Dict[str, List[str]] = {}
    
    for col in columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if col.startswith(MOVING_BAR_PREFIX):
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                movie_dir = parts[0]
                groups.setdefault(movie_dir, []).append(col)
        else:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                movie = parts[0]
                groups.setdefault(movie, []).append(col)
    
    for movie in groups:
        groups[movie] = sorted(groups[movie], key=lambda x: int(x.rsplit("_", 1)[1]))
    return groups


def validate_units(df: pd.DataFrame, movie_groups: Dict[str, List[str]]) -> Tuple[Set[str], Dict[str, int], Dict]:
    """Validate units: reject any with >1 frame difference from mode."""
    rejected_units: Set[str] = set()
    mode_lengths: Dict[str, int] = {}
    rejection_reasons: Dict[str, List[str]] = {}
    
    for movie, columns in movie_groups.items():
        lengths_df = pd.DataFrame(index=df.index)
        for col in columns:
            lengths_df[col] = df[col].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
            )
        
        all_lengths = lengths_df.values.flatten()
        all_lengths = all_lengths[all_lengths > 0]
        if len(all_lengths) == 0:
            continue
        
        mode_length = Counter(all_lengths).most_common(1)[0][0]
        mode_lengths[movie] = mode_length
        
        for unit_idx in df.index.unique():
            row_data = lengths_df.loc[unit_idx]
            # Handle duplicate indices: loc returns DataFrame, need to get first row
            if isinstance(row_data, pd.DataFrame):
                row_data = row_data.iloc[0]
            unit_lengths = row_data.values
            for i, length in enumerate(unit_lengths):
                # Ensure length is a scalar (handle any remaining array edge cases)
                if hasattr(length, '__len__') and not isinstance(length, str):
                    length = int(length.flat[0]) if hasattr(length, 'flat') else int(length[0])
                if length == 0:
                    continue
                if abs(length - mode_length) > 1:
                    rejected_units.add(unit_idx)
                    rejection_reasons.setdefault(unit_idx, []).append(
                        f"{movie} trial {columns[i]}: {length} (mode={mode_length})"
                    )
    
    return rejected_units, mode_lengths, rejection_reasons


def reshape_to_movies(df: pd.DataFrame, movie_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Stack trials into 2D arrays per movie."""
    # Compute target lengths (minimum)
    target_lengths: Dict[str, int] = {}
    for movie, columns in movie_groups.items():
        min_len = float('inf')
        for col in columns:
            col_lens = df[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0)
            col_lens = col_lens[col_lens > 0]
            if len(col_lens) > 0:
                min_len = min(min_len, col_lens.min())
        if min_len < float('inf'):
            target_lengths[movie] = int(min_len)
    
    # Build result
    result_data = {movie: [] for movie in movie_groups}
    result_index = []
    
    for unit_idx in df.index:
        result_index.append(unit_idx)
        for movie, columns in movie_groups.items():
            target_len = target_lengths.get(movie, 0)
            if target_len == 0:
                result_data[movie].append(None)
                continue
            
            trials = []
            for col in columns:
                trial_data = df.loc[unit_idx, col]
                if trial_data is not None and isinstance(trial_data, (list, np.ndarray)) and len(trial_data) > 0:
                    arr = np.array(trial_data, dtype=np.float32)[:target_len]
                    trials.append(arr)
            
            if trials:
                result_data[movie].append(np.stack(trials, axis=0))
            else:
                result_data[movie].append(None)
    
    return pd.DataFrame(result_data, index=result_index)


# =============================================================================
# STEP 3: PLOT FIRING RATES
# =============================================================================

def plot_dataset(df: pd.DataFrame, dataset_id: str, movies: list) -> plt.Figure:
    """Create a plot for a single dataset."""
    n_units = min(len(df), MAX_UNITS_PER_PLOT)
    n_movies = len(movies)
    truncated = len(df) > MAX_UNITS_PER_PLOT
    df = df.iloc[:n_units]
    
    # Identify moving_h_bar movies for consistent y-axis scaling
    mbar_movies = [m for m in movies if m.startswith(MOVING_BAR_PREFIX)]
    mbar_indices = {m: i for i, m in enumerate(movies) if m.startswith(MOVING_BAR_PREFIX)}
    
    # Pre-compute max y-value for moving_h_bar plots per unit
    unit_mbar_ymax: Dict[str, float] = {}
    for unit_idx, unit_row in df.iterrows():
        max_val = 0.0
        for movie in mbar_movies:
            data = unit_row[movie]
            if data is not None and not (isinstance(data, float) and np.isnan(data)):
                if isinstance(data, list):
                    data = np.array([np.array(row) for row in data])
                elif isinstance(data, np.ndarray) and data.dtype == object:
                    data = np.array([np.array(row) for row in data])
                if data.size > 0:
                    max_val = max(max_val, np.nanmax(data))
        unit_mbar_ymax[unit_idx] = max_val * 1.1 if max_val > 0 else 1.0
    
    fig, axes = plt.subplots(
        n_units, n_movies,
        figsize=(FIGSIZE_PER_SUBPLOT[0] * n_movies, FIGSIZE_PER_SUBPLOT[1] * n_units),
        squeeze=False
    )
    
    for row_idx, (unit_idx, unit_row) in enumerate(df.iterrows()):
        unit_id = unit_idx.split("_")[-1]
        
        for col_idx, movie in enumerate(movies):
            ax = axes[row_idx, col_idx]
            data = unit_row[movie]
            
            if data is None or (isinstance(data, float) and np.isnan(data)):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if isinstance(data, list):
                    data = np.array([np.array(row) for row in data])
                elif isinstance(data, np.ndarray) and data.dtype == object:
                    data = np.array([np.array(row) for row in data])
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                n_trials, n_bins = data.shape
                x = np.arange(n_bins)
                
                for trial_idx in range(n_trials):
                    ax.plot(x, data[trial_idx], color=TRIAL_COLOR, alpha=TRIAL_ALPHA, linewidth=0.5)
                ax.plot(x, np.mean(data, axis=0), color=MEAN_COLOR, linewidth=MEAN_LINEWIDTH)
                
                ax.set_xlim(0, n_bins)
                
                # Use consistent y-axis for moving_h_bar movies, individual for others
                if movie in mbar_indices:
                    y_max = unit_mbar_ymax[unit_idx]
                else:
                    y_max = np.nanmax(data) * 1.1 if np.nanmax(data) > 0 else 1
                ax.set_ylim(0, y_max)
            
            if row_idx == 0:
                short_name = movie.replace("moving_h_bar_s5_d8_3x_", "mbar_").replace("step_up_5s_5i_b0_", "step_")
                ax.set_title(short_name, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(unit_id, fontsize=7)
            ax.tick_params(axis="both", labelsize=5)
            if row_idx < n_units - 1:
                ax.set_xticklabels([])
    
    title = f"Dataset: {dataset_id}"
    if truncated:
        title += f" (showing {MAX_UNITS_PER_PLOT} units)"
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def generate_plots(df: pd.DataFrame):
    """Generate plots for all datasets."""
    movies = df.columns.tolist()
    df["dataset_id"] = df.index.map(lambda x: "_".join(x.split("_")[:-1]))
    datasets = df["dataset_id"].unique()
    
    print(f"\nGenerating {len(datasets)} plots...")
    for dataset_id in tqdm(datasets, desc="Plotting"):
        dataset_df = df[df["dataset_id"] == dataset_id].drop(columns=["dataset_id"])
        fig = plot_dataset(dataset_df, dataset_id, movies)
        fig.savefig(PLOTS_DIR / f"firing_rate_{dataset_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # Create HTML viewer
    html_path = PLOTS_DIR / "view_plots.html"
    with open(html_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html><head><title>Firing Rate Plots</title>")
        f.write("<style>body{font-family:sans-serif;}img{max-width:100%;margin:10px 0;}</style>")
        f.write("</head><body><h1>Firing Rate by Movie</h1>")
        for dataset_id in sorted(datasets):
            f.write(f"<h2>{dataset_id}</h2><img src='firing_rate_{dataset_id}.png'>")
        f.write("</body></html>")
    print(f"Created HTML viewer: {html_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Firing Rate Pipeline: Generate, reshape, and plot firing rates from HDF5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_firing_rate.py                      # Process all files
  python pipeline_firing_rate.py --start 0 --end 10   # Process files 0-9
  python pipeline_firing_rate.py --start 5            # Process from file 5 to end
  python pipeline_firing_rate.py --end 5              # Process first 5 files
  python pipeline_firing_rate.py --list               # List all files with indices
  python pipeline_firing_rate.py --no-plots           # Skip plot generation
  python pipeline_firing_rate.py --strict             # Fail on validation errors
  python pipeline_firing_rate.py --incremental        # Skip processed, append new
  python pipeline_firing_rate.py --start-step 2       # Start from step 2 (reshape)
  python pipeline_firing_rate.py --start-step 3       # Start from step 3 (plots only)
  python pipeline_firing_rate.py --input /path/to/hdf5 --output suffix

Note: By default, units with inconsistent trial lengths are excluded with a warning.
        """
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start file index (inclusive, 0-based). Default: 0"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End file index (exclusive). Default: process all files"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=HDF5_DIR,
        help=f"Input directory containing HDF5 files. Default: {HDF5_DIR}"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename suffix (e.g., '_partial'). Default: no suffix"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_files",
        help="List all HDF5 files with indices and exit"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Skip plot generation step"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        dest="skip_validation",
        help="Skip validation step (useful for partial runs with known inconsistencies)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        dest="strict",
        help="Fail on validation errors instead of excluding units and continuing"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        dest="incremental",
        help="Skip already-processed files and append new data to existing output"
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3],
        dest="start_step",
        help="Start from a specific step (1=generate, 2=reshape, 3=plots). Default: 1"
    )
    
    return parser.parse_args()


def list_files(h5_files: List[Path]):
    """Print a list of all HDF5 files with their indices."""
    print("=" * 80)
    print("Available HDF5 Files")
    print("=" * 80)
    print(f"\nTotal: {len(h5_files)} files\n")
    
    for idx, path in enumerate(h5_files):
        print(f"  [{idx:3d}] {path.name}")
    
    print()


def main():
    args = parse_args()
    
    # Get all HDF5 files
    all_h5_files = sorted(args.input.glob("*.h5"))
    
    # List mode
    if args.list_files:
        list_files(all_h5_files)
        return
    
    print("=" * 80)
    print("Firing Rate Pipeline")
    print("=" * 80)
    print(f"\nInput: {args.input}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    
    print(f"\nFound {len(all_h5_files)} HDF5 files")
    
    if len(all_h5_files) == 0:
        print("No HDF5 files found!")
        return
    
    # Apply start/end filters
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(all_h5_files)
    
    # Validate indices
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(all_h5_files):
        end_idx = len(all_h5_files)
    if start_idx >= end_idx:
        print(f"\nError: Invalid range [{start_idx}:{end_idx}]. No files to process.")
        return
    
    h5_files = all_h5_files[start_idx:end_idx]
    
    print(f"Processing files [{start_idx}:{end_idx}] ({len(h5_files)} files)")
    if len(h5_files) > 0:
        print(f"  First: {h5_files[0].name}")
        print(f"  Last:  {h5_files[-1].name}")
    
    # Determine output filename suffix
    if args.output:
        suffix = args.output
    elif start_idx != 0 or end_idx != len(all_h5_files):
        suffix = f"_{start_idx}_{end_idx}"
    else:
        suffix = "20260104"
    
    # Define output paths early for incremental mode
    raw_path = OUTPUT_DIR / f"interframe_firing_rate{suffix}.parquet"
    movies_path = OUTPUT_DIR / f"firing_rate_by_movie{suffix}.parquet"
    
    # Incremental mode: check for existing data and filter files
    existing_df_raw = None
    existing_df_movies = None
    if args.incremental and raw_path.exists():
        print(f"\n\033[94mIncremental mode: Loading existing data from {raw_path.name}\033[0m")
        existing_df_raw = pd.read_parquet(raw_path)
        
        # Extract dataset_ids from existing data
        # Index format: "2024.02.26-10.53.19-Rec_unit_001" -> dataset_id: "2024.02.26-10.53.19-Rec"
        existing_dataset_ids = set()
        for idx in existing_df_raw.index:
            # Split by "_unit_" to correctly extract dataset ID
            parts = idx.rsplit("_unit_", 1)
            if len(parts) == 2:
                existing_dataset_ids.add(parts[0])
            else:
                # Fallback: use the full index as dataset ID
                existing_dataset_ids.add(idx)
        
        print(f"  Found {len(existing_dataset_ids)} already-processed datasets")
        
        # Filter h5_files to only include new ones (compare with h5 file stem)
        new_h5_files = [
            f for f in h5_files 
            if f.stem not in existing_dataset_ids
        ]
        
        skipped_count = len(h5_files) - len(new_h5_files)
        if skipped_count > 0:
            print(f"  Skipping {skipped_count} already-processed files")
        
        if len(new_h5_files) == 0:
            print(f"\n\033[92mAll files already processed. Nothing to do.\033[0m")
            return
        
        h5_files = new_h5_files
        print(f"  Processing {len(h5_files)} new files")
        
        # Also load existing movies dataframe if it exists
        if movies_path.exists():
            existing_df_movies = pd.read_parquet(movies_path)
    
    # =========================================================================
    # STEP 1: Generate interframe firing rates
    # =========================================================================
    if args.start_step <= 1:
        print("\n" + "=" * 80)
        print("STEP 1: Generate Interframe Firing Rates")
        print("=" * 80)
        
        # Use all selected files (including already-processed) for target length determination
        # This ensures consistency across incremental runs
        target_lengths = determine_target_lengths(all_h5_files[start_idx:end_idx])
        print(f"Target lengths determined for {len(target_lengths)} columns")
        
        df_raw = process_all_data(h5_files, target_lengths)
        print(f"\nNew DataFrame: {df_raw.shape} (units x trials)")
        
        # Append to existing data if incremental mode
        if existing_df_raw is not None:
            # Align columns (new data might have different columns)
            all_columns = list(set(existing_df_raw.columns) | set(df_raw.columns))
            existing_df_raw = existing_df_raw.reindex(columns=all_columns)
            df_raw = df_raw.reindex(columns=all_columns)
            
            df_raw = pd.concat([existing_df_raw, df_raw], axis=0)
            
            # Remove any duplicate indices (keep first occurrence)
            n_before = len(df_raw)
            df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
            n_removed = n_before - len(df_raw)
            if n_removed > 0:
                print(f"\033[93m  Removed {n_removed} duplicate units\033[0m")
            
            print(f"Combined DataFrame: {df_raw.shape} (units x trials)")
        
        # Save raw
        df_raw.to_parquet(raw_path)
        print(f"Saved: {raw_path}")
    else:
        print("\n" + "=" * 80)
        print("STEP 1: Generate Interframe Firing Rates (SKIPPED - loading from file)")
        print("=" * 80)
        
        if not raw_path.exists():
            print(f"\n\033[91mERROR: Cannot start from step {args.start_step}.\033[0m")
            print(f"\033[91mRequired file not found: {raw_path}\033[0m")
            print("\033[93mRun the pipeline from step 1 first.\033[0m")
            return
        
        df_raw = pd.read_parquet(raw_path)
        print(f"Loaded: {raw_path}")
        print(f"DataFrame: {df_raw.shape} (units x trials)")
    
    # =========================================================================
    # STEP 2: Reshape into movie-based format
    # =========================================================================
    if args.start_step <= 2:
        print("\n" + "=" * 80)
        print("STEP 2: Reshape into Movie-Based Format")
        print("=" * 80)
        
        movie_groups = parse_column_groups(df_raw.columns.tolist())
        print(f"Movie groups: {len(movie_groups)}")
        for movie, cols in sorted(movie_groups.items()):
            print(f"  {movie}: {len(cols)} trials")
        
        # Validate
        if args.skip_validation:
            print("\n\033[93mValidation SKIPPED (--skip-validation)\033[0m")
            rejected = set()
        else:
            rejected, mode_lengths, reasons = validate_units(df_raw, movie_groups)
            
            if rejected:
                if args.strict:
                    # Strict mode: fail on validation errors
                    print(f"\n\033[91mERROR: {len(rejected)} units rejected!\033[0m")
                    for unit in sorted(rejected)[:10]:
                        print(f"\033[91m  {unit}: {reasons.get(unit, [''])[0]}\033[0m")
                    print("\n\033[93mTip: Remove --strict to exclude invalid units and continue.\033[0m")
                    raise ValueError(f"{len(rejected)} units have inconsistent trial lengths")
                else:
                    # Default: warn and exclude rejected units
                    print(f"\n\033[93mWARNING: {len(rejected)} units excluded due to inconsistent trial lengths\033[0m")
                    for unit in sorted(rejected)[:10]:
                        print(f"\033[93m  {unit}: {reasons.get(unit, [''])[0]}\033[0m")
                    if len(rejected) > 10:
                        print(f"\033[93m  ... and {len(rejected) - 10} more\033[0m")
                    
                    # Remove rejected units from dataframe
                    df_raw = df_raw.drop(index=list(rejected), errors='ignore')
                    print(f"Continuing with {len(df_raw)} valid units")
            else:
                print("All units passed validation")
        
        # Reshape
        df_movies = reshape_to_movies(df_raw, movie_groups)
        print(f"\nReshaped DataFrame: {df_movies.shape} (units x movies)")
        
        # Save
        df_save = df_movies.copy()
        for col in df_save.columns:
            df_save[col] = df_save[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_save.to_parquet(movies_path)
        print(f"Saved: {movies_path}")
    else:
        print("\n" + "=" * 80)
        print("STEP 2: Reshape into Movie-Based Format (SKIPPED - loading from file)")
        print("=" * 80)
        
        if not movies_path.exists():
            print(f"\n\033[91mERROR: Cannot start from step {args.start_step}.\033[0m")
            print(f"\033[91mRequired file not found: {movies_path}\033[0m")
            print("\033[93mRun the pipeline from step 2 first.\033[0m")
            return
        
        df_movies = pd.read_parquet(movies_path)
        # Convert lists back to numpy arrays
        for col in df_movies.columns:
            df_movies[col] = df_movies[col].apply(
                lambda x: np.array(x) if isinstance(x, list) else x
            )
        print(f"Loaded: {movies_path}")
        print(f"DataFrame: {df_movies.shape} (units x movies)")
    
    # =========================================================================
    # STEP 3: Generate plots
    # =========================================================================
    if args.no_plots:
        print("\n" + "=" * 80)
        print("STEP 3: Generate Plots (SKIPPED)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("STEP 3: Generate Plots")
        print("=" * 80)
        
        generate_plots(df_movies)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Build summary based on what was actually run
    summary_lines = [
        "",
        "Pipeline Complete:",
        "------------------",
        f"- Units: {len(df_movies)}",
        f"- Movies: {len(df_movies.columns)}",
    ]
    if args.start_step == 1:
        summary_lines.append(f"- Datasets: {len(h5_files)} (indices {start_idx}-{end_idx-1})")
    else:
        summary_lines.append(f"- Started from step {args.start_step}")
    
    summary_lines.extend([
        "",
        "Output Files:",
        f"- {raw_path.name}",
        f"- {movies_path.name}",
    ])
    print("\n".join(summary_lines))
    
    if not args.no_plots:
        print(f"- {PLOTS_DIR.name}/ (plots)")
    
    print(f"""
Array shapes per movie:
""")
    for movie in sorted(df_movies.columns):
        col_data = df_movies[movie].dropna()
        if len(col_data) > 0:
            sample = col_data.iloc[0]
            if sample is not None and isinstance(sample, np.ndarray):
                print(f"  {movie}: {sample.shape}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

