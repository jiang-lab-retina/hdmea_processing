"""
Generate Interframe Firing Rate DataFrame

Computes inter-frame firing rates for all stimuli (except dense noise) and saves
as a wide-format DataFrame with:
- Index: {dataset_id}_{unit_id}
- Columns: {movie_name}_{trial_num} (or {movie_name}_{direction}_{trial_num} for moving bar)
- Cell values: Array of firing rates (Hz) per inter-frame bin

Data sources:
- Frame timestamps: metadata/frame_timestamps
- Trial boundaries: units/{unit_id}/spike_times_sectioned/{movie}/trials_start_end
- Moving bar directions: units/{unit_id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/{dir}/section_bounds

Usage:
    python generate_interframe_firing_rate.py                    # Process all files
    python generate_interframe_firing_rate.py --start 0 --end 10 # Process files 0-9
    python generate_interframe_firing_rate.py --start 5          # Process from file 5 to end
    python generate_interframe_firing_rate.py --end 5            # Process first 5 files
    python generate_interframe_firing_rate.py --list             # List all files with indices
"""

import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configuration
HDF5_DIR = project_root / "Projects/unified_pipeline/export_all_steps"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Movies to process (exclude dense noise)
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

# Excluded
EXCLUDED_MOVIES = ["perfect_dense_noise_15x15_15hz_r42_3min"]


def get_frame_aligned_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    all_frame_samples: np.ndarray,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, float]:
    """
    Convert spike times to frame-aligned firing rate.
    
    Args:
        spike_times: Spike times in absolute sample indices
        trial_start_sample: Trial start in samples
        trial_end_sample: Trial end in samples
        all_frame_samples: All frame timestamps in samples
        acquisition_rate: Samples per second
    
    Returns:
        firing_rate: Firing rate in Hz for each inter-frame interval
        mean_frame_rate: Average frame rate in Hz
    """
    # Get frames within this trial
    mask = (all_frame_samples >= trial_start_sample) & (all_frame_samples < trial_end_sample)
    trial_frames = all_frame_samples[mask]
    
    if len(trial_frames) < 2:
        return np.array([], dtype=np.float32), 0.0
    
    # Convert spikes to relative times
    relative_spikes = spike_times.astype(np.int64) - int(trial_start_sample)
    
    # Filter to spikes within trial
    trial_duration = int(trial_end_sample) - int(trial_start_sample)
    spike_mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[spike_mask]
    
    # Convert frame samples to relative
    relative_frames = trial_frames.astype(np.int64) - int(trial_start_sample)
    
    # Count spikes per inter-frame interval
    counts, _ = np.histogram(trial_spikes, bins=relative_frames)
    
    # Calculate inter-frame intervals (in seconds)
    frame_intervals = np.diff(relative_frames) / acquisition_rate
    
    # Calculate firing rate: spikes / interval_duration
    firing_rate = np.zeros_like(counts, dtype=np.float32)
    valid_intervals = frame_intervals > 0
    firing_rate[valid_intervals] = counts[valid_intervals] / frame_intervals[valid_intervals]
    
    # Calculate mean frame rate
    mean_frame_rate = 1.0 / np.mean(frame_intervals[valid_intervals]) if np.any(valid_intervals) else 0.0
    
    return firing_rate, mean_frame_rate


def get_sample_based_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    target_rate_hz: float = 60.0,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, int]:
    """
    Convert spike times to firing rate using sample-based bins at target Hz.
    Used for iprgc_test which has no video frames.
    
    Args:
        spike_times: Spike times in absolute sample indices
        trial_start_sample: Trial start in samples
        trial_end_sample: Trial end in samples
        target_rate_hz: Target bin rate (Hz)
        acquisition_rate: Samples per second
    
    Returns:
        firing_rate: Firing rate in Hz for each time bin
        n_bins: Number of bins
    """
    # Calculate bin size in samples
    samples_per_bin = acquisition_rate / target_rate_hz
    
    # Convert spikes to relative times
    relative_spikes = spike_times.astype(np.int64) - int(trial_start_sample)
    
    # Filter to spikes within trial
    trial_duration = int(trial_end_sample) - int(trial_start_sample)
    mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[mask]
    
    # Calculate number of bins
    n_bins = int(np.ceil(trial_duration / samples_per_bin))
    
    # Create bin edges
    bin_edges = np.arange(0, n_bins + 1) * samples_per_bin
    
    # Count spikes per bin
    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
    
    # Convert to firing rate (Hz)
    bin_duration_s = 1.0 / target_rate_hz
    firing_rate = counts.astype(np.float32) / bin_duration_s
    
    return firing_rate, n_bins


def determine_target_lengths(h5_paths: List[Path]) -> Dict[str, int]:
    """
    First pass: Determine minimum frame counts per trial type for normalization.
    
    Returns:
        Dict mapping column_prefix -> minimum bin count
    """
    frame_counts: Dict[str, List[int]] = {}
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            # Get frame timestamps
            if "metadata/frame_timestamps" in f:
                all_frames = f["metadata/frame_timestamps"][:]
            else:
                continue
            
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            
            sample_unit = unit_ids[0]
            
            # Check regular movies
            for movie_name in MOVIES_FRAME_ALIGNED:
                trials_path = f"units/{sample_unit}/spike_times_sectioned/{movie_name}/trials_start_end"
                if trials_path not in f:
                    continue
                
                trials_start_end = f[trials_path][:]
                
                for trial_idx, (start, end) in enumerate(trials_start_end):
                    mask = (all_frames >= start) & (all_frames < end)
                    n_frames = np.sum(mask)
                    n_bins = n_frames - 1  # inter-frame intervals
                    
                    col_key = f"{movie_name}_{trial_idx}"
                    if col_key not in frame_counts:
                        frame_counts[col_key] = []
                    frame_counts[col_key].append(n_bins)
            
            # Check moving_h_bar direction sections
            dir_section_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/direction_section"
            if dir_section_path in f:
                dir_group = f[dir_section_path]
                
                for direction in dir_group.keys():
                    if direction == "_attrs":
                        continue
                    
                    dir_data = dir_group[direction]
                    if "section_bounds" not in dir_data:
                        continue
                    
                    # section_bounds are movie-relative frame indices
                    bounds = dir_data["section_bounds"][:]
                    
                    # Get movie start to find absolute frames
                    movie_trials_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/trials_start_end"
                    if movie_trials_path not in f:
                        continue
                    
                    movie_start_end = f[movie_trials_path][:]
                    movie_start_sample = movie_start_end[0, 0]
                    
                    # Find movie start frame index
                    movie_start_frame_idx = np.searchsorted(all_frames, movie_start_sample)
                    
                    for rep_idx, (start_frame_rel, end_frame_rel) in enumerate(bounds):
                        n_frames = end_frame_rel - start_frame_rel
                        if n_frames <= 0:
                            continue
                        n_bins = n_frames - 1
                        
                        col_key = f"{MOVIE_DIRECTION_SECTION}_{direction}_{rep_idx}"
                        if col_key not in frame_counts:
                            frame_counts[col_key] = []
                        frame_counts[col_key].append(n_bins)
            
            # Check iprgc_test (sample-based)
            iprgc_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_start_end"
            if iprgc_path in f:
                trials_start_end = f[iprgc_path][:]
                samples_per_bin = acq_rate / IPRGC_TARGET_RATE_HZ
                
                for trial_idx, (start, end) in enumerate(trials_start_end):
                    duration = end - start
                    n_bins = int(np.ceil(duration / samples_per_bin))
                    
                    col_key = f"{MOVIE_SAMPLE_BASED}_{trial_idx}"
                    if col_key not in frame_counts:
                        frame_counts[col_key] = []
                    frame_counts[col_key].append(n_bins)
    
    # Get minimum for each column
    target_lengths = {}
    for col_key, counts in frame_counts.items():
        target_lengths[col_key] = min(counts)
    
    return target_lengths


def process_all_data(h5_paths: List[Path], target_lengths: Dict[str, int]) -> pd.DataFrame:
    """
    Second pass: Compute firing rates and build wide DataFrame.
    
    Returns:
        Wide DataFrame with index={dataset_id}_{unit_id}, columns={movie}_{trial}
    """
    # Collect all data as rows first, then pivot
    rows = []
    
    for h5_path in tqdm(h5_paths, desc="Processing HDF5 files"):
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            
            # Get frame timestamps and acquisition rate
            if "metadata/frame_timestamps" not in f:
                continue
            all_frames = f["metadata/frame_timestamps"][:]
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            for unit_id in f["units"].keys():
                row_index = f"{dataset_id}_{unit_id}"
                row_data = {"row_index": row_index}
                
                # Process regular frame-aligned movies
                for movie_name in MOVIES_FRAME_ALIGNED:
                    trials_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_spike_times"
                    starts_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_start_end"
                    
                    if trials_path not in f or starts_path not in f:
                        continue
                    
                    trials_start_end = f[starts_path][:]
                    trials_group = f[trials_path]
                    
                    for trial_idx_str in trials_group.keys():
                        trial_idx = int(trial_idx_str)
                        spike_times = trials_group[trial_idx_str][:]
                        
                        trial_start = trials_start_end[trial_idx, 0]
                        trial_end = trials_start_end[trial_idx, 1]
                        
                        fr, _ = get_frame_aligned_firing_rate(
                            spike_times=spike_times,
                            trial_start_sample=trial_start,
                            trial_end_sample=trial_end,
                            all_frame_samples=all_frames,
                            acquisition_rate=acq_rate,
                        )
                        
                        col_key = f"{movie_name}_{trial_idx}"
                        target_len = target_lengths.get(col_key, len(fr))
                        
                        # Normalize length
                        if len(fr) > target_len:
                            fr = fr[:target_len]
                        elif len(fr) < target_len:
                            fr = np.pad(fr, (0, target_len - len(fr)), constant_values=0)
                        
                        row_data[col_key] = fr.tolist()
                
                # Process moving_h_bar direction sections
                dir_section_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/direction_section"
                if dir_section_path in f:
                    dir_group = f[dir_section_path]
                    
                    # Get movie start to convert relative frame indices to absolute
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
                            
                            bounds = dir_data["section_bounds"][:]  # Movie-relative frame indices
                            trials_group = dir_data["trials"]
                            
                            for rep_idx_str in trials_group.keys():
                                rep_idx = int(rep_idx_str)
                                start_frame_rel, end_frame_rel = bounds[rep_idx]
                                
                                # Skip invalid sections
                                if end_frame_rel <= start_frame_rel:
                                    continue
                                
                                # Get spike frame indices (movie-relative)
                                spike_frames = trials_group[rep_idx_str][:]
                                
                                # Convert to absolute frame indices
                                start_frame_abs = movie_start_frame_idx + start_frame_rel
                                end_frame_abs = movie_start_frame_idx + end_frame_rel
                                
                                # Get actual frame timestamps for this section
                                if start_frame_abs >= len(all_frames) or end_frame_abs > len(all_frames):
                                    continue
                                
                                section_frames = all_frames[int(start_frame_abs):int(end_frame_abs)]
                                
                                if len(section_frames) < 2:
                                    continue
                                
                                # Count spikes per frame interval
                                # spike_frames are movie-relative frame indices, need to shift to section-relative
                                n_frames = len(section_frames)
                                counts = np.zeros(n_frames - 1, dtype=np.float32)
                                
                                for spike_frame in spike_frames:
                                    bin_idx = spike_frame - start_frame_rel
                                    if 0 <= bin_idx < (n_frames - 1):
                                        counts[int(bin_idx)] += 1
                                
                                # Calculate inter-frame intervals (in seconds)
                                frame_intervals = np.diff(section_frames) / acq_rate
                                
                                # Calculate firing rate
                                fr = np.zeros_like(counts, dtype=np.float32)
                                valid = frame_intervals > 0
                                fr[valid] = counts[valid] / frame_intervals[valid]
                                
                                col_key = f"{MOVIE_DIRECTION_SECTION}_{direction}_{rep_idx}"
                                target_len = target_lengths.get(col_key, len(fr))
                                
                                # Normalize length
                                if len(fr) > target_len:
                                    fr = fr[:target_len]
                                elif len(fr) < target_len:
                                    fr = np.pad(fr, (0, target_len - len(fr)), constant_values=0)
                                
                                row_data[col_key] = fr.tolist()
                
                # Process iprgc_test (sample-based)
                iprgc_trials_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_spike_times"
                iprgc_starts_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_start_end"
                
                if iprgc_trials_path in f and iprgc_starts_path in f:
                    trials_start_end = f[iprgc_starts_path][:]
                    trials_group = f[iprgc_trials_path]
                    
                    for trial_idx_str in trials_group.keys():
                        trial_idx = int(trial_idx_str)
                        spike_times = trials_group[trial_idx_str][:]
                        
                        trial_start = trials_start_end[trial_idx, 0]
                        trial_end = trials_start_end[trial_idx, 1]
                        
                        fr, _ = get_sample_based_firing_rate(
                            spike_times=spike_times,
                            trial_start_sample=trial_start,
                            trial_end_sample=trial_end,
                            target_rate_hz=IPRGC_TARGET_RATE_HZ,
                            acquisition_rate=acq_rate,
                        )
                        
                        col_key = f"{MOVIE_SAMPLE_BASED}_{trial_idx}"
                        target_len = target_lengths.get(col_key, len(fr))
                        
                        # Normalize length
                        if len(fr) > target_len:
                            fr = fr[:target_len]
                        elif len(fr) < target_len:
                            fr = np.pad(fr, (0, target_len - len(fr)), constant_values=0)
                        
                        row_data[col_key] = fr.tolist()
                
                rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    df = df.set_index("row_index")
    
    return df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Interframe Firing Rate DataFrame from HDF5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_interframe_firing_rate.py                    # Process all files
  python generate_interframe_firing_rate.py --start 0 --end 10 # Process files 0-9
  python generate_interframe_firing_rate.py --start 5          # Process from file 5 to end
  python generate_interframe_firing_rate.py --end 5            # Process first 5 files
  python generate_interframe_firing_rate.py --list             # List all files with indices
  python generate_interframe_firing_rate.py --input /path/to/hdf5  # Custom input directory
  python generate_interframe_firing_rate.py --output suffix    # Add suffix to output filename
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
    
    return parser.parse_args()


def list_files(h5_files: List[Path]):
    """Print a list of all HDF5 files with their indices."""
    print("=" * 70)
    print("Available HDF5 Files")
    print("=" * 70)
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
    
    print("=" * 70)
    print("Generate Interframe Firing Rate DataFrame")
    print("=" * 70)
    
    print(f"\nFound {len(all_h5_files)} HDF5 files in {args.input}")
    
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
    
    # ==========================================================================
    # 1. Determine target lengths (use ALL files for consistency)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. Determining Target Lengths (First Pass - ALL files)")
    print("=" * 70)
    
    # Always use all files for target length determination to ensure consistency
    target_lengths = determine_target_lengths(all_h5_files)
    
    print(f"\nTarget lengths determined for {len(target_lengths)} columns:")
    
    # Group by movie for display
    movies = {}
    for col_key, length in target_lengths.items():
        parts = col_key.rsplit("_", 1)
        if len(parts) == 2:
            movie_prefix = parts[0]
            if movie_prefix not in movies:
                movies[movie_prefix] = []
            movies[movie_prefix].append((col_key, length))
    
    for movie_prefix in sorted(movies.keys()):
        cols = movies[movie_prefix]
        lengths = [l for _, l in cols]
        print(f"  {movie_prefix}: {len(cols)} columns, {min(lengths)}-{max(lengths)} bins")
    
    # ==========================================================================
    # 2. Process selected data
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"2. Processing Selected Data ({len(h5_files)} files)")
    print("=" * 70)
    
    df = process_all_data(h5_files, target_lengths)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"  Rows (units): {len(df)}")
    print(f"  Columns (trials): {len(df.columns)}")
    
    # ==========================================================================
    # 3. Validate output
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. Validation")
    print("=" * 70)
    
    # Check for missing values
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"\nColumns with missing values: {len(cols_with_missing)}")
        for col, count in cols_with_missing.head(10).items():
            print(f"  {col}: {count} missing")
    else:
        print("\nNo missing values found.")
    
    # Check array lengths
    print("\nArray length consistency by stimulus:")
    for movie_prefix in sorted(movies.keys()):
        cols = [col for col, _ in movies[movie_prefix] if col in df.columns]
        if not cols:
            continue
        
        lengths = []
        for col in cols:
            col_lengths = df[col].dropna().apply(len)
            if len(col_lengths) > 0:
                lengths.extend(col_lengths.tolist())
        
        if lengths:
            min_len = min(lengths)
            max_len = max(lengths)
            status = "OK" if min_len == max_len else f"VARIES ({min_len}-{max_len})"
            print(f"  {movie_prefix}: {min_len} bins [{status}]")
    
    # ==========================================================================
    # 4. Save output
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. Saving Output")
    print("=" * 70)
    
    # Determine output filename
    if args.output:
        output_filename = f"interframe_firing_rate{args.output}.parquet"
    elif start_idx != 0 or end_idx != len(all_h5_files):
        # Add range to filename for partial processing
        output_filename = f"interframe_firing_rate_{start_idx}_{end_idx}.parquet"
    else:
        output_filename = "interframe_firing_rate.parquet"
    
    output_path = OUTPUT_DIR / output_filename
    df.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # ==========================================================================
    # 5. Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. Summary")
    print("=" * 70)
    
    print(f"""
Interframe Firing Rate DataFrame Generated:
--------------------------------------------
- Index: {{dataset_id}}_{{unit_id}}
- Total units: {len(df)}
- Total columns: {len(df.columns)}
- Files processed: {len(h5_files)} (indices {start_idx}-{end_idx-1})

Stimuli processed:
""")
    
    for movie_prefix in sorted(movies.keys()):
        cols = [col for col, _ in movies[movie_prefix] if col in df.columns]
        if cols:
            sample_col = cols[0]
            col_data = df[sample_col].dropna()
            if len(col_data) > 0:
                sample_len = col_data.iloc[0]
                n_bins = len(sample_len) if isinstance(sample_len, list) else 0
                print(f"  - {movie_prefix}: {len(cols)} trials, {n_bins} bins each")
    
    print(f"""
Output file: {output_path.name}
""")
    
    print("=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

