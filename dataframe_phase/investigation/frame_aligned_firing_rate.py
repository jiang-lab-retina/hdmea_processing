"""
Frame-Aligned Firing Rate Calculation

Uses frame_timestamps from HDF5 to calculate firing rates as spikes per frame.
This ensures consistent output lengths across recordings since all recordings
should have the same number of frames per trial.

Output: firing rate in Hz (spikes/frame × frame_rate)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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


def explore_frame_timestamps(h5_path: Path) -> Dict:
    """Explore the frame_timestamps structure in an HDF5 file."""
    info = {
        "file": h5_path.name,
        "frame_timestamps_path": None,
        "frame_time_path": None,
        "n_frames": 0,
        "acquisition_rate": 20000.0,
        "frame_rate_hz": None,
        "total_duration_s": None,
    }
    
    with h5py.File(h5_path, "r") as f:
        # Check for frame_timestamps in different locations
        possible_paths = [
            "metadata/frame_timestamps",
            "stimulus/frame_times/default",
            "frame_timestamps",
        ]
        
        for path in possible_paths:
            if path in f:
                timestamps = f[path][:]
                info["frame_timestamps_path"] = path
                info["n_frames"] = len(timestamps)
                info["timestamps_sample"] = timestamps[:5].tolist()
                info["timestamps_last"] = timestamps[-5:].tolist()
                break
        
        # Check for frame_time (time in seconds)
        if "metadata/frame_time" in f:
            frame_time = f["metadata/frame_time"][:]
            info["frame_time_path"] = "metadata/frame_time"
            info["frame_time_sample"] = frame_time[:5].tolist()
            info["frame_time_last"] = frame_time[-5:].tolist()
            info["total_duration_s"] = frame_time[-1] - frame_time[0]
            info["frame_rate_hz"] = len(frame_time) / info["total_duration_s"]
        
        # Get acquisition rate
        if "metadata/acquisition_rate" in f:
            info["acquisition_rate"] = float(f["metadata/acquisition_rate"][()])
        
        # Check section_time for frame indices
        if "stimulus/section_time" in f:
            info["section_time_movies"] = list(f["stimulus/section_time"].keys())
    
    return info


def get_frame_timestamps_for_trial(
    h5file: h5py.File,
    movie_name: str,
    trial_idx: int,
    trial_start_sample: int,
    trial_end_sample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get frame timestamps (in samples) that fall within a trial.
    
    Returns:
        frame_samples: Sample indices of frames within the trial
        frame_times_relative: Frame times relative to trial start (in samples)
    """
    # Get all frame timestamps
    if "stimulus/frame_times/default" in h5file:
        all_frame_samples = h5file["stimulus/frame_times/default"][:]
    elif "metadata/frame_timestamps" in h5file:
        all_frame_samples = h5file["metadata/frame_timestamps"][:]
    else:
        raise KeyError("No frame_timestamps found in HDF5 file")
    
    # Find frames within this trial
    mask = (all_frame_samples >= trial_start_sample) & (all_frame_samples < trial_end_sample)
    trial_frame_samples = all_frame_samples[mask]
    
    # Make relative to trial start
    frame_times_relative = trial_frame_samples - trial_start_sample
    
    return trial_frame_samples, frame_times_relative


def spike_times_to_frame_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    frame_samples: np.ndarray,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, float]:
    """
    Convert spike times to frame-aligned firing rate.
    
    Args:
        spike_times: Spike times in absolute sample indices
        trial_start_sample: Trial start in samples
        trial_end_sample: Trial end in samples
        frame_samples: Absolute sample indices of each frame
        acquisition_rate: Samples per second
    
    Returns:
        firing_rate: Firing rate in Hz for each inter-frame interval
        mean_frame_rate: Average frame rate in Hz
    """
    # Convert spikes to relative times
    relative_spikes = spike_times.astype(np.int64) - int(trial_start_sample)
    
    # Filter to spikes within trial
    trial_duration = int(trial_end_sample) - int(trial_start_sample)
    mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[mask]
    
    # Convert frame samples to relative
    relative_frames = frame_samples.astype(np.int64) - int(trial_start_sample)
    
    # Filter frames to those within trial
    frame_mask = (relative_frames >= 0) & (relative_frames < trial_duration)
    trial_frames = relative_frames[frame_mask]
    
    if len(trial_frames) < 2:
        return np.array([]), 0.0
    
    # Create bin edges from frame times
    # Each bin is [frame_i, frame_i+1)
    bin_edges = trial_frames
    
    # Count spikes per inter-frame interval
    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
    
    # Calculate inter-frame intervals (in seconds)
    frame_intervals = np.diff(trial_frames) / acquisition_rate
    
    # Calculate firing rate: spikes / interval_duration
    # Avoid division by zero
    firing_rate = np.zeros_like(counts, dtype=np.float32)
    valid_intervals = frame_intervals > 0
    firing_rate[valid_intervals] = counts[valid_intervals] / frame_intervals[valid_intervals]
    
    # Calculate mean frame rate
    mean_frame_rate = 1.0 / np.mean(frame_intervals[valid_intervals]) if np.any(valid_intervals) else 0.0
    
    return firing_rate, mean_frame_rate


def analyze_frame_consistency(h5_paths: List[Path]) -> pd.DataFrame:
    """
    Analyze frame count consistency across recordings for each trial.
    """
    results = []
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            # Get frame timestamps
            if "stimulus/frame_times/default" in f:
                all_frames = f["stimulus/frame_times/default"][:]
            elif "metadata/frame_timestamps" in f:
                all_frames = f["metadata/frame_timestamps"][:]
            else:
                continue
            
            # Get first unit to read trial boundaries
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            
            sample_unit = unit_ids[0]
            
            for movie_name in f[f"units/{sample_unit}/spike_times_sectioned"].keys():
                trials_path = f"units/{sample_unit}/spike_times_sectioned/{movie_name}/trials_start_end"
                if trials_path not in f:
                    continue
                
                trials_start_end = f[trials_path][:]
                
                for trial_idx, (start, end) in enumerate(trials_start_end):
                    # Count frames in this trial
                    mask = (all_frames >= start) & (all_frames < end)
                    n_frames = np.sum(mask)
                    
                    # Calculate frame rate
                    trial_duration = (end - start) / acq_rate
                    frame_rate = n_frames / trial_duration if trial_duration > 0 else 0
                    
                    results.append({
                        "dataset_id": dataset_id,
                        "movie_name": movie_name,
                        "trial_idx": trial_idx,
                        "n_frames": int(n_frames),
                        "trial_duration_s": trial_duration,
                        "frame_rate_hz": frame_rate,
                        "trial_start": int(start),
                        "trial_end": int(end),
                    })
    
    return pd.DataFrame(results)


def create_frame_aligned_firing_rate_dataframe(
    h5_paths: List[Path],
    movie_name: str,
    normalize_to_min_frames: bool = True,
) -> pd.DataFrame:
    """
    Create DataFrame with frame-aligned firing rate traces.
    
    Args:
        h5_paths: List of HDF5 file paths
        movie_name: Name of stimulus to process
        normalize_to_min_frames: If True, truncate to minimum frame count
    
    Returns:
        DataFrame with columns: dataset_id, unit_id, trial_idx, firing_rate, n_frames, frame_rate_hz
    """
    # First pass: determine target frame count
    frame_counts = []
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if "stimulus/frame_times/default" in f:
                all_frames = f["stimulus/frame_times/default"][:]
            elif "metadata/frame_timestamps" in f:
                all_frames = f["metadata/frame_timestamps"][:]
            else:
                continue
            
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            
            sample_unit = unit_ids[0]
            trials_path = f"units/{sample_unit}/spike_times_sectioned/{movie_name}/trials_start_end"
            
            if trials_path not in f:
                continue
            
            trials_start_end = f[trials_path][:]
            
            for trial_idx, (start, end) in enumerate(trials_start_end):
                mask = (all_frames >= start) & (all_frames < end)
                n_frames = np.sum(mask)
                frame_counts.append({
                    "file": h5_path.name,
                    "trial_idx": trial_idx,
                    "n_frames": int(n_frames),
                })
    
    if not frame_counts:
        print(f"No trials found for '{movie_name}'")
        return pd.DataFrame()
    
    frame_df = pd.DataFrame(frame_counts)
    
    # Determine target frame count
    if normalize_to_min_frames:
        target_frames = frame_df.groupby("trial_idx")["n_frames"].min().min()
    else:
        target_frames = frame_df.groupby("trial_idx")["n_frames"].max().max()
    
    # Since firing rate is per inter-frame interval, we get n_frames - 1 values
    target_bins = target_frames - 1
    
    print(f"\nProcessing '{movie_name}':")
    print(f"  Frame counts per trial: {frame_df['n_frames'].min()} - {frame_df['n_frames'].max()}")
    print(f"  Target output length: {target_bins} bins")
    
    # Second pass: compute firing rates
    rows = []
    
    for h5_path in tqdm(h5_paths, desc=f"Processing {movie_name}"):
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            # Get frame timestamps
            if "stimulus/frame_times/default" in f:
                all_frames = f["stimulus/frame_times/default"][:]
            elif "metadata/frame_timestamps" in f:
                all_frames = f["metadata/frame_timestamps"][:]
            else:
                continue
            
            for unit_id in f["units"].keys():
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
                    
                    # Get frames for this trial
                    mask = (all_frames >= trial_start) & (all_frames < trial_end)
                    trial_frames = all_frames[mask]
                    
                    if len(trial_frames) < 2:
                        continue
                    
                    # Compute frame-aligned firing rate
                    fr, mean_frame_rate = spike_times_to_frame_firing_rate(
                        spike_times=spike_times,
                        trial_start_sample=trial_start,
                        trial_end_sample=trial_end,
                        frame_samples=trial_frames,
                        acquisition_rate=acq_rate,
                    )
                    
                    # Truncate or pad to target length
                    if len(fr) > target_bins:
                        fr = fr[:target_bins]
                    elif len(fr) < target_bins:
                        fr = np.pad(fr, (0, target_bins - len(fr)), constant_values=0)
                    
                    rows.append({
                        "dataset_id": dataset_id,
                        "unit_id": unit_id,
                        "trial_idx": trial_idx,
                        "firing_rate": fr,
                        "n_frames": len(trial_frames),
                        "n_bins": len(fr),
                        "frame_rate_hz": mean_frame_rate,
                        "n_spikes": len(spike_times),
                    })
    
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Frame-Aligned Firing Rate Analysis")
    print("=" * 70)
    
    h5_files = sorted(HDF5_DIR.glob("*.h5"))
    print(f"\nFound {len(h5_files)} HDF5 files")
    
    # ==========================================================================
    # 1. Explore frame_timestamps structure
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. Frame Timestamps Structure")
    print("=" * 70)
    
    for h5_path in h5_files[:2]:  # Just first 2 for brevity
        info = explore_frame_timestamps(h5_path)
        print(f"\n{info['file']}:")
        print(f"  Frame timestamps path: {info['frame_timestamps_path']}")
        print(f"  Number of frames: {info['n_frames']:,}")
        print(f"  Frame rate: {info.get('frame_rate_hz', 'N/A'):.2f} Hz" if info.get('frame_rate_hz') else "")
        print(f"  Total duration: {info.get('total_duration_s', 'N/A'):.2f}s" if info.get('total_duration_s') else "")
        if 'timestamps_sample' in info:
            print(f"  First 5 frames (samples): {info['timestamps_sample']}")
    
    # ==========================================================================
    # 2. Analyze frame count consistency
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. Frame Count Consistency Analysis")
    print("=" * 70)
    
    frame_df = analyze_frame_consistency(h5_files)
    
    if frame_df.empty:
        print("No frame data found!")
        return
    
    # Summary by movie and trial
    summary = frame_df.groupby(['movie_name', 'trial_idx']).agg({
        'n_frames': ['min', 'max', 'mean'],
        'frame_rate_hz': 'mean',
    }).round(2)
    
    print("\nFrame counts per trial (across recordings):")
    print(summary.to_string())
    
    # Check consistency
    print("\n\nFrame Count Variation Summary:")
    for movie in frame_df['movie_name'].unique():
        movie_df = frame_df[frame_df['movie_name'] == movie]
        
        # Group by trial_idx
        for trial_idx in sorted(movie_df['trial_idx'].unique()):
            trial_df = movie_df[movie_df['trial_idx'] == trial_idx]
            n_min = trial_df['n_frames'].min()
            n_max = trial_df['n_frames'].max()
            variation = n_max - n_min
            
            status = "[OK]" if variation == 0 else f"[!] {variation} frames variation"
            print(f"  {movie} Trial {trial_idx}: {n_min}-{n_max} frames {status}")
    
    # Save frame analysis
    frame_summary_path = OUTPUT_DIR / "frame_count_per_trial.csv"
    frame_df.to_csv(frame_summary_path, index=False)
    print(f"\nSaved frame analysis to: {frame_summary_path}")
    
    # ==========================================================================
    # 3. Create frame-aligned firing rate DataFrames
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. Creating Frame-Aligned Firing Rate DataFrames")
    print("=" * 70)
    
    # Get all movie names
    movies = frame_df['movie_name'].unique()
    
    for movie_name in movies:
        print(f"\n{'='*50}")
        
        df = create_frame_aligned_firing_rate_dataframe(
            h5_files,
            movie_name=movie_name,
            normalize_to_min_frames=True,
        )
        
        if df.empty:
            continue
        
        # Summary
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Unique datasets: {df['dataset_id'].nunique()}")
        print(f"  Unique units: {df['unit_id'].nunique()}")
        
        # Check firing rate array shapes
        fr_lengths = df["firing_rate"].apply(len)
        print(f"  Firing rate length: {fr_lengths.min()} - {fr_lengths.max()} bins")
        print(f"    All equal: {fr_lengths.min() == fr_lengths.max()}")
        print(f"  Mean frame rate: {df['frame_rate_hz'].mean():.2f} Hz")
        
        # Save to parquet
        output_path = OUTPUT_DIR / f"firing_rate_frame_aligned_{movie_name}.parquet"
        df_save = df.copy()
        df_save["firing_rate"] = df_save["firing_rate"].apply(lambda x: x.tolist())
        df_save.to_parquet(output_path, index=False)
        print(f"  Saved to: {output_path.name}")
    
    # ==========================================================================
    # 4. Comparison: Frame-aligned vs Fixed-bin
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. Comparison Summary")
    print("=" * 70)
    
    print("\n+------------------------+------------------+------------------+")
    print("| Stimulus               | Frame-Aligned    | Fixed 10ms Bins  |")
    print("+------------------------+------------------+------------------+")
    
    for movie in movies:
        movie_frame_df = frame_df[frame_df['movie_name'] == movie]
        n_frames_min = movie_frame_df.groupby('trial_idx')['n_frames'].min().min()
        n_frames_max = movie_frame_df.groupby('trial_idx')['n_frames'].max().max()
        
        # Get sample duration for 10ms bin calculation
        sample_dur = movie_frame_df['trial_duration_s'].min()
        n_10ms_bins = int(sample_dur * 100)
        
        frame_str = f"{n_frames_min-1} bins" if n_frames_min == n_frames_max else f"{n_frames_min-1}-{n_frames_max-1} bins"
        
        print(f"| {movie[:22]:<22} | {frame_str:<16} | {n_10ms_bins:<16} |")
    
    print("+------------------------+------------------+------------------+")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

