"""
Direction Section Analysis for Moving Bar Stimulus

The section_bounds are in MOVIE-RELATIVE FRAME INDICES (not samples).
Spikes in trials are also stored as movie-relative frame indices.

This analysis:
1. Examines the frame-based section bounds
2. Calculates firing rates per frame within each direction section
3. Analyzes variation across recordings
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, List, Tuple, Any
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

MOVIE_NAME = "moving_h_bar_s5_d8_3x"


def explore_hdf5_structure(h5_path: Path) -> Dict[str, Any]:
    """Explore the direction_section structure in an HDF5 file."""
    info = {
        "path": h5_path.name,
        "directions": [],
        "structure": {}
    }
    
    with h5py.File(h5_path, "r") as f:
        # Get acquisition rate
        acq_rate = float(f["metadata/acquisition_rate"][()])
        info["acquisition_rate"] = acq_rate
        
        # Get frame timestamps
        if "metadata/frame_timestamps" in f:
            frame_ts = f["metadata/frame_timestamps"][:]
            info["n_frames"] = len(frame_ts)
            
            # Calculate frame rate properly
            diffs = np.diff(frame_ts)
            # frame_timestamps are in samples, convert to seconds
            frame_intervals_s = diffs / acq_rate
            info["frame_rate"] = 1.0 / np.mean(frame_intervals_s)
            info["frame_interval_samples_mean"] = np.mean(diffs)
        
        # Get section_time for movie to find movie start
        if f"stimulus/section_time/{MOVIE_NAME}" in f:
            section_time = f[f"stimulus/section_time/{MOVIE_NAME}"][:]
            movie_start_sample = section_time[0, 0]
            info["movie_start_sample"] = int(movie_start_sample)
            
            # Find movie start frame
            if "metadata/frame_timestamps" in f:
                frame_ts = f["metadata/frame_timestamps"][:]
                movie_start_frame = np.searchsorted(frame_ts, movie_start_sample)
                info["movie_start_frame"] = int(movie_start_frame)
        
        # Get first unit to explore structure
        unit_ids = list(f["units"].keys())
        if not unit_ids:
            return info
        
        sample_unit = unit_ids[0]
        dir_section_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
        
        if dir_section_path not in f:
            info["error"] = f"No direction_section found at {dir_section_path}"
            return info
        
        dir_group = f[dir_section_path]
        
        # Get directions (excluding _attrs)
        info["directions"] = [k for k in dir_group.keys() if k != "_attrs"]
        
        # Get attributes
        if "_attrs" in dir_group:
            attrs_group = dir_group["_attrs"]
            info["attrs"] = {
                key: attrs_group[key][()] for key in attrs_group.keys()
            }
        
        # Explore first direction
        if info["directions"]:
            first_dir = info["directions"][0]
            first_dir_group = dir_group[first_dir]
            info["structure"][first_dir] = {
                "keys": list(first_dir_group.keys()),
            }
            
            # Get section_bounds if exists (FRAME INDICES!)
            if "section_bounds" in first_dir_group:
                bounds = first_dir_group["section_bounds"][:]
                info["structure"][first_dir]["section_bounds_shape"] = bounds.shape
                info["structure"][first_dir]["section_bounds_values"] = bounds.tolist()
                info["structure"][first_dir]["section_bounds_note"] = "These are MOVIE-RELATIVE FRAME indices"
            
            # Get trials structure
            if "trials" in first_dir_group:
                trials_group = first_dir_group["trials"]
                trial_keys = list(trials_group.keys())
                info["structure"][first_dir]["trial_keys"] = trial_keys
                
                if trial_keys:
                    first_trial = trials_group[trial_keys[0]][:]
                    info["structure"][first_dir]["first_trial_len"] = len(first_trial)
                    info["structure"][first_dir]["first_trial_sample"] = first_trial[:5].tolist() if len(first_trial) > 0 else []
                    info["structure"][first_dir]["trial_note"] = "Spike FRAME indices relative to movie start"
    
    return info


def analyze_direction_section_bounds(h5_paths: List[Path]) -> pd.DataFrame:
    """Analyze section_bounds across all recordings and directions."""
    results = []
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            # Get frame timestamps
            if "metadata/frame_timestamps" not in f:
                continue
            frame_ts = f["metadata/frame_timestamps"][:]
            frame_intervals = np.diff(frame_ts)
            mean_frame_interval = np.mean(frame_intervals)
            
            # Get movie start frame
            if f"stimulus/section_time/{MOVIE_NAME}" not in f:
                continue
            section_time = f[f"stimulus/section_time/{MOVIE_NAME}"][:]
            movie_start_sample = section_time[0, 0]
            movie_start_frame = np.searchsorted(frame_ts, movie_start_sample)
            
            # Get first unit to get section bounds (same across all units)
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            
            sample_unit = unit_ids[0]
            dir_section_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
            
            if dir_section_path not in f:
                continue
            
            dir_group = f[dir_section_path]
            
            for direction in dir_group.keys():
                if direction == "_attrs":
                    continue
                    
                dir_data = dir_group[direction]
                
                if "section_bounds" not in dir_data:
                    continue
                
                bounds = dir_data["section_bounds"][:]  # Movie-relative FRAME indices
                
                for rep_idx, (start_frame_rel, end_frame_rel) in enumerate(bounds):
                    # Convert to absolute frame indices
                    start_frame_abs = movie_start_frame + start_frame_rel
                    end_frame_abs = movie_start_frame + end_frame_rel
                    
                    # Number of frames in section
                    n_frames = end_frame_rel - start_frame_rel
                    
                    # Calculate duration in samples (approximate)
                    duration_samples = int(n_frames * mean_frame_interval)
                    duration_s = duration_samples / acq_rate
                    
                    results.append({
                        "dataset_id": dataset_id,
                        "direction": int(direction),
                        "rep_idx": rep_idx,
                        "start_frame_rel": int(start_frame_rel),
                        "end_frame_rel": int(end_frame_rel),
                        "start_frame_abs": int(start_frame_abs),
                        "end_frame_abs": int(end_frame_abs),
                        "n_frames": int(n_frames),
                        "duration_samples_approx": duration_samples,
                        "duration_s_approx": duration_s,
                        "movie_start_frame": int(movie_start_frame),
                    })
    
    return pd.DataFrame(results)


def calculate_direction_firing_rates(
    h5_paths: List[Path],
    target_frames_per_direction: Dict[int, int] = None,
) -> pd.DataFrame:
    """
    Calculate per-frame spike counts for each direction section.
    
    The spikes are already in movie-relative frame indices, so we just count
    spikes per frame within the section bounds.
    
    Args:
        h5_paths: List of HDF5 file paths
        target_frames_per_direction: Dict mapping direction -> target frame count
    
    Returns:
        DataFrame with spike counts per frame for each direction section
    """
    rows = []
    
    for h5_path in tqdm(h5_paths, desc="Processing direction sections"):
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            # Get frame timestamps to calculate frame rate
            if "metadata/frame_timestamps" not in f:
                continue
            frame_ts = f["metadata/frame_timestamps"][:]
            frame_intervals = np.diff(frame_ts) / acq_rate  # in seconds
            mean_frame_interval_s = np.mean(frame_intervals)
            frame_rate = 1.0 / mean_frame_interval_s
            
            for unit_id in f["units"].keys():
                dir_section_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
                
                if dir_section_path not in f:
                    continue
                
                dir_group = f[dir_section_path]
                
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
                        start_frame, end_frame = bounds[rep_idx]
                        n_frames = end_frame - start_frame
                        
                        # Skip invalid sections (negative frame count)
                        if n_frames <= 0:
                            continue
                        
                        # Get spike frame indices (movie-relative)
                        spike_frames = trials_group[rep_idx_str][:]
                        
                        # Create per-frame spike counts
                        # Each bin is one frame
                        counts = np.zeros(n_frames, dtype=np.float32)
                        
                        for spike_frame in spike_frames:
                            # spike_frame is movie-relative
                            bin_idx = spike_frame - start_frame
                            if 0 <= bin_idx < n_frames:
                                counts[bin_idx] += 1
                        
                        # Convert to firing rate (Hz) = spikes/frame * frames/second
                        firing_rate = counts * frame_rate
                        
                        original_bins = len(firing_rate)
                        
                        # Normalize to target if specified
                        dir_int = int(direction)
                        if target_frames_per_direction and dir_int in target_frames_per_direction:
                            target = target_frames_per_direction[dir_int]
                            if len(firing_rate) > target:
                                firing_rate = firing_rate[:target]
                            elif len(firing_rate) < target:
                                firing_rate = np.pad(firing_rate, (0, target - len(firing_rate)), constant_values=0)
                        
                        rows.append({
                            "dataset_id": dataset_id,
                            "unit_id": unit_id,
                            "direction": dir_int,
                            "rep_idx": rep_idx,
                            "firing_rate": firing_rate,
                            "n_bins": len(firing_rate),
                            "original_bins": original_bins,
                            "n_spikes": len(spike_frames),
                            "frame_rate_hz": frame_rate,
                        })
    
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Direction Section Analysis for Moving Bar Stimulus")
    print("=" * 70)
    
    h5_files = sorted(HDF5_DIR.glob("*.h5"))
    print(f"\nFound {len(h5_files)} HDF5 files")
    
    # ==========================================================================
    # 1. Explore HDF5 structure
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. HDF5 Structure Exploration")
    print("=" * 70)
    
    sample_info = explore_hdf5_structure(h5_files[0])
    print(f"\nSample file: {sample_info['path']}")
    print(f"Acquisition rate: {sample_info.get('acquisition_rate', 'N/A')} Hz")
    print(f"Total frames: {sample_info.get('n_frames', 'N/A')}")
    print(f"Frame rate: {sample_info.get('frame_rate', 0):.2f} Hz")
    print(f"Frame interval (samples): {sample_info.get('frame_interval_samples_mean', 0):.1f}")
    print(f"Movie start frame: {sample_info.get('movie_start_frame', 'N/A')}")
    print(f"Directions: {sample_info.get('directions', [])}")
    
    if sample_info.get("attrs"):
        print("\nDirection Section Attributes:")
        for k, v in sample_info["attrs"].items():
            print(f"  {k}: {v}")
    
    if sample_info.get("structure"):
        print("\nStructure of first direction:")
        for k, v in sample_info["structure"].items():
            print(f"  Direction {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
    
    # ==========================================================================
    # 2. Analyze section bounds
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. Section Bounds Analysis (Frame Indices)")
    print("=" * 70)
    
    bounds_df = analyze_direction_section_bounds(h5_files)
    
    if bounds_df.empty:
        print("No direction sections found!")
        return
    
    print(f"\nTotal direction sections: {len(bounds_df)}")
    print(f"Unique directions: {sorted(bounds_df['direction'].unique())}")
    print(f"Repetitions per direction: {bounds_df.groupby('direction')['rep_idx'].max().iloc[0] + 1}")
    
    # Per-direction statistics
    print("\n" + "-" * 50)
    print("Per-Direction Frame Statistics:")
    print("-" * 50)
    
    dir_stats = bounds_df.groupby("direction").agg({
        "n_frames": ["min", "max", "mean", lambda x: x.max() - x.min()],
        "duration_s_approx": ["min", "max", "mean"],
    })
    dir_stats.columns = [
        "frames_min", "frames_max", "frames_mean", "frames_var",
        "duration_min", "duration_max", "duration_mean",
    ]
    
    print("\nDirection | Frames (min-max) [variation] | Duration (s)")
    print("-" * 70)
    for direction in sorted(bounds_df["direction"].unique()):
        row = dir_stats.loc[direction]
        status = "✓" if row["frames_var"] == 0 else f"!"
        print(f"    {direction:>3}   | {int(row['frames_min']):>4} - {int(row['frames_max']):<4} [{int(row['frames_var']):>2}] {status}      | "
              f"{row['duration_min']:.3f} - {row['duration_max']:.3f}")
    
    # ==========================================================================
    # 3. Detailed per-recording breakdown
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. Per-Recording Frame Count Breakdown")
    print("=" * 70)
    
    pivot = bounds_df.pivot_table(
        index=["dataset_id", "rep_idx"],
        columns="direction",
        values="n_frames",
        aggfunc="first"
    )
    print("\nFrame counts per direction (rows=recording/rep, cols=direction):")
    print(pivot.to_string())
    
    # Check consistency per direction and repetition
    print("\n" + "-" * 50)
    print("Consistency Check by Direction and Repetition:")
    print("-" * 50)
    
    for direction in sorted(bounds_df["direction"].unique()):
        for rep_idx in sorted(bounds_df["rep_idx"].unique()):
            dir_rep_df = bounds_df[(bounds_df["direction"] == direction) & (bounds_df["rep_idx"] == rep_idx)]
            frames_unique = dir_rep_df["n_frames"].unique()
            
            status = "✓" if len(frames_unique) == 1 else f"✗ VARIES: {sorted(frames_unique)}"
            print(f"  Dir {direction:>3}, Rep {rep_idx}: {sorted(frames_unique)} {status}")
    
    # ==========================================================================
    # 4. Determine target frame counts
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. Target Frame Counts for Normalization")
    print("=" * 70)
    
    target_frames = {}
    for direction in sorted(bounds_df["direction"].unique()):
        dir_df = bounds_df[bounds_df["direction"] == direction]
        # Filter out invalid (negative) frame counts
        valid_frames = dir_df[dir_df["n_frames"] > 0]["n_frames"]
        
        if len(valid_frames) == 0:
            print(f"  Direction {direction:>3}: ✗ NO VALID DATA")
            continue
        
        min_frames = int(valid_frames.min())
        max_frames = int(valid_frames.max())
        variation = max_frames - min_frames
        target_frames[direction] = min_frames
        
        invalid_count = len(dir_df) - len(valid_frames)
        invalid_note = f" ({invalid_count} invalid excluded)" if invalid_count > 0 else ""
        status = "✓ consistent" if variation == 0 else f"! truncating {variation} frames"
        print(f"  Direction {direction:>3}: {min_frames:>3} - {max_frames:>3} frames -> {min_frames:>3} bins (target) {status}{invalid_note}")
    
    # ==========================================================================
    # 5. Calculate inter-frame firing rates
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. Calculating Per-Frame Firing Rates")
    print("=" * 70)
    
    fr_df = calculate_direction_firing_rates(h5_files, target_frames)
    
    if fr_df.empty:
        print("No firing rates calculated!")
        return
    
    print(f"\nDataFrame shape: {fr_df.shape}")
    print(f"Unique datasets: {fr_df['dataset_id'].nunique()}")
    print(f"Unique units: {fr_df['unit_id'].nunique()}")
    print(f"Unique directions: {sorted(fr_df['direction'].unique())}")
    print(f"Frame rate: {fr_df['frame_rate_hz'].iloc[0]:.2f} Hz")
    
    # Check output consistency per direction
    print("\nOutput bin counts per direction:")
    for direction in sorted(fr_df["direction"].unique()):
        dir_fr = fr_df[fr_df["direction"] == direction]
        bins = dir_fr["n_bins"]
        orig_bins = dir_fr["original_bins"]
        
        status = "✓ ALL EQUAL" if bins.min() == bins.max() else f"✗ {bins.min()}-{bins.max()}"
        duration_s = bins.iloc[0] / fr_df["frame_rate_hz"].iloc[0]
        print(f"  Direction {direction:>3}: {bins.iloc[0]:>3} bins (~{duration_s:.2f}s) {status} (original: {orig_bins.min()}-{orig_bins.max()})")
    
    # ==========================================================================
    # 6. Save results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("6. Saving Results")
    print("=" * 70)
    
    # Save section bounds analysis
    bounds_path = OUTPUT_DIR / "direction_section_bounds.csv"
    bounds_df.to_csv(bounds_path, index=False)
    print(f"  Saved bounds analysis to: {bounds_path.name}")
    
    # Save firing rates per direction
    for direction in sorted(fr_df["direction"].unique()):
        dir_fr = fr_df[fr_df["direction"] == direction].copy()
        dir_fr["firing_rate"] = dir_fr["firing_rate"].apply(lambda x: x.tolist())
        
        output_path = OUTPUT_DIR / f"firing_rate_direction_{direction}.parquet"
        dir_fr.to_parquet(output_path, index=False)
        print(f"  Saved direction {direction} firing rates to: {output_path.name}")
    
    # Save combined
    all_fr = fr_df.copy()
    all_fr["firing_rate"] = all_fr["firing_rate"].apply(lambda x: x.tolist())
    combined_path = OUTPUT_DIR / "firing_rate_all_directions.parquet"
    all_fr.to_parquet(combined_path, index=False)
    print(f"  Saved all directions to: {combined_path.name}")
    
    # ==========================================================================
    # 7. Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("7. Summary")
    print("=" * 70)
    
    print(f"""
Moving Bar Direction Section Analysis:
--------------------------------------
- Stimulus: {MOVIE_NAME}
- Directions: {len(bounds_df['direction'].unique())} (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Repetitions per direction: {bounds_df.groupby('direction')['rep_idx'].max().iloc[0] + 1}
- Total recordings: {bounds_df['dataset_id'].nunique()}
- Frame rate: ~{fr_df['frame_rate_hz'].iloc[0]:.2f} Hz

Frame Count Variation by Direction:
""")
    
    for direction in sorted(bounds_df["direction"].unique()):
        dir_df = bounds_df[bounds_df["direction"] == direction]
        frames_min = int(dir_df["n_frames"].min())
        frames_max = int(dir_df["n_frames"].max())
        frames_var = frames_max - frames_min
        status = "✓ Consistent" if frames_var == 0 else f"! {frames_var} frame variation"
        print(f"  Direction {direction:>3}°: {frames_min}-{frames_max} frames {status}")
    
    print(f"""
Output Firing Rate Arrays:
""")
    total_rows = 0
    for direction in sorted(fr_df["direction"].unique()):
        dir_fr = fr_df[fr_df["direction"] == direction]
        n_bins = dir_fr["n_bins"].iloc[0]
        duration_s = n_bins / fr_df["frame_rate_hz"].iloc[0]
        n_rows = len(dir_fr)
        total_rows += n_rows
        print(f"  Direction {direction:>3}°: {n_bins:>3} bins (~{duration_s:.2f}s), {n_rows} rows")
    
    print(f"\n  Total rows across all directions: {total_rows}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
