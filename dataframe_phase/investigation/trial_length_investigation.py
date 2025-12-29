"""
Trial Length Investigation - Focus on trials_start_end

This script investigates the trial boundaries stored in:
    /units/{unit_id}/spike_times_sectioned/{movie_name}/trials_start_end

Each stimulus may have multiple trials, and we need to understand:
1. How many trials per stimulus
2. The start/end times for each trial
3. Whether trial lengths are consistent across recordings
"""

import sys
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# =============================================================================
# Configuration
# =============================================================================

HDF5_DIR = project_root / "Projects/unified_pipeline/export_all_steps"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def investigate_trials_start_end(h5_path: Path):
    """
    Investigate trials_start_end structure for all stimuli in an HDF5 file.
    """
    results = []
    
    with h5py.File(h5_path, "r") as f:
        dataset_id = h5_path.stem
        
        # Get acquisition rate
        acq_rate = 20000.0
        if "metadata/acquisition_rate" in f:
            acq_rate = float(f["metadata/acquisition_rate"][()])
        
        # Get first unit to examine structure
        unit_ids = list(f["units"].keys())
        if not unit_ids:
            return results
        
        sample_unit = unit_ids[0]
        unit_path = f"units/{sample_unit}"
        
        if "spike_times_sectioned" not in f[unit_path]:
            return results
        
        # Iterate through all stimuli
        for movie_name in f[f"{unit_path}/spike_times_sectioned"].keys():
            trials_path = f"{unit_path}/spike_times_sectioned/{movie_name}/trials_start_end"
            
            if trials_path not in f:
                continue
            
            trials_start_end = f[trials_path][:]
            n_trials = len(trials_start_end)
            
            for trial_idx, (start, end) in enumerate(trials_start_end):
                duration_samples = end - start
                duration_s = duration_samples / acq_rate
                
                results.append({
                    "dataset_id": dataset_id,
                    "movie_name": movie_name,
                    "trial_idx": trial_idx,
                    "n_trials_total": n_trials,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "duration_samples": int(duration_samples),
                    "duration_s": duration_s,
                    "acq_rate": acq_rate,
                })
    
    return results


def main():
    print("=" * 80)
    print("Trial Length Investigation - Focus on trials_start_end")
    print("=" * 80)
    
    # Find HDF5 files
    h5_files = sorted(HDF5_DIR.glob("*.h5"))
    print(f"\nFound {len(h5_files)} HDF5 files in {HDF5_DIR}")
    
    if not h5_files:
        print("No HDF5 files found!")
        return
    
    # Collect all trial data
    all_trials = []
    for h5_path in h5_files:
        print(f"  Processing: {h5_path.name}")
        trials = investigate_trials_start_end(h5_path)
        all_trials.extend(trials)
    
    df = pd.DataFrame(all_trials)
    
    if df.empty:
        print("No trial data found!")
        return
    
    # ==========================================================================
    # 1. Summary by stimulus type
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. Trial Count and Duration by Stimulus Type")
    print("=" * 80)
    
    for movie_name in sorted(df["movie_name"].unique()):
        movie_df = df[df["movie_name"] == movie_name]
        
        print(f"\n--- {movie_name} ---")
        
        # Group by dataset to show trials per recording
        for dataset_id in sorted(movie_df["dataset_id"].unique()):
            ds_trials = movie_df[movie_df["dataset_id"] == dataset_id]
            n_trials = ds_trials["n_trials_total"].iloc[0]
            
            print(f"\n  Dataset: {dataset_id}")
            print(f"    Number of trials: {n_trials}")
            
            for _, row in ds_trials.iterrows():
                print(f"    Trial {row['trial_idx']}: "
                      f"start={row['start_sample']:,} end={row['end_sample']:,} "
                      f"duration={row['duration_s']:.3f}s ({row['duration_samples']:,} samples)")
    
    # ==========================================================================
    # 2. Cross-recording trial length comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. Trial Length Comparison Across Recordings (per trial index)")
    print("=" * 80)
    
    summary_rows = []
    
    for movie_name in sorted(df["movie_name"].unique()):
        movie_df = df[df["movie_name"] == movie_name]
        
        # Get unique trial indices
        trial_indices = sorted(movie_df["trial_idx"].unique())
        
        print(f"\n--- {movie_name} ({len(trial_indices)} trials per recording) ---")
        
        for trial_idx in trial_indices:
            trial_df = movie_df[movie_df["trial_idx"] == trial_idx]
            
            durations = trial_df["duration_samples"].values
            durations_s = trial_df["duration_s"].values
            
            dur_min = durations.min()
            dur_max = durations.max()
            dur_range = dur_max - dur_min
            dur_mean = durations.mean()
            
            print(f"  Trial {trial_idx}:")
            print(f"    Recordings: {len(trial_df)}")
            print(f"    Duration range: {durations_s.min():.4f}s to {durations_s.max():.4f}s")
            print(f"    Sample variation: {dur_range} samples ({dur_range / 20000 * 1000:.2f} ms)")
            print(f"    All same length: {dur_min == dur_max}")
            
            summary_rows.append({
                "movie_name": movie_name,
                "trial_idx": trial_idx,
                "n_recordings": len(trial_df),
                "dur_min_s": durations_s.min(),
                "dur_max_s": durations_s.max(),
                "dur_mean_s": durations_s.mean(),
                "dur_range_s": durations_s.max() - durations_s.min(),
                "samples_min": int(dur_min),
                "samples_max": int(dur_max),
                "samples_range": int(dur_range),
                "all_same_length": dur_min == dur_max,
            })
    
    # ==========================================================================
    # 3. Summary table
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. Summary Table (per trial index)")
    print("=" * 80)
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    
    # Save to CSV
    summary_path = OUTPUT_DIR / "trial_start_end_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    
    # Save full trial data
    full_path = OUTPUT_DIR / "trial_start_end_full.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved full data to: {full_path}")
    
    # ==========================================================================
    # 4. Recommended target lengths
    # ==========================================================================
    print("\n" + "=" * 80)
    print("4. Recommended Target Lengths for Equal-Length Traces")
    print("=" * 80)
    
    target_lengths = []
    
    for movie_name in sorted(df["movie_name"].unique()):
        movie_df = df[df["movie_name"] == movie_name]
        
        # For each trial index, find minimum duration across recordings
        trial_indices = sorted(movie_df["trial_idx"].unique())
        
        trial_targets = []
        for trial_idx in trial_indices:
            trial_df = movie_df[movie_df["trial_idx"] == trial_idx]
            min_samples = trial_df["duration_samples"].min()
            trial_targets.append({
                "trial_idx": trial_idx,
                "min_samples": int(min_samples),
                "min_duration_s": min_samples / 20000,
            })
        
        # Check if all trials have same target length
        unique_targets = set(t["min_samples"] for t in trial_targets)
        
        print(f"\n--- {movie_name} ---")
        print(f"  Trials per recording: {len(trial_indices)}")
        
        if len(unique_targets) == 1:
            target = list(unique_targets)[0]
            bin_size_ms = 10.0
            n_bins = int(target / (bin_size_ms / 1000 * 20000))
            print(f"  [OK] All trials can use same target: {target:,} samples ({target/20000:.3f}s)")
            print(f"       At 10ms bins: {n_bins} bins")
            
            target_lengths.append({
                "movie_name": movie_name,
                "n_trials": len(trial_indices),
                "target_samples": target,
                "target_duration_s": target / 20000,
                "target_bins_10ms": n_bins,
                "consistent": True,
            })
        else:
            print(f"  [!] Trials have different target lengths:")
            for t in trial_targets:
                print(f"      Trial {t['trial_idx']}: {t['min_samples']:,} samples ({t['min_duration_s']:.3f}s)")
            
            # Use the minimum across all trials
            overall_min = min(t["min_samples"] for t in trial_targets)
            n_bins = int(overall_min / (10.0 / 1000 * 20000))
            print(f"  Recommended: Use {overall_min:,} samples ({overall_min/20000:.3f}s) for all trials")
            
            target_lengths.append({
                "movie_name": movie_name,
                "n_trials": len(trial_indices),
                "target_samples": overall_min,
                "target_duration_s": overall_min / 20000,
                "target_bins_10ms": n_bins,
                "consistent": False,
            })
    
    # Print target lengths table
    print("\n" + "-" * 60)
    print("Target Lengths Summary:")
    print("-" * 60)
    
    target_df = pd.DataFrame(target_lengths)
    print("\n" + target_df.to_string(index=False))
    
    # Save target lengths
    target_path = OUTPUT_DIR / "recommended_target_lengths.csv"
    target_df.to_csv(target_path, index=False)
    print(f"\nSaved target lengths to: {target_path}")
    
    print("\n" + "=" * 80)
    print("Investigation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
