"""
ipRGC Firing Rate at 60Hz Sample-Based Bins

Since iprgc_test occurs outside the video frame period (no frame_timestamps),
we use sample-based bins at 60Hz (equivalent to frame rate).

At 20kHz acquisition:
- 60Hz = 20000/60 = 333.33 samples per bin
- 120s trial = 7200 bins
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import List, Tuple
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

MOVIE_NAME = "iprgc_test"
TARGET_RATE_HZ = 60.0  # Target firing rate bin size (60 Hz like video frames)


def spike_times_to_sample_based_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    target_rate_hz: float = 60.0,
    acquisition_rate: float = 20000.0,
    target_bins: int = None,
) -> Tuple[np.ndarray, int]:
    """
    Convert spike times to firing rate using sample-based bins at target Hz.
    
    Args:
        spike_times: Spike times in absolute sample indices
        trial_start_sample: Trial start in samples
        trial_end_sample: Trial end in samples
        target_rate_hz: Target bin rate (Hz)
        acquisition_rate: Samples per second
        target_bins: If set, truncate/pad to this many bins
    
    Returns:
        firing_rate: Firing rate in Hz for each time bin
        n_bins: Number of bins before truncation/padding
    """
    # Calculate bin size in samples
    samples_per_bin = acquisition_rate / target_rate_hz  # 333.33 for 60Hz at 20kHz
    
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
    
    original_bins = len(firing_rate)
    
    # Normalize length if requested
    if target_bins is not None:
        if len(firing_rate) > target_bins:
            firing_rate = firing_rate[:target_bins]
        elif len(firing_rate) < target_bins:
            firing_rate = np.pad(firing_rate, (0, target_bins - len(firing_rate)), constant_values=0)
    
    return firing_rate, original_bins


def analyze_iprgc_trial_lengths(h5_paths: List[Path]) -> pd.DataFrame:
    """Analyze iprgc_test trial lengths across recordings."""
    results = []
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            # Get first unit's trial boundaries
            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            
            sample_unit = unit_ids[0]
            trials_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_NAME}/trials_start_end"
            
            if trials_path not in f:
                continue
            
            trials_start_end = f[trials_path][:]
            
            for trial_idx, (start, end) in enumerate(trials_start_end):
                duration_samples = end - start
                duration_s = duration_samples / acq_rate
                
                # Calculate bins at 60Hz
                samples_per_bin = acq_rate / TARGET_RATE_HZ
                n_bins = int(np.ceil(duration_samples / samples_per_bin))
                
                results.append({
                    "dataset_id": dataset_id,
                    "trial_idx": trial_idx,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "duration_samples": int(duration_samples),
                    "duration_s": duration_s,
                    "n_bins_60hz": n_bins,
                    "expected_bins_60hz": int(duration_s * TARGET_RATE_HZ),
                })
    
    return pd.DataFrame(results)


def create_iprgc_firing_rate_dataframe(
    h5_paths: List[Path],
    target_rate_hz: float = 60.0,
) -> pd.DataFrame:
    """
    Create DataFrame with sample-based firing rate for iprgc_test.
    
    Returns:
        DataFrame with firing rate at specified Hz
    """
    # First pass: analyze trial lengths
    trial_df = analyze_iprgc_trial_lengths(h5_paths)
    
    if trial_df.empty:
        print("No iprgc_test trials found!")
        return pd.DataFrame()
    
    # Determine target bin count (minimum to ensure consistency)
    target_bins = trial_df["n_bins_60hz"].min()
    
    print(f"\niprgc_test Trial Length Analysis:")
    print(f"  Trials found: {len(trial_df)}")
    print(f"  Duration (samples): {trial_df['duration_samples'].min():,} - {trial_df['duration_samples'].max():,}")
    print(f"  Duration (seconds): {trial_df['duration_s'].min():.4f} - {trial_df['duration_s'].max():.4f}")
    print(f"  Bins at {target_rate_hz}Hz: {trial_df['n_bins_60hz'].min()} - {trial_df['n_bins_60hz'].max()}")
    print(f"  Target output: {target_bins} bins")
    
    # Group by trial to show variation
    print("\n  Per-trial breakdown:")
    for trial_idx in sorted(trial_df["trial_idx"].unique()):
        tdf = trial_df[trial_df["trial_idx"] == trial_idx]
        bins_min = tdf["n_bins_60hz"].min()
        bins_max = tdf["n_bins_60hz"].max()
        dur_min = tdf["duration_s"].min()
        dur_max = tdf["duration_s"].max()
        status = "[OK]" if bins_min == bins_max else f"[!] {bins_max - bins_min} bins variation"
        print(f"    Trial {trial_idx}: {bins_min}-{bins_max} bins, {dur_min:.3f}-{dur_max:.3f}s {status}")
    
    # Second pass: compute firing rates
    rows = []
    
    for h5_path in tqdm(h5_paths, desc="Processing iprgc_test"):
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            acq_rate = float(f["metadata/acquisition_rate"][()])
            
            for unit_id in f["units"].keys():
                trials_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/trials_spike_times"
                starts_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/trials_start_end"
                
                if trials_path not in f or starts_path not in f:
                    continue
                
                trials_start_end = f[starts_path][:]
                trials_group = f[trials_path]
                
                for trial_idx_str in trials_group.keys():
                    trial_idx = int(trial_idx_str)
                    spike_times = trials_group[trial_idx_str][:]
                    
                    trial_start = trials_start_end[trial_idx, 0]
                    trial_end = trials_start_end[trial_idx, 1]
                    
                    # Compute firing rate at target Hz
                    fr, original_bins = spike_times_to_sample_based_firing_rate(
                        spike_times=spike_times,
                        trial_start_sample=trial_start,
                        trial_end_sample=trial_end,
                        target_rate_hz=target_rate_hz,
                        acquisition_rate=acq_rate,
                        target_bins=target_bins,
                    )
                    
                    rows.append({
                        "dataset_id": dataset_id,
                        "unit_id": unit_id,
                        "trial_idx": trial_idx,
                        "firing_rate": fr,
                        "n_bins": len(fr),
                        "original_bins": original_bins,
                        "n_spikes": len(spike_times),
                        "bin_rate_hz": target_rate_hz,
                    })
    
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("ipRGC Firing Rate Analysis (60Hz Sample-Based Bins)")
    print("=" * 70)
    
    h5_files = sorted(HDF5_DIR.glob("*.h5"))
    print(f"\nFound {len(h5_files)} HDF5 files")
    
    # ==========================================================================
    # 1. Analyze trial lengths
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. Trial Length Comparison")
    print("=" * 70)
    
    trial_df = analyze_iprgc_trial_lengths(h5_files)
    
    if trial_df.empty:
        print("No iprgc_test trials found!")
        return
    
    print("\nDetailed trial information:")
    print(trial_df.to_string(index=False))
    
    # Check consistency
    print("\n" + "-" * 50)
    print("Consistency Check:")
    print("-" * 50)
    
    for col in ["duration_samples", "duration_s", "n_bins_60hz"]:
        min_val = trial_df[col].min()
        max_val = trial_df[col].max()
        variation = max_val - min_val
        
        if col == "duration_s":
            print(f"  {col}: {min_val:.6f} - {max_val:.6f} (variation: {variation:.6f}s)")
        else:
            print(f"  {col}: {min_val:,} - {max_val:,} (variation: {variation:,})")
    
    # ==========================================================================
    # 2. Compare with other stimuli
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. Comparison with Other Stimuli (at 60Hz)")
    print("=" * 70)
    
    # Calculate 60Hz bins for other stimuli from previous analysis
    frame_count_path = OUTPUT_DIR / "frame_count_per_trial.csv"
    if frame_count_path.exists():
        frame_df = pd.read_csv(frame_count_path)
        
        print("\nBins at 60Hz (frame-based vs sample-based):")
        print("-" * 60)
        
        for movie in frame_df["movie_name"].unique():
            movie_df = frame_df[frame_df["movie_name"] == movie]
            n_frames_min = movie_df["n_frames"].min()
            n_frames_max = movie_df["n_frames"].max()
            
            # For iprgc_test, use sample-based calculation
            if movie == MOVIE_NAME:
                bins_min = trial_df["n_bins_60hz"].min()
                bins_max = trial_df["n_bins_60hz"].max()
                method = "sample-based"
            else:
                bins_min = n_frames_min - 1  # n_frames - 1 = n_intervals
                bins_max = n_frames_max - 1
                method = "frame-based"
            
            variation = bins_max - bins_min
            status = "✓ consistent" if variation == 0 else f"! {variation} bins variation"
            
            print(f"  {movie[:35]:<35} {bins_min:>6} bins ({method}) {status}")
    
    # ==========================================================================
    # 3. Create firing rate DataFrame
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. Creating Firing Rate DataFrame")
    print("=" * 70)
    
    df = create_iprgc_firing_rate_dataframe(h5_files, target_rate_hz=TARGET_RATE_HZ)
    
    if df.empty:
        return
    
    print(f"\nDataFrame Summary:")
    print(f"  Shape: {df.shape}")
    print(f"  Unique datasets: {df['dataset_id'].nunique()}")
    print(f"  Unique units: {df['unit_id'].nunique()}")
    print(f"  Trials per unit: {df.groupby('unit_id')['trial_idx'].count().mean():.1f}")
    
    # Check output consistency
    fr_lengths = df["firing_rate"].apply(len)
    print(f"\n  Firing rate length: {fr_lengths.min()} - {fr_lengths.max()} bins")
    print(f"  All equal: {fr_lengths.min() == fr_lengths.max()}")
    print(f"  Duration at 60Hz: {fr_lengths.min() / TARGET_RATE_HZ:.2f}s")
    
    # Original vs normalized
    print(f"\n  Original bins (before normalization):")
    orig_bins = df["original_bins"]
    print(f"    Range: {orig_bins.min()} - {orig_bins.max()}")
    print(f"    Truncated by: {orig_bins.max() - fr_lengths.min()} bins")
    
    # Save to parquet
    output_path = OUTPUT_DIR / f"firing_rate_60hz_sample_based_{MOVIE_NAME}.parquet"
    df_save = df.copy()
    df_save["firing_rate"] = df_save["firing_rate"].apply(lambda x: x.tolist())
    df_save.to_parquet(output_path, index=False)
    print(f"\n  Saved to: {output_path.name}")
    
    # Save trial length analysis
    trial_path = OUTPUT_DIR / f"trial_length_analysis_{MOVIE_NAME}.csv"
    trial_df.to_csv(trial_path, index=False)
    print(f"  Saved trial analysis to: {trial_path.name}")
    
    # ==========================================================================
    # 4. Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. Summary")
    print("=" * 70)
    
    print(f"""
iprgc_test Analysis:
--------------------
- Trial duration: {trial_df['duration_s'].iloc[0]:.1f} seconds (exactly {trial_df['duration_samples'].iloc[0]:,} samples)
- Trials per recording: {len(trial_df['trial_idx'].unique())}
- Total recordings: {len(trial_df['dataset_id'].unique())}

Bin Calculation:
- Acquisition rate: 20,000 Hz
- Target rate: {TARGET_RATE_HZ} Hz
- Samples per bin: {20000 / TARGET_RATE_HZ:.2f}
- Output bins: {fr_lengths.min()} (= {fr_lengths.min() / TARGET_RATE_HZ:.2f}s at 60Hz)

Consistency:
- Duration samples: {"✓ ALL EQUAL" if trial_df['duration_samples'].nunique() == 1 else "✗ Variable"}
- Output bins: {"✓ ALL EQUAL" if fr_lengths.min() == fr_lengths.max() else "✗ Variable"}
""")
    
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

