"""
Create Firing Rate DataFrames from HDF5 Files

This script converts spike times to firing rate traces and creates
pandas DataFrames suitable for machine learning and analysis.

Key features:
- Equal-length traces for all trials of the same stimulus type
- Multiple normalization strategies (truncate, pad, resample)
- Support for saving to Parquet format with nested arrays
"""

import sys
from pathlib import Path
from typing import Optional, List, Literal, Union
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# =============================================================================
# Configuration
# =============================================================================

HDF5_DIR = project_root / "Projects/unified_pipeline/export_all_steps"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Core Functions
# =============================================================================

def spike_times_to_firing_rate(
    spike_times: np.ndarray,
    trial_start: int,
    trial_end: int,
    bin_size_ms: float = 10.0,
    acquisition_rate: float = 20000.0,
    normalize_length: Optional[int] = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Convert spike times to firing rate trace.
    
    Args:
        spike_times: Spike times in sample indices (absolute)
        trial_start: Trial start in samples
        trial_end: Trial end in samples
        bin_size_ms: Time bin size in milliseconds
        acquisition_rate: Samples per second
        normalize_length: If set, pad/truncate to this many bins
        pad_value: Value to use for padding (0.0 or np.nan)
    
    Returns:
        Firing rate in Hz for each time bin
    """
    # Convert to relative times (samples from trial start)
    relative_spikes = spike_times.astype(np.int64) - int(trial_start)
    
    # Filter to spikes within trial
    trial_duration = int(trial_end) - int(trial_start)
    mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[mask]
    
    # Create bins
    bin_size_samples = int(bin_size_ms / 1000.0 * acquisition_rate)
    n_bins = int(np.ceil(trial_duration / bin_size_samples))
    
    # Count spikes per bin
    bin_edges = np.arange(0, n_bins + 1) * bin_size_samples
    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
    
    # Convert to firing rate (Hz)
    bin_duration_s = bin_size_ms / 1000.0
    firing_rate = counts.astype(np.float32) / bin_duration_s
    
    # Normalize length if requested
    if normalize_length is not None:
        if len(firing_rate) > normalize_length:
            firing_rate = firing_rate[:normalize_length]  # Truncate
        elif len(firing_rate) < normalize_length:
            padding = np.full(normalize_length - len(firing_rate), pad_value, dtype=np.float32)
            firing_rate = np.concatenate([firing_rate, padding])
    
    return firing_rate


def get_trial_info_for_movie(h5file: h5py.File, movie_name: str) -> dict:
    """Get trial boundaries and metadata for a movie."""
    result = {
        "movie_name": movie_name,
        "trial_starts": [],
        "trial_ends": [],
        "trial_durations_samples": [],
        "acquisition_rate": 20000.0,
    }
    
    # Get acquisition rate
    if "metadata" in h5file:
        result["acquisition_rate"] = float(h5file["metadata/acquisition_rate"][()])
    
    # Try stimulus/section_time
    section_time_path = f"stimulus/section_time/{movie_name}"
    if section_time_path in h5file:
        section_time = h5file[section_time_path][:]
        result["trial_starts"] = section_time[:, 0].tolist()
        result["trial_ends"] = section_time[:, 1].tolist()
        result["trial_durations_samples"] = (section_time[:, 1] - section_time[:, 0]).tolist()
    
    return result


def create_firing_rate_dataframe(
    h5_path: Union[str, Path],
    movie_name: str,
    bin_size_ms: float = 10.0,
    normalize_strategy: Literal["min", "max", "median", "none"] = "min",
    target_bins: Optional[int] = None,
    pad_value: float = 0.0,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """
    Create DataFrame with firing rate traces for all units and trials.
    
    Args:
        h5_path: Path to HDF5 file
        movie_name: Name of movie/stimulus to extract
        bin_size_ms: Bin size in milliseconds
        normalize_strategy: How to normalize lengths ("min", "max", "median", "none")
        target_bins: Override target length (number of bins)
        pad_value: Value for padding (0.0 or np.nan)
        include_metadata: Include unit metadata columns
    
    Returns:
        DataFrame with columns:
        - dataset_id, unit_id, trial_idx
        - firing_rate: numpy array of firing rates
        - n_spikes, n_bins, trial_duration_s
        - (optional) row, col, cell_type, etc.
    """
    h5_path = Path(h5_path)
    
    with h5py.File(h5_path, "r") as f:
        dataset_id = f["metadata/dataset_id"][()].decode() if "metadata/dataset_id" in f else h5_path.stem
        trial_info = get_trial_info_for_movie(f, movie_name)
        acq_rate = trial_info["acquisition_rate"]
        
        if not trial_info["trial_starts"]:
            print(f"Warning: No trials found for '{movie_name}' in {h5_path.name}")
            return pd.DataFrame()
        
        # Determine target length
        durations = np.array(trial_info["trial_durations_samples"])
        bin_size_samples = int(bin_size_ms / 1000.0 * acq_rate)
        
        if target_bins is not None:
            n_target_bins = target_bins
        elif normalize_strategy == "min":
            n_target_bins = int(durations.min() / bin_size_samples)
        elif normalize_strategy == "max":
            n_target_bins = int(np.ceil(durations.max() / bin_size_samples))
        elif normalize_strategy == "median":
            n_target_bins = int(np.median(durations) / bin_size_samples)
        else:
            n_target_bins = None  # Variable length
        
        print(f"  Processing '{movie_name}' from {h5_path.name}")
        print(f"    Trials: {len(trial_info['trial_starts'])}")
        print(f"    Target bins: {n_target_bins} ({n_target_bins * bin_size_ms if n_target_bins else 'variable'} ms)")
        
        rows = []
        unit_ids = list(f["units"].keys())
        
        for unit_id in tqdm(unit_ids, desc=f"Units", leave=False):
            unit_path = f"units/{unit_id}"
            
            # Get spike times for this movie
            trials_path = f"{unit_path}/spike_times_sectioned/{movie_name}/trials_spike_times"
            starts_path = f"{unit_path}/spike_times_sectioned/{movie_name}/trials_start_end"
            
            if trials_path not in f or starts_path not in f:
                continue
            
            trials_start_end = f[starts_path][:]
            trials_group = f[trials_path]
            
            # Get unit metadata
            unit_meta = {}
            if include_metadata:
                unit_meta["row"] = f[unit_path].attrs.get("row", -1)
                unit_meta["col"] = f[unit_path].attrs.get("col", -1)
                
                if f"{unit_path}/auto_label/axon_type" in f:
                    axon_type = f[f"{unit_path}/auto_label/axon_type"][()]
                    unit_meta["axon_type"] = axon_type.decode() if isinstance(axon_type, bytes) else str(axon_type)
            
            # Process each trial
            for trial_idx_str in trials_group.keys():
                trial_idx = int(trial_idx_str)
                spike_times = trials_group[trial_idx_str][:]
                
                trial_start = trials_start_end[trial_idx, 0]
                trial_end = trials_start_end[trial_idx, 1]
                trial_duration_s = (trial_end - trial_start) / acq_rate
                
                # Compute firing rate
                fr = spike_times_to_firing_rate(
                    spike_times=spike_times,
                    trial_start=trial_start,
                    trial_end=trial_end,
                    bin_size_ms=bin_size_ms,
                    acquisition_rate=acq_rate,
                    normalize_length=n_target_bins,
                    pad_value=pad_value,
                )
                
                row = {
                    "dataset_id": dataset_id,
                    "unit_id": unit_id,
                    "trial_idx": trial_idx,
                    "firing_rate": fr,  # numpy array
                    "n_spikes": len(spike_times),
                    "n_bins": len(fr),
                    "trial_duration_s": trial_duration_s,
                    **unit_meta,
                }
                rows.append(row)
        
        return pd.DataFrame(rows)


def create_firing_rate_dataframe_multi(
    h5_paths: List[Union[str, Path]],
    movie_name: str,
    bin_size_ms: float = 10.0,
    normalize_strategy: Literal["min", "max", "median", "none"] = "min",
    global_normalize: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Create firing rate DataFrame from multiple HDF5 files.
    
    If global_normalize=True, determines target length from ALL files first,
    ensuring consistent output across recordings.
    """
    h5_paths = [Path(p) for p in h5_paths]
    
    # Determine global target length if needed
    target_bins = None
    if global_normalize and normalize_strategy != "none":
        all_durations = []
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as f:
                trial_info = get_trial_info_for_movie(f, movie_name)
                all_durations.extend(trial_info["trial_durations_samples"])
        
        if all_durations:
            acq_rate = 20000.0  # Default
            bin_size_samples = int(bin_size_ms / 1000.0 * acq_rate)
            durations = np.array(all_durations)
            
            if normalize_strategy == "min":
                target_bins = int(durations.min() / bin_size_samples)
            elif normalize_strategy == "max":
                target_bins = int(np.ceil(durations.max() / bin_size_samples))
            elif normalize_strategy == "median":
                target_bins = int(np.median(durations) / bin_size_samples)
            
            print(f"Global target for '{movie_name}': {target_bins} bins "
                  f"({target_bins * bin_size_ms:.1f} ms) from {len(all_durations)} trials")
    
    # Process each file
    dfs = []
    for h5_path in h5_paths:
        df = create_firing_rate_dataframe(
            h5_path, movie_name, bin_size_ms,
            normalize_strategy="none" if target_bins else normalize_strategy,
            target_bins=target_bins,
            **kwargs,
        )
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Main Demo
# =============================================================================

def main():
    """Demo: Create firing rate DataFrames for all stimuli."""
    print("=" * 80)
    print("Creating Firing Rate DataFrames")
    print("=" * 80)
    
    h5_files = sorted(HDF5_DIR.glob("*.h5"))
    print(f"\nFound {len(h5_files)} HDF5 files")
    
    if not h5_files:
        print("No HDF5 files found!")
        return
    
    # Get all movie names from first file
    with h5py.File(h5_files[0], "r") as f:
        movies = list(f["stimulus/section_time"].keys())
    
    print(f"Found stimulus types: {movies}")
    
    # Process each movie type
    for movie_name in movies:
        print(f"\n{'='*60}")
        print(f"Processing: {movie_name}")
        print("=" * 60)
        
        df = create_firing_rate_dataframe_multi(
            h5_files,
            movie_name=movie_name,
            bin_size_ms=10.0,
            normalize_strategy="min",
            global_normalize=True,
            include_metadata=True,
        )
        
        if df.empty:
            print(f"  No data for {movie_name}")
            continue
        
        # Summary
        print(f"\n  DataFrame shape: {df.shape}")
        print(f"  Unique datasets: {df['dataset_id'].nunique()}")
        print(f"  Unique units: {df['unit_id'].nunique()}")
        print(f"  Trials per unit: {df.groupby('unit_id')['trial_idx'].count().mean():.1f}")
        
        # Check firing rate array shapes
        fr_lengths = df["firing_rate"].apply(len)
        print(f"  Firing rate length: {fr_lengths.min()} - {fr_lengths.max()} bins")
        print(f"    (all equal: {fr_lengths.min() == fr_lengths.max()})")
        
        # Save to Parquet (with nested array support)
        output_path = OUTPUT_DIR / f"firing_rate_{movie_name}.parquet"
        
        # Convert firing_rate arrays to lists for Parquet storage
        df_save = df.copy()
        df_save["firing_rate"] = df_save["firing_rate"].apply(lambda x: x.tolist())
        
        df_save.to_parquet(output_path, index=False)
        print(f"\n  Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")
    
    print("\n" + "=" * 80)
    print("Complete! All firing rate DataFrames saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

