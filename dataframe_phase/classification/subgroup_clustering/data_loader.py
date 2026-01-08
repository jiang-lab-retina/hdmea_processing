"""
Data Loading and Preprocessing for Subgroup Clustering.

Loads the parquet file, classifies RGCs into subgroups, and prepares
concatenated traces for each subgroup.

Preprocessing:
- All traces are low-pass filtered at 10 Hz (Bessel filter)
- All traces are resampled from 60 Hz to 10 Hz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import signal
from tqdm import tqdm

# =============================================================================
# Preprocessing Constants
# =============================================================================
ORIGINAL_SAMPLE_RATE = 60.0  # Hz (original trace sampling rate)
TARGET_SAMPLE_RATE = 10.0    # Hz (target after resampling)
LOWPASS_CUTOFF = 10.0        # Hz (low-pass filter cutoff)
FILTER_ORDER = 5             # Bessel filter order


def preprocess_trace(trace: np.ndarray) -> np.ndarray:
    """
    Apply 10 Hz low-pass filter and resample to 10 Hz.
    
    Parameters
    ----------
    trace : np.ndarray
        Input trace at 60 Hz sampling rate
        
    Returns
    -------
    np.ndarray
        Filtered and resampled trace at 10 Hz
    """
    if len(trace) < FILTER_ORDER + 1:
        # Trace too short for filtering, just downsample
        downsample_factor = int(ORIGINAL_SAMPLE_RATE / TARGET_SAMPLE_RATE)
        return trace[::downsample_factor]
    
    # Design Bessel low-pass filter
    nyquist = 0.5 * ORIGINAL_SAMPLE_RATE
    normalized_cutoff = LOWPASS_CUTOFF / nyquist
    b, a = signal.bessel(FILTER_ORDER, normalized_cutoff, btype='low', analog=False)
    
    # Apply zero-phase filtering
    filtered = signal.filtfilt(b, a, trace)
    
    # Resample: 60 Hz -> 10 Hz (take every 6th sample)
    downsample_factor = int(ORIGINAL_SAMPLE_RATE / TARGET_SAMPLE_RATE)
    resampled = filtered[::downsample_factor]
    
    return resampled


from .config import (
    INPUT_PARQUET,
    MOVIE_COLUMNS,
    SUBGROUP_MOVIE_COLUMNS,
    DS_P_THRESHOLD,
    OS_P_THRESHOLD,
    IPRGC_QI_THRESHOLD,
    SUBGROUPS,
)


def load_and_classify_rgc(parquet_path: Path = None) -> pd.DataFrame:
    """
    Load parquet, filter for RGC, and classify into subgroups.
    
    Parameters
    ----------
    parquet_path : Path, optional
        Path to parquet file. Uses default if not provided.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with RGC units and 'rgc_subtype' column
    """
    if parquet_path is None:
        parquet_path = INPUT_PARQUET
    
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Total units: {len(df)}")
    
    # Filter for RGC only
    rgc_mask = df["axon_type"].str.lower() == "rgc"
    df_rgc = df[rgc_mask].copy()
    print(f"  RGC units: {len(df_rgc)}")
    
    # Classify subgroups with priority: ipRGC > DSGC > OSGC > Other
    df_rgc["rgc_subtype"] = "Other"
    
    osgc_mask = df_rgc["os_p_value"] < OS_P_THRESHOLD
    df_rgc.loc[osgc_mask, "rgc_subtype"] = "OSGC"
    
    dsgc_mask = df_rgc["ds_p_value"] < DS_P_THRESHOLD
    df_rgc.loc[dsgc_mask, "rgc_subtype"] = "DSGC"
    
    iprgc_mask = df_rgc["iprgc_2hz_QI"] > IPRGC_QI_THRESHOLD
    df_rgc.loc[iprgc_mask, "rgc_subtype"] = "ipRGC"
    
    # Print distribution
    print("\n  Subtype distribution:")
    for subtype in SUBGROUPS:
        count = (df_rgc["rgc_subtype"] == subtype).sum()
        pct = 100 * count / len(df_rgc)
        print(f"    {subtype}: {count} ({pct:.1f}%)")
    
    return df_rgc


def compute_mean_trace(trial_data) -> np.ndarray:
    """
    Compute mean trace from trial data, then apply preprocessing.
    
    Preprocessing includes:
    - 10 Hz Bessel low-pass filter
    - Resampling from 60 Hz to 10 Hz
    """
    if trial_data is None:
        return np.array([])
    
    if isinstance(trial_data, float) and np.isnan(trial_data):
        return np.array([])
    
    try:
        valid_trials = [np.array(trial) for trial in trial_data if trial is not None]
        if len(valid_trials) == 0:
            return np.array([])
        trials_array = np.vstack(valid_trials)
        mean_trace = np.mean(trials_array, axis=0)
        
        # Apply low-pass filter and resample to 10 Hz
        return preprocess_trace(mean_trace)
    except Exception:
        return np.array([])


def prepare_traces_for_subgroup(
    df: pd.DataFrame,
    subgroup: str,
    movie_columns: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare concatenated traces for a specific subgroup.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with rgc_subtype column
    subgroup : str
        Subgroup name ('ipRGC', 'DSGC', 'OSGC', 'Other')
    movie_columns : list, optional
        List of movie column names. If None, uses subgroup-specific columns.
        
    Returns
    -------
    tuple
        (trace_array, valid_indices)
    """
    if movie_columns is None:
        # Use subgroup-specific movie columns
        movie_columns = SUBGROUP_MOVIE_COLUMNS.get(subgroup, MOVIE_COLUMNS)
    
    # Filter for subgroup
    df_sub = df[df["rgc_subtype"] == subgroup].copy()
    print(f"\n  Processing {subgroup}: {len(df_sub)} units")
    print(f"    Using {len(movie_columns)} movie columns" + 
          (" (with iprgc_test)" if "iprgc_test" in movie_columns else " (without iprgc_test)"))
    
    # Check columns
    existing_cols = [col for col in movie_columns if col in df_sub.columns]
    
    # Compute concatenated traces
    traces = []
    valid_indices = []
    
    for idx in tqdm(df_sub.index, desc=f"    Concatenating traces", leave=False):
        row = df_sub.loc[idx]
        traces_to_concat = []
        valid = True
        
        for col in existing_cols:
            mean_trace = compute_mean_trace(row.get(col))
            if len(mean_trace) == 0:
                valid = False
                break
            traces_to_concat.append(mean_trace)
        
        if valid and len(traces_to_concat) > 0:
            concatenated = np.concatenate(traces_to_concat)
            traces.append(concatenated)
            valid_indices.append(idx)
    
    if len(traces) == 0:
        print(f"    WARNING: No valid traces for {subgroup}")
        return np.array([]), []
    
    # Ensure consistent length
    trace_lengths = [len(t) for t in traces]
    min_len = min(trace_lengths)
    traces = [t[:min_len] for t in traces]
    
    X = np.vstack(traces)
    print(f"    Valid traces: {len(valid_indices)}, shape: {X.shape}")
    
    return X, valid_indices


def normalize_traces(X: np.ndarray) -> np.ndarray:
    """Z-score normalize each trace."""
    X = np.nan_to_num(X, nan=0.0)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    return (X - means) / stds


def load_subgroup_data(
    parquet_path: Path = None,
    normalize: bool = True,
) -> Dict[str, Tuple[np.ndarray, List[str], pd.DataFrame]]:
    """
    Load and prepare data for all subgroups.
    
    Parameters
    ----------
    parquet_path : Path, optional
        Path to parquet file
    normalize : bool
        Whether to z-score normalize traces
        
    Returns
    -------
    dict
        Dictionary mapping subgroup name to (traces, indices, dataframe_subset)
    """
    print("=" * 70)
    print("Loading Subgroup Data")
    print("=" * 70)
    
    # Load and classify
    df = load_and_classify_rgc(parquet_path)
    
    # Prepare data for each subgroup
    subgroup_data = {}
    
    for subgroup in SUBGROUPS:
        X, valid_indices = prepare_traces_for_subgroup(df, subgroup)
        
        if len(X) == 0:
            continue
        
        if normalize:
            X = normalize_traces(X)
        
        df_subset = df.loc[valid_indices].copy()
        subgroup_data[subgroup] = (X, valid_indices, df_subset)
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for subgroup, (X, indices, _) in subgroup_data.items():
        print(f"  {subgroup}: {X.shape[0]} units, {X.shape[1]} features")
    
    return subgroup_data


if __name__ == "__main__":
    # Test data loading
    data = load_subgroup_data()

