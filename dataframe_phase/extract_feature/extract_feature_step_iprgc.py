"""
Feature extraction from firing rate data.

Computes quality index (QI) for step_up_5s_5i_b0_3x stimulus using
variance ratio method from Legacy code.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


def bessel_lowpass_filter(
    trace: np.ndarray,
    cutoff_freq: float = 10.0,
    order: int = 5,
    sampling_rate: float = 60.0,
) -> np.ndarray:
    """
    Apply Bessel low-pass filter to a trace.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of firing rate data
    cutoff_freq : float
        Cutoff frequency in Hz (default 10 Hz)
    order : int
        Filter order
    sampling_rate : float
        Sampling rate in Hz (default 60 Hz)
        
    Returns
    -------
    np.ndarray
        Filtered trace
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    
    # Ensure cutoff is valid (must be < 1 for normalized frequency)
    if normalized_cutoff >= 1.0:
        raise ValueError(
            f"Cutoff frequency {cutoff_freq} Hz is too high for sampling rate "
            f"{sampling_rate} Hz (Nyquist = {nyquist} Hz)"
        )
    
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False, output='ba')
    filtered_trace = signal.filtfilt(b, a, trace)
    return filtered_trace


def get_quality_index(array: np.ndarray) -> float:
    """
    Calculate quality index as variance ratio (from Legacy code).
    
    QI = Var(mean_across_trials) / Mean(var_within_each_trial)
    
    Parameters
    ----------
    array : np.ndarray
        2D array of shape (n_trials, n_timepoints)
        
    Returns
    -------
    float
        Quality index value
    """
    if array.ndim != 2:
        return np.nan
    
    # Calculate mean and variance
    mean_across_trials = array.mean(axis=0)  # Shape: (n_timepoints,)
    var_within_trials = array.var(axis=1)    # Shape: (n_trials,)
    
    # Check for NaN values
    if np.isnan(mean_across_trials).any() or np.isnan(var_within_trials).any():
        return np.nan
    
    # Check for zero variance to avoid division by zero
    mean_var_within_trials = np.mean(var_within_trials)
    if mean_var_within_trials == 0:
        return np.nan
    
    # Calculate the variance ratio
    variance_ratio = np.var(mean_across_trials) / mean_var_within_trials
    return variance_ratio


def get_quality_index_pearson(array: np.ndarray) -> float:
    """
    Calculate quality index as mean Pearson correlation of trials vs mean.
    
    QI = mean(corr(trial_i, mean_trace)) for all trials
    
    Parameters
    ----------
    array : np.ndarray
        2D array of shape (n_trials, n_timepoints)
        
    Returns
    -------
    float
        Quality index value (mean correlation coefficient)
    """
    if array.ndim != 2 or array.shape[0] < 2:
        return np.nan
    
    # Check for NaN values
    if np.isnan(array).any():
        return np.nan
    
    mean_trace = array.mean(axis=0)
    
    # Check if mean trace has zero variance (constant)
    if np.std(mean_trace) == 0:
        return np.nan
    
    correlations = []
    for trial in array:
        # Check if trial has zero variance
        if np.std(trial) == 0:
            continue
        r, _ = stats.pearsonr(trial, mean_trace)
        if not np.isnan(r):
            correlations.append(r)
    
    if len(correlations) == 0:
        return np.nan
    
    return np.mean(correlations)


def compute_step_up_qi(
    df: pd.DataFrame,
    movie_col: str = "step_up_5s_5i_b0_3x",
    apply_filter: bool = True,
    cutoff_freq: float = 10.0,
    filter_order: int = 5,
    sampling_rate: float = 60.0,
) -> pd.Series:
    """
    Compute quality index for step_up stimulus data.
    
    Each trial is baseline-subtracted (median of first 30 data points) before
    computing the quality index.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with firing rate data (each cell contains array of trials)
    movie_col : str
        Column name for the step_up movie
    apply_filter : bool
        Whether to apply low-pass filter before QI calculation
    cutoff_freq : float
        Low-pass filter cutoff frequency in Hz (default 10 Hz)
    filter_order : int
        Bessel filter order
    sampling_rate : float
        Sampling rate of firing rate data in Hz (default 60 Hz)
        
    Returns
    -------
    pd.Series
        Quality index for each unit
    """
    qi_values = []
    
    for idx in tqdm(df.index, desc="Computing step_up QI"):
        data = df.loc[idx, movie_col]
        
        # Handle missing data
        if data is None or (isinstance(data, float) and np.isnan(data)):
            qi_values.append(np.nan)
            continue
        
        # Stack trials into 2D array (filter out None trials)
        try:
            valid_trials = [np.array(trial) for trial in data if trial is not None]
            if len(valid_trials) == 0:
                qi_values.append(np.nan)
                continue
            trials_array = np.vstack(valid_trials)
        except Exception:
            qi_values.append(np.nan)
            continue
        
        # Apply low-pass filter to each trial
        if apply_filter:
            filtered_trials = np.zeros_like(trials_array)
            for i in range(trials_array.shape[0]):
                try:
                    filtered_trials[i] = bessel_lowpass_filter(
                        trials_array[i],
                        cutoff_freq=cutoff_freq,
                        order=filter_order,
                        sampling_rate=sampling_rate,
                    )
                except Exception:
                    filtered_trials[i] = trials_array[i]
            trials_array = filtered_trials
        
        # Baseline subtraction: subtract median of first 30 data points from each trial
        baseline_n_samples = 30
        for i in range(trials_array.shape[0]):
            baseline_median = np.median(trials_array[i, :baseline_n_samples])
            trials_array[i] = trials_array[i] - baseline_median
        
        # Compute quality index
        qi = get_quality_index(trials_array)
        qi_values.append(qi)
    
    return pd.Series(qi_values, index=df.index, name="step_up_QI")


def compute_iprgc_qi(
    df: pd.DataFrame,
    movie_col: str = "iprgc_test",
    filter_order: int = 5,
    sampling_rate: float = 60.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute ipRGC quality indices using Pearson correlation.
    
    Each trial is baseline-subtracted (mean of last 1 second) before
    computing the quality index.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with firing rate data (each cell contains array of trials)
    movie_col : str
        Column name for the iprgc_test movie
    filter_order : int
        Bessel filter order
    sampling_rate : float
        Sampling rate of firing rate data in Hz (default 60 Hz)
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        iprgc_2hz_QI: 2 Hz lowpass, 2s to end, Pearson correlation
        iprgc_20hz_QI: 20 Hz lowpass, 2s to 10s, Pearson correlation
    """
    qi_2hz_values = []
    qi_20hz_values = []
    
    # Time range: start at 2 seconds, end at 10 seconds for 20Hz QI
    samples_baseline = int(1 * sampling_rate)  # 1 second = 60 samples at 60 Hz for baseline
    samples_start = int(2 * sampling_rate)     # 2 seconds = 120 samples at 60 Hz
    samples_10s = int(10 * sampling_rate)      # 10 seconds = 600 samples at 60 Hz
    
    for idx in tqdm(df.index, desc="Computing ipRGC QI"):
        data = df.loc[idx, movie_col]
        
        # Handle missing data
        if data is None or (isinstance(data, float) and np.isnan(data)):
            qi_2hz_values.append(np.nan)
            qi_20hz_values.append(np.nan)
            continue
        
        # Stack trials into 2D array (filter out None trials)
        try:
            valid_trials = [np.array(trial) for trial in data if trial is not None]
            if len(valid_trials) == 0:
                qi_2hz_values.append(np.nan)
                qi_20hz_values.append(np.nan)
                continue
            trials_array = np.vstack(valid_trials)
        except Exception:
            qi_2hz_values.append(np.nan)
            qi_20hz_values.append(np.nan)
            continue
        
        n_trials, n_timepoints = trials_array.shape
        
        # Baseline subtraction: zero by mean of last second (60 samples at 60 Hz)
        for i in range(n_trials):
            baseline_mean = np.mean(trials_array[i, -samples_baseline:])
            trials_array[i] = trials_array[i] - baseline_mean
        
        # --- 2 Hz lowpass filter, from 2s to end ---
        try:
            filtered_2hz = np.zeros_like(trials_array)
            for i in range(n_trials):
                filtered_2hz[i] = bessel_lowpass_filter(
                    trials_array[i],
                    cutoff_freq=2.0,
                    order=filter_order,
                    sampling_rate=sampling_rate,
                )
            # Trim from 2 seconds to end
            trimmed_2hz = filtered_2hz[:, samples_start:]
            
            # Check if any trial has mean firing rate < 1 Hz (use raw trimmed data)
            raw_trimmed_2hz = trials_array[:, samples_start:]
            has_low_firing_trial = any(np.mean(raw_trimmed_2hz[i]) < 1.0 for i in range(n_trials))
            
            # If any trial is constant, all near zero, or low firing rate, set QI to 0
            has_constant_trial = any(np.std(trimmed_2hz[i]) == 0 for i in range(n_trials))
            all_near_zero = np.max(np.abs(trimmed_2hz)) < 1.0
            if has_constant_trial or all_near_zero or has_low_firing_trial:
                qi_2hz = 0.0
            else:
                qi_2hz = get_quality_index_pearson(trimmed_2hz)
        except Exception:
            qi_2hz = np.nan
        qi_2hz_values.append(qi_2hz)
        
        # --- 20 Hz lowpass filter, from 2s to 10s ---
        try:
            filtered_20hz = np.zeros_like(trials_array)
            for i in range(n_trials):
                filtered_20hz[i] = bessel_lowpass_filter(
                    trials_array[i],
                    cutoff_freq=20.0,
                    order=filter_order,
                    sampling_rate=sampling_rate,
                )
            # Trim from 2 seconds to 10 seconds
            trimmed_20hz = filtered_20hz[:, samples_start:samples_10s]
            
            # Check if any trial has mean firing rate < 1 Hz (use raw trimmed data)
            raw_trimmed_20hz = trials_array[:, samples_start:samples_10s]
            has_low_firing_trial = any(np.mean(raw_trimmed_20hz[i]) < 1.0 for i in range(n_trials))
            
            # If any trial is constant, all near zero, or low firing rate, set QI to 0
            has_constant_trial = any(np.std(trimmed_20hz[i]) == 0 for i in range(n_trials))
            all_near_zero = np.max(np.abs(trimmed_20hz)) < 1.0
            if has_constant_trial or all_near_zero or has_low_firing_trial:
                qi_20hz = 0.0
            else:
                qi_20hz = get_quality_index_pearson(trimmed_20hz)
        except Exception:
            qi_20hz = np.nan
        qi_20hz_values.append(qi_20hz)
    
    return (
        pd.Series(qi_2hz_values, index=df.index, name="iprgc_2hz_QI"),
        pd.Series(qi_20hz_values, index=df.index, name="iprgc_20hz_QI"),
    )


def add_good_cell_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add good_count and good_rgc_count columns to the DataFrame.
    
    Counts are computed per recording file based on:
    - good_count: cells with step_up_QI > 0.5
    - good_rgc_count: cells with step_up_QI > 0.5 AND axon_type == 'rgc'
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index format "filename_unit_id"
        
    Returns
    -------
    pd.DataFrame
        DataFrame with good_count and good_rgc_count columns added
    """
    # Extract filename from index (format: "filename_unit_id")
    df['_filename'] = df.index.str.rsplit('_unit_', n=1).str[0]
    
    # Count good cells per file (step_up_QI > 0.5)
    good_mask = df['step_up_QI'] > 0.5
    good_counts = df[good_mask].groupby('_filename').size()
    
    # Count good RGC cells per file
    good_rgc_mask = good_mask & (df['axon_type'] == 'rgc')
    good_rgc_counts = df[good_rgc_mask].groupby('_filename').size()
    
    # Map counts back to each row
    df['good_count'] = df['_filename'].map(good_counts).fillna(0).astype(int)
    df['good_rgc_count'] = df['_filename'].map(good_rgc_counts).fillna(0).astype(int)
    
    # Remove helper column
    df = df.drop(columns=['_filename'])
    
    return df


def main():
    """Main function to extract features and save to parquet."""
    # Paths
    #input_path = Path("dataframe_phase/load_feature/firing_rate_with_dsgc_features_typed20251230.parquet")
    #output_path = Path("dataframe_phase/extract_feature/firing_rate_with_features20251230.parquet")
    
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet")
    output_path = Path("dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet")
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Compute quality index for step_up stimulus
    print("\nComputing step_up quality index...")
    print("  - Applying 10 Hz low-pass Bessel filter (order 5)")
    print("  - Sampling rate: 60 Hz")
    
    step_up_qi = compute_step_up_qi(
        df,
        movie_col="step_up_5s_5i_b0_3x",
        apply_filter=True,
        cutoff_freq=10.0,  # 10 Hz low-pass
        filter_order=5,
        sampling_rate=60.0,
    )
    
    # Add QI to dataframe
    df["step_up_QI"] = step_up_qi
    
    # Print summary statistics
    print(f"\nstep_up Quality Index Statistics:")
    print(f"  Valid values: {step_up_qi.notna().sum()} / {len(step_up_qi)}")
    print(f"  Mean: {step_up_qi.mean():.4f}")
    print(f"  Median: {step_up_qi.median():.4f}")
    print(f"  Std: {step_up_qi.std():.4f}")
    print(f"  Min: {step_up_qi.min():.4f}")
    print(f"  Max: {step_up_qi.max():.4f}")
    
    # Compute ipRGC quality indices
    print("\nComputing ipRGC quality indices...")
    print("  - iprgc_2hz_QI: 2 Hz low-pass Bessel filter, 2s to end, Pearson correlation")
    print("  - iprgc_20hz_QI: 20 Hz low-pass Bessel filter, 2s to 10s, Pearson correlation")
    print("  - Sampling rate: 60 Hz")
    
    iprgc_2hz_qi, iprgc_20hz_qi = compute_iprgc_qi(
        df,
        movie_col="iprgc_test",
        filter_order=5,
        sampling_rate=60.0,
    )
    
    # Add ipRGC QI columns to dataframe
    df["iprgc_2hz_QI"] = iprgc_2hz_qi
    df["iprgc_20hz_QI"] = iprgc_20hz_qi
    
    # Print ipRGC QI statistics
    print(f"\niprgc_2hz_QI Statistics:")
    print(f"  Valid values: {iprgc_2hz_qi.notna().sum()} / {len(iprgc_2hz_qi)}")
    print(f"  Mean: {iprgc_2hz_qi.mean():.4f}")
    print(f"  Median: {iprgc_2hz_qi.median():.4f}")
    print(f"  Std: {iprgc_2hz_qi.std():.4f}")
    print(f"  Min: {iprgc_2hz_qi.min():.4f}")
    print(f"  Max: {iprgc_2hz_qi.max():.4f}")
    
    print(f"\niprgc_20hz_QI Statistics:")
    print(f"  Valid values: {iprgc_20hz_qi.notna().sum()} / {len(iprgc_20hz_qi)}")
    print(f"  Mean: {iprgc_20hz_qi.mean():.4f}")
    print(f"  Median: {iprgc_20hz_qi.median():.4f}")
    print(f"  Std: {iprgc_20hz_qi.std():.4f}")
    print(f"  Min: {iprgc_20hz_qi.min():.4f}")
    print(f"  Max: {iprgc_20hz_qi.max():.4f}")
    
    # Add good cell counts per recording file
    print("\nComputing good cell counts per recording...")
    df = add_good_cell_counts(df)
    print(f"  Added good_count and good_rgc_count columns")
    
    # Print good count statistics
    n_files = df.index.str.rsplit('_unit_', n=1).str[0].nunique()
    print(f"  Number of recordings: {n_files}")
    print(f"  good_count range: {df['good_count'].min()} - {df['good_count'].max()}")
    print(f"  good_rgc_count range: {df['good_rgc_count'].min()} - {df['good_rgc_count'].max()}")
    
    # Save to parquet
    print(f"\nSaving to: {output_path}")
    df.to_parquet(output_path)
    print("Done!")
    
    return df


if __name__ == "__main__":
    main()

