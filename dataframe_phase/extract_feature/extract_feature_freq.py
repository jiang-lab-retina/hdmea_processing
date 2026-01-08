"""
Frequency step response feature extraction.

Extracts sine wave fit features from frequency step stimulus traces.
Adapted from legacy code in cluster_analysis_2.py (lines 1083-1146) for 60 Hz data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple
from scipy import signal
from scipy.optimize import curve_fit

from step_config import (
    FREQ_PRE_MARGIN,
    FREQ_GAP_DURATION,
    FREQ_STEP_DURATION,
    FREQ_STEP_FREQUENCIES,
    FREQ_STEP_BOUNDS,
    FREQ_TRACE_COLUMN,
    FREQ_AMP_GUESS,
    FREQ_AMP_UPPER_LIM,
    FREQ_OFFSET_UPPER_LIM,
    FREQ_OFFSET_LOWER_LIM,
    FREQ_MAXFEV,
    FREQ_R_SQUARED_THRESHOLD,
    FREQ_FIT_SKIP_FRAMES,
    SAMPLING_RATE,
    LOWPASS_CUTOFF_FREQ,
    LOWPASS_FILTER_ORDER,
)


# =============================================================================
# Helper Functions
# =============================================================================

def bessel_lowpass_filter(
    trace: np.ndarray,
    cutoff_freq: float = LOWPASS_CUTOFF_FREQ,
    order: int = LOWPASS_FILTER_ORDER,
    sampling_rate: float = SAMPLING_RATE,
) -> np.ndarray:
    """
    Apply Bessel low-pass filter to a trace.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of values to filter
    cutoff_freq : float
        Cutoff frequency in Hz
    order : int
        Filter order
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    np.ndarray
        Filtered trace
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        return trace  # Can't filter, return original
    
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False, output='ba')
    filtered_trace = signal.filtfilt(b, a, trace)
    return filtered_trace


def fit_sine_wave_fixed_freq(
    x: np.ndarray,
    y: np.ndarray,
    freq_fixed: float,
    amp_guess: float = FREQ_AMP_GUESS,
    phase_guess: float = 0.0,
    offset_guess: float = 0.0,
    amp_upper_lim: float = FREQ_AMP_UPPER_LIM,
    offset_upper_lim: float = FREQ_OFFSET_UPPER_LIM,
    offset_lower_lim: float = FREQ_OFFSET_LOWER_LIM,
    maxfev: int = FREQ_MAXFEV,
) -> Dict[str, float]:
    """
    Fit a sine wave with fixed frequency to data.
    
    Model: y = amplitude * sin(2 * pi * freq_fixed * x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Time array in seconds
    y : np.ndarray
        Data values to fit
    freq_fixed : float
        Fixed frequency in Hz
    amp_guess : float
        Initial amplitude guess
    phase_guess : float
        Initial phase guess
    offset_guess : float
        Initial offset guess
    amp_upper_lim : float
        Upper bound for amplitude
    offset_upper_lim : float
        Upper bound for offset
    offset_lower_lim : float
        Lower bound for offset
    maxfev : int
        Maximum function evaluations for curve_fit
        
    Returns
    -------
    dict
        Dictionary with keys: 'amp', 'freq', 'phase', 'r_squared', 'offset'
    """
    def sine_wave_fixed_freq(x, amplitude, phase, offset):
        return amplitude * np.sin(2 * np.pi * freq_fixed * x + phase) + offset
    
    # Handle length mismatch
    if x.shape[0] != y.shape[0]:
        min_length = min(x.shape[0], y.shape[0])
        x = x[:min_length]
        y = y[:min_length]
    
    initial_guess = [amp_guess, phase_guess, offset_guess]
    bounds = ([0, -np.pi, offset_lower_lim], [amp_upper_lim, np.pi, offset_upper_lim])
    
    try:
        params, _ = curve_fit(
            sine_wave_fixed_freq, x, y, 
            p0=initial_guess, 
            bounds=bounds, 
            maxfev=maxfev
        )
        amplitude, phase, offset = params
        
        # Calculate r-squared value
        residuals = y - sine_wave_fixed_freq(x, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        if ss_tot < 1e-10:
            # Data is nearly constant, R^2 is undefined - treat as poor fit
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
            # Clamp to valid range in case of numerical issues
            if not np.isfinite(r_squared):
                r_squared = 0.0
        
        return {
            "amp": amplitude,
            "freq": freq_fixed,
            "phase": phase,
            "r_squared": r_squared,
            "offset": offset,
        }
    except Exception:
        # Return default values if fitting fails
        return {
            "amp": 0.0,
            "freq": freq_fixed,
            "phase": np.nan,
            "r_squared": 0.0,
            "offset": 0.0,
        }


def compute_mean_trace(
    trials_data,
    apply_filter: bool = True,
    cutoff_freq: float = LOWPASS_CUTOFF_FREQ,
    filter_order: int = LOWPASS_FILTER_ORDER,
    sampling_rate: float = SAMPLING_RATE,
) -> np.ndarray | None:
    """
    Compute mean trace from multiple trials with optional low-pass filtering.
    
    Parameters
    ----------
    trials_data : list or array-like
        List of trial arrays
    apply_filter : bool
        Whether to apply low-pass Bessel filter to each trial before averaging
    cutoff_freq : float
        Low-pass filter cutoff frequency in Hz
    filter_order : int
        Bessel filter order
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    np.ndarray or None
        Mean trace (filtered if apply_filter=True), or None if data is invalid
    """
    if trials_data is None:
        return None
    
    if isinstance(trials_data, float) and np.isnan(trials_data):
        return None
    
    try:
        valid_trials = [np.array(trial) for trial in trials_data if trial is not None]
        if len(valid_trials) == 0:
            return None
        trials_array = np.vstack(valid_trials)
        
        # Apply low-pass filter to each trial
        if apply_filter:
            for i in range(trials_array.shape[0]):
                try:
                    trials_array[i] = bessel_lowpass_filter(
                        trials_array[i],
                        cutoff_freq=cutoff_freq,
                        order=filter_order,
                        sampling_rate=sampling_rate,
                    )
                except Exception:
                    pass  # Keep original if filtering fails
        
        return trials_array.mean(axis=0)
    except Exception:
        return None


def freq_to_column_str(freq: float) -> str:
    """
    Convert frequency value to column name string.
    
    Examples:
        0.5 -> "05"
        1 -> "1"
        10 -> "10"
    """
    return str(freq).replace(".", "")


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_freq_step_features_from_trace(
    trace: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
    skip_frames: int = FREQ_FIT_SKIP_FRAMES,
) -> Dict[str, float]:
    """
    Extract all frequency step response features from a single mean trace.
    
    Fits a sine wave to each frequency segment and extracts amplitude, phase,
    r_squared, and offset.
    
    For all frequencies except 0.5 Hz, skips the first `skip_frames` frames
    to avoid transient response at the start of each frequency step.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of firing rate values
    sampling_rate : float
        Sampling rate in Hz
    skip_frames : int
        Number of frames to skip at start for fitting (except 0.5 Hz)
        
    Returns
    -------
    dict
        Dictionary of feature name -> value
    """
    features = {}
    
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        start, end = FREQ_STEP_BOUNDS[freq]
        
        # Extract segment for this frequency
        # Skip first frames for all frequencies except 0.5 Hz (to avoid transient)
        if freq != 0.5:
            fit_start = start + skip_frames
        else:
            fit_start = start
        
        segment = trace[fit_start:end]
        
        # Create time array in seconds
        n_samples = len(segment)
        time_array = np.arange(n_samples) / sampling_rate
        
        # Fit sine wave
        fit_result = fit_sine_wave_fixed_freq(
            x=time_array,
            y=segment,
            freq_fixed=freq,
        )
        
        # Store raw fit results
        amp = fit_result["amp"]
        phase = fit_result["phase"]
        r_squared = fit_result["r_squared"]
        offset = fit_result["offset"]
        
        # Calculate std of the fitting region
        segment_std = np.std(segment)
        
        # Apply R^2 threshold: if R^2 < 0.1, set amp to 0 and phase to NaN
        if r_squared < FREQ_R_SQUARED_THRESHOLD:
            amp = 0.0
            phase = np.nan
        
        # Store features
        features[f"freq_step_{freq_str}hz_amp"] = amp
        features[f"freq_step_{freq_str}hz_phase"] = phase
        features[f"freq_step_{freq_str}hz_r_squared"] = r_squared
        features[f"freq_step_{freq_str}hz_offset"] = offset
        features[f"freq_step_{freq_str}hz_std"] = segment_std
    
    return features


def extract_freq_step_features(
    df: pd.DataFrame,
    trace_column: str = FREQ_TRACE_COLUMN,
) -> pd.DataFrame:
    """
    Extract frequency step response features from all units in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data
    trace_column : str
        Column name containing the frequency step response trials
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns added
    """
    # Build list of feature column names
    feature_names = []
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        feature_names.extend([
            f"freq_step_{freq_str}hz_amp",
            f"freq_step_{freq_str}hz_phase",
            f"freq_step_{freq_str}hz_r_squared",
            f"freq_step_{freq_str}hz_offset",
            f"freq_step_{freq_str}hz_std",
        ])
    
    # Initialize feature columns with NaN
    for name in feature_names:
        df[name] = np.nan
    
    # Extract features for each unit
    valid_count = 0
    for idx in tqdm(df.index, desc="Extracting freq step features"):
        trials_data = df.loc[idx, trace_column]
        
        # Compute mean trace (no filtering applied)
        mean_trace = compute_mean_trace(trials_data, apply_filter=False)
        if mean_trace is None:
            continue
        
        # Check if trace is long enough
        max_end = max(end for _, end in FREQ_STEP_BOUNDS.values())
        if len(mean_trace) < max_end:
            continue
        
        # Extract features
        try:
            features = extract_freq_step_features_from_trace(mean_trace)
            
            # Assign features to DataFrame
            for name, value in features.items():
                df.loc[idx, name] = value
            
            valid_count += 1
        except Exception:
            # Skip units with errors
            continue
    
    print(f"Extracted features for {valid_count} / {len(df)} units")
    
    return df


# =============================================================================
# Sectioned Trace Extraction
# =============================================================================

# Mapping of frequency to column name suffix
FREQ_SECTION_COLUMNS = {
    0.5: "freq_section_0p5hz",
    1: "freq_section_1hz",
    2: "freq_section_2hz",
    4: "freq_section_4hz",
    10: "freq_section_10hz",
}


def extract_freq_sectioned_traces(
    df: pd.DataFrame,
    trace_column: str = FREQ_TRACE_COLUMN,
) -> pd.DataFrame:
    """
    Extract sectioned frequency traces as separate columns.
    
    Each frequency section is extracted from the mean trace and stored as a
    separate column containing the trace segment for that frequency period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with frequency step trace data
    trace_column : str
        Column name containing trial data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 5 new columns for sectioned traces:
        - freq_section_0p5hz: frames 30-270
        - freq_section_1hz: frames 330-570
        - freq_section_2hz: frames 630-870
        - freq_section_4hz: frames 930-1170
        - freq_section_10hz: frames 1230-1470
    """
    print(f"\nExtracting sectioned frequency traces from '{trace_column}'...")
    print("Frame ranges:")
    for freq, (start, end) in FREQ_STEP_BOUNDS.items():
        col_name = FREQ_SECTION_COLUMNS[freq]
        print(f"  {freq} Hz: [{start}:{end}] -> {col_name}")
    
    # Initialize columns with None
    for col_name in FREQ_SECTION_COLUMNS.values():
        df[col_name] = None
    
    valid_count = 0
    max_end = max(end for _, end in FREQ_STEP_BOUNDS.values())
    
    for idx in tqdm(df.index, desc="Extracting freq sections"):
        trials_data = df.loc[idx, trace_column]
        
        # Compute mean trace
        mean_trace = compute_mean_trace(trials_data, apply_filter=False)
        if mean_trace is None:
            continue
        
        # Check if trace is long enough
        if len(mean_trace) < max_end:
            continue
        
        # Extract each frequency section
        try:
            for freq, (start, end) in FREQ_STEP_BOUNDS.items():
                col_name = FREQ_SECTION_COLUMNS[freq]
                section = mean_trace[start:end].copy()
                df.at[idx, col_name] = section
            valid_count += 1
        except Exception:
            continue
    
    print(f"Extracted sectioned traces for {valid_count} / {len(df)} units")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to extract frequency step features and save to parquet."""
    # Paths
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step_gb.parquet")
    output_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step_gb_freq.parquet")
    
    print("=" * 60)
    print("Frequency Step Response Feature Extraction")
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  FREQ_PRE_MARGIN: {FREQ_PRE_MARGIN}")
    print(f"  FREQ_GAP_DURATION: {FREQ_GAP_DURATION}")
    print(f"  FREQ_STEP_DURATION: {FREQ_STEP_DURATION}")
    print(f"  Frequencies: {FREQ_STEP_FREQUENCIES}")
    print("\nFrame indices per frequency:")
    for freq, (start, end) in FREQ_STEP_BOUNDS.items():
        print(f"  {freq} Hz: [{start}:{end}]")
    print(f"\nSine fit parameters:")
    print(f"  amp_guess: {FREQ_AMP_GUESS}")
    print(f"  amp_upper_lim: {FREQ_AMP_UPPER_LIM}")
    print(f"  offset_bounds: [{FREQ_OFFSET_LOWER_LIM}, {FREQ_OFFSET_UPPER_LIM}]")
    print(f"  maxfev: {FREQ_MAXFEV}")
    print(f"  R^2 threshold: {FREQ_R_SQUARED_THRESHOLD}")
    print(f"  skip_frames (except 0.5Hz): {FREQ_FIT_SKIP_FRAMES}")
    print(f"\n  Trace column: {FREQ_TRACE_COLUMN}")
    print(f"  Low-pass filter: DISABLED (apply_filter=False)")
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Extract features
    print("\nExtracting frequency step response features...")
    df = extract_freq_step_features(df, trace_column=FREQ_TRACE_COLUMN)
    
    # Print feature statistics
    print("\nFeature statistics:")
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        amp_col = f"freq_step_{freq_str}hz_amp"
        r2_col = f"freq_step_{freq_str}hz_r_squared"
        
        if amp_col in df.columns:
            amp_valid = df[amp_col].dropna()
            r2_valid = df[r2_col].dropna()
            low_r2_count = (r2_valid < FREQ_R_SQUARED_THRESHOLD).sum()
            print(f"\n  {freq} Hz:")
            print(f"    amp: mean={amp_valid.mean():.2f}, std={amp_valid.std():.2f}, "
                  f"min={amp_valid.min():.2f}, max={amp_valid.max():.2f}")
            print(f"    r_squared: mean={r2_valid.mean():.3f}, "
                  f"low R^2 (<{FREQ_R_SQUARED_THRESHOLD}): {low_r2_count} ({100*low_r2_count/len(r2_valid):.1f}%)")
    
    # Save output
    print(f"\nSaving to: {output_path}")
    df.to_parquet(output_path)
    print(f"Saved DataFrame shape: {df.shape}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

