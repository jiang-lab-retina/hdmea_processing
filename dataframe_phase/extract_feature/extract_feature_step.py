"""
Step response feature extraction.

Extracts ON/OFF response features from step stimulus traces.
Adapted from legacy code in cluster_analysis_2.py (lines 839-915) for 60 Hz data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from scipy import signal
from scipy.signal import find_peaks

from step_config import (
    PRE_MARGIN,
    TRANSIENT_START,
    TRANSIENT_END,
    SUSTAINED_START,
    SUSTAINED_END,
    OFF_TRANSIENT_START,
    OFF_TRANSIENT_END,
    OFF_SUSTAINED_START,
    OFF_SUSTAINED_END,
    VALID_MAX_FIRING_RATE,
    STEP_TRACE_COLUMN,
    SAMPLING_RATE,
    LOWPASS_CUTOFF_FREQ,
    LOWPASS_FILTER_ORDER,
    PROMINENCE_STD_THRESHOLD,
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


def find_extreme_peak(
    data: np.ndarray,
    baseline_value: float,
    baseline_std: float,
    prominence_std_threshold: float = PROMINENCE_STD_THRESHOLD,
) -> Dict[str, Any]:
    """
    Find the most extreme peak (positive or negative) in data using scipy peak detection.
    
    1. Find most prominent positive peak (local maxima)
    2. Find most prominent negative peak (local minima)
    3. Compare absolute values relative to baseline
    4. Return the larger one with sign preserved
    
    If no peaks found: value = NaN, location = NaN
    
    Parameters
    ----------
    data : np.ndarray
        1D array of values to search
    baseline_value : float
        Reference baseline value for computing peak amplitude
    baseline_std : float
        Standard deviation of baseline, used to compute adaptive prominence threshold
    prominence_std_threshold : float
        Prominence threshold as multiple of baseline_std.
        prominence = prominence_std_threshold * baseline_std
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'value': peak value relative to baseline (with sign)
        - 'location': index of the peak within the window (NaN if no peak)
        - 'is_peak': True if a real peak was found, False otherwise
    """
    data = np.array(data)
    
    # Calculate adaptive prominence threshold based on baseline noise
    prominence = prominence_std_threshold * baseline_std
    
    # Find positive peaks (local maxima)
    pos_peaks, pos_props = find_peaks(data, prominence=prominence)
    
    # Find negative peaks (local minima) by inverting
    neg_peaks, neg_props = find_peaks(-data, prominence=prominence)
    
    pos_value, pos_loc = None, None
    neg_value, neg_loc = None, None
    
    # Get best positive peak by prominence
    if len(pos_peaks) > 0:
        best_pos_idx = np.argmax(pos_props["prominences"])
        pos_loc = pos_peaks[best_pos_idx]
        pos_value = data[pos_loc] - baseline_value
    
    # Get best negative peak by prominence
    if len(neg_peaks) > 0:
        best_neg_idx = np.argmax(neg_props["prominences"])
        neg_loc = neg_peaks[best_neg_idx]
        neg_value = data[neg_loc] - baseline_value  # Will be negative
    
    # Compare absolute values and select extreme
    if pos_value is None and neg_value is None:
        # No peaks found - return 0 for value, NaN for location
        return {
            "value": 0.0,
            "location": np.nan,
            "is_peak": False,
        }
    elif pos_value is None:
        return {"value": neg_value, "location": neg_loc, "is_peak": True}
    elif neg_value is None:
        return {"value": pos_value, "location": pos_loc, "is_peak": True}
    else:
        # Both exist - compare absolute values
        if abs(pos_value) >= abs(neg_value):
            return {"value": pos_value, "location": pos_loc, "is_peak": True}
        else:
            return {"value": neg_value, "location": neg_loc, "is_peak": True}


def compute_ratio_with_tanh(numerator: float, denominator: float) -> float:
    """
    Compute ratio with tanh normalization to bound to [-1, 1].
    
    Handles division by zero by returning ±1 based on numerator sign.
    """
    if denominator == 0:
        if numerator == 0:
            return 0.0
        return np.sign(numerator) * 1.0  # tanh(±inf) = ±1
    
    ratio = numerator / denominator
    return np.tanh(ratio)


def get_on_off_ratio(
    on_extreme: float,
    off_extreme: float,
) -> float:
    """
    Compute ON/OFF transient response ratio.
    
    Returns tanh(on_extreme / off_extreme)
    """
    return compute_ratio_with_tanh(on_extreme, off_extreme)


def get_on_trans_sus_ratio(
    on_extreme: float,
    on_sustained: float,
) -> float:
    """
    Compute ON transient/sustained ratio.
    
    Returns tanh(on_extreme / on_sustained)
    """
    return compute_ratio_with_tanh(on_extreme, on_sustained)


def get_off_trans_sus_ratio(
    off_extreme: float,
    off_sustained: float,
) -> float:
    """
    Compute OFF transient/sustained ratio.
    
    Returns tanh(off_extreme / off_sustained)
    """
    return compute_ratio_with_tanh(off_extreme, off_sustained)


def get_on_off_sus_ratio(
    on_sustained: float,
    off_sustained: float,
) -> float:
    """
    Compute ON/OFF sustained response ratio.
    
    Returns tanh(on_sustained / off_sustained)
    """
    return compute_ratio_with_tanh(on_sustained, off_sustained)


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_step_features_from_trace(trace: np.ndarray) -> Dict[str, float]:
    """
    Extract all step response features from a single mean trace.
    
    Uses scipy peak detection to find the most extreme peak (positive or negative)
    in ON and OFF transient windows.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of firing rate values (length 599 at 60 Hz)
        
    Returns
    -------
    dict
        Dictionary of feature name -> value
    """
    # Baseline statistics
    baseline = trace[:PRE_MARGIN]
    base_mean = baseline.mean()
    base_std = baseline.std()
    
    # ON transient peak detection
    on_trans_window = trace[TRANSIENT_START:TRANSIENT_END]
    on_peak_result = find_extreme_peak(on_trans_window, baseline_value=base_mean, baseline_std=base_std)
    on_peak_extreme = on_peak_result["value"]  # NaN if no peak
    time_to_on_peak_extreme = on_peak_result["location"]  # NaN if no peak
    
    # ON sustained
    on_sus_window = trace[SUSTAINED_START:SUSTAINED_END]
    on_sustained = on_sus_window.mean() - base_mean
    
    # OFF transient peak detection
    off_trans_window = trace[OFF_TRANSIENT_START:OFF_TRANSIENT_END]
    off_peak_result = find_extreme_peak(off_trans_window, baseline_value=base_mean, baseline_std=base_std)
    off_peak_extreme = off_peak_result["value"]  # NaN if no peak
    time_to_off_peak_extreme = off_peak_result["location"]  # NaN if no peak
    
    # OFF sustained
    off_sus_window = trace[OFF_SUSTAINED_START:OFF_SUSTAINED_END]
    off_sustained = off_sus_window.mean() - base_mean
    
    # Ratios (will be NaN if peak values are NaN)
    on_off_ratio = get_on_off_ratio(on_peak_extreme, off_peak_extreme)
    on_trans_sus_ratio = get_on_trans_sus_ratio(on_peak_extreme, on_sustained)
    off_trans_sus_ratio = get_off_trans_sus_ratio(off_peak_extreme, off_sustained)
    on_off_sus_ratio = get_on_off_sus_ratio(on_sustained, off_sustained)
    
    return {
        "on_peak_extreme": on_peak_extreme,
        "on_sustained": on_sustained,
        "off_peak_extreme": off_peak_extreme,
        "off_sustained": off_sustained,
        "base_mean": base_mean,
        "base_std": base_std,
        "time_to_on_peak_extreme": time_to_on_peak_extreme,
        "time_to_off_peak_extreme": time_to_off_peak_extreme,
        "on_off_ratio": on_off_ratio,
        "on_trans_sus_ratio": on_trans_sus_ratio,
        "off_trans_sus_ratio": off_trans_sus_ratio,
        "on_off_sus_ratio": on_off_sus_ratio,
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


def extract_step_features(
    df: pd.DataFrame,
    trace_column: str = STEP_TRACE_COLUMN,
    max_firing_rate: float = VALID_MAX_FIRING_RATE,
    skip_filtering: bool = False,
) -> pd.DataFrame:
    """
    Extract step response features from all units in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data
    trace_column : str
        Column name containing the step response trials
    max_firing_rate : float
        Maximum valid firing rate for QC filtering
    skip_filtering : bool
        If True, skip QC filtering (keep all rows, just add NaN for invalid)
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns added
    """
    feature_names = [
        "on_peak_extreme", "on_sustained", "off_peak_extreme", "off_sustained",
        "base_mean", "base_std", "time_to_on_peak_extreme", "time_to_off_peak_extreme",
        "on_off_ratio", "on_trans_sus_ratio", "off_trans_sus_ratio", "on_off_sus_ratio"
    ]
    
    # Initialize feature columns with NaN
    for name in feature_names:
        df[name] = np.nan
    
    # Extract features for each unit
    valid_count = 0
    for idx in tqdm(df.index, desc="Extracting step features"):
        trials_data = df.loc[idx, trace_column]
        
        # Compute mean trace
        mean_trace = compute_mean_trace(trials_data)
        if mean_trace is None:
            continue
        
        # Extract features
        try:
            features = extract_step_features_from_trace(mean_trace)
            
            # Assign features to DataFrame
            for name, value in features.items():
                df.loc[idx, name] = value
            
            valid_count += 1
        except Exception as e:
            # Skip units with errors
            continue
    
    print(f"Extracted features for {valid_count} / {len(df)} units")
    
    # QC filtering: remove rows with extreme values
    if not skip_filtering:
        features_to_check = ["on_peak_extreme", "on_sustained", "off_peak_extreme", "off_sustained", "base_mean"]
        
        original_count = len(df)
        for col in features_to_check:
            mask = (df[col].abs() <= max_firing_rate) | df[col].isna()
            removed = (~mask).sum()
            if removed > 0:
                print(f"  {col}: removed {removed} rows exceeding ±{max_firing_rate} Hz")
            df = df[mask]
        
        # Drop rows with NaN in ratio columns
        ratio_cols = ["on_off_ratio", "on_trans_sus_ratio", "off_trans_sus_ratio", "on_off_sus_ratio"]
        before_dropna = len(df)
        df = df.dropna(subset=ratio_cols)
        dropped = before_dropna - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN in ratio columns")
        
        print(f"Final: {len(df)} / {original_count} units retained after QC")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to extract step features and save to parquet."""
    # Paths
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected.parquet")
    output_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step.parquet")
    
    print("=" * 60)
    print("Step Response Feature Extraction")
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  PRE_MARGIN: {PRE_MARGIN}")
    print(f"  ON transient window: [{TRANSIENT_START}:{TRANSIENT_END}]")
    print(f"  ON sustained window: [{SUSTAINED_START}:{SUSTAINED_END}]")
    print(f"  OFF transient window: [{OFF_TRANSIENT_START}:{OFF_TRANSIENT_END}]")
    print(f"  OFF sustained window: [{OFF_SUSTAINED_START}:{OFF_SUSTAINED_END}]")
    print(f"  Max firing rate: {VALID_MAX_FIRING_RATE} Hz")
    print(f"  Peak prominence: {PROMINENCE_STD_THRESHOLD} * baseline_std")
    print(f"  Trace column: {STEP_TRACE_COLUMN}")
    print(f"  Low-pass filter: {LOWPASS_CUTOFF_FREQ} Hz Bessel (order {LOWPASS_FILTER_ORDER})")
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Extract features
    print("\nExtracting step response features...")
    df = extract_step_features(df, trace_column=STEP_TRACE_COLUMN)
    
    # Print feature statistics
    print("\nFeature statistics:")
    feature_cols = [
        "on_peak_extreme", "on_sustained", "off_peak_extreme", "off_sustained",
        "base_mean", "base_std", "on_off_ratio", "on_trans_sus_ratio",
        "off_trans_sus_ratio", "on_off_sus_ratio"
    ]
    for col in feature_cols:
        if col in df.columns:
            valid = df[col].dropna()
            print(f"  {col}: mean={valid.mean():.3f}, std={valid.std():.3f}, "
                  f"min={valid.min():.3f}, max={valid.max():.3f}")
    
    # Save output
    print(f"\nSaving to: {output_path}")
    df.to_parquet(output_path)
    print(f"Saved DataFrame shape: {df.shape}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

