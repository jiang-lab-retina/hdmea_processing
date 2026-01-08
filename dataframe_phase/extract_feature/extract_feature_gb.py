"""
Green-blue chromatic response feature extraction.

Extracts ON/OFF response features from green-blue stimulus traces.
Adapted from legacy code in cluster_analysis_2.py (lines 917-994) for 60 Hz data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from scipy import signal
from scipy.signal import find_peaks

from step_config import (
    GB_PRE_MARGIN,
    GB_STEP_DURATION,
    GB_TRANSIENT_DURATION,
    GB_TRACE_COLUMN,
    GB_VALID_MAX_FIRING_RATE,
    SAMPLING_RATE,
    LOWPASS_CUTOFF_FREQ,
    LOWPASS_FILTER_ORDER,
    PROMINENCE_STD_THRESHOLD,
)

# =============================================================================
# Calculated Frame Indices (from base parameters)
# =============================================================================

# Baseline window
GB_BASELINE_START = 0
GB_BASELINE_END = GB_PRE_MARGIN

# Green ON transient window
GB_GREEN_TRANSIENT_START = GB_PRE_MARGIN
GB_GREEN_TRANSIENT_END = GB_PRE_MARGIN + GB_TRANSIENT_DURATION

# Green OFF transient window (after green step ends)
GB_GREEN_OFF_TRANSIENT_START = GB_PRE_MARGIN + GB_STEP_DURATION
GB_GREEN_OFF_TRANSIENT_END = GB_PRE_MARGIN + GB_STEP_DURATION + GB_TRANSIENT_DURATION

# Blue ON transient window
GB_BLUE_TRANSIENT_START = GB_PRE_MARGIN + 2 * GB_STEP_DURATION
GB_BLUE_TRANSIENT_END = GB_PRE_MARGIN + 2 * GB_STEP_DURATION + GB_TRANSIENT_DURATION

# Blue OFF transient window
GB_BLUE_OFF_TRANSIENT_START = GB_PRE_MARGIN + 3 * GB_STEP_DURATION
GB_BLUE_OFF_TRANSIENT_END = GB_PRE_MARGIN + 3 * GB_STEP_DURATION + GB_TRANSIENT_DURATION


# =============================================================================
# Helper Functions (reused from extract_feature_step.py)
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
    
    If no peaks found: value = 0.0, location = NaN
    
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


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_gb_features_from_trace(trace: np.ndarray) -> Dict[str, float]:
    """
    Extract all green-blue response features from a single mean trace.
    
    Uses scipy peak detection to find the most extreme peak (positive or negative)
    in each transient window.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of firing rate values
        
    Returns
    -------
    dict
        Dictionary of feature name -> value
    """
    # Baseline statistics
    baseline = trace[GB_BASELINE_START:GB_BASELINE_END]
    base_mean = baseline.mean()
    base_std = baseline.std()
    
    # Green ON transient peak detection
    green_on_window = trace[GB_GREEN_TRANSIENT_START:GB_GREEN_TRANSIENT_END]
    green_on_result = find_extreme_peak(green_on_window, baseline_value=base_mean, baseline_std=base_std)
    green_on_peak_extreme = green_on_result["value"]
    time_to_green_on_peak = green_on_result["location"]
    
    # Blue ON transient peak detection
    blue_on_window = trace[GB_BLUE_TRANSIENT_START:GB_BLUE_TRANSIENT_END]
    blue_on_result = find_extreme_peak(blue_on_window, baseline_value=base_mean, baseline_std=base_std)
    blue_on_peak_extreme = blue_on_result["value"]
    time_to_blue_on_peak = blue_on_result["location"]
    
    # Green OFF transient peak detection
    green_off_window = trace[GB_GREEN_OFF_TRANSIENT_START:GB_GREEN_OFF_TRANSIENT_END]
    green_off_result = find_extreme_peak(green_off_window, baseline_value=base_mean, baseline_std=base_std)
    green_off_peak_extreme = green_off_result["value"]
    time_to_green_off_peak = green_off_result["location"]
    
    # Blue OFF transient peak detection
    blue_off_window = trace[GB_BLUE_OFF_TRANSIENT_START:GB_BLUE_OFF_TRANSIENT_END]
    blue_off_result = find_extreme_peak(blue_off_window, baseline_value=base_mean, baseline_std=base_std)
    blue_off_peak_extreme = blue_off_result["value"]
    time_to_blue_off_peak = blue_off_result["location"]
    
    # Ratios (both tanh normalized)
    green_blue_on_ratio = compute_ratio_with_tanh(green_on_peak_extreme, blue_on_peak_extreme)
    green_blue_off_ratio = compute_ratio_with_tanh(green_off_peak_extreme, blue_off_peak_extreme)
    
    return {
        "green_on_peak_extreme": green_on_peak_extreme,
        "blue_on_peak_extreme": blue_on_peak_extreme,
        "green_off_peak_extreme": green_off_peak_extreme,
        "blue_off_peak_extreme": blue_off_peak_extreme,
        "gb_base_mean": base_mean,
        "gb_base_std": base_std,
        "time_to_green_on_peak": time_to_green_on_peak,
        "time_to_blue_on_peak": time_to_blue_on_peak,
        "time_to_green_off_peak": time_to_green_off_peak,
        "time_to_blue_off_peak": time_to_blue_off_peak,
        "green_blue_on_ratio": green_blue_on_ratio,
        "green_blue_off_ratio": green_blue_off_ratio,
    }


def extract_gb_features(
    df: pd.DataFrame,
    trace_column: str = GB_TRACE_COLUMN,
    max_firing_rate: float = GB_VALID_MAX_FIRING_RATE,
    skip_filtering: bool = False,
) -> pd.DataFrame:
    """
    Extract green-blue response features from all units in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data
    trace_column : str
        Column name containing the green-blue response trials
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
        "green_on_peak_extreme", "blue_on_peak_extreme",
        "green_off_peak_extreme", "blue_off_peak_extreme",
        "gb_base_mean", "gb_base_std",
        "time_to_green_on_peak", "time_to_blue_on_peak",
        "time_to_green_off_peak", "time_to_blue_off_peak",
        "green_blue_on_ratio", "green_blue_off_ratio",
    ]
    
    # Initialize feature columns with NaN
    for name in feature_names:
        df[name] = np.nan
    
    # Extract features for each unit
    valid_count = 0
    for idx in tqdm(df.index, desc="Extracting green-blue features"):
        trials_data = df.loc[idx, trace_column]
        
        # Compute mean trace
        mean_trace = compute_mean_trace(trials_data)
        if mean_trace is None:
            continue
        
        # Extract features
        try:
            features = extract_gb_features_from_trace(mean_trace)
            
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
        features_to_check = [
            "green_on_peak_extreme", "blue_on_peak_extreme",
            "green_off_peak_extreme", "blue_off_peak_extreme",
            "gb_base_mean",
        ]
        
        original_count = len(df)
        for col in features_to_check:
            mask = (df[col].abs() <= max_firing_rate) | df[col].isna()
            removed = (~mask).sum()
            if removed > 0:
                print(f"  {col}: removed {removed} rows exceeding ±{max_firing_rate} Hz")
            df = df[mask]
        
        # Drop rows with NaN in ratio columns
        ratio_cols = ["green_blue_on_ratio", "green_blue_off_ratio"]
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
    """Main function to extract green-blue features and save to parquet."""
    # Paths
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step.parquet")
    output_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step_gb.parquet")
    
    print("=" * 60)
    print("Green-Blue Response Feature Extraction")
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  GB_PRE_MARGIN: {GB_PRE_MARGIN}")
    print(f"  GB_STEP_DURATION: {GB_STEP_DURATION}")
    print(f"  GB_TRANSIENT_DURATION: {GB_TRANSIENT_DURATION}")
    print("\nCalculated frame indices:")
    print(f"  Baseline: [{GB_BASELINE_START}:{GB_BASELINE_END}]")
    print(f"  Green ON transient: [{GB_GREEN_TRANSIENT_START}:{GB_GREEN_TRANSIENT_END}]")
    print(f"  Green OFF transient: [{GB_GREEN_OFF_TRANSIENT_START}:{GB_GREEN_OFF_TRANSIENT_END}]")
    print(f"  Blue ON transient: [{GB_BLUE_TRANSIENT_START}:{GB_BLUE_TRANSIENT_END}]")
    print(f"  Blue OFF transient: [{GB_BLUE_OFF_TRANSIENT_START}:{GB_BLUE_OFF_TRANSIENT_END}]")
    print(f"\n  Max firing rate: {GB_VALID_MAX_FIRING_RATE} Hz")
    print(f"  Peak prominence: {PROMINENCE_STD_THRESHOLD} * baseline_std")
    print(f"  Trace column: {GB_TRACE_COLUMN}")
    print(f"  Low-pass filter: {LOWPASS_CUTOFF_FREQ} Hz Bessel (order {LOWPASS_FILTER_ORDER})")
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Extract features
    print("\nExtracting green-blue response features...")
    df = extract_gb_features(df, trace_column=GB_TRACE_COLUMN)
    
    # Print feature statistics
    print("\nFeature statistics:")
    feature_cols = [
        "green_on_peak_extreme", "blue_on_peak_extreme",
        "green_off_peak_extreme", "blue_off_peak_extreme",
        "gb_base_mean", "gb_base_std",
        "green_blue_on_ratio", "green_blue_off_ratio",
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

