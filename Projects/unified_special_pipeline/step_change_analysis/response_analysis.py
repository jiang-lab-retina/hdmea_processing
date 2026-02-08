"""
Response Analysis for Step Change Analysis Pipeline

This module extracts and analyzes response features across time,
particularly for comparing pre- and post-treatment responses.

Ported from: Legacy_code/.../low_glucose/A04_step_analysis_v2.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from .specific_config import (
    ResponseAnalysisConfig,
    PipelineConfig,
    default_config,
    AGONIST_START_TIME_S,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_response_feature(
    step_responses: np.ndarray,
    baseline_range: Tuple[int, int] = (0, 5),
    peak_range: Tuple[int, int] = (10, 20),
) -> np.ndarray:
    """
    Extract response feature (peak - baseline) for each trial.
    
    Args:
        step_responses: Array of shape (n_trials, n_timepoints)
        baseline_range: Sample indices for baseline period
        peak_range: Sample indices for peak response period
    
    Returns:
        1D array of response magnitudes for each trial
    """
    if step_responses.size == 0:
        return np.array([])
    
    baseline = step_responses[:, baseline_range[0]:baseline_range[1]].mean(axis=1)
    peak = step_responses[:, peak_range[0]:peak_range[1]].max(axis=1)
    
    return np.abs(peak - baseline)


def get_trace_features_for_chain(
    recordings: Dict[str, Dict[str, Any]],
    chain: pd.Series,
    config: Optional[ResponseAnalysisConfig] = None,
    recording_lead_times: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Extract response features for a single alignment chain.
    
    Args:
        recordings: Dict mapping recording names to data
        chain: Series mapping recording names to unit IDs
        config: Response analysis configuration
        recording_lead_times: Dict mapping recording names to start times (minutes)
    
    Returns:
        Tuple of (feature_series, raw_traces_series)
    """
    if config is None:
        config = default_config.response_analysis
    
    if recording_lead_times is None:
        recording_lead_times = {}
    
    feature_series = pd.Series(dtype=float)
    raw_traces_series = pd.Series(dtype=object)
    
    rec_names = sorted(recordings.keys())
    
    for j, rec_name in enumerate(rec_names):
        unit_id = chain.get(rec_name)
        
        if pd.isna(unit_id) or unit_id is None:
            continue
        
        unit_id = str(unit_id).strip()
        if not unit_id:
            continue
        
        rec_data = recordings.get(rec_name, {})
        unit_data = rec_data.get("units", {}).get(unit_id, {})
        
        if "step_responses" not in unit_data:
            continue
        
        step_responses = np.array(unit_data["step_responses"])
        
        if step_responses.size == 0:
            continue
        
        # Extract response feature
        feature = extract_response_feature(
            step_responses,
            baseline_range=config.baseline_range,
            peak_range=config.on_peak_range,  # Default to ON response
        )
        
        # Calculate time points
        lead_time = recording_lead_times.get(rec_name, 0)
        file_offset = j * config.file_interval_minutes * 60  # seconds
        
        time_points = (
            np.arange(len(feature)) * config.trial_interval_s +
            lead_time * 60 +  # Convert minutes to seconds
            file_offset
        )
        
        # Create series with time as index
        feature_df = pd.Series(feature, index=time_points.astype(int))
        raw_df = pd.Series(
            [trace for trace in step_responses],
            index=time_points.astype(int),
            dtype=object,
        )
        
        # Concatenate
        feature_series = pd.concat([feature_series, feature_df])
        raw_traces_series = pd.concat([raw_traces_series, raw_df])
    
    return feature_series, raw_traces_series


def get_all_trace_features(
    grouped_data: Dict[str, Any],
    config: Optional[ResponseAnalysisConfig] = None,
    recording_lead_times: Optional[Dict[str, float]] = None,
    use_fixed_chains: bool = True,
    peak_range: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract response features for all chains in aligned group.
    
    Args:
        grouped_data: Grouped data from create_aligned_group()
        config: Response analysis configuration
        recording_lead_times: Dict mapping recording names to start times
        use_fixed_chains: Use fixed reference alignment chains
        peak_range: Override peak range for feature extraction
    
    Returns:
        Tuple of (features_df, raw_traces_df)
            - features_df: Columns are chains, rows are time points
            - raw_traces_df: Same structure but with raw trace arrays
    """
    if config is None:
        config = default_config.response_analysis
    
    # Override peak range if provided
    if peak_range is not None:
        config = ResponseAnalysisConfig(
            baseline_range=config.baseline_range,
            on_peak_range=peak_range,
            off_peak_range=config.off_peak_range,
            trial_interval_s=config.trial_interval_s,
            file_interval_minutes=config.file_interval_minutes,
            normalize_mode=config.normalize_mode,
        )
    
    # Get alignment chains
    if use_fixed_chains and "fixed_alignment_chains" in grouped_data:
        chains_df = grouped_data["fixed_alignment_chains"]
    else:
        chains_df = grouped_data.get("alignment_chains", pd.DataFrame())
    
    if chains_df.empty:
        logger.warning("No alignment chains found")
        return pd.DataFrame(), pd.DataFrame()
    
    # Drop rows with any NaN (incomplete chains)
    chains_df = chains_df.dropna(how="any")
    
    if chains_df.empty:
        logger.warning("No complete chains found")
        return pd.DataFrame(), pd.DataFrame()
    
    recordings = grouped_data.get("recordings", {})
    
    all_features = pd.DataFrame()
    all_raw_traces = pd.DataFrame()
    
    for idx, (row_idx, chain) in enumerate(chains_df.iterrows()):
        feature_series, raw_series = get_trace_features_for_chain(
            recordings,
            chain,
            config,
            recording_lead_times,
        )
        
        if feature_series.empty:
            continue
        
        # Name the series for the dataframe column
        feature_series.name = f"chain_{row_idx}"
        raw_series.name = f"chain_{row_idx}"
        
        all_features = pd.concat([all_features, feature_series], axis=1)
        all_raw_traces = pd.concat([all_raw_traces, raw_series], axis=1)
    
    logger.info(f"Extracted features for {len(all_features.columns)} chains")
    
    return all_features, all_raw_traces


# =============================================================================
# Normalization and Comparison
# =============================================================================

def normalize_features(
    features_df: pd.DataFrame,
    mode: str = "first",
    n_baseline: int = 5,
) -> pd.DataFrame:
    """
    Normalize response features.
    
    Args:
        features_df: Features DataFrame (columns=chains, rows=time)
        mode: Normalization mode:
            - "first": Normalize to mean of first N points
            - "max": Normalize to maximum value
            - "no_normalize": Return as-is
        n_baseline: Number of points for baseline in "first" mode
    
    Returns:
        Normalized features DataFrame
    """
    if features_df.empty:
        return features_df
    
    df = features_df.T  # Transpose so rows are chains
    df = df.apply(pd.to_numeric, errors="coerce")
    
    if mode == "max":
        max_vals = df.mean(axis=0, skipna=True).max()
        if max_vals > 0:
            df = df / max_vals
    
    elif mode == "first":
        baseline_vals = df.iloc[:, :n_baseline].mean(axis=1, skipna=True)
        # Avoid division by zero
        baseline_vals = baseline_vals.replace(0, np.nan)
        df = df.div(baseline_vals, axis=0)
    
    elif mode == "no_normalize":
        pass
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    
    return df.T  # Transpose back


def compute_binned_statistics(
    features_df: pd.DataFrame,
    bin_size: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Compute binned mean and SEM for features.
    
    Args:
        features_df: Normalized features DataFrame
        bin_size: Number of time points per bin
    
    Returns:
        Dictionary with:
            - time_points: Center time of each bin
            - mean: Mean response per bin
            - sem: Standard error of mean per bin
            - per_bin_values: List of per-chain values for each bin
    """
    if features_df.empty:
        return {
            "time_points": np.array([]),
            "mean": np.array([]),
            "sem": np.array([]),
            "per_bin_values": [],
        }
    
    df = features_df.T  # Rows = chains, columns = time
    time_vals = np.array(df.columns)
    n_points = len(time_vals)
    
    time_centers = []
    means = []
    sems = []
    per_bin_values = []
    
    for start in range(0, n_points, bin_size):
        stop = min(start + bin_size, n_points)
        cols = slice(start, stop)
        
        # Time center
        time_centers.append(np.nanmean(time_vals[cols]))
        
        # Per-chain mean within bin
        bin_per_chain = df.iloc[:, cols].mean(axis=1, skipna=True)
        per_bin_values.append(bin_per_chain)
        
        # Across-chain statistics
        means.append(bin_per_chain.mean(skipna=True))
        
        count = bin_per_chain.count()
        if count > 0:
            sems.append(bin_per_chain.std(skipna=True, ddof=1) / np.sqrt(count))
        else:
            sems.append(np.nan)
    
    return {
        "time_points": np.array(time_centers),
        "mean": np.array(means),
        "sem": np.array(sems),
        "per_bin_values": per_bin_values,
    }


def compare_groups_statistics(
    group1_stats: Dict[str, np.ndarray],
    group2_stats: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute statistical comparison between two groups.
    
    Args:
        group1_stats: Statistics from compute_binned_statistics()
        group2_stats: Statistics from compute_binned_statistics()
    
    Returns:
        Dictionary with:
            - time_points: Common time points
            - p_values: P-value for each time bin
            - significant: Boolean array (p < 0.05)
    """
    time_points = group1_stats["time_points"]
    per_bin_1 = group1_stats["per_bin_values"]
    per_bin_2 = group2_stats["per_bin_values"]
    
    p_values = []
    
    for i in range(len(time_points)):
        if i >= len(per_bin_1) or i >= len(per_bin_2):
            p_values.append(np.nan)
            continue
        
        vals1 = per_bin_1[i].dropna()
        vals2 = per_bin_2[i].dropna()
        
        if len(vals1) > 1 and len(vals2) > 1:
            _, p = ttest_ind(vals1, vals2, equal_var=False)
            p_values.append(p)
        else:
            p_values.append(np.nan)
    
    p_values = np.array(p_values)
    
    return {
        "time_points": time_points,
        "p_values": p_values,
        "significant": p_values < 0.05,
    }


# =============================================================================
# Time-based Analysis
# =============================================================================

def split_pre_post_treatment(
    features_df: pd.DataFrame,
    treatment_time_s: float = AGONIST_START_TIME_S,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features into pre- and post-treatment periods.
    
    Args:
        features_df: Features DataFrame with time as index
        treatment_time_s: Treatment time in seconds
    
    Returns:
        Tuple of (pre_treatment_df, post_treatment_df)
    """
    if features_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    time_index = features_df.index.astype(float)
    
    pre_mask = time_index < treatment_time_s
    post_mask = time_index >= treatment_time_s
    
    pre_df = features_df.loc[pre_mask]
    post_df = features_df.loc[post_mask]
    
    return pre_df, post_df


def compute_treatment_effect(
    features_df: pd.DataFrame,
    treatment_time_s: float = AGONIST_START_TIME_S,
    baseline_window_s: float = 60.0,
    post_window_s: float = 60.0,
) -> Dict[str, Any]:
    """
    Compute treatment effect by comparing pre and post responses.
    
    Args:
        features_df: Features DataFrame
        treatment_time_s: Treatment application time
        baseline_window_s: Window before treatment for baseline
        post_window_s: Window after treatment for comparison
    
    Returns:
        Dictionary with effect statistics
    """
    if features_df.empty:
        return {}
    
    time_index = features_df.index.astype(float)
    
    # Baseline period (just before treatment)
    baseline_start = treatment_time_s - baseline_window_s
    baseline_mask = (time_index >= baseline_start) & (time_index < treatment_time_s)
    
    # Post-treatment period
    post_end = treatment_time_s + post_window_s
    post_mask = (time_index >= treatment_time_s) & (time_index < post_end)
    
    baseline_df = features_df.loc[baseline_mask]
    post_df = features_df.loc[post_mask]
    
    if baseline_df.empty or post_df.empty:
        return {}
    
    # Compute per-chain means
    baseline_means = baseline_df.mean(axis=0)
    post_means = post_df.mean(axis=0)
    
    # Effect size (percent change)
    effect = (post_means - baseline_means) / baseline_means * 100
    
    return {
        "baseline_mean": baseline_means.mean(),
        "baseline_std": baseline_means.std(),
        "post_mean": post_means.mean(),
        "post_std": post_means.std(),
        "effect_percent": effect.mean(),
        "effect_std": effect.std(),
        "n_chains": len(baseline_means.dropna()),
    }


# =============================================================================
# Summary Statistics
# =============================================================================

def summarize_response_timecourse(
    grouped_data: Dict[str, Any],
    peak_type: str = "ON",
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Generate summary statistics for response timecourse.
    
    Args:
        grouped_data: Grouped aligned data
        peak_type: "ON" or "OFF" response
        config: Pipeline configuration
    
    Returns:
        Dictionary with summary statistics and data
    """
    if config is None:
        config = default_config
    
    # Select peak range
    if peak_type.upper() == "ON":
        peak_range = config.response_analysis.on_peak_range
    else:
        peak_range = config.response_analysis.off_peak_range
    
    # Extract features
    features_df, raw_traces_df = get_all_trace_features(
        grouped_data,
        config.response_analysis,
        peak_range=peak_range,
    )
    
    if features_df.empty:
        return {}
    
    # Normalize
    norm_features = normalize_features(
        features_df,
        mode=config.response_analysis.normalize_mode,
    )
    
    # Compute binned statistics
    stats = compute_binned_statistics(
        norm_features,
        bin_size=config.visualization.bin_size,
    )
    
    # Compute treatment effect
    effect = compute_treatment_effect(
        features_df,
        treatment_time_s=config.agonist_start_time_s,
    )
    
    return {
        "peak_type": peak_type,
        "features_df": features_df,
        "normalized_df": norm_features,
        "raw_traces_df": raw_traces_df,
        "binned_stats": stats,
        "treatment_effect": effect,
        "n_chains": len(features_df.columns),
        "treatment_time_s": config.agonist_start_time_s,
    }
