"""
Signal preprocessing for the DEC-refined RGC clustering pipeline.

This module handles:
- Per-segment signal conditioning (filtering, downsampling)
- Segment concatenation (moving bar directions)
- Segment index mapping
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import config
from .data_loader import extract_trace_arrays, extract_last_trial_column

logger = logging.getLogger(__name__)


def preprocess_segment(
    traces: np.ndarray,
    segment_name: str,
    sampling_rate: float | None = None,
    lowpass_cutoff: float | None = None,
    target_rate: float | None = None,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Preprocess a single segment's traces for all cells.
    
    Args:
        traces: (n_cells, trace_length) raw trace data.
        segment_name: Segment identifier for segment-specific logic.
        sampling_rate: Original sampling rate (Hz). Defaults to config.SAMPLING_RATE.
        lowpass_cutoff: Low-pass filter cutoff (Hz), None to skip.
        target_rate: Target sampling rate after downsampling, None to skip.
        filter_order: Butterworth filter order.
    
    Returns:
        (n_cells, new_length) preprocessed traces.
    
    Notes:
        - For freq_section_10hz: no filtering, no downsampling, edge crop
        - For sta_time_course: no processing (already 60 samples)
        - For iprgc_test: crop to 30s, 10 Hz lowpass, 10 Hz target rate
        - For others: 10 Hz lowpass, 10 Hz target rate
    """
    sampling_rate = sampling_rate if sampling_rate is not None else config.SAMPLING_RATE
    
    # Determine segment-specific processing parameters
    if segment_name == "freq_section_10hz":
        # No filtering, no downsampling — just crop edges (first/last 1 second)
        edge_samples = int(sampling_rate)  # 60 samples at 60 Hz
        
        if traces.shape[1] > 2 * edge_samples:
            processed = traces[:, edge_samples:-edge_samples]
        else:
            processed = traces.copy()
        
        logger.debug(f"  {segment_name}: edge crop, shape {traces.shape} -> {processed.shape}")
        return processed
    
    elif segment_name == "sta_time_course":
        # No filtering, no downsampling — keep as-is
        logger.debug(f"  {segment_name}: no processing, shape {traces.shape}")
        return traces.copy()
    
    elif segment_name == "iprgc_test":
        # Crop to first 30 seconds (sustained response period) before filtering
        crop_samples = int(30 * sampling_rate)  # 1800 at 60 Hz
        if traces.shape[1] > crop_samples:
            traces = traces[:, :crop_samples]
        logger.debug(f"  {segment_name}: cropped to 30s ({traces.shape[1]} samples)")
        lowpass_cutoff = config.LOWPASS_IPRGC
        target_rate = config.TARGET_RATE_IPRGC
        
    else:
        # Default processing: 10 Hz lowpass, 10 Hz target rate
        if lowpass_cutoff is None:
            lowpass_cutoff = config.LOWPASS_DEFAULT
        if target_rate is None:
            target_rate = config.TARGET_RATE_DEFAULT
    
    processed = traces.copy()
    
    # Minimum trace length for filtering
    min_filter_len = 3 * (filter_order + 1) + 1  # = 16 for order 4
    
    # Apply low-pass filter if specified
    if lowpass_cutoff is not None and lowpass_cutoff > 0:
        nyq = sampling_rate / 2
        if lowpass_cutoff >= nyq:
            logger.warning(f"Lowpass cutoff {lowpass_cutoff} >= Nyquist {nyq}, skipping filter")
        else:
            sos = butter(filter_order, lowpass_cutoff / nyq, btype='low', output='sos')
            
            # Apply filter to each trace
            for i in range(len(processed)):
                trace_len = processed[i].shape[0]
                if trace_len >= min_filter_len:
                    processed[i] = sosfiltfilt(sos, processed[i])
                elif i == 0:
                    logger.debug(f"  {segment_name}: traces too short for filter, skipping")
    
    # Downsample if specified
    if target_rate is not None and target_rate < sampling_rate:
        downsample_factor = int(round(sampling_rate / target_rate))
        processed = processed[:, ::downsample_factor]
        logger.debug(f"  {segment_name}: downsampled by {downsample_factor}x")
    
    logger.debug(f"  {segment_name}: shape {traces.shape} -> {processed.shape}")
    return processed


def preprocess_all_segments(
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Preprocess all trace segments for the entire dataset.
    
    Args:
        df: DataFrame with all trace columns.
    
    Returns:
        Tuple of (segments, full_segments):
        - segments: Dict of (n_cells, length) arrays for clustering.
        - full_segments: Copy of segments for prototype plotting.
    
    Notes:
        - bar_concat is concatenation of 8 directions in fixed order
        - freq_section_10hz and sta_time_course: no filtering, no downsampling
        - iprgc_test: cropped to 30s, 10 Hz lowpass, 10 Hz target rate
        - Other segments: 10 Hz lowpass, 10 Hz target rate
    """
    logger.info("Preprocessing all segments...")
    
    # Extract raw traces
    raw_traces = extract_trace_arrays(df)
    
    result = {}
    
    # Process frequency sections
    freq_cols = [
        "freq_section_0p5hz",
        "freq_section_1hz",
        "freq_section_2hz",
        "freq_section_4hz",
        "freq_section_10hz",
    ]
    for col in freq_cols:
        if col in raw_traces:
            result[col] = preprocess_segment(raw_traces[col], col)
    
    # Process color trace
    color_col = "green_blue_3s_3i_3x"
    if color_col in raw_traces:
        result[color_col] = preprocess_segment(raw_traces[color_col], color_col)
    
    # Process and concatenate moving bar directions
    bar_traces = []
    for direction in config.BAR_DIRECTIONS:
        col = config.BAR_COL_TEMPLATE.format(direction=direction)
        if col in raw_traces:
            processed = preprocess_segment(raw_traces[col], col)
            bar_traces.append(processed)
    
    if bar_traces:
        # Concatenate in direction order (0, 45, 90, ..., 315)
        result["bar_concat"] = np.concatenate(bar_traces, axis=1)
        logger.info(f"  bar_concat: concatenated {len(bar_traces)} directions, "
                   f"shape {result['bar_concat'].shape}")
    
    # Process STA time course
    rf_col = "sta_time_course"
    if rf_col in raw_traces:
        result[rf_col] = preprocess_segment(raw_traces[rf_col], rf_col)
    
    # Process ipRGC test trace (last trial only — avoids averaging over adaptation)
    iprgc_col = "iprgc_test"
    if iprgc_col in df.columns:
        iprgc_raw_last = extract_last_trial_column(df, iprgc_col)
        result[iprgc_col] = preprocess_segment(iprgc_raw_last, iprgc_col)
        logger.info(f"  {iprgc_col}: using last trial, shape {result[iprgc_col].shape}")
    
    # Process step-up trace
    step_col = "step_up_5s_5i_b0_3x"
    if step_col in raw_traces:
        result[step_col] = preprocess_segment(raw_traces[step_col], step_col)
    
    # Ensure numeric dtype, replace NaN with 0, and normalize
    for name in result:
        # Convert to float64 to handle any object arrays
        try:
            arr = np.asarray(result[name], dtype=np.float64)
            if np.any(np.isnan(arr)):
                nan_count = np.isnan(arr).sum()
                logger.warning(f"  {name}: replacing {nan_count} NaN values with 0")
                arr = np.nan_to_num(arr, nan=0.0)
            result[name] = arr
        except (ValueError, TypeError) as e:
            logger.warning(f"  {name}: could not convert to float64: {e}")
            # Try to salvage by replacing problematic values
            arr = result[name]
            if arr.dtype == object:
                # Handle object arrays by converting element-wise
                arr = np.array([np.asarray(x, dtype=np.float64) if x is not None else np.zeros(1) 
                               for x in arr.flat]).reshape(arr.shape)
            result[name] = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    
    # Normalize traces if enabled in config
    if getattr(config, 'NORMALIZE_TRACES', True):
        result = normalize_traces(result, method=getattr(config, 'NORMALIZE_METHOD', 'zscore'))
    
    # Store full-length traces for prototype plotting (same as clustering input)
    full_segments = {name: arr.copy() for name, arr in result.items()}
    
    # Log summary
    total_length = sum(s.shape[1] for s in result.values())
    logger.info(f"Preprocessing complete: {len(result)} segments, total length {total_length}")
    
    return result, full_segments


def extract_iprgc_last_trial(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and preprocess the last trial of iprgc_test for each cell.

    Used for prototype plotting: the last trial shows the sustained ipRGC
    response without averaging over adaptation effects.

    Args:
        df: Group DataFrame containing the 'iprgc_test' column.

    Returns:
        (n_cells, trace_length) array of preprocessed last-trial iprgc traces.
    """
    col = "iprgc_test"
    if col not in df.columns:
        logger.warning(f"Column '{col}' not found in DataFrame")
        return None

    raw_last = extract_last_trial_column(df, col)
    processed = preprocess_segment(raw_last, col)

    # Clean up: float64, replace NaN
    processed = np.asarray(processed, dtype=np.float64)
    if np.any(np.isnan(processed)):
        nan_count = np.isnan(processed).sum()
        logger.warning(f"  iprgc_test_last_trial: replacing {nan_count} NaN values with 0")
        processed = np.nan_to_num(processed, nan=0.0)

    logger.info(f"  iprgc_test_last_trial: shape {processed.shape}")
    return processed


def normalize_traces(
    segments: dict[str, np.ndarray],
    method: str = "zscore",
) -> dict[str, np.ndarray]:
    """
    Normalize traces to remove magnitude differences across cells.
    
    Args:
        segments: Dict of segment arrays (n_cells, segment_length).
        method: Normalization method:
            - "zscore": Per-cell z-score (mean=0, std=1) per segment
            - "maxabs": Divide by max absolute value per cell
            - "minmax": Scale to [0, 1] per cell
            - "baseline": Subtract baseline (first 10% of trace)
    
    Returns:
        Dict of normalized segment arrays.
    """
    result = {}
    
    for name, arr in segments.items():
        if method == "zscore":
            # Z-score normalization per cell (across time)
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
            result[name] = (arr - mean) / std
            
        elif method == "maxabs":
            # Divide by max absolute value per cell
            max_abs = np.abs(arr).max(axis=1, keepdims=True)
            max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
            result[name] = arr / max_abs
            
        elif method == "minmax":
            # Scale to [0, 1] per cell
            arr_min = arr.min(axis=1, keepdims=True)
            arr_max = arr.max(axis=1, keepdims=True)
            range_val = arr_max - arr_min
            range_val = np.where(range_val < 1e-8, 1.0, range_val)
            result[name] = (arr - arr_min) / range_val
            
        elif method == "baseline":
            # Subtract baseline (first 10% of trace)
            n_baseline = max(1, int(arr.shape[1] * 0.1))
            baseline = arr[:, :n_baseline].mean(axis=1, keepdims=True)
            result[name] = arr - baseline
            
        else:
            logger.warning(f"Unknown normalization method: {method}, skipping")
            result[name] = arr
    
    logger.info(f"Normalized traces using method='{method}'")
    return result


def get_segment_lengths(segments: dict[str, np.ndarray]) -> dict[str, int]:
    """
    Get the length of each segment after preprocessing.
    
    Args:
        segments: Dict of preprocessed segment arrays.
    
    Returns:
        Dict mapping segment_name to length.
    """
    return {name: arr.shape[1] for name, arr in segments.items()}


def concatenate_segments(segments: dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate all segments into a single feature vector per cell.
    
    Args:
        segments: Dict of preprocessed segment arrays.
    
    Returns:
        (n_cells, total_length) concatenated array.
    """
    # Use fixed order from config
    ordered_segments = []
    for segment_name in config.SEGMENT_NAMES:
        if segment_name in segments:
            ordered_segments.append(segments[segment_name])
    
    return np.concatenate(ordered_segments, axis=1)
