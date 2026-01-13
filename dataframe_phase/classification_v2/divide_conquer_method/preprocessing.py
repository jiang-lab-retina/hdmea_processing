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
from .data_loader import extract_trace_arrays

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
        - For iprgc_test: 4 Hz lowpass, 2 Hz target rate
        - For others: 10 Hz lowpass, 10 Hz target rate
    """
    sampling_rate = sampling_rate if sampling_rate is not None else config.SAMPLING_RATE
    
    # Determine segment-specific processing parameters
    if segment_name == "freq_section_10hz":
        # No filtering, no downsampling, just crop edges (first/last 1 second)
        edge_samples = int(sampling_rate)  # 60 samples at 60 Hz
        
        if traces.shape[1] > 2 * edge_samples:
            processed = traces[:, edge_samples:-edge_samples]
        else:
            processed = traces.copy()
        
        logger.debug(f"  {segment_name}: edge crop, shape {traces.shape} -> {processed.shape}")
        return processed
    
    elif segment_name == "sta_time_course":
        # No filtering, no downsampling for sta_time_course (keep as-is)
        logger.debug(f"  {segment_name}: no processing, shape {traces.shape}")
        return traces.copy()
    
    elif segment_name == "iprgc_test":
        # Special handling: 4 Hz lowpass, 2 Hz target rate
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
) -> dict[str, np.ndarray]:
    """
    Preprocess all trace segments for the entire dataset.
    
    Args:
        df: DataFrame with all trace columns.
    
    Returns:
        Dict mapping segment_name to (n_cells, segment_length) arrays.
        Keys match config.SEGMENT_NAMES.
    
    Notes:
        - bar_concat is concatenation of 8 directions in fixed order
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
    
    # Process ipRGC test trace
    iprgc_col = "iprgc_test"
    if iprgc_col in raw_traces:
        result[iprgc_col] = preprocess_segment(raw_traces[iprgc_col], iprgc_col)
    
    # Process step-up trace
    step_col = "step_up_5s_5i_b0_3x"
    if step_col in raw_traces:
        result[step_col] = preprocess_segment(raw_traces[step_col], step_col)
    
    # Replace NaN with 0
    for name in result:
        if np.any(np.isnan(result[name])):
            nan_count = np.isnan(result[name]).sum()
            logger.warning(f"  {name}: replacing {nan_count} NaN values with 0")
            result[name] = np.nan_to_num(result[name], nan=0.0)
    
    # Log summary
    total_length = sum(s.shape[1] for s in result.values())
    logger.info(f"Preprocessing complete: {len(result)} segments, total length {total_length}")
    
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
