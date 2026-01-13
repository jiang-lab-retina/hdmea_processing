"""
Signal preprocessing for the Autoencoder-based RGC clustering pipeline.

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
    filter_order: int | None = None,
) -> np.ndarray:
    """
    Preprocess a single segment's traces for all cells.
    
    Args:
        traces: (n_cells, trace_length) raw trace data.
        segment_name: Segment identifier for segment-specific logic.
        sampling_rate: Original sampling rate (Hz). Defaults to config.SAMPLING_RATE.
        lowpass_cutoff: Low-pass filter cutoff (Hz), None to skip.
        target_rate: Target sampling rate after downsampling, None to skip.
        filter_order: Butterworth filter order. Defaults to config.FILTER_ORDER.
    
    Returns:
        (n_cells, new_length) preprocessed traces.
    
    Notes:
        - For freq_section_10hz: no filtering, no downsampling, slice edges
        - For iprgc_test: 2 Hz lowpass, 2 Hz target rate
        - For others: 10 Hz lowpass, 10 Hz target rate
    """
    sampling_rate = sampling_rate if sampling_rate is not None else config.SAMPLING_RATE
    filter_order = filter_order if filter_order is not None else config.FILTER_ORDER
    
    # Determine segment-specific processing parameters
    if segment_name == "freq_section_10hz":
        # No filtering, no downsampling, just slice edges
        start_offset = config.FREQ_10HZ_START_OFFSET
        end_offset = config.FREQ_10HZ_END_OFFSET
        
        if end_offset < 0:
            processed = traces[:, start_offset:end_offset]
        else:
            processed = traces[:, start_offset:]
        
        logger.debug(f"  {segment_name}: sliced edges, shape {traces.shape} -> {processed.shape}")
        return processed
    
    elif segment_name == "sta_time_course":
        # No filtering, no downsampling for sta_time_course (already smooth)
        logger.debug(f"  {segment_name}: no processing, shape {traces.shape}")
        return traces.copy()
    
    elif segment_name == "iprgc_test":
        # Special handling for ipRGC: 2 Hz lowpass, 2 Hz target rate
        lowpass_cutoff = config.IPRGC_LOWPASS_CUTOFF
        target_rate = config.IPRGC_TARGET_RATE
        
    else:
        # Default processing: 10 Hz lowpass, 10 Hz target rate
        # Check if this segment should be filtered
        if segment_name in config.FREQ_SECTION_FILTER:
            if not config.FREQ_SECTION_FILTER[segment_name]:
                # No filtering for this segment
                lowpass_cutoff = None
        
        if lowpass_cutoff is None:
            lowpass_cutoff = config.LOWPASS_CUTOFF
        if target_rate is None:
            target_rate = config.TARGET_SAMPLING_RATE
    
    processed = traces.copy()
    
    # Minimum trace length for filtering (padlen = 3 * (filter_order + 1) for sos = 15 for order 4)
    min_filter_len = 3 * (filter_order + 1) + 1  # = 16
    
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
                else:
                    # Trace too short for filter, skip (use raw values)
                    if i == 0:  # Only log once
                        logger.debug(f"  {segment_name}: traces too short for filter ({trace_len} < {min_filter_len}), skipping filter")
    
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
        Keys: freq_section_*, green_blue_3s_3i_3x, bar_concat, sta_time_course, 
              iprgc_test, step_up_5s_5i_b0_3x
    
    Notes:
        - bar_concat is concatenation of 8 directions in fixed order
    """
    logger.info("Preprocessing all segments...")
    
    # Extract raw traces
    raw_traces = extract_trace_arrays(df)
    
    result = {}
    
    # Process frequency sections
    for col in config.FREQ_SECTION_COLS:
        result[col] = preprocess_segment(raw_traces[col], col)
    
    # Process color trace
    result[config.COLOR_COL] = preprocess_segment(raw_traces[config.COLOR_COL], config.COLOR_COL)
    
    # Process and concatenate moving bar directions
    bar_traces = []
    for col in config.BAR_COLS:
        processed = preprocess_segment(raw_traces[col], col)
        bar_traces.append(processed)
    
    # Concatenate in direction order (0, 45, 90, ..., 315)
    result["bar_concat"] = np.concatenate(bar_traces, axis=1)
    logger.info(f"  bar_concat: concatenated 8 directions, shape {result['bar_concat'].shape}")
    
    # Process STA time course
    result[config.RF_COL] = preprocess_segment(raw_traces[config.RF_COL], config.RF_COL)
    
    # Process ipRGC test trace
    result[config.IPRGC_COL] = preprocess_segment(raw_traces[config.IPRGC_COL], config.IPRGC_COL)
    
    # Process step-up trace
    result[config.STEP_UP_COL] = preprocess_segment(raw_traces[config.STEP_UP_COL], config.STEP_UP_COL)
    
    # Replace NaN with 0 (some traces may have missing data)
    for name in result:
        if np.any(np.isnan(result[name])):
            nan_count = np.isnan(result[name]).sum()
            logger.warning(f"  {name}: replacing {nan_count} NaN values with 0")
            result[name] = np.nan_to_num(result[name], nan=0.0)
    
    # Log summary
    total_length = sum(s.shape[1] for s in result.values())
    logger.info(f"Preprocessing complete: {len(result)} segments, total length {total_length}")
    
    return result


def build_segment_map(
    segments: dict[str, np.ndarray],
) -> dict[str, Tuple[int, int]]:
    """
    Build index map for segment positions in concatenated vector.
    
    Args:
        segments: Dict of preprocessed segment arrays.
    
    Returns:
        Dict mapping segment_name to (start_idx, end_idx) in concatenated vector.
    """
    segment_map = {}
    current_idx = 0
    
    # Use fixed order from config
    for segment_name in config.SEGMENT_NAMES:
        # Map segment name to actual key in segments dict
        if segment_name in segments:
            key = segment_name
        elif segment_name == "bar_concat":
            key = "bar_concat"
        else:
            # Try to find matching key
            key = None
            for k in segments:
                if segment_name in k or k in segment_name:
                    key = k
                    break
            if key is None:
                logger.warning(f"Segment {segment_name} not found in preprocessed segments")
                continue
        
        segment_length = segments[key].shape[1]
        segment_map[segment_name] = (current_idx, current_idx + segment_length)
        current_idx += segment_length
    
    logger.info(f"Segment map: {len(segment_map)} segments, total dim {current_idx}")
    return segment_map


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
    # Use fixed order
    ordered_segments = []
    for segment_name in config.SEGMENT_NAMES:
        if segment_name in segments:
            ordered_segments.append(segments[segment_name])
        elif segment_name == "bar_concat" and "bar_concat" in segments:
            ordered_segments.append(segments["bar_concat"])
    
    return np.concatenate(ordered_segments, axis=1)
