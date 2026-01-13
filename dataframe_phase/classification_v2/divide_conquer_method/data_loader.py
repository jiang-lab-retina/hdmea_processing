"""
Data loading and filtering for the DEC-refined RGC clustering pipeline.

This module handles:
- Loading parquet data
- RGC-only filtering (axon_type == "rgc")
- Reject reason tracking for audit
"""

import logging
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from . import config

logger = logging.getLogger(__name__)


def load_and_filter_data(
    input_path: Path | str | None = None,
    require_complete_traces: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load parquet file and filter to RGC cells only.
    
    Args:
        input_path: Path to input parquet file. Defaults to config.INPUT_PATH.
        require_complete_traces: If True, drop cells with NaN in required columns.
    
    Returns:
        Tuple of (filtered_dataframe, reject_reasons_dict)
        reject_reasons_dict maps reason to count.
    
    Raises:
        FileNotFoundError: If input_path doesn't exist.
        ValueError: If required columns are missing.
    """
    input_path = Path(input_path) if input_path else config.INPUT_PATH
    
    logger.info(f"Loading data from {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_parquet(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} cells")
    
    # Track rejection reasons
    reject_reasons: Counter = Counter()
    
    # Validate required columns exist
    required_cols = [config.AXON_COL, config.DS_PVAL_COL, config.OS_PVAL_COL]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add iprgc_2hz_QI column with NaN if missing
    if config.IPRGC_QI_COL not in df.columns:
        logger.warning(f"Column {config.IPRGC_QI_COL} not found, adding as NaN")
        df[config.IPRGC_QI_COL] = np.nan
    
    # Step 1: Filter to RGC cells only (key difference from Autoencoder_method)
    rgc_mask = df[config.AXON_COL].str.lower() == "rgc"
    non_rgc_count = (~rgc_mask).sum()
    reject_reasons["not_rgc"] = non_rgc_count
    df = df[rgc_mask].copy()
    logger.info(f"After RGC filter: {len(df)} cells ({non_rgc_count} non-RGC removed)")
    
    # Step 2: Drop rows with NaN in required metadata columns
    if require_complete_traces:
        metadata_cols = [config.DS_PVAL_COL, config.OS_PVAL_COL]
        
        for col in metadata_cols:
            nan_mask = df[col].isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                reject_reasons[f"nan_{col}"] = nan_count
                df = df[~nan_mask].copy()
                logger.info(f"  Removed {nan_count} cells with NaN in {col}")
    
    final_count = len(df)
    total_removed = initial_count - final_count
    
    logger.info(f"Filtering complete: {initial_count} -> {final_count} cells "
                f"({total_removed} removed)")
    
    # Log reject summary
    if reject_reasons:
        logger.info("Reject reasons:")
        for reason, count in reject_reasons.most_common():
            logger.info(f"  {reason}: {count}")
    
    return df, dict(reject_reasons)


def extract_trace_arrays(
    df: pd.DataFrame,
    trace_columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract trace data from DataFrame into numpy arrays with trial averaging.
    
    Handles variable-length traces by truncating to mode length.
    
    Args:
        df: DataFrame with trace columns.
        trace_columns: List of column names to extract.
    
    Returns:
        Dict mapping column name to (n_cells, trace_length) array.
    """
    if trace_columns is None:
        # Get default trace columns from config
        trace_columns = _get_required_trace_columns()
    
    # Check all columns exist
    missing = set(trace_columns) - set(df.columns)
    if missing:
        logger.warning(f"Missing trace columns (will skip): {missing}")
        trace_columns = [c for c in trace_columns if c in df.columns]
    
    result = {}
    
    for col in trace_columns:
        logger.debug(f"Extracting trace: {col}")
        
        # Extract and average trials for each cell
        trial_means = []
        for i, val in enumerate(df[col].values):
            try:
                mean_trace = _average_trials(val)
                mean_trace = np.atleast_1d(np.asarray(mean_trace).flatten())
                trial_means.append(mean_trace)
            except Exception as e:
                logger.warning(f"  Cell {i} in {col}: {e}")
                # Use zeros as fallback
                trial_means.append(np.zeros(1))
        
        # Handle variable-length traces by using mode length
        lengths = np.array([t.shape[0] for t in trial_means])
        min_len = lengths.min()
        max_len = lengths.max()
        
        if min_len != max_len:
            mode_len = int(stats.mode(lengths, keepdims=False).mode)
            logger.debug(f"  {col}: variable lengths [{min_len}, {max_len}], using mode={mode_len}")
            
            # Truncate or pad to mode length
            trial_means = [
                t[:mode_len] if len(t) >= mode_len else 
                np.pad(t, (0, mode_len - len(t)), mode='constant', constant_values=0)
                for t in trial_means
            ]
        
        # Stack into 2D array
        result[col] = np.vstack(trial_means)
        logger.debug(f"  {col}: shape {result[col].shape}")
    
    return result


def _average_trials(trace_data) -> np.ndarray:
    """
    Average across trials if trace has multiple trials.
    
    Handles multiple data formats:
    - 1D array: single trial, return as-is
    - 2D array: multiple trials, average across axis=0
    - Object array: ragged trials (different lengths), stack and average
    
    Args:
        trace_data: Raw trace data (1D array, 2D [trials, time], or object array).
    
    Returns:
        1D trial-mean trace.
    """
    # Handle None or empty data
    if trace_data is None:
        raise ValueError("Trace data is None")
    
    arr = np.asarray(trace_data)
    
    # Handle scalar (0D) or empty arrays
    if arr.ndim == 0 or arr.size == 0:
        raise ValueError(f"Unexpected trace shape: {arr.shape}")
    
    # KEY: Handle object dtype arrays (ragged trials with different lengths)
    # This is the format used in the parquet file for multi-trial data
    if arr.dtype == object and arr.ndim == 1:
        try:
            # Stack trials into 2D array and average
            trials = np.vstack([np.asarray(t, dtype=np.float64) for t in arr])
            return np.mean(trials, axis=0)
        except Exception as e:
            # Fallback: use first trial if stacking fails (truly ragged data)
            logger.debug(f"Falling back to first trial due to: {e}")
            return np.asarray(arr[0], dtype=np.float64)
    
    if arr.ndim == 1:
        return arr.astype(np.float64)
    elif arr.ndim == 2:
        # Average across first axis (trials)
        return np.mean(arr, axis=0).astype(np.float64)
    else:
        raise ValueError(f"Unexpected trace shape: {arr.shape}")


def _get_required_trace_columns() -> list[str]:
    """
    Get list of required trace columns based on config.
    
    Returns:
        List of column names.
    """
    columns = []
    
    # Frequency sections
    columns.extend([
        "freq_section_0p5hz",
        "freq_section_1hz", 
        "freq_section_2hz",
        "freq_section_4hz",
        "freq_section_10hz",
    ])
    
    # Color
    columns.append("green_blue_3s_3i_3x")
    
    # Moving bar (8 directions)
    for direction in config.BAR_DIRECTIONS:
        columns.append(config.BAR_COL_TEMPLATE.format(direction=direction))
    
    # Other segments
    columns.extend([
        "sta_time_course",
        "iprgc_test",
        "step_up_5s_5i_b0_3x",
    ])
    
    return columns
