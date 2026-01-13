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
    qi_threshold: float | None = None,
    baseline_max: float | None = None,
    min_batch_cells: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load parquet file and filter to RGC cells only.
    
    Args:
        input_path: Path to input parquet file. Defaults to config.INPUT_PATH.
        require_complete_traces: If True, drop cells with NaN in required columns.
        qi_threshold: Minimum quality index for inclusion. Defaults to config.QI_THRESHOLD.
        baseline_max: Maximum baseline firing rate (Hz). Defaults to config.BASELINE_MAX_THRESHOLD.
        min_batch_cells: Minimum cells per batch. Defaults to config.MIN_BATCH_GOOD_CELLS.
    
    Returns:
        Tuple of (filtered_dataframe, reject_reasons_dict)
        reject_reasons_dict maps reason to count.
    
    Raises:
        FileNotFoundError: If input_path doesn't exist.
        ValueError: If required columns are missing.
    """
    # Apply defaults from config
    qi_threshold = qi_threshold if qi_threshold is not None else getattr(config, 'QI_THRESHOLD', None)
    baseline_max = baseline_max if baseline_max is not None else getattr(config, 'BASELINE_MAX_THRESHOLD', None)
    min_batch_cells = min_batch_cells if min_batch_cells is not None else getattr(config, 'MIN_BATCH_GOOD_CELLS', None)
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
    
    # Step 3: Drop rows with None values in trace columns
    trace_cols = _get_required_trace_columns()
    trace_cols_present = [c for c in trace_cols if c in df.columns]
    
    total_none_removed = 0
    for col in trace_cols_present:
        # Check for None values in trace column (not NaN - these are object arrays)
        none_mask = df[col].apply(lambda x: x is None)
        none_count = none_mask.sum()
        if none_count > 0:
            reject_reasons[f"none_{col}"] = none_count
            df = df[~none_mask].copy()
            total_none_removed += none_count
            logger.warning(f"  Removed {none_count} cells with None in trace column '{col}'")
    
    if total_none_removed > 0:
        logger.warning(f"Total {total_none_removed} cells removed due to None trace values. "
                      f"Check data source for missing recordings.")
    
    # Step 4: Filter by quality index (QI)
    if qi_threshold is not None and config.STEP_UP_QI_COL in df.columns:
        qi_mask = df[config.STEP_UP_QI_COL] >= qi_threshold
        low_qi_count = (~qi_mask).sum()
        if low_qi_count > 0:
            reject_reasons["low_qi"] = low_qi_count
            df = df[qi_mask].copy()
            logger.info(f"  Removed {low_qi_count} cells with QI < {qi_threshold}")
    
    # Step 5: Filter by baseline firing rate
    if baseline_max is not None and config.BASELINE_TRACE_COL in df.columns:
        # Compute baseline from first 1 second of baseline trace
        baselines = _compute_baselines(df, config.BASELINE_TRACE_COL)
        if baselines is not None:
            high_baseline_mask = baselines > baseline_max
            high_baseline_count = high_baseline_mask.sum()
            if high_baseline_count > 0:
                reject_reasons["high_baseline"] = high_baseline_count
                df = df[~high_baseline_mask].copy()
                logger.info(f"  Removed {high_baseline_count} cells with baseline > {baseline_max} Hz")
    
    # Step 6: Filter batches with too few cells
    if min_batch_cells is not None and 'batch_id' in df.columns:
        batch_counts = df['batch_id'].value_counts()
        small_batches = batch_counts[batch_counts < min_batch_cells].index
        if len(small_batches) > 0:
            small_batch_mask = df['batch_id'].isin(small_batches)
            small_batch_count = small_batch_mask.sum()
            reject_reasons["small_batch"] = small_batch_count
            df = df[~small_batch_mask].copy()
            logger.info(f"  Removed {small_batch_count} cells from {len(small_batches)} batches with < {min_batch_cells} cells")
    
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


def _compute_baselines(
    df: pd.DataFrame,
    trace_col: str,
    sampling_rate: float = 60.0,
    baseline_seconds: float = 1.0,
) -> np.ndarray | None:
    """
    Compute baseline firing rate from the first N seconds of a trace.
    
    Args:
        df: DataFrame with trace column.
        trace_col: Column name for baseline trace.
        sampling_rate: Sampling rate in Hz.
        baseline_seconds: Number of seconds at start to use for baseline.
    
    Returns:
        Array of baseline values (Hz), or None if column not found.
    """
    if trace_col not in df.columns:
        logger.warning(f"Baseline column {trace_col} not found, skipping baseline filter")
        return None
    
    n_baseline_samples = int(sampling_rate * baseline_seconds)
    baselines = []
    
    for val in df[trace_col].values:
        try:
            trace = _average_trials(val)
            if len(trace) >= n_baseline_samples:
                baseline = np.mean(trace[:n_baseline_samples])
            else:
                baseline = np.mean(trace)
            baselines.append(baseline)
        except Exception:
            baselines.append(0.0)
    
    return np.array(baselines)
