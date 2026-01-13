"""
Data loading and filtering for the Autoencoder-based RGC clustering pipeline.

This module reuses the Baden pipeline's proven filtering logic for consistency.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Import Baden pipeline's preprocessing functions for consistency
from Baden_method import preprocessing as baden_preprocessing

logger = logging.getLogger(__name__)


def load_and_filter_data(
    input_path: Path | str | None = None,
    qi_threshold: float | None = None,
    baseline_max: float | None = None,
    min_batch_cells: int | None = None,
) -> pd.DataFrame:
    """
    Load parquet file and apply cell filtering using Baden pipeline logic.
    
    Args:
        input_path: Path to input parquet file. Defaults to config.INPUT_PATH.
        qi_threshold: Minimum quality index for inclusion. Defaults to config.QI_THRESHOLD.
        baseline_max: Maximum baseline firing rate (Hz). Defaults to config.BASELINE_MAX_THRESHOLD.
        min_batch_cells: Minimum cells per batch after filtering. Defaults to config.MIN_BATCH_GOOD_CELLS.
    
    Returns:
        DataFrame with filtered cells.
    
    Raises:
        FileNotFoundError: If input_path doesn't exist.
        ValueError: If required columns are missing.
    """
    # Apply defaults from config
    input_path = Path(input_path) if input_path else config.INPUT_PATH
    qi_threshold = qi_threshold if qi_threshold is not None else config.QI_THRESHOLD
    baseline_max = baseline_max if baseline_max is not None else config.BASELINE_MAX_THRESHOLD
    min_batch_cells = min_batch_cells if min_batch_cells is not None else config.MIN_BATCH_GOOD_CELLS
    
    # Use Baden pipeline's load function
    logger.info(f"Loading data from {input_path}")
    df = baden_preprocessing.load_data(input_path)
    initial_count = len(df)
    
    # Add iprgc_2hz_QI column with NaN if missing (not in Baden pipeline)
    if config.IPRGC_QI_COL not in df.columns:
        logger.warning(f"Column {config.IPRGC_QI_COL} not found, adding as NaN")
        df[config.IPRGC_QI_COL] = np.nan
    
    # Use Baden pipeline's filtering functions
    logger.info("Applying filters (using Baden pipeline logic)...")
    
    # Step 1: Filter by NaN and QI
    df = baden_preprocessing.filter_rows(df, qi_threshold=qi_threshold)
    
    # Step 2: Filter by baseline
    if baseline_max is not None:
        df = baden_preprocessing.filter_by_baseline(df, baseline_max=baseline_max)
    
    # Step 3: Filter by batch size
    if min_batch_cells is not None and min_batch_cells > 0:
        df = baden_preprocessing.filter_by_batch_size(df, min_cells_per_batch=min_batch_cells)
    
    final_count = len(df)
    logger.info(f"Filtering complete: {initial_count} -> {final_count} cells "
                f"({initial_count - final_count} removed)")
    
    return df


def extract_trace_arrays(
    df: pd.DataFrame,
    trace_columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract trace data from DataFrame into numpy arrays with trial averaging.
    
    Uses Baden pipeline's trial averaging logic for consistency.
    Handles variable-length traces by truncating to minimum length.
    
    Args:
        df: DataFrame with trace columns.
        trace_columns: List of column names to extract. Defaults to config.REQUIRED_TRACE_COLS.
    
    Returns:
        Dict mapping column name to (n_cells, trace_length) array.
    """
    trace_columns = trace_columns if trace_columns is not None else config.REQUIRED_TRACE_COLS
    
    # Check all columns exist
    missing = set(trace_columns) - set(df.columns)
    if missing:
        raise KeyError(f"Missing trace columns: {missing}")
    
    result = {}
    
    for col in trace_columns:
        logger.debug(f"Extracting trace: {col}")
        
        # Use Baden's average_trials for each cell
        trial_means = []
        for i, val in enumerate(df[col].values):
            try:
                mean_trace = baden_preprocessing.average_trials(val)
                # Ensure it's a 1D array
                mean_trace = np.atleast_1d(np.asarray(mean_trace).flatten())
                trial_means.append(mean_trace)
            except Exception as e:
                logger.warning(f"  Cell {i} in {col}: {e}")
                raise
        
        # Handle variable-length traces by using the most common (mode) length
        lengths = np.array([t.shape[0] for t in trial_means])
        min_len = lengths.min()
        max_len = lengths.max()
        
        if min_len != max_len:
            # Use mode (most common length) and truncate
            from scipy import stats
            mode_len = int(stats.mode(lengths, keepdims=False).mode)
            logger.debug(f"  {col}: variable lengths [{min_len}, {max_len}], using mode={mode_len}")
            
            # Truncate all traces to mode length
            trial_means = [t[:mode_len] if len(t) >= mode_len else 
                          np.pad(t, (0, mode_len - len(t)), mode='constant', constant_values=0)
                          for t in trial_means]
        
        # Stack into 2D array
        result[col] = np.vstack(trial_means)
        logger.debug(f"  {col}: shape {result[col].shape}")
    
    return result
