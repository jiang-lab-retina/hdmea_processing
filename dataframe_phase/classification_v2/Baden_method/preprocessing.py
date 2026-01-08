"""
Preprocessing module for the Baden-method RGC clustering pipeline.

This module handles:
- Data loading from parquet files
- Row filtering (NaN, quality index, axon type)
- DS/non-DS population splitting
- Signal conditioning (low-pass filter, baseline subtraction, normalization)
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load data from a parquet file.
    
    Args:
        path: Path to parquet file. Defaults to config.INPUT_PATH.
        
    Returns:
        DataFrame with all columns from the parquet file.
        
    Raises:
        FileNotFoundError: If the parquet file doesn't exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        path = config.INPUT_PATH
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    logger.info(f"Loading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} cells")
    
    # Validate required columns exist
    missing_cols = set(config.REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


# =============================================================================
# Row Filtering
# =============================================================================

def filter_rows(
    df: pd.DataFrame,
    qi_threshold: float = None,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter rows based on NaN values, quality index, and axon type.
    
    Args:
        df: Input DataFrame.
        qi_threshold: Minimum quality index value. Defaults to config.QI_THRESHOLD.
        required_columns: Columns that must not have NaN. Defaults to config.REQUIRED_COLS.
        
    Returns:
        Filtered DataFrame with only valid cells.
    """
    if qi_threshold is None:
        qi_threshold = config.QI_THRESHOLD
    if required_columns is None:
        required_columns = config.REQUIRED_COLS
    
    initial_count = len(df)
    logger.info(f"Starting filtering with {initial_count} cells")
    
    # Step 1: Filter cells with NaN in required columns
    # For scalar columns, use standard notna check
    scalar_cols = [config.QI_COL, config.DS_PVAL_COL, config.AXON_COL]
    df_filtered = df.dropna(subset=scalar_cols)
    after_scalar_nan = len(df_filtered)
    logger.info(f"After scalar NaN filter: {after_scalar_nan} cells ({initial_count - after_scalar_nan} removed)")
    
    # Step 2: Filter cells with NaN values inside trace arrays
    # Note: Trace columns may contain nested arrays (multiple trials)
    # We need to handle both flat arrays and nested trial arrays
    # Include baseline trace column in addition to feature trace columns
    trace_cols = config.REQUIRED_TRACE_COLS + [config.BASELINE_TRACE_COL]
    # Remove duplicates if baseline trace is already in required trace cols
    trace_cols = list(dict.fromkeys(trace_cols))
    mask_valid_traces = pd.Series(True, index=df_filtered.index)
    
    for col in trace_cols:
        def has_nan_in_trace(x):
            if x is None:
                return True
            if isinstance(x, (list, np.ndarray)):
                arr = np.asarray(x)
                # Check if this is a nested array (multiple trials)
                if arr.dtype == object and len(arr.shape) == 1:
                    # Nested array - check each trial
                    try:
                        for trial in arr:
                            trial_arr = np.asarray(trial, dtype=np.float64)
                            if np.any(np.isnan(trial_arr)):
                                return True
                        return False
                    except (ValueError, TypeError):
                        return True
                else:
                    # Flat array
                    try:
                        arr_f = np.asarray(x, dtype=np.float64)
                        return np.any(np.isnan(arr_f))
                    except (ValueError, TypeError):
                        return True
            return pd.isna(x)
        
        mask_valid_traces &= ~df_filtered[col].apply(has_nan_in_trace)
    
    df_filtered = df_filtered[mask_valid_traces]
    after_trace_nan = len(df_filtered)
    logger.info(f"After trace NaN filter: {after_trace_nan} cells ({after_scalar_nan - after_trace_nan} removed)")
    
    # Step 3: Filter by quality index
    df_filtered = df_filtered[df_filtered[config.QI_COL] > qi_threshold]
    after_qi = len(df_filtered)
    logger.info(f"After QI > {qi_threshold} filter: {after_qi} cells ({after_trace_nan - after_qi} removed)")
    
    # Step 4: Filter by axon type (allow multiple valid types)
    df_filtered = df_filtered[df_filtered[config.AXON_COL].isin(config.VALID_AXON_TYPES)]
    after_axon = len(df_filtered)
    logger.info(f"After axon_type in {config.VALID_AXON_TYPES} filter: {after_axon} cells ({after_qi - after_axon} removed)")
    
    logger.info(f"Filtering complete: {after_axon}/{initial_count} cells retained ({100*after_axon/initial_count:.1f}%)")
    
    if after_axon == 0:
        raise ValueError("No valid cells remain after filtering")
    
    return df_filtered.copy()


def filter_by_baseline(
    df: pd.DataFrame,
    baseline_max: float = None,
) -> pd.DataFrame:
    """
    Filter cells by baseline value computed from the step_up trace.
    
    The baseline is computed the same way as in preprocessing:
    1. Average trials
    2. Low-pass filter at 10 Hz
    3. Downsample to 10 Hz
    4. Take median of first N samples
    
    Args:
        df: Input DataFrame with step_up trace column.
        baseline_max: Maximum allowed baseline value. Defaults to config.BASELINE_MAX_THRESHOLD.
                      If None, no filtering is applied.
        
    Returns:
        Filtered DataFrame with high-baseline cells removed.
    """
    if baseline_max is None:
        baseline_max = getattr(config, 'BASELINE_MAX_THRESHOLD', None)
    
    if baseline_max is None:
        logger.info("Baseline filtering disabled (BASELINE_MAX_THRESHOLD is None)")
        return df
    
    initial_count = len(df)
    logger.info(f"Computing baselines for {initial_count} cells...")
    
    # Compute baseline for each cell
    def compute_baseline_for_filter(trace):
        """Compute baseline using the same method as preprocessing."""
        try:
            # Average trials
            trace = average_trials(trace)
            # Low-pass filter
            trace = lowpass_filter(trace)
            # Downsample to 10 Hz
            trace = downsample(trace)
            # Compute baseline: median of first N samples
            return compute_cell_baseline(trace)
        except Exception:
            return np.nan
    
    baselines = df[config.BASELINE_TRACE_COL].apply(compute_baseline_for_filter)
    
    # Log baseline statistics
    valid_baselines = baselines.dropna()
    logger.info(f"Baseline range: {valid_baselines.min():.2f} to {valid_baselines.max():.2f}")
    logger.info(f"Baseline median: {valid_baselines.median():.2f}")
    
    # Filter by baseline threshold
    mask = baselines <= baseline_max
    df_filtered = df[mask].copy()
    
    after_baseline = len(df_filtered)
    removed = initial_count - after_baseline
    logger.info(f"After baseline <= {baseline_max} filter: {after_baseline} cells ({removed} removed)")
    
    if after_baseline == 0:
        raise ValueError(f"No valid cells remain after baseline filtering (threshold: {baseline_max})")
    
    return df_filtered


def filter_by_batch_size(
    df: pd.DataFrame,
    min_cells_per_batch: int = None,
) -> pd.DataFrame:
    """
    Filter out all cells from batches that have too few good cells.
    
    Batch is determined by the index (filename without unit ID).
    
    Args:
        df: Input DataFrame (should already be filtered by QI).
        min_cells_per_batch: Minimum number of cells required per batch.
                            Defaults to config.MIN_BATCH_GOOD_CELLS.
                            If None or 0, no filtering is applied.
        
    Returns:
        Filtered DataFrame with small-batch cells removed.
    """
    if min_cells_per_batch is None:
        min_cells_per_batch = getattr(config, 'MIN_BATCH_GOOD_CELLS', None)
    
    if min_cells_per_batch is None or min_cells_per_batch <= 0:
        logger.info("Batch size filtering disabled (MIN_BATCH_GOOD_CELLS is None or 0)")
        return df
    
    initial_count = len(df)
    
    # Extract batch from index (filename without unit ID)
    def get_batch(idx):
        s = str(idx)
        parts = s.rsplit('_', 1)
        return parts[0] if len(parts) == 2 else s
    
    # Count cells per batch
    batch_series = pd.Series([get_batch(idx) for idx in df.index], index=df.index)
    batch_counts = batch_series.value_counts()
    
    # Find batches that meet the threshold
    valid_batches = batch_counts[batch_counts >= min_cells_per_batch].index.tolist()
    excluded_batches = batch_counts[batch_counts < min_cells_per_batch].index.tolist()
    
    logger.info(f"Batch size filtering: min {min_cells_per_batch} cells/batch")
    logger.info(f"  Total batches: {len(batch_counts)}")
    logger.info(f"  Valid batches (>= {min_cells_per_batch} cells): {len(valid_batches)}")
    logger.info(f"  Excluded batches (< {min_cells_per_batch} cells): {len(excluded_batches)}")
    
    # Filter to keep only cells from valid batches
    mask = batch_series.isin(valid_batches)
    df_filtered = df[mask].copy()
    
    after_batch_filter = len(df_filtered)
    removed = initial_count - after_batch_filter
    logger.info(f"After batch size filter: {after_batch_filter} cells ({removed} removed from {len(excluded_batches)} batches)")
    
    if after_batch_filter == 0:
        raise ValueError(f"No valid cells remain after batch size filtering (threshold: {min_cells_per_batch})")
    
    return df_filtered


def split_ds_nds(
    df: pd.DataFrame,
    p_threshold: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into DS (direction-selective) and non-DS populations.
    
    Args:
        df: Input DataFrame with ds_p_value column.
        p_threshold: P-value threshold for DS classification. Defaults to config.DS_P_THRESHOLD.
        
    Returns:
        Tuple of (df_ds, df_nds) DataFrames.
    """
    if p_threshold is None:
        p_threshold = config.DS_P_THRESHOLD
    
    ds_mask = df[config.DS_PVAL_COL] < p_threshold
    
    df_ds = df[ds_mask].copy()
    df_nds = df[~ds_mask].copy()
    
    logger.info(f"DS/non-DS split: {len(df_ds)} DS cells, {len(df_nds)} non-DS cells")
    
    return df_ds, df_nds


# =============================================================================
# Signal Processing
# =============================================================================

def lowpass_filter(
    trace: np.ndarray,
    fs: float = None,
    cutoff: float = None,
    order: int = None,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter to a trace.
    
    Args:
        trace: 1D array of signal values.
        fs: Sampling frequency in Hz. Defaults to config.SAMPLING_RATE.
        cutoff: Cutoff frequency in Hz. Defaults to config.LOWPASS_CUTOFF.
        order: Filter order. Defaults to config.FILTER_ORDER.
        
    Returns:
        Filtered trace as 1D array.
    """
    if fs is None:
        fs = config.SAMPLING_RATE
    if cutoff is None:
        cutoff = config.LOWPASS_CUTOFF
    if order is None:
        order = config.FILTER_ORDER
    
    trace = np.asarray(trace, dtype=np.float64)
    
    # Handle edge case of very short traces
    if len(trace) < 3 * order:
        logger.warning(f"Trace too short for filtering ({len(trace)} samples), returning original")
        return trace
    
    # Design Butterworth filter
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    
    # Apply zero-phase filter
    filtered = sosfiltfilt(sos, trace)
    
    return filtered


def downsample(
    trace: np.ndarray,
    factor: int = None,
) -> np.ndarray:
    """
    Downsample a trace by taking every nth sample.
    
    Args:
        trace: 1D array of signal values (should be low-pass filtered first).
        factor: Downsampling factor. Defaults to config.DOWNSAMPLE_FACTOR.
        
    Returns:
        Downsampled trace.
    """
    if factor is None:
        factor = config.DOWNSAMPLE_FACTOR
    
    trace = np.asarray(trace, dtype=np.float64)
    
    # Take every nth sample
    return trace[::factor]


def baseline_subtract(
    trace: np.ndarray,
    baseline_value: float = None,
) -> np.ndarray:
    """
    Subtract baseline from trace.
    
    Args:
        trace: 1D array of signal values.
        baseline_value: Pre-computed baseline value to subtract. 
                       If None, computes from the trace itself (legacy behavior).
        
    Returns:
        Baseline-subtracted trace.
    """
    trace = np.asarray(trace, dtype=np.float64)
    
    if baseline_value is None:
        # Legacy behavior: compute from this trace (first 5 samples at 10 Hz)
        n_samples = min(config.BASELINE_N_SAMPLES, len(trace))
        baseline_value = np.median(trace[:n_samples])
    
    return trace - baseline_value


def compute_cell_baseline(
    trace: np.ndarray,
    n_samples: int = None,
) -> float:
    """
    Compute baseline value from a filtered and downsampled trace.
    
    This computes the median of the first N samples of the trace at 10 Hz.
    The trace should already be:
    1. Trial-averaged
    2. Low-pass filtered at 10 Hz
    3. Downsampled to 10 Hz
    
    Args:
        trace: 1D array of signal values at 10 Hz.
        n_samples: Number of samples for baseline. Defaults to config.BASELINE_N_SAMPLES.
        
    Returns:
        Baseline value (median of first N samples).
    """
    if n_samples is None:
        n_samples = config.BASELINE_N_SAMPLES
    
    trace = np.asarray(trace, dtype=np.float64)
    n_samples = min(n_samples, len(trace))
    
    baseline = np.median(trace[:n_samples])
    logger.debug(f"Computed baseline: {baseline:.4f} from first {n_samples} samples")
    
    return baseline


def normalize_trace(
    trace: np.ndarray,
    eps: float = None,
) -> np.ndarray:
    """
    Normalize trace using max-absolute value.
    
    Args:
        trace: 1D array of signal values.
        eps: Small value for numerical stability. Defaults to config.NORMALIZE_EPS.
        
    Returns:
        Normalized trace with values in range [-1, 1].
    """
    if eps is None:
        eps = config.NORMALIZE_EPS
    
    trace = np.asarray(trace, dtype=np.float64)
    
    max_abs = np.max(np.abs(trace))
    
    if max_abs < eps:
        # Trace is effectively zero
        return np.zeros_like(trace)
    
    return trace / (max_abs + eps)


def average_trials(trace: np.ndarray) -> np.ndarray:
    """
    Average across trials if trace contains multiple repetitions.
    
    Args:
        trace: Either a 1D array (single trial) or object array containing
               multiple trial arrays.
               
    Returns:
        1D array of trial-averaged signal values.
    """
    arr = np.asarray(trace)
    
    # Check if this is a nested array (multiple trials)
    if arr.dtype == object and len(arr.shape) == 1:
        # Stack trials and average
        try:
            trials = np.vstack([np.asarray(t, dtype=np.float64) for t in arr])
            return np.mean(trials, axis=0)
        except Exception as e:
            logger.warning(f"Failed to average trials: {e}")
            # Fall back to first trial
            return np.asarray(arr[0], dtype=np.float64)
    else:
        # Already a flat array
        return np.asarray(arr, dtype=np.float64)


def preprocess_single_trace(
    trace: np.ndarray,
    apply_filter: bool = True,
    apply_downsample: bool = True,
    baseline_value: float = None,
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a single trace.
    
    Pipeline:
    1. Average trials (if multiple trials present) - at 60 Hz
    2. Low-pass filter at 10 Hz (optional, for stimulus traces) - at 60 Hz
    3. Downsample from 60 Hz to 10 Hz (optional)
    4. Baseline subtraction using provided baseline value - at 10 Hz
    5. Max-absolute normalization
    
    Args:
        trace: 1D array of signal values, or nested array of trials.
        apply_filter: Whether to apply low-pass filter. Set False for RF traces.
        apply_downsample: Whether to downsample after filtering.
        baseline_value: Pre-computed baseline value from step-up trace.
                       If None, computes from this trace's first 5 samples.
        
    Returns:
        Preprocessed trace at 10 Hz sampling rate (if downsampled).
    """
    # Step 0: Average trials if needed (at 60 Hz)
    trace = average_trials(trace)
    
    # Step 1: Low-pass filter (optional, at 60 Hz)
    if apply_filter:
        trace = lowpass_filter(trace)
    
    # Step 2: Downsample from 60 Hz to 10 Hz (before baseline subtraction)
    if apply_downsample:
        trace = downsample(trace)
    
    # Step 3: Baseline subtraction (optional, controlled by config)
    if config.APPLY_BASELINE_ZEROING:
        trace = baseline_subtract(trace, baseline_value=baseline_value)
    
    # Step 4: Max-abs normalization (optional, controlled by config)
    if config.APPLY_MAX_ABS_NORMALIZATION:
        trace = normalize_trace(trace)
    
    return trace


def preprocess_traces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to all stimulus traces in DataFrame.
    
    The baseline is computed from the step-up trace (filtered + downsampled to 10 Hz)
    using the median of the first 5 samples. This same baseline value is then
    subtracted from ALL traces of each cell.
    
    Applies low-pass filter, baseline subtraction, and normalization to:
    - Chirp traces
    - Color traces  
    - Moving bar traces (8 directions)
    
    RF time course (sta_time_course) is only baseline-subtracted and normalized
    (no low-pass filter per specification).
    
    Args:
        df: DataFrame with trace columns.
        
    Returns:
        DataFrame with preprocessed trace columns (same column names, modified values).
    """
    logger.info(f"Preprocessing traces for {len(df)} cells...")
    
    df = df.copy()
    
    # Columns that get full preprocessing (filter + downsample + baseline + normalize)
    # Includes color, bar, and filtered freq sections
    filter_cols = [config.COLOR_COL] + config.BAR_COLS
    
    # Add freq section columns that should be filtered (0.5, 1, 2, 4 Hz)
    for col in config.FREQ_SECTION_COLS:
        if config.FREQ_SECTION_FILTER.get(col, True) and col != "freq_section_10hz":
            filter_cols.append(col)
    
    # RF column: no filter, but still downsample + baseline + normalize
    no_filter_cols = [config.RF_COL]
    
    # freq_section_10hz: no filter, no downsample (stays at 60 Hz), only baseline + normalize
    no_filter_no_downsample_cols = []
    for col in config.FREQ_SECTION_COLS:
        if not config.FREQ_SECTION_FILTER.get(col, True):
            no_filter_no_downsample_cols.append(col)
    
    # Step 1: Compute baseline for each cell from the step-up trace
    # The baseline is computed from the filtered + downsampled trace at 10 Hz
    logger.info(f"Computing baselines from {config.BASELINE_TRACE_COL}...")
    baselines = {}
    
    for idx in df.index:
        trace = df.loc[idx, config.BASELINE_TRACE_COL]
        
        # Average trials
        trace = average_trials(trace)
        
        # Low-pass filter at 10 Hz
        trace = lowpass_filter(trace)
        
        # Downsample to 10 Hz
        trace = downsample(trace)
        
        # Compute baseline: median of first 5 samples at 10 Hz
        baselines[idx] = compute_cell_baseline(trace)
    
    logger.info(f"Computed baselines for {len(baselines)} cells")
    logger.info(f"Baseline range: {min(baselines.values()):.4f} to {max(baselines.values()):.4f}")
    
    # Step 2: Preprocess all traces using the computed baseline for each cell
    # Preprocess columns with filter
    for col in filter_cols:
        logger.debug(f"Preprocessing column (with filter): {col}")
        df[col] = df.apply(
            lambda row: preprocess_single_trace(
                row[col], 
                apply_filter=True, 
                baseline_value=baselines[row.name]
            ), 
            axis=1
        )
    
    # Preprocess RF column without filter (but still downsample + baseline)
    for col in no_filter_cols:
        logger.debug(f"Preprocessing column (no filter): {col}")
        df[col] = df.apply(
            lambda row: preprocess_single_trace(
                row[col], 
                apply_filter=False,
                apply_downsample=True,
                baseline_value=baselines[row.name]
            ), 
            axis=1
        )
    
    # Preprocess freq_section_10hz: no filter, no downsample (stays at 60 Hz)
    for col in no_filter_no_downsample_cols:
        logger.debug(f"Preprocessing column (no filter, no downsample): {col}")
        df[col] = df.apply(
            lambda row: preprocess_single_trace(
                row[col], 
                apply_filter=False,
                apply_downsample=False,
                baseline_value=baselines[row.name]
            ), 
            axis=1
        )
    
    logger.info("Trace preprocessing complete")
    
    return df


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_traces(df: pd.DataFrame) -> bool:
    """
    Validate that all traces are properly preprocessed.
    
    Args:
        df: DataFrame with preprocessed traces.
        
    Returns:
        True if all traces are valid.
        
    Raises:
        ValueError: If any traces contain NaN or Inf values.
    """
    trace_cols = config.REQUIRED_TRACE_COLS
    
    for col in trace_cols:
        for idx, trace in df[col].items():
            trace = np.asarray(trace)
            if np.any(np.isnan(trace)):
                raise ValueError(f"NaN found in {col} for cell {idx}")
            if np.any(np.isinf(trace)):
                raise ValueError(f"Inf found in {col} for cell {idx}")
    
    logger.info("All traces validated successfully")
    return True


def check_trace_lengths(df: pd.DataFrame) -> dict:
    """
    Check trace lengths for each column.
    
    Args:
        df: DataFrame with trace columns.
        
    Returns:
        Dictionary mapping column name to (min_length, max_length, mode_length).
    """
    trace_cols = config.REQUIRED_TRACE_COLS
    length_info = {}
    
    for col in trace_cols:
        lengths = df[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
        length_info[col] = {
            'min': lengths.min(),
            'max': lengths.max(),
            'mode': lengths.mode().iloc[0] if len(lengths.mode()) > 0 else 0,
        }
    
    return length_info

