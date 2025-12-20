"""
Core computation functions for Electrode Image STA (eimage_sta).

This module provides the main computation logic for computing spike-triggered
averages from HD-MEA sensor data.

Key functions:
    - compute_eimage_sta: Main entry point for processing a recording
    - compute_sta_for_unit: Compute STA for a single unit
    - write_eimage_sta_to_hdf5: Write results to HDF5 file

Performance optimizations:
    - Vectorized high-pass filtering (scipy filtfilt with axis parameter)
    - Vectorized spike window extraction (NumPy fancy indexing)
    - Memory-mapped sensor data access
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from hdmea.io.cmcr import load_sensor_data, load_sensor_data_chunked, get_sensor_data_info
from hdmea.preprocess.filtering import apply_highpass_filter_3d, apply_highpass_filter_3d_legacy


logger = logging.getLogger(__name__)


def _get_cache_path(cmcr_path: Path, cache_path: Optional[Path]) -> Path:
    """Generate cache path if not provided."""
    if cache_path is not None:
        return cache_path
    return cmcr_path.with_suffix(".filtered_cache.h5")


def _load_cached_filtered_data(
    cache_path: Path,
    cutoff_hz: float,
    filter_order: int,
    duration_samples: Optional[int],
) -> Optional[np.ndarray]:
    """
    Load cached filtered data if available and parameters match.
    
    Args:
        cache_path: Path to cache file.
        cutoff_hz: Expected filter cutoff.
        filter_order: Expected filter order.
        duration_samples: Expected number of samples (None for all data).
    
    Returns:
        Filtered data array if cache valid, None otherwise.
    """
    if not cache_path.exists():
        logger.debug(f"Cache file not found: {cache_path}")
        return None
    
    try:
        with h5py.File(cache_path, "r") as f:
            if "filtered_data" not in f:
                logger.debug("No filtered_data in cache")
                return None
            
            # Validate parameters match
            attrs = f["filtered_data"].attrs
            if attrs.get("cutoff_hz") != cutoff_hz:
                logger.debug(f"Cache cutoff mismatch: {attrs.get('cutoff_hz')} != {cutoff_hz}")
                return None
            if attrs.get("filter_order") != filter_order:
                logger.debug(f"Cache filter_order mismatch")
                return None
            
            cached_data = f["filtered_data"][:]
            
            # If duration_samples is None, use all cached data
            if duration_samples is None:
                logger.info(f"Loaded cached filtered data: shape={cached_data.shape}")
                return cached_data
            
            if cached_data.shape[0] < duration_samples:
                logger.debug(f"Cache too short: {cached_data.shape[0]} < {duration_samples}")
                return None
            
            logger.info(f"Loaded cached filtered data: shape={cached_data.shape}")
            return cached_data[:duration_samples]
            
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def _save_filtered_data_cache(
    cache_path: Path,
    filtered_data: np.ndarray,
    cutoff_hz: float,
    filter_order: int,
    sampling_rate: float,
) -> None:
    """
    Save filtered data to cache file.
    
    Args:
        cache_path: Path to cache file.
        filtered_data: Filtered sensor data array.
        cutoff_hz: Filter cutoff used.
        filter_order: Filter order used.
        sampling_rate: Sampling rate in Hz.
    """
    try:
        with h5py.File(cache_path, "w") as f:
            ds = f.create_dataset(
                "filtered_data",
                data=filtered_data,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            ds.attrs["cutoff_hz"] = cutoff_hz
            ds.attrs["filter_order"] = filter_order
            ds.attrs["sampling_rate"] = sampling_rate
            ds.attrs["version"] = VERSION
        
        logger.info(f"Saved filtered data cache: {cache_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

# Feature version for cache invalidation
VERSION = "1.0.0"


@dataclass
class EImageSTAConfig:
    """Configuration for eimage_sta computation.
    
    Attributes:
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        pre_samples: Number of samples before spike in window.
        post_samples: Number of samples after spike in window.
        spike_limit: Maximum spikes per unit (-1 for no limit).
        duration_s: Duration of sensor data to process in seconds.
        skip_highpass: If True, skip high-pass filter and mean-center instead.
        use_cache: If True, cache filtered data for reuse.
        cache_path: Path to cache file (auto-generated if None).
        force: If True, overwrite existing eimage_sta features.
    """
    cutoff_hz: float = 100.0
    filter_order: int = 2
    pre_samples: int = 10
    post_samples: int = 40
    spike_limit: int = 10000
    duration_s: Optional[float] = None
    skip_highpass: bool = False
    use_cache: bool = False
    cache_path: Optional[Path] = None
    force: bool = False


@dataclass
class EImageSTAResult:
    """Result of eimage_sta computation.
    
    Attributes:
        hdf5_path: Path to the HDF5 file processed.
        cmcr_path: Path to the CMCR file used.
        units_processed: Number of units successfully processed.
        units_failed: Number of units that failed.
        units_skipped: Number of units skipped (already existed).
        elapsed_seconds: Total computation time.
        filter_time_seconds: Time spent on filtering.
        config: Configuration used for computation.
        warnings: List of warning messages generated.
        failed_units: List of unit IDs that failed.
    """
    hdf5_path: Path
    cmcr_path: Path
    units_processed: int
    units_failed: int
    units_skipped: int
    elapsed_seconds: float
    filter_time_seconds: float
    config: EImageSTAConfig
    warnings: List[str] = field(default_factory=list)
    failed_units: List[str] = field(default_factory=list)


@dataclass
class STAAccumulator:
    """Accumulates STA sum and count across chunks for memory-efficient processing.
    
    This class enables computing STA from data that is too large to fit in memory
    by maintaining a running sum and count that can be updated chunk by chunk.
    
    Attributes:
        sum_array: Running sum of spike-triggered windows (window_len, rows, cols).
        spike_count: Total number of spikes used so far.
        excluded_count: Total number of spikes excluded due to edge effects.
        window_length: Length of the STA window in samples.
        n_rows: Number of electrode rows.
        n_cols: Number of electrode columns.
    """
    sum_array: np.ndarray
    spike_count: int = 0
    excluded_count: int = 0
    window_length: int = 0
    n_rows: int = 0
    n_cols: int = 0
    
    @classmethod
    def create(cls, window_length: int, n_rows: int, n_cols: int) -> "STAAccumulator":
        """Create a new accumulator with zero-initialized sum array."""
        return cls(
            sum_array=np.zeros((window_length, n_rows, n_cols), dtype=np.float64),
            spike_count=0,
            excluded_count=0,
            window_length=window_length,
            n_rows=n_rows,
            n_cols=n_cols,
        )
    
    def add_windows(self, windows: np.ndarray, n_excluded: int = 0) -> None:
        """Add spike windows from current chunk to running sum.
        
        Args:
            windows: 4D array (n_spikes, window_len, rows, cols) of extracted windows.
            n_excluded: Number of spikes excluded from this chunk due to edge effects.
        """
        if windows.shape[0] > 0:
            # Sum windows and add to running sum (use float64 for precision)
            self.sum_array += windows.sum(axis=0).astype(np.float64)
            self.spike_count += windows.shape[0]
        self.excluded_count += n_excluded
    
    def finalize(self) -> np.ndarray:
        """Return final STA = sum / count.
        
        Returns:
            STA array as float32. Returns NaN-filled array if no spikes were accumulated.
        """
        if self.spike_count == 0:
            logger.warning("No spikes accumulated for STA, returning NaN array")
            return np.full(
                (self.window_length, self.n_rows, self.n_cols),
                np.nan,
                dtype=np.float32
            )
        return (self.sum_array / self.spike_count).astype(np.float32)


def compute_sta_for_unit(
    filtered_data: np.ndarray,
    spike_samples: np.ndarray,
    pre_samples: int = 10,
    post_samples: int = 40,
    spike_limit: int = -1,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute eimage STA for a single unit using vectorized window extraction.
    
    Uses NumPy fancy indexing to extract all spike windows in a single
    operation, providing significant speedup over per-spike loops.
    
    Args:
        filtered_data: 3D array (time, rows, cols) of filtered sensor data.
        spike_samples: 1D array of spike times as sample indices.
        pre_samples: Samples before spike in window.
        post_samples: Samples after spike in window.
        spike_limit: Max spikes to use (-1 for no limit).
    
    Returns:
        Tuple of (sta_array, n_spikes_used, n_spikes_excluded).
        sta_array has shape (window_length, rows, cols) as float32.
        Returns NaN-filled array if no valid spikes.
    """
    window_length = pre_samples + post_samples
    n_samples = filtered_data.shape[0]
    n_rows = filtered_data.shape[1]
    n_cols = filtered_data.shape[2]
    
    logger.debug(
        f"STA compute input - Filtered data shape: {filtered_data.shape}, dtype: {filtered_data.dtype}, "
        f"n_spikes: {len(spike_samples)}, window_length: {window_length} "
        f"(pre={pre_samples}, post={post_samples})"
    )
    
    # Apply spike limit if specified
    if spike_limit > 0 and len(spike_samples) > spike_limit:
        spike_samples = spike_samples[:spike_limit]
    
    # Filter valid spikes (edge handling)
    # Spikes must have enough samples before and after
    valid_mask = (
        (spike_samples >= pre_samples) &
        (spike_samples + post_samples <= n_samples)
    )
    
    valid_spikes = spike_samples[valid_mask]
    n_used = len(valid_spikes)
    n_excluded = len(spike_samples) - n_used
    
    # Handle case of no valid spikes
    if n_used == 0:
        logger.warning("No valid spikes for STA computation (all excluded by edge effects)")
        sta = np.full(
            (window_length, n_rows, n_cols),
            np.nan,
            dtype=np.float32
        )
        return sta, 0, n_excluded
    
    # Build all window indices at once: (n_spikes, window_length)
    window_offsets = np.arange(-pre_samples, post_samples)
    all_indices = valid_spikes[:, np.newaxis] + window_offsets  # (n_spikes, window_length)
    
    # Extract all windows using fancy indexing: (n_spikes, window_length, rows, cols)
    windows = filtered_data[all_indices]
    
    # Compute STA as mean across spikes (axis=0)
    sta = windows.mean(axis=0).astype(np.float32)
    
    return sta, n_used, n_excluded


def compute_sta_for_unit_legacy(
    filtered_data: np.ndarray,
    spike_samples: np.ndarray,
    pre_samples: int = 10,
    post_samples: int = 40,
    spike_limit: int = -1,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute eimage STA using the EXACT legacy algorithm.
    
    This function replicates the legacy behavior precisely, including:
    1. Strict boundary conditions: spike - pre > 0 AND spike + post < n_samples
       (Note: Legacy uses STRICT inequalities, excluding more edge spikes)
    2. Division by TOTAL spike count (not just valid spikes)
       (Note: This is a bug in legacy code but we replicate it for consistency)
    
    Legacy code reference (load_raw_data.py lines 350-360):
        def calculate_network_sta(spike_time_stamp, sensor_data, pre_spike_num=10, post_spike_num=40):
            field_sta = np.zeros((pre_spike_num + post_spike_num, ...))
            for time_stamp in spike_time_stamp:
                if time_stamp - pre_spike_num > 0 and time_stamp + post_spike_num < sensor_data.shape[0]:
                    field_sta += sensor_data[time_stamp - pre_spike_num:time_stamp + post_spike_num, :, :]
            field_sta = field_sta / len(spike_time_stamp)  # <-- divides by TOTAL, not valid
            return field_sta
    
    Args:
        filtered_data: 3D array (time, rows, cols) of filtered sensor data.
        spike_samples: 1D array of spike times as sample indices.
        pre_samples: Samples before spike in window.
        post_samples: Samples after spike in window.
        spike_limit: Max spikes to use (-1 for no limit).
    
    Returns:
        Tuple of (sta_array, n_spikes_used, n_spikes_excluded).
        sta_array has shape (window_length, rows, cols) as float32.
    """
    window_length = pre_samples + post_samples
    n_samples = filtered_data.shape[0]
    n_rows = filtered_data.shape[1]
    n_cols = filtered_data.shape[2]
    
    # Apply spike limit if specified (BEFORE counting total)
    if spike_limit > 0 and len(spike_samples) > spike_limit:
        spike_samples = spike_samples[:spike_limit]
    
    # Total spike count for division (legacy behavior: divide by total, not valid)
    total_spike_count = len(spike_samples)
    
    logger.debug(
        f"Legacy STA compute - Filtered data shape: {filtered_data.shape}, "
        f"total_spikes: {total_spike_count}, window_length: {window_length} "
        f"(pre={pre_samples}, post={post_samples})"
    )
    
    # Legacy boundary conditions use STRICT inequalities:
    # time_stamp - pre_spike_num > 0  -->  spike > pre_samples  (NOT >=)
    # time_stamp + post_spike_num < n_samples  -->  spike + post < n_samples  (NOT <=)
    valid_mask = (
        (spike_samples - pre_samples > 0) &  # Legacy: time_stamp - pre > 0
        (spike_samples + post_samples < n_samples)  # Legacy: time_stamp + post < n_samples
    )
    
    valid_spikes = spike_samples[valid_mask]
    n_used = len(valid_spikes)
    n_excluded = total_spike_count - n_used
    
    # Handle case of no valid spikes
    if n_used == 0:
        logger.warning("No valid spikes for STA computation (all excluded by edge effects)")
        sta = np.full(
            (window_length, n_rows, n_cols),
            np.nan,
            dtype=np.float32
        )
        return sta, 0, n_excluded
    
    # Build all window indices at once: (n_spikes, window_length)
    window_offsets = np.arange(-pre_samples, post_samples)
    all_indices = valid_spikes[:, np.newaxis] + window_offsets
    
    # Extract all windows and sum them
    windows = filtered_data[all_indices]  # (n_valid_spikes, window_length, rows, cols)
    windows_sum = windows.sum(axis=0)  # (window_length, rows, cols)
    
    # LEGACY BEHAVIOR: Divide by TOTAL spike count, not valid spike count
    # This is technically a bug in the legacy code but we replicate it for consistency
    sta = (windows_sum / total_spike_count).astype(np.float32)
    
    logger.debug(
        f"Legacy STA result - valid_spikes: {n_used}, excluded: {n_excluded}, "
        f"dividing by total: {total_spike_count}"
    )
    
    return sta, n_used, n_excluded


def extract_windows_for_chunk(
    chunk_data: np.ndarray,
    spike_samples: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    pre_samples: int = 10,
    post_samples: int = 40,
) -> Tuple[np.ndarray, int, int]:
    """
    Extract spike windows from a single chunk of data.
    
    Handles spikes that fall within this chunk and have enough margin
    for the full window.
    
    Args:
        chunk_data: 3D array (time, rows, cols) of filtered chunk data.
        spike_samples: 1D array of spike times as GLOBAL sample indices.
        chunk_start: Global sample index where this chunk starts.
        chunk_end: Global sample index where this chunk ends (exclusive).
        pre_samples: Samples before spike in window.
        post_samples: Samples after spike in window.
    
    Returns:
        Tuple of (windows, n_used, n_excluded):
            - windows: 4D array (n_spikes, window_len, rows, cols)
            - n_used: Number of spikes used
            - n_excluded: Number of spikes in chunk but excluded due to edges
    """
    chunk_length = chunk_data.shape[0]
    window_length = pre_samples + post_samples
    
    # Find spikes within this chunk's valid window range
    # Spike must be in [chunk_start + pre_samples, chunk_end - post_samples)
    # So that window [spike - pre_samples, spike + post_samples) fits in chunk
    valid_start = chunk_start + pre_samples
    valid_end = chunk_end - post_samples
    
    # Spikes in chunk range (including those that might be edge-excluded)
    in_chunk_mask = (spike_samples >= chunk_start) & (spike_samples < chunk_end)
    spikes_in_chunk = spike_samples[in_chunk_mask]
    
    # Valid spikes (have enough margin for full window)
    valid_mask = (spikes_in_chunk >= valid_start) & (spikes_in_chunk < valid_end)
    valid_spikes = spikes_in_chunk[valid_mask]
    
    n_used = len(valid_spikes)
    n_excluded = len(spikes_in_chunk) - n_used
    
    if n_used == 0:
        # Return empty array with correct shape
        return np.empty((0, window_length, chunk_data.shape[1], chunk_data.shape[2]), dtype=np.float32), 0, n_excluded
    
    # Convert global spike indices to local chunk indices
    local_spikes = valid_spikes - chunk_start
    
    # Build window indices: (n_spikes, window_length)
    window_offsets = np.arange(-pre_samples, post_samples)
    all_indices = local_spikes[:, np.newaxis] + window_offsets
    
    # Extract windows: (n_spikes, window_length, rows, cols)
    windows = chunk_data[all_indices].astype(np.float32)
    
    return windows, n_used, n_excluded


def _process_chunk(
    chunk_data: np.ndarray,
    skip_highpass: bool,
    cutoff_hz: float,
    sampling_rate: float,
    filter_order: int,
) -> np.ndarray:
    """
    Process a single chunk: either high-pass filter or mean-center.
    
    Args:
        chunk_data: Raw chunk data (time, rows, cols) as int16.
        skip_highpass: If True, mean-center instead of filtering.
        cutoff_hz: High-pass filter cutoff.
        sampling_rate: Acquisition rate.
        filter_order: Filter order.
    
    Returns:
        Processed chunk as float32.
    """
    logger.debug(f"_process_chunk: input shape={chunk_data.shape}, dtype={chunk_data.dtype}, skip_highpass={skip_highpass}")
    
    if skip_highpass:
        # Mean-center each electrode
        logger.debug("_process_chunk: Converting to float32...")
        t0 = time.time()
        chunk_float = chunk_data.astype(np.float32)
        logger.debug(f"_process_chunk: Float conversion done in {time.time()-t0:.2f}s")
        
        logger.debug("_process_chunk: Computing electrode means...")
        t0 = time.time()
        electrode_means = chunk_float.mean(axis=0, keepdims=True)
        logger.debug(f"_process_chunk: Means computed in {time.time()-t0:.2f}s")
        
        logger.debug("_process_chunk: Subtracting means...")
        t0 = time.time()
        result = (chunk_float - electrode_means).astype(np.float32)
        logger.debug(f"_process_chunk: Subtraction done in {time.time()-t0:.2f}s")
        
        return result
    else:
        # Apply high-pass filter
        logger.debug("_process_chunk: Applying high-pass filter...")
        return apply_highpass_filter_3d(
            chunk_data,
            cutoff_hz=cutoff_hz,
            sampling_rate=sampling_rate,
            filter_order=filter_order,
            show_progress=False,  # Don't show nested progress
        )


def write_eimage_sta_to_hdf5(
    hdf5_file: h5py.File,
    unit_id: str,
    sta: np.ndarray,
    n_spikes: int,
    n_spikes_excluded: int,
    config: EImageSTAConfig,
    sampling_rate: float,
    force: bool = False,
) -> bool:
    """
    Write eimage_sta to HDF5 file.
    
    Creates group structure: units/{unit_id}/features/eimage_sta/
    
    Args:
        hdf5_file: Open HDF5 file in write mode.
        unit_id: Unit identifier.
        sta: Computed STA array.
        n_spikes: Number of spikes used in average.
        n_spikes_excluded: Number of spikes excluded due to edge effects.
        config: Configuration used for computation.
        sampling_rate: Acquisition rate in Hz.
        force: If True, overwrite existing. If False, skip if exists.
    
    Returns:
        True if written, False if skipped (already exists and force=False).
    """
    unit_group = hdf5_file[f"units/{unit_id}"]
    
    # Create features group if not exists
    if "features" not in unit_group:
        unit_group.create_group("features")
    
    features_group = unit_group["features"]
    
    # Check if eimage_sta already exists
    if "eimage_sta" in features_group:
        if force:
            del features_group["eimage_sta"]
            logger.debug(f"Overwriting existing eimage_sta for {unit_id}")
        else:
            logger.debug(f"eimage_sta already exists for {unit_id}, skipping")
            return False
    
    # Create eimage_sta group
    eimage_group = features_group.create_group("eimage_sta")
    
    # Write STA data
    eimage_group.create_dataset("data", data=sta, dtype=np.float32)
    
    # Write metadata as attributes
    eimage_group.attrs["n_spikes"] = n_spikes
    eimage_group.attrs["n_spikes_excluded"] = n_spikes_excluded
    eimage_group.attrs["pre_samples"] = config.pre_samples
    eimage_group.attrs["post_samples"] = config.post_samples
    eimage_group.attrs["cutoff_hz"] = config.cutoff_hz
    eimage_group.attrs["filter_order"] = config.filter_order
    eimage_group.attrs["sampling_rate"] = sampling_rate
    eimage_group.attrs["spike_limit"] = config.spike_limit
    eimage_group.attrs["version"] = VERSION
    
    logger.debug(
        f"Wrote eimage_sta for {unit_id}: "
        f"shape={sta.shape}, n_spikes={n_spikes}"
    )
    
    return True


def _get_sampling_rate(hdf5_file: h5py.File) -> float:
    """Get sampling rate from HDF5 metadata."""
    # Try metadata group with acquisition_rate dataset (standard format)
    if "metadata" in hdf5_file:
        meta = hdf5_file["metadata"]
        if "acquisition_rate" in meta:
            # Access as dataset and get scalar value
            return float(np.array(meta["acquisition_rate"]).flat[0])
        if "acquisition_rate" in meta.attrs:
            return float(meta.attrs["acquisition_rate"])
    
    # Try root-level attribute
    if "acquisition_rate" in hdf5_file.attrs:
        return float(hdf5_file.attrs["acquisition_rate"])
    
    # Default to 20kHz
    logger.warning("Could not find acquisition_rate in HDF5, using default 20000 Hz")
    return 20000.0


def compute_eimage_sta(
    hdf5_path: Union[str, Path],
    cmcr_path: Union[str, Path],
    *,
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    pre_samples: int = 10,
    post_samples: int = 40,
    spike_limit: int = 10000,
    duration_s: Optional[float] = None,
    skip_highpass: bool = False,
    use_cache: bool = False,
    cache_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> EImageSTAResult:
    """
    Compute Electrode Image STA for all units using sensor data from CMCR.
    
    This is the main entry point for eimage_sta computation. It:
    1. Loads sensor data from CMCR file
    2. Applies vectorized high-pass filtering
    3. Computes STA for each unit using vectorized window extraction
    4. Writes results to HDF5 file
    
    Args:
        hdf5_path: Path to HDF5 file containing units with spike_times.
        cmcr_path: Path to CMCR file containing sensor data.
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        pre_samples: Number of samples before spike in window.
        post_samples: Number of samples after spike in window.
        spike_limit: Maximum spikes to use per unit (-1 for no limit).
        duration_s: Duration of sensor data to process in seconds (None for all data).
        skip_highpass: If True, skip high-pass filter and mean-center data instead.
        use_cache: If True, cache filtered data for reuse.
        cache_path: Path to cache file (auto-generated if None and use_cache=True).
        force: If True, overwrite existing eimage_sta features.
    
    Returns:
        EImageSTAResult with processing summary.
    
    Raises:
        FileNotFoundError: If hdf5_path or cmcr_path does not exist.
        ValueError: If no units found in HDF5 file.
        ValueError: If invalid filter parameters.
        DataLoadError: If sensor data cannot be loaded from CMCR.
    """
    start_time = time.time()
    
    hdf5_path = Path(hdf5_path)
    cmcr_path = Path(cmcr_path)
    
    # Build config
    config = EImageSTAConfig(
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        pre_samples=pre_samples,
        post_samples=post_samples,
        spike_limit=spike_limit,
        duration_s=duration_s,
        skip_highpass=skip_highpass,
        use_cache=use_cache,
        cache_path=Path(cache_path) if cache_path else None,
        force=force,
    )
    
    warnings_list: List[str] = []
    failed_units: List[str] = []
    
    # Validate filter parameters (only if not skipping)
    if not skip_highpass:
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
        if filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {filter_order}")
    
    # Check files exist
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    # Get sampling rate from HDF5
    with h5py.File(hdf5_path, "r") as f:
        sampling_rate = _get_sampling_rate(f)
    
    # Validate cutoff vs Nyquist (only if not skipping high-pass filter)
    if not skip_highpass:
        nyquist = 0.5 * sampling_rate
        if cutoff_hz >= nyquist:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be less than Nyquist frequency ({nyquist})"
            )
    
    # Calculate expected samples (None means use all available data)
    duration_samples = int(duration_s * sampling_rate) if duration_s is not None else None
    
    # Try to load from cache if enabled (only when using high-pass filter)
    filtered_data = None
    filter_time = 0.0
    
    if use_cache and not skip_highpass:
        cache_file = _get_cache_path(cmcr_path, config.cache_path)
        filtered_data = _load_cached_filtered_data(
            cache_file,
            cutoff_hz=cutoff_hz,
            filter_order=filter_order,
            duration_samples=duration_samples,
        )
        if filtered_data is not None:
            logger.info("Using cached filtered data (skipping filtering)")
    
    # If no cache, load and process data
    if filtered_data is None:
        # Load sensor data
        logger.info(f"Loading sensor data from {cmcr_path}...")
        sensor_data = load_sensor_data(cmcr_path, duration_s=duration_s)
        
        if skip_highpass:
            # Mean-center each electrode along time axis (subtract mean per electrode)
            logger.info("Skipping high-pass filter, applying mean-centering...")
            filter_start = time.time()
            # Compute mean along time axis (axis=0) and subtract
            sensor_data_float = sensor_data.astype(np.float32)
            electrode_means = sensor_data_float.mean(axis=0, keepdims=True)
            filtered_data = (sensor_data_float - electrode_means).astype(np.float32)
            filter_time = time.time() - filter_start
            logger.info(f"Mean-centering complete in {filter_time:.1f}s")
            del sensor_data_float
        else:
            # Apply high-pass filter
            logger.info(f"Applying high-pass filter (cutoff={cutoff_hz}Hz, order={filter_order})...")
            filter_start = time.time()
            filtered_data = apply_highpass_filter_3d(
                sensor_data,
                cutoff_hz=cutoff_hz,
                sampling_rate=sampling_rate,
                filter_order=filter_order,
            )
            filter_time = time.time() - filter_start
            logger.info(f"Filtering complete in {filter_time:.1f}s")
        
        # Free original sensor data memory
        del sensor_data
        
        # Save to cache if enabled (only for high-pass filtered data)
        if use_cache and not skip_highpass:
            cache_file = _get_cache_path(cmcr_path, config.cache_path)
            _save_filtered_data_cache(
                cache_file,
                filtered_data,
                cutoff_hz=cutoff_hz,
                filter_order=filter_order,
                sampling_rate=sampling_rate,
            )
    
    # Process each unit
    units_processed = 0
    units_skipped = 0
    
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        # Get list of units
        if "units" not in hdf5_file:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        unit_ids = list(hdf5_file["units"].keys())
        if not unit_ids:
            raise ValueError(f"No units found in {hdf5_path}")
        
        logger.info(f"Processing {len(unit_ids)} units...")
        
        for unit_id in tqdm(unit_ids, desc="Computing eimage_sta"):
            try:
                unit_group = hdf5_file[f"units/{unit_id}"]
                
                # Get spike times
                if "spike_times" not in unit_group:
                    warnings_list.append(f"No spike_times for {unit_id}")
                    continue
                
                spike_times_us = unit_group["spike_times"][:]
                
                # Convert microseconds to sample indices
                spike_samples = (spike_times_us / 1e6 * sampling_rate).astype(np.int64)
                
                # Compute STA
                sta, n_used, n_excluded = compute_sta_for_unit(
                    filtered_data,
                    spike_samples,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    spike_limit=spike_limit,
                )
                
                # Write to HDF5
                written = write_eimage_sta_to_hdf5(
                    hdf5_file,
                    unit_id,
                    sta,
                    n_used,
                    n_excluded,
                    config,
                    sampling_rate,
                    force=force,
                )
                
                if written:
                    units_processed += 1
                else:
                    units_skipped += 1
                    
            except Exception as e:
                logger.error(f"Failed to process unit {unit_id}: {e}")
                failed_units.append(unit_id)
    
    elapsed = time.time() - start_time
    
    result = EImageSTAResult(
        hdf5_path=hdf5_path,
        cmcr_path=cmcr_path,
        units_processed=units_processed,
        units_failed=len(failed_units),
        units_skipped=units_skipped,
        elapsed_seconds=elapsed,
        filter_time_seconds=filter_time,
        config=config,
        warnings=warnings_list,
        failed_units=failed_units,
    )
    
    logger.info(
        f"eimage_sta computation complete: {units_processed} processed, "
        f"{units_skipped} skipped, {len(failed_units)} failed, "
        f"{elapsed:.1f}s elapsed"
    )
    
    return result


def compute_eimage_sta_chunked(
    hdf5_path: Union[str, Path],
    cmcr_path: Union[str, Path],
    *,
    chunk_duration_s: float = 60.0,
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    window_range: Tuple[int, int] = (-10, 40),
    duration_s: float = 120.0,
    spike_limit: int = 10000,
    unit_ids: Optional[List[str]] = None,
    skip_highpass: bool = False,
    force: bool = False,
) -> EImageSTAResult:
    """
    Compute Electrode Image STA using chunk-based processing for large files.
    
    This is the memory-efficient version that processes sensor data in chunks,
    enabling processing of 100GB+ CMCR files with limited RAM. Uses streaming
    I/O and accumulates STA incrementally.
    
    Pipeline:
    1. Load spike times for all units upfront
    2. Initialize STA accumulators for each unit
    3. For each chunk:
       a. Load chunk from CMCR
       b. Filter/mean-center chunk
       c. For each unit, extract windows and accumulate
       d. Free chunk memory
    4. Finalize STAs (divide sum by count)
    5. Write results to HDF5
    
    Args:
        hdf5_path: Path to HDF5 file containing units with spike_times.
        cmcr_path: Path to CMCR file containing sensor data.
        chunk_duration_s: Duration of each processing chunk in seconds.
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        window_range: Tuple (start, end) relative to spike in samples, e.g. (-10, 40).
        duration_s: Total duration to analyze in seconds (default 120s, -1 for all).
        spike_limit: Maximum spikes to use per unit (default 10000, -1 for no limit).
        unit_ids: List of unit IDs to process (default None = all units).
        skip_highpass: If True, skip high-pass filter and mean-center instead.
        force: If True, overwrite existing eimage_sta features.
    
    Returns:
        EImageSTAResult with processing summary.
    
    Raises:
        FileNotFoundError: If hdf5_path or cmcr_path does not exist.
        ValueError: If no units found in HDF5 file.
        DataLoadError: If sensor data cannot be loaded from CMCR.
    """
    start_time = time.time()
    
    hdf5_path = Path(hdf5_path)
    cmcr_path = Path(cmcr_path)
    
    # Parse window_range to pre_samples and post_samples
    # window_range = (start, end) relative to spike, e.g. (-10, 40)
    # pre_samples = samples before spike = -start (e.g., 10)
    # post_samples = samples after spike = end (e.g., 40)
    pre_samples = -window_range[0]
    post_samples = window_range[1]
    window_length = pre_samples + post_samples
    
    if pre_samples < 0 or post_samples < 0:
        raise ValueError(
            f"Invalid window_range {window_range}: start must be <= 0 and end must be >= 0"
        )
    
    logger.info(f"Window range: {window_range} -> pre_samples={pre_samples}, post_samples={post_samples}, window_length={window_length}")
    
    # Build config
    config = EImageSTAConfig(
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        pre_samples=pre_samples,
        post_samples=post_samples,
        spike_limit=spike_limit,
        duration_s=duration_s if duration_s > 0 else None,
        skip_highpass=skip_highpass,
        use_cache=False,  # Cache not used in chunked mode
        cache_path=None,
        force=force,
    )
    warnings_list: List[str] = []
    failed_units: List[str] = []
    filter_time = 0.0
    
    # Validate filter parameters
    if not skip_highpass:
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
        if filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {filter_order}")
    
    # Check files exist
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    # Get sensor data info without loading
    sensor_info = get_sensor_data_info(cmcr_path)
    n_rows = sensor_info["n_rows"]
    n_cols = sensor_info["n_cols"]
    total_samples_available = sensor_info["n_samples"]
    
    # Get sampling rate from HDF5
    with h5py.File(hdf5_path, "r") as f:
        sampling_rate = _get_sampling_rate(f)
    
    # Limit total samples based on duration_s
    if duration_s > 0:
        max_samples = int(duration_s * sampling_rate)
        total_samples = min(max_samples, total_samples_available)
        logger.info(
            f"Limiting analysis to {duration_s:.1f}s ({total_samples} samples) "
            f"out of {total_samples_available} available ({sensor_info['duration_s']:.1f}s)"
        )
    else:
        total_samples = total_samples_available
        logger.info(
            f"Analyzing full recording: {total_samples} samples ({sensor_info['duration_s']:.1f}s), "
            f"{n_rows}x{n_cols} electrodes"
        )
    
    # Validate cutoff vs Nyquist
    if not skip_highpass:
        nyquist = 0.5 * sampling_rate
        if cutoff_hz >= nyquist:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be less than Nyquist frequency ({nyquist})"
            )
    
    # =========================================================================
    # PHASE 1: Load all spike times and initialize accumulators
    # =========================================================================
    logger.info("Phase 1: Loading spike times and initializing accumulators...")
    
    unit_spike_samples: Dict[str, np.ndarray] = {}
    accumulators: Dict[str, STAAccumulator] = {}
    
    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "units" not in hdf5_file:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        available_units = list(hdf5_file["units"].keys())
        if not available_units:
            raise ValueError(f"No units found in {hdf5_path}")
        
        # Use specified unit_ids or all available units
        if unit_ids is not None:
            # Filter to only requested units that exist
            units_to_process = [u for u in unit_ids if u in available_units]
            missing = set(unit_ids) - set(units_to_process)
            if missing:
                warnings_list.append(f"Requested units not found: {missing}")
            logger.info(f"Processing {len(units_to_process)} specified units (of {len(available_units)} available)")
        else:
            units_to_process = available_units
            logger.info(f"Processing all {len(units_to_process)} units")
        
        for unit_id in units_to_process:
            unit_group = hdf5_file[f"units/{unit_id}"]
            
            if "spike_times" not in unit_group:
                warnings_list.append(f"No spike_times for {unit_id}")
                continue
            
            # Load spike times (microseconds) and convert to samples
            spike_times_us = unit_group["spike_times"][:]
            spike_samples = (spike_times_us / 1e6 * sampling_rate).astype(np.int64)
            
            # Filter spikes to only include those within duration limit
            spike_samples = spike_samples[spike_samples < total_samples]
            
            # Apply spike limit if specified
            if spike_limit > 0 and len(spike_samples) > spike_limit:
                spike_samples = spike_samples[:spike_limit]
            
            unit_spike_samples[unit_id] = spike_samples
            accumulators[unit_id] = STAAccumulator.create(window_length, n_rows, n_cols)
    
    total_spikes_loaded = sum(len(s) for s in unit_spike_samples.values())
    logger.info(
        f"Loaded spike times for {len(unit_spike_samples)} units "
        f"({total_spikes_loaded} total spikes, spike_limit={spike_limit})"
    )
    
    # =========================================================================
    # PHASE 2: Process chunks and accumulate STAs
    # =========================================================================
    logger.info("Phase 2: Processing chunks...")
    
    chunk_samples = int(chunk_duration_s * sampling_rate)
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    chunk_generator = load_sensor_data_chunked(
        cmcr_path,
        chunk_duration_s=chunk_duration_s,
        sampling_rate=sampling_rate,
        max_samples=total_samples,
    )
    
    # Wrap generator in tqdm for progress
    chunk_idx = 0
    for chunk_data, chunk_start, chunk_end, _ in tqdm(
        chunk_generator,
        total=n_chunks,
        desc="Processing chunks",
        unit="chunk",
    ):
        chunk_idx += 1
        chunk_loop_start = time.time()
        
        logger.debug(f"Chunk {chunk_idx}/{n_chunks}: Starting processing...")
        
        # Step 1: Process chunk (filter or mean-center)
        logger.debug(f"Chunk {chunk_idx}: Step 1 - Starting filter/mean-center...")
        step_start = time.time()
        processed_chunk = _process_chunk(
            chunk_data,
            skip_highpass=skip_highpass,
            cutoff_hz=cutoff_hz,
            sampling_rate=sampling_rate,
            filter_order=filter_order,
        )
        step_filter_time = time.time() - step_start
        filter_time += step_filter_time
        logger.debug(f"Chunk {chunk_idx}: Step 1 - Filter complete in {step_filter_time:.2f}s")
        
        # Free raw chunk memory
        del chunk_data
        
        # Step 2: Accumulate STAs for each unit
        logger.debug(f"Chunk {chunk_idx}: Step 2 - Starting STA accumulation for {len(unit_spike_samples)} units...")
        step_start = time.time()
        total_windows_extracted = 0
        total_spikes_in_chunk = 0
        units_with_spikes = 0
        
        for unit_idx, (unit_id, spike_samples) in enumerate(unit_spike_samples.items()):
            if unit_idx % 50 == 0:
                logger.debug(f"Chunk {chunk_idx}: Processing unit {unit_idx+1}/{len(unit_spike_samples)}...")
            
            windows, n_used, n_excluded = extract_windows_for_chunk(
                processed_chunk,
                spike_samples,
                chunk_start,
                chunk_end,
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            
            total_windows_extracted += n_used
            total_spikes_in_chunk += n_used + n_excluded
            
            if n_used > 0 or n_excluded > 0:
                units_with_spikes += 1
                accumulators[unit_id].add_windows(windows, n_excluded)
            
            # Free windows memory
            del windows
        
        step_accumulate_time = time.time() - step_start
        logger.debug(f"Chunk {chunk_idx}: Step 2 - Accumulation complete in {step_accumulate_time:.2f}s")
        
        # Free processed chunk memory
        del processed_chunk
        
        chunk_total_time = time.time() - chunk_loop_start
        
        logger.debug(
            f"Chunk {chunk_idx}/{n_chunks} [{chunk_start}:{chunk_end}] COMPLETE - "
            f"Total: {chunk_total_time:.2f}s | "
            f"Filter: {step_filter_time:.2f}s | "
            f"Accumulate: {step_accumulate_time:.2f}s | "
            f"Units w/spikes: {units_with_spikes}, Windows: {total_windows_extracted}"
        )
    
    # =========================================================================
    # PHASE 3: Finalize and write results
    # =========================================================================
    logger.info("Phase 3: Finalizing and writing results...")
    
    units_processed = 0
    units_skipped = 0
    
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        for unit_id in tqdm(accumulators.keys(), desc="Writing results", unit="unit"):
            try:
                acc = accumulators[unit_id]
                sta = acc.finalize()
                
                written = write_eimage_sta_to_hdf5(
                    hdf5_file,
                    unit_id,
                    sta,
                    acc.spike_count,
                    acc.excluded_count,
                    config,
                    sampling_rate,
                    force=force,
                )
                
                if written:
                    units_processed += 1
                else:
                    units_skipped += 1
                    
            except Exception as e:
                logger.error(f"Failed to process unit {unit_id}: {e}")
                failed_units.append(unit_id)
    
    elapsed = time.time() - start_time
    
    result = EImageSTAResult(
        hdf5_path=hdf5_path,
        cmcr_path=cmcr_path,
        units_processed=units_processed,
        units_failed=len(failed_units),
        units_skipped=units_skipped,
        elapsed_seconds=elapsed,
        filter_time_seconds=filter_time,
        config=config,
        warnings=warnings_list,
        failed_units=failed_units,
    )
    
    logger.info(
        f"Chunked eimage_sta complete: {units_processed} processed, "
        f"{units_skipped} skipped, {len(failed_units)} failed, "
        f"{elapsed:.1f}s total, {filter_time:.1f}s filtering"
    )
    
    return result


def compute_eimage_sta_legacy(
    hdf5_path: Union[str, Path],
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
    *,
    duration_s: float = 120.0,
    spike_limit: int = 10000,
    unit_ids: Optional[List[str]] = None,
    window_range: Tuple[int, int] = (-10, 40),
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    skip_highpass: bool = False,
    force: bool = False,
) -> EImageSTAResult:
    """
    Legacy-style STA computation using EXACT legacy approach.
    
    This function replicates the legacy approach precisely:
    1. Load sensor data from CMCR using McsPy dictionary access
    2. Load spike times from CMTR using McsPy (NOT from HDF5)
    3. Apply filter using legacy ba/filtfilt method
    4. Compute STA with legacy boundary conditions and division
    5. Write results to HDF5
    
    Args:
        hdf5_path: Path to HDF5 file for storing results.
        cmcr_path: Path to CMCR file containing sensor data.
        cmtr_path: Path to CMTR file containing spike times (legacy source).
        duration_s: Total duration to analyze in seconds (default 120s).
        spike_limit: Maximum spikes to use per unit (default 10000).
        unit_ids: List of unit IDs to process (default None = all units).
            Note: unit_ids should match legacy format, e.g. ["1", "2"] not ["unit_001"].
        window_range: Tuple (start, end) relative to spike in samples, e.g. (-10, 40).
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        skip_highpass: If True, skip high-pass filter and mean-center instead.
        force: If True, overwrite existing eimage_sta features.
    
    Returns:
        EImageSTAResult with processing summary.
    
    Raises:
        FileNotFoundError: If any input file does not exist.
        ImportError: If McsPy is not available.
        ValueError: If no units found.
    
    Example:
        >>> result = compute_eimage_sta_legacy(
        ...     hdf5_path="artifacts/recording.h5",
        ...     cmcr_path="O:/data/recording.cmcr",
        ...     cmtr_path="O:/data/recording.cmtr",
        ...     duration_s=120.0,
        ...     spike_limit=10000,
        ...     unit_ids=["1", "2", "3"],  # Legacy unit numbers
        ...     window_range=(-10, 40),
        ... )
    """
    start_time = time.time()
    
    hdf5_path = Path(hdf5_path)
    cmcr_path = Path(cmcr_path)
    cmtr_path = Path(cmtr_path)
    
    # Parse window_range to pre_samples and post_samples
    pre_samples = -window_range[0]
    post_samples = window_range[1]
    window_length = pre_samples + post_samples
    
    if pre_samples < 0 or post_samples < 0:
        raise ValueError(
            f"Invalid window_range {window_range}: start must be <= 0 and end must be >= 0"
        )
    
    logger.info(
        f"Legacy STA: window_range={window_range}, "
        f"duration_s={duration_s}, spike_limit={spike_limit}"
    )
    
    # Build config
    config = EImageSTAConfig(
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        pre_samples=pre_samples,
        post_samples=post_samples,
        spike_limit=spike_limit,
        duration_s=duration_s,
        skip_highpass=skip_highpass,
        use_cache=False,
        cache_path=None,
        force=force,
    )
    
    warnings_list: List[str] = []
    failed_units: List[str] = []
    
    # Validate filter parameters
    if not skip_highpass:
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
        if filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {filter_order}")
    
    # Check files exist
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    if not cmtr_path.exists():
        raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
    
    # Get sampling rate from HDF5
    with h5py.File(hdf5_path, "r") as f:
        sampling_rate = _get_sampling_rate(f)
    
    # Validate cutoff vs Nyquist
    if not skip_highpass:
        nyquist = 0.5 * sampling_rate
        if cutoff_hz >= nyquist:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be less than Nyquist frequency ({nyquist})"
            )
    
    # =========================================================================
    # STEP 1: Load sensor data using LEGACY McsPy approach
    # =========================================================================
    logger.info(f"Step 1: Loading sensor data using LEGACY McsPy approach ({duration_s}s)...")
    load_start = time.time()
    
    # Legacy approach: Use McsPy with dictionary-style access
    # Reference: cmcr_data['Acquisition']['Sensor Data']["SensorData 1 1"][:int(acq_rate * 120),:,:]
    try:
        from McsPy import McsCMOSMEA
        cmcr_data = McsCMOSMEA.McsData(str(cmcr_path))
        
        # Calculate number of samples to load
        n_samples_to_load = int(sampling_rate * duration_s)
        
        # Access using legacy dictionary-style path
        sensor_data = cmcr_data['Acquisition']['Sensor Data']["SensorData 1 1"][:n_samples_to_load, :, :]
        sensor_data = np.array(sensor_data).astype(np.int16)
        
        logger.info(f"Legacy McsPy load complete: shape={sensor_data.shape}")
        
    except ImportError:
        raise ImportError(
            "McsPy library not available. Install with: pip install McsPyDataTools"
        )
    except KeyError as e:
        # Try alternative paths if the standard one fails
        logger.warning(f"Standard path failed ({e}), trying alternative paths...")
        try:
            # Try different possible paths
            acq = cmcr_data['Acquisition']
            sensor_streams = acq['Sensor Data']
            
            # Find the sensor data key
            sensor_key = None
            for key in sensor_streams.keys():
                if 'SensorData' in str(key):
                    sensor_key = key
                    break
            
            if sensor_key is None:
                raise KeyError("No SensorData found in CMCR file")
            
            n_samples_to_load = int(sampling_rate * duration_s)
            sensor_data = sensor_streams[sensor_key][:n_samples_to_load, :, :]
            sensor_data = np.array(sensor_data).astype(np.int16)
            
            logger.info(f"Legacy McsPy load (alt path) complete: shape={sensor_data.shape}")
            
        except Exception as inner_e:
            raise ValueError(f"Failed to load sensor data from CMCR: {inner_e}")
    
    load_time = time.time() - load_start
    logger.info(f"Loaded sensor data: shape={sensor_data.shape}, time={load_time:.1f}s")
    
    n_samples = sensor_data.shape[0]
    n_rows = sensor_data.shape[1]
    n_cols = sensor_data.shape[2]
    
    # =========================================================================
    # STEP 2: Apply filter or mean-center
    # =========================================================================
    filter_start = time.time()
    
    if skip_highpass:
        logger.info("Step 2: Mean-centering sensor data...")
        sensor_data_float = sensor_data.astype(np.float32)
        electrode_means = sensor_data_float.mean(axis=0, keepdims=True)
        filtered_data = (sensor_data_float - electrode_means).astype(np.float32)
        del sensor_data_float
    else:
        # Use LEGACY filter approach: ba format + filtfilt + nested loops
        logger.info(f"Step 2: Applying LEGACY high-pass filter (cutoff={cutoff_hz}Hz)...")
        logger.info("WARNING: Legacy filter is slow (~1 hour for 120s data). Use skip_highpass=True for faster testing.")
        filtered_data = apply_highpass_filter_3d_legacy(
            sensor_data,
            cutoff_hz=cutoff_hz,
            sampling_rate=sampling_rate,
            filter_order=filter_order,
            show_progress=True,
        )
        # Legacy keeps float64 from filtfilt - do NOT convert to float32
    
    filter_time = time.time() - filter_start
    logger.info(f"Filtering complete: time={filter_time:.1f}s")
    
    # Free original sensor data
    del sensor_data
    
    # =========================================================================
    # STEP 3: Load spike data from CMTR using LEGACY approach
    # =========================================================================
    logger.info("Step 3: Loading spike data from CMTR using LEGACY McsPy approach...")
    
    # Legacy approach: Load CMTR using McsPy
    # Reference: cmtr_data['Spike Sorter'][f"Unit {unit_num}"]["Peaks"]
    try:
        from McsPy import McsCMOSMEA
        cmtr_data = McsCMOSMEA.McsData(str(cmtr_path))
        
        # Get available units from CMTR
        spike_sorter = cmtr_data['Spike Sorter']
        
        # Get list of unit keys from CMTR
        cmtr_unit_keys = [key for key in spike_sorter.keys() if key.startswith('Unit')]
        logger.info(f"Found {len(cmtr_unit_keys)} units in CMTR: {cmtr_unit_keys[:5]}...")
        
    except ImportError:
        raise ImportError("McsPy library not available. Install with: pip install McsPyDataTools")
    except KeyError as e:
        raise ValueError(f"Failed to access Spike Sorter in CMTR: {e}")
    
    # =========================================================================
    # STEP 4: Process each unit
    # =========================================================================
    logger.info("Step 4: Computing STA for each unit...")
    
    units_processed = 0
    units_skipped = 0
    
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        # Create units group if not exists (for storing results)
        if "units" not in hdf5_file:
            hdf5_file.create_group("units")
        
        # Determine which units to process
        if unit_ids is not None:
            # User specified unit numbers (e.g., ["1", "2", "3"])
            units_to_process = unit_ids
            logger.info(f"Processing {len(units_to_process)} specified units")
        else:
            # Extract unit numbers from CMTR unit keys (e.g., "Unit 1" -> "1")
            units_to_process = []
            for key in cmtr_unit_keys:
                # Extract number from "Unit X" format
                parts = key.split()
                if len(parts) >= 2:
                    units_to_process.append(parts[1])
            logger.info(f"Processing all {len(units_to_process)} units from CMTR")
        
        for unit_num in tqdm(units_to_process, desc="Computing eimage_sta"):
            try:
                # Legacy format: "Unit {unit_num}"
                cmtr_unit_key = f"Unit {unit_num}"
                
                if cmtr_unit_key not in spike_sorter:
                    warnings_list.append(f"Unit {unit_num} not found in CMTR")
                    continue
                
                # =========================================================
                # LEGACY spike loading from CMTR
                # Reference: spike_data = cmtr_data['Spike Sorter'][f"Unit {unit_num}"]["Peaks"][:int(spike_limit)]
                # =========================================================
                unit_data = spike_sorter[cmtr_unit_key]
                
                if "Peaks" not in unit_data:
                    warnings_list.append(f"No Peaks data for Unit {unit_num}")
                    continue
                
                # Legacy: Apply spike_limit BEFORE loading/conversion
                if spike_limit > 0:
                    spike_data = unit_data["Peaks"][:spike_limit]
                else:
                    spike_data = unit_data["Peaks"][:]
                
                # Legacy conversion: spike[0]/1_000_000 * acq_rate
                # spike_data is array of tuples/arrays, spike[0] is timestamp in microseconds
                spike_time_stamp = [spike[0] / 1_000_000 * sampling_rate for spike in spike_data]
                spike_samples = np.int64(spike_time_stamp)
                
                logger.debug(f"Unit {unit_num}: {len(spike_samples)} spikes loaded from CMTR")
                
                # Compute STA using legacy algorithm (exact replication)
                sta, n_used, n_excluded = compute_sta_for_unit_legacy(
                    filtered_data,
                    spike_samples,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    spike_limit=-1,  # Already applied above
                )
                
                # Create unit group in HDF5 if not exists
                unit_id = f"unit_{int(unit_num):03d}"
                if unit_id not in hdf5_file["units"]:
                    hdf5_file["units"].create_group(unit_id)
                
                # Write to HDF5
                written = write_eimage_sta_to_hdf5(
                    hdf5_file,
                    unit_id,
                    sta,
                    n_used,
                    n_excluded,
                    config,
                    sampling_rate,
                    force=force,
                )
                
                if written:
                    units_processed += 1
                else:
                    units_skipped += 1
                    
            except Exception as e:
                logger.error(f"Failed to process unit {unit_id}: {e}")
                failed_units.append(unit_id)
    
    elapsed = time.time() - start_time
    
    result = EImageSTAResult(
        hdf5_path=hdf5_path,
        cmcr_path=cmcr_path,
        units_processed=units_processed,
        units_failed=len(failed_units),
        units_skipped=units_skipped,
        elapsed_seconds=elapsed,
        filter_time_seconds=filter_time,
        config=config,
        warnings=warnings_list,
        failed_units=failed_units,
    )
    
    logger.info(
        f"Legacy eimage_sta complete: {units_processed} processed, "
        f"{units_skipped} skipped, {len(failed_units)} failed, "
        f"{elapsed:.1f}s total"
    )
    
    return result


def compute_eimage_sta_legacy_improved(
    hdf5_path: Union[str, Path],
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
    *,
    duration_s: float = 120.0,
    spike_limit: int = 10000,
    unit_ids: Optional[List[str]] = None,
    window_range: Tuple[int, int] = (-10, 40),
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    skip_highpass: bool = False,
    chunk_duration_s: float = 30.0,
    force: bool = False,
) -> EImageSTAResult:
    """
    Legacy-style STA computation with CHUNK-BASED processing and single-threaded SOS filtering.
    
    This function processes sensor data in memory-efficient chunks:
    1. Load CMCR sensor data in chunks (e.g., 30s at a time)
    2. Filter each chunk using single-threaded SOS filter (no multi-threading)
    3. Extract spike windows from each chunk and accumulate STA
    4. Use legacy spike loading from CMTR
    
    Memory-efficient: Only one chunk (~5-10 GB) in memory at a time instead of
    full recording (~20-40 GB).
    
    Same as legacy:
    - CMTR spike loading via McsPy
    - Spike limit applied BEFORE conversion
    - Legacy STA boundary conditions and division
    
    Args:
        hdf5_path: Path to HDF5 file for storing results.
        cmcr_path: Path to CMCR file containing sensor data.
        cmtr_path: Path to CMTR file containing spike times (legacy source).
        duration_s: Total duration to analyze in seconds (default 120s).
        spike_limit: Maximum spikes to use per unit (default 10000).
        unit_ids: List of unit IDs to process (default None = all units).
            Note: unit_ids should match legacy format, e.g. ["1", "2"] not ["unit_001"].
        window_range: Tuple (start, end) relative to spike in samples, e.g. (-10, 40).
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        skip_highpass: If True, skip high-pass filter and mean-center instead.
        chunk_duration_s: Duration of each chunk in seconds (default 30s).
        force: If True, overwrite existing eimage_sta features.
    
    Returns:
        EImageSTAResult with processing summary.
    
    Example:
        >>> result = compute_eimage_sta_legacy_improved(
        ...     hdf5_path="artifacts/recording.h5",
        ...     cmcr_path="O:/data/recording.cmcr",
        ...     cmtr_path="O:/data/recording.cmtr",
        ...     duration_s=120.0,
        ...     spike_limit=10000,
        ...     unit_ids=["1", "2", "3"],
        ...     window_range=(-10, 40),
        ...     chunk_duration_s=30.0,  # 30-second chunks
        ... )
    """
    from scipy.signal import butter, sosfiltfilt
    
    start_time = time.time()
    
    hdf5_path = Path(hdf5_path)
    cmcr_path = Path(cmcr_path)
    cmtr_path = Path(cmtr_path)
    
    # Parse window_range to pre_samples and post_samples
    pre_samples = -window_range[0]
    post_samples = window_range[1]
    window_length = pre_samples + post_samples
    
    if pre_samples < 0 or post_samples < 0:
        raise ValueError(
            f"Invalid window_range {window_range}: start must be <= 0 and end must be >= 0"
        )
    
    logger.info(
        f"Legacy Improved (Chunked) STA: window_range={window_range}, "
        f"duration_s={duration_s}, spike_limit={spike_limit}, chunk_duration_s={chunk_duration_s}"
    )
    
    # Build config
    config = EImageSTAConfig(
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        pre_samples=pre_samples,
        post_samples=post_samples,
        spike_limit=spike_limit,
        duration_s=duration_s,
        skip_highpass=skip_highpass,
        use_cache=False,
        cache_path=None,
        force=force,
    )
    
    warnings_list: List[str] = []
    failed_units: List[str] = []
    
    # Validate filter parameters
    if not skip_highpass:
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
        if filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {filter_order}")
    
    # Check files exist
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    if not cmtr_path.exists():
        raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
    
    # Get sampling rate from HDF5
    with h5py.File(hdf5_path, "r") as f:
        sampling_rate = _get_sampling_rate(f)
    
    # Validate cutoff vs Nyquist
    if not skip_highpass:
        nyquist = 0.5 * sampling_rate
        if cutoff_hz >= nyquist:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be less than Nyquist frequency ({nyquist})"
            )
        # Design filter once (reused for all chunks)
        normalized_cutoff = cutoff_hz / nyquist
        sos = butter(filter_order, normalized_cutoff, btype='high', analog=False, output='sos')
    
    # =========================================================================
    # STEP 1: Open CMCR and get sensor data info
    # =========================================================================
    logger.info(f"Step 1: Opening CMCR file...")
    
    try:
        from McsPy import McsCMOSMEA
        cmcr_data = McsCMOSMEA.McsData(str(cmcr_path))
        sensor_stream = cmcr_data['Acquisition']['Sensor Data']["SensorData 1 1"]
        
        total_samples_available = sensor_stream.shape[0]
        n_rows = sensor_stream.shape[1]
        n_cols = sensor_stream.shape[2]
        
        # Calculate total samples to process
        total_samples = min(int(sampling_rate * duration_s), total_samples_available)
        
        logger.info(f"Sensor data: shape=({total_samples_available}, {n_rows}, {n_cols}), "
                   f"processing {total_samples} samples ({total_samples/sampling_rate:.1f}s)")
        
    except ImportError:
        raise ImportError("McsPy library not available. Install with: pip install McsPyDataTools")
    except KeyError as e:
        # Try alternative path
        logger.warning(f"Standard path failed ({e}), trying alternative...")
        try:
            acq = cmcr_data['Acquisition']
            sensor_streams = acq['Sensor Data']
            sensor_key = None
            for key in sensor_streams.keys():
                if 'SensorData' in str(key):
                    sensor_key = key
                    break
            if sensor_key is None:
                raise KeyError("No SensorData found in CMCR file")
            sensor_stream = sensor_streams[sensor_key]
            total_samples_available = sensor_stream.shape[0]
            n_rows = sensor_stream.shape[1]
            n_cols = sensor_stream.shape[2]
            total_samples = min(int(sampling_rate * duration_s), total_samples_available)
        except Exception as inner_e:
            raise ValueError(f"Failed to access sensor data: {inner_e}")
    # =========================================================================
    # STEP 2: Load spike data from CMTR using LEGACY approach
    # =========================================================================
    logger.info("Step 2: Loading spike data from CMTR using LEGACY McsPy approach...")
    
    cmtr_data = McsCMOSMEA.McsData(str(cmtr_path))
    spike_sorter = cmtr_data['Spike Sorter']
    cmtr_unit_keys = [key for key in spike_sorter.keys() if key.startswith('Unit')]
    logger.info(f"Found {len(cmtr_unit_keys)} units in CMTR: {cmtr_unit_keys[:5]}...")
    
    # Determine which units to process
    if unit_ids is not None:
        units_to_process = unit_ids
        logger.info(f"Processing {len(units_to_process)} specified units")
    else:
        units_to_process = []
        for key in cmtr_unit_keys:
            parts = key.split()
            if len(parts) >= 2:
                units_to_process.append(parts[1])
        logger.info(f"Processing all {len(units_to_process)} units from CMTR")
    
    # Pre-load all spike samples for all units (they're small)
    # Dictionary: unit_num -> spike_samples (in samples, not time)
    unit_spike_samples: Dict[str, np.ndarray] = {}
    total_spikes_for_division: Dict[str, int] = {}  # For legacy division
    
    for unit_num in units_to_process:
        cmtr_unit_key = f"Unit {unit_num}"
        if cmtr_unit_key not in spike_sorter:
            warnings_list.append(f"Unit {unit_num} not found in CMTR")
            continue
        
        unit_data = spike_sorter[cmtr_unit_key]
        if "Peaks" not in unit_data:
            warnings_list.append(f"No Peaks data for Unit {unit_num}")
            continue
        
        # Legacy: Apply spike_limit BEFORE loading/conversion
        if spike_limit > 0:
            spike_data = unit_data["Peaks"][:spike_limit]
        else:
            spike_data = unit_data["Peaks"][:]
        
        # Legacy conversion: spike[0]/1_000_000 * acq_rate
        spike_time_stamp = [spike[0] / 1_000_000 * sampling_rate for spike in spike_data]
        spike_samples_arr = np.int64(spike_time_stamp)
        
        # Filter spikes to only those within the duration we're processing
        valid_mask = spike_samples_arr < total_samples
        spike_samples_arr = spike_samples_arr[valid_mask]
        
        unit_spike_samples[unit_num] = spike_samples_arr
        total_spikes_for_division[unit_num] = len(spike_samples_arr)  # For legacy division
        
        logger.debug(f"Unit {unit_num}: {len(spike_samples_arr)} spikes within duration")
    
    # =========================================================================
    # STEP 3: Chunk-based processing
    # =========================================================================
    logger.info("Step 3: Processing sensor data in chunks (load -> filter -> accumulate STA)...")
    
    chunk_samples = int(sampling_rate * chunk_duration_s)
    # Add overlap for spike windows at chunk boundaries
    overlap_samples = window_length
    
    # Initialize STA accumulators for each unit
    sta_accumulators: Dict[str, np.ndarray] = {}
    spike_counts: Dict[str, int] = {}  # Count of actually used spikes per unit
    
    for unit_num in unit_spike_samples.keys():
        sta_accumulators[unit_num] = np.zeros((window_length, n_rows, n_cols), dtype=np.float64)
        spike_counts[unit_num] = 0
    
    # Calculate chunks
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    filter_time_total = 0.0
    
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
        chunk_start_time = time.time()
        
        # Calculate chunk boundaries
        chunk_start = chunk_idx * chunk_samples
        chunk_end = min(chunk_start + chunk_samples, total_samples)
        
        # For filtering, we need overlap at chunk boundaries to avoid edge effects
        # Load extra samples before and after if available
        load_start = max(0, chunk_start - overlap_samples)
        load_end = min(chunk_end + overlap_samples, total_samples)
        
        # Calculate the valid range within the loaded data (excluding padding)
        valid_start_in_chunk = chunk_start - load_start
        valid_end_in_chunk = valid_start_in_chunk + (chunk_end - chunk_start)
        
        # ---------------------------------------------------------------------
        # Load chunk
        # ---------------------------------------------------------------------
        load_time_start = time.time()
        chunk_data = sensor_stream[load_start:load_end, :, :]
        chunk_data = np.array(chunk_data).astype(np.int16)
        load_time = time.time() - load_time_start
        
        # ---------------------------------------------------------------------
        # Filter chunk (single-threaded SOS)
        # ---------------------------------------------------------------------
        filter_time_start = time.time()
        
        if skip_highpass:
            # Mean-center
            chunk_float = chunk_data.astype(np.float32)
            chunk_mean = chunk_float.mean(axis=0, keepdims=True)
            filtered_chunk = chunk_float - chunk_mean
        else:
            # Single-threaded SOS filter along time axis
            # Reshape to 2D for better memory layout: (time, n_electrodes)
            chunk_float = chunk_data.astype(np.float32)
            original_shape = chunk_float.shape
            reshaped = chunk_float.reshape(original_shape[0], -1)
            
            # Apply sosfiltfilt along axis=0 (time)
            filtered_reshaped = sosfiltfilt(sos, reshaped, axis=0).astype(np.float32)
            filtered_chunk = filtered_reshaped.reshape(original_shape)
            
            del reshaped, filtered_reshaped
        
        filter_time = time.time() - filter_time_start
        filter_time_total += filter_time
        
        # Free raw chunk data
        del chunk_data, chunk_float
        
        # Extract valid portion (without padding)
        filtered_valid = filtered_chunk[valid_start_in_chunk:valid_end_in_chunk, :, :]
        del filtered_chunk
        
        # ---------------------------------------------------------------------
        # Accumulate STA for each unit
        # ---------------------------------------------------------------------
        sta_time_start = time.time()
        
        for unit_num, spike_samples_arr in unit_spike_samples.items():
            # Find spikes within this chunk's valid range
            # Spike must have full window within this chunk
            min_spike = chunk_start + pre_samples
            max_spike = chunk_end - post_samples
            
            chunk_spikes_mask = (spike_samples_arr >= min_spike) & (spike_samples_arr < max_spike)
            chunk_spikes = spike_samples_arr[chunk_spikes_mask]
            
            if len(chunk_spikes) == 0:
                continue
            
            # Convert to local chunk indices
            local_spikes = chunk_spikes - chunk_start
            
            # Accumulate STA
            for local_spike in local_spikes:
                window_start = local_spike - pre_samples
                window_end = local_spike + post_samples
                
                if window_start >= 0 and window_end <= filtered_valid.shape[0]:
                    sta_accumulators[unit_num] += filtered_valid[window_start:window_end, :, :]
                    spike_counts[unit_num] += 1
        
        sta_time = time.time() - sta_time_start
        chunk_time = time.time() - chunk_start_time
        
        logger.debug(
            f"Chunk {chunk_idx+1}/{n_chunks}: samples [{chunk_start}:{chunk_end}], "
            f"load={load_time:.2f}s, filter={filter_time:.2f}s, sta={sta_time:.2f}s, "
            f"total={chunk_time:.2f}s"
        )
        
        # Free filtered data
        del filtered_valid
    
    # =========================================================================
    # STEP 4: Finalize STAs and write to HDF5
    # =========================================================================
    logger.info("Step 4: Finalizing STAs and writing to HDF5...")
    
    units_processed = 0
    units_skipped = 0
    
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        if "units" not in hdf5_file:
            hdf5_file.create_group("units")
        
        for unit_num in tqdm(unit_spike_samples.keys(), desc="Saving eimage_sta"):
            try:
                # Legacy division: divide by TOTAL spikes (before edge exclusion)
                total_for_div = total_spikes_for_division.get(unit_num, 0)
                n_used = spike_counts.get(unit_num, 0)
                n_excluded = total_for_div - n_used
                
                if total_for_div == 0:
                    sta = np.full_like(sta_accumulators[unit_num], np.nan)
                else:
                    sta = sta_accumulators[unit_num] / total_for_div
                
                logger.debug(f"Unit {unit_num}: {n_used} used, {n_excluded} excluded, dividing by {total_for_div}")
                
                unit_id = f"unit_{int(unit_num):03d}"
                if unit_id not in hdf5_file["units"]:
                    hdf5_file["units"].create_group(unit_id)
                
                written = write_eimage_sta_to_hdf5(
                    hdf5_file,
                    unit_id,
                    sta,
                    n_used,
                    n_excluded,
                    config,
                    sampling_rate,
                    force=force,
                )
                
                if written:
                    units_processed += 1
                else:
                    units_skipped += 1
                    
            except Exception as e:
                logger.error(f"Failed to save unit {unit_num}: {e}")
                failed_units.append(str(unit_num))
    
    elapsed = time.time() - start_time
    
    result = EImageSTAResult(
        hdf5_path=hdf5_path,
        cmcr_path=cmcr_path,
        units_processed=units_processed,
        units_failed=len(failed_units),
        units_skipped=units_skipped,
        elapsed_seconds=elapsed,
        filter_time_seconds=filter_time_total,
        config=config,
        warnings=warnings_list,
        failed_units=failed_units,
    )
    
    logger.info(
        f"Legacy Improved (Chunked) eimage_sta complete: {units_processed} processed, "
        f"{units_skipped} skipped, {len(failed_units)} failed, {elapsed:.1f}s total, "
        f"filter_time={filter_time_total:.1f}s"
    )
    
    return result

