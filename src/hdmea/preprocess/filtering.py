"""
Signal filtering and preprocessing utilities.

Includes firing rate computation, signal smoothing, and high-pass filtering
for sensor data preprocessing.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


logger = logging.getLogger(__name__)


def apply_highpass_filter_3d(
    sensor_data: np.ndarray,
    cutoff_hz: float,
    sampling_rate: float,
    filter_order: int = 2,
    n_workers: int = 4,
    chunk_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to 3D sensor array (optimized).
    
    Uses multiple optimizations for maximum speed:
    - SOS (second-order sections) format for numerical stability and speed
    - Reshapes to 2D for more efficient memory access
    - Processes electrodes in parallel chunks using ThreadPoolExecutor
    - Uses float32 throughout to reduce memory bandwidth
    - Shows progress bar without sacrificing performance
    
    Args:
        sensor_data: 3D array (time, rows, cols) of sensor readings.
        cutoff_hz: Filter cutoff frequency in Hz.
        sampling_rate: Data sampling rate in Hz.
        filter_order: Butterworth filter order (default: 2).
        n_workers: Number of parallel workers (default: 4).
        chunk_size: Number of electrodes per chunk (default: 256).
        show_progress: Whether to show progress bar (default: True).
    
    Returns:
        Filtered array as float32, same shape as input.
    
    Raises:
        ValueError: If cutoff_hz is invalid (<=0 or >= Nyquist).
        ValueError: If filter_order < 1.
    
    Performance:
        ~10-15s for 120s of data at 20kHz, 64x64 electrodes (4x faster).
    """
    # Validate parameters
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    
    if filter_order < 1:
        raise ValueError(f"filter_order must be >= 1, got {filter_order}")
    
    nyquist = 0.5 * sampling_rate
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be less than Nyquist frequency ({nyquist})"
        )
    
    original_shape = sensor_data.shape
    n_samples = original_shape[0]
    n_electrodes = np.prod(original_shape[1:])
    
    logger.info(
        f"Applying high-pass filter: cutoff={cutoff_hz}Hz, order={filter_order}, "
        f"shape={original_shape}, n_electrodes={n_electrodes}"
    )
    
    # Normalize cutoff frequency
    normalized_cutoff = cutoff_hz / nyquist
    
    # Design Butterworth high-pass filter using SOS format (more stable & faster)
    sos = butter(filter_order, normalized_cutoff, btype='high', analog=False, output='sos')
    
    # Convert to float32 and reshape to 2D: (time, n_electrodes)
    # This gives better memory locality for filtering along time axis
    data_2d = sensor_data.astype(np.float32).reshape(n_samples, n_electrodes)
    
    logger.info(
        f"Before filtering - Input shape: {original_shape}, dtype: {sensor_data.dtype}, "
        f"reshaped to 2D: {data_2d.shape}"
    )
    
    # For small arrays, just filter directly (with simple progress indication)
    if n_electrodes <= chunk_size or n_workers <= 1:
        logger.info("Using single-threaded filtering")
        if show_progress:
            # Show indeterminate progress for single-threaded mode
            with tqdm(total=1, desc="High-pass filtering", unit="batch") as pbar:
                filtered_2d = sosfiltfilt(sos, data_2d, axis=0).astype(np.float32)
                pbar.update(1)
        else:
            filtered_2d = sosfiltfilt(sos, data_2d, axis=0).astype(np.float32)
    else:
        # Process in chunks using thread pool for larger arrays
        logger.info(f"Using multi-threaded filtering with {n_workers} workers")
        filtered_2d = _filter_parallel(data_2d, sos, n_workers, chunk_size, show_progress)
    
    # Reshape back to original 3D shape
    filtered = filtered_2d.reshape(original_shape)
    
    logger.info(
        f"After filtering - Output shape: {filtered.shape}, dtype: {filtered.dtype}"
    )
    logger.info("High-pass filtering complete")
    
    return filtered


def _filter_parallel(
    data_2d: np.ndarray,
    sos: np.ndarray,
    n_workers: int,
    chunk_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Filter 2D array in parallel chunks with progress bar.
    
    Args:
        data_2d: 2D array (time, electrodes).
        sos: SOS filter coefficients.
        n_workers: Number of parallel workers.
        chunk_size: Electrodes per chunk.
        show_progress: Whether to show progress bar.
    
    Returns:
        Filtered 2D array.
    """
    n_samples, n_electrodes = data_2d.shape
    
    # Pre-allocate output
    filtered_2d = np.empty_like(data_2d)
    
    # Create chunk indices
    chunk_starts = list(range(0, n_electrodes, chunk_size))
    n_chunks = len(chunk_starts)
    
    def filter_chunk(start_idx: int) -> int:
        """Filter a chunk of electrodes in-place. Returns start_idx for tracking."""
        end_idx = min(start_idx + chunk_size, n_electrodes)
        chunk = data_2d[:, start_idx:end_idx]
        filtered_2d[:, start_idx:end_idx] = sosfiltfilt(sos, chunk, axis=0).astype(np.float32)
        return start_idx
    
    # Use ThreadPoolExecutor with progress bar
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(filter_chunk, start): start for start in chunk_starts}
        
        # Track completion with progress bar
        if show_progress:
            with tqdm(total=n_chunks, desc="High-pass filtering", unit="chunk") as pbar:
                for future in as_completed(futures):
                    future.result()  # Raises exception if any
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                future.result()
    
    return filtered_2d


def apply_highpass_filter_3d_simple(
    sensor_data: np.ndarray,
    cutoff_hz: float,
    sampling_rate: float,
    filter_order: int = 2,
) -> np.ndarray:
    """
    Simple high-pass filter without parallelization (for comparison/fallback).
    
    Args:
        sensor_data: 3D array (time, rows, cols) of sensor readings.
        cutoff_hz: Filter cutoff frequency in Hz.
        sampling_rate: Data sampling rate in Hz.
        filter_order: Butterworth filter order (default: 2).
    
    Returns:
        Filtered array as float32.
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(filter_order, normalized_cutoff, btype='high', analog=False, output='sos')
    
    # Simple direct filtering
    filtered = sosfiltfilt(sos, sensor_data.astype(np.float32), axis=0)
    
    return filtered.astype(np.float32)


def apply_highpass_filter_3d_legacy(
    sensor_data: np.ndarray,
    cutoff_hz: float,
    sampling_rate: float,
    filter_order: int = 2,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter using the EXACT legacy algorithm.
    
    This function replicates the original legacy implementation precisely:
    - Uses butter() with output='ba' (transfer function format)
    - Uses filtfilt() for zero-phase filtering
    - Loops over each electrode individually (row by row, column by column)
    
    Legacy code reference (load_raw_data.py lines 339-348):
        def apply_high_pass_filter(sensor_data, cutoff_freq, sampling_freq, filter_order=2):
            nyquist = 0.5 * sampling_freq
            normal_cutoff = cutoff_freq / nyquist
            b, a = butter(filter_order, normal_cutoff, btype='high', analog=False, output='ba')
            sensor_data_filtered = np.zeros(sensor_data.shape)
            for i in tqdm(range(sensor_data.shape[1]), leave=False, desc="Row"):
                for j in tqdm(range(sensor_data.shape[2]), leave=False, desc="Column"):
                    sensor_data_filtered[:,i,j] = filtfilt(b, a, sensor_data[:,i,j])
            return sensor_data_filtered
    
    WARNING: This is MUCH slower than apply_highpass_filter_3d (~1 hour vs ~15s for 120s data).
    Use only when exact legacy behavior is required for validation.
    
    Args:
        sensor_data: 3D array (time, rows, cols) of sensor readings.
        cutoff_hz: Filter cutoff frequency in Hz (legacy: cutoff_freq).
        sampling_rate: Data sampling rate in Hz (legacy: sampling_freq).
        filter_order: Butterworth filter order (default: 2).
        show_progress: Whether to show progress bars (default: True).
    
    Returns:
        Filtered array with same shape as input.
    """
    from scipy.signal import filtfilt
    
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_hz / nyquist
    
    # Legacy uses 'ba' format (transfer function coefficients)
    b, a = butter(filter_order, normal_cutoff, btype='high', analog=False, output='ba')
    
    # Pre-allocate output array (legacy uses np.zeros)
    sensor_data_filtered = np.zeros(sensor_data.shape, dtype=np.float64)
    
    n_rows = sensor_data.shape[1]
    n_cols = sensor_data.shape[2]
    
    logger.info(
        f"Legacy filter: cutoff={cutoff_hz}Hz, order={filter_order}, "
        f"shape={sensor_data.shape}, electrodes={n_rows}x{n_cols}"
    )
    logger.warning("Using legacy filter - this is SLOW (~1 hour for 120s data)")
    
    # Legacy nested loop: iterate over each electrode individually
    if show_progress:
        row_iter = tqdm(range(n_rows), leave=False, desc="Row")
    else:
        row_iter = range(n_rows)
    
    for i in row_iter:
        if show_progress:
            col_iter = tqdm(range(n_cols), leave=False, desc="Column")
        else:
            col_iter = range(n_cols)
        
        for j in col_iter:
            # Filter single electrode time series using filtfilt (zero-phase)
            sensor_data_filtered[:, i, j] = filtfilt(b, a, sensor_data[:, i, j])
    
    logger.info("Legacy filter complete")
    
    return sensor_data_filtered


def compute_firing_rate(
    spike_times: np.ndarray,
    duration_us: float,
    bin_rate_hz: float = 10,
) -> np.ndarray:
    """
    Compute binned firing rate from spike times.
    
    Args:
        spike_times: Spike timestamps in microseconds
        duration_us: Total recording duration in microseconds
        bin_rate_hz: Output bin rate in Hz (default: 10Hz = 100ms bins)
    
    Returns:
        Firing rate array in spikes/second
    """
    if len(spike_times) == 0:
        return np.array([], dtype=np.float32)
    
    # Calculate bin size in microseconds
    bin_size_us = 1e6 / bin_rate_hz
    
    # Number of bins
    n_bins = int(np.ceil(duration_us / bin_size_us))
    
    if n_bins <= 0:
        return np.array([], dtype=np.float32)
    
    # Create histogram
    bin_edges = np.arange(0, (n_bins + 1) * bin_size_us, bin_size_us)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    
    # Convert to rate (spikes/second)
    # Each bin is (1/bin_rate_hz) seconds, so multiply by bin_rate_hz
    firing_rate = counts.astype(np.float32) * bin_rate_hz
    
    return firing_rate


def smooth_signal(
    signal: np.ndarray,
    window_size: int = 5,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Smooth a signal using a moving window.
    
    Args:
        signal: Input signal array
        window_size: Size of smoothing window
        method: Smoothing method ("gaussian", "boxcar", "triangular")
    
    Returns:
        Smoothed signal
    """
    if len(signal) < window_size:
        return signal
    
    if method == "boxcar":
        kernel = np.ones(window_size) / window_size
    elif method == "triangular":
        kernel = np.concatenate([
            np.arange(1, window_size // 2 + 1),
            np.arange(window_size // 2, 0, -1)
        ])
        kernel = kernel / kernel.sum()
    else:  # gaussian
        x = np.linspace(-2, 2, window_size)
        kernel = np.exp(-x**2)
        kernel = kernel / kernel.sum()
    
    # Use same padding to preserve array length
    smoothed = np.convolve(signal, kernel, mode="same")
    
    return smoothed.astype(signal.dtype)


def filter_by_firing_rate(
    spike_times: np.ndarray,
    duration_us: float,
    min_rate_hz: float = 0.1,
    max_rate_hz: float = 100.0,
) -> bool:
    """
    Check if a unit's firing rate is within acceptable range.
    
    Args:
        spike_times: Spike timestamps in microseconds
        duration_us: Total recording duration in microseconds
        min_rate_hz: Minimum acceptable firing rate
        max_rate_hz: Maximum acceptable firing rate
    
    Returns:
        True if firing rate is within range, False otherwise
    """
    if len(spike_times) == 0 or duration_us <= 0:
        return False
    
    mean_rate = len(spike_times) / (duration_us / 1e6)
    
    return min_rate_hz <= mean_rate <= max_rate_hz

