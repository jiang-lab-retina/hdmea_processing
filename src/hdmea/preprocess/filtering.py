"""
Signal filtering and preprocessing utilities.

Includes firing rate computation and signal smoothing.
"""

import logging
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


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

