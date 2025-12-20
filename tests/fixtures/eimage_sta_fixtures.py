"""
Test fixtures for eimage_sta feature testing.

Provides synthetic data generators for unit testing without requiring
real CMCR/HDF5 files.
"""

import numpy as np
from typing import Tuple


def generate_synthetic_sensor_data(
    n_samples: int = 1000,
    n_rows: int = 64,
    n_cols: int = 64,
    sampling_rate: float = 20000.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic sensor data for testing.
    
    Creates random noise with realistic properties for HD-MEA sensor data.
    
    Args:
        n_samples: Number of time samples.
        n_rows: Number of electrode rows.
        n_cols: Number of electrode columns.
        sampling_rate: Sampling rate in Hz.
        seed: Random seed for reproducibility.
    
    Returns:
        3D array of shape (n_samples, n_rows, n_cols) as int16.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random noise in typical sensor data range
    data = rng.normal(0, 100, size=(n_samples, n_rows, n_cols))
    
    return data.astype(np.int16)


def generate_synthetic_spike_times(
    n_spikes: int = 100,
    duration_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic spike times for testing.
    
    Args:
        n_spikes: Number of spikes to generate.
        duration_samples: Total duration in samples.
        seed: Random seed for reproducibility.
    
    Returns:
        1D array of spike sample indices, sorted ascending.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random spike times, leaving margin at edges
    margin = 50  # samples
    spikes = rng.integers(
        margin, 
        duration_samples - margin, 
        size=n_spikes
    )
    
    return np.sort(spikes)


def generate_test_data_pair(
    n_samples: int = 2000,
    n_spikes: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a matched pair of sensor data and spike times for testing.
    
    Args:
        n_samples: Number of time samples.
        n_spikes: Number of spikes.
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (sensor_data, spike_samples).
    """
    sensor_data = generate_synthetic_sensor_data(
        n_samples=n_samples,
        seed=seed,
    )
    spike_samples = generate_synthetic_spike_times(
        n_spikes=n_spikes,
        duration_samples=n_samples,
        seed=seed + 1,
    )
    
    return sensor_data, spike_samples

