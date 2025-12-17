"""
Fixtures for analog section time detection tests.

Provides synthetic Zarr archives with known pulse locations for testing
the add_section_time_analog() function.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pytest
import zarr


def create_synthetic_pulses(
    n_samples: int,
    acquisition_rate: float,
    pulse_times_seconds: List[float],
    pulse_amplitude: float = 1e7,
    noise_level: float = 1e4,
) -> np.ndarray:
    """
    Create a synthetic analog signal with known pulse locations.
    
    Args:
        n_samples: Total number of samples
        acquisition_rate: Sampling rate in Hz
        pulse_times_seconds: List of pulse onset times in seconds
        pulse_amplitude: Step height at each pulse
        noise_level: Gaussian noise standard deviation
    
    Returns:
        Signal array with step pulses at specified times
    """
    signal = np.zeros(n_samples, dtype=np.float32)
    
    # Add baseline noise
    signal += np.random.normal(0, noise_level, n_samples).astype(np.float32)
    
    # Add step pulses at specified times
    for pulse_time in pulse_times_seconds:
        pulse_sample = int(pulse_time * acquisition_rate)
        if 0 <= pulse_sample < n_samples:
            signal[pulse_sample:] += pulse_amplitude
    
    return signal


@pytest.fixture
def analog_zarr_with_pulses(tmp_path):
    """
    Create a Zarr archive with synthetic pulses at known locations.
    
    Returns tuple of (zarr_path, expected_onset_samples, acquisition_rate)
    """
    zarr_path = tmp_path / "test_analog.zarr"
    
    # Recording parameters
    acquisition_rate = 20000.0  # 20 kHz
    recording_duration = 300.0  # 5 minutes
    n_samples = int(recording_duration * acquisition_rate)
    
    # Define pulse times (in seconds)
    pulse_times = [10.0, 50.0, 100.0, 150.0, 200.0]  # 5 pulses
    
    # Create Zarr structure
    root = zarr.open(str(zarr_path), mode='w')
    
    # Create metadata
    metadata = root.create_group("metadata")
    acq_rate_arr = np.array([acquisition_rate])
    metadata.create_array("acquisition_rate", data=acq_rate_arr)
    
    # Create frame_timestamps (simulate ~50 Hz display)
    n_frames = int(recording_duration * 50)
    frame_timestamps = np.linspace(0, n_samples - 1, n_frames).astype(np.uint64)
    metadata.create_array("frame_timestamps", data=frame_timestamps)
    
    # Create stimulus group with light reference
    stimulus = root.create_group("stimulus")
    lr_group = stimulus.create_group("light_reference")
    
    # Create synthetic signal with pulses
    raw_ch1 = create_synthetic_pulses(
        n_samples=n_samples,
        acquisition_rate=acquisition_rate,
        pulse_times_seconds=pulse_times,
        pulse_amplitude=1e7,
        noise_level=1e4,
    )
    lr_group.create_array("raw_ch1", data=raw_ch1)
    
    # Also create raw_ch2 for compatibility
    raw_ch2 = np.zeros(n_samples, dtype=np.float32)
    lr_group.create_array("raw_ch2", data=raw_ch2)
    
    # Mark stage1 complete
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_analog_dataset"
    
    # Calculate expected onset samples
    expected_onset_samples = [int(t * acquisition_rate) for t in pulse_times]
    
    return zarr_path, expected_onset_samples, acquisition_rate


@pytest.fixture
def analog_zarr_no_pulses(tmp_path):
    """Create a Zarr archive with no detectable pulses (flat signal)."""
    zarr_path = tmp_path / "test_no_pulses.zarr"
    
    acquisition_rate = 20000.0
    n_samples = 1000000  # ~50 seconds
    
    root = zarr.open(str(zarr_path), mode='w')
    
    # Metadata
    metadata = root.create_group("metadata")
    acq_rate_arr = np.array([acquisition_rate])
    metadata.create_array("acquisition_rate", data=acq_rate_arr)
    
    n_frames = 2500
    frame_timestamps = np.linspace(0, n_samples - 1, n_frames).astype(np.uint64)
    metadata.create_array("frame_timestamps", data=frame_timestamps)
    
    # Flat signal with small noise (no detectable pulses)
    stimulus = root.create_group("stimulus")
    lr_group = stimulus.create_group("light_reference")
    raw_ch1 = np.random.normal(0, 1000, n_samples).astype(np.float32)
    lr_group.create_array("raw_ch1", data=raw_ch1)
    
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_no_pulses_dataset"
    
    return zarr_path


@pytest.fixture
def analog_zarr_missing_raw_ch1(tmp_path):
    """Create a Zarr archive without raw_ch1."""
    zarr_path = tmp_path / "test_missing_raw_ch1.zarr"
    
    root = zarr.open(str(zarr_path), mode='w')
    
    metadata = root.create_group("metadata")
    acq_rate = np.array([20000.0])
    metadata.create_array("acquisition_rate", data=acq_rate)
    frame_ts = np.arange(1000, dtype=np.uint64)
    metadata.create_array("frame_timestamps", data=frame_ts)
    
    stimulus = root.create_group("stimulus")
    stimulus.create_group("light_reference")  # Empty - no raw_ch1
    
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_missing_raw_ch1"
    
    return zarr_path


@pytest.fixture
def analog_zarr_missing_frame_timestamps(tmp_path):
    """Create a Zarr archive without frame_timestamps."""
    zarr_path = tmp_path / "test_missing_timestamps.zarr"
    
    root = zarr.open(str(zarr_path), mode='w')
    
    metadata = root.create_group("metadata")
    acq_rate = np.array([20000.0])
    metadata.create_array("acquisition_rate", data=acq_rate)
    # No frame_timestamps!
    
    stimulus = root.create_group("stimulus")
    lr_group = stimulus.create_group("light_reference")
    raw_ch1 = np.zeros(100000, dtype=np.float32)
    lr_group.create_array("raw_ch1", data=raw_ch1)
    
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_missing_timestamps"
    
    return zarr_path


@pytest.fixture
def analog_zarr_missing_acquisition_rate(tmp_path):
    """Create a Zarr archive without acquisition_rate."""
    zarr_path = tmp_path / "test_missing_rate.zarr"
    
    root = zarr.open(str(zarr_path), mode='w')
    
    metadata = root.create_group("metadata")
    # No acquisition_rate!
    frame_ts = np.arange(1000, dtype=np.uint64)
    metadata.create_array("frame_timestamps", data=frame_ts)
    
    stimulus = root.create_group("stimulus")
    lr_group = stimulus.create_group("light_reference")
    raw_ch1 = np.zeros(100000, dtype=np.float32)
    lr_group.create_array("raw_ch1", data=raw_ch1)
    
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_missing_rate"
    
    return zarr_path

