"""
Fixtures for STA (Spike Triggered Average) unit tests.

Provides synthetic data generators for testing STA computation
without requiring real recording files.
"""

import numpy as np
import h5py
import tempfile
from pathlib import Path
from typing import Tuple, Optional


def create_synthetic_movie(
    n_frames: int = 100,
    height: int = 10,
    width: int = 10,
    dtype: np.dtype = np.uint8,
    seed: int = 42,
) -> np.ndarray:
    """
    Create a synthetic noise movie for testing.
    
    Args:
        n_frames: Number of frames in the movie.
        height: Height of each frame in pixels.
        width: Width of each frame in pixels.
        dtype: Data type for the movie array.
        seed: Random seed for reproducibility.
    
    Returns:
        3D array of shape (n_frames, height, width).
    """
    rng = np.random.default_rng(seed)
    
    if dtype == np.uint8:
        movie = rng.integers(0, 256, size=(n_frames, height, width), dtype=dtype)
    else:
        movie = rng.random(size=(n_frames, height, width)).astype(dtype)
    
    return movie


def create_synthetic_spikes(
    n_spikes: int = 50,
    movie_length: int = 100,
    cover_range: Tuple[int, int] = (-60, 0),
    include_edge_spikes: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic spike times (as frame numbers) for testing.
    
    Args:
        n_spikes: Number of spikes to generate.
        movie_length: Total number of frames in the movie.
        cover_range: Frame window for STA computation.
        include_edge_spikes: If True, include some spikes at edges.
        seed: Random seed for reproducibility.
    
    Returns:
        1D array of spike times as frame indices.
    """
    rng = np.random.default_rng(seed)
    
    # Safe range for spikes (avoiding edge effects)
    safe_start = abs(cover_range[0])
    safe_end = movie_length - abs(cover_range[1]) - 1
    
    if safe_end <= safe_start:
        raise ValueError(
            f"Movie too short ({movie_length}) for cover_range {cover_range}"
        )
    
    if include_edge_spikes:
        # Include some spikes at the very edges
        n_edge = min(5, n_spikes // 5)
        n_safe = n_spikes - n_edge
        
        safe_spikes = rng.integers(safe_start, safe_end, size=n_safe)
        edge_spikes = np.concatenate([
            rng.integers(0, safe_start, size=n_edge // 2),
            rng.integers(safe_end, movie_length, size=n_edge - n_edge // 2),
        ])
        spikes = np.concatenate([safe_spikes, edge_spikes])
    else:
        spikes = rng.integers(safe_start, safe_end, size=n_spikes)
    
    return np.sort(spikes).astype(np.int64)


def create_test_hdf5(
    tmp_path: Path,
    movie_name: str = "test_noise_movie",
    n_units: int = 3,
    n_spikes_per_unit: int = 50,
    acquisition_rate: float = 20000.0,
    frame_rate: float = 15.0,
) -> Path:
    """
    Create a minimal HDF5 file for testing STA computation.
    
    Args:
        tmp_path: Directory to create the file in.
        movie_name: Name of the movie (should contain 'noise' for detection).
        n_units: Number of units to create.
        n_spikes_per_unit: Number of spikes per unit.
        acquisition_rate: Acquisition rate in Hz.
        frame_rate: Movie frame rate in Hz.
    
    Returns:
        Path to the created HDF5 file.
    """
    hdf5_path = tmp_path / "test_recording.h5"
    
    samples_per_frame = acquisition_rate / frame_rate
    
    with h5py.File(hdf5_path, "w") as f:
        # Add metadata
        metadata = f.create_group("metadata")
        metadata.create_dataset("acquisition_rate", data=[acquisition_rate])
        
        # Add units group
        units = f.create_group("units")
        
        for i in range(n_units):
            unit_id = f"unit_{i:03d}"
            unit_group = units.create_group(unit_id)
            
            # Add spike_times_sectioned with the movie
            sectioned = unit_group.create_group("spike_times_sectioned")
            movie_group = sectioned.create_group(movie_name)
            
            # Create spike times as sampling indices
            spike_frames = create_synthetic_spikes(
                n_spikes=n_spikes_per_unit,
                movie_length=100,
                seed=42 + i,
            )
            spike_samples = (spike_frames * samples_per_frame).astype(np.int64)
            
            # Store as trials_spike_times
            trials = movie_group.create_group("trials_spike_times")
            trials.create_dataset("0", data=spike_samples)
            
            # Add trial boundaries (for compatibility)
            boundaries = np.array([[0, int(100 * samples_per_frame)]])
            movie_group.create_dataset("trials_start_end", data=boundaries)
    
    return hdf5_path


def create_stimulus_file(
    stimuli_dir: Path,
    movie_name: str,
    movie: Optional[np.ndarray] = None,
) -> Path:
    """
    Create a stimulus .npy file for testing.
    
    Args:
        stimuli_dir: Directory to save the file.
        movie_name: Name for the movie file (without .npy extension).
        movie: Movie array to save. If None, creates a synthetic movie.
    
    Returns:
        Path to the created .npy file.
    """
    if movie is None:
        movie = create_synthetic_movie()
    
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    npy_path = stimuli_dir / f"{movie_name}.npy"
    np.save(npy_path, movie)
    
    return npy_path

