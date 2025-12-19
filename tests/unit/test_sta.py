"""
Unit tests for STA (Spike Triggered Average) computation.

Tests cover:
    - Core STA computation algorithm
    - Noise movie detection
    - Edge effect handling
    - Multiprocessing functionality
    - Error handling and retry logic
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

# Import fixtures
from tests.fixtures.sta_fixtures import (
    create_synthetic_movie,
    create_synthetic_spikes,
    create_test_hdf5,
    create_stimulus_file,
)

# Module under test
from hdmea.features.sta import (
    compute_sta,
    _compute_sta_for_unit,
    _find_noise_movie,
    _load_stimulus_movie,
    STAResult,
)
from hdmea.io.section_time import convert_sample_index_to_frame


class TestSTAComputation:
    """Tests for core STA computation."""
    
    def test_compute_sta_for_unit_basic(self):
        """Test basic STA computation with synthetic data."""
        # Create simple test data
        movie = create_synthetic_movie(n_frames=100, height=5, width=5, seed=42)
        spikes = create_synthetic_spikes(n_spikes=20, movie_length=100, cover_range=(-10, 0), seed=42)
        
        sta, n_used, n_excluded = _compute_sta_for_unit(spikes, movie, cover_range=(-10, 0))
        
        # Check shape: (window_length, height, width)
        assert sta.shape == (10, 5, 5)
        assert n_used == 20
        assert n_excluded == 0
    
    def test_compute_sta_for_unit_correct_shape(self):
        """Test STA output shape matches cover_range and movie dimensions."""
        movie = create_synthetic_movie(n_frames=200, height=15, width=15, seed=42)
        spikes = create_synthetic_spikes(n_spikes=30, movie_length=200, cover_range=(-60, 0), seed=42)
        
        sta, n_used, n_excluded = _compute_sta_for_unit(spikes, movie, cover_range=(-60, 0))
        
        # Shape should be (60, 15, 15) for cover_range=(-60, 0)
        assert sta.shape == (60, 15, 15)
        assert sta.dtype == np.float32


class TestNoiseMovieDetection:
    """Tests for noise movie detection."""
    
    def test_find_noise_movie_single_match(self, tmp_path):
        """Test finding noise movie when exactly one exists."""
        hdf5_path = create_test_hdf5(tmp_path, movie_name="test_noise_15hz")
        
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            movie_name = _find_noise_movie(f, "unit_000")
        
        assert movie_name == "test_noise_15hz"
    
    def test_find_noise_movie_zero_matches(self, tmp_path):
        """Test error when no noise movie exists."""
        hdf5_path = create_test_hdf5(tmp_path, movie_name="test_grating_movie")
        
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            with pytest.raises(ValueError, match="No noise movie found"):
                _find_noise_movie(f, "unit_000")
    
    def test_find_noise_movie_case_insensitive(self, tmp_path):
        """Test that noise detection is case-insensitive."""
        hdf5_path = create_test_hdf5(tmp_path, movie_name="Dense_NOISE_30hz")
        
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            movie_name = _find_noise_movie(f, "unit_000")
        
        assert movie_name == "Dense_NOISE_30hz"


class TestSpikeConversion:
    """Tests for spike time conversion using frame_timestamps."""
    
    def test_convert_sample_to_frame_basic(self):
        """Test basic sample-to-frame conversion using frame_timestamps."""
        # Create frame_timestamps: frames at sample indices 0, 1000, 2000, 3000, 4000
        frame_timestamps = np.array([0, 1000, 2000, 3000, 4000], dtype=np.int64)
        
        # Spikes at exact frame boundaries
        spike_samples = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        
        frames = convert_sample_index_to_frame(spike_samples, frame_timestamps)
        
        # Floor behavior: spike at sample 1000 is at/after frame 1
        expected = np.array([0, 1, 2, 3], dtype=np.int64)
        np.testing.assert_array_equal(frames, expected)
    
    def test_convert_sample_to_frame_floor_behavior(self):
        """Test that conversion uses floor (returns frame that started at or before sample)."""
        # Create frame_timestamps: frames at sample indices 0, 1000, 2000, 3000
        frame_timestamps = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        
        # Spikes between frames
        spike_samples = np.array([500, 1500, 2999], dtype=np.int64)
        
        frames = convert_sample_index_to_frame(spike_samples, frame_timestamps)
        
        # Floor: 500 is after frame 0, before frame 1 -> frame 0
        # 1500 is after frame 1, before frame 2 -> frame 1
        # 2999 is after frame 2, before frame 3 -> frame 2
        expected = np.array([0, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(frames, expected)


class TestEdgeEffects:
    """Tests for edge effect handling."""
    
    def test_edge_exclusion_at_start(self):
        """Test spikes near start are excluded."""
        movie = create_synthetic_movie(n_frames=100, height=5, width=5, seed=42)
        # For cover_range=(-10, 0), spike at frame N needs N >= 10
        # Spikes at frames 5, 9 should be excluded (need window [-10, 0) + spike)
        spikes = np.array([5, 9, 50, 60, 70], dtype=np.int64)
        
        sta, n_used, n_excluded = _compute_sta_for_unit(spikes, movie, cover_range=(-10, 0))
        
        assert n_used == 3  # Only 50, 60, 70 are valid (10+ works)
        assert n_excluded == 2
    
    def test_edge_exclusion_at_end(self):
        """Test spikes near end are excluded."""
        movie = create_synthetic_movie(n_frames=100, height=5, width=5, seed=42)
        # For cover_range=(-5, 5) with 100 frames, need spike + 5 <= 100, so spike <= 95
        # Spikes at frames 96, 99 should be excluded
        spikes = np.array([30, 40, 50, 96, 99], dtype=np.int64)
        
        sta, n_used, n_excluded = _compute_sta_for_unit(spikes, movie, cover_range=(-5, 5))
        
        assert n_used == 3  # Only 30, 40, 50 are valid
        assert n_excluded == 2
    
    def test_zero_valid_spikes_returns_nan(self):
        """Test that no valid spikes returns NaN STA."""
        movie = create_synthetic_movie(n_frames=100, height=5, width=5, seed=42)
        # For cover_range=(-60, 0), need spike >= 60
        # All spikes at frames 0-59 should be excluded
        spikes = np.array([0, 1, 2, 55, 59], dtype=np.int64)
        
        sta, n_used, n_excluded = _compute_sta_for_unit(spikes, movie, cover_range=(-60, 0))
        
        assert n_used == 0
        assert n_excluded == 5
        assert np.all(np.isnan(sta))


class TestStimulusLoading:
    """Tests for stimulus movie loading."""
    
    def test_load_stimulus_movie_success(self, tmp_path):
        """Test successful loading of stimulus movie."""
        movie = create_synthetic_movie(n_frames=50, height=10, width=10, seed=42)
        create_stimulus_file(tmp_path, "test_movie", movie)
        
        loaded = _load_stimulus_movie("test_movie", tmp_path)
        
        np.testing.assert_array_equal(loaded, movie)
    
    def test_load_stimulus_movie_not_found(self, tmp_path):
        """Test error when stimulus file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Stimulus file not found"):
            _load_stimulus_movie("nonexistent_movie", tmp_path)


class TestCoverRangeValidation:
    """Tests for cover_range parameter validation."""
    
    def test_invalid_cover_range_raises_error(self, tmp_path):
        """Test that invalid cover_range raises ValueError."""
        hdf5_path = create_test_hdf5(tmp_path, movie_name="test_noise")
        
        with pytest.raises(ValueError, match="Invalid cover_range"):
            compute_sta(hdf5_path, cover_range=(0, -60), stimuli_dir=tmp_path)
    
    def test_equal_cover_range_raises_error(self, tmp_path):
        """Test that equal start/end in cover_range raises ValueError."""
        hdf5_path = create_test_hdf5(tmp_path, movie_name="test_noise")
        
        with pytest.raises(ValueError, match="Invalid cover_range"):
            compute_sta(hdf5_path, cover_range=(0, 0), stimuli_dir=tmp_path)


class TestMultiprocessing:
    """Tests for multiprocessing functionality."""
    
    def test_get_worker_count(self):
        """Test worker count is 80% of CPU count."""
        from hdmea.features.sta import _get_worker_count
        from multiprocessing import cpu_count
        
        n_workers = _get_worker_count()
        expected = max(1, int(cpu_count() * 0.8))
        
        assert n_workers == expected
        assert n_workers >= 1


class TestErrorHandling:
    """Tests for error handling and retry logic."""
    
    def test_failed_units_tracked(self, tmp_path):
        """Test that failed units are tracked in result."""
        # STAResult should have failed_units list
        result = STAResult(
            hdf5_path=tmp_path / "test.h5",
            movie_name="test",
            units_processed=5,
            units_failed=2,
            cover_range=(-60, 0),
            elapsed_seconds=10.0,
            failed_units=["unit_001", "unit_003"],
        )
        
        assert result.units_failed == 2
        assert "unit_001" in result.failed_units
        assert "unit_003" in result.failed_units
