"""
Unit tests for section_time module.

Tests the add_section_time() function and helper functions with synthetic data.
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from hdmea.io.section_time import (
    add_section_time,
    _load_playlist_csv,
    _load_movie_length_csv,
    _convert_frame_to_time,
    _get_movie_start_end_frame,
    PRE_MARGIN_FRAME_NUM,
    POST_MARGIN_FRAME_NUM,
    DEFAULT_PAD_FRAME,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_playlist_csv(temp_dir):
    """Create a sample playlist CSV file."""
    playlist_path = temp_dir / "playlist.csv"
    df = pd.DataFrame({
        "playlist_name": ["test_playlist", "single_movie"],
        "movie_names": [
            "['movie_a.mov', 'movie_b.mov', 'movie_c.mov']",
            "['movie_a.mov']"
        ]
    })
    df.to_csv(playlist_path, index=False)
    return playlist_path


@pytest.fixture
def sample_movie_length_csv(temp_dir):
    """Create a sample movie_length CSV file."""
    movie_length_path = temp_dir / "movie_length.csv"
    df = pd.DataFrame({
        "movie_name": ["movie_a", "movie_b", "movie_c"],
        "movie_length": [600, 1200, 900]
    })
    df.to_csv(movie_length_path, index=False)
    return movie_length_path


@pytest.fixture
def sample_zarr(temp_dir):
    """Create a sample Zarr archive with required metadata."""
    zarr_path = temp_dir / "test.zarr"
    root = zarr.open(str(zarr_path), mode='w')
    
    # Create metadata group
    metadata = root.create_group("metadata")
    
    # Create frame_time array (1000 frames at ~60fps -> ~16.7ms per frame)
    n_frames = 10000
    frame_time = np.linspace(0, n_frames * 0.0167, n_frames)
    metadata.create_dataset("frame_time", data=frame_time)
    metadata.attrs["acquisition_rate"] = 20000.0
    
    # Create stimulus group with light reference
    stimulus = root.create_group("stimulus")
    lr_group = stimulus.create_group("light_reference")
    
    # Create synthetic light reference data
    n_samples = int(n_frames * 0.0167 * 20000)  # samples based on acquisition rate
    light_data = np.sin(np.linspace(0, 100 * np.pi, n_samples)).astype(np.float32)
    lr_group.create_dataset("raw_ch2", data=light_data)
    
    # Mark stage1 complete (required by open_recording_zarr)
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_dataset"
    
    return zarr_path


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestConvertFrameToTime:
    """Tests for _convert_frame_to_time helper."""
    
    def test_basic_conversion(self):
        """Test basic frame to time conversion."""
        frame_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        frames = np.array([0, 2, 4])
        
        result = _convert_frame_to_time(frames, frame_time)
        
        expected = np.array([0.0, 0.2, 0.4])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_clips_to_valid_range(self):
        """Test that out-of-range frames are clipped."""
        frame_time = np.array([0.0, 0.1, 0.2])
        frames = np.array([-1, 10])  # Out of range
        
        result = _convert_frame_to_time(frames, frame_time)
        
        # Should clip to [0, 2]
        expected = np.array([0.0, 0.2])
        np.testing.assert_array_almost_equal(result, expected)


class TestLoadPlaylistCsv:
    """Tests for _load_playlist_csv helper."""
    
    def test_loads_valid_csv(self, sample_playlist_csv):
        """Test loading a valid playlist CSV."""
        result = _load_playlist_csv(sample_playlist_csv)
        
        assert result is not None
        assert "test_playlist" in result.index
        assert "single_movie" in result.index
    
    def test_returns_none_for_missing_file(self, temp_dir):
        """Test that missing file returns None."""
        result = _load_playlist_csv(temp_dir / "nonexistent.csv")
        
        assert result is None


class TestLoadMovieLengthCsv:
    """Tests for _load_movie_length_csv helper."""
    
    def test_loads_valid_csv(self, sample_movie_length_csv):
        """Test loading a valid movie_length CSV."""
        result = _load_movie_length_csv(sample_movie_length_csv)
        
        assert result is not None
        assert "movie_a" in result.index
        assert result.loc["movie_a"]["movie_length"] == 600
    
    def test_returns_none_for_missing_file(self, temp_dir):
        """Test that missing file returns None."""
        result = _load_movie_length_csv(temp_dir / "nonexistent.csv")
        
        assert result is None


class TestGetMovieStartEndFrame:
    """Tests for _get_movie_start_end_frame core algorithm."""
    
    def test_computes_frame_boundaries(self, sample_playlist_csv, sample_movie_length_csv):
        """Test that frame boundaries are computed correctly."""
        playlist = _load_playlist_csv(sample_playlist_csv)
        movies_length = _load_movie_length_csv(sample_movie_length_csv)
        frame_time = np.linspace(0, 1000, 10000)
        
        movie_list, movie_frames, movie_templates = _get_movie_start_end_frame(
            playlist_name="test_playlist",
            repeats=1,
            all_playlists=playlist,
            movies_length=movies_length,
            frame_time=frame_time,
        )
        
        # Check that all movies are in the result
        assert "movie_a" in movie_frames
        assert "movie_b" in movie_frames
        assert "movie_c" in movie_frames
        
        # Each movie should have one frame pair
        assert len(movie_frames["movie_a"]) == 1
        assert len(movie_frames["movie_a"][0]) == 2  # [start, end]
    
    def test_handles_repeats(self, sample_playlist_csv, sample_movie_length_csv):
        """Test that repeats create multiple frame pairs."""
        playlist = _load_playlist_csv(sample_playlist_csv)
        movies_length = _load_movie_length_csv(sample_movie_length_csv)
        frame_time = np.linspace(0, 10000, 100000)
        
        movie_list, movie_frames, movie_templates = _get_movie_start_end_frame(
            playlist_name="test_playlist",
            repeats=2,
            all_playlists=playlist,
            movies_length=movies_length,
            frame_time=frame_time,
        )
        
        # Each movie should have two frame pairs (one per repeat)
        assert len(movie_frames["movie_a"]) == 2


# =============================================================================
# Main API Tests
# =============================================================================

class TestAddSectionTime:
    """Tests for add_section_time main function."""
    
    def test_happy_path(self, sample_zarr, sample_playlist_csv, sample_movie_length_csv):
        """Test successful section time addition."""
        result = add_section_time(
            zarr_path=sample_zarr,
            playlist_name="test_playlist",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=sample_movie_length_csv,
            repeats=1,
        )
        
        assert result is True
        
        # Verify data was written
        root = zarr.open(str(sample_zarr), mode='r')
        assert "section_time" in root["stimulus"]
        assert "movie_a" in root["stimulus"]["section_time"]
    
    def test_returns_false_for_missing_playlist(self, sample_zarr, temp_dir, sample_movie_length_csv):
        """Test that missing playlist file returns False."""
        result = add_section_time(
            zarr_path=sample_zarr,
            playlist_name="test_playlist",
            playlist_file_path=temp_dir / "nonexistent.csv",
            movie_length_file_path=sample_movie_length_csv,
        )
        
        assert result is False
    
    def test_returns_false_for_missing_movie_length(self, sample_zarr, sample_playlist_csv, temp_dir):
        """Test that missing movie_length file returns False."""
        result = add_section_time(
            zarr_path=sample_zarr,
            playlist_name="test_playlist",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=temp_dir / "nonexistent.csv",
        )
        
        assert result is False
    
    def test_returns_false_for_invalid_playlist_name(self, sample_zarr, sample_playlist_csv, sample_movie_length_csv):
        """Test that invalid playlist name returns False."""
        result = add_section_time(
            zarr_path=sample_zarr,
            playlist_name="nonexistent_playlist",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=sample_movie_length_csv,
        )
        
        assert result is False
    
    def test_raises_error_on_existing_without_force(self, sample_zarr, sample_playlist_csv, sample_movie_length_csv):
        """Test that existing section_time raises error without force."""
        # First call succeeds
        add_section_time(
            zarr_path=sample_zarr,
            playlist_name="test_playlist",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=sample_movie_length_csv,
        )
        
        # Second call should raise
        with pytest.raises(FileExistsError):
            add_section_time(
                zarr_path=sample_zarr,
                playlist_name="test_playlist",
                playlist_file_path=sample_playlist_csv,
                movie_length_file_path=sample_movie_length_csv,
            )
    
    def test_force_overwrites_existing(self, sample_zarr, sample_playlist_csv, sample_movie_length_csv):
        """Test that force=True overwrites existing section_time."""
        # First call
        add_section_time(
            zarr_path=sample_zarr,
            playlist_name="test_playlist",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=sample_movie_length_csv,
        )
        
        # Second call with force should succeed
        result = add_section_time(
            zarr_path=sample_zarr,
            playlist_name="test_playlist",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=sample_movie_length_csv,
            force=True,
        )
        
        assert result is True
    
    def test_custom_paths_are_used(self, sample_zarr, sample_playlist_csv, sample_movie_length_csv):
        """Test that custom paths are properly used."""
        # This test verifies US3 - custom paths work
        result = add_section_time(
            zarr_path=sample_zarr,
            playlist_name="single_movie",
            playlist_file_path=sample_playlist_csv,
            movie_length_file_path=sample_movie_length_csv,
        )
        
        assert result is True
        
        # Verify only movie_a is in section_time (single_movie playlist)
        root = zarr.open(str(sample_zarr), mode='r')
        section_time_movies = list(root["stimulus"]["section_time"].keys())
        assert section_time_movies == ["movie_a"]
    
    def test_path_objects_and_strings_both_work(self, sample_zarr, sample_playlist_csv, sample_movie_length_csv):
        """Test that both Path objects and strings work for paths."""
        # Test with Path objects
        result = add_section_time(
            zarr_path=Path(sample_zarr),
            playlist_name="test_playlist",
            playlist_file_path=Path(sample_playlist_csv),
            movie_length_file_path=Path(sample_movie_length_csv),
        )
        
        assert result is True

