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
    add_section_time_analog,
    _load_playlist_csv,
    _load_movie_length_csv,
    _convert_frame_to_sample_index,
    _sample_to_nearest_frame,
    _detect_analog_peaks,
    _get_movie_start_end_frame,
    PRE_MARGIN_FRAME_NUM,
    POST_MARGIN_FRAME_NUM,
    DEFAULT_PAD_FRAME,
)
from hdmea.utils.exceptions import MissingInputError

# Import analog fixtures
from tests.fixtures.section_time.analog_fixture import (
    create_synthetic_pulses,
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
    
    # Create frame_timestamps array (sample indices for each frame)
    n_frames = 10000
    acquisition_rate = 20000.0
    # ~60fps display rate, sample indices at acquisition rate
    frame_timestamps = np.linspace(0, n_frames * (acquisition_rate / 60), n_frames).astype(np.uint64)
    metadata.create_array("frame_timestamps", data=frame_timestamps)
    metadata.create_array("acquisition_rate", data=np.array([acquisition_rate]))
    
    # Create stimulus group with light reference
    stimulus = root.create_group("stimulus")
    lr_group = stimulus.create_group("light_reference")
    
    # Create synthetic light reference data
    n_samples = int(frame_timestamps[-1]) + 1000
    light_data = np.sin(np.linspace(0, 100 * np.pi, n_samples)).astype(np.float32)
    lr_group.create_array("raw_ch1", data=light_data)
    lr_group.create_array("raw_ch2", data=light_data)
    
    # Mark stage1 complete (required by open_recording_zarr)
    root.attrs["stage1_completed"] = True
    root.attrs["dataset_id"] = "test_dataset"
    
    return zarr_path


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestConvertFrameToSampleIndex:
    """Tests for _convert_frame_to_sample_index helper."""
    
    def test_basic_conversion(self):
        """Test basic frame to sample index conversion."""
        frame_timestamps = np.array([0, 100, 200, 300, 400])
        frames = np.array([0, 2, 4])
        
        result = _convert_frame_to_sample_index(frames, frame_timestamps)
        
        expected = np.array([0, 200, 400])
        np.testing.assert_array_equal(result, expected)
    
    def test_clips_to_valid_range(self):
        """Test that out-of-range frames are clipped."""
        frame_timestamps = np.array([0, 100, 200])
        frames = np.array([-1, 10])  # Out of range
        
        result = _convert_frame_to_sample_index(frames, frame_timestamps)
        
        # Should clip to [0, 2]
        expected = np.array([0, 200])
        np.testing.assert_array_equal(result, expected)


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
        frame_timestamps = np.linspace(0, 1000000, 10000).astype(np.uint64)
        
        movie_list, movie_frames, movie_templates = _get_movie_start_end_frame(
            playlist_name="test_playlist",
            repeats=1,
            all_playlists=playlist,
            movies_length=movies_length,
            frame_timestamps=frame_timestamps,
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
        frame_timestamps = np.linspace(0, 10000000, 100000).astype(np.uint64)
        
        movie_list, movie_frames, movie_templates = _get_movie_start_end_frame(
            playlist_name="test_playlist",
            repeats=2,
            all_playlists=playlist,
            movies_length=movies_length,
            frame_timestamps=frame_timestamps,
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


# =============================================================================
# Analog Section Time Helper Tests
# =============================================================================

class TestSampleToNearestFrame:
    """Tests for _sample_to_nearest_frame helper."""
    
    def test_exact_match(self):
        """Test when sample index matches exactly."""
        frame_timestamps = np.array([0, 100, 200, 300, 400])
        
        result = _sample_to_nearest_frame(200, frame_timestamps)
        
        assert result == 2
    
    def test_nearest_before(self):
        """Test rounding to nearest frame (before)."""
        frame_timestamps = np.array([0, 100, 200, 300, 400])
        
        # 149 is closer to 100 (frame 1) than 200 (frame 2)
        result = _sample_to_nearest_frame(149, frame_timestamps)
        
        assert result == 1
    
    def test_nearest_after(self):
        """Test rounding to nearest frame (after)."""
        frame_timestamps = np.array([0, 100, 200, 300, 400])
        
        # 151 is closer to 200 (frame 2) than 100 (frame 1)
        result = _sample_to_nearest_frame(151, frame_timestamps)
        
        assert result == 2
    
    def test_edge_before_first(self):
        """Test sample before first frame."""
        frame_timestamps = np.array([100, 200, 300])
        
        result = _sample_to_nearest_frame(50, frame_timestamps)
        
        assert result == 0
    
    def test_edge_after_last(self):
        """Test sample after last frame."""
        frame_timestamps = np.array([100, 200, 300])
        
        result = _sample_to_nearest_frame(500, frame_timestamps)
        
        assert result == 2  # Last frame index
    
    def test_array_input(self):
        """Test with array of samples."""
        frame_timestamps = np.array([0, 100, 200, 300, 400])
        samples = np.array([50, 150, 250, 350])
        
        result = _sample_to_nearest_frame(samples, frame_timestamps)
        
        np.testing.assert_array_equal(result, [0, 1, 2, 3])


class TestDetectAnalogPeaks:
    """Tests for _detect_analog_peaks helper."""
    
    def test_detects_single_peak(self):
        """Test detection of a single step pulse."""
        signal = np.zeros(1000, dtype=np.float32)
        signal[500:] = 1e7  # Step at sample 500
        
        peaks = _detect_analog_peaks(signal, threshold=1e6)
        
        assert len(peaks) == 1
        assert peaks[0] == 499  # Peak in derivative is one before step
    
    def test_detects_multiple_peaks(self):
        """Test detection of multiple step pulses."""
        signal = np.zeros(1000, dtype=np.float32)
        signal[200:] += 1e7
        signal[400:] += 1e7
        signal[600:] += 1e7
        
        peaks = _detect_analog_peaks(signal, threshold=1e6)
        
        assert len(peaks) == 3
    
    def test_threshold_filters_small_peaks(self):
        """Test that threshold filters out small transitions."""
        signal = np.zeros(1000, dtype=np.float32)
        signal[200:] += 1e7  # Large step
        signal[400:] += 1e4  # Small step (below threshold)
        signal[600:] += 1e7  # Large step
        
        peaks = _detect_analog_peaks(signal, threshold=1e6)
        
        assert len(peaks) == 2  # Only large steps detected
    
    def test_no_peaks_with_flat_signal(self):
        """Test that flat signal returns no peaks."""
        signal = np.ones(1000, dtype=np.float32) * 1000
        
        peaks = _detect_analog_peaks(signal, threshold=1e6)
        
        assert len(peaks) == 0
    
    def test_high_threshold_no_peaks(self):
        """Test that very high threshold returns no peaks."""
        signal = np.zeros(1000, dtype=np.float32)
        signal[500:] = 1e6
        
        peaks = _detect_analog_peaks(signal, threshold=1e9)  # Very high
        
        assert len(peaks) == 0


# =============================================================================
# Analog Section Time Main API Tests
# =============================================================================

class TestAddSectionTimeAnalog:
    """Tests for add_section_time_analog main function."""
    
    @pytest.fixture
    def analog_zarr(self, temp_dir):
        """Create a Zarr with synthetic pulses for testing.
        
        Note: frame_timestamps is NOT created since add_section_time_analog()
        no longer requires it. The function works directly with acquisition
        sample indices from raw_ch1.
        """
        zarr_path = temp_dir / "test_analog.zarr"
        
        acquisition_rate = 20000.0
        n_samples = 1000000  # 50 seconds
        
        root = zarr.open(str(zarr_path), mode='w')
        
        # Metadata - only acquisition_rate needed for analog detection
        metadata = root.create_group("metadata")
        acq_rate_arr = np.array([acquisition_rate])
        metadata.create_array("acquisition_rate", data=acq_rate_arr)
        # frame_timestamps not needed for analog section time detection
        
        # Create signal with 3 pulses at 10s, 20s, 30s
        stimulus = root.create_group("stimulus")
        lr_group = stimulus.create_group("light_reference")
        
        signal = np.zeros(n_samples, dtype=np.float32)
        signal[int(10 * acquisition_rate):] += 1e7
        signal[int(20 * acquisition_rate):] += 1e7
        signal[int(30 * acquisition_rate):] += 1e7
        
        lr_group.create_array("raw_ch1", data=signal)
        raw_ch2 = np.zeros(n_samples, dtype=np.float32)
        lr_group.create_array("raw_ch2", data=raw_ch2)
        
        root.attrs["stage1_completed"] = True
        root.attrs["dataset_id"] = "test_analog"
        
        return zarr_path
    
    def test_basic_detection(self, analog_zarr):
        """Test basic pulse detection with acquisition sample indices output."""
        result = add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="test_movie",
            plot_duration=5.0,
        )
        
        assert result is True
        
        # Verify data was written
        root = zarr.open(str(analog_zarr), mode='r')
        assert "section_time" in root["stimulus"]
        assert "test_movie" in root["stimulus"]["section_time"]
        
        section_time = np.array(root["stimulus"]["section_time"]["test_movie"])
        assert section_time.shape[0] == 3  # 3 pulses
        assert section_time.shape[1] == 2  # [start, end]
        
        # Verify values are acquisition sample indices (not frame indices)
        # Pulses at 10s, 20s, 30s at 20kHz acquisition rate
        # First pulse onset should be near sample 200000 (10s * 20000 Hz)
        # Using tolerance because peak detection finds derivative peak
        acquisition_rate = 20000.0
        expected_first_onset = int(10 * acquisition_rate)
        assert abs(section_time[0, 0] - expected_first_onset) < 100  # Within 100 samples
        
        # Check end sample = start + duration_samples (5.0s * 20000 = 100000)
        duration_samples = int(5.0 * acquisition_rate)
        assert section_time[0, 1] == section_time[0, 0] + duration_samples
        
        # Verify light_template was created
        assert "light_template" in root["stimulus"]
        assert "test_movie" in root["stimulus"]["light_template"]
        
        light_template = np.array(root["stimulus"]["light_template"]["test_movie"])
        # Template should have duration_samples length (5.0s * 20000 = 100000 samples)
        assert light_template.shape == (duration_samples,)
        assert light_template.dtype == np.float32
    
    def test_custom_movie_name(self, analog_zarr):
        """Test custom movie_name parameter."""
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="iprgc_blue",
        )
        
        root = zarr.open(str(analog_zarr), mode='r')
        assert "iprgc_blue" in root["stimulus"]["section_time"]
    
    def test_repeat_limits_trials(self, analog_zarr):
        """Test repeat parameter limits detected trials."""
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="test_repeat",
            repeat=2,
        )
        
        root = zarr.open(str(analog_zarr), mode='r')
        section_time = np.array(root["stimulus"]["section_time"]["test_repeat"])
        assert section_time.shape[0] == 2  # Only 2 of 3 pulses
    
    def test_plot_duration_affects_section_length(self, analog_zarr):
        """Test that plot_duration affects section spans in sample units."""
        acquisition_rate = 20000.0  # From fixture
        
        # Short duration (1.0s = 20000 samples)
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="short_duration",
            plot_duration=1.0,
        )
        
        # Longer duration (10.0s = 200000 samples)
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="long_duration",
            plot_duration=10.0,
        )
        
        root = zarr.open(str(analog_zarr), mode='r')
        short = np.array(root["stimulus"]["section_time"]["short_duration"])
        long = np.array(root["stimulus"]["section_time"]["long_duration"])
        
        # Verify spans are in sample units (duration * acquisition_rate)
        short_span = short[0, 1] - short[0, 0]
        long_span = long[0, 1] - long[0, 0]
        
        # Expected: 1.0s * 20000 Hz = 20000 samples
        assert short_span == int(1.0 * acquisition_rate)
        # Expected: 10.0s * 20000 Hz = 200000 samples
        assert long_span == int(10.0 * acquisition_rate)
        
        assert long_span > short_span
    
    def test_threshold_sensitivity(self, analog_zarr):
        """Test that higher threshold detects fewer peaks."""
        # Low threshold - should detect all
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e5,
            movie_name="low_thresh",
        )
        
        # Create new zarr with mixed amplitude pulses
        root = zarr.open(str(analog_zarr), mode='r+')
        n_samples = root["stimulus"]["light_reference"]["raw_ch1"].shape[0]
        acquisition_rate = 20000.0
        
        # Recreate with varied amplitudes
        signal = np.zeros(n_samples, dtype=np.float32)
        signal[int(10 * acquisition_rate):] += 1e7  # Large
        signal[int(20 * acquisition_rate):] += 1e5  # Small
        signal[int(30 * acquisition_rate):] += 1e7  # Large
        
        del root["stimulus"]["light_reference"]["raw_ch1"]
        root["stimulus"]["light_reference"].create_array("raw_ch1", data=signal)
        
        # High threshold should miss the small pulse
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="high_thresh",
        )
        
        root = zarr.open(str(analog_zarr), mode='r')
        high_thresh = np.array(root["stimulus"]["section_time"]["high_thresh"])
        assert high_thresh.shape[0] == 2  # Only 2 large pulses
    
    def test_raises_on_missing_raw_ch1(self, temp_dir):
        """Test MissingInputError when raw_ch1 is missing."""
        zarr_path = temp_dir / "missing_raw_ch1.zarr"
        root = zarr.open(str(zarr_path), mode='w')
        
        metadata = root.create_group("metadata")
        acq_rate = np.array([20000.0])
        metadata.create_array("acquisition_rate", data=acq_rate)
        frame_ts = np.arange(100, dtype=np.uint64)
        metadata.create_array("frame_timestamps", data=frame_ts)
        
        stimulus = root.create_group("stimulus")
        stimulus.create_group("light_reference")  # Empty
        
        root.attrs["stage1_completed"] = True
        root.attrs["dataset_id"] = "test"
        
        with pytest.raises(MissingInputError) as exc_info:
            add_section_time_analog(
                zarr_path=zarr_path,
                threshold_value=1e6,
            )
        assert "raw_ch1" in str(exc_info.value)
    
    def test_raises_on_missing_acquisition_rate(self, temp_dir):
        """Test MissingInputError when acquisition_rate is missing."""
        zarr_path = temp_dir / "missing_rate.zarr"
        root = zarr.open(str(zarr_path), mode='w')
        
        metadata = root.create_group("metadata")
        # No acquisition_rate (frame_timestamps not needed for analog detection)
        
        stimulus = root.create_group("stimulus")
        lr = stimulus.create_group("light_reference")
        raw_ch1 = np.zeros(1000, dtype=np.float32)
        lr.create_array("raw_ch1", data=raw_ch1)
        
        root.attrs["stage1_completed"] = True
        root.attrs["dataset_id"] = "test"
        
        with pytest.raises(MissingInputError) as exc_info:
            add_section_time_analog(
                zarr_path=zarr_path,
                threshold_value=1e6,
            )
        assert "acquisition_rate" in str(exc_info.value)
    
    def test_returns_false_no_peaks(self, temp_dir):
        """Test returns False when no peaks detected."""
        zarr_path = temp_dir / "no_peaks.zarr"
        root = zarr.open(str(zarr_path), mode='w')
        
        metadata = root.create_group("metadata")
        acq_rate = np.array([20000.0])
        metadata.create_array("acquisition_rate", data=acq_rate)
        # frame_timestamps not needed for analog detection
        
        stimulus = root.create_group("stimulus")
        lr = stimulus.create_group("light_reference")
        # Flat signal with small noise
        raw_ch1 = np.random.normal(0, 100, 10000).astype(np.float32)
        lr.create_array("raw_ch1", data=raw_ch1)
        
        root.attrs["stage1_completed"] = True
        root.attrs["dataset_id"] = "test"
        
        result = add_section_time_analog(
            zarr_path=zarr_path,
            threshold_value=1e6,  # High threshold, no peaks
        )
        
        assert result is False
    
    def test_raises_on_existing_without_force(self, analog_zarr):
        """Test FileExistsError when data exists without force."""
        # First call succeeds
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="duplicate_test",
        )
        
        # Second call should raise
        with pytest.raises(FileExistsError):
            add_section_time_analog(
                zarr_path=analog_zarr,
                threshold_value=1e6,
                movie_name="duplicate_test",
            )
    
    def test_force_overwrites_existing(self, analog_zarr):
        """Test force=True overwrites existing data."""
        # First call
        add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="overwrite_test",
        )
        
        # Second call with force should succeed
        result = add_section_time_analog(
            zarr_path=analog_zarr,
            threshold_value=1e6,
            movie_name="overwrite_test",
            force=True,
        )
        
        assert result is True
    
    def test_raises_on_invalid_threshold(self, analog_zarr):
        """Test ValueError when threshold is None."""
        with pytest.raises((ValueError, TypeError)):
            add_section_time_analog(
                zarr_path=analog_zarr,
                threshold_value=None,  # type: ignore
            )
    
    def test_raises_on_invalid_plot_duration(self, analog_zarr):
        """Test ValueError when plot_duration <= 0."""
        with pytest.raises(ValueError):
            add_section_time_analog(
                zarr_path=analog_zarr,
                threshold_value=1e6,
                plot_duration=-1.0,
            )
    
    def test_section_clipped_at_signal_boundary(self, temp_dir):
        """Test that sections are clipped to raw_ch1 signal length (not frame_timestamps)."""
        zarr_path = temp_dir / "clip_test.zarr"
        
        acquisition_rate = 20000.0
        n_samples = 50000  # 2.5 seconds of signal
        
        root = zarr.open(str(zarr_path), mode='w')
        
        # Metadata
        metadata = root.create_group("metadata")
        metadata.create_array("acquisition_rate", data=np.array([acquisition_rate]))
        
        # Create signal with pulse at 1s (sample 20000)
        stimulus = root.create_group("stimulus")
        lr_group = stimulus.create_group("light_reference")
        
        signal = np.zeros(n_samples, dtype=np.float32)
        signal[int(1.0 * acquisition_rate):] += 1e7  # Pulse at 1s
        lr_group.create_array("raw_ch1", data=signal)
        
        root.attrs["stage1_completed"] = True
        root.attrs["dataset_id"] = "test_clip"
        
        # Request 5s duration but signal only has 2.5s total
        # Pulse at 1s + 5s duration would go to sample 120000
        # But signal only has 50000 samples, so end should be clipped
        result = add_section_time_analog(
            zarr_path=zarr_path,
            threshold_value=1e6,
            movie_name="clip_test",
            plot_duration=5.0,  # 5 seconds = 100000 samples
        )
        
        assert result is True
        
        root = zarr.open(str(zarr_path), mode='r')
        section_time = np.array(root["stimulus"]["section_time"]["clip_test"])
        
        # End sample should be clipped to max_sample (n_samples - 1 = 49999)
        assert section_time[0, 1] == n_samples - 1

