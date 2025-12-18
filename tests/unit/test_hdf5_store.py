"""
Unit tests for HDF5 store operations.

Tests the hdmea.io.hdf5_store module which provides HDF5 file operations
for the HD-MEA pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import h5py

from hdmea.io.hdf5_store import (
    create_recording_hdf5,
    open_recording_hdf5,
    write_units,
    write_stimulus,
    write_metadata,
    write_source_files,
    mark_stage1_complete,
    get_stage1_status,
    list_units,
    list_features,
    write_feature_to_unit,
)


class TestCreateRecordingHdf5:
    """Tests for create_recording_hdf5()."""
    
    def test_creates_file_with_correct_structure(self, tmp_path):
        """Test that created HDF5 has expected groups."""
        hdf5_path = tmp_path / "test.h5"
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            assert "units" in f
            assert "stimulus" in f
            assert "metadata" in f
            assert f.attrs["dataset_id"] == "test_dataset"
    
    def test_sets_required_attributes(self, tmp_path):
        """Test that all required root attributes are set."""
        hdf5_path = tmp_path / "test.h5"
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            assert "dataset_id" in f.attrs
            assert "hdmea_pipeline_version" in f.attrs
            assert "created_at" in f.attrs
            assert "updated_at" in f.attrs
            assert "stage1_completed" in f.attrs
            assert f.attrs["stage1_completed"] == False
    
    def test_raises_if_file_exists_and_no_overwrite(self, tmp_path):
        """Test FileExistsError when file exists and overwrite=False."""
        hdf5_path = tmp_path / "test.h5"
        
        # Create file first
        with create_recording_hdf5(hdf5_path, "test_dataset"):
            pass
        
        # Try to create again without overwrite
        with pytest.raises(FileExistsError):
            create_recording_hdf5(hdf5_path, "test_dataset", overwrite=False)
    
    def test_overwrites_existing_file(self, tmp_path):
        """Test that overwrite=True replaces existing file."""
        hdf5_path = tmp_path / "test.h5"
        
        # Create file first
        with create_recording_hdf5(hdf5_path, "first_dataset"):
            pass
        
        # Overwrite with new dataset
        with create_recording_hdf5(hdf5_path, "second_dataset", overwrite=True) as f:
            assert f.attrs["dataset_id"] == "second_dataset"


class TestOpenRecordingHdf5:
    """Tests for open_recording_hdf5()."""
    
    def test_opens_existing_file(self, tmp_path):
        """Test opening an existing HDF5 file."""
        hdf5_path = tmp_path / "test.h5"
        
        # Create file first
        with create_recording_hdf5(hdf5_path, "test_dataset"):
            pass
        
        # Open and verify
        with open_recording_hdf5(hdf5_path, mode="r") as f:
            assert f.attrs["dataset_id"] == "test_dataset"
    
    def test_raises_if_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent file."""
        hdf5_path = tmp_path / "nonexistent.h5"
        
        with pytest.raises(FileNotFoundError):
            open_recording_hdf5(hdf5_path)


class TestWriteUnits:
    """Tests for write_units()."""
    
    def test_writes_unit_data(self, tmp_path):
        """Test writing spike times and waveforms for units."""
        hdf5_path = tmp_path / "test.h5"
        
        units_data = {
            "unit_000": {
                "spike_times": np.array([100, 200, 300], dtype=np.uint64),
                "waveform": np.random.randn(50).astype(np.float32),
                "row": 5,
                "col": 10,
                "global_id": 0,
            },
        }
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_units(f, units_data)
            
            # Verify unit was written
            assert "unit_000" in f["units"]
            assert "spike_times" in f["units/unit_000"]
            assert "waveform" in f["units/unit_000"]
            assert "features" in f["units/unit_000"]
            
            # Verify data
            np.testing.assert_array_equal(
                f["units/unit_000/spike_times"][:],
                units_data["unit_000"]["spike_times"]
            )
            
            # Verify attributes
            assert f["units/unit_000"].attrs["row"] == 5
            assert f["units/unit_000"].attrs["col"] == 10
            assert f["units/unit_000"].attrs["spike_count"] == 3


class TestWriteStimulus:
    """Tests for write_stimulus()."""
    
    def test_writes_light_reference(self, tmp_path):
        """Test writing light reference data."""
        hdf5_path = tmp_path / "test.h5"
        
        light_ref = {
            "raw_ch1": np.random.randn(1000).astype(np.float32),
        }
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_stimulus(f, light_ref)
            
            assert "light_reference" in f["stimulus"]
            assert "raw_ch1" in f["stimulus/light_reference"]
    
    def test_writes_section_times(self, tmp_path):
        """Test writing section time boundaries."""
        hdf5_path = tmp_path / "test.h5"
        
        section_times = {
            "movie_a": np.array([[0, 100], [200, 300]], dtype=np.uint64),
        }
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_stimulus(f, {}, section_times=section_times)
            
            assert "section_time" in f["stimulus"]
            assert "movie_a" in f["stimulus/section_time"]
            np.testing.assert_array_equal(
                f["stimulus/section_time/movie_a"][:],
                section_times["movie_a"]
            )


class TestWriteMetadata:
    """Tests for write_metadata()."""
    
    def test_writes_scalar_metadata(self, tmp_path):
        """Test writing scalar values as single-element datasets."""
        hdf5_path = tmp_path / "test.h5"
        
        metadata = {
            "acquisition_rate": 20000.0,
            "frame_time": 0.01,
        }
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_metadata(f, metadata)
            
            assert "acquisition_rate" in f["metadata"]
            assert f["metadata/acquisition_rate"][0] == 20000.0


class TestRoundTrip:
    """Tests for data round-trip (write then read)."""
    
    def test_spike_times_round_trip(self, tmp_path):
        """Test that spike times are preserved exactly."""
        hdf5_path = tmp_path / "test.h5"
        
        original_spikes = np.array([100, 200, 300, 500, 1000], dtype=np.uint64)
        
        units_data = {
            "unit_000": {
                "spike_times": original_spikes,
                "waveform": np.array([]),
                "row": 0,
                "col": 0,
                "global_id": 0,
            },
        }
        
        # Write
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_units(f, units_data)
        
        # Read back
        with open_recording_hdf5(hdf5_path, mode="r") as f:
            read_spikes = f["units/unit_000/spike_times"][:]
            np.testing.assert_array_equal(read_spikes, original_spikes)
            assert read_spikes.dtype == np.uint64


class TestWriteFeatureToUnit:
    """Tests for write_feature_to_unit()."""
    
    def test_writes_feature_data(self, tmp_path):
        """Test writing feature data for a unit."""
        hdf5_path = tmp_path / "test.h5"
        
        units_data = {
            "unit_000": {
                "spike_times": np.array([100], dtype=np.uint64),
                "waveform": np.array([]),
                "row": 0,
                "col": 0,
                "global_id": 0,
            },
        }
        
        feature_data = {
            "on_index": 0.85,
            "response_curve": np.random.randn(100).astype(np.float32),
        }
        
        metadata = {
            "version": "1.0.0",
            "params_hash": "abc123",
        }
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_units(f, units_data)
            write_feature_to_unit(f, "unit_000", "step_up", feature_data, metadata)
            
            assert "step_up" in f["units/unit_000/features"]
            assert f["units/unit_000/features/step_up"].attrs["on_index"] == 0.85
            assert "response_curve" in f["units/unit_000/features/step_up"]


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_list_units(self, tmp_path):
        """Test listing all unit IDs."""
        hdf5_path = tmp_path / "test.h5"
        
        units_data = {
            "unit_000": {"spike_times": np.array([100], dtype=np.uint64), "waveform": np.array([]), "row": 0, "col": 0, "global_id": 0},
            "unit_001": {"spike_times": np.array([200], dtype=np.uint64), "waveform": np.array([]), "row": 0, "col": 1, "global_id": 1},
        }
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_units(f, units_data)
            units = list_units(f)
            
            assert len(units) == 2
            assert "unit_000" in units
            assert "unit_001" in units
    
    def test_mark_stage1_complete(self, tmp_path):
        """Test marking Stage 1 as complete."""
        hdf5_path = tmp_path / "test.h5"
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            assert f.attrs["stage1_completed"] == False
            
            mark_stage1_complete(f)
            
            assert f.attrs["stage1_completed"] == True
    
    def test_get_stage1_status(self, tmp_path):
        """Test getting Stage 1 status."""
        hdf5_path = tmp_path / "test.h5"
        
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            status = get_stage1_status(f)
            
            assert status["completed"] == False
            assert "params_hash" in status
            assert "created_at" in status

