"""
Unit tests for Zarr store operations.
"""

import pytest
import numpy as np
from pathlib import Path

from hdmea.io.zarr_store import (
    create_recording_zarr,
    open_recording_zarr,
    write_units,
    write_stimulus,
    write_metadata,
    mark_stage1_complete,
    get_stage1_status,
    list_units,
    list_features,
    write_feature_to_unit,
)


class TestCreateRecordingZarr:
    """Tests for create_recording_zarr."""
    
    def test_creates_zarr_with_structure(self, temp_zarr_path):
        """Test that Zarr is created with correct structure."""
        root = create_recording_zarr(
            temp_zarr_path,
            dataset_id="TEST001",
        )
        
        assert "units" in root
        assert "stimulus" in root
        assert "metadata" in root
        assert root.attrs["dataset_id"] == "TEST001"
        assert root.attrs["stage1_completed"] is False
    
    def test_raises_if_exists_without_overwrite(self, temp_zarr_path):
        """Test that existing Zarr raises error without overwrite."""
        create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        
        with pytest.raises(FileExistsError):
            create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
    
    def test_overwrites_with_flag(self, temp_zarr_path):
        """Test that existing Zarr can be overwritten."""
        create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        root = create_recording_zarr(
            temp_zarr_path,
            dataset_id="TEST002",
            overwrite=True,
        )
        
        assert root.attrs["dataset_id"] == "TEST002"


class TestWriteUnits:
    """Tests for write_units."""
    
    def test_writes_unit_data(self, temp_zarr_path):
        """Test writing unit data to Zarr."""
        root = create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        
        units_data = {
            "unit_001": {
                "spike_times": np.array([1000, 2000, 3000], dtype=np.uint64),
                "waveform": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "row": 10,
                "col": 20,
                "global_id": 1,
            }
        }
        
        write_units(root, units_data)
        
        assert "unit_001" in root["units"]
        unit = root["units"]["unit_001"]
        assert np.array_equal(unit["spike_times"][:], [1000, 2000, 3000])
        assert unit.attrs["row"] == 10
        assert unit.attrs["col"] == 20
        assert "features" in unit


class TestWriteStimulus:
    """Tests for write_stimulus."""
    
    def test_writes_light_reference(self, temp_zarr_path):
        """Test writing light reference to Zarr."""
        root = create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        
        light_ref = {
            "20kHz": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "10Hz": np.array([0.15], dtype=np.float32),
        }
        
        write_stimulus(root, light_ref)
        
        assert "light_reference" in root["stimulus"]
        assert "20kHz" in root["stimulus"]["light_reference"]


class TestStage1Status:
    """Tests for stage1 completion status."""
    
    def test_mark_complete(self, temp_zarr_path):
        """Test marking Stage 1 as complete."""
        root = create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        
        assert root.attrs["stage1_completed"] is False
        
        mark_stage1_complete(root)
        
        assert root.attrs["stage1_completed"] is True
    
    def test_get_status(self, temp_zarr_path):
        """Test getting Stage 1 status."""
        root = create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        mark_stage1_complete(root)
        
        status = get_stage1_status(root)
        
        assert status["completed"] is True
        assert status["params_hash"] is not None


class TestFeatureOperations:
    """Tests for feature reading/writing."""
    
    def test_write_and_list_features(self, temp_zarr_path):
        """Test writing and listing features."""
        root = create_recording_zarr(temp_zarr_path, dataset_id="TEST001")
        
        # Create a unit
        units_data = {
            "unit_001": {
                "spike_times": np.array([1000, 2000], dtype=np.uint64),
                "waveform": np.array([1.0, 2.0], dtype=np.float32),
                "row": 0,
                "col": 0,
                "global_id": 1,
            }
        }
        write_units(root, units_data)
        
        # Write a feature
        feature_data = {
            "on_response_flag": True,
            "on_peak_value": 25.5,
        }
        metadata = {
            "feature_name": "step_up_5s_5i_3x",
            "extractor_version": "1.0.0",
        }
        
        write_feature_to_unit(
            root, "unit_001", "step_up_5s_5i_3x", feature_data, metadata
        )
        
        features = list_features(root, "unit_001")
        assert "step_up_5s_5i_3x" in features

