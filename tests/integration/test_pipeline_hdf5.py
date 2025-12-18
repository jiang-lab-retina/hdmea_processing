"""
Integration tests for HDF5 pipeline output.

Tests the full pipeline flow from Stage 1 (load) through Stage 2 (extract)
with HDF5 as the storage format.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from hdmea.io.hdf5_store import (
    create_recording_hdf5,
    open_recording_hdf5,
    write_units,
    write_stimulus,
    write_metadata,
    mark_stage1_complete,
    get_stage1_status,
    list_units,
    list_features,
    write_feature_to_unit,
)


class TestHDF5PipelineIntegration:
    """Integration tests for full HDF5 pipeline."""
    
    def test_full_stage1_workflow(self, tmp_path):
        """Test complete Stage 1 workflow: create, write all data, mark complete."""
        hdf5_path = tmp_path / "test_recording.h5"
        
        # Stage 1: Create and populate HDF5
        with create_recording_hdf5(hdf5_path, "test_dataset_001") as f:
            # Write units
            units_data = {
                f"unit_{i:03d}": {
                    "spike_times": np.sort(np.random.randint(0, 1000000, 500).astype(np.uint64)),
                    "waveform": np.random.randn(60).astype(np.float32),
                    "row": i // 10,
                    "col": i % 10,
                    "global_id": i,
                }
                for i in range(10)
            }
            write_units(f, units_data)
            
            # Write stimulus
            light_ref = {
                "raw_ch1": np.random.randn(100000).astype(np.float32),
                "raw_ch2": np.random.randn(100000).astype(np.float32),
            }
            section_times = {
                "movie_a": np.array([[0, 10000], [20000, 30000], [40000, 50000]], dtype=np.uint64),
            }
            write_stimulus(f, light_ref, section_times=section_times)
            
            # Write metadata
            metadata = {
                "acquisition_rate": 20000.0,
                "sample_interval": 0.00005,
                "sys_meta": {
                    "device": "MaxOne",
                    "software_version": "1.0.0",
                },
            }
            write_metadata(f, metadata)
            
            # Mark complete
            mark_stage1_complete(f)
        
        # Verify file was closed and can be reopened
        with open_recording_hdf5(hdf5_path, mode="r") as f:
            # Verify structure
            assert "units" in f
            assert "stimulus" in f
            assert "metadata" in f
            
            # Verify units
            units = list_units(f)
            assert len(units) == 10
            
            # Verify Stage 1 status
            status = get_stage1_status(f)
            assert status["completed"] is True
            
            # Verify spike times round-trip
            original_spikes = units_data["unit_000"]["spike_times"]
            read_spikes = f["units/unit_000/spike_times"][:]
            np.testing.assert_array_equal(read_spikes, original_spikes)
    
    def test_stage2_feature_extraction_workflow(self, tmp_path):
        """Test Stage 2 workflow: open HDF5, extract features, write back."""
        hdf5_path = tmp_path / "test_recording.h5"
        
        # Create Stage 1 data
        with create_recording_hdf5(hdf5_path, "test_dataset_002") as f:
            units_data = {
                "unit_000": {
                    "spike_times": np.array([100, 200, 300, 400, 500], dtype=np.uint64),
                    "waveform": np.random.randn(60).astype(np.float32),
                    "row": 0,
                    "col": 0,
                    "global_id": 0,
                },
            }
            write_units(f, units_data)
            mark_stage1_complete(f)
        
        # Stage 2: Open, compute features, write
        with open_recording_hdf5(hdf5_path, mode="r+") as f:
            # Simulate feature extraction
            unit_id = "unit_000"
            feature_name = "test_feature"
            
            feature_data = {
                "response_curve": np.random.randn(100).astype(np.float32),
                "on_index": 0.75,
                "off_index": 0.25,
            }
            
            feature_metadata = {
                "version": "1.0.0",
                "params_hash": "abc123",
            }
            
            write_feature_to_unit(f, unit_id, feature_name, feature_data, feature_metadata)
        
        # Verify features were written
        with open_recording_hdf5(hdf5_path, mode="r") as f:
            features = list_features(f, "unit_000")
            assert "test_feature" in features
            
            # Verify feature data
            feature_group = f["units/unit_000/features/test_feature"]
            assert "response_curve" in feature_group
            assert feature_group.attrs["on_index"] == 0.75
    
    def test_multiple_features_per_unit(self, tmp_path):
        """Test writing multiple features to the same unit."""
        hdf5_path = tmp_path / "test_recording.h5"
        
        # Create Stage 1 data
        with create_recording_hdf5(hdf5_path, "test_dataset_003") as f:
            units_data = {
                "unit_000": {
                    "spike_times": np.array([100, 200, 300], dtype=np.uint64),
                    "waveform": np.array([]),
                    "row": 0,
                    "col": 0,
                    "global_id": 0,
                },
            }
            write_units(f, units_data)
            mark_stage1_complete(f)
        
        # Write multiple features
        with open_recording_hdf5(hdf5_path, mode="r+") as f:
            for feature_name in ["feature_a", "feature_b", "feature_c"]:
                feature_data = {
                    "value": np.random.rand(),
                    "curve": np.random.randn(50).astype(np.float32),
                }
                write_feature_to_unit(f, "unit_000", feature_name, feature_data, {"version": "1.0"})
        
        # Verify all features exist
        with open_recording_hdf5(hdf5_path, mode="r") as f:
            features = list_features(f, "unit_000")
            assert set(features) == {"feature_a", "feature_b", "feature_c"}
    
    def test_hdf5_file_portability(self, tmp_path):
        """Test that HDF5 file can be read with raw h5py (not just hdmea)."""
        hdf5_path = tmp_path / "test_recording.h5"
        
        # Create file with hdmea
        with create_recording_hdf5(hdf5_path, "test_dataset") as f:
            write_units(f, {
                "unit_000": {
                    "spike_times": np.array([100, 200, 300], dtype=np.uint64),
                    "waveform": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    "row": 5,
                    "col": 10,
                    "global_id": 0,
                },
            })
        
        # Read with raw h5py (simulating external tool like HDFView or MATLAB)
        with h5py.File(str(hdf5_path), "r") as f:
            # Verify standard HDF5 access works
            assert "units" in f
            assert "unit_000" in f["units"]
            
            spikes = f["units/unit_000/spike_times"][:]
            np.testing.assert_array_equal(spikes, [100, 200, 300])
            
            # Verify attributes are accessible
            assert f["units/unit_000"].attrs["row"] == 5
            assert f["units/unit_000"].attrs["col"] == 10

