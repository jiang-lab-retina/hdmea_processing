"""
Integration tests for load_recording (Stage 1).

These tests use synthetic data and mocked McsPy to test the full loading flow.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from hdmea.pipeline.runner import load_recording, LoadResult
from hdmea.io.zarr_store import open_recording_zarr, list_units


class TestLoadRecordingIntegration:
    """Integration tests for the load_recording function."""
    
    def test_loads_without_real_files(self, temp_dir):
        """Test that load_recording works with mocked file readers."""
        # Create mock CMTR data
        mock_units = {
            "unit_000": {
                "spike_times": np.array([1000, 2000, 3000], dtype=np.uint64),
                "waveform": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "row": 10,
                "col": 20,
                "global_id": 0,
            }
        }
        
        mock_light_ref = {
            "20kHz": np.random.randn(1000).astype(np.float32),
        }
        
        with patch("hdmea.io.cmtr.load_cmtr_data") as mock_cmtr, \
             patch("hdmea.io.cmcr.load_cmcr_data") as mock_cmcr, \
             patch("hdmea.utils.validation.validate_path_exists") as mock_path:
            
            # Configure mocks
            mock_cmtr.return_value = {
                "units": mock_units,
                "metadata": {},
                "source_path": "/fake/path.cmtr",
            }
            mock_cmcr.return_value = {
                "light_reference": mock_light_ref,
                "metadata": {"recording_duration_s": 10.0},
                "acquisition_rate": 20000,
                "source_path": "/fake/path.cmcr",
            }
            mock_path.side_effect = lambda p, ft: Path(p)
            
            # Run load_recording
            result = load_recording(
                cmcr_path="/fake/path.cmcr",
                cmtr_path="/fake/path.cmtr",
                dataset_id="TEST001_2025-01-01",
                output_dir=temp_dir,
            )
            
            # Verify result
            assert isinstance(result, LoadResult)
            assert result.stage1_completed is True
            assert result.num_units == 1
            assert result.zarr_path.exists()
            
            # Verify Zarr contents
            root = open_recording_zarr(result.zarr_path)
            assert "units" in root
            assert "unit_000" in root["units"]
            assert "stimulus" in root
    
    def test_caching_skips_reload(self, temp_dir):
        """Test that cached Zarr is reused on second call."""
        mock_units = {
            "unit_000": {
                "spike_times": np.array([1000], dtype=np.uint64),
                "waveform": np.array([1.0], dtype=np.float32),
                "row": 0,
                "col": 0,
                "global_id": 0,
            }
        }
        
        with patch("hdmea.io.cmtr.load_cmtr_data") as mock_cmtr, \
             patch("hdmea.io.cmcr.load_cmcr_data") as mock_cmcr, \
             patch("hdmea.utils.validation.validate_path_exists") as mock_path:
            
            mock_cmtr.return_value = {"units": mock_units, "metadata": {}}
            mock_cmcr.return_value = {"light_reference": {}, "metadata": {}, "acquisition_rate": 20000}
            mock_path.side_effect = lambda p, ft: Path(p)
            
            # First call
            result1 = load_recording(
                cmtr_path="/fake/path.cmtr",
                dataset_id="TEST002",
                output_dir=temp_dir,
            )
            
            # Second call (should use cache)
            result2 = load_recording(
                cmtr_path="/fake/path.cmtr",
                dataset_id="TEST002",
                output_dir=temp_dir,
            )
            
            # CMTR should only be called once
            assert mock_cmtr.call_count == 1
            assert result2.stage1_completed is True
            assert "cached" in result2.warnings[0].lower()


class TestLoadRecordingValidation:
    """Tests for input validation in load_recording."""
    
    def test_requires_at_least_one_file(self, temp_dir):
        """Test that at least one of cmcr/cmtr must be provided."""
        from hdmea.utils.exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError, match="At least one"):
            load_recording(output_dir=temp_dir)
    
    def test_validates_dataset_id_format(self, temp_dir):
        """Test that invalid dataset_id format is rejected."""
        from hdmea.utils.exceptions import ConfigurationError
        
        with patch("hdmea.utils.validation.validate_path_exists") as mock_path:
            mock_path.side_effect = lambda p, ft: Path(p)
            
            with pytest.raises(ConfigurationError, match="Invalid dataset_id"):
                load_recording(
                    cmtr_path="/fake/path.cmtr",
                    dataset_id="invalid-format",
                    output_dir=temp_dir,
                )

