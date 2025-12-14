"""
Integration tests for end-to-end pipeline flow.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from hdmea.pipeline.flows import run_flow, list_available_flows
from hdmea.io.parquet_export import export_features_to_parquet


class TestRunFlow:
    """Integration tests for run_flow function."""
    
    def test_run_flow_with_mocked_data(self, temp_dir):
        """Test running a flow with mocked file readers."""
        # Create flow config
        flows_dir = temp_dir / "config" / "flows"
        flows_dir.mkdir(parents=True)
        
        flow_config = {
            "name": "test_flow",
            "version": "1.0.0",
            "stages": {
                "load": {"enabled": True},
                "features": {
                    "enabled": True,
                    "feature_sets": ["baseline_127"]
                }
            },
            "defaults": {}
        }
        
        with open(flows_dir / "test_flow.json", "w") as f:
            json.dump(flow_config, f)
        
        # Create mock data
        mock_units = {
            "unit_000": {
                "spike_times": np.arange(0, 10_000_000, 100_000, dtype=np.uint64),
                "waveform": np.random.randn(50).astype(np.float32),
                "row": 10,
                "col": 20,
                "global_id": 0,
            }
        }
        
        with patch("hdmea.io.cmtr.load_cmtr_data") as mock_cmtr, \
             patch("hdmea.io.cmcr.load_cmcr_data") as mock_cmcr, \
             patch("hdmea.utils.validation.validate_path_exists") as mock_path:
            
            mock_cmtr.return_value = {"units": mock_units, "metadata": {}}
            mock_cmcr.return_value = {
                "light_reference": {},
                "metadata": {"recording_duration_s": 10.0},
                "acquisition_rate": 20000
            }
            mock_path.side_effect = lambda p, ft: Path(p)
            
            result = run_flow(
                flow_name="test_flow",
                cmtr_path="/fake/test.cmtr",
                dataset_id="TEST001",
                output_dir=temp_dir / "artifacts",
                config_dir=temp_dir / "config",
            )
            
            assert result.success is True
            assert result.zarr_path.exists()
            
            if result.load_result:
                assert result.load_result.stage1_completed is True
            
            if result.extraction_result:
                assert "baseline_127" in result.extraction_result.features_extracted


class TestListFlows:
    """Tests for list_available_flows."""
    
    def test_list_flows_returns_names(self, temp_dir):
        """Test that list_flows returns flow names."""
        flows_dir = temp_dir / "config" / "flows"
        flows_dir.mkdir(parents=True)
        
        # Create test flows
        for name in ["flow1", "flow2"]:
            with open(flows_dir / f"{name}.json", "w") as f:
                json.dump({"name": name, "stages": {}}, f)
        
        flows = list_available_flows(temp_dir / "config")
        
        assert "flow1" in flows
        assert "flow2" in flows


class TestParquetExport:
    """Integration tests for Parquet export."""
    
    def test_export_creates_parquet(self, temp_dir):
        """Test that export creates valid Parquet file."""
        import zarr
        
        # Create test Zarr with features
        zarr_path = temp_dir / "test.zarr"
        store = zarr.DirectoryStore(str(zarr_path))
        root = zarr.group(store=store)
        
        root.attrs["dataset_id"] = "TEST001"
        root.create_group("units")
        
        unit = root["units"].create_group("unit_001")
        unit.create_dataset("spike_times", data=np.arange(100, dtype=np.uint64))
        unit.attrs["row"] = 10
        unit.attrs["col"] = 20
        unit.attrs["spike_count"] = 100
        
        features = unit.create_group("features")
        baseline = features.create_group("baseline_127")
        baseline.create_dataset("mean_firing_rate", data=np.array(5.0))
        baseline.attrs["version"] = "1.0.0"
        
        # Export
        output_path = temp_dir / "export.parquet"
        result_path = export_features_to_parquet(
            zarr_paths=[zarr_path],
            output_path=output_path,
        )
        
        assert result_path.exists()
        
        # Read back and verify
        import pandas as pd
        df = pd.read_parquet(result_path)
        
        assert len(df) == 1
        assert df.iloc[0]["dataset_id"] == "TEST001"
        assert df.iloc[0]["unit_id"] == "unit_001"

