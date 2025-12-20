"""
End-to-end integration test for eimage_sta feature.

Tests the full workflow with real test data to validate:
- Correct computation and storage
- Performance target (<5 minutes)
- Output shape validation
"""

import time
from pathlib import Path

import h5py
import numpy as np
import pytest


# Test data paths - update these to match your test environment
TEST_CMCR_PATH = Path("O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr")
TEST_CMTR_PATH = Path("O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr")
TEST_HDF5_PATH = Path("artifacts/2025.04.10-11.12.57-Rec.h5")


def check_test_data_available():
    """Check if test data files exist."""
    return TEST_CMCR_PATH.exists() and TEST_HDF5_PATH.exists()


@pytest.mark.skipif(
    not check_test_data_available(),
    reason="Test data not available"
)
class TestEImageSTAEndToEnd:
    """End-to-end tests for eimage_sta computation."""
    
    def test_compute_eimage_sta_basic(self):
        """Test basic eimage_sta computation with default parameters."""
        from hdmea.features.eimage_sta import compute_eimage_sta
        
        result = compute_eimage_sta(
            hdf5_path=TEST_HDF5_PATH,
            cmcr_path=TEST_CMCR_PATH,
            force=True,  # Overwrite for testing
        )
        
        assert result.units_processed > 0, "Should process at least one unit"
        assert result.units_failed == 0, f"No units should fail: {result.failed_units}"
        
        # Verify data was written to HDF5
        with h5py.File(TEST_HDF5_PATH, "r") as f:
            unit_ids = list(f["units"].keys())
            assert len(unit_ids) > 0
            
            # Check first unit has eimage_sta
            first_unit = unit_ids[0]
            eimage_path = f"units/{first_unit}/features/eimage_sta/data"
            assert eimage_path in f, f"eimage_sta not found for {first_unit}"
    
    def test_performance_target(self):
        """Test that computation completes within 5 minute target."""
        from hdmea.features.eimage_sta import compute_eimage_sta
        
        start_time = time.time()
        
        result = compute_eimage_sta(
            hdf5_path=TEST_HDF5_PATH,
            cmcr_path=TEST_CMCR_PATH,
            duration_s=120.0,  # 120 seconds as per spec
            force=True,
        )
        
        elapsed = time.time() - start_time
        
        # Performance target: <5 minutes (300 seconds)
        assert elapsed < 300, (
            f"Computation took {elapsed:.1f}s, exceeds 5 minute target. "
            f"Filter time: {result.filter_time_seconds:.1f}s"
        )
        
        print(f"\nPerformance: {elapsed:.1f}s total, {result.filter_time_seconds:.1f}s filtering")
        print(f"Units processed: {result.units_processed}")
    
    def test_output_shape(self):
        """Test that output has expected shape (window_length, 64, 64)."""
        from hdmea.features.eimage_sta import compute_eimage_sta
        
        pre_samples = 10
        post_samples = 40
        expected_window = pre_samples + post_samples
        
        result = compute_eimage_sta(
            hdf5_path=TEST_HDF5_PATH,
            cmcr_path=TEST_CMCR_PATH,
            pre_samples=pre_samples,
            post_samples=post_samples,
            force=True,
        )
        
        with h5py.File(TEST_HDF5_PATH, "r") as f:
            unit_ids = list(f["units"].keys())
            
            for unit_id in unit_ids[:5]:  # Check first 5 units
                data_path = f"units/{unit_id}/features/eimage_sta/data"
                if data_path in f:
                    data = f[data_path][:]
                    
                    assert data.shape[0] == expected_window, (
                        f"Expected window length {expected_window}, got {data.shape[0]}"
                    )
                    assert data.shape[1] == 64, f"Expected 64 rows, got {data.shape[1]}"
                    assert data.shape[2] == 64, f"Expected 64 cols, got {data.shape[2]}"
    
    def test_metadata_storage(self):
        """Test that metadata is correctly stored."""
        from hdmea.features.eimage_sta import compute_eimage_sta
        
        result = compute_eimage_sta(
            hdf5_path=TEST_HDF5_PATH,
            cmcr_path=TEST_CMCR_PATH,
            cutoff_hz=150.0,  # Custom value to verify
            filter_order=3,
            spike_limit=5000,
            force=True,
        )
        
        with h5py.File(TEST_HDF5_PATH, "r") as f:
            unit_ids = list(f["units"].keys())
            first_unit = unit_ids[0]
            
            group_path = f"units/{first_unit}/features/eimage_sta"
            if group_path in f:
                attrs = f[group_path].attrs
                
                assert attrs["cutoff_hz"] == 150.0
                assert attrs["filter_order"] == 3
                assert attrs["spike_limit"] == 5000
                assert "n_spikes" in attrs
                assert "version" in attrs
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        from hdmea.features.eimage_sta import compute_eimage_sta
        
        # Run twice with same parameters
        result1 = compute_eimage_sta(
            hdf5_path=TEST_HDF5_PATH,
            cmcr_path=TEST_CMCR_PATH,
            spike_limit=100,  # Small limit for faster test
            force=True,
        )
        
        # Read result
        with h5py.File(TEST_HDF5_PATH, "r") as f:
            unit_ids = list(f["units"].keys())
            first_unit = unit_ids[0]
            data1 = f[f"units/{first_unit}/features/eimage_sta/data"][:]
        
        # Run again
        result2 = compute_eimage_sta(
            hdf5_path=TEST_HDF5_PATH,
            cmcr_path=TEST_CMCR_PATH,
            spike_limit=100,
            force=True,
        )
        
        # Read result again
        with h5py.File(TEST_HDF5_PATH, "r") as f:
            data2 = f[f"units/{first_unit}/features/eimage_sta/data"][:]
        
        # Results should be identical
        np.testing.assert_array_equal(data1, data2, "Results should be reproducible")


class TestEImageSTASynthetic:
    """Unit tests with synthetic data (no real files needed)."""
    
    def test_compute_sta_for_unit_basic(self):
        """Test STA computation with synthetic data."""
        from hdmea.features.eimage_sta.compute import compute_sta_for_unit
        from tests.fixtures.eimage_sta_fixtures import generate_test_data_pair
        
        sensor_data, spike_samples = generate_test_data_pair(
            n_samples=1000,
            n_spikes=20,
        )
        
        # Convert to float32 for filtered data
        filtered_data = sensor_data.astype(np.float32)
        
        sta, n_used, n_excluded = compute_sta_for_unit(
            filtered_data,
            spike_samples,
            pre_samples=10,
            post_samples=40,
        )
        
        assert sta.shape == (50, 64, 64)
        assert n_used > 0
        assert n_used + n_excluded == len(spike_samples)
    
    def test_compute_sta_for_unit_no_valid_spikes(self):
        """Test STA computation with all spikes at edges."""
        from hdmea.features.eimage_sta.compute import compute_sta_for_unit
        
        # Create small data where all spikes are at edges
        filtered_data = np.random.randn(100, 64, 64).astype(np.float32)
        spike_samples = np.array([5, 95])  # Too close to edges
        
        sta, n_used, n_excluded = compute_sta_for_unit(
            filtered_data,
            spike_samples,
            pre_samples=10,
            post_samples=40,
        )
        
        assert n_used == 0
        assert n_excluded == 2
        assert np.all(np.isnan(sta))
    
    def test_compute_sta_for_unit_spike_limit(self):
        """Test spike limiting."""
        from hdmea.features.eimage_sta.compute import compute_sta_for_unit
        from tests.fixtures.eimage_sta_fixtures import generate_test_data_pair
        
        sensor_data, spike_samples = generate_test_data_pair(
            n_samples=2000,
            n_spikes=100,
        )
        
        filtered_data = sensor_data.astype(np.float32)
        
        sta, n_used, n_excluded = compute_sta_for_unit(
            filtered_data,
            spike_samples,
            pre_samples=10,
            post_samples=40,
            spike_limit=20,  # Only use 20 spikes
        )
        
        # Should use at most 20 spikes (some may be excluded for edge effects)
        assert n_used <= 20


class TestHighpassFilter:
    """Tests for vectorized high-pass filter."""
    
    def test_apply_highpass_filter_3d_basic(self):
        """Test basic filter operation."""
        from hdmea.preprocess.filtering import apply_highpass_filter_3d
        
        # Create test data
        data = np.random.randn(1000, 8, 8).astype(np.float32)
        
        filtered = apply_highpass_filter_3d(
            data,
            cutoff_hz=100.0,
            sampling_rate=20000.0,
            filter_order=2,
        )
        
        assert filtered.shape == data.shape
        assert filtered.dtype == np.float32
    
    def test_apply_highpass_filter_3d_validation(self):
        """Test parameter validation."""
        from hdmea.preprocess.filtering import apply_highpass_filter_3d
        
        data = np.random.randn(100, 8, 8).astype(np.float32)
        
        # Invalid cutoff
        with pytest.raises(ValueError, match="cutoff_hz must be positive"):
            apply_highpass_filter_3d(data, cutoff_hz=-10, sampling_rate=20000)
        
        # Invalid order
        with pytest.raises(ValueError, match="filter_order must be >= 1"):
            apply_highpass_filter_3d(data, cutoff_hz=100, sampling_rate=20000, filter_order=0)
        
        # Cutoff >= Nyquist
        with pytest.raises(ValueError, match="must be less than Nyquist"):
            apply_highpass_filter_3d(data, cutoff_hz=15000, sampling_rate=20000)

