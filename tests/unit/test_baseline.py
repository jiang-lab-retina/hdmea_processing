"""
Unit tests for baseline feature extractor.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from hdmea.features.baseline.baseline_127 import BaselineFeatureExtractor


class TestBaselineFeatureExtractor:
    """Tests for BaselineFeatureExtractor."""
    
    def test_extractor_has_required_attributes(self):
        """Test that extractor has all required attributes."""
        extractor = BaselineFeatureExtractor()
        
        assert extractor.name == "baseline_127"
        assert "mean_firing_rate" in extractor.output_schema
        assert "quality_index" in extractor.output_schema
    
    def test_extract_with_regular_spikes(self):
        """Test extraction with regular spike train."""
        extractor = BaselineFeatureExtractor()
        
        # Regular spikes at 10 Hz for 10 seconds
        spike_times = np.arange(0, 10_000_000, 100_000).astype(np.uint64)
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: spike_times
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data, config={
            "start_time_us": 0,
            "end_time_us": 10_000_000,
        })
        
        assert 9 < result["mean_firing_rate"] < 11  # Should be ~10 Hz
        assert result["spike_count"] == 100
        assert result["quality_index"] > 0
    
    def test_extract_with_empty_spikes(self):
        """Test extraction with no spikes."""
        extractor = BaselineFeatureExtractor()
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: np.array([], dtype=np.uint64)
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data)
        
        assert result["mean_firing_rate"] == 0.0
        assert result["spike_count"] == 0
        assert result["quality_index"] == 0.0
    
    def test_cv_isi_calculation(self):
        """Test coefficient of variation of ISI."""
        extractor = BaselineFeatureExtractor()
        
        # Regular spikes should have low CV
        regular_spikes = np.arange(0, 10_000_000, 100_000).astype(np.uint64)
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: regular_spikes
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data, config={
            "end_time_us": 10_000_000,
        })
        
        # Perfect regularity should have CV near 0
        assert result["cv_isi"] < 0.1

