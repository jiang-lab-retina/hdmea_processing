"""
Unit tests for ON/OFF response feature extractor.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from hdmea.features.on_off.step_up import StepUpFeatureExtractor


class TestStepUpFeatureExtractor:
    """Tests for StepUpFeatureExtractor."""
    
    def test_extractor_has_required_attributes(self):
        """Test that extractor has all required attributes."""
        extractor = StepUpFeatureExtractor()
        
        assert extractor.name == "step_up_5s_5i_3x"
        assert extractor.version == "1.0.0"
        assert "spike_times" in extractor.required_inputs
        assert "on_response_flag" in extractor.output_schema
        assert "off_response_flag" in extractor.output_schema
    
    def test_extract_with_on_response(self):
        """Test extraction with clear ON response."""
        extractor = StepUpFeatureExtractor()
        
        # Create spikes with burst at stimulus onset (5s)
        baseline_spikes = np.arange(0, 5_000_000, 200_000)  # 5 spikes/s baseline
        on_burst = np.arange(5_000_000, 5_500_000, 10_000)  # 100 spikes/s ON burst
        sustained = np.arange(5_500_000, 10_000_000, 100_000)  # 10 spikes/s sustained
        after_spikes = np.arange(10_000_000, 15_000_000, 200_000)
        
        all_spikes = np.concatenate([baseline_spikes, on_burst, sustained, after_spikes])
        all_spikes = np.sort(all_spikes).astype(np.uint64)
        
        # Create mock unit data
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: all_spikes
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data)
        
        assert result["on_response_flag"] is True
        assert result["on_peak_value"] > 0
        assert result["response_quality"] > 0
    
    def test_extract_with_empty_spikes(self):
        """Test extraction with no spikes."""
        extractor = StepUpFeatureExtractor()
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: np.array([], dtype=np.uint64)
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data)
        
        assert result["on_response_flag"] is False
        assert result["off_response_flag"] is False
        assert result["on_peak_value"] == 0.0
    
    def test_on_off_ratio_calculation(self):
        """Test that ON/OFF ratio is calculated correctly."""
        extractor = StepUpFeatureExtractor()
        
        # Create spikes with known ON and OFF responses
        on_burst = np.arange(5_000_000, 5_100_000, 2_000)  # Strong ON
        off_burst = np.arange(10_000_000, 10_050_000, 2_000)  # Weaker OFF
        
        all_spikes = np.concatenate([on_burst, off_burst]).astype(np.uint64)
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: all_spikes
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data)
        
        # ON is stronger, so ratio should be positive
        assert result["on_off_ratio"] > 0
        assert -1 <= result["on_off_ratio"] <= 1

