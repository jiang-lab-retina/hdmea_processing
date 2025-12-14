"""
Unit tests for feature extractor extensibility.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from hdmea.features.registry import FeatureRegistry
from hdmea.features.example.example_feature import ExampleFeatureExtractor


class TestExampleExtractorRegistration:
    """Tests for example extractor registration and functionality."""
    
    def test_example_extractor_is_registered(self):
        """Test that example extractor is registered in registry."""
        assert FeatureRegistry.is_registered("example_feature")
    
    def test_example_extractor_can_be_retrieved(self):
        """Test that example extractor can be retrieved by name."""
        extractor_class = FeatureRegistry.get("example_feature")
        assert extractor_class is ExampleFeatureExtractor
    
    def test_example_extractor_has_metadata(self):
        """Test that example extractor metadata is correct."""
        metadata = FeatureRegistry.get_metadata("example_feature")
        
        assert metadata["name"] == "example_feature"
        assert metadata["version"] == "1.0.0"
        assert "spike_times" in metadata["required_inputs"]
        assert "total_spikes" in metadata["output_schema"]
    
    def test_example_extractor_extract_works(self):
        """Test that example extractor produces valid output."""
        extractor = ExampleFeatureExtractor()
        
        # Create mock data
        spike_times = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.uint64)
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: spike_times
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data)
        
        assert result["total_spikes"] == 5
        assert result["first_spike_time_us"] == 1000
        assert result["last_spike_time_us"] == 5000
        assert result["mean_rate_hz"] > 0
    
    def test_example_extractor_handles_empty_spikes(self):
        """Test that example extractor handles empty spike array."""
        extractor = ExampleFeatureExtractor()
        
        unit_data = MagicMock()
        unit_data.__getitem__ = lambda self, key: np.array([], dtype=np.uint64)
        
        stimulus_data = MagicMock()
        
        result = extractor.extract(unit_data, stimulus_data)
        
        assert result["total_spikes"] == 0
        assert result["mean_rate_hz"] == 0.0


class TestNewExtractorPattern:
    """Tests for adding new extractors via the registry pattern."""
    
    def setup_method(self):
        """Store original registry."""
        self._original = FeatureRegistry._registry.copy()
    
    def teardown_method(self):
        """Restore original registry."""
        FeatureRegistry._registry = self._original
    
    def test_new_extractor_can_be_added(self):
        """Test that a new extractor can be added dynamically."""
        from hdmea.features.base import FeatureExtractor
        
        @FeatureRegistry.register("dynamic_test_feature")
        class DynamicExtractor(FeatureExtractor):
            version = "1.0.0"
            required_inputs = []
            output_schema = {"value": {"dtype": "float64"}}
            
            def extract(self, unit_data, stimulus_data, config=None):
                return {"value": 42.0}
        
        # Verify registration
        assert FeatureRegistry.is_registered("dynamic_test_feature")
        
        # Verify can be used
        extractor = FeatureRegistry.get("dynamic_test_feature")()
        result = extractor.extract(MagicMock(), MagicMock())
        
        assert result["value"] == 42.0
    
    def test_new_extractor_appears_in_list(self):
        """Test that new extractors appear in list_all."""
        from hdmea.features.base import FeatureExtractor
        
        @FeatureRegistry.register("list_test_feature")
        class ListTestExtractor(FeatureExtractor):
            version = "1.0.0"
            required_inputs = []
            output_schema = {}
            
            def extract(self, unit_data, stimulus_data, config=None):
                return {}
        
        all_features = FeatureRegistry.list_all()
        assert "list_test_feature" in all_features

