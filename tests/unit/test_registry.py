"""
Unit tests for FeatureRegistry.
"""

import pytest

from hdmea.features.registry import FeatureRegistry
from hdmea.features.base import FeatureExtractor


class TestFeatureRegistry:
    """Tests for the FeatureRegistry class."""
    
    def setup_method(self):
        """Clear registry before each test."""
        # Store original registry
        self._original_registry = FeatureRegistry._registry.copy()
    
    def teardown_method(self):
        """Restore original registry after each test."""
        FeatureRegistry._registry = self._original_registry
    
    def test_register_decorator(self):
        """Test that @register decorator adds class to registry."""
        @FeatureRegistry.register("test_feature")
        class TestExtractor(FeatureExtractor):
            version = "1.0.0"
            required_inputs = []
            output_schema = {}
            
            def extract(self, unit_data, stimulus_data, config=None):
                return {}
        
        assert "test_feature" in FeatureRegistry.list_all()
        assert FeatureRegistry.get("test_feature") is TestExtractor
    
    def test_duplicate_registration_raises(self):
        """Test that registering same name twice raises error."""
        @FeatureRegistry.register("duplicate_feature")
        class Extractor1(FeatureExtractor):
            version = "1.0.0"
            required_inputs = []
            output_schema = {}
            
            def extract(self, unit_data, stimulus_data, config=None):
                return {}
        
        with pytest.raises(ValueError, match="already registered"):
            @FeatureRegistry.register("duplicate_feature")
            class Extractor2(FeatureExtractor):
                version = "1.0.0"
                required_inputs = []
                output_schema = {}
                
                def extract(self, unit_data, stimulus_data, config=None):
                    return {}
    
    def test_get_unknown_feature_raises(self):
        """Test that getting unknown feature raises KeyError."""
        with pytest.raises(KeyError, match="Unknown feature"):
            FeatureRegistry.get("nonexistent_feature")
    
    def test_list_all_returns_sorted(self):
        """Test that list_all returns sorted feature names."""
        features = FeatureRegistry.list_all()
        assert features == sorted(features)
    
    def test_get_metadata(self):
        """Test that get_metadata returns correct info."""
        @FeatureRegistry.register("metadata_test")
        class MetadataExtractor(FeatureExtractor):
            version = "2.0.0"
            required_inputs = ["spike_times"]
            output_schema = {"value": {"dtype": "float64"}}
            runtime_class = "slow"
            
            def extract(self, unit_data, stimulus_data, config=None):
                return {"value": 1.0}
        
        metadata = FeatureRegistry.get_metadata("metadata_test")
        
        assert metadata["name"] == "metadata_test"
        assert metadata["version"] == "2.0.0"
        assert metadata["required_inputs"] == ["spike_times"]
        assert metadata["runtime_class"] == "slow"
    
    def test_is_registered(self):
        """Test is_registered method."""
        @FeatureRegistry.register("registered_feature")
        class RegisteredExtractor(FeatureExtractor):
            version = "1.0.0"
            required_inputs = []
            output_schema = {}
            
            def extract(self, unit_data, stimulus_data, config=None):
                return {}
        
        assert FeatureRegistry.is_registered("registered_feature") is True
        assert FeatureRegistry.is_registered("not_registered") is False

