"""
Smoke tests for HD-MEA pipeline.

Verify that all modules can be imported without errors.
"""

import pytest


class TestPackageImports:
    """Test that the main package and subpackages can be imported."""
    
    def test_import_hdmea(self):
        """Test main package import."""
        import hdmea
        assert hdmea.__version__ == "0.1.0"
    
    def test_import_hdmea_io(self):
        """Test io subpackage import."""
        from hdmea import io
        assert io is not None
    
    def test_import_hdmea_preprocess(self):
        """Test preprocess subpackage import."""
        from hdmea import preprocess
        assert preprocess is not None
    
    def test_import_hdmea_features(self):
        """Test features subpackage import."""
        from hdmea import features
        assert features is not None
    
    def test_import_hdmea_analysis(self):
        """Test analysis subpackage import."""
        from hdmea import analysis
        assert analysis is not None
    
    def test_import_hdmea_viz(self):
        """Test viz subpackage import."""
        from hdmea import viz
        assert viz is not None
    
    def test_import_hdmea_pipeline(self):
        """Test pipeline subpackage import."""
        from hdmea import pipeline
        assert pipeline is not None
    
    def test_import_hdmea_utils(self):
        """Test utils subpackage import."""
        from hdmea import utils
        assert utils is not None


class TestUtilityImports:
    """Test that utility modules can be imported."""
    
    def test_import_logging(self):
        """Test logging utility import."""
        from hdmea.utils.logging import setup_logging, get_logger
        assert setup_logging is not None
        assert get_logger is not None
    
    def test_import_hashing(self):
        """Test hashing utility import."""
        from hdmea.utils.hashing import hash_config, verify_hash
        assert hash_config is not None
        assert verify_hash is not None
    
    def test_import_validation(self):
        """Test validation utility import."""
        from hdmea.utils.validation import validate_dataset_id, validate_input_files
        assert validate_dataset_id is not None
        assert validate_input_files is not None
    
    def test_import_exceptions(self):
        """Test exceptions import."""
        from hdmea.utils.exceptions import (
            HDMEAError,
            ConfigurationError,
            DataLoadError,
            FeatureExtractionError,
            MissingInputError,
            CacheConflictError,
        )
        assert issubclass(ConfigurationError, HDMEAError)
        assert issubclass(DataLoadError, HDMEAError)
        assert issubclass(FeatureExtractionError, HDMEAError)
        assert issubclass(MissingInputError, HDMEAError)
        assert issubclass(CacheConflictError, HDMEAError)


class TestFeatureImports:
    """Test that feature module can be imported."""
    
    def test_import_feature_registry(self):
        """Test FeatureRegistry import."""
        from hdmea.features.registry import FeatureRegistry
        assert FeatureRegistry is not None
        assert hasattr(FeatureRegistry, "register")
        assert hasattr(FeatureRegistry, "get")
        assert hasattr(FeatureRegistry, "list_all")
    
    def test_import_feature_extractor_base(self):
        """Test FeatureExtractor base class import."""
        from hdmea.features.base import FeatureExtractor
        assert FeatureExtractor is not None
        assert hasattr(FeatureExtractor, "extract")


class TestConfigImports:
    """Test that config module can be imported."""
    
    def test_import_config_models(self):
        """Test config model imports."""
        from hdmea.pipeline.config import (
            FlowConfig,
            StimulusConfig,
            PipelineConfig,
            DefaultsConfig,
        )
        assert FlowConfig is not None
        assert StimulusConfig is not None
        assert PipelineConfig is not None
        assert DefaultsConfig is not None
    
    def test_import_config_loaders(self):
        """Test config loader imports."""
        from hdmea.pipeline.config import (
            load_flow_config,
            load_stimulus_config,
            load_defaults,
        )
        assert load_flow_config is not None
        assert load_stimulus_config is not None
        assert load_defaults is not None

