"""
Base class for feature extractors.

All feature extractors MUST inherit from FeatureExtractor and implement
the required interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
import logging

import zarr


logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """
    Base class for all feature extractors.
    
    Feature extractors MUST:
    - Declare name, version, required_inputs, output_schema, runtime_class
    - Implement the extract() method
    - Be pure functions: same input → same output
    
    Example:
        @FeatureRegistry.register("my_feature")
        class MyFeatureExtractor(FeatureExtractor):
            name = "my_feature"
            version = "1.0.0"
            required_inputs = ["spike_times", "stimulus/light_reference"]
            output_schema = {
                "value": {"dtype": "float64", "unit": "spikes/s"}
            }
            runtime_class = "fast"
            
            def extract(self, unit_data, stimulus_data, config=None):
                # Implementation
                return {"value": computed_value}
    """
    
    # Required class attributes (MUST be overridden)
    name: str = ""
    version: str = "0.0.0"
    required_inputs: List[str] = []
    output_schema: Dict[str, Dict[str, Any]] = {}
    runtime_class: Literal["fast", "slow"] = "fast"
    
    # Optional class attributes
    description: str = ""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def extract(
        self,
        unit_data: zarr.Group,
        stimulus_data: zarr.Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[zarr.Group] = None,
    ) -> Dict[str, Any]:
        """
        Extract features for a single unit.
        
        Args:
            unit_data: Zarr group for the unit (contains spike_times, waveform, etc.)
            stimulus_data: Zarr group with stimulus information
            config: Optional runtime configuration overrides
            metadata: Optional Zarr group with recording metadata (acquisition_rate, etc.)
        
        Returns:
            Dictionary mapping feature names to values (scalars or arrays)
        
        Note:
            This method MUST be pure: given the same inputs, it MUST produce
            identical outputs. Random operations MUST use explicit seeds
            from the config.
        """
        pass
    
    def validate_inputs(self, root: zarr.Group) -> List[str]:
        """
        Check that all required inputs are present in the Zarr.
        
        Args:
            root: Root Zarr group for the recording
        
        Returns:
            List of missing input paths (empty if all present)
        """
        missing = []
        for input_path in self.required_inputs:
            try:
                # Navigate path (supports nested paths like "stimulus/light_reference")
                parts = input_path.split("/")
                current = root
                for part in parts:
                    current = current[part]
            except KeyError:
                missing.append(input_path)
        return missing
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for this extractor.
        
        Returns:
            Dictionary with extractor metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "required_inputs": self.required_inputs,
            "output_schema": self.output_schema,
            "runtime_class": self.runtime_class,
            "description": self.description,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"

