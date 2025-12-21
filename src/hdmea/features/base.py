"""
Base class for feature extractors.

All feature extractors MUST inherit from FeatureExtractor and implement
the required interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union
import logging

import h5py
import numpy as np


logger = logging.getLogger(__name__)

# Type alias for HDF5 groups (compatible with both File and Group)
HDF5Group = Union[h5py.File, h5py.Group]


# =============================================================================
# Dictionary Adapters for Session-Mode Feature Extraction
# =============================================================================

class ArrayAdapter:
    """
    Wraps a value to support [:] slice access like HDF5 datasets.
    
    This allows Python values (scalars, numpy arrays, lists) to be accessed
    using the same syntax as HDF5 datasets: value[:] or value[0].
    
    Example:
        >>> adapter = ArrayAdapter(np.array([1, 2, 3]))
        >>> adapter[:]  # Returns array([1, 2, 3])
        >>> adapter[0]  # Returns 1
        
        >>> scalar_adapter = ArrayAdapter(20000.0)
        >>> scalar_adapter[:]  # Returns array(20000.0)
    """
    
    def __init__(self, value: Any):
        self._value = value
    
    def __getitem__(self, key):
        if isinstance(self._value, np.ndarray):
            return self._value[key]
        elif key == slice(None):  # [:] access
            # Convert scalars and lists to numpy arrays
            return np.asarray(self._value)
        elif isinstance(key, int):
            # For scalar values, just return the value for index 0
            if isinstance(self._value, (int, float, np.number)):
                if key == 0:
                    return self._value
                raise IndexError(f"index {key} is out of bounds for scalar")
            # For sequences, index directly
            return self._value[key]
        return self._value
    
    @property
    def shape(self):
        """Return shape like numpy arrays."""
        if isinstance(self._value, np.ndarray):
            return self._value.shape
        elif isinstance(self._value, (list, tuple)):
            return (len(self._value),)
        else:
            return ()
    
    @property
    def size(self):
        """Return size like numpy arrays."""
        if isinstance(self._value, np.ndarray):
            return self._value.size
        elif isinstance(self._value, (list, tuple)):
            return len(self._value)
        else:
            return 1
    
    @property
    def flat(self):
        """Return flat iterator like numpy arrays."""
        if isinstance(self._value, np.ndarray):
            return self._value.flat
        else:
            return np.asarray(self._value).flat


class DictAdapter:
    """
    Wraps a Python dict to behave like an HDF5 Group.
    
    This enables feature extractors to use the same code for both HDF5 
    and session data without modification. Supports:
    - Subscript access: adapter["key"]
    - Containment check: "key" in adapter
    - Nested dict access (returns nested DictAdapter)
    - Array/scalar value access (returns ArrayAdapter)
    
    Example:
        >>> data = {"spike_times": np.array([1, 2, 3]), "rate": 20000.0}
        >>> adapter = DictAdapter(data)
        >>> adapter["spike_times"][:]  # Returns array([1, 2, 3])
        >>> adapter["rate"][:]  # Returns array(20000.0)
        >>> "spike_times" in adapter  # Returns True
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getitem__(self, key: str):
        value = self._data[key]
        if isinstance(value, dict):
            return DictAdapter(value)
        return ArrayAdapter(value)
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def keys(self):
        """Return keys like HDF5 groups."""
        return self._data.keys()
    
    def items(self):
        """Return items like HDF5 groups."""
        return self._data.items()
    
    def get(self, key: str, default: Any = None):
        """Get with default like dicts."""
        return self._data.get(key, default)


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
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """
        Extract features for a single unit.
        
        Args:
            unit_data: HDF5 group for the unit (contains spike_times, waveform, etc.)
            stimulus_data: HDF5 group with stimulus information
            config: Optional runtime configuration overrides
            metadata: Optional HDF5 group with recording metadata (acquisition_rate, etc.)
        
        Returns:
            Dictionary mapping feature names to values (scalars or arrays)
        
        Note:
            This method MUST be pure: given the same inputs, it MUST produce
            identical outputs. Random operations MUST use explicit seeds
            from the config.
        """
        pass
    
    def validate_inputs(self, root: HDF5Group) -> List[str]:
        """
        Check that all required inputs are present in the HDF5 file.
        
        Args:
            root: Root HDF5 group for the recording
        
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

