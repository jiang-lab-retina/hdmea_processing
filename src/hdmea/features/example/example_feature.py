"""
Example feature extractor - Template for creating new extractors.

This file demonstrates the correct pattern for implementing a new feature
extractor. Copy this file and modify it for your own features.

Usage:
    1. Copy this file to a new subpackage under src/hdmea/features/
    2. Rename the class and change the @register decorator name
    3. Modify required_inputs, output_schema, and the extract() method
    4. Add your feature to the appropriate stimulus config
    5. Run tests to verify it works

The feature will be automatically available after import due to the
registry pattern.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import zarr

from hdmea.features.base import FeatureExtractor
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("example_feature")
class ExampleFeatureExtractor(FeatureExtractor):
    """
    Example feature extractor demonstrating the correct pattern.
    
    This extractor computes simple statistics as an example.
    Replace this logic with your actual feature computation.
    
    Attributes:
        name: Unique identifier for this feature (set by @register decorator)
        version: Version string - increment when algorithm changes
        required_inputs: List of Zarr paths this extractor needs
        output_schema: Documentation of output columns
        runtime_class: "fast" (<1s per unit) or "slow" (>1s per unit)
    """
    
    # Required class attributes
    name = "example_feature"  # Will be set by @register
    version = "1.0.0"  # Increment when algorithm changes
    description = "Example feature extractor for documentation"
    
    # Declare what inputs are needed from the Zarr
    # These paths are relative to the root group
    required_inputs = [
        "spike_times",  # Means root["units"][unit_id]["spike_times"]
    ]
    
    # Document the output schema
    # This helps users understand what features are produced
    output_schema = {
        "total_spikes": {
            "dtype": "int64",
            "description": "Total number of spikes",
        },
        "mean_rate_hz": {
            "dtype": "float64",
            "unit": "Hz",
            "description": "Mean firing rate in Hz",
        },
        "first_spike_time_us": {
            "dtype": "uint64",
            "unit": "μs",
            "description": "Time of first spike",
        },
        "last_spike_time_us": {
            "dtype": "uint64",
            "unit": "μs",
            "description": "Time of last spike",
        },
    }
    
    # Runtime class for selective execution
    # "fast" = <1s per unit, "slow" = >1s per unit
    runtime_class = "fast"
    
    def extract(
        self,
        unit_data: zarr.Group,
        stimulus_data: zarr.Group,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract features for a single unit.
        
        This method is called once per unit. It receives:
        - unit_data: The Zarr group for this unit (root["units"][unit_id])
        - stimulus_data: The Zarr group for stimulus info (root["stimulus"])
        - config: Optional runtime configuration overrides
        
        Args:
            unit_data: Zarr group containing unit data
            stimulus_data: Zarr group containing stimulus timing
            config: Optional configuration dict
        
        Returns:
            Dictionary mapping output column names to values.
            Values can be scalars, 1D arrays, or nested dicts.
        
        Note:
            This method MUST be pure: same inputs → same outputs.
            Use explicit random seeds from config for any randomness.
        """
        # Get config with defaults
        config = config or {}
        
        # Load required data from Zarr
        # Note: unit_data["spike_times"] accesses the array
        spike_times = unit_data["spike_times"][:]  # [:] loads into memory
        
        # Handle empty case
        if len(spike_times) == 0:
            return self._empty_result()
        
        # Compute features
        total_spikes = len(spike_times)
        first_spike = spike_times[0]
        last_spike = spike_times[-1]
        
        # Duration in seconds
        duration_us = last_spike - first_spike
        duration_s = duration_us / 1e6 if duration_us > 0 else 1.0
        
        # Mean rate
        mean_rate = total_spikes / duration_s
        
        # Return results
        # Keys must match output_schema
        return {
            "total_spikes": int(total_spikes),
            "mean_rate_hz": float(mean_rate),
            "first_spike_time_us": int(first_spike),
            "last_spike_time_us": int(last_spike),
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """
        Return default values when no spikes are present.
        
        Always implement this to handle edge cases gracefully.
        """
        return {
            "total_spikes": 0,
            "mean_rate_hz": 0.0,
            "first_spike_time_us": 0,
            "last_spike_time_us": 0,
        }

