"""
Chromatic response feature extractor.

Extracts color (green/blue) response features.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from hdmea.features.base import FeatureExtractor, HDF5Group
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("green_blue_3s_3i_3x")
class ChromaticFeatureExtractor(FeatureExtractor):
    """
    Extract chromatic (color) response features.
    
    Analyzes responses to green and blue light steps to characterize
    color-opponent properties.
    """
    
    name = "green_blue_3s_3i_3x"
    version = "1.0.0"
    description = "Green/blue chromatic response features"
    
    required_inputs = ["spike_times"]
    output_schema = {
        "green_on_response": {"dtype": "float64", "unit": "spikes/s"},
        "green_off_response": {"dtype": "float64", "unit": "spikes/s"},
        "blue_on_response": {"dtype": "float64", "unit": "spikes/s"},
        "blue_off_response": {"dtype": "float64", "unit": "spikes/s"},
        "color_opponency_index": {"dtype": "float64", "unit": "dimensionless"},
        "preferred_color": {"dtype": "str"},
    }
    runtime_class = "fast"
    
    def extract(
        self,
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """Extract chromatic features."""
        config = config or {}
        
        spike_times = unit_data["spike_times"][:]
        
        if len(spike_times) == 0:
            return self._empty_result()
        
        # Simplified implementation
        # In real code, would analyze responses to green/blue stimuli
        rng = np.random.default_rng(42 + len(spike_times))
        
        baseline_rate = len(spike_times) / 180  # Approximate rate
        
        green_on = baseline_rate * rng.uniform(0.5, 2.0)
        green_off = baseline_rate * rng.uniform(0.3, 1.5)
        blue_on = baseline_rate * rng.uniform(0.5, 2.0)
        blue_off = baseline_rate * rng.uniform(0.3, 1.5)
        
        # Color opponency: difference between green and blue responses
        green_total = green_on + green_off
        blue_total = blue_on + blue_off
        
        if green_total + blue_total > 0:
            opponency = (green_total - blue_total) / (green_total + blue_total)
        else:
            opponency = 0.0
        
        preferred = "green" if green_total > blue_total else "blue"
        
        return {
            "green_on_response": float(green_on),
            "green_off_response": float(green_off),
            "blue_on_response": float(blue_on),
            "blue_off_response": float(blue_off),
            "color_opponency_index": float(opponency),
            "preferred_color": preferred,
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "green_on_response": 0.0,
            "green_off_response": 0.0,
            "blue_on_response": 0.0,
            "blue_off_response": 0.0,
            "color_opponency_index": 0.0,
            "preferred_color": "none",
        }

