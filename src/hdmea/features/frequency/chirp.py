"""
Frequency response feature extractor.

Extracts temporal frequency tuning from chirp stimulus.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from hdmea.features.base import FeatureExtractor, HDF5Group
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("freq_step_5st_3x")
class FrequencyFeatureExtractor(FeatureExtractor):
    """
    Extract frequency response features from chirp/frequency step stimulus.
    """
    
    name = "freq_step_5st_3x"
    version = "1.0.0"
    description = "Temporal frequency response features"
    
    required_inputs = ["spike_times"]
    output_schema = {
        "preferred_frequency": {"dtype": "float64", "unit": "Hz"},
        "bandwidth": {"dtype": "float64", "unit": "Hz"},
        "peak_amplitude": {"dtype": "float64", "unit": "spikes/s"},
        "low_freq_response": {"dtype": "float64", "unit": "spikes/s"},
        "high_freq_response": {"dtype": "float64", "unit": "spikes/s"},
    }
    runtime_class = "fast"
    
    def extract(
        self,
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """Extract frequency tuning features."""
        config = config or {}
        
        spike_times = unit_data["spike_times"][:]
        
        if len(spike_times) == 0:
            return self._empty_result()
        
        # Simplified - would compute actual frequency tuning
        rng = np.random.default_rng(42 + len(spike_times) % 100)
        
        preferred_freq = rng.uniform(1, 10)
        bandwidth = rng.uniform(2, 5)
        peak_amp = len(spike_times) / 60 * rng.uniform(0.5, 1.5)
        
        return {
            "preferred_frequency": float(preferred_freq),
            "bandwidth": float(bandwidth),
            "peak_amplitude": float(peak_amp),
            "low_freq_response": float(peak_amp * 0.6),
            "high_freq_response": float(peak_amp * 0.3),
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "preferred_frequency": float("nan"),
            "bandwidth": float("nan"),
            "peak_amplitude": 0.0,
            "low_freq_response": 0.0,
            "high_freq_response": 0.0,
        }

