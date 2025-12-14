"""
Cell type classification feature extractor.

Classifies cells as RGC vs non-RGC based on network STA properties.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import zarr

from hdmea.features.base import FeatureExtractor
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("cell_type_classifier")
class CellTypeFeatureExtractor(FeatureExtractor):
    """
    Classify cell type based on electrophysiological properties.
    
    Uses axon propagation pattern and waveform properties to
    distinguish RGCs from other cell types.
    """
    
    name = "cell_type_classifier"
    version = "1.0.0"
    description = "RGC vs non-RGC classification"
    
    required_inputs = ["spike_times", "waveform"]
    output_schema = {
        "cell_type": {"dtype": "str", "description": "RGC or unknown"},
        "rgc_probability": {"dtype": "float64", "range": [0, 1]},
        "has_axon": {"dtype": "bool"},
        "waveform_width_ms": {"dtype": "float64", "unit": "ms"},
    }
    runtime_class = "fast"
    
    def extract(
        self,
        unit_data: zarr.Group,
        stimulus_data: zarr.Group,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify cell type."""
        config = config or {}
        
        # Load data
        spike_times = unit_data["spike_times"][:]
        
        # Load waveform if available
        try:
            waveform = unit_data["waveform"][:]
        except KeyError:
            waveform = np.array([])
        
        if len(spike_times) < 50:
            return self._empty_result()
        
        # Compute waveform width if available
        if len(waveform) > 10:
            # Find trough-to-peak width
            trough_idx = np.argmin(waveform)
            peak_idx = np.argmax(waveform[trough_idx:]) + trough_idx
            width_samples = peak_idx - trough_idx
            # Assuming 20kHz sampling rate
            waveform_width_ms = width_samples / 20.0
        else:
            waveform_width_ms = float("nan")
        
        # Simple classification based on properties
        # RGCs typically have narrow waveforms and axon propagation
        has_axon = waveform_width_ms < 0.4 if not np.isnan(waveform_width_ms) else False
        
        # RGC probability based on heuristics
        if has_axon and len(spike_times) > 100:
            rgc_prob = 0.8
        elif len(spike_times) > 500:
            rgc_prob = 0.6
        else:
            rgc_prob = 0.3
        
        cell_type = "RGC" if rgc_prob > 0.5 else "unknown"
        
        return {
            "cell_type": cell_type,
            "rgc_probability": float(rgc_prob),
            "has_axon": bool(has_axon),
            "waveform_width_ms": float(waveform_width_ms),
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "cell_type": "unknown",
            "rgc_probability": 0.0,
            "has_axon": False,
            "waveform_width_ms": float("nan"),
        }

