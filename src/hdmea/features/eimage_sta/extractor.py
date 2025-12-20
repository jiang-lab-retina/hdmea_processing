"""
EImageSTAExtractor - Feature extractor for electrode image STA.

This extractor integrates with the FeatureRegistry pattern for
consistent feature extraction across the hdmea package.
"""

import logging
from typing import Any, Dict, Literal, Optional

import numpy as np

from hdmea.features.base import FeatureExtractor, HDF5Group
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("eimage_sta")
class EImageSTAExtractor(FeatureExtractor):
    """
    Electrode Image STA feature extractor.
    
    Computes spike-triggered average of sensor data across the electrode array.
    Unlike visual STA (sta.py), this captures the electrical footprint of the
    neuron's activity pattern across the HD-MEA.
    
    Note:
        This extractor requires additional context (filtered sensor data)
        that is typically pre-computed at the recording level. For full
        workflow, use compute_eimage_sta() which handles data loading,
        filtering, and iteration over all units.
    
    Attributes:
        name: "eimage_sta"
        version: "1.0.0"
        runtime_class: "slow" (requires CMCR sensor data access)
        required_inputs: ["spike_times"]
        output_schema: See class definition
    """
    
    name = "eimage_sta"
    version = "1.0.0"
    runtime_class: Literal["fast", "slow"] = "slow"
    description = "Electrode image spike-triggered average from sensor data"
    
    required_inputs = ["spike_times"]
    
    output_schema = {
        "data": {
            "dtype": "float32",
            "shape": "(window_length, rows, cols)",
            "description": "Average electrode activity around spikes",
        },
        "n_spikes": {
            "dtype": "int64",
            "description": "Number of spikes used in average",
        },
        "n_spikes_excluded": {
            "dtype": "int64",
            "description": "Spikes excluded due to edge effects",
        },
        "pre_samples": {
            "dtype": "int64",
            "description": "Samples before spike in window",
        },
        "post_samples": {
            "dtype": "int64",
            "description": "Samples after spike in window",
        },
        "cutoff_hz": {
            "dtype": "float64",
            "description": "High-pass filter cutoff frequency",
        },
        "filter_order": {
            "dtype": "int64",
            "description": "Butterworth filter order",
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor.
        
        Args:
            config: Optional configuration with keys:
                - filtered_data: Pre-filtered sensor data array (required for extract)
                - cutoff_hz: Filter cutoff (default: 100.0)
                - filter_order: Filter order (default: 2)
                - pre_samples: Samples before spike (default: 10)
                - post_samples: Samples after spike (default: 40)
                - spike_limit: Max spikes per unit (default: 10000)
                - sampling_rate: Acquisition rate in Hz (default: 20000.0)
        """
        super().__init__(config)
        
        # Set defaults
        self.config.setdefault("cutoff_hz", 100.0)
        self.config.setdefault("filter_order", 2)
        self.config.setdefault("pre_samples", 10)
        self.config.setdefault("post_samples", 40)
        self.config.setdefault("spike_limit", 10000)
        self.config.setdefault("sampling_rate", 20000.0)
    
    def extract(
        self,
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """
        Extract eimage_sta for a single unit.
        
        Note:
            This method requires filtered_data to be passed in config.
            For full workflow, use compute_eimage_sta() instead.
        
        Args:
            unit_data: HDF5 group containing unit data (must have spike_times)
            stimulus_data: HDF5 group containing stimulus timing (not used)
            config: Must contain 'filtered_data' key with pre-filtered sensor array
            metadata: Optional HDF5 group with recording metadata
        
        Returns:
            Dictionary with eimage_sta results and metadata.
        
        Raises:
            ValueError: If filtered_data not provided in config.
        """
        # Merge config
        cfg = {**self.config, **(config or {})}
        
        # Check for required filtered data
        if "filtered_data" not in cfg:
            raise ValueError(
                "filtered_data must be provided in config for EImageSTAExtractor. "
                "Use compute_eimage_sta() for full workflow."
            )
        
        filtered_data = cfg["filtered_data"]
        
        # Get spike times
        if "spike_times" not in unit_data:
            return self._empty_result(cfg)
        
        spike_times_us = unit_data["spike_times"][:]
        
        if len(spike_times_us) == 0:
            return self._empty_result(cfg)
        
        # Convert to sample indices
        sampling_rate = cfg["sampling_rate"]
        spike_samples = (spike_times_us / 1e6 * sampling_rate).astype(np.int64)
        
        # Import compute function here to avoid circular imports
        from hdmea.features.eimage_sta.compute import compute_sta_for_unit
        
        # Compute STA
        sta, n_used, n_excluded = compute_sta_for_unit(
            filtered_data,
            spike_samples,
            pre_samples=cfg["pre_samples"],
            post_samples=cfg["post_samples"],
            spike_limit=cfg["spike_limit"],
        )
        
        return {
            "data": sta,
            "n_spikes": n_used,
            "n_spikes_excluded": n_excluded,
            "pre_samples": cfg["pre_samples"],
            "post_samples": cfg["post_samples"],
            "cutoff_hz": cfg["cutoff_hz"],
            "filter_order": cfg["filter_order"],
        }
    
    def _empty_result(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Return empty result for units with no spikes."""
        window_length = cfg["pre_samples"] + cfg["post_samples"]
        
        return {
            "data": np.full((window_length, 64, 64), np.nan, dtype=np.float32),
            "n_spikes": 0,
            "n_spikes_excluded": 0,
            "pre_samples": cfg["pre_samples"],
            "post_samples": cfg["post_samples"],
            "cutoff_hz": cfg["cutoff_hz"],
            "filter_order": cfg["filter_order"],
        }

