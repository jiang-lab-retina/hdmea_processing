"""
ON/OFF response feature extractor for step-up stimulus.

Extracts features related to light onset (ON) and offset (OFF) responses.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from hdmea.features.base import FeatureExtractor, HDF5Group
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("step_up_5s_5i_3x")
class StepUpFeatureExtractor(FeatureExtractor):
    """
    Extract ON/OFF response features from step-up stimulus.
    
    Analyzes neural responses to sustained light steps, quantifying:
    - Transient ON response at light onset
    - Transient OFF response at light offset
    - Sustained response during light-on period
    """
    
    name = "step_up_5s_5i_3x"
    version = "1.0.0"
    description = "ON/OFF response features for step-up intensity stimulus"
    
    required_inputs = ["spike_times"]
    output_schema = {
        "on_response_flag": {"dtype": "bool", "description": "Significant ON response detected"},
        "off_response_flag": {"dtype": "bool", "description": "Significant OFF response detected"},
        "on_peak_value": {"dtype": "float64", "unit": "spikes/s", "description": "Peak firing at ON"},
        "off_peak_value": {"dtype": "float64", "unit": "spikes/s", "description": "Peak firing at OFF"},
        "on_sustained_response": {"dtype": "float64", "unit": "spikes/s", "description": "Sustained rate after ON"},
        "off_sustained_response": {"dtype": "float64", "unit": "spikes/s", "description": "Sustained rate after OFF"},
        "response_quality": {"dtype": "float64", "unit": "dimensionless", "description": "Quality metric [0,1]"},
        "on_latency_ms": {"dtype": "float64", "unit": "ms", "description": "Latency to ON peak"},
        "off_latency_ms": {"dtype": "float64", "unit": "ms", "description": "Latency to OFF peak"},
        "on_off_ratio": {"dtype": "float64", "unit": "dimensionless", "description": "ON/OFF balance [-1,1]"},
    }
    runtime_class = "fast"
    
    # Default stimulus parameters (can be overridden via config)
    DEFAULT_STIM_ONSET_US = 5_000_000  # 5 seconds
    DEFAULT_STIM_OFFSET_US = 10_000_000  # 10 seconds
    DEFAULT_TRANSIENT_WINDOW_US = 500_000  # 500 ms
    DEFAULT_SUSTAINED_WINDOW_US = 4_000_000  # 4 seconds
    DEFAULT_BASELINE_WINDOW_US = 4_000_000  # 4 seconds before onset
    
    def extract(
        self,
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """
        Extract ON/OFF response features for a unit.
        
        Args:
            unit_data: HDF5 group for the unit
            stimulus_data: HDF5 group with stimulus information
            config: Optional runtime configuration
        
        Returns:
            Dictionary of feature values
        """
        config = config or {}
        
        # Get parameters from config or use defaults
        stim_onset = config.get("stim_onset_us", self.DEFAULT_STIM_ONSET_US)
        stim_offset = config.get("stim_offset_us", self.DEFAULT_STIM_OFFSET_US)
        transient_window = config.get("transient_window_us", self.DEFAULT_TRANSIENT_WINDOW_US)
        sustained_window = config.get("sustained_window_us", self.DEFAULT_SUSTAINED_WINDOW_US)
        baseline_window = config.get("baseline_window_us", self.DEFAULT_BASELINE_WINDOW_US)
        
        # Load spike times
        spike_times = unit_data["spike_times"][:]
        
        if len(spike_times) == 0:
            return self._empty_result()
        
        # Calculate baseline rate
        baseline_spikes = spike_times[
            (spike_times >= stim_onset - baseline_window) & 
            (spike_times < stim_onset)
        ]
        baseline_rate = len(baseline_spikes) / (baseline_window / 1e6)
        
        # Calculate ON response (transient)
        on_transient_spikes = spike_times[
            (spike_times >= stim_onset) & 
            (spike_times < stim_onset + transient_window)
        ]
        on_transient_rate = len(on_transient_spikes) / (transient_window / 1e6)
        
        # Calculate ON sustained response
        on_sustained_spikes = spike_times[
            (spike_times >= stim_onset + transient_window) & 
            (spike_times < stim_offset)
        ]
        on_sustained_duration = (stim_offset - stim_onset - transient_window) / 1e6
        on_sustained_rate = len(on_sustained_spikes) / on_sustained_duration if on_sustained_duration > 0 else 0
        
        # Calculate OFF response (transient)
        off_transient_spikes = spike_times[
            (spike_times >= stim_offset) & 
            (spike_times < stim_offset + transient_window)
        ]
        off_transient_rate = len(off_transient_spikes) / (transient_window / 1e6)
        
        # Calculate OFF sustained response
        off_sustained_spikes = spike_times[
            (spike_times >= stim_offset + transient_window) & 
            (spike_times < stim_offset + transient_window + sustained_window)
        ]
        off_sustained_rate = len(off_sustained_spikes) / (sustained_window / 1e6)
        
        # Determine response flags (significant if 2x baseline)
        threshold_factor = config.get("threshold_factor", 2.0)
        on_response_flag = on_transient_rate > baseline_rate * threshold_factor
        off_response_flag = off_transient_rate > baseline_rate * threshold_factor
        
        # Calculate latencies (time to first spike after onset/offset)
        on_latency = self._calculate_latency(spike_times, stim_onset, transient_window)
        off_latency = self._calculate_latency(spike_times, stim_offset, transient_window)
        
        # Calculate ON/OFF ratio
        total_response = on_transient_rate + off_transient_rate
        if total_response > 0:
            on_off_ratio = (on_transient_rate - off_transient_rate) / total_response
        else:
            on_off_ratio = 0.0
        
        # Calculate response quality (based on signal-to-noise)
        max_response = max(on_transient_rate, off_transient_rate)
        if max_response > 0 and baseline_rate >= 0:
            snr = (max_response - baseline_rate) / (baseline_rate + 1)  # +1 to avoid division by zero
            response_quality = min(1.0, snr / 10)  # Normalize to [0, 1]
        else:
            response_quality = 0.0
        
        return {
            "on_response_flag": bool(on_response_flag),
            "off_response_flag": bool(off_response_flag),
            "on_peak_value": float(on_transient_rate),
            "off_peak_value": float(off_transient_rate),
            "on_sustained_response": float(on_sustained_rate),
            "off_sustained_response": float(off_sustained_rate),
            "response_quality": float(response_quality),
            "on_latency_ms": float(on_latency),
            "off_latency_ms": float(off_latency),
            "on_off_ratio": float(on_off_ratio),
            "baseline_rate": float(baseline_rate),
        }
    
    def _calculate_latency(
        self,
        spike_times: np.ndarray,
        event_time: int,
        window: int,
    ) -> float:
        """Calculate latency to first spike after event."""
        spikes_after = spike_times[
            (spike_times >= event_time) & 
            (spike_times < event_time + window)
        ]
        if len(spikes_after) > 0:
            return (spikes_after[0] - event_time) / 1000  # Convert to ms
        return float("nan")
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for units with no spikes."""
        return {
            "on_response_flag": False,
            "off_response_flag": False,
            "on_peak_value": 0.0,
            "off_peak_value": 0.0,
            "on_sustained_response": 0.0,
            "off_sustained_response": 0.0,
            "response_quality": 0.0,
            "on_latency_ms": float("nan"),
            "off_latency_ms": float("nan"),
            "on_off_ratio": 0.0,
            "baseline_rate": 0.0,
        }

