"""
Baseline firing statistics feature extractor.

Extracts basic firing statistics during baseline (gray screen) periods.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import zarr

from hdmea.features.base import FeatureExtractor
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("baseline_127")
class BaselineFeatureExtractor(FeatureExtractor):
    """
    Extract baseline firing statistics.
    
    Computes mean firing rate, standard deviation, and other
    statistics during baseline periods (gray screen at intensity 127).
    """
    
    name = "baseline_127"
    version = "1.0.0"
    description = "Baseline firing statistics during gray screen"
    
    required_inputs = ["spike_times"]
    output_schema = {
        "mean_firing_rate": {"dtype": "float64", "unit": "spikes/s"},
        "std_firing_rate": {"dtype": "float64", "unit": "spikes/s"},
        "cv_isi": {"dtype": "float64", "unit": "dimensionless", "description": "Coefficient of variation of ISI"},
        "median_isi_ms": {"dtype": "float64", "unit": "ms"},
        "spike_count": {"dtype": "int64"},
        "quality_index": {"dtype": "float64", "unit": "dimensionless", "description": "Quality metric [0,1]"},
    }
    runtime_class = "fast"
    
    def extract(
        self,
        unit_data: zarr.Group,
        stimulus_data: zarr.Group,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract baseline statistics for a unit.
        
        Args:
            unit_data: Zarr group for the unit
            stimulus_data: Zarr group with stimulus information
            config: Optional runtime configuration
        
        Returns:
            Dictionary of feature values
        """
        config = config or {}
        
        # Get analysis window from config or use full recording
        start_time = config.get("start_time_us", 0)
        end_time = config.get("end_time_us", None)
        
        # Load spike times
        spike_times = unit_data["spike_times"][:]
        
        if len(spike_times) == 0:
            return self._empty_result()
        
        # Filter to analysis window
        if end_time is not None:
            mask = (spike_times >= start_time) & (spike_times < end_time)
            spike_times = spike_times[mask]
        else:
            mask = spike_times >= start_time
            spike_times = spike_times[mask]
            end_time = spike_times[-1] if len(spike_times) > 0 else start_time
        
        if len(spike_times) == 0:
            return self._empty_result()
        
        # Calculate duration in seconds
        duration_s = (end_time - start_time) / 1e6
        
        if duration_s <= 0:
            return self._empty_result()
        
        # Mean firing rate
        mean_rate = len(spike_times) / duration_s
        
        # Calculate ISIs
        if len(spike_times) > 1:
            isis = np.diff(spike_times) / 1000  # Convert to ms
            median_isi = float(np.median(isis))
            
            # Coefficient of variation of ISI
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
            
            # Standard deviation of firing rate (using 1-second bins)
            bin_size_us = 1_000_000  # 1 second
            n_bins = max(1, int(duration_s))
            bin_edges = np.linspace(start_time, end_time, n_bins + 1)
            counts, _ = np.histogram(spike_times, bins=bin_edges)
            std_rate = float(np.std(counts))
        else:
            median_isi = float("nan")
            cv_isi = 0.0
            std_rate = 0.0
        
        # Quality index based on firing rate and regularity
        # Higher is better: good rate, not too irregular
        quality_index = self._calculate_quality_index(
            mean_rate, cv_isi, len(spike_times)
        )
        
        return {
            "mean_firing_rate": float(mean_rate),
            "std_firing_rate": float(std_rate),
            "cv_isi": float(cv_isi),
            "median_isi_ms": float(median_isi),
            "spike_count": int(len(spike_times)),
            "quality_index": float(quality_index),
        }
    
    def _calculate_quality_index(
        self,
        mean_rate: float,
        cv_isi: float,
        spike_count: int,
    ) -> float:
        """
        Calculate a quality index for the unit.
        
        Based on:
        - Sufficient spike count (>100)
        - Reasonable firing rate (0.5-50 Hz)
        - Not too irregular (CV < 2)
        """
        if spike_count < 10:
            return 0.0
        
        # Rate component
        if 0.5 <= mean_rate <= 50:
            rate_score = 1.0
        elif mean_rate < 0.5:
            rate_score = mean_rate / 0.5
        else:  # > 50
            rate_score = max(0, 1 - (mean_rate - 50) / 50)
        
        # Regularity component
        if cv_isi < 1:
            cv_score = 1.0
        elif cv_isi < 2:
            cv_score = 2 - cv_isi
        else:
            cv_score = 0.0
        
        # Spike count component
        if spike_count >= 100:
            count_score = 1.0
        else:
            count_score = spike_count / 100
        
        return (rate_score + cv_score + count_score) / 3
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for units with no spikes."""
        return {
            "mean_firing_rate": 0.0,
            "std_firing_rate": 0.0,
            "cv_isi": 0.0,
            "median_isi_ms": float("nan"),
            "spike_count": 0,
            "quality_index": 0.0,
        }

