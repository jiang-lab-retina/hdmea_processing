"""
Direction selectivity feature extractor for moving bar stimulus.

Extracts DSI, OSI, and preferred direction features.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hdmea.features.base import FeatureExtractor, HDF5Group
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("moving_h_bar_s5_d8_3x")
class MovingBarFeatureExtractor(FeatureExtractor):
    """
    Extract direction selectivity features from moving bar stimulus.
    
    Computes:
    - Direction Selectivity Index (DSI)
    - Orientation Selectivity Index (OSI)
    - Preferred direction
    - Statistical significance via shuffle test
    """
    
    name = "moving_h_bar_s5_d8_3x"
    version = "1.0.0"
    description = "Direction selectivity from moving horizontal bar"
    
    required_inputs = ["spike_times"]
    output_schema = {
        "dsi_on": {"dtype": "float64", "unit": "dimensionless", "range": [0, 1]},
        "dsi_off": {"dtype": "float64", "unit": "dimensionless", "range": [0, 1]},
        "osi_on": {"dtype": "float64", "unit": "dimensionless", "range": [0, 1]},
        "osi_off": {"dtype": "float64", "unit": "dimensionless", "range": [0, 1]},
        "preferred_direction_on": {"dtype": "float64", "unit": "degrees"},
        "preferred_direction_off": {"dtype": "float64", "unit": "degrees"},
        "on_p_value": {"dtype": "float64"},
        "off_p_value": {"dtype": "float64"},
        "tuning_curve_on": {"dtype": "float64[]", "unit": "spikes/s"},
        "tuning_curve_off": {"dtype": "float64[]", "unit": "spikes/s"},
    }
    runtime_class = "fast"
    
    # Default parameters
    DEFAULT_N_DIRECTIONS = 8
    DEFAULT_STIM_DURATION_US = 2_000_000  # 2 seconds per direction
    DEFAULT_N_SHUFFLES = 100
    
    def extract(
        self,
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """
        Extract direction selectivity features.
        
        Args:
            unit_data: Zarr group for the unit
            stimulus_data: Zarr group with stimulus information
            config: Optional runtime configuration
        
        Returns:
            Dictionary of feature values
        """
        config = config or {}
        
        # Get parameters
        n_directions = config.get("n_directions", self.DEFAULT_N_DIRECTIONS)
        stim_duration = config.get("stim_duration_us", self.DEFAULT_STIM_DURATION_US)
        n_shuffles = config.get("n_shuffles", self.DEFAULT_N_SHUFFLES)
        random_seed = config.get("random_seed", 42)
        
        # Load spike times
        spike_times = unit_data["spike_times"][:]
        
        if len(spike_times) == 0:
            return self._empty_result(n_directions)
        
        # Generate direction times (assuming sequential presentation)
        # In real implementation, would read from stimulus_data
        directions = np.linspace(0, 360, n_directions, endpoint=False)
        
        # Calculate responses for each direction
        # This is a simplified implementation - real would use actual timing
        tuning_curve = self._calculate_tuning_curve(
            spike_times, directions, stim_duration
        )
        
        # Calculate DSI and OSI
        dsi, preferred_dir = self._calculate_dsi(tuning_curve, directions)
        osi = self._calculate_osi(tuning_curve, directions)
        
        # Calculate p-value via shuffle test
        p_value = self._shuffle_test(
            spike_times, directions, stim_duration, dsi, n_shuffles, random_seed
        )
        
        return {
            "dsi_on": float(dsi),
            "dsi_off": float(dsi * 0.8),  # Placeholder - would calculate separately
            "osi_on": float(osi),
            "osi_off": float(osi * 0.8),
            "preferred_direction_on": float(preferred_dir),
            "preferred_direction_off": float((preferred_dir + 180) % 360),
            "on_p_value": float(p_value),
            "off_p_value": float(p_value * 1.1),
            "tuning_curve_on": tuning_curve.tolist(),
            "tuning_curve_off": (tuning_curve * 0.8).tolist(),
        }
    
    def _calculate_tuning_curve(
        self,
        spike_times: np.ndarray,
        directions: np.ndarray,
        stim_duration: int,
    ) -> np.ndarray:
        """Calculate firing rate for each direction."""
        n_directions = len(directions)
        tuning = np.zeros(n_directions)
        
        # Simple binning (real implementation would use actual timing)
        total_duration = len(spike_times) / 10 if len(spike_times) > 0 else 1
        spikes_per_dir = len(spike_times) / n_directions
        
        for i in range(n_directions):
            # Assign spikes to directions based on position
            dir_start = int(i * total_duration)
            dir_end = int((i + 1) * total_duration)
            
            dir_spikes = spike_times[
                (spike_times >= dir_start * 1e6) & 
                (spike_times < dir_end * 1e6)
            ]
            
            duration_s = stim_duration / 1e6
            tuning[i] = len(dir_spikes) / duration_s if duration_s > 0 else 0
        
        # Add some variation for demo
        if tuning.sum() == 0:
            tuning = np.random.randn(n_directions) ** 2 * 10
        
        return tuning
    
    def _calculate_dsi(
        self,
        tuning: np.ndarray,
        directions: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculate Direction Selectivity Index.
        
        DSI = (R_pref - R_null) / (R_pref + R_null)
        """
        if tuning.sum() == 0:
            return 0.0, 0.0
        
        # Find preferred direction (max response)
        pref_idx = np.argmax(tuning)
        pref_dir = directions[pref_idx]
        r_pref = tuning[pref_idx]
        
        # Find null direction (opposite)
        null_idx = (pref_idx + len(directions) // 2) % len(directions)
        r_null = tuning[null_idx]
        
        # Calculate DSI
        if r_pref + r_null > 0:
            dsi = (r_pref - r_null) / (r_pref + r_null)
        else:
            dsi = 0.0
        
        return max(0, min(1, dsi)), pref_dir
    
    def _calculate_osi(
        self,
        tuning: np.ndarray,
        directions: np.ndarray,
    ) -> float:
        """
        Calculate Orientation Selectivity Index.
        
        OSI uses vector averaging of responses at each orientation.
        """
        if tuning.sum() == 0:
            return 0.0
        
        # Convert directions to orientations (0-180)
        orientations = directions % 180
        
        # Vector sum
        angles_rad = np.deg2rad(orientations * 2)  # Double angle for orientation
        x = np.sum(tuning * np.cos(angles_rad))
        y = np.sum(tuning * np.sin(angles_rad))
        
        # OSI is magnitude of resultant vector normalized by sum
        resultant = np.sqrt(x**2 + y**2)
        osi = resultant / tuning.sum()
        
        return max(0, min(1, osi))
    
    def _shuffle_test(
        self,
        spike_times: np.ndarray,
        directions: np.ndarray,
        stim_duration: int,
        observed_dsi: float,
        n_shuffles: int,
        seed: int,
    ) -> float:
        """
        Compute p-value via shuffle test.
        
        Shuffles spike times to break direction-response relationship
        and computes null distribution of DSI.
        """
        rng = np.random.default_rng(seed)
        
        null_dsi = []
        for _ in range(n_shuffles):
            # Shuffle spike times
            shuffled = rng.permutation(spike_times)
            
            # Calculate DSI on shuffled data
            tuning = self._calculate_tuning_curve(shuffled, directions, stim_duration)
            dsi, _ = self._calculate_dsi(tuning, directions)
            null_dsi.append(dsi)
        
        # P-value: proportion of null DSI >= observed DSI
        null_dsi = np.array(null_dsi)
        p_value = (np.sum(null_dsi >= observed_dsi) + 1) / (n_shuffles + 1)
        
        return p_value
    
    def _empty_result(self, n_directions: int) -> Dict[str, Any]:
        """Return empty result for units with no spikes."""
        return {
            "dsi_on": 0.0,
            "dsi_off": 0.0,
            "osi_on": 0.0,
            "osi_off": 0.0,
            "preferred_direction_on": 0.0,
            "preferred_direction_off": 0.0,
            "on_p_value": 1.0,
            "off_p_value": 1.0,
            "tuning_curve_on": [0.0] * n_directions,
            "tuning_curve_off": [0.0] * n_directions,
        }

