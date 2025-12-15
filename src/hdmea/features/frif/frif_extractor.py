"""
FRIF (Firing Rate Inter-Frame) feature extractor.

Computes the firing rate for each inter-frame interval, aligning spike
activity to video frame timing. This is essential for correlating neural
activity with visual stimuli presented at discrete frame intervals.

The implementation follows the legacy approach from:
jianglab.common_functions.add_frame_time_to_pkl_data() and
feature_analysis.create_FRIF()
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import zarr

from hdmea.features.base import FeatureExtractor
from hdmea.features.registry import FeatureRegistry


logger = logging.getLogger(__name__)


@FeatureRegistry.register("frif")
class FRIFExtractor(FeatureExtractor):
    """
    Compute Firing Rate Inter-Frame (FRIF) for each unit.
    
    FRIF computes the instantaneous firing rate aligned to video frame
    intervals. For each pair of consecutive frame timestamps, it counts
    the spikes in that interval and converts to rate (Hz).
    
    Outputs:
        - FRIF: Array of firing rates (Hz) for each frame interval
        - FRIF_x_axis: Array of frame start times (seconds) for plotting
    
    Required inputs:
        - spike_times: Spike timestamps in microseconds (from unit_data)
        - metadata/frame_timestamps: Frame start times in samples
        - metadata/acquisition_rate: Sampling rate in Hz
    
    Note:
        Frame timestamps are detected from the light reference signal
        during Stage 1 loading. If frame_timestamps are not available,
        this extractor will return empty arrays.
    """
    
    name = "frif"
    version = "1.0.0"
    description = "Firing Rate Inter-Frame - spike rate aligned to video frames"
    
    # Note: spike_times is validated per-unit in extract(), not at root level
    # frame_timestamps and acquisition_rate are accessed from metadata
    required_inputs = []
    
    output_schema = {
        "FRIF": {
            "dtype": "float32",
            "shape": "(n_frames-1,)",
            "unit": "Hz",
            "description": "Firing rate for each inter-frame interval",
        },
        "FRIF_x_axis": {
            "dtype": "float64",
            "shape": "(n_frames-1,)",
            "unit": "s",
            "description": "Frame start times in seconds (for x-axis plotting)",
        },
    }
    
    runtime_class = "fast"
    
    def extract(
        self,
        unit_data: zarr.Group,
        stimulus_data: zarr.Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[zarr.Group] = None,
    ) -> Dict[str, Any]:
        """
        Extract FRIF features for a single unit.
        
        Args:
            unit_data: Zarr group containing spike_times
            stimulus_data: Zarr group containing stimulus info
            config: Optional configuration overrides
            metadata: Zarr group containing acquisition_rate and frame_timestamps
        
        Returns:
            Dictionary with FRIF and FRIF_x_axis arrays
        """
        config = config or {}
        
        # Get frame timestamps and acquisition rate from metadata
        if metadata is None:
            logger.warning("No metadata provided - cannot compute FRIF")
            return self._empty_result()
        
        # Load frame_timestamps
        frame_timestamps = None
        if "frame_timestamps" in metadata:
            frame_timestamps = metadata["frame_timestamps"][:]
        
        if frame_timestamps is None or len(frame_timestamps) < 2:
            logger.warning("No frame_timestamps found - cannot compute FRIF")
            return self._empty_result()
        
        # Load acquisition_rate
        acquisition_rate = None
        if "acquisition_rate" in metadata:
            arr = metadata["acquisition_rate"][:]
            acquisition_rate = float(arr.flat[0]) if arr.size > 0 else None
        
        if acquisition_rate is None or acquisition_rate <= 0:
            logger.warning("Invalid acquisition_rate - cannot compute FRIF")
            return self._empty_result()
        
        # Load spike times (in microseconds)
        spike_times = unit_data["spike_times"][:]
        
        if len(spike_times) == 0:
            # No spikes - return zeros
            n_intervals = len(frame_timestamps) - 1
            return {
                "FRIF": np.zeros(n_intervals, dtype=np.float32),
                "FRIF_x_axis": (frame_timestamps[:-1] / acquisition_rate).astype(np.float64),
            }
        
        # Convert frame timestamps to microseconds
        # frame_timestamps are in samples, need to convert to µs
        # time_us = samples * 1e6 / acquisition_rate
        frame_times_us = frame_timestamps.astype(np.float64) * 1e6 / acquisition_rate
        
        # Compute FRIF
        n_intervals = len(frame_times_us) - 1
        frif = np.zeros(n_intervals, dtype=np.float32)
        
        for i in range(n_intervals):
            start_us = frame_times_us[i]
            end_us = frame_times_us[i + 1]
            
            # Count spikes in this interval
            spikes_in_interval = np.sum(
                (spike_times >= start_us) & (spike_times < end_us)
            )
            
            # Convert to rate (Hz)
            interval_duration_s = (end_us - start_us) * 1e-6
            if interval_duration_s > 0:
                frif[i] = spikes_in_interval / interval_duration_s
        
        # Compute x-axis (frame start times in seconds)
        frif_x_axis = (frame_timestamps[:-1] / acquisition_rate).astype(np.float64)
        
        logger.debug(
            f"Computed FRIF: {n_intervals} intervals, "
            f"mean rate: {np.mean(frif):.2f} Hz"
        )
        
        return {
            "FRIF": frif,
            "FRIF_x_axis": frif_x_axis,
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when computation is not possible."""
        return {
            "FRIF": np.array([], dtype=np.float32),
            "FRIF_x_axis": np.array([], dtype=np.float64),
        }

