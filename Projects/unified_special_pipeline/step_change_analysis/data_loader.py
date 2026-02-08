"""
Data Loader for Step Change Analysis Pipeline

This module handles loading CMCR/CMTR data and saving to HDF5 format.
It extracts step responses and computes quality indices for each unit.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.signal import find_peaks

from .specific_config import (
    PipelineConfig,
    StepDetectionConfig,
    QualityConfig,
    default_config,
    get_output_hdf5_path,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Index Calculation
# =============================================================================

def calculate_quality_index(trace_array: np.ndarray) -> float:
    """
    Calculate the quality index for a trace array.
    
    Quality index is the ratio of signal variance to noise variance:
    QI = Var(mean_trace) / Mean(Var(individual_traces))
    
    Higher values indicate more consistent, reliable responses.
    
    Args:
        trace_array: 2D array of shape (n_trials, n_timepoints)
    
    Returns:
        Quality index value (float), or np.nan if invalid
    """
    if trace_array.ndim != 2:
        return np.nan
    
    if trace_array.shape[0] < 2:
        return np.nan
    
    # Mean trace across trials
    mean_across_trials = trace_array.mean(axis=0)
    
    # Variance of each trial
    var_across_timepoints = trace_array.var(axis=1)
    
    # Check for NaN values
    if np.isnan(mean_across_trials).any() or np.isnan(var_across_timepoints).any():
        return np.nan
    
    # Mean variance across trials
    mean_var = np.mean(var_across_timepoints)
    
    if mean_var == 0:
        return np.nan
    
    # Quality index: variance of mean / mean variance
    quality_index = np.var(mean_across_trials) / mean_var
    
    return float(quality_index)


# =============================================================================
# Step Detection
# =============================================================================

def detect_step_times(
    light_reference: np.ndarray,
    config: Optional[StepDetectionConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect step onset and offset times from light reference signal.
    
    Args:
        light_reference: 1D array of light reference signal (at 10Hz)
        config: Step detection configuration
    
    Returns:
        Tuple of (on_times, off_times) arrays
    """
    if config is None:
        config = default_config.step_detection
    
    # Find step onsets (positive edges)
    diff_signal = np.diff(light_reference.astype(float))
    on_times, _ = find_peaks(
        diff_signal,
        height=config.threshold_height,
        distance=config.min_peak_distance,
    )
    
    # Find step offsets (negative edges)
    off_times, _ = find_peaks(
        -diff_signal,
        height=config.threshold_height,
        distance=config.min_peak_distance,
    )
    
    # Apply repeat range to skip incomplete steps
    start_idx, end_idx = config.repeat_range
    if end_idx is None:
        on_times = on_times[start_idx:]
        off_times = off_times[start_idx:]
    else:
        on_times = on_times[start_idx:end_idx]
        off_times = off_times[start_idx:end_idx]
    
    logger.debug(f"Detected {len(on_times)} step onsets, {len(off_times)} step offsets")
    
    return on_times, off_times


def extract_step_responses(
    firing_rate: np.ndarray,
    on_times: np.ndarray,
    config: Optional[StepDetectionConfig] = None,
) -> np.ndarray:
    """
    Extract response segments around each step onset.
    
    Args:
        firing_rate: 1D array of firing rate (at 10Hz)
        on_times: Array of step onset times (sample indices)
        config: Step detection configuration
    
    Returns:
        2D array of shape (n_trials, pre_margin + post_margin)
    """
    if config is None:
        config = default_config.step_detection
    
    responses = []
    window_size = config.pre_margin + config.post_margin
    
    for on_time in on_times:
        start = on_time - config.pre_margin
        end = on_time + config.post_margin
        
        # Skip if window extends beyond data
        if start < 0 or end > len(firing_rate):
            continue
        
        responses.append(firing_rate[start:end])
    
    if not responses:
        return np.array([]).reshape(0, window_size)
    
    return np.array(responses)


# =============================================================================
# Data Loading (using hdmea modules)
# =============================================================================

def load_cmcr_cmtr_data(
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load data from CMCR and CMTR files.
    
    Uses hdmea.io modules for loading.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
    
    Returns:
        Dictionary with:
            - units: Dict[unit_id, unit_data]
            - light_reference: Light reference arrays
            - metadata: Recording metadata
    """
    from hdmea.io.cmcr import load_cmcr_data
    from hdmea.io.cmtr import load_cmtr_data
    
    cmcr_path = Path(cmcr_path)
    cmtr_path = Path(cmtr_path)
    
    logger.info(f"Loading CMCR: {cmcr_path.name}")
    cmcr_data = load_cmcr_data(cmcr_path)
    
    logger.info(f"Loading CMTR: {cmtr_path.name}")
    cmtr_data = load_cmtr_data(cmtr_path)
    
    # Combine data
    result = {
        "units": cmtr_data.get("units", {}),
        "light_reference": cmcr_data.get("light_reference", {}),
        "metadata": {
            "cmcr": cmcr_data.get("metadata", {}),
            "cmtr": cmtr_data.get("metadata", {}),
            "acquisition_rate": cmcr_data.get("acquisition_rate", 20000.0),
            "source_cmcr": str(cmcr_path),
            "source_cmtr": str(cmtr_path),
        },
    }
    
    logger.info(f"Loaded {len(result['units'])} units")
    
    return result


def compute_firing_rate_10hz(
    spike_times_us: np.ndarray,
    recording_duration_us: float,
    interval_us: float = 100_000,
) -> np.ndarray:
    """
    Compute firing rate histogram at 10Hz (100ms bins).
    
    Args:
        spike_times_us: Spike times in microseconds
        recording_duration_us: Total recording duration in microseconds
        interval_us: Bin size in microseconds (default 100ms = 100,000us)
    
    Returns:
        1D array of firing rates (spikes/second)
    """
    bins = np.arange(0, recording_duration_us + interval_us, interval_us)
    hist, _ = np.histogram(spike_times_us, bins=bins)
    
    # Convert to Hz (spikes per second)
    firing_rate = hist / (interval_us / 1_000_000)
    
    return firing_rate.astype(np.float32)


# =============================================================================
# HDF5 Operations
# =============================================================================

def save_recording_to_hdf5(
    data: Dict[str, Any],
    hdf5_path: Union[str, Path],
    step_config: Optional[StepDetectionConfig] = None,
    quality_config: Optional[QualityConfig] = None,
    overwrite: bool = False,
) -> Path:
    """
    Save recording data to HDF5 format with step responses.
    
    Args:
        data: Data dictionary from load_cmcr_cmtr_data()
        hdf5_path: Output HDF5 file path
        step_config: Step detection configuration
        quality_config: Quality calculation configuration
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to the created HDF5 file
    """
    if step_config is None:
        step_config = default_config.step_detection
    if quality_config is None:
        quality_config = default_config.quality
    
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hdf5_path.exists() and not overwrite:
        logger.warning(f"HDF5 file exists, skipping: {hdf5_path}")
        return hdf5_path
    
    logger.info(f"Saving to HDF5: {hdf5_path}")
    
    # Get light reference for step detection
    # Use the channel specified in config (default: channel 2)
    light_ref = data.get("light_reference", {})
    channel = getattr(step_config, 'light_channel', 2)
    
    # Try 1kHz first (better resolution), then 10Hz, then raw
    light_ref_10hz = None
    
    # Try 1kHz channel and downsample to 10Hz
    key_1khz = f"1khz_ch{channel}"
    if key_1khz in light_ref:
        light_ref_1khz = light_ref[key_1khz]
        light_ref_10hz = light_ref_1khz[::100]  # 1kHz -> 10Hz
        logger.info(f"Using {key_1khz} for step detection (downsampled to 10Hz)")
    
    # Fallback to 10Hz if available
    if light_ref_10hz is None:
        key_10hz = f"10hz_ch{channel}"
        if key_10hz in light_ref:
            light_ref_10hz = light_ref[key_10hz]
            logger.info(f"Using {key_10hz} for step detection")
    
    # Fallback to raw and downsample
    if light_ref_10hz is None:
        key_raw = f"raw_ch{channel}"
        if key_raw in light_ref:
            light_ref_raw = light_ref[key_raw]
            # Downsample from raw to 10Hz (assuming 20kHz -> 10Hz = factor of 2000)
            light_ref_10hz = light_ref_raw[::2000]
            logger.info(f"Using {key_raw} for step detection (downsampled to 10Hz)")
    
    if light_ref_10hz is None:
        logger.warning(f"No light reference found for channel {channel}, cannot detect steps")
        light_ref_10hz = np.array([])
    
    # Detect step times
    if len(light_ref_10hz) > 0:
        on_times, off_times = detect_step_times(light_ref_10hz, step_config)
    else:
        on_times, off_times = np.array([]), np.array([])
    
    with h5py.File(hdf5_path, "w") as f:
        # Create groups
        units_grp = f.create_group("units")
        stim_grp = f.create_group("stimulus")
        meta_grp = f.create_group("metadata")
        
        # Save stimulus data
        if len(light_ref_10hz) > 0:
            stim_grp.create_dataset("light_reference_10hz", data=light_ref_10hz)
        if len(on_times) > 0:
            stim_grp.create_dataset("step_on_times", data=on_times)
        if len(off_times) > 0:
            stim_grp.create_dataset("step_off_times", data=off_times)
        
        # Save raw light reference if available
        for key in ["raw_ch1", "raw_ch2", "1khz_ch1", "1khz_ch2"]:
            if key in light_ref:
                stim_grp.create_dataset(f"light_reference_{key}", data=light_ref[key])
        
        # Save metadata
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            if isinstance(value, dict):
                sub_grp = meta_grp.create_group(key)
                for sub_key, sub_value in value.items():
                    try:
                        if isinstance(sub_value, (str, bytes)):
                            sub_grp.attrs[sub_key] = sub_value
                        elif isinstance(sub_value, (int, float, np.number)):
                            sub_grp.attrs[sub_key] = sub_value
                        elif isinstance(sub_value, np.ndarray):
                            sub_grp.create_dataset(sub_key, data=sub_value)
                    except Exception as e:
                        logger.debug(f"Could not save metadata {key}/{sub_key}: {e}")
            else:
                try:
                    if isinstance(value, (str, bytes)):
                        meta_grp.attrs[key] = value
                    elif isinstance(value, (int, float, np.number)):
                        meta_grp.attrs[key] = value
                except Exception as e:
                    logger.debug(f"Could not save metadata {key}: {e}")
        
        # Save units
        units = data.get("units", {})
        high_quality_count = 0
        
        for unit_id, unit_data in units.items():
            unit_grp = units_grp.create_group(str(unit_id))
            
            # Save basic unit data
            if "spike_times" in unit_data:
                unit_grp.create_dataset(
                    "spike_times",
                    data=np.array(unit_data["spike_times"], dtype=np.uint64),
                )
            
            if "waveform" in unit_data:
                unit_grp.create_dataset(
                    "waveform",
                    data=np.array(unit_data["waveform"], dtype=np.float32),
                )
            
            # Save electrode position
            if "row" in unit_data:
                unit_grp.attrs["row"] = unit_data["row"]
            if "col" in unit_data:
                unit_grp.attrs["col"] = unit_data["col"]
            
            # Compute firing rate at 10Hz
            if "spike_times" in unit_data and "spike_times" in unit_data:
                spike_times = unit_data["spike_times"]
                if len(spike_times) > 0:
                    # Get recording duration from metadata
                    rec_duration = metadata.get("cmcr", {}).get("LB.RecordingDuration", 0)
                    if rec_duration == 0 and len(light_ref_10hz) > 0:
                        rec_duration = len(light_ref_10hz) * 100_000  # 100ms bins
                    
                    if rec_duration > 0:
                        firing_rate = compute_firing_rate_10hz(spike_times, rec_duration)
                        unit_grp.create_dataset("firing_rate_10hz", data=firing_rate)
                        
                        # Extract step responses
                        if len(on_times) > 0:
                            step_responses = extract_step_responses(
                                firing_rate, on_times, step_config
                            )
                            
                            if step_responses.size > 0:
                                unit_grp.create_dataset("step_responses", data=step_responses)
                                
                                # Calculate quality index
                                qi = calculate_quality_index(step_responses)
                                unit_grp.attrs["quality_index"] = qi
                                
                                # Calculate response signature (mean response)
                                response_sig = step_responses.mean(axis=0)
                                unit_grp.create_dataset("response_signature", data=response_sig)
                                
                                if qi >= quality_config.quality_threshold:
                                    high_quality_count += 1
        
        # Save pipeline info
        f.attrs["pipeline"] = "step_change_analysis"
        f.attrs["total_units"] = len(units)
        f.attrs["high_quality_units"] = high_quality_count
    
    logger.info(f"Saved {len(units)} units ({high_quality_count} high quality)")
    
    return hdf5_path


def load_recording_from_hdf5(
    hdf5_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load recording data from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary with units, stimulus, and metadata
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    logger.info(f"Loading HDF5: {hdf5_path}")
    
    data = {
        "units": {},
        "stimulus": {},
        "metadata": {},
        "source_path": str(hdf5_path),
    }
    
    with h5py.File(hdf5_path, "r") as f:
        # Load units
        if "units" in f:
            for unit_id in f["units"]:
                unit_grp = f["units"][unit_id]
                unit_data = {}
                
                # Load datasets
                for key in unit_grp.keys():
                    unit_data[key] = unit_grp[key][:]
                
                # Load attributes
                for key in unit_grp.attrs.keys():
                    unit_data[key] = unit_grp.attrs[key]
                
                data["units"][unit_id] = unit_data
        
        # Load stimulus
        if "stimulus" in f:
            for key in f["stimulus"].keys():
                data["stimulus"][key] = f["stimulus"][key][:]
        
        # Load metadata
        if "metadata" in f:
            for key in f["metadata"].attrs.keys():
                data["metadata"][key] = f["metadata"].attrs[key]
            
            for key in f["metadata"].keys():
                if isinstance(f["metadata"][key], h5py.Group):
                    data["metadata"][key] = {}
                    for sub_key in f["metadata"][key].attrs.keys():
                        data["metadata"][key][sub_key] = f["metadata"][key].attrs[sub_key]
                else:
                    data["metadata"][key] = f["metadata"][key][:]
        
        # Load root attributes
        for key in f.attrs.keys():
            data["metadata"][key] = f.attrs[key]
    
    logger.info(f"Loaded {len(data['units'])} units")
    
    return data


# =============================================================================
# High-level API
# =============================================================================

def load_and_save_recording(
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    overwrite: bool = False,
) -> Tuple[Dict[str, Any], Path]:
    """
    Load CMCR/CMTR data, process step responses, and save to HDF5.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        output_path: Output HDF5 path (auto-generated if None)
        config: Pipeline configuration
        overwrite: Whether to overwrite existing output
    
    Returns:
        Tuple of (loaded_data, hdf5_path)
    """
    if config is None:
        config = default_config
    
    cmcr_path = Path(cmcr_path)
    cmtr_path = Path(cmtr_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_hdf5_path(
            cmcr_path.name,
            config.output_dir,
        )
    
    # Load data
    data = load_cmcr_cmtr_data(cmcr_path, cmtr_path)
    
    # Save to HDF5
    hdf5_path = save_recording_to_hdf5(
        data,
        output_path,
        step_config=config.step_detection,
        quality_config=config.quality,
        overwrite=overwrite,
    )
    
    # Reload from HDF5 to get processed data
    processed_data = load_recording_from_hdf5(hdf5_path)
    
    return processed_data, hdf5_path


def get_high_quality_units(
    data: Dict[str, Any],
    threshold: Optional[float] = None,
) -> List[str]:
    """
    Get list of high quality unit IDs.
    
    Args:
        data: Recording data dictionary
        threshold: Quality index threshold (uses default if None)
    
    Returns:
        List of unit IDs with quality above threshold
    """
    if threshold is None:
        threshold = default_config.quality.quality_threshold
    
    high_quality = []
    
    for unit_id, unit_data in data.get("units", {}).items():
        qi = unit_data.get("quality_index", 0)
        if qi >= threshold:
            high_quality.append(unit_id)
    
    return sorted(high_quality, key=lambda x: int(x) if x.isdigit() else x)
