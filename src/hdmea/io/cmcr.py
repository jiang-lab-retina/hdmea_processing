"""
CMCR file reading via McsPy.

CMCR files contain raw sensor data and light reference from HD-MEA recordings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hdmea.utils.exceptions import DataLoadError


logger = logging.getLogger(__name__)


def load_cmcr_data(cmcr_path: Path) -> Dict[str, Any]:
    """
    Load raw sensor data from CMCR file.
    
    Extracts light reference data which is used for stimulus timing alignment.
    
    Args:
        cmcr_path: Path to .cmcr file (may be UNC path)
    
    Returns:
        Dictionary with keys:
            - light_reference: Dict with six entries:
                - raw_ch1, 10hz_ch1, 1khz_ch1 (channel 1 at raw/10Hz/1kHz)
                - raw_ch2, 10hz_ch2, 1khz_ch2 (channel 2 at raw/10Hz/1kHz)
            - metadata: Recording metadata
            - acquisition_rate: Sampling rate in Hz
    
    Raises:
        FileNotFoundError: If cmcr_path does not exist
        DataLoadError: If file cannot be read
    """
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    logger.info(f"Loading CMCR file: {cmcr_path}")
    
    try:
        # Import McsPy here to allow graceful failure if not installed
        from McsPy.McsCMOSMEA import McsCMOSMEAData
        
        cmcr_data = McsCMOSMEAData(str(cmcr_path))
        
        light_reference = {}
        metadata = {}
        
        # Extract file attributes as metadata
        if hasattr(cmcr_data, "attrs"):
            for key in cmcr_data.attrs.keys():
                try:
                    val = cmcr_data.attrs[key]
                    # Decode bytes to string
                    if isinstance(val, bytes):
                        metadata[key] = val.decode('utf-8', errors='ignore')
                    elif hasattr(val, '__len__') and len(val) == 1:
                        metadata[key] = val[0]
                    else:
                        metadata[key] = val
                except Exception:
                    pass
        
        # Get recording duration from attributes (in microseconds)
        recording_duration_us = metadata.get("LB.RecordingDuration", 0)
        if recording_duration_us > 0:
            metadata["recording_duration_s"] = recording_duration_us / 1e6
        
        # Default acquisition rate (20 kHz typical for MaxOne/MaxTwo)
        acquisition_rate = 20000.0
        
        # Extract light reference from Analog_Data
        # Light reference is typically stored in Acquisition.Analog_Data
        if hasattr(cmcr_data, "Acquisition") and cmcr_data.Acquisition is not None:
            acq = cmcr_data.Acquisition
            
            if hasattr(acq, "Analog_Data") and acq.Analog_Data is not None:
                analog_data = acq.Analog_Data
                
                # Try to get ChannelData_1 which contains analog channels
                if hasattr(analog_data, "ChannelData_1"):
                    try:
                        channel_data = analog_data.ChannelData_1[:]
                        
                        # channel_data shape is typically (num_channels, num_samples)
                        if channel_data is not None and len(channel_data) > 0:
                            # Use first channel as light reference
                            if channel_data.ndim == 2:
                                light_data = channel_data[0].astype(np.float32)
                            else:
                                light_data = channel_data.astype(np.float32)
                            
                            # Estimate acquisition rate from data and duration
                            if recording_duration_us > 0:
                                acquisition_rate = len(light_data) / (recording_duration_us / 1e6)
                                logger.info(f"Estimated acquisition rate: {acquisition_rate:.0f} Hz")
                            
                            # Store channel 1 at all sample rates
                            light_reference["raw_ch1"] = light_data
                            light_reference["10hz_ch1"] = _downsample_light_reference(
                                light_data, acquisition_rate, target_rate=10
                            )
                            light_reference["1khz_ch1"] = _downsample_light_reference(
                                light_data, acquisition_rate, target_rate=1000
                            )
                            
                            logger.info(f"Loaded light reference ch1: {len(light_data)} samples from Analog_Data")
                            
                            # Store channel 2 at all sample rates if available
                            if channel_data.ndim == 2 and channel_data.shape[0] > 1:
                                light_data_ch2 = channel_data[1].astype(np.float32)
                                light_reference["raw_ch2"] = light_data_ch2
                                light_reference["10hz_ch2"] = _downsample_light_reference(
                                    light_data_ch2, acquisition_rate, target_rate=10
                                )
                                light_reference["1khz_ch2"] = _downsample_light_reference(
                                    light_data_ch2, acquisition_rate, target_rate=1000
                                )
                                logger.info(f"Loaded light reference ch2: {len(light_data_ch2)} samples")
                                
                    except Exception as e:
                        logger.warning(f"Could not extract light reference from ChannelData_1: {e}")
                
                # Also check for Data attribute
                elif hasattr(analog_data, "Data"):
                    try:
                        # Data might be a McsStreamList
                        data_list = analog_data.Data
                        if hasattr(data_list, '__iter__'):
                            for i, data_item in enumerate(data_list):
                                if hasattr(data_item, 'shape'):
                                    light_data = data_item[:].astype(np.float32)
                                    light_reference[f"channel_{i}"] = light_data
                                    logger.info(f"Loaded light reference channel {i}: {len(light_data)} samples")
                    except Exception as e:
                        logger.warning(f"Could not extract light reference from Data: {e}")
        
        if not light_reference:
            logger.warning("No light reference data found in CMCR file")
        
        return {
            "light_reference": light_reference,
            "metadata": metadata,
            "acquisition_rate": acquisition_rate,
            "source_path": str(cmcr_path),
        }
        
    except ImportError:
        raise DataLoadError(
            "McsPy library not available. Install with: pip install McsPyDataTools",
            file_path=str(cmcr_path),
        )
    except Exception as e:
        logger.error(f"Failed to read CMCR: {cmcr_path}: {e}")
        raise DataLoadError(
            f"Cannot read CMCR file: {cmcr_path}",
            file_path=str(cmcr_path),
            original_error=e,
        )


def _downsample_light_reference(
    data: np.ndarray,
    source_rate: float,
    target_rate: float,
) -> np.ndarray:
    """
    Downsample light reference to target rate.
    
    Args:
        data: Source data array
        source_rate: Source sampling rate (Hz)
        target_rate: Target sampling rate (Hz)
    
    Returns:
        Downsampled array
    """
    if target_rate >= source_rate:
        return data
    
    factor = int(source_rate / target_rate)
    
    # Use reshape and mean for downsampling
    n_samples = (len(data) // factor) * factor
    reshaped = data[:n_samples].reshape(-1, factor)
    downsampled = reshaped.mean(axis=1).astype(np.float32)
    
    return downsampled


def find_cmcr_file_from_cmtr(cmtr_path: Path) -> Optional[Path]:
    """
    Try to find matching CMCR file for a given CMTR file.
    
    Searches in the same directory for a file with matching base name
    but .cmcr extension.
    
    Args:
        cmtr_path: Path to .cmtr file
    
    Returns:
        Path to matching .cmcr file if found, None otherwise
    """
    # Try exact stem match
    cmcr_path = cmtr_path.with_suffix(".cmcr")
    if cmcr_path.exists():
        return cmcr_path
    
    # Try common variations
    stem = cmtr_path.stem
    directory = cmtr_path.parent
    
    for suffix in ["_sorted", "_spikes"]:
        if stem.endswith(suffix):
            base_stem = stem[: -len(suffix)]
            candidate = directory / f"{base_stem}.cmcr"
            if candidate.exists():
                return candidate
    
    # Search for any .cmcr file with similar name
    for cmcr_file in directory.glob("*.cmcr"):
        if cmcr_file.stem.startswith(stem[:10]):  # Match first 10 chars
            logger.info(f"Found potential matching CMCR: {cmcr_file}")
            return cmcr_file
    
    return None

