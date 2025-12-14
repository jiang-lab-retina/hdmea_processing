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
            - light_reference: Dict with different sample rates
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
        from McsPy import McsCMOSMEA
        
        cmcr_data = McsCMOSMEA.McsCMOSMEAData(str(cmcr_path))
        
        light_reference = {}
        metadata = {}
        
        # Extract recording info
        if hasattr(cmcr_data, "recording_info"):
            metadata["recording_info"] = cmcr_data.recording_info
        
        # Get acquisition rate
        acquisition_rate = 20000.0  # Default 20kHz
        if hasattr(cmcr_data, "acquisition_rate"):
            acquisition_rate = cmcr_data.acquisition_rate
        
        # Extract light reference (auxiliary channel)
        # Light reference is typically in an auxiliary stream
        if hasattr(cmcr_data, "auxiliary_source") and cmcr_data.auxiliary_source is not None:
            aux_source = cmcr_data.auxiliary_source
            
            # Get the light reference channel
            if hasattr(aux_source, "get_data"):
                try:
                    light_data = aux_source.get_data()
                    if light_data is not None and len(light_data) > 0:
                        light_reference["20kHz"] = light_data.astype(np.float32)
                        
                        # Create downsampled versions
                        light_reference["10Hz"] = _downsample_light_reference(
                            light_data, acquisition_rate, target_rate=10
                        )
                        light_reference["1kHz"] = _downsample_light_reference(
                            light_data, acquisition_rate, target_rate=1000
                        )
                        
                        logger.info(f"Loaded light reference: {len(light_data)} samples")
                except Exception as e:
                    logger.warning(f"Could not extract light reference: {e}")
        
        # Get recording duration
        if hasattr(cmcr_data, "duration"):
            metadata["recording_duration_s"] = cmcr_data.duration / 1e6  # Convert from μs
        
        return {
            "light_reference": light_reference,
            "metadata": metadata,
            "acquisition_rate": acquisition_rate,
            "source_path": str(cmcr_path),
        }
        
    except ImportError:
        raise DataLoadError(
            "McsPy library not available. Install it to read CMCR files.",
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

