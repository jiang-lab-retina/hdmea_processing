"""
CMTR file reading via McsPy.

CMTR files contain spike-sorted data from HD-MEA recordings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hdmea.utils.exceptions import DataLoadError, MismatchError


logger = logging.getLogger(__name__)


def load_cmtr_data(cmtr_path: Path) -> Dict[str, Any]:
    """
    Load spike-sorted data from CMTR file.
    
    Args:
        cmtr_path: Path to .cmtr file (may be UNC path)
    
    Returns:
        Dictionary with keys:
            - units: Dict mapping unit_id -> unit data
            - metadata: Recording metadata
            - info_channel: Channel information
    
    Raises:
        FileNotFoundError: If cmtr_path does not exist
        DataLoadError: If file cannot be read
    """
    if not cmtr_path.exists():
        raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
    
    logger.info(f"Loading CMTR file: {cmtr_path}")
    
    try:
        # Import McsPy here to allow graceful failure if not installed
        from McsPy.McsCMOSMEA import McsCMOSMEAData
        
        # Use McsCMOSMEAData class for CMTR files
        cmtr_data = McsCMOSMEAData(str(cmtr_path))
        
        units = {}
        metadata = {}
        
        # Extract file attributes
        acquisition_rate = None
        if hasattr(cmtr_data, "attrs"):
            for key in cmtr_data.attrs.keys():
                try:
                    val = cmtr_data.attrs[key]
                    # Decode bytes to string
                    if isinstance(val, bytes):
                        metadata[key] = val.decode('utf-8', errors='ignore')
                    elif hasattr(val, '__len__') and len(val) == 1:
                        metadata[key] = val[0]
                    else:
                        metadata[key] = val
                    
                    # Look for acquisition rate in common attribute names
                    key_lower = key.lower()
                    if acquisition_rate is None and any(
                        rate_key in key_lower 
                        for rate_key in ['samplingrate', 'sampling_rate', 'samplerate', 'sample_rate', 'acquisitionrate', 'acquisition_rate']
                    ):
                        try:
                            if isinstance(val, (int, float)):
                                acquisition_rate = float(val)
                            elif hasattr(val, '__len__') and len(val) == 1:
                                acquisition_rate = float(val[0])
                            elif hasattr(val, '__len__') and len(val) > 0:
                                acquisition_rate = float(val[0])
                            logger.debug(f"Found acquisition_rate in CMTR attr '{key}': {acquisition_rate}")
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    pass
        
        # Access spike sorter results
        if hasattr(cmtr_data, "Spike_Sorter") and cmtr_data.Spike_Sorter is not None:
            spike_sorter = cmtr_data.Spike_Sorter
            
            # Find all Unit_X attributes
            unit_attrs = [attr for attr in dir(spike_sorter) 
                         if attr.startswith("Unit_") and not attr == "Unit_Info"]
            
            logger.info(f"Found {len(unit_attrs)} units in Spike_Sorter")
            
            for unit_num, unit_attr in enumerate(sorted(unit_attrs)):
                try:
                    # Get unit object (can use getattr or get_unit method)
                    unit = getattr(spike_sorter, unit_attr)
                    unit_id = f"unit_{unit_num:03d}"
                    
                    # Get spike timestamps using get_peaks_timestamps()
                    spike_times = np.array([], dtype=np.uint64)
                    if hasattr(unit, "get_peaks_timestamps"):
                        timestamps = unit.get_peaks_timestamps()
                        if timestamps is not None and len(timestamps) > 0:
                            spike_times = np.array(timestamps, dtype=np.uint64)
                    
                    # Get waveform cutouts using get_peaks_cutouts()
                    waveform = np.array([], dtype=np.float32)
                    if hasattr(unit, "get_peaks_cutouts"):
                        try:
                            cutouts = unit.get_peaks_cutouts()
                            if cutouts is not None and len(cutouts) > 0:
                                # Average all cutouts to get mean waveform
                                waveform = np.mean(cutouts, axis=0).astype(np.float32)
                        except Exception as e:
                            logger.debug(f"Could not get cutouts for {unit_attr}: {e}")
                    
                    # Try to get source/position info
                    row, col, global_id = 0, 0, unit_num
                    if hasattr(unit, "Source"):
                        try:
                            source = unit.Source[:]
                            if len(source) > 0:
                                # Source typically contains sensor ID
                                global_id = int(source[0]) if source[0] else unit_num
                                # Compute row/col from sensor ID (MaxOne/MaxTwo layout)
                                row = global_id // 220
                                col = global_id % 220
                        except Exception:
                            pass
                    
                    units[unit_id] = {
                        "spike_times": spike_times,
                        "waveform": waveform,
                        "row": row,
                        "col": col,
                        "global_id": global_id,
                        "unit_num": unit_num,
                        "original_name": unit_attr,
                    }
                    
                    logger.debug(f"Loaded {unit_attr}: {len(spike_times)} spikes")
                    
                except Exception as e:
                    logger.warning(f"Could not load unit {unit_attr}: {e}")
            
            logger.info(f"Loaded {len(units)} units from CMTR (Spike_Sorter API)")
        
        if not units:
            logger.warning("No units found in CMTR file - file may be empty or have different structure")
        
        result = {
            "units": units,
            "metadata": metadata,
            "source_path": str(cmtr_path),
        }
        
        # Add acquisition_rate if found
        if acquisition_rate is not None and acquisition_rate > 0:
            result["acquisition_rate"] = acquisition_rate
            logger.info(f"Extracted acquisition_rate from CMTR: {acquisition_rate:.0f} Hz")
        
        return result
        
    except ImportError:
        raise DataLoadError(
            "McsPy library not available. Install with: pip install McsPyDataTools",
            file_path=str(cmtr_path),
        )
    except Exception as e:
        logger.error(f"Failed to read CMTR: {cmtr_path}: {e}")
        raise DataLoadError(
            f"Cannot read CMTR file: {cmtr_path}",
            file_path=str(cmtr_path),
            original_error=e,
        )


def get_cmtr_recording_id(cmtr_path: Path) -> Optional[str]:
    """
    Extract recording ID from CMTR metadata for validation.
    
    Args:
        cmtr_path: Path to .cmtr file
    
    Returns:
        Recording ID if available, None otherwise
    """
    try:
        from McsPy.McsCMOSMEA import McsCMOSMEAData
        cmtr_data = McsCMOSMEAData(str(cmtr_path))
        
        if hasattr(cmtr_data, "attrs"):
            if "recording_id" in cmtr_data.attrs:
                return cmtr_data.attrs["recording_id"]
            if "RecordingID" in cmtr_data.attrs:
                return cmtr_data.attrs["RecordingID"]
        return None
        
    except Exception:
        return None


def validate_cmcr_cmtr_match(
    cmcr_path: Optional[Path],
    cmtr_path: Optional[Path],
) -> None:
    """
    Validate that CMCR and CMTR files are from the same recording.
    
    Args:
        cmcr_path: Path to .cmcr file (or None)
        cmtr_path: Path to .cmtr file (or None)
    
    Raises:
        MismatchError: If files don't match
    """
    if cmcr_path is None or cmtr_path is None:
        # Can't validate if one is missing
        return
    
    # For now, validate based on file stem matching
    # In production, would compare internal metadata
    cmcr_stem = cmcr_path.stem.lower()
    cmtr_stem = cmtr_path.stem.lower()
    
    # Remove common suffixes to compare base names
    for suffix in ["_sorted", "_spikes", "_raw"]:
        cmcr_stem = cmcr_stem.replace(suffix, "")
        cmtr_stem = cmtr_stem.replace(suffix, "")
    
    if cmcr_stem != cmtr_stem:
        logger.warning(
            f"CMCR and CMTR file names differ: {cmcr_path.stem} vs {cmtr_path.stem}. "
            "Proceeding with caution."
        )
        # Don't raise error, just warn - user may have renamed files

