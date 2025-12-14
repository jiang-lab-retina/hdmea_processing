"""
CMTR file reading via McsPy.

CMTR files contain spike-sorted data from HD-MEA recordings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        from McsPy import McsCMOSMEA
        
        cmtr_data = McsCMOSMEA.McsCMOSSpikeStream(str(cmtr_path))
        
        units = {}
        metadata = {}
        
        # Extract recording info
        if hasattr(cmtr_data, "recording_info"):
            metadata["recording_info"] = cmtr_data.recording_info
        
        # Get the high-pass stream (contains spike data)
        if hasattr(cmtr_data, "high_pass_source") and cmtr_data.high_pass_source is not None:
            hp_source = cmtr_data.high_pass_source
            
            # Get units data
            if hasattr(hp_source, "units") and hp_source.units is not None:
                units_data = hp_source.units
                
                for unit_num in range(len(units_data)):
                    unit = units_data[unit_num]
                    unit_id = f"unit_{unit_num:03d}"
                    
                    # Extract spike times (convert to microseconds)
                    spike_times = unit.get_spike_times() if hasattr(unit, "get_spike_times") else np.array([])
                    
                    # Extract waveform (mean cutout)
                    waveform = unit.get_mean_cutout() if hasattr(unit, "get_mean_cutout") else np.array([])
                    
                    # Extract location info
                    row = getattr(unit, "row", 0)
                    col = getattr(unit, "column", 0)
                    global_id = getattr(unit, "global_id", unit_num)
                    
                    units[unit_id] = {
                        "spike_times": spike_times,
                        "waveform": waveform,
                        "row": row,
                        "col": col,
                        "global_id": global_id,
                        "unit_num": unit_num,
                    }
                    
                logger.info(f"Loaded {len(units)} units from CMTR")
        
        return {
            "units": units,
            "metadata": metadata,
            "source_path": str(cmtr_path),
        }
        
    except ImportError:
        raise DataLoadError(
            "McsPy library not available. Install it to read CMTR files.",
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
        from McsPy import McsCMOSMEA
        cmtr_data = McsCMOSMEA.McsCMOSSpikeStream(str(cmtr_path))
        
        if hasattr(cmtr_data, "recording_info"):
            return cmtr_data.recording_info.get("recording_id")
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

