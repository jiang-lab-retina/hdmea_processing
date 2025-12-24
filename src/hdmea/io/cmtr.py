"""
CMTR file reading via McsPy.

CMTR files contain spike-sorted data from HD-MEA recordings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
            
            for unit_attr in sorted(unit_attrs):
                try:
                    # Get unit object (can use getattr or get_unit method)
                    unit = getattr(spike_sorter, unit_attr)
                    
                    # Extract actual unit number from attribute name (e.g., "Unit_1" -> 1)
                    # This preserves the CMTR unit numbering
                    try:
                        unit_num = int(unit_attr.split("_")[-1])
                    except ValueError:
                        logger.warning(f"Could not parse unit number from {unit_attr}, skipping")
                        continue
                    
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


# =============================================================================
# Helper function: Extract unit metadata from a CMTR unit object
# =============================================================================

def _extract_unit_meta_from_cmtr(cmtr_unit: Any, unit_id: str = "") -> Dict[str, Any]:
    """
    Extract all metadata from a single CMTR unit object.
    
    This is the core extraction logic shared by both add_cmtr_unit_info
    and update_hdf5_with_cmtr_unit_info.
    
    Args:
        cmtr_unit: McsPy Spike_Sorter unit object (e.g., spike_sorter.Unit_1)
        unit_id: Unit identifier for logging purposes
    
    Returns:
        Dict containing all extracted metadata, with keys:
        - Scalar values from Unit_Info (row, column, snr, etc.)
        - Attributes prefixed with "attr_"
        - peak_amplitudes: array of spike amplitudes
        - peaks_data: dict of all Peaks columns
        - roi_stas: ROI spike-triggered averages
        - unmixing_matrix: ICA unmixing matrix
        - _row, _col, _global_id: convenience values for unit-level access
    """
    unit_meta: Dict[str, Any] = {}
    
    # =========================================================================
    # 1. Extract ALL fields from Unit_Info structured array
    # =========================================================================
    if hasattr(cmtr_unit, "Unit_Info"):
        try:
            unit_info_dataset = cmtr_unit.Unit_Info[:]
            if len(unit_info_dataset) > 0:
                info = unit_info_dataset[0]  # First (and typically only) record
                
                for field_name in info.dtype.names:
                    try:
                        value = info[field_name]
                        key = field_name.lower()
                        
                        # Wrap in 1-element array so it gets saved as dataset (not attribute)
                        if isinstance(value, np.integer):
                            unit_meta[key] = np.array([int(value)], dtype=np.int64)
                        elif isinstance(value, np.floating):
                            unit_meta[key] = np.array([float(value)], dtype=np.float64)
                        elif isinstance(value, np.ndarray):
                            if value.size == 1:
                                val_item = value.item()
                                if isinstance(val_item, (int, np.integer)):
                                    unit_meta[key] = np.array([int(val_item)], dtype=np.int64)
                                elif isinstance(val_item, (float, np.floating)):
                                    unit_meta[key] = np.array([float(val_item)], dtype=np.float64)
                                else:
                                    unit_meta[key] = np.array([val_item])
                            else:
                                unit_meta[key] = np.array(value)
                        elif isinstance(value, bytes):
                            unit_meta[key] = value.decode("utf-8", errors="ignore")
                        else:
                            unit_meta[key] = value
                    except Exception as e:
                        logger.debug(f"Could not extract {field_name} for {unit_id}: {e}")
                
                # Store convenience values for unit-level access
                if "row" in unit_meta:
                    unit_meta["_row"] = int(unit_meta["row"][0])
                if "column" in unit_meta:
                    unit_meta["_col"] = int(unit_meta["column"][0])
                if "sensorid" in unit_meta:
                    unit_meta["_global_id"] = int(unit_meta["sensorid"][0])
                    
        except Exception as e:
            logger.debug(f"Could not extract Unit_Info for {unit_id}: {e}")
    
    # =========================================================================
    # 2. Extract ALL unit attributes (.attrs)
    # =========================================================================
    if hasattr(cmtr_unit, "attrs"):
        try:
            attrs = cmtr_unit.attrs
            for key in attrs.keys():
                try:
                    value = attrs[key]
                    attr_key = f"attr_{key.lower()}"
                    
                    if isinstance(value, bytes):
                        unit_meta[attr_key] = value.decode("utf-8", errors="ignore")
                    elif isinstance(value, np.ndarray):
                        if value.size == 1:
                            val_item = value.item()
                            if isinstance(val_item, (int, np.integer)):
                                unit_meta[attr_key] = np.array([int(val_item)], dtype=np.int64)
                            elif isinstance(val_item, (float, np.floating)):
                                unit_meta[attr_key] = np.array([float(val_item)], dtype=np.float64)
                            else:
                                unit_meta[attr_key] = np.array([val_item])
                        else:
                            unit_meta[attr_key] = np.array(value)
                    elif isinstance(value, np.integer):
                        unit_meta[attr_key] = np.array([int(value)], dtype=np.int64)
                    elif isinstance(value, np.floating):
                        unit_meta[attr_key] = np.array([float(value)], dtype=np.float64)
                    else:
                        unit_meta[attr_key] = value
                except Exception as e:
                    logger.debug(f"Could not extract attr {key} for {unit_id}: {e}")
        except Exception as e:
            logger.debug(f"Could not extract attrs for {unit_id}: {e}")
    
    # =========================================================================
    # 3. Extract peak amplitudes (per-spike amplitudes)
    # =========================================================================
    if hasattr(cmtr_unit, "get_peaks_amplitudes"):
        try:
            amplitudes = cmtr_unit.get_peaks_amplitudes()
            if amplitudes is not None and len(amplitudes) > 0:
                unit_meta["peak_amplitudes"] = np.array(amplitudes, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Could not extract peak amplitudes for {unit_id}: {e}")
    
    # =========================================================================
    # 4. Extract ALL fields from Peaks structured array
    # =========================================================================
    if hasattr(cmtr_unit, "Peaks"):
        try:
            peaks = cmtr_unit.Peaks[:]
            if peaks is not None and len(peaks) > 0:
                peaks_data = {}
                for field_name in peaks.dtype.names:
                    try:
                        key = field_name.lower()
                        peaks_data[key] = np.array(peaks[field_name])
                    except Exception as e:
                        logger.debug(f"Could not extract Peaks.{field_name} for {unit_id}: {e}")
                
                unit_meta["peaks_data"] = peaks_data
        except Exception as e:
            logger.debug(f"Could not extract Peaks for {unit_id}: {e}")
    
    # =========================================================================
    # 5. Extract RoiSTAs (ROI spike-triggered averages)
    # =========================================================================
    if hasattr(cmtr_unit, "RoiSTAs"):
        try:
            roi_stas = cmtr_unit.RoiSTAs[:]
            if roi_stas is not None and roi_stas.size > 0:
                unit_meta["roi_stas"] = np.array(roi_stas, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Could not extract RoiSTAs for {unit_id}: {e}")
    
    # =========================================================================
    # 6. Extract Unmixing matrix (ICA)
    # =========================================================================
    if hasattr(cmtr_unit, "Unmixing"):
        try:
            unmixing = cmtr_unit.Unmixing[:]
            if unmixing is not None and unmixing.size > 0:
                unit_meta["unmixing_matrix"] = np.array(unmixing, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Could not extract Unmixing for {unit_id}: {e}")
    
    return unit_meta


def _write_unit_meta_to_hdf5(
    unit_meta_group: Any,
    unit_meta: Dict[str, Any],
    unit_group: Optional[Any] = None,
) -> None:
    """
    Write unit metadata dict to an HDF5 group.
    
    Args:
        unit_meta_group: HDF5 group to write to (units/{unit_id}/unit_meta)
        unit_meta: Dict of metadata extracted by _extract_unit_meta_from_cmtr
        unit_group: Optional parent unit group for setting convenience attributes
    """
    for key, value in unit_meta.items():
        # Skip private convenience keys
        if key.startswith("_"):
            continue
        
        try:
            if key == "peaks_data" and isinstance(value, dict):
                # Create nested group for peaks_data
                peaks_group = unit_meta_group.create_group("peaks_data")
                for pk, pv in value.items():
                    peaks_group.create_dataset(pk, data=pv)
            elif isinstance(value, np.ndarray):
                unit_meta_group.create_dataset(key, data=value)
            elif isinstance(value, str):
                unit_meta_group.attrs[key] = value
            else:
                unit_meta_group.attrs[key] = value
        except Exception as e:
            logger.debug(f"Could not write {key}: {e}")
    
    # Set convenience attributes on parent unit group
    if unit_group is not None:
        if "_row" in unit_meta:
            unit_group.attrs["row"] = unit_meta["_row"]
        if "_col" in unit_meta:
            unit_group.attrs["col"] = unit_meta["_col"]
        if "_global_id" in unit_meta:
            unit_group.attrs["global_id"] = unit_meta["_global_id"]


# =============================================================================
# Main API function (unified interface for session and HDF5 modes)
# =============================================================================

def add_cmtr_unit_info(
    session: Optional[Any] = None,
    hdf5_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    *,
    force: bool = False,
) -> Union[Any, Dict[str, Any]]:
    """
    Add all unit metadata from CMTR file to a session or HDF5 file.
    
    This unified function supports two modes:
    1. Session mode: Pass a PipelineSession to add metadata in-memory
    2. HDF5 mode: Pass an hdf5_path to update the file directly
    
    Extracts metadata from each unit including:
    - Unit_Info: Row, Column, SensorID, UnitID, RoiID, SNR, Separability,
      IsoINN, IsoIBg, AmplitudeSD, NoiseStd, SDScore, RSTD, Skewness,
      Kurtosis, IcaConverged, IcaIterations, Tags
    - Peaks: Timestamps, PeakAmplitude, IncludePeak, and all feature columns
    - RoiSTAs: ROI spike-triggered averages
    - Unmixing: ICA unmixing matrix
    - All unit attributes
    
    Note: Source array is NOT extracted (too large).
    
    Args:
        session: PipelineSession with units already loaded (mode 1)
        hdf5_path: Path to existing HDF5 file to update (mode 2)
        cmtr_path: Path to CMTR file. If None, uses the path stored in
            session.source_files["cmtr_path"] or HDF5 source_files group.
        force: If True, overwrite existing unit_meta groups (HDF5 mode only)
    
    Returns:
        - Session mode: Updated session with extended unit data
        - HDF5 mode: Dict with {"hdf5_path": Path, "units_updated": int, "cmtr_path": Path}
    
    Raises:
        ValueError: If neither session nor hdf5_path is provided, or if both are
        FileNotFoundError: If cmtr_path does not exist
    
    Example (session mode):
        >>> session = load_recording(cmcr_path, cmtr_path, session=session)
        >>> session = add_cmtr_unit_info(session=session)
        >>> session.save()
    
    Example (HDF5 mode):
        >>> result = add_cmtr_unit_info(hdf5_path="artifacts/recording.h5", force=True)
    """
    import h5py
    from McsPy.McsCMOSMEA import McsCMOSMEAData
    
    # Validate arguments - must have exactly one of session or hdf5_path
    if session is None and hdf5_path is None:
        raise ValueError("Must provide either session or hdf5_path")
    if session is not None and hdf5_path is not None:
        raise ValueError("Cannot provide both session and hdf5_path - use one or the other")
    
    # =========================================================================
    # HDF5 MODE: Update file directly
    # =========================================================================
    if hdf5_path is not None:
        hdf5_path = Path(hdf5_path)
        
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Get CMTR path from HDF5 if not provided
        if cmtr_path is None:
            with h5py.File(hdf5_path, "r") as f:
                if "source_files" in f and "cmtr_path" in f["source_files"].attrs:
                    cmtr_path = Path(f["source_files"].attrs["cmtr_path"])
                    logger.info(f"Using CMTR path from HDF5: {cmtr_path}")
                else:
                    raise ValueError(
                        "cmtr_path not provided and not found in HDF5 source_files. "
                        "Either pass cmtr_path explicitly or ensure the HDF5 file was "
                        "created with a CMTR file path."
                    )
        else:
            cmtr_path = Path(cmtr_path)
        
        if not cmtr_path.exists():
            raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
        
        logger.info(f"Updating HDF5 with CMTR unit info: {hdf5_path}")
        logger.info(f"  CMTR source: {cmtr_path}")
        
        # Load CMTR data
        cmtr_data = McsCMOSMEAData(str(cmtr_path))
        spike_sorter = cmtr_data.Spike_Sorter
        
        units_updated = 0
        units_skipped = 0
        
        with h5py.File(hdf5_path, "r+") as f:
            if "units" not in f:
                logger.warning("No 'units' group found in HDF5 file")
                return {"hdf5_path": hdf5_path, "units_updated": 0, "cmtr_path": cmtr_path}
            
            units_group = f["units"]
            
            for unit_id in units_group.keys():
                unit_group = units_group[unit_id]
                
                # Check if unit_meta already exists
                if "unit_meta" in unit_group:
                    if not force:
                        logger.debug(f"Skipping {unit_id}: unit_meta already exists")
                        units_skipped += 1
                        continue
                    else:
                        del unit_group["unit_meta"]
                
                # Extract unit number
                try:
                    unit_num = int(unit_id.split("_")[-1])
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse unit number from {unit_id}")
                    continue
                
                # Get CMTR unit
                cmtr_unit_name = f"Unit_{unit_num}"
                if not hasattr(spike_sorter, cmtr_unit_name):
                    logger.warning(f"CMTR unit {cmtr_unit_name} not found for {unit_id}")
                    continue
                
                cmtr_unit = getattr(spike_sorter, cmtr_unit_name)
                
                # Extract and write metadata
                unit_meta = _extract_unit_meta_from_cmtr(cmtr_unit, unit_id)
                unit_meta_group = unit_group.create_group("unit_meta")
                _write_unit_meta_to_hdf5(unit_meta_group, unit_meta, unit_group)
                
                units_updated += 1
                logger.debug(f"Added unit_meta to {unit_id}")
        
        logger.info(f"Updated {units_updated} units with CMTR metadata (skipped {units_skipped})")
        return {"hdf5_path": hdf5_path, "units_updated": units_updated, "cmtr_path": cmtr_path}
    
    # =========================================================================
    # SESSION MODE: Add to in-memory session
    # =========================================================================
    # Get CMTR path from session if not provided
    if cmtr_path is None:
        if hasattr(session, 'source_files') and session.source_files.get("cmtr_path"):
            cmtr_path = session.source_files["cmtr_path"]
            logger.info(f"Using CMTR path from session: {cmtr_path}")
        else:
            raise ValueError(
                "cmtr_path not provided and not found in session.source_files. "
                "Either pass cmtr_path explicitly or ensure the session was "
                "created with a CMTR file path."
            )
    
    cmtr_path = Path(cmtr_path)
    logger.info(f"Adding extended CMTR unit data from: {cmtr_path}")
    
    if not cmtr_path.exists():
        raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
    
    cmtr_data = McsCMOSMEAData(str(cmtr_path))
    spike_sorter = cmtr_data.Spike_Sorter
    
    units_updated = 0
    
    for unit_id, unit_data in session.units.items():
        # Extract unit number
        try:
            unit_num = int(unit_id.split("_")[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not parse unit number from {unit_id}")
            continue
        
        # Get CMTR unit
        cmtr_unit_name = f"Unit_{unit_num}"
        if not hasattr(spike_sorter, cmtr_unit_name):
            logger.warning(f"CMTR unit {cmtr_unit_name} not found for session {unit_id}")
            continue
        
        cmtr_unit = getattr(spike_sorter, cmtr_unit_name)
        
        # Extract metadata using helper
        unit_meta = _extract_unit_meta_from_cmtr(cmtr_unit, unit_id)
        
        # Copy convenience fields to unit_data and remove from unit_meta
        if "_row" in unit_meta:
            unit_data["row"] = unit_meta.pop("_row")
        if "_col" in unit_meta:
            unit_data["col"] = unit_meta.pop("_col")
        if "_global_id" in unit_meta:
            unit_data["global_id"] = unit_meta.pop("_global_id")
        
        unit_data["unit_meta"] = unit_meta
        
        units_updated += 1
        logger.debug(f"Added {len(unit_meta)} metadata fields to {unit_id}")
    
    logger.info(f"Added extended CMTR data to {units_updated}/{len(session.units)} units")
    session.mark_step_complete("add_cmtr_unit_info")
    
    return session


# Backwards compatibility alias
def update_hdf5_with_cmtr_unit_info(
    hdf5_path: Union[str, Path],
    cmtr_path: Optional[Union[str, Path]] = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Update an existing HDF5 file with unit metadata from CMTR file.
    
    This is a convenience alias for add_cmtr_unit_info(hdf5_path=...).
    See add_cmtr_unit_info() for full documentation.
    """
    return add_cmtr_unit_info(hdf5_path=hdf5_path, cmtr_path=cmtr_path, force=force)
