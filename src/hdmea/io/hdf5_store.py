"""
HDF5 store operations for HD-MEA pipeline.

Handles reading and writing of HDF5 archives that store recording data and features.
This module replaces zarr_store.py, providing equivalent functionality with HDF5 format.

Key differences from Zarr:
- Single file (.h5) instead of directory (.zarr)
- Context manager required for proper file handling
- Single-writer access model (no concurrent writes)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from hdmea.utils.hashing import hash_config


logger = logging.getLogger(__name__)


def create_recording_hdf5(
    hdf5_path: Path,
    dataset_id: str,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> h5py.File:
    """
    Create a new HDF5 archive for a recording.
    
    Args:
        hdf5_path: Path for the HDF5 file (should end with .h5)
        dataset_id: Unique identifier for the recording
        config: Configuration dictionary for Stage 1
        overwrite: If True, overwrite existing file
    
    Returns:
        Open h5py.File handle (caller must close or use with statement)
    
    Raises:
        FileExistsError: If file exists and overwrite=False
        OSError: If file is locked by another process
    """
    hdf5_path = Path(hdf5_path)
    
    # Warn if extension is not .h5 or .hdf5
    if hdf5_path.suffix.lower() not in ('.h5', '.hdf5'):
        logger.warning(f"HDF5 file has non-standard extension: {hdf5_path.suffix}")
    
    if hdf5_path.exists() and not overwrite:
        raise FileExistsError(f"HDF5 file already exists: {hdf5_path}")
    
    logger.info(f"Creating HDF5 archive: {hdf5_path}")
    
    # Delete existing file if overwriting
    if overwrite and hdf5_path.exists():
        hdf5_path.unlink()
    
    # Ensure parent directory exists
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        root = h5py.File(str(hdf5_path), mode='w')
    except OSError as e:
        if "already open" in str(e).lower() or "locked" in str(e).lower():
            raise OSError(
                f"File is already open for writing: {hdf5_path}. "
                "HDF5 supports single-writer access only. "
                "Close the file in other processes before writing."
            ) from e
        raise
    
    # Set root metadata as attributes
    now = datetime.now(timezone.utc).isoformat()
    root.attrs["dataset_id"] = dataset_id
    root.attrs["hdmea_pipeline_version"] = "0.1.0"
    root.attrs["created_at"] = now
    root.attrs["updated_at"] = now
    root.attrs["stage1_completed"] = False
    root.attrs["stage1_params_hash"] = hash_config(config or {})
    # Store features_extracted as JSON string (HDF5 doesn't support list attrs well)
    root.attrs["features_extracted"] = "[]"
    
    # Create main groups
    root.create_group("units")
    root.create_group("stimulus")
    root.create_group("metadata")
    
    return root


def open_recording_hdf5(
    hdf5_path: Path,
    mode: str = "r",
) -> h5py.File:
    """
    Open an existing HDF5 archive.
    
    Args:
        hdf5_path: Path to HDF5 file
        mode: Open mode ("r" for read, "r+" for read/write, "a" for append)
    
    Returns:
        Open h5py.File handle
    
    Raises:
        FileNotFoundError: If file does not exist
        OSError: If file is locked (write modes only)
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    try:
        return h5py.File(str(hdf5_path), mode=mode)
    except OSError as e:
        error_str = str(e).lower()
        if "already open" in error_str or "locked" in error_str:
            raise OSError(
                f"File is already open for writing: {hdf5_path}. "
                "HDF5 supports single-writer access only. "
                "Close the file in other processes before writing."
            ) from e
        if "unable to open" in error_str or "cannot read" in error_str:
            raise OSError(
                f"Cannot read HDF5 file: {hdf5_path}. "
                "The file may be corrupted or incomplete."
            ) from e
        raise


def _write_metadata_to_group(
    group: h5py.Group,
    data: Dict[str, Any],
) -> None:
    """
    Write metadata dictionary to an HDF5 group.
    
    Helper function that recursively writes metadata values as datasets.
    
    Args:
        group: HDF5 group to write to
        data: Dictionary of values to write
    """
    for key, value in data.items():
        # Delete existing if present
        if key in group:
            del group[key]
        
        if isinstance(value, (int, float)):
            # Store scalars as 1-element arrays (visible in tree)
            arr = np.array([value])
            group.create_dataset(key, data=arr, dtype=arr.dtype)
        elif isinstance(value, str):
            # Store strings as variable-length UTF-8
            dt = h5py.string_dtype(encoding='utf-8')
            group.create_dataset(key, data=value, dtype=dt)
        elif isinstance(value, dict):
            # Create subgroup for nested dicts (e.g., sys_meta)
            if key in group:
                subgroup = group[key]
            else:
                subgroup = group.create_group(key)
            _write_metadata_to_group(subgroup, value)
        elif isinstance(value, np.ndarray):
            # Handle numpy arrays directly
            if value.ndim > 0:
                group.create_dataset(key, data=value, dtype=value.dtype)
            else:
                # Scalar numpy array -> wrap in 1-element array
                arr = np.array([value.item()])
                group.create_dataset(key, data=arr, dtype=arr.dtype)


def write_units(
    root: h5py.File,
    units_data: Dict[str, Dict[str, Any]],
) -> None:
    """
    Write unit data to HDF5.
    
    Args:
        root: Open h5py.File handle
        units_data: Dictionary mapping unit_id -> unit data
            Expected keys per unit:
            - spike_times: np.ndarray (uint64)
            - waveform: np.ndarray (float32)
            - firing_rate_10hz: np.ndarray (float32) [optional]
            - row: int
            - col: int
            - global_id: int
    """
    units_group = root["units"]
    
    for unit_id, unit_info in units_data.items():
        logger.debug(f"Writing unit: {unit_id}")
        
        # Delete existing unit group if present
        if unit_id in units_group:
            del units_group[unit_id]
        
        # Create unit group
        unit_group = units_group.create_group(unit_id)
        
        # Write spike times (in sample indices)
        spike_times = unit_info.get("spike_times", np.array([]))
        if len(spike_times) > 0:
            spike_arr = spike_times.astype(np.uint64)
            spike_ds = unit_group.create_dataset(
                "spike_times",
                data=spike_arr,
                dtype=np.uint64,
            )
            # Add unit metadata attribute
            spike_ds.attrs["unit"] = "sample_index"
        
        # Write waveform
        waveform = unit_info.get("waveform", np.array([]))
        if len(waveform) > 0:
            waveform_arr = waveform.astype(np.float32)
            unit_group.create_dataset(
                "waveform",
                data=waveform_arr,
                dtype=np.float32,
            )
        
        # Write firing rate if available
        firing_rate = unit_info.get("firing_rate_10hz")
        if firing_rate is not None and len(firing_rate) > 0:
            fr_arr = firing_rate.astype(np.float32)
            unit_group.create_dataset(
                "firing_rate_10hz",
                data=fr_arr,
                dtype=np.float32,
            )
        
        # Set unit metadata as attributes
        unit_group.attrs["row"] = unit_info.get("row", 0)
        unit_group.attrs["col"] = unit_info.get("col", 0)
        unit_group.attrs["global_id"] = unit_info.get("global_id", 0)
        unit_group.attrs["spike_count"] = len(spike_times)
        
        # Create features group for this unit
        unit_group.create_group("features")
    
    logger.info(f"Wrote {len(units_data)} units to HDF5")


def write_stimulus(
    root: h5py.File,
    light_reference: Dict[str, np.ndarray],
    frame_times: Optional[Dict[str, np.ndarray]] = None,
    section_times: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Write stimulus data to HDF5.
    
    Args:
        root: Open h5py.File handle
        light_reference: Dict mapping rate_name -> light data array
        frame_times: Optional dict mapping movie_name -> frame timestamps
        section_times: Optional dict mapping movie_name -> (n_trials, 2) boundaries
    """
    stimulus_group = root["stimulus"]
    
    # Write light reference at different sample rates
    if light_reference:
        # Delete and recreate light_reference group
        if "light_reference" in stimulus_group:
            del stimulus_group["light_reference"]
        lr_group = stimulus_group.create_group("light_reference")
        
        for rate_name, data in light_reference.items():
            if data is not None and len(data) > 0:
                data_arr = data.astype(np.float32)
                lr_group.create_dataset(
                    rate_name,
                    data=data_arr,
                    dtype=np.float32,
                )
        logger.info(f"Wrote light reference: {list(light_reference.keys())}")
    
    # Write frame times
    if frame_times:
        if "frame_time" in stimulus_group:
            del stimulus_group["frame_time"]
        ft_group = stimulus_group.create_group("frame_time")
        
        for movie_name, times in frame_times.items():
            if times is not None and len(times) > 0:
                times_arr = times.astype(np.uint64)
                ft_group.create_dataset(
                    movie_name,
                    data=times_arr,
                    dtype=np.uint64,
                )
    
    # Write section times
    if section_times:
        if "section_time" in stimulus_group:
            del stimulus_group["section_time"]
        st_group = stimulus_group.create_group("section_time")
        
        for section_name, times in section_times.items():
            if times is not None:
                times_arr = np.array(times, dtype=np.uint64)
                st_group.create_dataset(
                    section_name,
                    data=times_arr,
                    dtype=np.uint64,
                )


def write_metadata(
    root: h5py.File,
    metadata: Dict[str, Any],
) -> None:
    """
    Write recording metadata to HDF5.
    
    Top-level timing metadata (acquisition_rate, frame_time, frame_timestamps)
    is stored directly in the metadata group. System metadata from raw files
    (CMCR/CMTR) is stored under metadata/sys_meta subgroup.
    
    Args:
        root: Open h5py.File handle
        metadata: Metadata dictionary (may contain nested 'sys_meta' dict)
    """
    metadata_group = root["metadata"]
    
    # Write metadata recursively (handles sys_meta as subgroup)
    _write_metadata_to_group(metadata_group, metadata)
    
    logger.debug(f"Wrote metadata: {list(metadata.keys())}")


def write_source_files(
    root: h5py.File,
    cmcr_path: Optional[Path],
    cmtr_path: Optional[Path],
) -> None:
    """
    Write source file information to HDF5 metadata.
    
    Args:
        root: Open h5py.File handle
        cmcr_path: Path to CMCR file (or None)
        cmtr_path: Path to CMTR file (or None)
    """
    # Store as JSON string since HDF5 attrs don't handle dicts well
    import json
    source_info = {
        "cmcr_path": str(cmcr_path) if cmcr_path else None,
        "cmtr_path": str(cmtr_path) if cmtr_path else None,
        "cmcr_exists": cmcr_path is not None,
        "cmtr_exists": cmtr_path is not None,
    }
    root.attrs["source_files"] = json.dumps(source_info)


def mark_stage1_complete(root: h5py.File) -> None:
    """Mark Stage 1 as complete in HDF5 metadata."""
    now = datetime.now(timezone.utc).isoformat()
    root.attrs["stage1_completed"] = True
    root.attrs["updated_at"] = now
    logger.info("Stage 1 marked as complete")


def get_stage1_status(root: h5py.File) -> Dict[str, Any]:
    """
    Get Stage 1 completion status.
    
    Returns:
        Dictionary with: completed, params_hash, created_at, updated_at
    """
    return {
        "completed": bool(root.attrs.get("stage1_completed", False)),
        "params_hash": root.attrs.get("stage1_params_hash"),
        "created_at": root.attrs.get("created_at"),
        "updated_at": root.attrs.get("updated_at"),
    }


def list_units(root: h5py.File) -> List[str]:
    """List all unit IDs in the HDF5 file."""
    return list(root["units"].keys())


def list_features(root: h5py.File, unit_id: str) -> List[str]:
    """List all features extracted for a unit."""
    try:
        return list(root["units"][unit_id]["features"].keys())
    except KeyError:
        return []


def write_feature_to_unit(
    root: h5py.File,
    unit_id: str,
    feature_name: str,
    feature_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Write extracted features for a unit to HDF5.
    
    Args:
        root: Open h5py.File handle
        unit_id: Unit identifier (e.g., "unit_000")
        feature_name: Feature name (e.g., "step_up_5s_5i_3x")
        feature_data: Dictionary of feature values (scalars or arrays)
        metadata: Feature metadata (version, params_hash, extracted_at)
    """
    import json
    
    features_group = root["units"][unit_id]["features"]
    
    # Delete existing feature group if present
    if feature_name in features_group:
        del features_group[feature_name]
    
    # Create feature group
    feature_group = features_group.create_group(feature_name)
    
    # Write feature data
    for key, value in feature_data.items():
        if isinstance(value, np.ndarray):
            if value.ndim > 0:
                feature_group.create_dataset(key, data=value, dtype=value.dtype)
            else:
                # Scalar numpy array -> store as attribute
                feature_group.attrs[key] = value.item()
        elif isinstance(value, (list, tuple)):
            arr = np.array(value)
            if arr.ndim > 0:
                feature_group.create_dataset(key, data=arr, dtype=arr.dtype)
            else:
                feature_group.attrs[key] = arr.item()
        elif isinstance(value, dict):
            # Nested dict -> create subgroup
            subgroup = feature_group.create_group(key)
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray) and subvalue.ndim > 0:
                    subgroup.create_dataset(subkey, data=subvalue, dtype=subvalue.dtype)
                elif isinstance(subvalue, np.ndarray):
                    subgroup.attrs[subkey] = subvalue.item()
                else:
                    subgroup.attrs[subkey] = subvalue
        else:
            # Scalar values -> store as attributes
            feature_group.attrs[key] = value
    
    # Set feature metadata as attributes
    for meta_key, meta_value in metadata.items():
        feature_group.attrs[meta_key] = meta_value
    
    # Update root-level feature list
    features_json = root.attrs.get("features_extracted", "[]")
    features_list = json.loads(features_json)
    if feature_name not in features_list:
        features_list.append(feature_name)
        root.attrs["features_extracted"] = json.dumps(features_list)
    
    logger.debug(f"Wrote feature {feature_name} for unit {unit_id}")

