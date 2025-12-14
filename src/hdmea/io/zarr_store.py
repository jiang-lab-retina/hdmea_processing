"""
Zarr store operations for HD-MEA pipeline.

Handles reading and writing of Zarr archives that store recording data and features.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import zarr

from hdmea.utils.hashing import hash_config


logger = logging.getLogger(__name__)


def create_recording_zarr(
    zarr_path: Path,
    dataset_id: str,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> zarr.Group:
    """
    Create a new Zarr archive for a recording.
    
    Args:
        zarr_path: Path for the Zarr store
        dataset_id: Unique identifier for the recording
        config: Configuration dictionary for Stage 1
        overwrite: If True, overwrite existing Zarr
    
    Returns:
        Root Zarr group
    
    Raises:
        FileExistsError: If Zarr exists and overwrite=False
    """
    if zarr_path.exists() and not overwrite:
        raise FileExistsError(f"Zarr already exists: {zarr_path}")
    
    logger.info(f"Creating Zarr archive: {zarr_path}")
    
    # Create store
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=overwrite)
    
    # Set root metadata
    now = datetime.now(timezone.utc).isoformat()
    root.attrs["dataset_id"] = dataset_id
    root.attrs["hdmea_pipeline_version"] = "0.1.0"
    root.attrs["created_at"] = now
    root.attrs["updated_at"] = now
    root.attrs["stage1_completed"] = False
    root.attrs["stage1_params_hash"] = hash_config(config or {})
    root.attrs["features_extracted"] = []
    
    # Create main groups
    root.create_group("units")
    root.create_group("stimulus")
    root.create_group("metadata")
    
    return root


def open_recording_zarr(
    zarr_path: Path,
    mode: str = "r",
) -> zarr.Group:
    """
    Open an existing Zarr archive.
    
    Args:
        zarr_path: Path to Zarr store
        mode: Open mode ("r" for read, "r+" for read/write, "a" for append)
    
    Returns:
        Root Zarr group
    
    Raises:
        FileNotFoundError: If Zarr does not exist
    """
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")
    
    store = zarr.DirectoryStore(str(zarr_path))
    return zarr.open_group(store=store, mode=mode)


def write_units(
    root: zarr.Group,
    units_data: Dict[str, Dict[str, Any]],
) -> None:
    """
    Write unit data to Zarr.
    
    Args:
        root: Root Zarr group
        units_data: Dictionary mapping unit_id -> unit data
    """
    units_group = root["units"]
    
    for unit_id, unit_info in units_data.items():
        logger.debug(f"Writing unit: {unit_id}")
        
        # Create unit group
        unit_group = units_group.create_group(unit_id, overwrite=True)
        
        # Write spike times (in microseconds)
        spike_times = unit_info.get("spike_times", np.array([]))
        if len(spike_times) > 0:
            unit_group.create_dataset(
                "spike_times",
                data=spike_times.astype(np.uint64),
                dtype=np.uint64,
            )
        
        # Write waveform
        waveform = unit_info.get("waveform", np.array([]))
        if len(waveform) > 0:
            unit_group.create_dataset(
                "waveform",
                data=waveform.astype(np.float32),
                dtype=np.float32,
            )
        
        # Write firing rate if available
        firing_rate = unit_info.get("firing_rate_10hz")
        if firing_rate is not None and len(firing_rate) > 0:
            unit_group.create_dataset(
                "firing_rate_10hz",
                data=firing_rate.astype(np.float32),
                dtype=np.float32,
            )
        
        # Set unit metadata
        unit_group.attrs["row"] = unit_info.get("row", 0)
        unit_group.attrs["col"] = unit_info.get("col", 0)
        unit_group.attrs["global_id"] = unit_info.get("global_id", 0)
        unit_group.attrs["spike_count"] = len(spike_times)
        
        # Create features group for this unit
        unit_group.create_group("features")
    
    logger.info(f"Wrote {len(units_data)} units to Zarr")


def write_stimulus(
    root: zarr.Group,
    light_reference: Dict[str, np.ndarray],
    frame_times: Optional[Dict[str, np.ndarray]] = None,
    section_times: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Write stimulus data to Zarr.
    
    Args:
        root: Root Zarr group
        light_reference: Dict mapping sample rate -> light data
        frame_times: Optional dict mapping movie name -> frame timestamps
        section_times: Optional dict mapping section name -> (start, end) arrays
    """
    stimulus_group = root["stimulus"]
    
    # Write light reference at different sample rates
    if light_reference:
        lr_group = stimulus_group.create_group("light_reference", overwrite=True)
        for rate_name, data in light_reference.items():
            if data is not None and len(data) > 0:
                lr_group.create_dataset(
                    rate_name,
                    data=data.astype(np.float32),
                    dtype=np.float32,
                )
        logger.info(f"Wrote light reference: {list(light_reference.keys())}")
    
    # Write frame times
    if frame_times:
        ft_group = stimulus_group.create_group("frame_time", overwrite=True)
        for movie_name, times in frame_times.items():
            if times is not None and len(times) > 0:
                ft_group.create_dataset(
                    movie_name,
                    data=times.astype(np.uint64),
                    dtype=np.uint64,
                )
    
    # Write section times
    if section_times:
        st_group = stimulus_group.create_group("section_time", overwrite=True)
        for section_name, times in section_times.items():
            if times is not None:
                st_group.create_dataset(
                    section_name,
                    data=np.array(times, dtype=np.uint64),
                    dtype=np.uint64,
                )


def write_metadata(
    root: zarr.Group,
    metadata: Dict[str, Any],
) -> None:
    """
    Write recording metadata to Zarr.
    
    Args:
        root: Root Zarr group
        metadata: Metadata dictionary
    """
    metadata_group = root["metadata"]
    
    # Write scalar metadata as datasets
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            metadata_group.create_dataset(
                key,
                data=np.array(value),
                overwrite=True,
            )
        elif isinstance(value, str):
            # Store strings as attributes
            metadata_group.attrs[key] = value
        elif isinstance(value, dict):
            # Store dicts as JSON attributes
            metadata_group.attrs[key] = json.dumps(value)
    
    logger.debug(f"Wrote metadata: {list(metadata.keys())}")


def write_source_files(
    root: zarr.Group,
    cmcr_path: Optional[Path],
    cmtr_path: Optional[Path],
) -> None:
    """
    Write source file information to Zarr metadata.
    
    Args:
        root: Root Zarr group
        cmcr_path: Path to CMCR file (or None)
        cmtr_path: Path to CMTR file (or None)
    """
    root.attrs["source_files"] = {
        "cmcr_path": str(cmcr_path) if cmcr_path else None,
        "cmtr_path": str(cmtr_path) if cmtr_path else None,
        "cmcr_exists": cmcr_path is not None,
        "cmtr_exists": cmtr_path is not None,
    }


def mark_stage1_complete(root: zarr.Group) -> None:
    """Mark Stage 1 as complete in Zarr metadata."""
    now = datetime.now(timezone.utc).isoformat()
    root.attrs["stage1_completed"] = True
    root.attrs["updated_at"] = now
    logger.info("Stage 1 marked as complete")


def get_stage1_status(root: zarr.Group) -> Dict[str, Any]:
    """
    Get Stage 1 completion status.
    
    Returns:
        Dictionary with stage1_completed, params_hash, etc.
    """
    return {
        "completed": root.attrs.get("stage1_completed", False),
        "params_hash": root.attrs.get("stage1_params_hash"),
        "created_at": root.attrs.get("created_at"),
        "updated_at": root.attrs.get("updated_at"),
    }


def list_units(root: zarr.Group) -> List[str]:
    """List all unit IDs in the Zarr."""
    return list(root["units"].keys())


def list_features(root: zarr.Group, unit_id: str) -> List[str]:
    """List all features extracted for a unit."""
    try:
        return list(root["units"][unit_id]["features"].keys())
    except KeyError:
        return []


def write_feature_to_unit(
    root: zarr.Group,
    unit_id: str,
    feature_name: str,
    feature_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Write extracted features for a unit to Zarr.
    
    Args:
        root: Root Zarr group
        unit_id: Unit identifier
        feature_name: Feature name (e.g., "step_up_5s_5i_3x")
        feature_data: Dictionary of feature values (scalars or arrays)
        metadata: Feature metadata (version, params_hash, etc.)
    """
    features_group = root["units"][unit_id]["features"]
    
    # Create feature group
    feature_group = features_group.create_group(feature_name, overwrite=True)
    
    # Write feature data
    for key, value in feature_data.items():
        if isinstance(value, np.ndarray):
            feature_group.create_dataset(key, data=value)
        elif isinstance(value, (list, tuple)):
            feature_group.create_dataset(key, data=np.array(value))
        elif isinstance(value, dict):
            # Nested dict -> create subgroup
            subgroup = feature_group.create_group(key)
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    subgroup.create_dataset(subkey, data=subvalue)
                else:
                    subgroup.attrs[subkey] = subvalue
        else:
            # Scalar values as datasets
            feature_group.create_dataset(key, data=np.array(value))
    
    # Set feature metadata
    feature_group.attrs.update(metadata)
    
    # Update root-level feature list
    features_list = list(root.attrs.get("features_extracted", []))
    if feature_name not in features_list:
        features_list.append(feature_name)
        root.attrs["features_extracted"] = features_list
    
    logger.debug(f"Wrote feature {feature_name} for unit {unit_id}")

