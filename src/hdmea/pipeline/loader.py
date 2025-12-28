"""
Universal HDF5 Loader for Pipeline Sessions.

Provides `load_session_from_hdf5()` function that loads all data from an existing
HDF5 file into a PipelineSession for further processing.

Features:
    - Recursive loading of all HDF5 groups and datasets
    - Optional feature filtering to reduce memory usage
    - Restoration of completed_steps for resume capability
    - Support for all data types stored in the HDF5 structure

Usage:
    >>> from hdmea.pipeline.loader import load_session_from_hdf5
    >>> session = load_session_from_hdf5("path/to/file.h5")
    >>> session = load_session_from_hdf5(
    ...     "path/to/file.h5",
    ...     load_features=["eimage_sta", "sta"]  # Only load specific features
    ... )
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import h5py
import numpy as np

from hdmea.pipeline.session import PipelineSession, SaveState, create_session

logger = logging.getLogger(__name__)


def load_session_from_hdf5(
    hdf5_path: Union[str, Path],
    *,
    dataset_id: Optional[str] = None,
    load_features: Optional[List[str]] = None,
) -> PipelineSession:
    """
    Load an existing HDF5 file into a PipelineSession.
    
    Recursively loads all data from the HDF5 file. Optionally filters
    which features to load to reduce memory usage.
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Override dataset_id (uses file stem if not provided)
        load_features: List of feature names to load. If None, loads all features.
                       Example: ["eimage_sta", "sta", "ap_tracking"]
    
    Returns:
        PipelineSession with loaded data (save_state=DEFERRED, ready for more processing)
    
    Raises:
        FileNotFoundError: If hdf5_path does not exist
        ValueError: If file is not a valid HDF5 or has no units group
    
    Example:
        >>> session = load_session_from_hdf5("path/to/file.h5")
        >>> session = load_session_from_hdf5(
        ...     "path/to/file.h5",
        ...     load_features=["eimage_sta"]
        ... )
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    # Determine dataset_id
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    logger.info(f"Loading session from: {hdf5_path}")
    
    # Create session
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path  # Track source file
    
    # Convert load_features to set for faster lookup
    feature_filter: Optional[Set[str]] = None
    if load_features is not None:
        feature_filter = set(load_features)
        logger.info(f"Filtering features to: {feature_filter}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get dataset_id from file attributes if available (prefer file's stored value)
        if 'dataset_id' in f.attrs:
            file_dataset_id = f.attrs['dataset_id']
            if isinstance(file_dataset_id, bytes):
                file_dataset_id = file_dataset_id.decode('utf-8')
            # Use file's dataset_id if we defaulted to filename
            if dataset_id == hdf5_path.stem and file_dataset_id:
                logger.debug(f"Using dataset_id from file: {file_dataset_id}")
                session.dataset_id = file_dataset_id
        
        # Load units
        if 'units' in f:
            session.units = _load_units(f['units'], feature_filter)
            logger.info(f"Loaded {len(session.units)} units")
        else:
            logger.warning(f"No 'units' group found in {hdf5_path}")
        
        # Load metadata
        if 'metadata' in f:
            session.metadata = _read_group_recursive(f['metadata'])
            logger.debug(f"Loaded metadata with {len(session.metadata)} keys")
        
        # Load stimulus
        if 'stimulus' in f:
            session.stimulus = _read_group_recursive(f['stimulus'])
            logger.debug(f"Loaded stimulus with {len(session.stimulus)} keys")
        
        # Load source files
        if 'source_files' in f:
            for key in f['source_files'].attrs:
                value = f['source_files'].attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                session.source_files[key] = Path(value) if value else None
        
        # Restore pipeline info
        if 'pipeline' in f:
            _restore_pipeline_info(f['pipeline'], session)
    
    # Mark as loaded from HDF5
    session.completed_steps.add('load_from_hdf5')
    
    # Keep session in DEFERRED state so it can be modified
    session.save_state = SaveState.DEFERRED
    
    logger.info(f"Session loaded: {session.unit_count} units, "
                f"{len(session.completed_steps)} completed steps")
    
    return session


def _load_units(
    units_group: h5py.Group,
    feature_filter: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all units from HDF5 units group.
    
    Args:
        units_group: HDF5 group containing unit data
        feature_filter: Optional set of feature names to load
    
    Returns:
        Dict mapping unit_id to unit data dict
    """
    units = {}
    
    for unit_id in units_group:
        unit_group = units_group[unit_id]
        unit_data = _load_single_unit(unit_group, feature_filter)
        units[unit_id] = unit_data
    
    return units


def _load_single_unit(
    unit_group: h5py.Group,
    feature_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Load data for a single unit.
    
    Args:
        unit_group: HDF5 group for the unit
        feature_filter: Optional set of feature names to load
    
    Returns:
        Dict with unit data
    """
    unit_data = {}
    
    # Load attributes
    for key in unit_group.attrs:
        value = unit_group.attrs[key]
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        unit_data[key] = value
    
    # Load datasets and subgroups
    for key in unit_group:
        item = unit_group[key]
        
        if key == 'features':
            # Handle features with optional filtering
            unit_data['features'] = _load_features(item, feature_filter)
        elif isinstance(item, h5py.Dataset):
            unit_data[key] = _read_dataset(item)
        elif isinstance(item, h5py.Group):
            unit_data[key] = _read_group_recursive(item)
    
    return unit_data


def _load_features(
    features_group: h5py.Group,
    feature_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Load features from HDF5 group with optional filtering.
    
    Args:
        features_group: HDF5 group containing features
        feature_filter: Optional set of feature names to load
    
    Returns:
        Dict mapping feature name to feature data
    """
    features = {}
    
    for feature_name in features_group:
        # Apply feature filter if specified
        if feature_filter is not None and feature_name not in feature_filter:
            continue
        
        feature_group = features_group[feature_name]
        if isinstance(feature_group, h5py.Group):
            features[feature_name] = _read_group_recursive(feature_group)
        elif isinstance(feature_group, h5py.Dataset):
            features[feature_name] = _read_dataset(feature_group)
    
    return features


def _read_group_recursive(group: h5py.Group) -> Dict[str, Any]:
    """
    Recursively read an HDF5 group into a dict.
    
    Args:
        group: HDF5 group to read
    
    Returns:
        Dict with group contents
    """
    result = {}
    
    # Read attributes
    for key in group.attrs:
        value = group.attrs[key]
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        result[key] = value
    
    # Read datasets and subgroups
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = _read_dataset(item)
        elif isinstance(item, h5py.Group):
            result[key] = _read_group_recursive(item)
    
    return result


def _read_dataset(dataset: h5py.Dataset) -> Any:
    """
    Read an HDF5 dataset and convert to appropriate Python type.
    
    Args:
        dataset: HDF5 dataset to read
    
    Returns:
        Dataset value (numpy array, scalar, or string)
    """
    value = dataset[()]
    
    # Handle scalar datasets
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            # Scalar array
            return value.item()
        elif value.size == 1 and value.ndim == 1:
            # Single-element array - return scalar
            item = value[0]
            if isinstance(item, bytes):
                return item.decode('utf-8')
            return item
    
    # Handle bytes/strings
    if isinstance(value, bytes):
        return value.decode('utf-8')
    
    # Handle numpy bytes arrays
    if isinstance(value, np.ndarray) and value.dtype.kind == 'S':
        # Byte string array
        if value.size == 1:
            return value.flat[0].decode('utf-8')
        return np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in value.flat])
    
    # Handle object dtype (variable-length strings)
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 1:
            item = value.flat[0]
            if isinstance(item, bytes):
                return item.decode('utf-8')
            return item
        return value
    
    return value


def _restore_pipeline_info(pipeline_group: h5py.Group, session: PipelineSession) -> None:
    """
    Restore pipeline tracking info from HDF5.
    
    Args:
        pipeline_group: HDF5 group containing pipeline info
        session: Session to update
    """
    # Restore session_info
    if 'session_info' in pipeline_group:
        session_info = pipeline_group['session_info']
        
        # Restore dataset_id if stored in session_info
        if 'dataset_id' in session_info.attrs:
            stored_id = session_info.attrs['dataset_id']
            if isinstance(stored_id, bytes):
                stored_id = stored_id.decode('utf-8')
            if stored_id and stored_id != session.dataset_id:
                logger.debug(f"Restoring dataset_id from session_info: {stored_id}")
                session.dataset_id = stored_id
        
        # Restore saved_at timestamp
        if 'saved_at' in session_info.attrs:
            session.saved_at = session_info.attrs['saved_at']
            if isinstance(session.saved_at, bytes):
                session.saved_at = session.saved_at.decode('utf-8')
        
        # Restore completed_steps
        if 'completed_steps' in session_info:
            steps = session_info['completed_steps'][:]
            session.completed_steps = set()
            for step in steps:
                if isinstance(step, bytes):
                    step = step.decode('utf-8')
                session.completed_steps.add(step)
        
        # Restore warnings
        if 'warnings' in session_info:
            warnings = session_info['warnings'][:]
            session.warnings = []
            for warning in warnings:
                if isinstance(warning, bytes):
                    warning = warning.decode('utf-8')
                session.warnings.append(warning)


# Alias for backwards compatibility
load_hdf5_to_session = load_session_from_hdf5

