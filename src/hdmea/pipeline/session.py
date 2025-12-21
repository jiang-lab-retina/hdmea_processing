"""
Pipeline session for deferred HDF5 saving.

Provides PipelineSession class that accumulates data across pipeline steps
and supports explicit save/checkpoint operations.
"""

import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import h5py
import numpy as np

from hdmea.utils.exceptions import SessionError, CheckpointError

logger = logging.getLogger(__name__)


def _safe_remove_file(path: Path, max_retries: int = 3, delay: float = 0.5) -> bool:
    """
    Safely remove a file with retries for Windows file locking issues.
    
    Returns True if file was removed or doesn't exist, False if removal failed.
    """
    if not path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            # Force garbage collection to release any Python references
            gc.collect()
            path.unlink()
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                logger.warning(f"File locked, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error(f"Failed to remove locked file after {max_retries} attempts: {path}")
                return False
    return False


class SaveState(Enum):
    """Session persistence state."""
    DEFERRED = "deferred"  # Data in memory only
    SAVED = "saved"        # Data persisted to HDF5


@dataclass
class PipelineSession:
    """
    Container for in-memory pipeline data accumulated across steps.
    
    Supports deferred HDF5 saving - data accumulates in memory until
    explicitly saved via save() or checkpoint().
    
    Attributes:
        dataset_id: Unique identifier for the recording (e.g., "2025.04.10-11.12.57-Rec")
        save_state: Current persistence state (DEFERRED or SAVED)
        hdf5_path: Path to HDF5 file (set after save(), None while deferred)
        output_dir: Default directory for save operations
        
        units: Unit data keyed by unit_id
        metadata: Recording metadata (acquisition_rate, timing, etc.)
        stimulus: Stimulus data (light_reference, frame_times, section_time)
        source_files: Paths to original CMCR/CMTR files
        
        completed_steps: Set of pipeline steps that have been run
        warnings: Accumulated warnings from pipeline operations
    
    Example:
        >>> session = PipelineSession(dataset_id="2025.04.10-11.12.57-Rec")
        >>> session = load_recording(..., session=session)
        >>> session = extract_features(..., session=session)
        >>> session.save()  # Write all data to HDF5
    """
    # Identity
    dataset_id: str
    save_state: SaveState = field(default=SaveState.DEFERRED)
    hdf5_path: Optional[Path] = field(default=None)
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))
    
    # Data containers (mirror HDF5 structure)
    units: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stimulus: Dict[str, Any] = field(default_factory=dict)
    source_files: Dict[str, Optional[Path]] = field(default_factory=dict)
    
    # Pipeline tracking
    completed_steps: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: Optional[str] = field(default=None)
    saved_at: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate and initialize session."""
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        
        self.dataset_id = self.dataset_id.strip()
        
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_saved(self) -> bool:
        """True if session has been saved to disk."""
        return self.save_state == SaveState.SAVED
    
    @property
    def is_deferred(self) -> bool:
        """True if session is in deferred (in-memory only) mode."""
        return self.save_state == SaveState.DEFERRED
    
    @property
    def unit_count(self) -> int:
        """Number of units in the session."""
        return len(self.units)
    
    @property
    def memory_estimate_gb(self) -> float:
        """
        Estimated memory usage in GB (approximate).
        
        Estimates based on numpy array sizes in units and stimulus data.
        """
        total_bytes = 0
        
        # Estimate units data
        for unit_id, unit_data in self.units.items():
            for key, value in unit_data.items():
                if isinstance(value, np.ndarray):
                    total_bytes += value.nbytes
                elif isinstance(value, dict):
                    # Nested dict (features, etc.)
                    for subkey, subval in value.items():
                        if isinstance(subval, np.ndarray):
                            total_bytes += subval.nbytes
        
        # Estimate stimulus data
        for key, value in self.stimulus.items():
            if isinstance(value, np.ndarray):
                total_bytes += value.nbytes
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    if isinstance(subval, np.ndarray):
                        total_bytes += subval.nbytes
        
        return total_bytes / (1024 ** 3)
    
    # =========================================================================
    # Data Accumulation Methods
    # =========================================================================
    
    def add_units(self, units_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Add or update unit data in the session.
        
        Args:
            units_data: Dict mapping unit_id to unit data dict
        
        Side effects:
            Updates self.units (merge, not replace)
        """
        for unit_id, unit_info in units_data.items():
            if unit_id in self.units:
                # Merge with existing
                self.units[unit_id].update(unit_info)
            else:
                self.units[unit_id] = unit_info.copy()
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Add or update metadata in the session.
        
        Args:
            metadata: Dict of metadata key-value pairs
        
        Side effects:
            Updates self.metadata (merge, not replace)
        """
        self.metadata.update(metadata)
    
    def add_stimulus(self, stimulus_data: Dict[str, Any]) -> None:
        """
        Add or update stimulus data in the session.
        
        Args:
            stimulus_data: Dict of stimulus data
        
        Side effects:
            Updates self.stimulus (merge, not replace)
        """
        for key, value in stimulus_data.items():
            if key in self.stimulus and isinstance(self.stimulus[key], dict) and isinstance(value, dict):
                # Deep merge for nested dicts
                self.stimulus[key].update(value)
            else:
                self.stimulus[key] = value
    
    def add_feature(
        self,
        unit_id: str,
        feature_name: str,
        feature_data: Any,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add extracted feature data to a unit.
        
        Args:
            unit_id: Target unit identifier
            feature_name: Name of the feature
            feature_data: Feature data (typically numpy array or dict)
            feature_metadata: Optional metadata about the extraction
        
        Raises:
            KeyError: If unit_id does not exist in session
        """
        if unit_id not in self.units:
            raise KeyError(f"Unit {unit_id} not found in session")
        
        if "features" not in self.units[unit_id]:
            self.units[unit_id]["features"] = {}
        
        self.units[unit_id]["features"][feature_name] = {
            "data": feature_data,
            "metadata": feature_metadata or {},
        }
    
    def mark_step_complete(self, step_name: str) -> None:
        """
        Record that a pipeline step has been completed.
        
        Args:
            step_name: Name of the completed step
        
        Side effects:
            Adds step_name to self.completed_steps
        """
        self.completed_steps.add(step_name)
    
    def set_source_files(
        self,
        cmcr_path: Optional[Union[str, Path]] = None,
        cmtr_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Set paths to original source files.
        
        Args:
            cmcr_path: Path to CMCR file
            cmtr_path: Path to CMTR file
        """
        if cmcr_path:
            self.source_files["cmcr_path"] = Path(cmcr_path) if cmcr_path else None
        if cmtr_path:
            self.source_files["cmtr_path"] = Path(cmtr_path) if cmtr_path else None
    
    # =========================================================================
    # Save/Checkpoint Methods
    # =========================================================================
    
    def save(
        self,
        output_path: Optional[Union[str, Path]] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        """
        Write all accumulated data to HDF5 and transition to SAVED state.
        
        After save(), the session's hdf5_path is set and HDF5-path-based
        operations can be performed.
        
        Args:
            output_path: Path for output file. If None, uses
                         output_dir / f"{dataset_id}.h5"
            overwrite: If True (default), overwrite existing file with warning.
                       If False, raise error if file exists.
        
        Returns:
            Path to the saved HDF5 file
        
        Raises:
            FileExistsError: If file exists and overwrite=False
            IOError: If write fails
        """
        if output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{self.dataset_id}.h5"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for existing file
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"File already exists: {output_path}. Use overwrite=True to replace."
                )
            logger.warning(f"Overwriting existing file: {output_path}")
        
        # Write to HDF5
        self.saved_at = datetime.now(timezone.utc).isoformat()
        self._write_session_to_hdf5(output_path)
        
        # Transition state
        self.save_state = SaveState.SAVED
        self.hdf5_path = output_path
        
        logger.info(f"Session saved to: {output_path}")
        return output_path
    
    def checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        *,
        checkpoint_name: Optional[str] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Save current state to a checkpoint file without ending the session.
        
        Unlike save(), checkpoint() does NOT transition the session to SAVED
        state. The session remains in DEFERRED mode and can continue
        accumulating data.
        
        Args:
            checkpoint_path: Path for checkpoint file
            checkpoint_name: Optional name for the checkpoint (stored in metadata)
            overwrite: If True (default), overwrite existing file with warning.
                       If False, raise error if file exists.
        
        Returns:
            Path to the checkpoint file
        
        Raises:
            FileExistsError: If file exists and overwrite=False
            IOError: If write fails
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for existing file
        if checkpoint_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Checkpoint file already exists: {checkpoint_path}. Use overwrite=True to replace."
                )
            logger.warning(f"Overwriting existing checkpoint: {checkpoint_path}")
        
        # Write checkpoint (without changing state)
        checkpoint_saved_at = datetime.now(timezone.utc).isoformat()
        self._write_session_to_hdf5(
            checkpoint_path,
            checkpoint_name=checkpoint_name,
            checkpoint_saved_at=checkpoint_saved_at,
        )
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
    
    def ensure_saved(
        self,
        default_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Ensure session is in SAVED state, auto-saving if necessary.
        
        This is called internally by functions that require an HDF5 path.
        If session is already SAVED, returns existing path.
        If DEFERRED, performs save() with warning and returns path.
        
        Args:
            default_path: Path to use for auto-save if not already saved
        
        Returns:
            Path to the HDF5 file
        """
        if self.is_saved and self.hdf5_path is not None:
            return self.hdf5_path
        
        # Auto-save with warning
        logger.warning(
            f"Auto-saving deferred session to satisfy HDF5-path requirement. "
            f"Consider calling session.save() explicitly."
        )
        return self.save(output_path=default_path)
    
    # =========================================================================
    # Load/Resume Methods
    # =========================================================================
    
    @classmethod
    def load(cls, checkpoint_path: Union[str, Path]) -> "PipelineSession":
        """
        Resume a session from a previously saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint HDF5 file
        
        Returns:
            PipelineSession with restored state (save_state=SAVED)
        
        Raises:
            FileNotFoundError: If checkpoint_path does not exist
            ValueError: If file is not a valid session checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        return cls._restore_session_from_hdf5(checkpoint_path)
    
    # =========================================================================
    # Internal Serialization Methods
    # =========================================================================
    
    def _write_session_to_hdf5(
        self,
        path: Path,
        checkpoint_name: Optional[str] = None,
        checkpoint_saved_at: Optional[str] = None,
    ) -> None:
        """
        Write session data to HDF5 file.
        
        Args:
            path: Output file path
            checkpoint_name: Optional checkpoint name (for checkpoint() calls)
            checkpoint_saved_at: Timestamp for checkpoint (for checkpoint() calls)
        """
        mode = "w"  # Always overwrite for now
        
        # Handle Windows file locking: try to remove existing file first
        if path.exists():
            if not _safe_remove_file(path):
                raise OSError(f"Cannot overwrite file (locked by another process): {path}")
        
        # Retry logic for file creation (Windows sometimes needs a moment after deletion)
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                with h5py.File(path, mode) as f:
                    self._write_session_content_to_hdf5(f, checkpoint_name, checkpoint_saved_at)
                return  # Success
            except OSError as e:
                last_error = e
                if attempt < max_retries - 1:
                    gc.collect()
                    time.sleep(0.5)
                    logger.warning(f"HDF5 file creation failed, retrying... (attempt {attempt + 1}/{max_retries})")
        
        raise OSError(f"Failed to create HDF5 file after {max_retries} attempts: {last_error}")
    
    def _write_session_content_to_hdf5(
        self,
        f: h5py.File,
        checkpoint_name: Optional[str] = None,
        checkpoint_saved_at: Optional[str] = None,
    ) -> None:
        """Write session content to an already-open HDF5 file."""
        # Write dataset_id attribute
        f.attrs["dataset_id"] = self.dataset_id
        f.attrs["created_at"] = self.created_at or ""
        f.attrs["hdmea_version"] = "0.1.0"  # TODO: get from package
        
        # Write units
        if self.units:
            units_group = f.create_group("units")
            for unit_id, unit_data in self.units.items():
                unit_group = units_group.create_group(unit_id)
                self._write_unit_to_hdf5(unit_group, unit_data)
        
        # Write metadata
        if self.metadata:
            meta_group = f.create_group("metadata")
            self._write_dict_to_hdf5(meta_group, self.metadata)
        
        # Write stimulus
        if self.stimulus:
            stim_group = f.create_group("stimulus")
            self._write_dict_to_hdf5(stim_group, self.stimulus)
        
        # Write source files
        if self.source_files:
            src_group = f.create_group("source_files")
            for key, value in self.source_files.items():
                if value is not None:
                    src_group.attrs[key] = str(value)
        
        # Write pipeline info
        pipeline_group = f.create_group("pipeline")
        pipeline_group.attrs["stage1_completed"] = True
        pipeline_group.attrs["stage1_timestamp"] = self.saved_at or checkpoint_saved_at or ""
        
        # Write session info
        session_info = pipeline_group.create_group("session_info")
        session_info.attrs["saved_at"] = self.saved_at or checkpoint_saved_at or ""
        if checkpoint_name:
            session_info.attrs["checkpoint_name"] = checkpoint_name
        
        # Write completed_steps as dataset
        if self.completed_steps:
            steps_list = list(self.completed_steps)
            dt = h5py.special_dtype(vlen=str)
            session_info.create_dataset(
                "completed_steps",
                data=steps_list,
                dtype=dt,
            )
        
        # Write warnings as dataset
        if self.warnings:
            dt = h5py.special_dtype(vlen=str)
            session_info.create_dataset(
                "warnings",
                data=self.warnings,
                dtype=dt,
            )
    
    def _write_unit_to_hdf5(self, group: h5py.Group, unit_data: Dict[str, Any]) -> None:
        """Write unit data to HDF5 group."""
        for key, value in unit_data.items():
            if key == "features":
                # Handle features as subgroup
                if value:
                    features_group = group.create_group("features")
                    for feat_name, feat_data in value.items():
                        feat_group = features_group.create_group(feat_name)
                        if isinstance(feat_data, dict):
                            self._write_dict_to_hdf5(feat_group, feat_data)
                        elif isinstance(feat_data, np.ndarray):
                            feat_group.create_dataset("data", data=feat_data)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_hdf5(subgroup, value)
            elif value is not None:
                # Store as attribute for simple types
                try:
                    group.attrs[key] = value
                except TypeError:
                    # Can't store this type as attr, skip
                    pass
    
    def _write_dict_to_hdf5(self, group: h5py.Group, data: Dict[str, Any]) -> None:
        """Recursively write dict to HDF5 group."""
        for key, value in data.items():
            # HDF5 requires string keys - convert integers to strings
            str_key = str(key) if not isinstance(key, str) else key
            
            if isinstance(value, np.ndarray):
                group.create_dataset(str_key, data=value)
            elif isinstance(value, dict):
                subgroup = group.create_group(str_key)
                self._write_dict_to_hdf5(subgroup, value)
            elif isinstance(value, (list, tuple)):
                # Convert list/tuple to numpy array if possible
                try:
                    arr = np.array(value)
                    group.create_dataset(str_key, data=arr)
                except (ValueError, TypeError):
                    # Store as string if can't convert
                    try:
                        group.attrs[str_key] = str(value)
                    except Exception:
                        pass
            elif value is not None:
                try:
                    group.attrs[str_key] = value
                except TypeError:
                    # Can't store this type, try string conversion
                    try:
                        group.attrs[str_key] = str(value)
                    except Exception:
                        pass
    
    @classmethod
    def _restore_session_from_hdf5(cls, path: Path) -> "PipelineSession":
        """
        Restore session from HDF5 file.
        
        Args:
            path: Path to HDF5 file
        
        Returns:
            Restored PipelineSession
        """
        with h5py.File(path, "r") as f:
            dataset_id = f.attrs.get("dataset_id", "")
            if not dataset_id:
                raise ValueError(f"Invalid session file: missing dataset_id in {path}")
            
            session = cls(
                dataset_id=dataset_id,
                save_state=SaveState.SAVED,
                hdf5_path=path,
            )
            
            session.created_at = f.attrs.get("created_at", None)
            
            # Restore units
            if "units" in f:
                for unit_id in f["units"]:
                    session.units[unit_id] = cls._read_unit_from_hdf5(f["units"][unit_id])
            
            # Restore metadata
            if "metadata" in f:
                session.metadata = cls._read_dict_from_hdf5(f["metadata"])
            
            # Restore stimulus
            if "stimulus" in f:
                session.stimulus = cls._read_dict_from_hdf5(f["stimulus"])
            
            # Restore source files
            if "source_files" in f:
                for key in f["source_files"].attrs:
                    session.source_files[key] = Path(f["source_files"].attrs[key])
            
            # Restore pipeline info
            if "pipeline" in f and "session_info" in f["pipeline"]:
                session_info = f["pipeline"]["session_info"]
                session.saved_at = session_info.attrs.get("saved_at", None)
                
                if "completed_steps" in session_info:
                    session.completed_steps = set(session_info["completed_steps"][:])
                
                if "warnings" in session_info:
                    session.warnings = list(session_info["warnings"][:])
            
            return session
    
    @classmethod
    def _read_unit_from_hdf5(cls, group: h5py.Group) -> Dict[str, Any]:
        """Read unit data from HDF5 group."""
        unit_data = {}
        
        # Read attributes
        for key in group.attrs:
            unit_data[key] = group.attrs[key]
        
        # Read datasets and subgroups
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                unit_data[key] = item[:]
            elif isinstance(item, h5py.Group):
                if key == "features":
                    unit_data["features"] = {}
                    for feat_name in item:
                        unit_data["features"][feat_name] = cls._read_dict_from_hdf5(item[feat_name])
                else:
                    unit_data[key] = cls._read_dict_from_hdf5(item)
        
        return unit_data
    
    @classmethod
    def _read_dict_from_hdf5(cls, group: h5py.Group) -> Dict[str, Any]:
        """Recursively read dict from HDF5 group."""
        result = {}
        
        # Read attributes
        for key in group.attrs:
            result[key] = group.attrs[key]
        
        # Read datasets and subgroups
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[:]
            elif isinstance(item, h5py.Group):
                result[key] = cls._read_dict_from_hdf5(item)
        
        return result


def create_session(
    dataset_id: Optional[str] = None,
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "artifacts",
) -> PipelineSession:
    """
    Convenience function to create a new PipelineSession.
    
    If dataset_id is not provided, derives it from file paths.
    
    Args:
        dataset_id: Unique recording identifier
        cmcr_path: Optional path to derive dataset_id from
        cmtr_path: Optional path to derive dataset_id from
        output_dir: Default output directory
    
    Returns:
        New PipelineSession in DEFERRED state
    
    Example:
        >>> session = create_session(cmcr_path="path/to/recording.cmcr")
        >>> session = load_recording(..., session=session)
        >>> session.save()
    """
    if dataset_id is None:
        # Derive from file paths
        from hdmea.utils.validation import derive_dataset_id
        dataset_id = derive_dataset_id(cmcr_path, cmtr_path)
    
    session = PipelineSession(
        dataset_id=dataset_id,
        output_dir=Path(output_dir),
    )
    
    if cmcr_path or cmtr_path:
        session.set_source_files(cmcr_path=cmcr_path, cmtr_path=cmtr_path)
    
    return session

