# API Contract: PipelineSession

**Module**: `hdmea.pipeline.session`  
**Version**: 1.0.0

## Class: PipelineSession

### Constructor

```python
def __init__(
    self,
    dataset_id: str,
    output_dir: Union[str, Path] = "artifacts",
) -> None:
    """
    Create a new pipeline session for deferred HDF5 saving.
    
    Args:
        dataset_id: Unique identifier for the recording
        output_dir: Default directory for save operations
    
    Raises:
        ValueError: If dataset_id is empty
    """
```

### Class Methods

#### load

```python
@classmethod
def load(
    cls,
    checkpoint_path: Union[str, Path],
) -> "PipelineSession":
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
```

### Instance Methods

#### save

```python
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
    
    Side effects:
        - Writes HDF5 file to disk
        - Sets self.hdf5_path
        - Sets self.save_state = SAVED
        - Logs warning if overwriting existing file
    """
```

#### checkpoint

```python
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
    
    Side effects:
        - Writes HDF5 file to disk
        - Does NOT change save_state
        - Does NOT set hdf5_path
        - Logs warning if overwriting existing file
    """
```

#### add_units

```python
def add_units(
    self,
    units_data: Dict[str, Dict[str, Any]],
) -> None:
    """
    Add or update unit data in the session.
    
    Args:
        units_data: Dict mapping unit_id to unit data dict
    
    Side effects:
        - Updates self.units (merge, not replace)
    """
```

#### add_metadata

```python
def add_metadata(
    self,
    metadata: Dict[str, Any],
) -> None:
    """
    Add or update metadata in the session.
    
    Args:
        metadata: Dict of metadata key-value pairs
    
    Side effects:
        - Updates self.metadata (merge, not replace)
    """
```

#### add_stimulus

```python
def add_stimulus(
    self,
    stimulus_data: Dict[str, Any],
) -> None:
    """
    Add or update stimulus data in the session.
    
    Args:
        stimulus_data: Dict of stimulus data
    
    Side effects:
        - Updates self.stimulus (merge, not replace)
    """
```

#### add_feature

```python
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
```

#### mark_step_complete

```python
def mark_step_complete(
    self,
    step_name: str,
) -> None:
    """
    Record that a pipeline step has been completed.
    
    Args:
        step_name: Name of the completed step
    
    Side effects:
        - Adds step_name to self.completed_steps
    """
```

#### ensure_saved

```python
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
    
    Side effects:
        - May trigger save() if in DEFERRED state
        - Logs warning if auto-save occurs
    """
```

### Properties

```python
@property
def is_saved(self) -> bool:
    """True if session has been saved to disk."""

@property
def is_deferred(self) -> bool:
    """True if session is in deferred (in-memory only) mode."""

@property
def unit_count(self) -> int:
    """Number of units in the session."""

@property
def memory_estimate_gb(self) -> float:
    """Estimated memory usage in GB (approximate)."""
```

