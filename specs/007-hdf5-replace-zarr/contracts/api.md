# API Contract: hdf5_store.py

**Module**: `hdmea.io.hdf5_store`  
**Replaces**: `hdmea.io.zarr_store`

---

## Core I/O Functions

### `create_recording_hdf5`

Creates a new HDF5 archive for a recording.

```python
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
    
    Side Effects:
        - Creates file with groups: /units, /stimulus, /metadata
        - Sets root attributes: dataset_id, hdmea_pipeline_version, 
          created_at, updated_at, stage1_completed, stage1_params_hash,
          features_extracted
    """
```

**Contract**:
- MUST create file with `.h5` extension (warn if different)
- MUST create groups: `/units`, `/stimulus`, `/metadata`
- MUST set all required root attributes
- MUST raise `FileExistsError` if file exists and `overwrite=False`
- MUST delete existing file before creating if `overwrite=True`
- MUST return open file handle (not closed)

---

### `open_recording_hdf5`

Opens an existing HDF5 archive.

```python
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
```

**Contract**:
- MUST raise `FileNotFoundError` if file doesn't exist
- MUST enforce single-writer model for write modes
- MUST return lazy-loading file handle

---

## Data Writing Functions

### `write_units`

Writes unit data (spike times, waveforms, metadata).

```python
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
    
    Side Effects:
        - Creates /units/{unit_id}/ groups
        - Creates spike_times, waveform datasets
        - Sets unit attributes (row, col, global_id, spike_count)
        - Creates empty /features subgroup
    """
```

**Contract**:
- MUST create unit group for each entry
- MUST store spike_times as uint64 dataset
- MUST store waveform as float32 dataset
- MUST set all required unit attributes
- MUST create empty features subgroup

---

### `write_stimulus`

Writes stimulus data (light reference, frame times, section times).

```python
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
    
    Side Effects:
        - Creates /stimulus/light_reference/{rate_name} datasets
        - Creates /stimulus/frame_time/{movie_name} datasets
        - Creates /stimulus/section_time/{movie_name} datasets
    """
```

**Contract**:
- MUST create subgroups with `overwrite=True` semantics
- MUST store light_reference as float32
- MUST store frame_times as uint64
- MUST store section_times as uint64 with shape (n, 2)

---

### `write_metadata`

Writes recording metadata.

```python
def write_metadata(
    root: h5py.File,
    metadata: Dict[str, Any],
) -> None:
    """
    Write recording metadata to HDF5.
    
    Args:
        root: Open h5py.File handle
        metadata: Metadata dictionary (may contain nested dicts)
    
    Side Effects:
        - Writes to /metadata group
        - Scalars stored as single-element datasets
        - Nested dicts become subgroups (e.g., sys_meta)
    """
```

**Contract**:
- MUST handle nested dictionaries as subgroups
- MUST store scalars (int, float) as single-element arrays
- MUST store strings as variable-length UTF-8

---

### `write_feature_to_unit`

Writes extracted feature data for a unit.

```python
def write_feature_to_unit(
    root: h5py.File,
    unit_id: str,
    feature_name: str,
    feature_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Write extracted features for a unit.
    
    Args:
        root: Open h5py.File handle
        unit_id: Unit identifier (e.g., "unit_000")
        feature_name: Feature name (e.g., "step_up_5s_5i_3x")
        feature_data: Dictionary of feature values (scalars or arrays)
        metadata: Feature metadata (version, params_hash, extracted_at)
    
    Side Effects:
        - Creates /units/{unit_id}/features/{feature_name} group
        - Stores arrays as datasets, scalars as attributes
        - Updates root features_extracted list
    """
```

**Contract**:
- MUST create feature group under unit
- MUST store numpy arrays as datasets
- MUST store scalars as group attributes
- MUST handle nested dicts as subgroups
- MUST update root `features_extracted` attribute

---

## Utility Functions

### `mark_stage1_complete`

```python
def mark_stage1_complete(root: h5py.File) -> None:
    """Mark Stage 1 as complete in HDF5 metadata."""
```

**Contract**:
- MUST set `root.attrs["stage1_completed"] = True`
- MUST update `root.attrs["updated_at"]` to current timestamp

---

### `get_stage1_status`

```python
def get_stage1_status(root: h5py.File) -> Dict[str, Any]:
    """
    Get Stage 1 completion status.
    
    Returns:
        Dictionary with: completed, params_hash, created_at, updated_at
    """
```

---

### `list_units`

```python
def list_units(root: h5py.File) -> List[str]:
    """List all unit IDs in the HDF5 file."""
```

**Contract**:
- MUST return list of unit group names
- MUST return empty list if no units

---

### `list_features`

```python
def list_features(root: h5py.File, unit_id: str) -> List[str]:
    """List all features extracted for a unit."""
```

**Contract**:
- MUST return list of feature group names
- MUST return empty list if no features or unit doesn't exist

---

### `write_source_files`

```python
def write_source_files(
    root: h5py.File,
    cmcr_path: Optional[Path],
    cmtr_path: Optional[Path],
) -> None:
    """Write source file information to HDF5 metadata."""
```

**Contract**:
- MUST store paths as strings in `root.attrs["source_files"]`
- MUST include `cmcr_exists` and `cmtr_exists` flags

---

## Error Handling

### Single-Writer Violations

When a file is already open for writing:

```python
raise OSError(
    f"File is already open for writing: {path}. "
    "HDF5 supports single-writer access only. "
    "Close the file in other processes before writing."
)
```

### Corrupt File Detection

When a file cannot be read:

```python
raise OSError(
    f"Cannot read HDF5 file: {path}. "
    "The file may be corrupted or incomplete."
)
```

---

## Type Aliases

```python
from pathlib import Path
from typing import Any, Dict, List, Optional
import h5py
import numpy as np
```

---

## Migration Compatibility

All functions maintain signature compatibility with `zarr_store.py`:

| zarr_store | hdf5_store | Notes |
|------------|------------|-------|
| Returns `zarr.Group` | Returns `h5py.File` | Both support dict-like access |
| No context manager needed | Context manager recommended | Explicit close required |
| `group.attrs` | `group.attrs` | Identical API |
| `group.create_dataset()` | `group.create_dataset()` | Nearly identical |

