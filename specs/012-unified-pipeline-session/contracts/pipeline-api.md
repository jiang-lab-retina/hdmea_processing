# Pipeline API Contract

**Branch**: `012-unified-pipeline-session` | **Date**: 2025-12-28

## Overview

This document defines the public API contracts for the unified pipeline session feature.

---

## Core Functions

### `create_session`

Creates a new PipelineSession in DEFERRED state.

```python
def create_session(
    dataset_id: Optional[str] = None,
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "artifacts",
) -> PipelineSession:
    """
    Create a new PipelineSession.
    
    Args:
        dataset_id: Unique recording identifier. If None, derived from file paths.
        cmcr_path: Optional CMCR file path (used to derive dataset_id if not provided)
        cmtr_path: Optional CMTR file path (used to derive dataset_id if not provided)
        output_dir: Default output directory for save operations
    
    Returns:
        New PipelineSession in DEFERRED state
    
    Raises:
        ValueError: If dataset_id cannot be determined
    
    Example:
        >>> session = create_session(dataset_id="2024.08.08-10.40.20-Rec")
        >>> session = create_session(cmcr_path="path/to/file.cmcr")
    """
```

---

### `load_session_from_hdf5`

Universal HDF5 loader - loads existing HDF5 into a session.

```python
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
        PipelineSession with loaded data (save_state=DEFERRED)
    
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
```

---

## PipelineSession Methods

### `save`

Write session to HDF5 file.

```python
def save(
    self,
    output_path: Optional[Union[str, Path]] = None,
    *,
    overwrite: bool = False,  # NOTE: Default changed from True to False
) -> Path:
    """
    Write all accumulated data to HDF5.
    
    Args:
        output_path: Output file path. If None, uses output_dir/{dataset_id}.h5
        overwrite: If True, overwrite existing file with warning.
                   If False (default), raise FileExistsError if file exists.
    
    Returns:
        Path to saved HDF5 file
    
    Raises:
        FileExistsError: If file exists and overwrite=False
        IOError: If write fails
    
    Example:
        >>> path = session.save()  # Uses default path
        >>> path = session.save("output/custom.h5")  # Explicit path
        >>> path = session.save("existing.h5", overwrite=True)  # Overwrite
    """
```

### `checkpoint`

Save intermediate state without ending session.

```python
def checkpoint(
    self,
    checkpoint_path: Union[str, Path],
    *,
    checkpoint_name: Optional[str] = None,
    overwrite: bool = True,
) -> Path:
    """
    Save current state to a checkpoint file.
    
    Unlike save(), checkpoint() does NOT transition the session to SAVED
    state. The session remains in DEFERRED mode and can continue.
    
    Args:
        checkpoint_path: Path for checkpoint file
        checkpoint_name: Optional name for the checkpoint
        overwrite: If True (default), overwrite existing checkpoint.
    
    Returns:
        Path to checkpoint file
    
    Raises:
        FileExistsError: If file exists and overwrite=False
    """
```

### `mark_step_complete`

Record pipeline step completion.

```python
def mark_step_complete(self, step_name: str) -> None:
    """
    Record that a pipeline step has been completed.
    
    Args:
        step_name: Name of the completed step
        
    Side effects:
        Adds step_name to self.completed_steps
    """
```

---

## Pipeline Step Pattern

All pipeline step functions MUST follow this signature pattern:

```python
def pipeline_step(
    # Required positional parameters
    param1: type,
    param2: type,
    *,
    # Keyword-only parameters
    session: Optional[PipelineSession] = None,
    option1: type = default,
) -> Union[ResultType, PipelineSession]:
    """
    Perform pipeline step.
    
    Args:
        param1: Description
        param2: Description
        session: If provided, operates in deferred mode and returns session.
                 If None, operates in immediate mode and returns result.
        option1: Optional parameter
    
    Returns:
        If session provided: Updated PipelineSession with step complete
        If session is None: Direct result
    """
    if session is not None:
        # Deferred mode
        for unit_id in tqdm(session.units, desc="Processing"):
            # Process unit
            pass
        session.mark_step_complete("step_name")
        return session
    else:
        # Immediate mode
        return result
```

---

## Step Wrappers

Each step wrapper in `Projects/unified_pipeline/steps/` follows this pattern:

```python
# steps/{step_name}.py

import logging
from tqdm import tqdm
from colorama import Fore, Style

from hdmea.pipeline import PipelineSession

logger = logging.getLogger(__name__)

def step_name(
    *,
    param: type,
    session: PipelineSession,
) -> PipelineSession:
    """
    Pipeline step: {description}
    
    Args:
        param: Step-specific parameter
        session: Pipeline session (required)
    
    Returns:
        Updated session with step complete
    """
    logger.info(f"Starting {step_name}...")
    
    try:
        # Call existing implementation
        from some_module import existing_function
        
        for unit_id in tqdm(session.units, desc=f"{step_name}"):
            result = existing_function(session.units[unit_id], param)
            session.units[unit_id]["feature"] = result
        
        session.mark_step_complete("step_name")
        logger.info(f"{step_name} complete")
        
    except ExternalServiceError as e:
        # Handle optional external dependency failure
        logger.warning(f"{Fore.RED}{step_name} failed: {e}. Skipping...{Style.RESET_ALL}")
        session.mark_step_complete("step_name:skipped")
    
    return session
```

---

## Error Handling

### Standard Exceptions

| Exception | When Raised |
|-----------|-------------|
| `FileNotFoundError` | Source file not found |
| `FileExistsError` | Output file exists and overwrite=False |
| `ValueError` | Invalid parameters or data |
| `KeyError` | Unit not found in session |
| `SessionError` | General session operation failure |
| `CheckpointError` | Checkpoint read/write failure |

### External Service Failures

For optional external services (e.g., Google Sheet):
1. Log RED warning
2. Mark step as `"{step_name}:skipped"` in completed_steps
3. Return session without failing

```python
# Check if step was skipped
if "add_gsheet_metadata:skipped" in session.completed_steps:
    # Handle missing gsheet data
```

---

## Progress Reporting

### Logging Levels

| Level | Usage |
|-------|-------|
| `INFO` | Step start/completion, summary statistics |
| `WARNING` | Skipped steps, recoverable issues (use RED for external failures) |
| `ERROR` | Failures that prevent step completion |
| `DEBUG` | Detailed per-unit processing info |

### Progress Bars

Use `tqdm` for unit iteration:

```python
from tqdm import tqdm

for unit_id in tqdm(session.units, desc="Processing units"):
    # ...
```

