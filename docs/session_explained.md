# PipelineSession: Deferred Save Mode Guide

**Last Updated**: 2025-12-20  
**Pipeline Version**: 0.3.0

## Overview

The `PipelineSession` is an in-memory container that accumulates data across multiple pipeline steps, enabling **deferred saving** to HDF5. Instead of writing intermediate files after each step, data stays in memory until you explicitly call `save()`.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Reduced I/O** | Single HDF5 write instead of multiple intermediate files |
| **Faster Pipelines** | No disk I/O between steps |
| **Checkpointing** | Save intermediate state and resume later |
| **Flexibility** | Inspect and modify data between steps |
| **Feature Extraction** | Extract features directly from session data via adapters |
| **Integrated STA** | Compute eimage_sta during loading with `load_recording_with_eimage_sta` |

---

## Quick Start

### Basic Pipeline (load_recording)

```python
from hdmea.pipeline import create_session, load_recording, extract_features
from hdmea.io import add_section_time, section_spike_times
from hdmea.features import compute_sta

# 1. Create a session
session = create_session(dataset_id="2025.04.10-11.12.57-Rec")

# 2. Run pipeline steps (data accumulates in memory)
session = load_recording(
    cmcr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr",
    cmtr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr",
    session=session,
)

session = add_section_time(
    playlist_name="play_optimization_set6_ipRGC_manual",
    session=session,
)

session = section_spike_times(
    pad_margin=(0.0, 0.0),
    session=session,
)

session = compute_sta(
    cover_range=(-60, 0),
    session=session,
)

session = extract_features(
    features=["frif"],
    session=session,
)

# 3. Save once at the end
hdf5_path = session.save()
print(f"Saved to: {hdf5_path}")
```

### Integrated EImage STA (load_recording_with_eimage_sta)

For computing eimage_sta during loading (more memory-efficient for large datasets):

```python
from hdmea.pipeline import create_session, load_recording_with_eimage_sta, extract_features
from hdmea.io import add_section_time, section_spike_times
from hdmea.features import compute_sta

# 1. Create a session
session = create_session(dataset_id="2025.04.10-11.12.57-Rec")

# 2. Load recording AND compute eimage_sta in one pass
session = load_recording_with_eimage_sta(
    cmcr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr",
    cmtr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr",
    duration_s=120.0,       # Analyze first 120 seconds
    spike_limit=10000,      # Max spikes per unit for STA
    window_range=(-10, 40), # Samples before/after spike
    session=session,
)

# Continue with rest of pipeline
session = add_section_time(playlist_name="play_optimization_set6_ipRGC_manual", session=session)
session = section_spike_times(pad_margin=(0.0, 0.0), session=session)
session = compute_sta(cover_range=(-60, 0), session=session)
session = extract_features(features=["frif"], session=session)

# 3. Save once at the end
hdf5_path = session.save()
print(f"Saved to: {hdf5_path}")
```

---

## Session Lifecycle

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ create_     │     │  Pipeline Steps  │     │   save()    │
│ session()   │────▶│  (accumulate)    │────▶│             │────▶ HDF5
└─────────────┘     └──────────────────┘     └─────────────┘
      │                      │                      │
      │                      ▼                      │
      │               ┌────────────┐                │
      │               │checkpoint()│────▶ Checkpoint HDF5
      │               └────────────┘      (resumable)
      │                      │
      ▼                      ▼
   DEFERRED              DEFERRED               SAVED
   (in memory)           (in memory)           (on disk)
```

### Session States

| State | `session.save_state` | Description |
|-------|---------------------|-------------|
| **DEFERRED** | `SaveState.DEFERRED` | Data in memory, not yet saved |
| **SAVED** | `SaveState.SAVED` | Data written to HDF5 |

---

## Session Data Structure

The session mirrors the HDF5 structure in memory:

```python
session = PipelineSession(dataset_id="recording_001")

# After load_recording():
session.units = {
    "unit_001": {
        "spike_times": np.array([...]),      # Sample indices
        "firing_rate_10hz": np.array([...]), # Binned rates
        "waveform": np.array([...]),         # Average waveform
    },
    "unit_002": {...},
    ...
}

session.stimulus = {
    "light_reference": {
        "raw_ch1": np.array([...]),  # Light intensity
        "raw_ch2": np.array([...]),  # Frame sync
    },
    "frame_times": {
        "frame_timestamps": np.array([...]),  # Frame→sample mapping
    },
}

session.metadata = {
    "acquisition_rate": 20000.0,
    "sample_interval": 5e-5,
    "recording_duration_s": 1189.7,
    ...
}

session.source_files = {
    "cmcr_path": Path("O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr"),
    "cmtr_path": Path("O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr"),
}

session.completed_steps = {"load_recording"}
session.warnings = []
```

### After Additional Steps

```python
# After add_section_time():
session.stimulus["section_time"] = {
    "movie_name": np.array([[start, end], ...]),  # Trial boundaries
}
session.stimulus["light_template"] = {
    "movie_name": np.array([...]),  # Averaged light trace
}
session.completed_steps = {"load_recording", "add_section_time"}

# After section_spike_times():
session.units["unit_001"]["spike_times_sectioned"] = {
    "movie_name": {
        "full_spike_times": np.array([...]),
        "trials_spike_times": {0: np.array([...]), 1: np.array([...])},
    }
}
session.completed_steps = {"load_recording", "add_section_time", "section_spike_times"}

# After load_recording_with_eimage_sta() (if used instead of load_recording):
session.units["unit_001"]["features"] = {
    "eimage_sta": {
        "data": np.array([...]),  # STA image (frames x height x width)
        "metadata": {
            "n_spikes_used": 5000,
            "n_spikes_excluded": 123,
            "cutoff_hz": 100.0,
            "window_range": [-10, 40],
        },
    }
}
session.completed_steps = {"load_recording_with_eimage_sta"}

# After extract_features():
session.units["unit_001"]["features"] = {
    "frif": {
        "data": {...},  # Feature data dict
        "metadata": {"extractor_version": "1.0.0"},
    },
    # eimage_sta preserved if present from load_recording_with_eimage_sta
}
session.completed_steps = {"load_recording", "add_section_time", "section_spike_times", "extract_features"}
```

---

## Supported Functions

All major pipeline functions support deferred saving via the `session` parameter:

| Function | Module | Description |
|----------|--------|-------------|
| `load_recording` | `hdmea.pipeline` | Load spike data from CMCR/CMTR files |
| `load_recording_with_eimage_sta` | `hdmea.pipeline` | Load + compute eimage_sta in one pass |
| `extract_features` | `hdmea.pipeline` | Extract features (FRIF, etc.) from session data |
| `add_section_time` | `hdmea.io` | Add stimulus section times from playlist |
| `section_spike_times` | `hdmea.io` | Section spike times by stimulus trials |
| `compute_sta` | `hdmea.features` | Compute spike-triggered average |

### Feature Extraction with DictAdapter

Feature extractors (like FRIF) are designed to work with HDF5 groups. In session mode, the pipeline uses `DictAdapter` and `ArrayAdapter` classes to provide an HDF5-like interface to Python dictionaries:

```python
# The pipeline internally wraps session data for extractors:
from hdmea.features.base import DictAdapter

unit_adapter = DictAdapter(session.units["unit_001"])
stimulus_adapter = DictAdapter(session.stimulus)
metadata_adapter = DictAdapter(session.metadata)

# Extractors can use HDF5-like access patterns:
spike_times = unit_adapter["spike_times"][:]  # Works like HDF5 dataset
```

This allows existing feature extractors to work with session data without modification.

---

## Core Operations

### Creating a Session

```python
from hdmea.pipeline import create_session, PipelineSession

# Method 1: Using helper function (recommended)
session = create_session(
    dataset_id="2025.04.10-11.12.57-Rec",
    output_dir="artifacts",  # Default save location
)

# Method 2: Direct instantiation
session = PipelineSession(
    dataset_id="2025.04.10-11.12.57-Rec",
    output_dir=Path("artifacts"),
)
```

### Saving a Session

```python
# Save to default location (artifacts/{dataset_id}.h5)
hdf5_path = session.save()

# Save to custom location
hdf5_path = session.save(output_path="custom/path/recording.h5")

# Overwrite existing file (default: True)
hdf5_path = session.save(overwrite=True)

# Raise error if file exists
hdf5_path = session.save(overwrite=False)  # Raises SessionError if exists
```

### Checkpointing

```python
# Save checkpoint (session remains in DEFERRED state, can continue)
session.checkpoint("artifacts/checkpoint_after_load.h5")

# Continue processing after checkpoint
session = add_section_time(..., session=session)

# Save final result
session.save()
```

### Resuming from Checkpoint

```python
from hdmea.pipeline import PipelineSession

# Load session from checkpoint
session = PipelineSession.load("artifacts/checkpoint_after_load.h5")

# Check completed steps
print(session.completed_steps)  # {'load_recording'}

# Continue from where you left off
session = add_section_time(..., session=session)
session = section_spike_times(..., session=session)
session.save()
```

---

## Session Properties and Methods

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session.is_saved` | `bool` | True if saved to disk |
| `session.is_deferred` | `bool` | True if still in memory |
| `session.unit_count` | `int` | Number of units loaded |
| `session.memory_estimate_gb` | `float` | Estimated memory usage in GB |
| `session.hdf5_path` | `Path` | Path to HDF5 (after save) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_units(units_data)` | `None` | Add/update unit data |
| `add_metadata(metadata)` | `None` | Add/update metadata |
| `add_stimulus(stimulus_data)` | `None` | Add/update stimulus data |
| `add_feature(unit_id, name, data, metadata)` | `None` | Add feature to a unit |
| `mark_step_complete(step_name)` | `None` | Mark a pipeline step as done |
| `save(output_path, overwrite)` | `Path` | Save session to HDF5 |
| `checkpoint(path, overwrite)` | `Path` | Save checkpoint (continue after) |
| `PipelineSession.load(path)` | `PipelineSession` | Load from checkpoint |

---

## Inspecting Session Data

```python
# Check what's loaded
print(f"Units: {session.unit_count}")
print(f"Memory: {session.memory_estimate_gb:.2f} GB")
print(f"Completed steps: {session.completed_steps}")
print(f"Warnings: {session.warnings}")

# Access unit data
unit_001 = session.units["unit_001"]
spike_times = unit_001["spike_times"]
print(f"Unit 001 has {len(spike_times)} spikes")

# Access stimulus data
section_time = session.stimulus.get("section_time", {})
for movie_name, trials in section_time.items():
    print(f"Movie '{movie_name}': {len(trials)} trials")

# Access metadata
acq_rate = session.metadata.get("acquisition_rate", 20000.0)
print(f"Acquisition rate: {acq_rate} Hz")
```

---

## Adding a New Module to the Pipeline

When creating a new pipeline function that supports deferred saving, follow this pattern:

### Step 1: Add Session Parameter

```python
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from hdmea.pipeline.session import PipelineSession

def my_new_function(
    hdf5_path: Optional[Union[str, Path]] = None,  # Required if session=None
    *,
    # ... other parameters ...
    session: Optional["PipelineSession"] = None,   # Optional session
) -> Union[MyResult, "PipelineSession"]:           # Return type changes based on mode
```

### Step 2: Implement Dual-Mode Logic

```python
def my_new_function(
    hdf5_path: Optional[Union[str, Path]] = None,
    *,
    param1: str = "default",
    param2: float = 1.0,
    force: bool = False,
    session: Optional["PipelineSession"] = None,
) -> Union[MyResult, "PipelineSession"]:
    """
    My new pipeline function.
    
    Supports deferred saving via the optional `session` parameter.
    
    Args:
        hdf5_path: Path to HDF5 file. Required if session=None.
        param1: Some parameter.
        param2: Another parameter.
        force: If True, overwrite existing data.
        session: Optional PipelineSession for deferred saving.
    
    Returns:
        MyResult if session is None (immediate save mode).
        PipelineSession if session is provided (deferred save mode).
    """
    warnings_list = []
    
    # =========================================================================
    # Session-based mode: read from session, write to session
    # =========================================================================
    if session is not None:
        # Validate prerequisites
        if "required_step" not in session.completed_steps:
            session.warnings.append("Required step not completed")
            return session
        
        # Read data from session
        units = session.units
        stimulus = session.stimulus
        metadata = session.metadata
        
        # Process data
        results = process_data(units, stimulus, metadata, param1, param2)
        
        # Write results back to session
        for unit_id, result in results.items():
            session.add_feature(unit_id, "my_feature", result["data"], result["metadata"])
        
        # Or add to stimulus/metadata
        session.stimulus["my_new_data"] = computed_data
        session.metadata["my_param"] = param1
        
        # Mark step complete
        session.warnings.extend(warnings_list)
        session.mark_step_complete("my_new_function")
        
        logger.info(f"my_new_function complete (deferred)")
        return session
    
    # =========================================================================
    # Immediate save mode: read from HDF5, write to HDF5
    # =========================================================================
    if hdf5_path is None:
        raise ValueError("hdf5_path is required when session is not provided")
    
    hdf5_path = Path(hdf5_path)
    
    # Open HDF5 and process
    with h5py.File(hdf5_path, "r+") as f:
        # Read data
        units = f["units"]
        # ... process ...
        # Write results
        f.create_dataset("my_data", data=results)
    
    return MyResult(
        hdf5_path=hdf5_path,
        # ... other fields ...
    )
```

### Step 3: Register Dependencies

In your function, check for prerequisite steps:

```python
# Common prerequisite patterns
STEP_PREREQUISITES = {
    "extract_features": ["load_recording"],
    "add_section_time": ["load_recording"],
    "section_spike_times": ["add_section_time"],
    "compute_sta": ["section_spike_times"],
    "my_new_function": ["load_recording", "add_section_time"],  # Your function
}

if session is not None:
    for prereq in STEP_PREREQUISITES.get("my_new_function", []):
        if prereq not in session.completed_steps:
            raise ConfigurationError(f"Required step '{prereq}' not completed")
```

### Step 4: Update Exports (Optional)

If your function should be importable from `hdmea.pipeline`:

```python
# In src/hdmea/pipeline/__init__.py
from hdmea.my_module import my_new_function

__all__ = [
    # ... existing exports ...
    "my_new_function",
]
```

---

## Complete Example: Custom Analysis Function

```python
# src/hdmea/analysis/custom.py

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import h5py
import numpy as np

if TYPE_CHECKING:
    from hdmea.pipeline.session import PipelineSession

logger = logging.getLogger(__name__)


def compute_burst_statistics(
    hdf5_path: Optional[Union[str, Path]] = None,
    *,
    burst_threshold_ms: float = 10.0,
    min_spikes_per_burst: int = 3,
    force: bool = False,
    session: Optional["PipelineSession"] = None,
) -> Union[Dict[str, Any], "PipelineSession"]:
    """
    Compute burst statistics for all units.
    
    Args:
        hdf5_path: Path to HDF5 file. Required if session=None.
        burst_threshold_ms: Maximum inter-spike interval for burst detection.
        min_spikes_per_burst: Minimum spikes to qualify as a burst.
        force: If True, overwrite existing burst statistics.
        session: Optional PipelineSession for deferred saving.
    
    Returns:
        Dict with statistics if session is None.
        PipelineSession if session is provided.
    """
    warnings_list: List[str] = []
    
    # =========================================================================
    # Session-based mode
    # =========================================================================
    if session is not None:
        if "load_recording" not in session.completed_steps:
            session.warnings.append("load_recording not completed")
            return session
        
        acquisition_rate = session.metadata.get("acquisition_rate", 20000.0)
        burst_threshold_samples = int(burst_threshold_ms * acquisition_rate / 1000)
        
        results = {}
        for unit_id, unit_data in session.units.items():
            spike_times = unit_data.get("spike_times")
            if spike_times is None or len(spike_times) < min_spikes_per_burst:
                continue
            
            # Detect bursts
            isi = np.diff(spike_times)
            burst_mask = isi < burst_threshold_samples
            
            # Count bursts and spikes in bursts
            burst_count = np.sum(np.diff(burst_mask.astype(int)) == 1)
            spikes_in_bursts = np.sum(burst_mask)
            
            burst_stats = {
                "burst_count": int(burst_count),
                "spikes_in_bursts": int(spikes_in_bursts),
                "burst_rate": float(burst_count / (len(spike_times) / acquisition_rate)),
                "burst_threshold_ms": burst_threshold_ms,
            }
            
            # Store in session
            session.add_feature(
                unit_id, 
                "burst_statistics", 
                burst_stats,
                {"version": "1.0.0"},
            )
            results[unit_id] = burst_stats
        
        session.warnings.extend(warnings_list)
        session.mark_step_complete("compute_burst_statistics")
        
        logger.info(f"Computed burst statistics for {len(results)} units (deferred)")
        return session
    
    # =========================================================================
    # Immediate save mode
    # =========================================================================
    if hdf5_path is None:
        raise ValueError("hdf5_path is required when session is not provided")
    
    hdf5_path = Path(hdf5_path)
    
    with h5py.File(hdf5_path, "r+") as f:
        acquisition_rate = float(f["metadata"]["acquisition_rate"][()])
        burst_threshold_samples = int(burst_threshold_ms * acquisition_rate / 1000)
        
        results = {}
        for unit_id in f["units"].keys():
            spike_times = f[f"units/{unit_id}/spike_times"][:]
            if len(spike_times) < min_spikes_per_burst:
                continue
            
            isi = np.diff(spike_times)
            burst_mask = isi < burst_threshold_samples
            burst_count = np.sum(np.diff(burst_mask.astype(int)) == 1)
            spikes_in_bursts = np.sum(burst_mask)
            
            burst_stats = {
                "burst_count": int(burst_count),
                "spikes_in_bursts": int(spikes_in_bursts),
                "burst_rate": float(burst_count / (len(spike_times) / acquisition_rate)),
            }
            
            # Write to HDF5
            features_grp = f[f"units/{unit_id}"].require_group("features")
            burst_grp = features_grp.require_group("burst_statistics")
            for key, value in burst_stats.items():
                if key in burst_grp:
                    del burst_grp[key]
                burst_grp.create_dataset(key, data=value)
            
            results[unit_id] = burst_stats
    
    logger.info(f"Computed burst statistics for {len(results)} units")
    return results
```

### Using the Custom Function

```python
from hdmea.pipeline import create_session, load_recording
from hdmea.analysis.custom import compute_burst_statistics

# Deferred mode
session = create_session(dataset_id="recording_001")
session = load_recording(..., session=session)
session = compute_burst_statistics(
    burst_threshold_ms=15.0,
    session=session,
)
session.save()

# Immediate mode
compute_burst_statistics(
    hdf5_path="artifacts/recording_001.h5",
    burst_threshold_ms=15.0,
)
```

---

## Best Practices

### 1. Always Check Prerequisites

```python
if session is not None:
    required_steps = ["load_recording", "add_section_time"]
    for step in required_steps:
        if step not in session.completed_steps:
            raise ConfigurationError(f"Step '{step}' must be completed first")
```

### 2. Use Meaningful Step Names

```python
# Good: descriptive, matches function name
session.mark_step_complete("compute_burst_statistics")

# Bad: vague or inconsistent
session.mark_step_complete("step3")
session.mark_step_complete("done")
```

### 3. Preserve Backwards Compatibility

Always make `session` optional with `None` as default:

```python
def my_function(
    hdf5_path: Optional[...] = None,  # Required if session=None
    session: Optional["PipelineSession"] = None,  # Optional
) -> Union[MyResult, "PipelineSession"]:
```

### 4. Handle Errors Gracefully

```python
if session is not None:
    try:
        result = risky_operation()
    except Exception as e:
        session.warnings.append(f"Operation failed: {e}")
        logger.error(f"Failed: {e}")
        return session  # Return session even on failure
```

### 5. Monitor Memory Usage

```python
if session is not None:
    mem_gb = session.memory_estimate_gb
    if mem_gb > 8.0:
        logger.warning(f"High memory usage: {mem_gb:.2f} GB")
```

### 6. Use Checkpoints for Long Pipelines

```python
session = create_session(dataset_id="long_recording")

# After expensive step
session = load_recording(..., session=session)
session.checkpoint("checkpoints/after_load.h5")

# After another expensive step
session = compute_expensive_features(..., session=session)
session.checkpoint("checkpoints/after_features.h5")

# Final save
session.save()
```

---

## Troubleshooting

### Session Not Saving Data

```python
# Wrong: forgot to reassign
session = create_session(...)
load_recording(..., session=session)  # Returns session, but not captured!

# Correct: reassign session
session = create_session(...)
session = load_recording(..., session=session)  # Capture returned session
```

### TypeError: 'float' object is not callable

```python
# Wrong: memory_estimate_gb is a property, not a method
mem = session.memory_estimate_gb()

# Correct: access as property
mem = session.memory_estimate_gb
```

### ConfigurationError: Step not completed

```python
# Check completed steps before calling a function
print(session.completed_steps)

# Run missing prerequisite
if "load_recording" not in session.completed_steps:
    session = load_recording(..., session=session)
```

### Windows File Locking Error (OSError: Unable to lock file)

On Windows, you may see this error when saving to an existing file:

```
OSError: [Errno 0] Unable to synchronously create file 
(unable to lock file, errno = 0, error message = 'No error', 
Win32 GetLastError() = 33)
```

**Cause**: The HDF5 file is being held by another process (antivirus, file explorer, or a previous unclosed handle).

**Solution**: The session now includes automatic retry logic with file deletion. If the error persists:

1. Close any programs that might have the file open
2. Wait a few seconds and retry
3. Delete the existing file manually before saving

```python
# Force garbage collection before save (helps release file handles)
import gc
gc.collect()

# Then save
session.save(overwrite=True)
```

### Feature Extraction Fails in Session Mode

If feature extraction fails with `KeyError` or attribute errors:

```python
# Ensure load_recording completed successfully
if "load_recording" not in session.completed_steps:
    raise ValueError("Must run load_recording first")

# Check that units have spike_times
for unit_id, unit_data in session.units.items():
    if "spike_times" not in unit_data:
        print(f"Warning: {unit_id} missing spike_times")
```

---

## Batch Processing

For processing multiple recordings, use a loop with error handling:

```python
import csv
from pathlib import Path
from hdmea.pipeline import create_session, load_recording_with_eimage_sta, extract_features

# Read CSV with recording paths
csv_path = Path("path/to/recordings.csv")
output_dir = Path("output/data")
output_dir.mkdir(parents=True, exist_ok=True)

with open(csv_path) as f:
    rows = [r for r in csv.DictReader(f) if r["matched"] == "True"]

for row in rows:
    dataset_id = Path(row["cmtr_path"]).stem.rstrip("-")
    output_path = output_dir / f"{dataset_id}.h5"
    
    if output_path.exists():
        print(f"Skipping {dataset_id} - already exists")
        continue
    
    try:
        session = create_session(dataset_id=dataset_id)
        session = load_recording_with_eimage_sta(
            cmcr_path=row["cmcr_path"],
            cmtr_path=row["cmtr_path"],
            session=session,
        )
        session = extract_features(features=["frif"], session=session)
        session.save(output_path=output_path)
        print(f"Saved: {dataset_id}")
    except Exception as e:
        print(f"Failed: {dataset_id} - {e}")
```

---

## Related Documentation

- [Pipeline Explained](pipeline_explained.md) - Overall pipeline flow
- [Constitution](../.specify/memory/constitution.md) - Project principles
- [Pipeline Log](pipeline_log.md) - Changelog

