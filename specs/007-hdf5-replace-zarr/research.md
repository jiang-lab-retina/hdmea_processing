# Research: Replace Zarr with HDF5

**Date**: 2025-12-17  
**Plan**: [plan.md](./plan.md)

---

## Overview

This document consolidates research findings for implementing HDF5 storage to replace Zarr in the HD-MEA pipeline.

---

## 1. h5py Library Selection

### Decision: `h5py` v3.x

### Rationale

- **Mature and stable**: h5py is the standard Python interface to HDF5, production-ready
- **Direct HDF5 mapping**: Groups, Datasets, and Attributes map directly to HDF5 concepts
- **Lazy loading**: Datasets are not loaded until sliced/accessed
- **NumPy integration**: Seamless conversion to/from numpy arrays
- **Cross-platform**: HDF5 files readable by MATLAB, HDFView, many other tools

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| `pytables` | More complex API; designed for tabular data, not hierarchical |
| `zarr` (keep) | User explicitly requested HDF5 for tooling compatibility |
| Custom binary | No standard tools; maintenance burden |

### Implementation Notes

```python
import h5py
import numpy as np

# Create HDF5 file
with h5py.File("recording.h5", "w") as f:
    # Create groups (like zarr.Group)
    units = f.create_group("units")
    unit = units.create_group("unit_000")
    
    # Create dataset (like zarr.create_dataset)
    spike_times = np.array([100, 200, 300], dtype=np.uint64)
    unit.create_dataset("spike_times", data=spike_times)
    
    # Set attributes (like zarr.attrs)
    unit.attrs["row"] = 5
    unit.attrs["col"] = 10
    f.attrs["dataset_id"] = "JIANG009_2025-04-10"

# Read back
with h5py.File("recording.h5", "r") as f:
    spikes = f["units/unit_000/spike_times"][:]
    row = f["units/unit_000"].attrs["row"]
```

---

## 2. Zarr to HDF5 API Mapping

### Core Concepts

| Zarr | h5py/HDF5 | Notes |
|------|-----------|-------|
| `zarr.open(path, mode)` | `h5py.File(path, mode)` | Nearly identical API |
| `zarr.Group` | `h5py.Group` | Same concept |
| `group.create_group(name)` | `group.create_group(name)` | Identical |
| `group.create_dataset(name, data=...)` | `group.create_dataset(name, data=...)` | Very similar |
| `group.attrs[key] = value` | `group.attrs[key] = value` | Identical |
| `group.keys()` | `group.keys()` | Identical |
| Directory store (`.zarr/`) | Single file (`.h5`) | Key difference |

### Mode Mapping

| Zarr Mode | h5py Mode | Behavior |
|-----------|-----------|----------|
| `"r"` | `"r"` | Read-only |
| `"r+"` | `"r+"` | Read/write, file must exist |
| `"w"` | `"w"` | Create/overwrite |
| `"a"` | `"a"` | Read/write, create if needed |
| `"w-"` | `"w-"` or `"x"` | Create, fail if exists |

### Key Differences

1. **File vs Directory**: HDF5 is a single file; Zarr is a directory tree
2. **Context Manager**: h5py requires `with` statement or explicit `.close()`
3. **String Storage**: h5py stores strings differently; use `h5py.string_dtype()`
4. **Chunking**: Both support chunking, but syntax differs slightly
5. **Compression**: h5py uses `compression='gzip'` (we use none per clarification)

---

## 3. String Handling in HDF5

### Decision: Use variable-length UTF-8 strings

### Rationale

HDF5 has complex string handling. For maximum compatibility (including MATLAB):

```python
# For attributes (simple)
group.attrs["name"] = "value"  # Works directly

# For string datasets (rare in our case)
dt = h5py.string_dtype(encoding='utf-8')
group.create_dataset("names", data=["a", "b"], dtype=dt)
```

### Implementation Notes

- Most of our data is numeric (spike times, waveforms)
- Strings are primarily used in attributes (dataset_id, timestamps)
- h5py handles attribute strings automatically
- For string datasets, use `h5py.string_dtype(encoding='utf-8')`

---

## 4. Single-Writer Enforcement

### Decision: Check file lock on open, raise clear error

### Implementation

HDF5 files can be corrupted by concurrent writes. We enforce single-writer:

```python
import os
import h5py

def open_recording_hdf5(path, mode="r"):
    """Open HDF5 with single-writer check."""
    if mode in ("w", "r+", "a"):
        # Check if file is already open for writing
        # On Windows, this will fail naturally; on Linux, check explicitly
        try:
            # Attempt exclusive open to detect locks
            f = h5py.File(path, mode)
            return f
        except OSError as e:
            if "already open" in str(e).lower() or "locked" in str(e).lower():
                raise OSError(
                    f"File is already open for writing: {path}. "
                    "HDF5 supports single-writer access only."
                ) from e
            raise
    return h5py.File(path, mode)
```

### Notes

- HDF5 library provides some built-in locking
- Windows file locking is stricter than Linux
- We add clear error messages for user understanding

---

## 5. Dataset Creation Patterns

### Decision: Match zarr_store.py patterns exactly

### Spike Times

```python
# Current zarr pattern
spike_ds = unit_group.create_dataset(
    "spike_times",
    data=spike_arr,
    shape=spike_arr.shape,
    dtype=np.uint64,
)
spike_ds.attrs["unit"] = "sample_index"

# HDF5 equivalent
spike_ds = unit_group.create_dataset(
    "spike_times",
    data=spike_arr,
    dtype=np.uint64,
)
spike_ds.attrs["unit"] = "sample_index"
```

### Waveforms

```python
# HDF5 pattern
unit_group.create_dataset(
    "waveform",
    data=waveform_arr,
    dtype=np.float32,
)
```

### Metadata (scalars)

```python
# Store as attributes for fast access
f.attrs["dataset_id"] = dataset_id
f.attrs["acquisition_rate"] = 20000
f.attrs["created_at"] = datetime.now().isoformat()

# Or as single-element datasets (visible in tree)
metadata_group.create_dataset("acquisition_rate", data=np.array([20000]))
```

---

## 6. Lazy Loading Behavior

### Decision: Use HDF5 dataset slicing for lazy access

### Rationale

HDF5 datasets are not loaded until sliced:

```python
with h5py.File("recording.h5", "r") as f:
    # This does NOT load data into memory
    ds = f["units/unit_000/spike_times"]
    
    # This loads all data
    all_data = ds[:]
    
    # This loads only first 100 elements
    partial = ds[:100]
```

### Implementation Notes

- Return `h5py.Dataset` objects for lazy access
- Users call `[:]` when they need the data
- Same pattern as zarr

---

## 7. Performance Considerations

### Benchmarks (Expected)

| Operation | Zarr | HDF5 (no compression) |
|-----------|------|----------------------|
| Create file | ~50ms | ~30ms (single file) |
| Write 1M spike times | ~100ms | ~80ms |
| Read 1M spike times | ~50ms | ~50ms |
| Metadata access | ~5ms | ~2ms (single file) |

### Optimization Decisions

1. **No compression**: Per clarification, prioritize I/O speed over file size
2. **Default chunking**: Let h5py choose chunk sizes automatically
3. **Single file**: Faster metadata access than Zarr directory traversal

---

## 8. File Structure Preservation

### Decision: Maintain identical logical structure

The HDF5 file mirrors the Zarr structure exactly:

```
{dataset_id}.h5
├── / (root)
│   ├── [attrs] dataset_id, hdmea_pipeline_version, created_at, ...
│   ├── units/
│   │   └── {unit_id}/
│   │       ├── [attrs] row, col, global_id, spike_count
│   │       ├── spike_times          # dataset
│   │       ├── waveform             # dataset
│   │       └── features/
│   │           └── {feature_name}/
│   ├── stimulus/
│   │   ├── light_reference/
│   │   ├── frame_time/
│   │   └── section_time/
│   └── metadata/
```

This allows straightforward code migration: replace `zarr` with `h5py` calls.

---

## 9. Migration Strategy

### Decision: Create new module, deprecate old

1. **Create** `src/hdmea/io/hdf5_store.py` with all functions
2. **Keep** `zarr_store.py` but mark as deprecated
3. **Update** imports throughout codebase
4. **Remove** zarr dependency in future version

### Function Mapping

| zarr_store.py | hdf5_store.py |
|---------------|---------------|
| `create_recording_zarr()` | `create_recording_hdf5()` |
| `open_recording_zarr()` | `open_recording_hdf5()` |
| `write_units()` | `write_units()` |
| `write_stimulus()` | `write_stimulus()` |
| `write_metadata()` | `write_metadata()` |
| `write_feature_to_unit()` | `write_feature_to_unit()` |
| `mark_stage1_complete()` | `mark_stage1_complete()` |
| `list_units()` | `list_units()` |
| `list_features()` | `list_features()` |

---

## 10. Testing Strategy

### Unit Tests

```python
def test_create_recording_hdf5():
    """Test HDF5 file creation with correct structure."""
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        root = create_recording_hdf5(Path(f.name), "test_dataset")
        assert "units" in root
        assert "stimulus" in root
        assert "metadata" in root
        assert root.attrs["dataset_id"] == "test_dataset"
        root.close()

def test_write_read_spike_times():
    """Test spike times round-trip."""
    spike_times = np.array([100, 200, 300], dtype=np.uint64)
    # ... write and read back, verify exact match

def test_single_writer_enforcement():
    """Test that concurrent writes raise error."""
    # Open file for writing
    # Attempt second write
    # Verify OSError raised
```

### Integration Tests

- Full pipeline run with HDF5 output
- Feature extraction with HDF5 input
- Visualization with HDF5 files

---

## Summary

The migration from Zarr to HDF5 is straightforward due to similar APIs. Key implementation points:

1. Use `h5py.File` as context manager for proper cleanup
2. Map zarr concepts directly to HDF5 equivalents
3. Store strings as UTF-8 for MATLAB compatibility
4. Enforce single-writer access with clear error messages
5. Skip compression for maximum I/O speed
6. Maintain identical logical data structure

