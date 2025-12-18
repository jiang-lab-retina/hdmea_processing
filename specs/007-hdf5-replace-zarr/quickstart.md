# Quickstart: HDF5 Store Module

**Feature**: 007-hdf5-replace-zarr  
**Module**: `hdmea.io.hdf5_store`

---

## Installation

The `h5py` library is required:

```bash
pip install h5py
```

Or with the hdmea package:

```bash
pip install -e ".[dev]"
```

---

## Basic Usage

### Creating a New Recording

```python
from pathlib import Path
from hdmea.io.hdf5_store import create_recording_hdf5, write_units, write_metadata
import numpy as np

# Create new HDF5 file
hdf5_path = Path("artifacts/JIANG009_2025-04-10.h5")

with create_recording_hdf5(hdf5_path, "JIANG009_2025-04-10") as f:
    # Prepare unit data
    units_data = {
        "unit_000": {
            "spike_times": np.array([1000, 2000, 3000], dtype=np.uint64),
            "waveform": np.random.randn(50).astype(np.float32),
            "row": 5,
            "col": 10,
            "global_id": 0,
        },
        "unit_001": {
            "spike_times": np.array([1500, 2500, 3500], dtype=np.uint64),
            "waveform": np.random.randn(50).astype(np.float32),
            "row": 6,
            "col": 10,
            "global_id": 1,
        },
    }
    
    # Write units
    write_units(f, units_data)
    
    # Write metadata
    metadata = {
        "acquisition_rate": 20000.0,
        "frame_time": 0.01,
    }
    write_metadata(f, metadata)
```

---

### Reading Data

```python
from hdmea.io.hdf5_store import open_recording_hdf5, list_units

# Open existing file
with open_recording_hdf5(hdf5_path, mode="r") as f:
    # List all units
    unit_ids = list_units(f)
    print(f"Found {len(unit_ids)} units: {unit_ids}")
    
    # Read spike times for a unit
    spike_times = f["units/unit_000/spike_times"][:]
    print(f"Spike times: {spike_times}")
    
    # Read unit attributes
    row = f["units/unit_000"].attrs["row"]
    col = f["units/unit_000"].attrs["col"]
    print(f"Unit position: ({row}, {col})")
    
    # Read metadata
    acq_rate = f["metadata/acquisition_rate"][0]
    print(f"Acquisition rate: {acq_rate} Hz")
```

---

### Writing Stimulus Data

```python
from hdmea.io.hdf5_store import open_recording_hdf5, write_stimulus
import numpy as np

with open_recording_hdf5(hdf5_path, mode="r+") as f:
    # Light reference data
    light_reference = {
        "raw_ch1": np.random.randn(10000).astype(np.float32),
        "raw_ch2": np.random.randn(10000).astype(np.float32),
    }
    
    # Section times (trial boundaries)
    section_times = {
        "baseline_127": np.array([
            [0, 40000],      # Trial 0: samples 0-40000
            [50000, 90000],  # Trial 1
            [100000, 140000] # Trial 2
        ], dtype=np.uint64),
    }
    
    write_stimulus(f, light_reference, section_times=section_times)
```

---

### Writing Features

```python
from hdmea.io.hdf5_store import open_recording_hdf5, write_feature_to_unit
import numpy as np
from datetime import datetime

with open_recording_hdf5(hdf5_path, mode="r+") as f:
    # Feature data
    feature_data = {
        "on_index": 0.85,
        "off_index": 0.42,
        "on_off_ratio": 0.43,
        "response_curve": np.random.randn(100).astype(np.float32),
    }
    
    # Feature metadata
    metadata = {
        "version": "1.0.0",
        "params_hash": "abc123",
        "extracted_at": datetime.now().isoformat(),
    }
    
    write_feature_to_unit(f, "unit_000", "step_up", feature_data, metadata)
```

---

### Inspecting with HDFView

HDF5 files can be opened with [HDFView](https://www.hdfgroup.org/downloads/hdfview/):

1. Download and install HDFView
2. Open your `.h5` file
3. Navigate the tree structure
4. Double-click datasets to view data

---

### MATLAB Access

```matlab
% Read spike times
spike_times = h5read('JIANG009_2025-04-10.h5', '/units/unit_000/spike_times');

% Read attribute
info = h5info('JIANG009_2025-04-10.h5', '/units/unit_000');
row = info.Attributes(strcmp({info.Attributes.Name}, 'row')).Value;

% Read all unit names
info = h5info('JIANG009_2025-04-10.h5', '/units');
unit_names = {info.Groups.Name};
```

---

## Migration from Zarr

### Before (Zarr)

```python
from hdmea.io.zarr_store import create_recording_zarr, open_recording_zarr

# Create
root = create_recording_zarr(Path("recording.zarr"), "dataset_id")

# Open
root = open_recording_zarr(Path("recording.zarr"), mode="r")
```

### After (HDF5)

```python
from hdmea.io.hdf5_store import create_recording_hdf5, open_recording_hdf5

# Create (use context manager)
with create_recording_hdf5(Path("recording.h5"), "dataset_id") as f:
    # ... write data ...
    pass  # auto-closes

# Open (use context manager)
with open_recording_hdf5(Path("recording.h5"), mode="r") as f:
    # ... read data ...
    pass  # auto-closes
```

### Key Differences

| Aspect | Zarr | HDF5 |
|--------|------|------|
| File extension | `.zarr` (directory) | `.h5` (single file) |
| Context manager | Optional | Recommended |
| Close file | Automatic | Explicit (via context manager) |
| Concurrent writes | Supported | Single-writer only |

---

## Common Patterns

### Check if Stage 1 Complete

```python
with open_recording_hdf5(hdf5_path, mode="r") as f:
    if f.attrs.get("stage1_completed", False):
        print("Stage 1 is complete")
    else:
        print("Stage 1 not yet run")
```

### Iterate Over All Units

```python
with open_recording_hdf5(hdf5_path, mode="r") as f:
    for unit_id in f["units"].keys():
        spike_count = f["units"][unit_id].attrs["spike_count"]
        print(f"{unit_id}: {spike_count} spikes")
```

### Read Section Times for a Movie

```python
with open_recording_hdf5(hdf5_path, mode="r") as f:
    if "section_time" in f["stimulus"]:
        for movie in f["stimulus/section_time"].keys():
            boundaries = f[f"stimulus/section_time/{movie}"][:]
            print(f"{movie}: {len(boundaries)} trials")
```

---

## Error Handling

### File Already Open

```python
try:
    with open_recording_hdf5(hdf5_path, mode="r+") as f:
        # ...
        pass
except OSError as e:
    if "already open" in str(e).lower():
        print("File is locked by another process")
    else:
        raise
```

### File Not Found

```python
try:
    with open_recording_hdf5(Path("nonexistent.h5"), mode="r") as f:
        pass
except FileNotFoundError:
    print("Recording file not found")
```

---

## Visualization

The visualization GUI works with HDF5 files:

```bash
python -m hdmea.viz.hdf5_viz "artifacts/JIANG009_2025-04-10.h5"
```

Or in Python:

```python
from hdmea.viz.hdf5_viz import launch_viewer

launch_viewer("artifacts/JIANG009_2025-04-10.h5")
```

