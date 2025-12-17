# Quickstart: Add frame_time and acquisition_rate to Metadata

**Feature Branch**: `003-add-metadata-fields`  
**Date**: 2024-12-15  
**Status**: ✅ IMPLEMENTED

## Overview

This feature adds timing metadata fields to Zarr archives:
- `acquisition_rate`: Sampling rate in Hz (e.g., 20000)
- `sample_interval`: Duration per sample in seconds (e.g., 0.00005)
- `frame_timestamps`: Array of sample indices where video frames start
- `frame_time`: Array of frame start times in seconds

Raw file metadata from CMCR/CMTR is stored under `metadata/sys_meta/` subgroup.

The `acquisition_rate` is extracted using a priority chain: **CMCR → CMTR → default (20000 Hz)**.

## Metadata Structure

```
/metadata/
├── acquisition_rate     # float64 - sampling rate in Hz
├── sample_interval      # float64 - 1/acquisition_rate
├── frame_timestamps     # uint64 array - sample indices of frame starts
├── frame_time           # float64 array - frame times in seconds
├── dataset_id           # string - dataset identifier
└── sys_meta/            # subgroup with raw file metadata
    ├── DateTime         # recording date/time
    ├── ProgramName      # e.g., "CMOS-MEA-Control"
    ├── recording_duration_s  # total duration
    └── ...              # other CMCR/CMTR metadata
```

## Usage

### Accessing Timing Metadata

```python
import zarr

# Open existing Zarr archive
root = zarr.open("artifacts/preprocessed/RECORDING_001.zarr", mode="r")

# Access timing metadata (stored as 1-element arrays)
metadata = root["metadata"]
acquisition_rate = metadata["acquisition_rate"][0]  # e.g., 20000.0
sample_interval = metadata["sample_interval"][0]    # e.g., 0.00005

# Access frame timestamps (arrays)
frame_timestamps = metadata["frame_timestamps"][:]  # sample indices
frame_time = metadata["frame_time"][:]              # seconds

print(f"Sampling rate: {acquisition_rate} Hz")
print(f"Sample interval: {sample_interval * 1e6:.2f} µs")
print(f"Number of frames: {len(frame_timestamps)}")
```

### Accessing Raw File Metadata (sys_meta)

```python
# Access system metadata from raw files
sys_meta = metadata["sys_meta"]
recording_duration = sys_meta["recording_duration_s"][0]
program_name = sys_meta["ProgramName"][0]
```

### Using in Analysis

```python
import numpy as np

# Convert sample indices to time
sample_indices = np.array([0, 100, 200, 300])
times_seconds = sample_indices * frame_time

# Convert spike times (in samples) to seconds
spike_samples = np.array([1000, 2500, 5000])
spike_times_s = spike_samples / acquisition_rate
```

### Viewing in zarr-viz GUI

```bash
# Launch the GUI
python -m hdmea.viz.zarr_viz path/to/recording.zarr

# Navigate to /metadata group in the tree
# Timing metadata is displayed prominently under "⏱️ Timing Metadata"
```

## Implementation Summary

### Completed Changes

| File | Change | Status |
|------|--------|--------|
| `src/hdmea/io/cmtr.py` | Added `acquisition_rate` extraction from file attributes | ✅ |
| `src/hdmea/pipeline/runner.py` | Added priority chain, `frame_time` computation, validation, logging | ✅ |
| `src/hdmea/viz/zarr_viz/app.py` | Enhanced group view to display timing metadata | ✅ |
| `tests/unit/test_metadata_fields.py` | New unit tests for validation and computation | ✅ |
| `tests/fixtures/synthetic_zarr.py` | Added fixtures for metadata testing | ✅ |

### Key Functions Added

- `validate_acquisition_rate(rate)` - Validates rate is positive, warns if outside 1000-100000 Hz
- `compute_frame_time(acquisition_rate)` - Computes `1 / acquisition_rate`
- `render_group_view(node)` - Displays group attributes including timing metadata in GUI

### Constants

- `DEFAULT_ACQUISITION_RATE = 20000.0 Hz`
- `MIN_TYPICAL_ACQUISITION_RATE = 1000.0 Hz`
- `MAX_TYPICAL_ACQUISITION_RATE = 100000.0 Hz`

## Testing

```bash
# Run unit tests
pytest tests/unit/test_metadata_fields.py -v

# Verify metadata in existing Zarr
python -c "
import zarr
root = zarr.open('artifacts/preprocessed/RECORDING.zarr')
print('acquisition_rate:', root['metadata'].attrs.get('acquisition_rate'))
print('frame_time:', root['metadata'].attrs.get('frame_time'))
"
```

## Acceptance Criteria Verification

- ✅ **SC-001**: Zarr files contain both `acquisition_rate` and `frame_time` in metadata group
- ✅ **SC-002**: `frame_time * acquisition_rate == 1.0` (mathematically consistent)
- ✅ **SC-003**: Single attribute lookup access (no computation required)
- ✅ **SC-004**: Timing metadata visible in zarr-viz GUI
- ✅ **SC-005**: CMCR extraction used when available
