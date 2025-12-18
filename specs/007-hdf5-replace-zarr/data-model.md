# Data Model: HDF5 Recording Archive

**Feature**: 007-hdf5-replace-zarr  
**Date**: 2025-12-17

---

## Overview

This document defines the HDF5 file structure for HD-MEA recording archives. The structure mirrors the previous Zarr format to ensure logical compatibility while providing the benefits of single-file storage.

---

## File Naming Convention

| Pattern | Example |
|---------|---------|
| `{dataset_id}.h5` | `JIANG009_2025-04-10.h5` |

---

## HDF5 Structure

```
{dataset_id}.h5
│
├── / (root group)
│   │
│   ├── [ATTRIBUTES]
│   │   ├── dataset_id: str              # Unique recording identifier
│   │   ├── hdmea_pipeline_version: str  # Package version (e.g., "0.1.0")
│   │   ├── created_at: str              # ISO 8601 timestamp
│   │   ├── updated_at: str              # ISO 8601 timestamp
│   │   ├── stage1_completed: bool       # Pipeline stage flag
│   │   ├── stage1_params_hash: str      # SHA256 of config
│   │   ├── features_extracted: list     # Feature names list
│   │   └── source_files: dict           # Source file paths
│   │
│   ├── units/                           # GROUP: All spike-sorted units
│   │   │
│   │   └── {unit_id}/                   # GROUP: Single unit (e.g., "unit_000")
│   │       │
│   │       ├── [ATTRIBUTES]
│   │       │   ├── row: int             # Electrode row position
│   │       │   ├── col: int             # Electrode column position
│   │       │   ├── global_id: int       # Global unit identifier
│   │       │   └── spike_count: int     # Number of spikes
│   │       │
│   │       ├── spike_times              # DATASET: (N,) uint64, sample indices
│   │       │   └── [attrs] unit: str    # "sample_index"
│   │       │
│   │       ├── waveform                 # DATASET: (M,) float32, average waveform
│   │       │
│   │       ├── firing_rate_10hz         # DATASET: (T,) float32, binned rate
│   │       │
│   │       ├── spike_times_sectioned/   # GROUP: Sectioned spike times
│   │       │   │
│   │       │   └── {movie_name}/        # GROUP: Per-movie sections
│   │       │       │
│   │       │       ├── full_spike_times     # DATASET: (K,) int64
│   │       │       │
│   │       │       └── trials_spike_times/  # GROUP: Per-trial data
│   │       │           ├── 0                # DATASET: (J,) int64
│   │       │           ├── 1                # DATASET: (J,) int64
│   │       │           └── ...
│   │       │
│   │       └── features/                # GROUP: Extracted features
│   │           │
│   │           └── {feature_name}/      # GROUP: Single feature
│   │               ├── [ATTRIBUTES]     # Feature metadata
│   │               │   ├── version: str
│   │               │   ├── params_hash: str
│   │               │   └── extracted_at: str
│   │               │
│   │               └── {values}         # DATASETS/ATTRS for feature data
│   │
│   ├── stimulus/                        # GROUP: Stimulus information
│   │   │
│   │   ├── light_reference/             # GROUP: Light sensor data
│   │   │   ├── raw_ch1                  # DATASET: (S,) float32
│   │   │   └── raw_ch2                  # DATASET: (S,) float32
│   │   │
│   │   ├── frame_time/                  # GROUP: Frame timing
│   │   │   └── {movie_name}             # DATASET: (F,) uint64
│   │   │
│   │   ├── section_time/                # GROUP: Trial boundaries
│   │   │   └── {movie_name}             # DATASET: (R, 2) uint64
│   │   │
│   │   └── light_template/              # GROUP: Averaged templates
│   │       └── {movie_name}             # DATASET: (T,) float32
│   │
│   └── metadata/                        # GROUP: Recording metadata
│       │
│       ├── acquisition_rate             # DATASET: (1,) float64, Hz
│       ├── frame_time                   # DATASET: (1,) float64
│       │
│       └── sys_meta/                    # GROUP: System metadata
│           ├── version                  # DATASET: str
│           └── ...                      # Additional system fields
```

---

## Entity Definitions

### Recording HDF5 File (Root)

The root group contains global metadata about the recording session.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_id` | str | Yes | Unique identifier (e.g., "JIANG009_2025-04-10") |
| `hdmea_pipeline_version` | str | Yes | Package version |
| `created_at` | str | Yes | ISO 8601 creation timestamp |
| `updated_at` | str | Yes | ISO 8601 last update timestamp |
| `stage1_completed` | bool | Yes | Whether Stage 1 processing is complete |
| `stage1_params_hash` | str | Yes | SHA256 hash of processing config |
| `features_extracted` | list[str] | Yes | Names of extracted features |
| `source_files` | dict | No | Paths to source CMCR/CMTR files |

---

### Unit Group (`/units/{unit_id}/`)

Each unit represents a spike-sorted neuron.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `row` | int | Yes | Electrode row (0-indexed) |
| `col` | int | Yes | Electrode column (0-indexed) |
| `global_id` | int | Yes | Global unit identifier |
| `spike_count` | int | Yes | Total number of spikes |

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `spike_times` | (N,) | uint64 | Spike timestamps in sample indices |
| `waveform` | (M,) | float32 | Average spike waveform |
| `firing_rate_10hz` | (T,) | float32 | Binned firing rate (10 Hz bins) |

---

### Spike Times Sectioned (`/units/{unit_id}/spike_times_sectioned/{movie}/`)

Spike times extracted for specific stimulus periods.

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `full_spike_times` | (K,) | int64 | All spikes from all trials combined |
| `trials_spike_times/{idx}` | (J,) | int64 | Spikes for trial index (0, 1, 2, ...) |

---

### Features Group (`/units/{unit_id}/features/{feature_name}/`)

Extracted feature values for a unit.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `version` | str | Yes | Feature extractor version |
| `params_hash` | str | Yes | SHA256 of extraction parameters |
| `extracted_at` | str | Yes | ISO 8601 timestamp |

Feature values are stored as:
- **Scalars**: HDF5 attributes on the feature group
- **Arrays**: HDF5 datasets within the feature group

---

### Stimulus Group (`/stimulus/`)

Contains all stimulus-related data.

#### Light Reference (`/stimulus/light_reference/`)

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `raw_ch1` | (S,) | float32 | Light sensor channel 1 |
| `raw_ch2` | (S,) | float32 | Light sensor channel 2 |

#### Frame Time (`/stimulus/frame_time/`)

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `{movie_name}` | (F,) | uint64 | Frame timestamps in sample indices |

#### Section Time (`/stimulus/section_time/`)

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `{movie_name}` | (R, 2) | uint64 | Trial boundaries: [start, end] pairs |

---

### Metadata Group (`/metadata/`)

Recording-level metadata and system information.

| Dataset/Attr | Type | Description |
|--------------|------|-------------|
| `acquisition_rate` | float64 | Sampling rate in Hz (e.g., 20000) |
| `frame_time` | float64 | Frame duration |
| `sys_meta/` | group | System metadata subgroup |

---

## Data Type Specifications

| Data | Python Type | HDF5 Type | Notes |
|------|-------------|-----------|-------|
| Spike times | `np.uint64` | `H5T_STD_U64LE` | Sample indices |
| Waveforms | `np.float32` | `H5T_IEEE_F32LE` | Voltage values |
| Firing rates | `np.float32` | `H5T_IEEE_F32LE` | Spikes per bin |
| Timestamps | `str` | Variable-length UTF-8 | ISO 8601 format |
| Counts/indices | `int` | `H5T_STD_I64LE` | 64-bit signed |
| Boolean flags | `bool` | `H5T_STD_I8LE` | 0 or 1 |

---

## Validation Rules

1. **Root attributes**: All required attributes must be present
2. **Unit naming**: Unit IDs must match pattern `unit_\d{3}` (e.g., `unit_000`)
3. **Spike times**: Must be sorted in ascending order
4. **Data types**: Must match specified dtypes exactly
5. **Non-negative**: Spike times and indices must be ≥ 0
6. **Consistency**: `spike_count` attribute must equal `len(spike_times)`

---

## Access Patterns

### Read spike times for a unit

```python
with h5py.File("recording.h5", "r") as f:
    spike_times = f["units/unit_000/spike_times"][:]
```

### List all units

```python
with h5py.File("recording.h5", "r") as f:
    unit_ids = list(f["units"].keys())
```

### Read section time for a movie

```python
with h5py.File("recording.h5", "r") as f:
    boundaries = f["stimulus/section_time/baseline_127"][:]
    # boundaries.shape = (num_trials, 2)
```

### Read feature value

```python
with h5py.File("recording.h5", "r") as f:
    feature_group = f["units/unit_000/features/step_up"]
    on_index = feature_group.attrs["on_index"]
    response_curve = feature_group["response_curve"][:]
```

---

## Comparison: Zarr vs HDF5

| Aspect | Zarr | HDF5 |
|--------|------|------|
| Storage | Directory tree | Single file |
| Extension | `.zarr/` | `.h5` |
| Python library | `zarr` | `h5py` |
| Parallel write | Supported (chunks) | Single-writer only |
| Cloud storage | Native support | Limited |
| External tools | Limited | HDFView, MATLAB, etc. |
| Attribute access | `group.attrs[key]` | `group.attrs[key]` |
| Dataset creation | `create_dataset()` | `create_dataset()` |

The logical structure is identical; only the storage format differs.

