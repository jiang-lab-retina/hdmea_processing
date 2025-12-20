# Data Model: Electrode Image STA (eimage_sta)

**Date**: 2025-12-19  
**Feature**: 001-eimage-sta-feature

## Entities

### 1. EImageSTA

Represents the computed electrode image spike-triggered average for a single unit.

| Field | Type | Description |
|-------|------|-------------|
| `data` | np.ndarray (float32) | 3D array: (window_length, rows, cols) |
| `n_spikes` | int | Number of spikes used in average |
| `n_spikes_excluded` | int | Spikes excluded due to edge effects |
| `pre_samples` | int | Number of samples before spike (default: 10) |
| `post_samples` | int | Number of samples after spike (default: 40) |
| `cutoff_hz` | float | High-pass filter cutoff frequency (default: 100.0) |
| `filter_order` | int | Butterworth filter order (default: 2) |
| `sampling_rate` | float | Acquisition rate in Hz |
| `spike_limit` | int | Max spikes used (-1 for no limit) |
| `version` | str | Extractor version string |

**Dimensions**:
- `window_length` = `pre_samples` + `post_samples` (default: 50)
- `rows` = 64 (HD-MEA electrode rows)
- `cols` = 64 (HD-MEA electrode columns)

**Storage**: HDF5 dataset at `units/{unit_id}/features/eimage_sta/data`

---

### 2. SensorData

Raw voltage recordings from the HD-MEA electrode array.

| Field | Type | Description |
|-------|------|-------------|
| `data` | np.ndarray (int16) | 3D array: (time_samples, rows, cols) |
| `sampling_rate` | float | Acquisition rate in Hz (typically 20,000) |
| `duration_s` | float | Recording duration in seconds |

**Source**: CMCR file via McsPy: `Acquisition.Sensor_Data.SensorData_1_1`

**Access Pattern**: Memory-mapped (lazy loading on slice access)

---

### 3. FilteredSensorData

High-pass filtered sensor data (optional cache).

| Field | Type | Description |
|-------|------|-------------|
| `data` | np.ndarray (float32) | 3D array: (time_samples, rows, cols) |
| `cutoff_hz` | float | Filter cutoff frequency |
| `filter_order` | int | Butterworth filter order |
| `sampling_rate` | float | Acquisition rate in Hz |

**Storage**: Optional HDF5 cache file or in-memory

---

### 4. EImageSTAConfig

Configuration for eimage_sta computation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cutoff_hz` | float | 100.0 | High-pass filter cutoff frequency |
| `filter_order` | int | 2 | Butterworth filter order |
| `pre_samples` | int | 10 | Samples before spike in window |
| `post_samples` | int | 40 | Samples after spike in window |
| `spike_limit` | int | 10000 | Max spikes per unit (-1 for no limit) |
| `duration_s` | float | 120.0 | Duration of sensor data to use |
| `use_cache` | bool | False | Enable filtered data caching |
| `cache_path` | str | None | Path to cache file (auto-generated if None) |
| `force` | bool | False | Overwrite existing eimage_sta |

---

### 5. EImageSTAResult

Result of eimage_sta computation for a recording.

| Field | Type | Description |
|-------|------|-------------|
| `hdf5_path` | Path | Path to HDF5 file processed |
| `units_processed` | int | Number of units successfully processed |
| `units_failed` | int | Number of units that failed |
| `elapsed_seconds` | float | Total computation time |
| `filter_time_seconds` | float | Time spent on filtering |
| `warnings` | list[str] | Warning messages generated |
| `failed_units` | list[str] | Unit IDs that failed |

---

## HDF5 Schema

### Input Structure (existing)

```
/units/
└── {unit_id}/
    ├── spike_times          # Dataset: uint64, shape (n_spikes,)
    └── features/            # Group: existing features
        └── ...
```

### Output Structure (added by this feature)

```
/units/
└── {unit_id}/
    └── features/
        └── eimage_sta/                    # Group: NEW
            ├── data                       # Dataset: float32, (50, 64, 64)
            └── [attrs]
                ├── n_spikes: 8542
                ├── n_spikes_excluded: 23
                ├── pre_samples: 10
                ├── post_samples: 40
                ├── cutoff_hz: 100.0
                ├── filter_order: 2
                ├── sampling_rate: 20000.0
                ├── spike_limit: 10000
                └── version: "1.0.0"
```

---

## Data Flow

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│ CMCR File   │────▶│ Sensor Data      │────▶│ Filtered Data  │
│ (McsPy)     │     │ (int16, mmap)    │     │ (float32)      │
└─────────────┘     └──────────────────┘     └───────┬────────┘
                                                     │
┌─────────────┐                                      │
│ HDF5 File   │◀────────────────────────────────────┐│
│ (spike_times)                                     ││
└──────┬──────┘                                     ││
       │                                            ││
       ▼                                            ▼│
┌──────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Spike Times  │────▶│ Window Indices   │────▶│ eimage_sta  │
│ (per unit)   │     │ (fancy indexing) │     │ (float32)   │
└──────────────┘     └──────────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ HDF5 Output     │
                                            │ (features group)│
                                            └─────────────────┘
```

---

## Validation Rules

### SensorData
- Shape must be 3D: `(time, rows, cols)`
- dtype must be int16 or compatible
- rows and cols typically 64×64 for HD-MEA

### SpikeTimes
- Must be sorted ascending
- Values in microseconds (uint64)
- Must have at least 1 spike (or NaN result returned)

### EImageSTA
- Shape must be `(pre_samples + post_samples, rows, cols)`
- dtype must be float32
- NaN values indicate no valid spikes

### EImageSTAConfig
- `cutoff_hz > 0`
- `filter_order >= 1`
- `pre_samples >= 0`
- `post_samples > 0`
- `spike_limit == -1` or `spike_limit > 0`
- `duration_s > 0`

