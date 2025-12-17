# Data Model: Analog Section Time Detection

**Feature**: 005-analog-section-time  
**Date**: 2025-12-17

## Entities

### SectionTimeArray

The core output entity storing detected stimulus boundaries.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| data | int64[N, 2] | acquisition samples | Array of [start_sample, end_sample] pairs |
| N | int | count | Number of detected trials/sections |

**Storage Location**: `stimulus/section_time/{movie_name}` in Zarr archive

**Example**:
```
[[  200000,  2600000],   # Trial 1: samples 200000-2600000 (~10s-130s at 20kHz)
 [ 3000000,  5400000],   # Trial 2: samples 3000000-5400000 (~150s-270s)
 [ 5800000,  8200000]]   # Trial 3: samples 5800000-8200000 (~290s-410s)
```

**Constraints**:
- `start_sample < end_sample` for all rows
- All values ≥ 0
- Values ≤ max sample index in recording
- Dtype: int64 (to handle large sample counts)

---

### raw_ch1 (Input)

Light reference signal at acquisition rate.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| data | float32[M] | ADC units | Raw analog signal from light sensor |
| M | int | samples | Total samples at acquisition rate (~20 kHz) |

**Storage Location**: `stimulus/light_reference/raw_ch1` in Zarr archive

**Characteristics**:
- Step-function transitions at stimulus onset
- Large derivative peaks at transitions
- Typical length: 20-50 million samples for 20-minute recording

---

### frame_timestamps (Input, Playlist-based only)

Mapping from display frames to acquisition samples.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| data | uint64[F] | acquisition samples | Sample index for each display frame |
| F | int | frames | Number of detected display frames |

**Storage Location**: `metadata/frame_timestamps` in Zarr archive

**Usage**: 
- Required for `add_section_time()` (playlist-based)
- NOT required for `add_section_time_analog()`

**Conversion Formula**:
```python
sample_index = frame_timestamps[frame_index]
```

---

### acquisition_rate (Input)

Sampling rate of the recording.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| value | float64 | Hz | Acquisition sampling rate |

**Storage Location**: `metadata/acquisition_rate` (scalar array) in Zarr archive

**Typical Value**: 20000.0 Hz (20 kHz)

**Usage**: Convert plot_duration (seconds) to sample count
```python
end_sample = onset_sample + int(plot_duration * acquisition_rate)
```

---

## Relationships

```
Zarr Archive
├── metadata/
│   ├── acquisition_rate      ─────────────────────┐
│   └── frame_timestamps      ──────────┐          │
│                                       │          │
├── stimulus/                           │          │
│   ├── light_reference/                │          │
│   │   └── raw_ch1           ──────────┼──────────┼─→ Peak Detection
│   │                                   │          │         │
│   └── section_time/                   │          │         │
│       ├── {movie_name_1}    ←─────────┴──────────┴─────────┘
│       ├── {movie_name_2}              (output)
│       └── iprgc_test        
│
└── units/                    (NOT affected by this feature)
```

---

## State Transitions

### Section Time Lifecycle

```
[Does Not Exist] 
       │
       │ add_section_time_analog() or add_section_time()
       │ (force=False)
       ▼
   [Created]
       │
       │ add_section_time_analog() or add_section_time()
       │ (force=True, same movie_name)
       ▼
  [Overwritten]
```

### Error States

| Condition | Result |
|-----------|--------|
| `raw_ch1` missing | `MissingInputError` |
| `acquisition_rate` missing | `MissingInputError` |
| `frame_timestamps` missing (analog) | OK - not required |
| `frame_timestamps` missing (playlist) | Error - required |
| No peaks detected | Returns `False`, no data written |
| Section exists, force=False | `FileExistsError` |

---

## Validation Rules

### add_section_time_analog()

| Parameter | Validation | Error |
|-----------|------------|-------|
| threshold_value | Must be provided (no default) | ValueError |
| plot_duration | Must be > 0 | ValueError |
| movie_name | Non-empty string | ValueError |

### add_section_time()

| Parameter | Validation | Error |
|-----------|------------|-------|
| playlist_name | Must exist in playlist CSV | Error logged, return False |
| repeats | Converted to 1 if ≤ 0 | None (silent fix) |

---

## Unit Consistency

**CRITICAL**: After this feature, ALL section_time arrays use acquisition sample indices.

| Function | Before | After |
|----------|--------|-------|
| `add_section_time_analog()` | Display frame indices | Acquisition sample indices |
| `add_section_time()` | Display frame indices | Acquisition sample indices |

**Conversion to time (if needed)**:
```python
time_seconds = sample_index / acquisition_rate
```
