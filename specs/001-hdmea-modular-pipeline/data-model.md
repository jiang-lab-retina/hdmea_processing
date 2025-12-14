# Data Model: HD-MEA Data Analysis Pipeline v1

**Date**: 2025-12-14  
**Plan**: [plan.md](./plan.md)

---

## Overview

This document defines the entity schemas for the HD-MEA pipeline. All entities are stored in Zarr format (hierarchical) or JSON (configuration).

---

## Core Entities

### 1. Recording

A single HD-MEA recording session, represented by external `.cmcr` and/or `.cmtr` files.

| Field | Type | Description |
|-------|------|-------------|
| `dataset_id` | string | Unique identifier (e.g., `"JIANG009_2024-01-15"`) |
| `cmcr_path` | string \| null | External path to raw sensor data |
| `cmtr_path` | string \| null | External path to spike-sorted data |
| `recording_duration_s` | float | Total duration in seconds |
| `acquisition_rate_hz` | float | Sampling rate |
| `num_units` | int | Number of sorted units |

**Location in Zarr**: Root `.zattrs`

**Validation Rules**:
- At least one of `cmcr_path` or `cmtr_path` MUST be non-null
- `dataset_id` MUST match regex `^[A-Z]+\d+(_[\d-]+)?$`

---

### 2. Unit (Cell)

A single sorted neuron/cell within a recording.

| Field | Type | Description |
|-------|------|-------------|
| `unit_id` | string | Unique within recording (e.g., `"unit_001"`) |
| `global_id` | int | Global electrode ID |
| `row` | int | Electrode array row |
| `col` | int | Electrode array column |
| `spike_count` | int | Total number of spikes |
| `mean_firing_rate_hz` | float | Average firing rate |

**Location in Zarr**: `units/{unit_id}/.zattrs`

**Arrays**:

| Array | Shape | Dtype | Unit | Description |
|-------|-------|-------|------|-------------|
| `spike_times` | (N,) | uint64 | μs | Spike timestamps |
| `waveform` | (T,) | float32 | μV | Mean spike waveform (T≈50 samples) |
| `firing_rate_10hz` | (M,) | float32 | spikes/s | Binned firing rate at 10Hz |

**Location in Zarr**: `units/{unit_id}/`

---

### 3. Stimulus

Stimulus presentation information and timing.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Stimulus identifier (e.g., `"step_up_5s_5i_3x"`) |
| `movie_length_frames` | int | Total frames in movie |
| `frame_rate_hz` | float | Presentation frame rate |
| `num_repeats` | int | Number of repeats presented |
| `sections` | dict | Mapping of section name → frame ranges |

**Location in Zarr**: `stimulus/.zattrs`

**Arrays**:

| Array | Path | Shape | Dtype | Description |
|-------|------|-------|-------|-------------|
| Light reference (20kHz) | `stimulus/light_reference/20kHz` | (N,) | float32 | High-res light sensor |
| Frame times | `stimulus/frame_time/{movie}` | (F,) | uint64 | Frame → timestamp (μs) |
| Section times | `stimulus/section_time/{section}` | (R, 2) | uint64 | Repeat → (start, end) μs |
| Template | `stimulus/template_auto/{movie}` | (F,) | float32 | Averaged light template |

---

### 4. Feature

Extracted feature for a single unit under a specific stimulus condition.

| Field | Type | Description |
|-------|------|-------------|
| `feature_name` | string | Feature set name (e.g., `"step_up_5s_5i_3x"`) |
| `extractor_version` | string | Version of extractor (e.g., `"1.0.0"`) |
| `params_hash` | string | SHA256 hash of extraction parameters |
| `extracted_at` | string | ISO 8601 timestamp |

**Location in Zarr**: `units/{unit_id}/features/{feature_name}/.zattrs`

**Feature-specific arrays** (examples):

#### ON/OFF Response Features (`step_up_5s_5i_3x`)

| Array | Dtype | Unit | Description |
|-------|-------|------|-------------|
| `on_response_flag` | bool | — | True if significant ON response |
| `off_response_flag` | bool | — | True if significant OFF response |
| `on_peak_value` | float64 | spikes/s | Peak firing at light ON |
| `off_peak_value` | float64 | spikes/s | Peak firing at light OFF |
| `on_sustained_response` | float64 | spikes/s | Sustained rate after ON |
| `off_sustained_response` | float64 | spikes/s | Sustained rate after OFF |
| `response_quality` | float64 | dimensionless | Quality metric [0, 1] |
| `filtered_trace` | float32[] | spikes/s | Averaged response trace |

#### Receptive Field Features (`perfect_dense_noise_15x15_15hz_r42_3min`)

| Array | Path | Dtype | Description |
|-------|------|-------|-------------|
| STA array | `sta/sta_array` | float32[T,H,W] | Spike-triggered average |
| Center coord | `sta/center_coordinate` | float64[3] | [time, y, x] of center |
| Spike frames | `sta/spike_as_frame_num` | int64[] | Spikes as frame indices |
| Gaussian max | `gaussian_fit/parameters_max` | float64[6] | Fit params at max |
| Gaussian min | `gaussian_fit/parameters_min` | float64[6] | Fit params at min |

#### Direction Selectivity Features (`moving_h_bar_s5_d8_3x`)

| Array | Dtype | Unit | Description |
|-------|-------|------|-------------|
| `dsi_on` | float64 | dimensionless | DSI for ON response [0, 1] |
| `dsi_off` | float64 | dimensionless | DSI for OFF response [0, 1] |
| `osi_on` | float64 | dimensionless | OSI for ON response [0, 1] |
| `osi_off` | float64 | dimensionless | OSI for OFF response [0, 1] |
| `preferred_direction_on` | float64 | degrees | Preferred direction (ON) |
| `preferred_direction_off` | float64 | degrees | Preferred direction (OFF) |
| `p_value/on_p_value` | float64 | — | Significance p-value |
| `p_value/off_p_value` | float64 | — | Significance p-value |
| `tuning_curve_on` | float64[D] | spikes/s | Response per direction |
| `tuning_curve_off` | float64[D] | spikes/s | Response per direction |

---

### 5. Pipeline Metadata

Versioning and provenance information at the Zarr root.

| Field | Type | Description |
|-------|------|-------------|
| `hdmea_pipeline_version` | string | Package version |
| `created_at` | string | ISO 8601 creation time |
| `updated_at` | string | ISO 8601 last modification |
| `stage1_completed` | bool | Data loading complete |
| `stage1_params_hash` | string | Hash of Stage 1 config |
| `features_extracted` | list[string] | List of extracted feature names |
| `source_files` | object | Source file provenance |

**Location in Zarr**: Root `.zattrs`

---

## Configuration Entities

### 6. Flow Configuration

Defines a named pipeline flow (Stage 1 + Stage 2 features).

```json
{
  "name": "set6a_full",
  "description": "Full set6a feature extraction workflow",
  "version": "1.0.0",
  "stages": {
    "load": {
      "enabled": true
    },
    "features": {
      "enabled": true,
      "feature_sets": [
        "baseline_127",
        "step_up_5s_5i_3x",
        "moving_h_bar_s5_d8_3x",
        "perfect_dense_noise_15x15_15hz_r42_3min"
      ]
    }
  },
  "defaults": {
    "force_recompute": false,
    "random_seed": 42
  }
}
```

**Location**: `config/flows/{flow_name}.json`

---

### 7. Stimulus Configuration

Defines timing and parameters for a stimulus type.

```json
{
  "name": "step_up_5s_5i_3x",
  "display_name": "Step Up 5s/5i 3x",
  "movie_length_frames": 300,
  "frame_rate_hz": 30.0,
  "num_repeats": 3,
  "sections": {
    "baseline": [0, 150],
    "step_up": [150, 300]
  },
  "analysis_windows": {
    "on_window_ms": [0, 500],
    "off_window_ms": [0, 500],
    "sustained_window_ms": [500, 5000]
  }
}
```

**Location**: `config/stimuli/{stimulus_name}.json`

---

## Zarr Structure Summary

```
{dataset_id}.zarr/
├── .zattrs                           # Recording + pipeline metadata
├── units/
│   └── {unit_id}/
│       ├── .zattrs                   # Unit metadata
│       ├── spike_times               # uint64 array
│       ├── waveform                  # float32 array
│       ├── firing_rate_10hz          # float32 array
│       └── features/
│           └── {feature_name}/
│               ├── .zattrs           # Feature metadata
│               ├── {scalar_feature}  # scalar or 1D array
│               └── {nested}/         # Optional subgroup
│                   └── {array}
├── stimulus/
│   ├── .zattrs                       # Stimulus metadata
│   ├── light_reference/
│   │   ├── 20kHz                     # float32 array
│   │   └── 10Hz                      # float32 array (downsampled)
│   ├── frame_time/
│   │   └── {movie_name}              # uint64 array
│   ├── section_time/
│   │   └── {section_name}            # uint64 array (R, 2)
│   └── template_auto/
│       └── {movie_name}              # float32 array
└── metadata/
    ├── acquisition_rate              # scalar
    ├── recording_duration            # scalar
    └── electrode_geometry            # optional array
```

---

## State Transitions

### Recording Lifecycle

```
┌─────────────────┐
│     EMPTY       │  (No Zarr exists)
└────────┬────────┘
         │ Stage 1: Data Loading
         ▼
┌─────────────────┐
│    LOADED       │  stage1_completed=true
│                 │  features_extracted=[]
└────────┬────────┘
         │ Stage 2: Feature Extraction
         ▼ (one or more features)
┌─────────────────┐
│   EXTRACTED     │  features_extracted=[...]
│                 │  Each feature has .zattrs
└────────┬────────┘
         │ Optional: Parquet Export
         ▼
┌─────────────────┐
│   EXPORTED      │  (Parquet file exists)
└─────────────────┘
```

### Feature Cache States

```
                    ┌─────────────────┐
                    │   NOT PRESENT   │
                    └────────┬────────┘
                             │ Run extractor
                             ▼
                    ┌─────────────────┐
       CACHE HIT ◄──│     VALID       │──► FORCE OVERWRITE
      (skip run)    │  version match  │    (force=True)
                    └────────┬────────┘
                             │ Version mismatch
                             ▼
                    ┌─────────────────┐
                    │     STALE       │──► Re-extract
                    │ version differs │
                    └─────────────────┘
```

---

## Validation Rules Summary

| Entity | Rule | Error Type |
|--------|------|------------|
| Recording | At least one of cmcr_path/cmtr_path non-null | `ConfigurationError` |
| Recording | dataset_id matches pattern | `ConfigurationError` |
| Unit | spike_times monotonically increasing | `DataLoadError` |
| Unit | waveform length matches expected | `DataLoadError` |
| Feature | Required inputs present in Zarr | `MissingInputError` |
| Feature | No overwrite without force=True | `FeatureExtractionError` |
| Config | JSON schema validates | `ConfigurationError` |

