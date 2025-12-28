# Data Model: Unified Pipeline Session

**Branch**: `012-unified-pipeline-session` | **Date**: 2025-12-28

## Overview

This document defines the data entities, their attributes, relationships, and the HDF5 storage structure for the unified pipeline session.

---

## Core Entities

### 1. PipelineSession

The central in-memory container for all pipeline data.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset_id` | `str` | Unique recording identifier (e.g., "2024.08.08-10.40.20-Rec") |
| `save_state` | `SaveState` | Current state: DEFERRED or SAVED |
| `hdf5_path` | `Optional[Path]` | Path to HDF5 file (set after save) |
| `output_dir` | `Path` | Default output directory |
| `units` | `Dict[str, Dict]` | Unit data keyed by unit_id |
| `metadata` | `Dict[str, Any]` | Recording metadata |
| `stimulus` | `Dict[str, Any]` | Stimulus data |
| `source_files` | `Dict[str, Path]` | Paths to source CMCR/CMTR files |
| `completed_steps` | `Set[str]` | Names of completed pipeline steps |
| `warnings` | `List[str]` | Accumulated warning messages |
| `created_at` | `str` | ISO 8601 creation timestamp |
| `saved_at` | `Optional[str]` | ISO 8601 save timestamp |

### 2. Unit

Individual neural unit with spike data and extracted features.

| Attribute | Type | Description |
|-----------|------|-------------|
| `unit_id` | `str` | Unique identifier (e.g., "unit_001") |
| `spike_times` | `np.ndarray` | Array of spike timestamps |
| `waveform` | `np.ndarray` | Representative waveform shape |
| `unit_meta` | `Dict` | Extended metadata from CMTR |
| `features` | `Dict[str, FeatureData]` | Extracted features by name |
| `auto_label` | `Dict` | Automatic classification labels |

### 3. FeatureData

Extracted feature results for a unit.

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | Primary feature data |
| `metadata` | `Dict` | Extraction parameters and metadata |
| `geometry` | `Optional[Dict]` | Geometric analysis results |

### 4. Metadata

Recording-level metadata.

| Attribute | Type | Description |
|-----------|------|-------------|
| `acquisition_rate` | `float` | Sampling rate in Hz |
| `duration` | `float` | Recording duration in seconds |
| `cmcr_meta` | `Dict` | Metadata from CMCR file |
| `cmtr_meta` | `Dict` | Metadata from CMTR file |
| `gsheet_row` | `Dict` | Google Sheet metadata row |
| `ap_tracking` | `Dict` | AP tracking global results |
| `rf_sta_geometry` | `Dict` | RF-STA geometry summary |

### 5. Stimulus

Stimulus timing and pattern data.

| Attribute | Type | Description |
|-----------|------|-------------|
| `light_reference` | `np.ndarray` | Light stimulus reference signal |
| `frame_times` | `np.ndarray` | Frame timing information |
| `light_template` | `np.ndarray` | Stimulus template pattern |
| `section_time` | `Dict[str, SectionTime]` | Section timing per playlist |

---

## HDF5 Storage Structure

```
{dataset_id}.h5
├── [attrs]
│   ├── dataset_id: str
│   ├── created_at: str (ISO 8601)
│   ├── hdmea_version: str
│   └── saved_at: str (ISO 8601)
│
├── units/
│   └── {unit_id}/                          # e.g., "unit_001"
│       ├── spike_times: float64[]
│       ├── waveform: float64[N]
│       ├── unit_meta/
│       │   ├── row: int64
│       │   ├── column: int64
│       │   ├── snr: float64
│       │   ├── separability: float64
│       │   └── ...
│       ├── auto_label/
│       │   └── axon_type: str              # "rgc", "ac", "other", "no_label"
│       └── features/
│           ├── eimage_sta/
│           │   ├── data: float64[T,H,W]
│           │   └── geometry/
│           │       ├── centroid_x: float64
│           │       ├── centroid_y: float64
│           │       ├── soma_area: float64
│           │       └── ...
│           ├── sta/
│           │   ├── data: float64[...]
│           │   └── geometry/
│           │       ├── gaussian_params: float64[N]
│           │       ├── dog_params: float64[N]
│           │       └── ...
│           ├── ap_tracking/
│           │   ├── prediction_sta_data: float64[T,H,W]
│           │   ├── centroids_raw: float64[N,2]
│           │   ├── line_fit/
│           │   │   ├── slope: float64
│           │   │   ├── intercept: float64
│           │   │   ├── r_squared: float64
│           │   │   └── ...
│           │   └── ...
│           └── dsgc_section/
│               ├── {movie_name}/
│               │   ├── direction_0/
│               │   │   ├── spike_times: float64[N]
│               │   │   ├── sta_section: float64[T,H,W]
│               │   │   └── ...
│               │   ├── direction_45/
│               │   └── ...
│
├── metadata/
│   ├── frame_time: float64[]
│   ├── frame_timestamps: float64[]
│   ├── cmcr_meta/
│   │   ├── recording_date: str
│   │   ├── electrode_count: int64
│   │   └── ...
│   ├── cmtr_meta/
│   │   ├── threshold: float64
│   │   └── ...
│   ├── gsheet_row/
│   │   ├── {column_name}: various types
│   │   └── ...
│   ├── rf_sta_geometry/
│   │   ├── summary_stats: float64[N]
│   │   └── ...
│   └── ap_tracking/
│       ├── all_ap_intersection/
│       │   ├── x: float64
│       │   ├── y: float64
│       │   ├── method: str
│       │   ├── rmse: float64
│       │   └── n_cells_used: int64
│       └── ...
│
├── stimulus/
│   ├── light_reference: float64[N]
│   ├── frame_times: float64[N]
│   ├── light_template: float64[...]
│   └── section_time/
│       └── {playlist_name}/
│           └── {movie_name}/
│               ├── start_frame: int64
│               ├── end_frame: int64
│               ├── start_time: float64
│               └── end_time: float64
│
├── source_files/
│   └── [attrs]
│       ├── cmcr_path: str
│       └── cmtr_path: str
│
└── pipeline/
    ├── [attrs]
    │   ├── stage1_completed: bool
    │   └── stage1_timestamp: str
    └── session_info/
        ├── [attrs]
        │   ├── saved_at: str
        │   └── checkpoint_name: str (optional)
        ├── completed_steps: str[]
        └── warnings: str[]
```

---

## Entity Relationships

```
┌─────────────────┐
│ PipelineSession │
└────────┬────────┘
         │
         │ contains
         ▼
    ┌────┴────┬───────────┬──────────────┐
    │         │           │              │
    ▼         ▼           ▼              ▼
┌───────┐ ┌────────┐ ┌──────────┐ ┌────────────┐
│ Units │ │Metadata│ │ Stimulus │ │Source Files│
└───┬───┘ └────────┘ └──────────┘ └────────────┘
    │
    │ 1:N
    ▼
┌───────┐
│ Unit  │
└───┬───┘
    │
    │ 1:N
    ▼
┌───────────┐
│FeatureData│
└───────────┘
```

---

## Validation Rules

| Entity | Rule | Enforcement |
|--------|------|-------------|
| `dataset_id` | Non-empty string | Validated in `__post_init__` |
| `unit_id` | Must match pattern `unit_\d{3}` | Validated on add |
| `spike_times` | Monotonically increasing | Validated on load |
| `save(overwrite)` | Default `False`; raises `FileExistsError` if file exists | Checked before write |
| `completed_steps` | Step names must be unique | Set data structure |

---

## State Transitions

```
                  create_session()
                        │
                        ▼
              ┌─────────────────┐
              │    DEFERRED     │◄──────────────────┐
              │  (in-memory)    │                   │
              └────────┬────────┘                   │
                       │                            │
        ┌──────────────┼──────────────┐            │
        │              │              │            │
        ▼              ▼              ▼            │
   save()        checkpoint()   load_from_hdf5()   │
        │              │              │            │
        ▼              │              │            │
   ┌─────────┐         │              │            │
   │  SAVED  │         │              │            │
   └─────────┘         │              │            │
                       ▼              │            │
              (remains DEFERRED)      │            │
                       │              │            │
                       └──────────────┴────────────┘
                              (can continue processing)
```

