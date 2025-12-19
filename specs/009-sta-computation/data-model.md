# Data Model: STA Computation

**Feature**: 009-sta-computation  
**Date**: 2025-12-18

## Entities

### STAResult

Represents the computed Spike Triggered Average for a single unit.

| Field | Type | Description |
|-------|------|-------------|
| unit_id | str | Unit identifier (e.g., "unit_000") |
| movie_name | str | Noise movie name used for computation |
| sta | ndarray[float32] | 3D STA array (time × height × width) |
| n_spikes_used | int | Number of valid spikes included in average |
| n_spikes_excluded | int | Number of spikes excluded due to edge effects |
| cover_range | tuple[int, int] | Frame range used for averaging |

### STAComputationConfig

Configuration for STA computation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| cover_range | tuple[int, int] | (-60, 0) | Frame window relative to spike |
| use_multiprocessing | bool | True | Enable parallel processing |
| n_workers | int | None | CPU cores to use (None = 80% of available) |
| stimuli_dir | Path | See spec | Directory containing .npy stimulus files |

### SpikeFrameMapping

Intermediate data for spike time conversion.

| Field | Type | Description |
|-------|------|-------------|
| spike_samples | ndarray[int64] | Original spike times in sampling indices |
| spike_frames | ndarray[int64] | Converted spike times in movie frames |
| samples_per_frame | float | Conversion factor (acquisition_rate / frame_rate) |

## HDF5 Schema Changes

### New Groups/Datasets

```
units/{unit_id}/
└── features/
    └── {noise_movie_name}/
        ├── sta                 # Dataset: float32, shape (time, height, width)
        └── sta_metadata/       # Group with attributes
            ├── n_spikes_used       # int
            ├── n_spikes_excluded   # int
            ├── cover_range         # (int, int)
            └── computed_at         # ISO 8601 timestamp
```

### Dataset Attributes

The `sta` dataset stores these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| dtype_warning | bool | True if stimulus was not uint8 |
| original_dtype | str | Original stimulus dtype name |
| cover_range | (int, int) | Frame range used |
| n_spikes | int | Number of spikes in average |

## Relationships

```
┌─────────────────┐
│   HDF5 File     │
└────────┬────────┘
         │ contains
         ▼
┌─────────────────┐     uses      ┌──────────────────────┐
│   units/{id}/   │◄─────────────►│ spike_times_sectioned│
│   features/     │               │ /{movie}/trials_*/   │
└────────┬────────┘               └──────────────────────┘
         │ produces
         ▼
┌─────────────────┐     loaded    ┌──────────────────────┐
│  {movie}/sta    │◄─────from─────│ Stimulus .npy file   │
└─────────────────┘               └──────────────────────┘
```

## Validation Rules

1. **cover_range**: `cover_range[0] < cover_range[1]` (start must be before end)
2. **movie_name**: Must contain "noise" (case-insensitive), exactly one match
3. **spike_frames**: Must be within valid range `[0, movie_length)` after adding cover_range offsets
4. **sta shape**: Must match `(abs(cover_range[1] - cover_range[0]), movie_height, movie_width)`

## State Transitions

```
┌─────────────────┐
│   Not Computed  │
└────────┬────────┘
         │ compute_sta()
         ▼
┌─────────────────┐
│   Computing...  │──── on failure ────► [Error logged, partial results]
└────────┬────────┘
         │ success
         ▼
┌─────────────────┐
│    Computed     │
└────────┬────────┘
         │ force=True
         ▼
┌─────────────────┐
│   Recomputed    │
└─────────────────┘
```

