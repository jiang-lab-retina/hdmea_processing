# Quickstart: JSON-Based Spike Sectioning

**Branch**: `008-json-spike-sectioning` | **Date**: 2024-12-18

## Overview

This feature modifies spike time sectioning to use JSON configuration files from `config/stimuli/` instead of calculating trial boundaries from the `trial_repeats` parameter.

## Prerequisites

1. HDF5 file with:
   - `units/{unit_id}/spike_times` datasets
   - `stimulus/section_time/{movie_name}` datasets
   - `metadata/frame_timestamps` array

2. JSON config files for each movie in `config/stimuli/{movie_name}.json`

## JSON Config Format

Each stimulus JSON file must contain a `section_kwargs` object:

```json
{
    "name": "moving_h_bar_s5_d8_3x",
    "section_kwargs": {
        "start_frame": 60,
        "trial_length_frame": 4520,
        "repeat": 3
    }
}
```

| Field | Description |
|-------|-------------|
| `start_frame` | First trial start frame (relative to movie content start) |
| `trial_length_frame` | Duration of each trial in display frames |
| `repeat` | Number of trial repetitions |

## Usage

```python
from hdmea.io.spike_sectioning import section_spike_times

# Basic usage (uses default config/stimuli/ directory)
result = section_spike_times(
    hdf5_path="artifacts/JIANG009_2025-04-10.h5",
    pad_margin=(2.0, 0.0),  # 2s pre-margin
)

# With custom config directory
result = section_spike_times(
    hdf5_path="artifacts/recording.h5",
    config_dir="path/to/custom/configs/",
)

# Check result
print(f"Processed {result.units_processed} units")
print(f"Movies: {result.movies_processed}")
```

## Validation

The function validates all JSON configs before processing:

```python
# This will fail with informative error if any movie lacks config
try:
    result = section_spike_times("recording.h5")
except ValueError as e:
    print(f"Config validation failed: {e}")
    # Error lists all missing/invalid configs
```

## Trial Boundary Calculation

For each trial `n` (0-indexed):

```
trial_start_frame = section_frame_start + PRE_MARGIN_FRAME_NUM + start_frame + (n * trial_length_frame)
trial_end_frame = trial_start_frame + trial_length_frame
trial_start_sample = frame_timestamps[trial_start_frame]
```

Where:
- `section_frame_start` = frame number from HDF5 section_time[0, 0]
- `PRE_MARGIN_FRAME_NUM` = 60 (constant)
- `start_frame`, `trial_length_frame` = from JSON config

## Output Structure

Output is unchanged from existing implementation:

```
units/{unit_id}/spike_times_sectioned/{movie_name}/
├── full_spike_times          # All trials combined (int64[])
└── trials_spike_times/
    ├── 0                     # Trial 0 spikes (int64[])
    ├── 1                     # Trial 1 spikes (int64[])
    └── ...
```

## Migration Notes

- The `trial_repeats` parameter is now **deprecated** and ignored
- Trial count is determined by JSON config `repeat` value
- All movies must have corresponding JSON config files (fail-fast validation)

