# Quickstart: Spike Times Unit Conversion and Stimulation Sectioning

**Feature Branch**: `006-spike-times-sectioning`  
**Date**: 2025-12-17  
**Status**: 📋 PLANNED (pending implementation)

## Overview

This feature provides:
1. Automatic conversion of spike timestamps to acquisition sample units during data loading
2. A pipeline step to section spike times by stimulation periods, storing BOTH:
   - `full_spike_times` - all trials combined
   - `trials_spike_times/{idx}` - per-trial split

## Usage (Post-Implementation)

### 1. Loading Data (Automatic Conversion)

```python
from hdmea.pipeline import load_recording

# Spike times are automatically converted to sample units
result = load_recording(
    cmcr_path="path/to/recording.cmcr.h5",
    cmtr_path="path/to/recording.cmtr.h5",
    dataset_id="JIANG009_2025-04-10",
)

# Access spike times (now in sample units, not nanoseconds)
import zarr
root = zarr.open(result.zarr_path, mode="r")
unit = root["units/unit_000"]
spike_times = unit["spike_times"][:]  # Sample indices at ~20 kHz

print(f"Spike count: {len(spike_times)}")
print(f"First spike at sample: {spike_times[0]}")
```

### 2. Sectioning Spike Times by Stimulation

```python
from hdmea.io.spike_sectioning import section_spike_times
from hdmea.io.section_time import add_section_time

# First, ensure section_time exists
add_section_time(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    playlist_name="playlist_set6a",
)

# Section spike times by stimulation periods
result = section_spike_times(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    trial_repeats=3,            # Process first 3 trials (default)
    pad_margin=(2.0, 0.0),      # 2s pre-margin, 0s post-margin (default)
    force=False,
)

print(f"Processed {result.units_processed} units")
print(f"Movies: {result.movies_processed}")
print(f"Trials per movie: {result.trial_repeats}")
print(f"Padding: {result.pre_samples} pre-samples, {result.post_samples} post-samples")
```

### 3. Accessing Sectioned Spike Times

```python
import zarr
import numpy as np

root = zarr.open("artifacts/JIANG009_2025-04-10.zarr", mode="r")
unit = root["units/unit_000"]

# List available movies
sectioned = unit["spike_times_sectioned"]
movies = list(sectioned.keys())
print(f"Movies with sectioned data: {movies}")

# Access COMBINED spikes (all trials)
movie_name = "movie_A"
full_spikes = unit[f"spike_times_sectioned/{movie_name}/full_spike_times"][:]
print(f"{movie_name} full_spike_times: {len(full_spikes)} total spikes")

# Access PER-TRIAL spikes
trials_group = unit[f"spike_times_sectioned/{movie_name}/trials_spike_times"]
n_trials = len(trials_group.keys())
print(f"Number of trials: {n_trials}")

for trial_idx in range(n_trials):
    trial_spikes = trials_group[str(trial_idx)][:]
    print(f"  Trial {trial_idx}: {len(trial_spikes)} spikes")
```

### 4. Converting Between Units

```python
# Get acquisition rate
acquisition_rate = root["metadata/acquisition_rate"][0]  # e.g., 20000.0 Hz

# Sample index → time in seconds
time_seconds = spike_times / acquisition_rate

# Sample index → time in milliseconds
time_ms = spike_times / acquisition_rate * 1000
```

### 5. Computing PSTH Using Per-Trial Data

```python
import numpy as np

# Get section_time for trial boundaries
section_time = root["stimulus/section_time/movie_A"][:]
acquisition_rate = root["metadata/acquisition_rate"][0]

# Collect trial-relative spike times
trials_group = unit["spike_times_sectioned/movie_A/trials_spike_times"]
all_relative_ms = []

for trial_idx in range(len(trials_group.keys())):
    trial_spikes = trials_group[str(trial_idx)][:]
    trial_start = section_time[trial_idx, 0]
    
    # Convert to trial-relative milliseconds
    relative_ms = (trial_spikes - trial_start) / acquisition_rate * 1000
    all_relative_ms.extend(relative_ms)

# PSTH histogram
bin_size_ms = 10
max_time_ms = 5000  # 5 seconds
bins = np.arange(0, max_time_ms + bin_size_ms, bin_size_ms)
psth, _ = np.histogram(all_relative_ms, bins=bins)
n_trials = len(trials_group.keys())
psth = psth / n_trials / (bin_size_ms / 1000)  # spikes/s
```

## Data Structure

```
artifacts/JIANG009_2025-04-10.zarr/
├── metadata/
│   ├── acquisition_rate    # 20000.0 Hz
│   └── sample_interval     # 0.00005 s
├── stimulus/
│   └── section_time/
│       ├── movie_A         # (N_trials, 2) - [start, end] sample indices
│       └── movie_B
└── units/
    └── unit_000/
        ├── spike_times             # (N,) uint64 - full recording (sample indices)
        └── spike_times_sectioned/  # Group
            └── movie_A/
                ├── full_spike_times     # (M,) int64 - ALL trials combined
                └── trials_spike_times/  # Group
                    ├── 0                # (K0,) int64 - trial 0 spikes
                    ├── 1                # (K1,) int64 - trial 1 spikes
                    └── 2                # (K2,) int64 - trial 2 spikes
```

## Test Dataset

Use `artifacts/JIANG009_2025-04-10.zarr` for validation and testing.

## Verification

```python
# Verify spike times are in sample units (not nanoseconds)
spike_times = unit["spike_times"][:]
print(f"Max spike time: {spike_times.max()}")
# Expected: ~millions (samples) not ~trillions (nanoseconds)

# Verify full_spike_times equals union of all trials
full_spikes = set(unit["spike_times_sectioned/movie_A/full_spike_times"][:])
trial_spikes = set()
trials_group = unit["spike_times_sectioned/movie_A/trials_spike_times"]
for trial_idx in trials_group.keys():
    trial_spikes.update(trials_group[trial_idx][:])
assert full_spikes == trial_spikes, "full_spike_times should equal union of all trials"

# Verify spikes are within section_time boundaries (with padding)
section_time = root["stimulus/section_time/movie_A"][:]
# Note: spikes may extend beyond boundaries due to pad_margin
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Very large spike_times values (~10^12) | Old data in nanoseconds | Re-run `load_recording(force=True)` |
| `MissingInputError: section_time` | No section_time data | Run `add_section_time()` first |
| `FileExistsError` | Sectioned data already exists | Use `force=True` to overwrite |
| Fewer trials than expected | `trial_repeats` > available trials | section_time has fewer trials; adjust `trial_repeats` |
