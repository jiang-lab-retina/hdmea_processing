# Quickstart: STA Computation

**Feature**: 009-sta-computation  
**Date**: 2025-12-18

## Prerequisites

1. HDF5 recording with sectioned spike times (run spike sectioning first)
2. Noise stimulus movie file (`.npy`) in the stimuli directory
3. Recording must have exactly one movie with "noise" in its name

## Basic Usage

```python
from hdmea.features import compute_sta

# Compute STA with default settings
result = compute_sta("artifacts/2025.04.10-11.12.57-Rec.h5")

print(f"Movie: {result.movie_name}")
print(f"Units processed: {result.units_processed}")
print(f"Units failed: {result.units_failed}")
print(f"Time: {result.elapsed_seconds:.1f}s")
```

## Configuration Options

### Custom Cover Range

```python
# Use 30 frames before spike instead of 60
result = compute_sta(
    "artifacts/recording.h5",
    cover_range=(-30, 0)
)

# Include frames after spike (e.g., for response analysis)
result = compute_sta(
    "artifacts/recording.h5",
    cover_range=(-30, 30)  # 30 before, 30 after
)
```

### Disable Multiprocessing

```python
# Sequential processing (useful for debugging)
result = compute_sta(
    "artifacts/recording.h5",
    use_multiprocessing=False
)
```

### Custom Stimuli Directory

```python
from pathlib import Path

result = compute_sta(
    "artifacts/recording.h5",
    stimuli_dir=Path("D:/stimuli/noise_movies")
)
```

### Force Recomputation

```python
# Overwrite existing STA results
result = compute_sta(
    "artifacts/recording.h5",
    force=True
)
```

## Reading Results

```python
import h5py
import numpy as np

with h5py.File("artifacts/recording.h5", "r") as f:
    # List all units
    units = list(f["units"].keys())
    
    for unit_id in units:
        # Check if STA exists for this unit
        sta_path = f"units/{unit_id}/features"
        if sta_path in f:
            for movie_name in f[sta_path].keys():
                sta = f[f"{sta_path}/{movie_name}/sta"][:]
                print(f"{unit_id}: STA shape = {sta.shape}")
                
                # Access metadata
                n_spikes = f[f"{sta_path}/{movie_name}/sta"].attrs["n_spikes"]
                print(f"  Used {n_spikes} spikes")
```

## Visualization

```python
import matplotlib.pyplot as plt
import h5py

with h5py.File("artifacts/recording.h5", "r") as f:
    sta = f["units/unit_000/features/noise_movie/sta"][:]

# Plot temporal evolution of STA
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    frame_idx = i * (sta.shape[0] // 10)
    ax.imshow(sta[frame_idx], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"Frame {frame_idx}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Common Issues

### No noise movie found

```
ValueError: No noise movie found in spike_times_sectioned
```

**Solution**: Ensure your HDF5 has sectioned spike times with a movie containing "noise" in its name.

### Multiple noise movies found

```
ValueError: Multiple noise movies found: ['noise_15hz', 'noise_30hz']
```

**Solution**: Currently the system requires exactly one noise movie. Process recordings separately or rename movies.

### Stimulus file not found

```
FileNotFoundError: Stimulus file not found: M:\...\noise_movie.npy
```

**Solution**: Ensure the `.npy` file exists and the `stimuli_dir` path is correct.

## Test File

For validation, use the test file specified in the spec:

```python
result = compute_sta("artifacts/2025.04.10-11.12.57-Rec.h5")
```

