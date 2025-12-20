# Quick Start: Electrode Image STA (eimage_sta)

**Date**: 2025-12-19  
**Feature**: 001-eimage-sta-feature

## Overview

Compute the Electrode Image Spike-Triggered Average (eimage_sta) to visualize how electrical activity propagates across the HD-MEA electrode array around each unit's spikes.

## Basic Usage

```python
from hdmea.features.eimage_sta import compute_eimage_sta

# Compute eimage_sta for all units in an HDF5 file
result = compute_eimage_sta(
    hdf5_path="artifacts/2025.04.10-11.12.57-Rec.h5",
    cmcr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr",
)

print(f"Processed {result.units_processed} units in {result.elapsed_seconds:.1f}s")
```

## With Custom Parameters

```python
result = compute_eimage_sta(
    hdf5_path="artifacts/recording.h5",
    cmcr_path="O:/data/recording.cmcr",
    # Filter settings
    cutoff_hz=100.0,        # High-pass filter cutoff (Hz)
    filter_order=2,         # Butterworth filter order
    # Time window
    pre_samples=10,         # Samples before spike
    post_samples=40,        # Samples after spike
    # Performance
    spike_limit=10000,      # Max spikes per unit
    duration_s=120.0,       # Seconds of sensor data to use
    # Cache (optional)
    use_cache=True,         # Cache filtered data
    # Overwrite
    force=False,            # Skip existing, set True to overwrite
)
```

## Reading Results

```python
import h5py

with h5py.File("artifacts/recording.h5", "r") as f:
    for unit_id in f["units"].keys():
        eimage = f[f"units/{unit_id}/features/eimage_sta/data"][:]
        n_spikes = f[f"units/{unit_id}/features/eimage_sta"].attrs["n_spikes"]
        
        print(f"Unit {unit_id}: shape={eimage.shape}, n_spikes={n_spikes}")
        # Output: Unit 1: shape=(50, 64, 64), n_spikes=8542
```

## Visualizing Results

```python
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("artifacts/recording.h5", "r") as f:
    eimage = f["units/1/features/eimage_sta/data"][:]

# Plot frame at spike time (sample 10, since pre_samples=10)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Before spike
axes[0].imshow(eimage[5], cmap='RdBu_r', vmin=-50, vmax=50)
axes[0].set_title("5 samples before spike")

# At spike
axes[1].imshow(eimage[10], cmap='RdBu_r', vmin=-50, vmax=50)
axes[1].set_title("At spike")

# After spike
axes[2].imshow(eimage[20], cmap='RdBu_r', vmin=-50, vmax=50)
axes[2].set_title("10 samples after spike")

plt.tight_layout()
plt.show()
```

## Test Data

For testing, use these paths:

```python
cmcr_path = "O:\\20250410\\set6\\2025.04.10-11.12.57-Rec.cmcr"
cmtr_path = "O:\\20250410\\set6\\2025.04.10-11.12.57-Rec-.cmtr"
hdf5_path = "artifacts/2025.04.10-11.12.57-Rec.h5"
```

## Performance Notes

- **First run**: ~2-3 minutes (includes filtering)
- **With cache**: ~1 minute (skips filtering)
- **Memory**: Uses memory-mapped access, works on systems with 16GB+ RAM
- **Scaling**: Linear with number of units (~0.5s per unit)

## Common Issues

### Missing Sensor Data
```
DataLoadError: Sensor data not found in CMCR file
```
→ Ensure CMCR file contains sensor data (some recordings may only have analog/light reference)

### Memory Errors
```
MemoryError: Unable to allocate...
```
→ Reduce `spike_limit` or `duration_s` parameters

### Slow Performance
→ Enable caching with `use_cache=True` for repeated runs on same data

