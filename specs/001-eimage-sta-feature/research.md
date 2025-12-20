# Research: Electrode Image STA (eimage_sta)

**Date**: 2025-12-19  
**Feature**: 001-eimage-sta-feature

## Overview

This document captures research findings for implementing the eimage_sta feature with intensive performance optimization.

---

## 1. Vectorized High-Pass Filtering

### Decision
Use `scipy.signal.filtfilt()` with `axis=0` parameter to apply Butterworth high-pass filter to entire 3D sensor array at once.

### Rationale
- Legacy code uses nested loops over 64×64 electrodes (4,096 iterations), each calling filtfilt
- scipy's filtfilt supports axis parameter for vectorized operation across all electrodes simultaneously
- Expected speedup: 100-1000x (eliminates Python loop overhead, enables SIMD vectorization)

### Implementation Pattern

```python
from scipy.signal import butter, filtfilt

def apply_highpass_filter_3d(
    sensor_data: np.ndarray,  # Shape: (time, rows, cols)
    cutoff_hz: float,
    sampling_rate: float,
    order: int = 2,
) -> np.ndarray:
    """Apply high-pass filter along time axis for all electrodes at once."""
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    
    # Apply along axis=0 (time dimension) - processes ALL electrodes in one call
    filtered = filtfilt(b, a, sensor_data, axis=0)
    return filtered.astype(np.float32)
```

### Alternatives Considered
1. **Per-electrode multiprocessing**: Moderate speedup but high memory overhead from data copies
2. **GPU/CuPy**: Fastest for very large data but adds dependency, not needed for target performance
3. **Chunked processing**: More complex, needed only if memory-constrained

---

## 2. Vectorized Spike Window Extraction

### Decision
Use NumPy fancy indexing to extract all spike windows in a single operation per unit.

### Rationale
- Legacy code loops over each spike, extracting one window at a time
- NumPy advanced indexing can extract all windows at once: `sensor_data[all_indices]`
- Memory overhead acceptable: 10,000 spikes × 50 samples × 64×64 = ~1.3 GB (float32)

### Implementation Pattern

```python
def compute_eimage_sta(
    filtered_data: np.ndarray,  # Shape: (time, rows, cols)
    spike_samples: np.ndarray,  # Shape: (n_spikes,)
    pre_samples: int = 10,
    post_samples: int = 40,
) -> tuple[np.ndarray, int, int]:
    """Compute eimage STA using vectorized window extraction."""
    window_length = pre_samples + post_samples
    n_samples = filtered_data.shape[0]
    
    # Filter valid spikes (edge handling)
    valid_mask = (
        (spike_samples >= pre_samples) &
        (spike_samples + post_samples <= n_samples)
    )
    valid_spikes = spike_samples[valid_mask]
    n_used = len(valid_spikes)
    n_excluded = len(spike_samples) - n_used
    
    if n_used == 0:
        return np.full((window_length, *filtered_data.shape[1:]), np.nan, dtype=np.float32), 0, n_excluded
    
    # Build all window indices at once: (n_spikes, window_length)
    window_offsets = np.arange(-pre_samples, post_samples)
    all_indices = valid_spikes[:, np.newaxis] + window_offsets
    
    # Extract all windows: (n_spikes, window_length, rows, cols)
    windows = filtered_data[all_indices]
    
    # Average across spikes
    sta = windows.mean(axis=0).astype(np.float32)
    
    return sta, n_used, n_excluded
```

### Alternatives Considered
1. **Sequential loop**: Simple but slow for high spike counts
2. **Parallel per-spike**: Complex, memory-intensive, minimal benefit over vectorized

---

## 3. Memory-Mapped Sensor Data Access

### Decision
Use HDF5 memory-mapped access via h5py with `driver='mpio'` or standard chunked access.

### Rationale
- Sensor data for 120s at 20kHz × 64×64 × int16 = ~20 GB
- Loading entirely into RAM may fail on systems with <32GB RAM
- Memory-mapped access lets OS manage paging, accessing only needed regions

### Implementation Pattern

```python
import h5py

def load_sensor_data_mmap(cmcr_path: str, duration_samples: int) -> np.ndarray:
    """Load sensor data with memory-mapped access."""
    from McsPy.McsCMOSMEA import McsCMOSMEAData
    
    cmcr = McsCMOSMEAData(cmcr_path)
    sensor_data = cmcr.Acquisition.Sensor_Data.SensorData_1_1
    
    # Access as memory-mapped array (returns view, not copy)
    return sensor_data[:duration_samples, :, :]
```

### Note on CMCR Access
McsPy's `McsCMOSMEAData` provides lazy access to sensor data. The actual data is read on-demand when sliced. For filtered data caching, we write the filtered result to a temporary HDF5 file.

### Filtered Data Cache Pattern

```python
def get_or_create_filtered_cache(
    cmcr_path: str,
    cache_path: str,
    cutoff_hz: float,
    sampling_rate: float,
    duration_samples: int,
) -> np.ndarray:
    """Load from cache or filter and cache sensor data."""
    cache_key = f"filtered_{cutoff_hz}hz"
    
    if Path(cache_path).exists():
        with h5py.File(cache_path, 'r') as f:
            if cache_key in f:
                return f[cache_key][:]
    
    # Load and filter
    sensor_data = load_sensor_data_mmap(cmcr_path, duration_samples)
    filtered = apply_highpass_filter_3d(sensor_data, cutoff_hz, sampling_rate)
    
    # Cache result
    with h5py.File(cache_path, 'a') as f:
        if cache_key in f:
            del f[cache_key]
        f.create_dataset(cache_key, data=filtered, compression='gzip')
    
    return filtered
```

---

## 4. Performance Target Analysis

### Target
Complete computation for typical recording (120s, ~100 units) in under 5 minutes.

### Breakdown

| Phase | Legacy Time | Optimized Estimate | Notes |
|-------|-------------|-------------------|-------|
| Load sensor data | ~1 min | ~30 sec | Memory-mapped, lazy access |
| High-pass filter | ~60 min | ~30 sec | Vectorized (1000x speedup) |
| STA per unit | ~10 sec/unit | ~0.5 sec/unit | Vectorized window extraction |
| Total (100 units) | ~70 min | ~2-3 min | Well under 5 min target |

### Verification Approach
1. Time each phase separately using `time.perf_counter()`
2. Log timing for filter operation
3. Log average time per unit
4. Integration test asserts total time < 5 minutes

---

## 5. HDF5 Output Schema

### Decision
Store eimage_sta in `units/{unit_id}/features/eimage_sta/data` with metadata as attributes.

### Schema

```
units/
└── {unit_id}/
    └── features/
        └── eimage_sta/
            ├── data           # Dataset: float32, shape (window_length, rows, cols)
            └── [attributes]
                ├── n_spikes           # int: Number of spikes used
                ├── n_spikes_excluded  # int: Spikes excluded (edge effects)
                ├── pre_samples        # int: Samples before spike
                ├── post_samples       # int: Samples after spike
                ├── cutoff_hz          # float: High-pass filter cutoff
                ├── filter_order       # int: Butterworth filter order
                ├── sampling_rate      # float: Acquisition rate in Hz
                ├── spike_limit        # int: Max spikes used (-1 if no limit)
                └── version            # str: Extractor version
```

---

## 6. Integration with Existing STA

### Relationship
- **Existing `sta.py`**: Computes Spike-Triggered Average from visual stimulus movie (receptive field mapping)
- **New `eimage_sta`**: Computes Spike-Triggered Average from electrode array data (axonal footprint mapping)

### Code Reuse
- Both use similar window extraction pattern - can share `_compute_sta_for_unit` logic
- Both use similar HDF5 write pattern - can share `_write_feature_to_hdf5` helper
- Different data sources: stimulus movie vs. sensor data

---

## Summary of Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Filter strategy | Vectorized (axis=0) | 1000x speedup over loops |
| Window extraction | NumPy fancy indexing | Single operation for all spikes |
| Data loading | Memory-mapped via McsPy | Handles large files, OS-managed caching |
| Cache strategy | Optional HDF5 cache | 2x speedup for repeated runs |
| Output format | HDF5 with attributes | Consistent with existing features |
| Performance target | <5 minutes | Achievable with vectorization |

