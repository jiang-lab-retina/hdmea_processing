# Research: STA Computation

**Feature**: 009-sta-computation  
**Date**: 2025-12-18

## 1. Shared Memory for Multiprocessing

**Decision**: Use `multiprocessing.shared_memory.SharedMemory` to share the stimulus array across worker processes.

**Rationale**: 
- The stimulus array can be large (e.g., 15x15x100k frames = ~22MB for uint8)
- Loading once and sharing avoids redundant I/O and memory duplication
- Python 3.8+ provides `SharedMemory` in the standard library

**Alternatives Considered**:
- `multiprocessing.Array`: Requires ctypes, less numpy-friendly
- Memory-mapped files (`np.memmap`): More complex, disk I/O overhead
- Per-worker loading: High memory usage, redundant I/O

**Implementation Pattern**:
```python
from multiprocessing import shared_memory
import numpy as np

# Create shared memory
shm = shared_memory.SharedMemory(create=True, size=movie_array.nbytes)
shared_arr = np.ndarray(movie_array.shape, dtype=movie_array.dtype, buffer=shm.buf)
shared_arr[:] = movie_array[:]

# In workers: attach to existing shared memory
shm_worker = shared_memory.SharedMemory(name=shm.name)
arr = np.ndarray(shape, dtype=dtype, buffer=shm_worker.buf)
```

## 2. Spike Time to Frame Conversion

**Decision**: Convert spike times (sampling indices at acquisition rate) to frame numbers using rounding to nearest frame.

**Rationale**:
- Spike times are recorded at acquisition rate (e.g., 20kHz)
- Movie frames are displayed at lower rate (e.g., 15Hz)
- Rounding to nearest frame minimizes timing error

**Formula**:
```python
frame_number = round(spike_sample / samples_per_frame)
# where samples_per_frame = acquisition_rate / frame_rate
```

**Alternatives Considered**:
- Floor: Biases toward earlier frames
- Ceiling: Biases toward later frames
- Linear interpolation: Overcomplicated for integer frame indices

## 3. Vectorized Window Extraction

**Decision**: Pre-compute all valid spike indices, then use numpy fancy indexing for batch extraction.

**Rationale**:
- Python loops are slow (~100x slower than vectorized ops)
- Pre-filtering invalid spikes avoids per-spike bounds checking in inner loop
- Fancy indexing allows single numpy call for all windows

**Implementation Pattern**:
```python
# Pre-compute valid indices
valid_mask = (spikes + cover_range[0] >= 0) & (spikes + cover_range[1] < movie_length)
valid_spikes = spikes[valid_mask]

# Build window indices (n_spikes, window_length)
window_offsets = np.arange(cover_range[0], cover_range[1])
all_indices = valid_spikes[:, np.newaxis] + window_offsets

# Extract all windows at once
windows = movie_array[all_indices]  # Shape: (n_spikes, window_length, height, width)

# Compute STA
sta = windows.mean(axis=0)
```

## 4. Progress Bar with Multiprocessing

**Decision**: Use `tqdm` with `multiprocessing.Pool.imap` for progress tracking.

**Rationale**:
- `tqdm` is already a project dependency (per assumptions in spec)
- `imap` allows iteration over results as they complete
- Progress bar updates per-unit completion

**Implementation Pattern**:
```python
from multiprocessing import Pool
from tqdm import tqdm

with Pool(n_workers) as pool:
    results = list(tqdm(
        pool.imap(compute_sta_for_unit, unit_args),
        total=len(unit_args),
        desc="Computing STA"
    ))
```

## 5. Error Handling with Retry

**Decision**: Wrap per-unit computation in try/except, retry once on failure, log warning and continue on second failure.

**Rationale**:
- Batch processing should not abort entirely for one bad unit
- Single retry handles transient issues (memory pressure, etc.)
- Logging failed units enables post-hoc debugging

**Implementation Pattern**:
```python
def compute_sta_with_retry(unit_id, *args, max_retries=1):
    for attempt in range(max_retries + 1):
        try:
            return compute_sta_for_unit(unit_id, *args)
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"Unit {unit_id} failed after {max_retries+1} attempts: {e}")
                return None  # Partial result
            logger.debug(f"Unit {unit_id} attempt {attempt+1} failed, retrying...")
```

## 6. Noise Movie Detection

**Decision**: Case-insensitive search for "noise" substring in movie names under `spike_times_sectioned`.

**Rationale**:
- Simple and robust pattern matching
- Case-insensitive handles variations like "Noise", "NOISE", "noise"
- Strict validation (exactly one match) prevents ambiguity

**Implementation Pattern**:
```python
def find_noise_movie(unit_group: h5py.Group) -> str:
    movies = list(unit_group["spike_times_sectioned"].keys())
    noise_movies = [m for m in movies if "noise" in m.lower()]
    
    if len(noise_movies) == 0:
        raise ValueError("No noise movie found")
    if len(noise_movies) > 1:
        raise ValueError(f"Multiple noise movies found: {noise_movies}")
    
    return noise_movies[0]
```

## 7. HDF5 Storage Location

**Decision**: Store STA in `units/{unit_id}/features/{movie_name}/sta` as a 3D array.

**Rationale**:
- Follows existing HDF5 structure with features under units
- Movie name in path enables multiple STA computations if needed
- 3D array preserves spatial structure (height × width × time)

**Alternatives Considered**:
- Separate features file: Fragments data, complicates access
- Root-level features group: Loses per-unit organization
- Flattened 1D array: Loses spatial structure

## Summary

All technical decisions resolved. Ready for Phase 1 design artifacts.

