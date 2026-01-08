---
name: Trace Preprocessing Pipeline
overview: ""
todos:
  - id: update-config-columns
    content: Update config.py to use corrected_moving_h_bar_s5_d8_3x_{angle:03d} columns
    status: completed
  - id: add-preprocess-function
    content: Add preprocess_trace() function with 10 Hz low-pass filter and resampling to data_loader.py
    status: completed
  - id: integrate-preprocessing
    content: Call preprocess_trace() in compute_mean_trace() after computing trial mean
    status: completed
---

# Add Low-Pass Filtering, Resampling, and Corrected Direction Traces

## Overview

Modify the subgroup clustering data loader to preprocess all traces with a 10 Hz low-pass filter, resample from 60 Hz to 10 Hz, and use corrected direction columns for moving bar stimuli.

## Key Files

- [`dataframe_phase/classification/subgroup_clustering/config.py`](dataframe_phase/classification/subgroup_clustering/config.py) - Update column names
- [`dataframe_phase/classification/subgroup_clustering/data_loader.py`](dataframe_phase/classification/subgroup_clustering/data_loader.py) - Add filtering/resampling

## Implementation

### 1. Update config.py: Use Corrected Direction Columns

Replace the 8 direction columns with their corrected versions (using 3-digit angle format):

```python
# Current (uncorrected)
"moving_h_bar_s5_d8_3x_0"   -> "corrected_moving_h_bar_s5_d8_3x_000"
"moving_h_bar_s5_d8_3x_45"  -> "corrected_moving_h_bar_s5_d8_3x_045"
# ... etc for all 8 directions
```



### 2. Update data_loader.py: Add Preprocessing

Add a `preprocess_trace()` function that:

1. Applies a 10 Hz Bessel low-pass filter (using `scipy.signal.bessel` + `filtfilt`)
2. Resamples from 60 Hz to 10 Hz (downsample by factor of 6)

This will be applied in `compute_mean_trace()` after averaging trials.

```python
from scipy import signal

# Add constants
ORIGINAL_SAMPLE_RATE = 60.0  # Hz
TARGET_SAMPLE_RATE = 10.0    # Hz
LOWPASS_CUTOFF = 10.0        # Hz
FILTER_ORDER = 5

def preprocess_trace(trace: np.ndarray) -> np.ndarray:
    """Apply 10 Hz low-pass filter and resample to 10 Hz."""
    # Low-pass filter
    nyquist = 0.5 * ORIGINAL_SAMPLE_RATE
    normalized_cutoff = LOWPASS_CUTOFF / nyquist
    b, a = signal.bessel(FILTER_ORDER, normalized_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, trace)
    
    # Resample: 60 Hz -> 10 Hz (take every 6th sample)
    downsample_factor = int(ORIGINAL_SAMPLE_RATE / TARGET_SAMPLE_RATE)
    resampled = filtered[::downsample_factor]
    return resampled
```

Call `preprocess_trace()` in `compute_mean_trace()` before returning the mean trace.

## Impact

- Trace length will reduce by 6x (e.g., 599 frames at 60 Hz becomes ~100 frames at 10 Hz)