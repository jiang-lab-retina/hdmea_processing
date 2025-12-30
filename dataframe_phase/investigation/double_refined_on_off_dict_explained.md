# Double-Refined ON/OFF Dictionary Generation

## Overview

The double-refined method creates `moving_h_bar_s5_d8_3x_on_off_dict_hd_double_refined.pkl` by combining two detection approaches to achieve both **accurate timing** and **reliable peak selection**.

## Problem Statement

### Original Single-Pixel Method (`on_off_dict_hd.pkl`)

The single-pixel method detects ON/OFF events by computing the derivative of a single pixel's intensity trace and finding the maximum positive (ON) and negative (OFF) derivatives.

**Advantage**: Precise timing - the exact frame when the bar enters/exits that specific pixel.

**Problem**: When multiple peaks exist in the derivative trace (due to noise or bar edge effects), `np.argmax()` arbitrarily selects one. This can result in selecting the wrong peak, often offset by ~440 frames (approximately one direction segment).

### Original Area-Based Method (`on_off_dict_area_hd.pkl`)

The area-based method averages a 50×50 pixel region around each pixel before computing derivatives. This spatial averaging smooths out noise and produces a cleaner signal.

**Advantage**: Reliable peak selection - the averaging suppresses spurious peaks, making the true onset peak dominant.

**Problem**: The timing is less precise because the averaged trace represents when the bar enters/exits the *region*, not the specific pixel. This introduces a systematic timing offset.

## Double-Refined Solution

The double-refined method combines both approaches:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Double-Refined Algorithm                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Single-Pixel Trace ──► Find ALL Peaks ──► Candidate List         │
│         (accurate)           (scipy)         (precise timing)       │
│                                                    │                │
│                                                    ▼                │
│   Area-Based Trace ───► Find Reference ──► Select Closest ──► ON   │
│      (reliable)           Peak              Candidate              │
│                                                                     │
│   ON Peak ──► Find First Significant Negative Derivative ──► OFF   │
│                     (threshold-based search)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Algorithm Details

#### Step 1: Find All Onset Candidates (Single-Pixel)

Instead of using `np.argmax()` which returns only one peak, we use `scipy.signal.find_peaks()` with prominence filtering:

```python
from scipy.signal import find_peaks

# Find all local maxima with at least 30% of max prominence
min_prominence = segment.max() * 0.3
peaks, _ = find_peaks(segment, prominence=min_prominence)
```

This returns ALL significant positive derivative peaks in the segment.

#### Step 2: Get Reference Onset (Area-Based)

Compute the averaged trace over a 50×50 region and find its maximum derivative:

```python
# Average 50x50 area centered on pixel
area_trace = ds_stim[:, x-25:x+25, y-25:y+25].mean(axis=(1, 2))
area_deriv = np.diff(area_trace)
area_ref_idx = np.argmax(area_segment)
```

The area-based peak reliably indicates WHICH peak is the true onset.

#### Step 3: Select Best Candidate

Choose the single-pixel candidate closest in time to the area-based reference:

```python
if len(candidates) == 1:
    on_peak_iter = candidates[0]
else:
    distances = np.abs(candidates - area_ref_idx)
    on_peak_iter = candidates[np.argmin(distances)]
```

This gives us the **precise timing** from single-pixel detection combined with the **correct peak selection** from area-based detection.

#### Step 4: OFF Peak Detection

The OFF peak is found using a refined threshold-based search that ensures OFF > ON:

```python
# Threshold: 50% of ON peak magnitude (negative)
threshold = -abs(on_peak_value) * 0.5

# Find first frame below threshold after ON
search_region = segment[on_peak_idx + 1:]
off_idx = np.where(search_region <= threshold)[0][0] + on_peak_idx + 1
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BAR_HEIGHT` | 50 | Size of averaging region (50×50 pixels) |
| `RECEPTIVE_FIELD_SIDE_LENGTH` | 25 | Half-width of averaging region |
| `min_prominence_ratio` | 0.3 | Minimum peak prominence (30% of max) |
| `OFF threshold ratio` | 0.5 | OFF detection threshold (50% of ON magnitude) |
| `min_threshold` | -50.0 | Minimum absolute threshold for OFF detection |

## Validation Results

### Comparison with Legacy

| Metric | Value |
|--------|-------|
| Total pixels | 90,000 |
| Exact matches with legacy | 49,819 (55.4%) |
| ON differences | 33,839 pixels |
| OFF differences | 38,804 pixels |

### ON Peak Difference Analysis

The differences between double-refined and legacy ON peaks cluster around **440 time units**:

| Statistic | Value |
|-----------|-------|
| Mean difference | 409.5 frames |
| Median difference | 440.0 frames |
| Std deviation | 144.8 frames |

**Distribution:**
- 95.2% of differences fall in [400, 500) range
- This corresponds to approximately one direction segment (550 frames)
- Indicates legacy method was selecting peaks from wrong segments

### Constraint Validation

All 90,000 pixels pass the OFF > ON constraint:

| Direction | Status |
|-----------|--------|
| 0° | ✓ |
| 45° | ✓ |
| 90° | ✓ |
| 135° | ✓ |
| 180° | ✓ |
| 225° | ✓ |
| 270° | ✓ |
| 315° | ✓ |

**Total problematic entries: 0**

## Output Files

| File | Path |
|------|------|
| Script | `dataframe_phase/investigation/recreate_on_off_dict_hd_double_refined.py` |
| Output | `dataframe_phase/investigation/output/moving_h_bar_s5_d8_3x_on_off_dict_hd_double_refined.pkl` |

## Dictionary Structure

```python
{
    (x, y): {
        "on_peak_location": [24 integers],   # Frame indices for ON peaks (3 reps × 8 directions)
        "off_peak_location": [24 integers]   # Frame indices for OFF peaks
    },
    ...  # 90,000 pixel entries total (300 × 300)
}
```

## Stimulus Parameters

| Parameter | Value |
|-----------|-------|
| Stimulus file | `moving_h_bar_s5_d8_3x.npy` |
| Shape | (13560, 300, 300) - frames × height × width |
| Repetitions | 3 |
| Directions | 8 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) |
| Frames per direction | 550 |
| Pre-margin frames | 120 |

## Processing Performance

- **CPU cores used**: 80% of available (25 of 32 cores)
- **Processing time**: ~373 seconds (~6.2 minutes)
- **Method**: Row-parallel multiprocessing with `pool.imap()`

