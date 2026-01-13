# Parquet File Structure Documentation

**File:** `firing_rate_with_all_features_loaded_extracted20260104.parquet`

**Location:** `dataframe_phase/extract_feature/`

---

## Overview

| Property | Value |
|----------|-------|
| Rows | 46,992 cells |
| Columns | 120 |
| Memory | ~198 MB |
| Index | Cell ID (string) |
| Unique Batches | 322 recordings |

### Index Format

Cell IDs follow the pattern: `{recording_session}_unit_{unit_number}`

**Example:** `2024.02.26-10.53.19-Rec_unit_001`

Batch ID is extracted as everything before the last `_unit_` or `_` separator.

---

## Column Categories

The 120 columns are organized into the following categories:

1. [Raw Stimulus Traces](#1-raw-stimulus-traces) (14 columns)
2. [Corrected Moving Bar Traces](#2-corrected-moving-bar-traces) (8 columns)
3. [Frequency Section Traces](#3-frequency-section-traces) (5 columns)
4. [Spatial Receptive Field](#4-spatial-receptive-field) (10 columns)
5. [STA Time Course](#5-sta-time-course) (1 column)
6. [LNL Model Parameters](#6-lnl-model-parameters) (14 columns)
7. [Quality Indices](#7-quality-indices) (3 columns)
8. [Direction/Orientation Selectivity](#8-directionorientation-selectivity) (5 columns)
9. [Step-Up Response Features](#9-step-up-response-features) (12 columns)
10. [Color Response Features](#10-color-response-features) (12 columns)
11. [Frequency Response Features](#11-frequency-response-features) (25 columns)
12. [Metadata](#12-metadata) (3 columns)
13. [Batch Statistics](#13-batch-statistics) (2 columns)

---

## 1. Raw Stimulus Traces

Nested arrays containing trial-by-trial firing rate responses at 60 Hz.

| Column | Trials | Samples | Duration | Description |
|--------|--------|---------|----------|-------------|
| `baseline_127` | 1 | 179 | ~3s | Baseline recording |
| `freq_step_5st_3x` | 3 | 1499 | ~25s | Frequency step stimulus (0.5, 1, 2, 4, 10 Hz) |
| `green_blue_3s_3i_3x` | 3 | 719 | ~12s | Color opponency (green/blue) |
| `step_up_5s_5i_b0_3x` | 3 | 599 | ~10s | Step-up stimulus (light ON/OFF) |
| `step_up_5s_5i_b0_30x` | 30 | 599 | ~10s | Step-up stimulus (30 trials for QI) |
| `moving_h_bar_s5_d8_3x_0` | 3 | 99 | ~1.6s | Moving bar, direction 0° |
| `moving_h_bar_s5_d8_3x_45` | 3 | 98 | ~1.6s | Moving bar, direction 45° |
| `moving_h_bar_s5_d8_3x_90` | 3 | 99 | ~1.6s | Moving bar, direction 90° |
| `moving_h_bar_s5_d8_3x_135` | 3 | 98 | ~1.6s | Moving bar, direction 135° |
| `moving_h_bar_s5_d8_3x_180` | 3 | 99 | ~1.6s | Moving bar, direction 180° |
| `moving_h_bar_s5_d8_3x_225` | 3 | 98 | ~1.6s | Moving bar, direction 225° |
| `moving_h_bar_s5_d8_3x_270` | 3 | 99 | ~1.6s | Moving bar, direction 270° |
| `moving_h_bar_s5_d8_3x_315` | 3 | 98 | ~1.6s | Moving bar, direction 315° |
| `iprgc_test` | 2 | 7199 | ~120s | ipRGC test stimulus |

**Data format:** `object` dtype containing nested NumPy arrays of shape `(n_trials, n_samples)`

---

## 2. Corrected Moving Bar Traces

Angle-corrected moving bar responses with standardized direction labels.

| Column | Trials | Samples | NaN Count |
|--------|--------|---------|-----------|
| `corrected_moving_h_bar_s5_d8_3x_000` | 3 | 98 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_045` | 3 | 99 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_090` | 3 | 98 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_135` | 3 | 99 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_180` | 3 | 98 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_225` | 3 | 99 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_270` | 3 | 98 | 2,922 |
| `corrected_moving_h_bar_s5_d8_3x_315` | 3 | 99 | 2,922 |

**Note:** 2,922 cells have NaN values (no angle correction applied).

---

## 3. Frequency Section Traces

Trial-averaged and sectioned frequency step responses.

| Column | Shape | Samples | Frame Range | Frequency |
|--------|-------|---------|-------------|-----------|
| `freq_section_0p5hz` | (240,) | 240 | 30–270 | 0.5 Hz |
| `freq_section_1hz` | (240,) | 240 | 330–570 | 1 Hz |
| `freq_section_2hz` | (240,) | 240 | 630–870 | 2 Hz |
| `freq_section_4hz` | (240,) | 240 | 930–1170 | 4 Hz |
| `freq_section_10hz` | (240,) | 240 | 1230–1470 | 10 Hz |

**Note:** These are pre-extracted sections from the full `freq_step_5st_3x` trace, trial-averaged.

---

## 4. Spatial Receptive Field

Parameters from Gaussian and Difference-of-Gaussians (DoG) fits to receptive field data.

### Gaussian Fit

| Column | Type | Description |
|--------|------|-------------|
| `gaussian_sigma_x` | float64 | Gaussian width (x-axis) |
| `gaussian_sigma_y` | float64 | Gaussian width (y-axis) |
| `gaussian_amp` | float64 | Gaussian amplitude |
| `gaussian_r2` | float64 | Goodness of fit |

### DoG Fit

| Column | Type | Description |
|--------|------|-------------|
| `dog_sigma_exc` | float64 | Excitatory center sigma |
| `dog_sigma_inh` | float64 | Inhibitory surround sigma |
| `dog_amp_exc` | float64 | Excitatory amplitude |
| `dog_amp_inh` | float64 | Inhibitory amplitude |
| `dog_r2` | float64 | Goodness of fit |

### Spatial Coordinates

| Column | Type | NaN Count | Description |
|--------|------|-----------|-------------|
| `angle_correction_applied` | float64 | 2,922 | Rotation angle applied |
| `transformed_x` | float64 | 2,219 | X coordinate (transformed) |
| `transformed_y` | float64 | 2,219 | Y coordinate (transformed) |
| `polar_radius` | float64 | 2,219 | Distance from optic disc |
| `polar_theta_deg` | float64 | 2,219 | Angle in polar coordinates |
| `polar_theta_deg_raw` | float64 | 2,219 | Raw polar angle |
| `cartesian_x` | float64 | 2,219 | Cartesian X |
| `cartesian_y` | float64 | 2,219 | Cartesian Y |

---

## 5. STA Time Course

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `sta_time_course` | object | (60,) | Spike-triggered average temporal kernel |

**Sampling:** 60 time points representing the temporal receptive field profile.

---

## 6. LNL Model Parameters

Linear-Nonlinear (LNL) model fit parameters.

| Column | Type | NaN Count | Description |
|--------|------|-----------|-------------|
| `lnl_a` | float64 | 1,395 | LNL parameter a |
| `lnl_b` | float64 | 1,395 | LNL parameter b |
| `lnl_a_norm` | float64 | 1,395 | Normalized LNL parameter a |
| `lnl_bits_per_spike` | float64 | 1,395 | Information content |
| `lnl_r_squared` | float64 | 1,395 | Model fit R² |
| `lnl_rectification_index` | float64 | 1,395 | Rectification index |
| `lnl_nonlinearity_index` | float64 | 1,395 | Nonlinearity index |
| `lnl_threshold_g` | float64 | 1,395 | Threshold value |
| `lnl_log_likelihood` | float64 | 1,395 | Log-likelihood |
| `lnl_null_log_likelihood` | float64 | 1,395 | Null model log-likelihood |
| `lnl_n_frames` | float64 | 1,395 | Number of frames |
| `lnl_n_spikes` | float64 | 1,395 | Number of spikes |
| `lnl_g_bin_centers` | object | 1,395 | Bin centers (50,) |
| `lnl_rate_vs_g` | object | 1,395 | Rate vs generator (50,) |

---

## 7. Quality Indices

Measures of response reliability across trials.

| Column | Type | NaN Count | Range | Median | Description |
|--------|------|-----------|-------|--------|-------------|
| `step_up_QI` | float64 | 1,324 | [0.14, 1.0] | 0.77 | Step-up response quality |
| `iprgc_2hz_QI` | float64 | 333 | [0.0, 1.0] | 0.0 | ipRGC 2 Hz modulation response |
| `iprgc_20hz_QI` | float64 | 353 | — | — | ipRGC 20 Hz modulation response |

**Quality Index Formula:**

$$\text{QI} = \frac{\text{Var}(\bar{r})}{\overline{\text{Var}(r_i)}}$$

---

## 8. Direction/Orientation Selectivity

| Column | Type | NaN Count | Range | Description |
|--------|------|-----------|-------|-------------|
| `dsi` | float64 | 2,922 | [0.0, 2.5] | Direction selectivity index |
| `osi` | float64 | 2,922 | [0.0, 1.0] | Orientation selectivity index |
| `preferred_direction` | float64 | 2,922 | [0, 360) | Preferred direction (degrees) |
| `ds_p_value` | float64 | 2,922 | [0.0, 1.0] | Direction selectivity p-value |
| `os_p_value` | float64 | 2,922 | [0.0, 1.0] | Orientation selectivity p-value |

**Direction-selective cells:** 10,776 cells have `ds_p_value < 0.05`

---

## 9. Step-Up Response Features

Features extracted from the step-up (light ON/OFF) stimulus.

### Amplitude Features

| Column | Type | Description |
|--------|------|-------------|
| `on_peak_extreme` | float64 | Maximum ON response |
| `on_sustained` | float64 | Sustained ON response |
| `off_peak_extreme` | float64 | Maximum OFF response |
| `off_sustained` | float64 | Sustained OFF response |
| `base_mean` | float64 | Baseline mean firing rate |
| `base_std` | float64 | Baseline standard deviation |

### Timing Features

| Column | Type | NaN Count | Description |
|--------|------|-----------|-------------|
| `time_to_on_peak_extreme` | float64 | 5,588 | Time to ON peak |
| `time_to_off_peak_extreme` | float64 | 8,502 | Time to OFF peak |

### Derived Ratios

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `on_off_ratio` | float64 | [-1, 1] | ON vs OFF response ratio |
| `on_trans_sus_ratio` | float64 | [-1, 1] | ON transient/sustained ratio |
| `off_trans_sus_ratio` | float64 | [-1, 1] | OFF transient/sustained ratio |
| `on_off_sus_ratio` | float64 | — | ON/OFF sustained ratio |

---

## 10. Color Response Features

Features extracted from the green/blue color opponency stimulus.

### Amplitude Features

| Column | Type | Description |
|--------|------|-------------|
| `green_on_peak_extreme` | float64 | Green ON response |
| `blue_on_peak_extreme` | float64 | Blue ON response |
| `green_off_peak_extreme` | float64 | Green OFF response |
| `blue_off_peak_extreme` | float64 | Blue OFF response |
| `gb_base_mean` | float64 | Baseline mean |
| `gb_base_std` | float64 | Baseline std |

### Timing Features

| Column | Type | NaN Count | Description |
|--------|------|-----------|-------------|
| `time_to_green_on_peak` | float64 | 5,062 | Time to green ON peak |
| `time_to_blue_on_peak` | float64 | 5,244 | Time to blue ON peak |
| `time_to_green_off_peak` | float64 | 7,204 | Time to green OFF peak |
| `time_to_blue_off_peak` | float64 | 7,495 | Time to blue OFF peak |

### Derived Ratios

| Column | Type | Description |
|--------|------|-------------|
| `green_blue_on_ratio` | float64 | Green/Blue ON ratio |
| `green_blue_off_ratio` | float64 | Green/Blue OFF ratio |

---

## 11. Frequency Response Features

Features extracted from each frequency section of the frequency step stimulus.

### Per-Frequency Features

For each frequency (0.5, 1, 2, 4, 10 Hz):

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `freq_step_{f}hz_amp` | float64 | Response amplitude |
| `freq_step_{f}hz_phase` | float64 | Response phase (high NaN count) |
| `freq_step_{f}hz_r_squared` | float64 | Sinusoidal fit R² |
| `freq_step_{f}hz_offset` | float64 | DC offset |
| `freq_step_{f}hz_std` | float64 | Response variability |

**Frequency labels:** `05` (0.5 Hz), `1`, `2`, `4`, `10`

**Note:** Phase columns have high NaN counts (10,837–22,882) because phase is undefined for weak responses.

---

## 12. Metadata

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `axon_type` | object | rgc, ac, unknown, other, no_label | Cell classification |

### Axon Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
| rgc | 17,418 | 37.1% |
| unknown | 12,566 | 26.7% |
| ac | 10,167 | 21.6% |
| other | 5,522 | 11.7% |
| no_label | 1,319 | 2.8% |

---

## 13. Batch Statistics

Per-batch (recording session) statistics.

| Column | Type | Description |
|--------|------|-------------|
| `total_unit_count` | int64 | Total units in batch |
| `good_count` | int64 | Units with QI > threshold |
| `good_rgc_count` | int64 | Good RGCs in batch |

---

## Data Types Summary

| Data Type | Count | Examples |
|-----------|-------|----------|
| float64 | 89 | QI, DSI, ratios, coordinates |
| object (array) | 28 | Trace data, STA |
| object (string) | 1 | axon_type |
| int64 | 3 | unit counts |

---

## Usage Notes

### Loading the File

```python
import pandas as pd

df = pd.read_parquet('dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet')
```

### Accessing Trace Data

```python
# Get trace for a single cell
trace = df.loc['2024.02.26-10.53.19-Rec_unit_001', 'step_up_5s_5i_b0_3x']

# Convert to numpy array
import numpy as np
arr = np.asarray(trace)  # Shape: (3, 599) for 3 trials

# Average across trials
mean_trace = arr.mean(axis=0)  # Shape: (599,)
```

### Filtering Cells

```python
# Filter by axon type
rgcs = df[df['axon_type'] == 'rgc']

# Filter by quality index
good_cells = df[df['step_up_QI'] > 0.7]

# Filter direction-selective cells
ds_cells = df[df['ds_p_value'] < 0.05]
```

### Extracting Batch Information

```python
# Get batch ID from cell ID
def get_batch(cell_id):
    return cell_id.rsplit('_', 1)[0]

df['batch'] = df.index.to_series().apply(get_batch)
```

---

## File History

- **Created:** 2026-01-04
- **Source:** Feature extraction pipeline
- **Predecessor:** Raw spike sorting output + stimulus timing data
