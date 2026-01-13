# Data Preprocessing Pipeline: Detailed Documentation

This document provides a comprehensive explanation of the data preprocessing approach used in the Baden-method RGC clustering pipeline for electrophysiology data.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Sources and Format](#2-data-sources-and-format)
3. [Cell Filtering](#3-cell-filtering)
4. [Signal Processing Chain](#4-signal-processing-chain)
5. [Stimulus-Specific Preprocessing](#5-stimulus-specific-preprocessing)
6. [Feature Extraction](#6-feature-extraction)
7. [Configuration Parameters](#7-configuration-parameters)
8. [Mathematical Details](#8-mathematical-details)

---

## 1. Overview

### Pipeline Purpose

The preprocessing pipeline transforms raw electrophysiology spike rate data into a standardized format suitable for unsupervised clustering. The goal is to extract functional features from multiple visual stimuli while handling the inherent differences between electrophysiology data and the calcium imaging data used in the original Baden paper.

### High-Level Data Flow

```
Raw Parquet Data
       │
       ▼
┌──────────────────────┐
│    Cell Filtering    │ ◄── QI threshold, axon type, batch size, baseline
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ DS / non-DS Splitting│ ◄── Direction selectivity p-value
└──────────────────────┘
       │
       ├─────────────────────┐
       ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│  DS Population  │  │ non-DS Population│
└─────────────────┘  └─────────────────┘
       │                     │
       ▼                     ▼
┌──────────────────────────────────────┐
│       Signal Conditioning            │ ◄── Filter, downsample, (optional normalize)
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│        Feature Extraction            │ ◄── SparsePCA, SVD, PCA
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Feature Standardization         │ ◄── Z-score normalization
└──────────────────────────────────────┘
       │
       ▼
   40D Feature Vector per Cell
```

---

## 2. Data Sources and Format

### Input Data

**File:** `firing_rate_with_all_features_loaded_extracted20260104.parquet`

**Format:** Apache Parquet with mixed column types:
- Scalar columns: `step_up_QI`, `ds_p_value`, `axon_type`, etc.
- Array columns: Trace data stored as NumPy arrays or nested lists

### Trace Columns Used

| Column | Description | Shape per Cell |
|--------|-------------|----------------|
| `step_up_5s_5i_b0_3x` | Step-up stimulus response (baseline source) | (N_trials, 600) |
| `freq_section_0p5hz` | 0.5 Hz frequency section | (240,) |
| `freq_section_1hz` | 1 Hz frequency section | (240,) |
| `freq_section_2hz` | 2 Hz frequency section | (240,) |
| `freq_section_4hz` | 4 Hz frequency section | (240,) |
| `freq_section_10hz` | 10 Hz frequency section | (240,) |
| `green_blue_3s_3i_3x` | Color opponency stimulus | (N_trials, T) |
| `corrected_moving_h_bar_s5_d8_3x_XXX` | Moving bar (8 directions) | (N_trials, T) |
| `sta_time_course` | Receptive field STA | (T,) |

### Index Convention

- **Cell ID:** `{filename}_{unit_id}` (e.g., `recording_2026_01_04_001_unit_5`)
- **Batch ID:** Extracted as filename prefix (everything before the last `_`)

---

## 3. Cell Filtering

Cells are filtered through multiple sequential stages. All thresholds are configurable in `config.py`.

### Stage 1: Missing Value Filter

**Columns checked:** `step_up_QI`, `ds_p_value`, `axon_type`

Removes cells with:
- `NaN` values in scalar metadata columns
- `NaN` values within any trace array (checks nested trial arrays)

### Stage 2: Quality Index Filter

**Threshold:** `QI_THRESHOLD = 0.7`

**Column:** `step_up_QI`

$$\text{QI} = \frac{\text{Var}(\bar{r})}{\overline{\text{Var}(r_i)}}$$

where:
- $\bar{r}$ = mean response across trials
- $r_i$ = individual trial response

**Interpretation:**
- QI = 1.0: Perfect trial-to-trial consistency
- QI > 0.7: High quality, reliable response
- QI < 0.5: Noisy or unreliable response

### Stage 3: Axon Type Filter

**Allowed types:** `VALID_AXON_TYPES = ["rgc", "ac"]`

Restricts analysis to retinal ganglion cells (RGC) and amacrine cells (AC).

### Stage 4: Baseline Filter

**Threshold:** `BASELINE_MAX_THRESHOLD = 200.0` Hz

**Purpose:** Exclude cells with abnormally high spontaneous firing rates.

**Baseline computation:**
1. Average trials of `step_up_5s_5i_b0_3x` trace
2. Apply 10 Hz low-pass Butterworth filter
3. Downsample from 60 Hz to 10 Hz
4. Compute median of first 5 samples (0.5 s at 10 Hz)

**Rationale:**
- Normal RGC baseline: typically < 50 Hz
- High baseline (> 200 Hz) may indicate:
  - Recording artifacts
  - Multi-unit contamination
  - Injured or dying cells

### Stage 5: Batch Size Filter

**Threshold:** `MIN_BATCH_GOOD_CELLS = 25`

**Definition:** A "batch" is all cells from the same recording session (same filename prefix).

**Exclusion rule:** Remove ALL cells from batches with fewer than 25 cells (after QI filtering).

**Rationale:**
- Small batches may have systematic differences (recording quality, animal health)
- Ensures statistical reliability of clustering

### Filtering Summary Table

| Filter | Threshold | Removes |
|--------|-----------|---------|
| NaN values | Any NaN | Incomplete data |
| Quality Index | QI ≤ 0.7 | Unreliable responses |
| Axon Type | Not in ["rgc", "ac"] | Other cell types |
| Baseline | > 200 Hz | High spontaneous firing |
| Batch Size | < 25 cells | Under-sampled batches |

---

## 4. Signal Processing Chain

### Core Processing Steps

Each stimulus trace undergoes a standardized processing chain:

```
Raw Trace (60 Hz, multiple trials)
         │
         ▼
┌─────────────────────────┐
│   Step 1: Average       │
│   across trials         │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Step 2: Low-pass      │  ◄── 10 Hz cutoff, 4th-order Butterworth
│   Filter (optional)     │      Zero-phase (sosfiltfilt)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Step 3: Downsample    │  ◄── 60 Hz → 10 Hz (factor = 6)
│   (optional)            │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Step 4: Baseline      │  ◄── DISABLED by default
│   Subtraction           │      (APPLY_BASELINE_ZEROING = False)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Step 5: Max-abs       │  ◄── DISABLED by default
│   Normalization         │      (APPLY_MAX_ABS_NORMALIZATION = False)
└─────────────────────────┘
         │
         ▼
   Preprocessed Trace
```

### Step 1: Trial Averaging

**Purpose:** Reduce trial-to-trial variability.

**Input:** Nested array of shape `(N_trials, T_samples)`

**Output:** 1D array of shape `(T_samples,)`

**Method:** Simple arithmetic mean across trials

$$\bar{r}(t) = \frac{1}{N} \sum_{i=1}^{N} r_i(t)$$

### Step 2: Low-Pass Filtering

**Purpose:** Remove high-frequency spike count variability.

**Parameters:**
- Cutoff: `LOWPASS_CUTOFF = 10.0` Hz
- Order: `FILTER_ORDER = 4`
- Type: Butterworth
- Application: Zero-phase (`scipy.signal.sosfiltfilt`)

**Why 10 Hz cutoff?**
- Mimics temporal smoothing inherent in calcium imaging
- Removes spike-timing jitter
- Preserves stimulus-relevant dynamics

**Transfer function:**

$$H(s) = \frac{1}{\left(1 + \frac{s}{\omega_c}\right)^4}$$

where $\omega_c = 2\pi \times 10$ rad/s.

**Zero-phase implementation:**
1. Filter forward: $y_1 = H(x)$
2. Filter backward: $y_2 = H(\text{flip}(y_1))$
3. Output: $y = \text{flip}(y_2)$

This eliminates phase distortion.

### Step 3: Downsampling

**Purpose:** Reduce data dimensionality and match Baden's temporal scale.

**Factor:** `DOWNSAMPLE_FACTOR = 6` (60 Hz → 10 Hz)

**Method:** Take every 6th sample

$$y[n] = x[6n]$$

**Important:** Low-pass filtering (Step 2) prevents aliasing.

### Step 4: Baseline Subtraction (Optional)

**Status:** `APPLY_BASELINE_ZEROING = False` (disabled by default)

**When enabled:**
1. Compute baseline from step-up trace (filtered + downsampled)
2. Take median of first `BASELINE_N_SAMPLES = 5` samples
3. Subtract this value from all traces of the cell

$$x_{\text{zeroed}}(t) = x(t) - \text{median}(x[0:5])$$

**Rationale for disabling:**
- Preserves absolute firing rate information
- Avoids issues with negative firing rates
- Original amplitude scale may carry biological meaning

### Step 5: Max-Absolute Normalization (Optional)

**Status:** `APPLY_MAX_ABS_NORMALIZATION = False` (disabled by default)

**When enabled:**

$$x_{\text{norm}}(t) = \frac{x(t)}{\max(|x|) + \epsilon}$$

where $\epsilon = 10^{-8}$ for numerical stability.

**Output range:** $[-1, 1]$

**Rationale for disabling:**
- Allows comparison of response magnitudes across cells
- Z-score standardization at feature level handles scale differences

---

## 5. Stimulus-Specific Preprocessing

Different stimuli require different preprocessing approaches based on their temporal characteristics.

### Frequency Section Traces

| Section | Frequency | Frame Range | Filter | Downsample | Final Rate |
|---------|-----------|-------------|--------|------------|------------|
| 0.5 Hz | 0.5 Hz | 30–270 | Yes | Yes (6×) | 10 Hz |
| 1 Hz | 1 Hz | 330–570 | Yes | Yes (6×) | 10 Hz |
| 2 Hz | 2 Hz | 630–870 | Yes | Yes (6×) | 10 Hz |
| 4 Hz | 4 Hz | 930–1170 | Yes | Yes (6×) | 10 Hz |
| 10 Hz | 10 Hz | 1230–1470 | **No** | **No** | 60 Hz |

**Why no filtering for 10 Hz section?**
- The 10 Hz modulation frequency would be attenuated by the 10 Hz low-pass filter
- Preserved at full 60 Hz to capture high-frequency response dynamics

**10 Hz Section Slicing:**
- `FREQ_10HZ_START_OFFSET = 60` (skip first 1s)
- `FREQ_10HZ_END_OFFSET = -60` (skip last 1s)
- Removes edge artifacts from filtering/stimulus transitions

### Color Stimulus Trace

**Column:** `green_blue_3s_3i_3x`

**Processing:**
1. Average trials
2. Low-pass filter (10 Hz)
3. Downsample (6×)
4. (Optional) Baseline subtract + normalize

### Moving Bar Traces (8 Directions)

**Columns:** `corrected_moving_h_bar_s5_d8_3x_{000,045,...,315}`

**Processing per direction:**
1. Average trials
2. Low-pass filter (10 Hz)
3. Downsample (6×)
4. (Optional) Baseline subtract + normalize

**Then for SVD:**
1. Stack 8 directions → matrix of shape (T, 8)
2. **Per-cell normalization:** Divide by max absolute value across all directions
3. SVD decomposition

**Per-cell normalization preserves:**
- Relative amplitude differences across directions
- Directional tuning information

### RF Time Course (STA)

**Column:** `sta_time_course`

**Processing:**
1. Average trials (if applicable)
2. **No** low-pass filter (already smooth)
3. Downsample (6×)
4. (Optional) Baseline subtract + normalize

---

## 6. Feature Extraction

After preprocessing, features are extracted using dimensionality reduction techniques.

### Feature Summary

| Feature Set | Method | Components | Non-zero per Comp | Total |
|-------------|--------|------------|-------------------|-------|
| Freq 0.5 Hz | SparsePCA | 4 | 4 | 4 |
| Freq 1 Hz | SparsePCA | 4 | 4 | 4 |
| Freq 2 Hz | SparsePCA | 4 | 4 | 4 |
| Freq 4 Hz | SparsePCA | 4 | 4 | 4 |
| Freq 10 Hz | SparsePCA | 4 | 4 | 4 |
| Color | SparsePCA | 6 | 10 | 6 |
| Bar Time Course | SparsePCA | 8 | 5 | 8 |
| Bar Derivative | SparsePCA | 4 | 6 | 4 |
| RF | PCA | 2 | — | 2 |
| **Total** | | | | **40** |

### Sparse PCA

**Purpose:** Extract interpretable components with localized temporal weights.

**Process:**
1. Fit `sklearn.decomposition.SparsePCA` with `alpha` regularization
2. Enforce hard sparsity by keeping only top-k weights per component
3. Renormalize components to unit length

**Hard sparsity enforcement:**

For each component $\mathbf{w}$:
1. Find indices of top-k weights by absolute value
2. Zero out all other weights
3. Renormalize: $\mathbf{w} \leftarrow \mathbf{w} / \|\mathbf{w}\|$

### SVD for Moving Bar

**Matrix construction:** For each cell, stack 8 direction traces as columns:

$$X \in \mathbb{R}^{T \times 8}$$

**Decomposition:**

$$X = U \Sigma V^T$$

**Time course extraction:**

$$\mathbf{tc} = \mathbf{u}_1 \cdot \sigma_1$$

This captures the dominant temporal pattern across all directions.

### Feature Standardization

After extraction, all features are Z-score standardized:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

This ensures all features contribute equally to clustering.

---

## 7. Configuration Parameters

All parameters are centralized in `config.py` for easy modification.

### Signal Processing

```python
SAMPLING_RATE = 60.0            # Original sampling rate (Hz)
LOWPASS_CUTOFF = 10.0           # Low-pass filter cutoff (Hz)
TARGET_SAMPLING_RATE = 10.0     # After downsampling (Hz)
DOWNSAMPLE_FACTOR = 6           # 60 / 10
FILTER_ORDER = 4                # Butterworth order
BASELINE_N_SAMPLES = 5          # Samples for baseline median
```

### Preprocessing Toggles

```python
APPLY_BASELINE_ZEROING = False      # Subtract baseline from traces
APPLY_MAX_ABS_NORMALIZATION = False # Scale traces to [-1, 1]
```

### Cell Filtering

```python
QI_THRESHOLD = 0.7              # Minimum quality index
DS_P_THRESHOLD = 0.05           # Direction selectivity p-value
VALID_AXON_TYPES = ["rgc", "ac"] # Allowed cell types
BASELINE_MAX_THRESHOLD = 200.0  # Maximum baseline firing rate
MIN_BATCH_GOOD_CELLS = 25       # Minimum cells per batch
```

### Frequency Section Settings

```python
FREQ_SECTION_FILTER = {
    "freq_section_0p5hz": True,   # Apply 10 Hz low-pass
    "freq_section_1hz": True,
    "freq_section_2hz": True,
    "freq_section_4hz": True,
    "freq_section_10hz": False,   # No filtering for 10 Hz
}
FREQ_10HZ_START_OFFSET = 60     # Skip first 60 frames (1s)
FREQ_10HZ_END_OFFSET = -60      # Skip last 60 frames (1s)
```

### Sparse PCA Settings

```python
FREQ_SECTION_N_COMPONENTS = 4   # Components per frequency section
FREQ_SECTION_TOP_K = 4          # Non-zero weights per component
FREQ_SECTION_ALPHA = 1.0        # Sparsity regularization

COLOR_N_COMPONENTS = 6
COLOR_TOP_K = 10
COLOR_ALPHA = 1.0

BAR_TC_N_COMPONENTS = 8
BAR_TC_TOP_K = 5
BAR_TC_ALPHA = 1.0

BAR_DERIV_N_COMPONENTS = 4
BAR_DERIV_TOP_K = 6
BAR_DERIV_ALPHA = 1.0

RF_N_COMPONENTS = 2             # Regular PCA (no sparsity)
```

---

## 8. Mathematical Details

### Butterworth Filter Design

The 4th-order Butterworth low-pass filter is designed using second-order sections (SOS) for numerical stability:

$$H(z) = \prod_{i=1}^{2} \frac{b_{0,i} + b_{1,i} z^{-1} + b_{2,i} z^{-2}}{1 + a_{1,i} z^{-1} + a_{2,i} z^{-2}}$$

**Cutoff normalization:**

$$\omega_n = \frac{f_c}{f_s / 2} = \frac{10}{30} = 0.333$$

### Quality Index Derivation

The QI measures signal-to-noise ratio based on trial variability:

$$\text{QI} = \frac{\text{Var}_t[\bar{r}(t)]}{\frac{1}{N}\sum_{i=1}^{N} \text{Var}_t[r_i(t)]}$$

**Interpretation:**
- Numerator: Variance of mean trace (signal power)
- Denominator: Average within-trial variance (noise power)
- Higher QI → stronger stimulus-locked response

### Sparse PCA Objective

SparsePCA minimizes:

$$\min_{D, \alpha} \frac{1}{2} \|X - D\alpha^T\|_F^2 + \lambda \|\alpha\|_1$$

where:
- $X$ = data matrix (samples × features)
- $D$ = dictionary (components)
- $\alpha$ = sparse codes
- $\lambda$ = sparsity parameter (`alpha` in sklearn)

### SVD Properties

For the direction matrix $X \in \mathbb{R}^{T \times 8}$:

$$X = U \Sigma V^T$$

- $U[:, 0]$: First left singular vector (temporal pattern)
- $\Sigma[0]$: First singular value (explained variance)
- $V[0, :]$: First right singular vector (direction weights)

**Time course = $U[:, 0] \times \Sigma[0]$** captures how much each time point contributes to the dominant response mode.

---

## Summary

The preprocessing pipeline transforms raw electrophysiology spike rate data through a carefully designed sequence of operations:

1. **Filtering:** Remove cells with missing data, low QI, inappropriate axon type, high baseline, or from small batches
2. **Trial averaging:** Reduce noise by averaging across repeated presentations
3. **Low-pass filtering:** Remove high-frequency jitter (10 Hz cutoff, 4th-order Butterworth)
4. **Downsampling:** Reduce dimensionality (60 Hz → 10 Hz)
5. **Feature extraction:** SparsePCA and SVD to create 40D feature vectors
6. **Standardization:** Z-score normalization for clustering

The pipeline is designed to be modular and configurable, with all parameters accessible in `config.py` for easy experimentation and optimization.
