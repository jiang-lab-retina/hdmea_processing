# iprgc_2hz_QI Calculation Pipeline

This document describes how the `iprgc_2hz_QI` (ipRGC 2 Hz Quality Index) is calculated from raw `.h5` files through the data processing pipeline.

## Overview

The calculation involves two main stages:

1. **Stage 1: Firing Rate Generation** (`pipeline_firing_rate.py`)  
   Raw spike times → Sample-based firing rate at 60 Hz

2. **Stage 2: Feature Extraction** (`extract_feature_step_iprgc.py`)  
   Firing rate arrays → Quality Index using Pearson correlation

---

## Stage 1: Firing Rate Generation

### 1.1 Data Source in HDF5 Files

The ipRGC test data is stored in the HDF5 file under:

```
units/{unit_id}/spike_times_sectioned/iprgc_test/
├── trials_spike_times/
│   ├── 0      # spike times for trial 0
│   ├── 1      # spike times for trial 1
│   └── ...
└── trials_start_end    # (n_trials, 2) array of [start_sample, end_sample]
```

- **Spike times**: Raw sample indices (at acquisition rate, typically 20 kHz) when spikes occurred
- **trials_start_end**: Start and end sample indices for each trial

### 1.2 Configuration Constants

```python
MOVIE_SAMPLE_BASED = "iprgc_test"
IPRGC_TARGET_RATE_HZ = 60.0       # Target binning rate
IPRGC_EXPECTED_BINS = 7200        # Expected bins (= 120 seconds × 60 Hz)
IPRGC_LENGTH_TOLERANCE = 0.10    # ±10% tolerance for valid trials
```

### 1.3 Trial Validation

Trials are validated based on expected length. Only trials within ±10% of the expected 7200 bins are included:

$$\text{min\_expected} = 7200 \times (1 - 0.10) = 6480$$
$$\text{max\_expected} = 7200 \times (1 + 0.10) = 7920$$

Trials outside this range are excluded with a warning.

### 1.4 Firing Rate Calculation (`get_sample_based_firing_rate`)

The function converts spike times to binned firing rate:

```python
def get_sample_based_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    target_rate_hz: float = 60.0,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, int]:
```

**Algorithm**:

1. **Compute bin width** in samples:
   $$\text{samples\_per\_bin} = \frac{\text{acquisition\_rate}}{\text{target\_rate\_hz}} = \frac{20000}{60} \approx 333.33 \text{ samples}$$

2. **Convert spike times to relative coordinates**:
   $$\text{relative\_spikes} = \text{spike\_times} - \text{trial\_start\_sample}$$

3. **Determine number of bins**:
   $$n_{\text{bins}} = \lceil \frac{\text{trial\_end\_sample} - \text{trial\_start\_sample}}{\text{samples\_per\_bin}} \rceil$$

4. **Create bin edges**:
   $$\text{bin\_edges}[i] = i \times \text{samples\_per\_bin}, \quad i = 0, 1, \ldots, n_{\text{bins}}$$

5. **Histogram spikes into bins**:
   ```python
   counts, _ = np.histogram(trial_spikes, bins=bin_edges)
   ```

6. **Convert counts to firing rate** (spikes/second):
   $$\text{firing\_rate}[i] = \frac{\text{counts}[i]}{\text{bin\_duration\_s}} = \text{counts}[i] \times 60$$

### 1.5 Output Format

The processed data is saved as a Parquet file with an `iprgc_test` column containing a 2D array for each unit:

- **Shape**: `(n_trials, n_timepoints)` where `n_timepoints ≈ 7200`
- **Units**: Firing rate in Hz (spikes per second)
- **Sampling rate**: 60 Hz

---

## Stage 2: Quality Index Extraction

### 2.1 Function Signature

```python
def compute_iprgc_qi(
    df: pd.DataFrame,
    movie_col: str = "iprgc_test",
    filter_order: int = 5,
    sampling_rate: float = 60.0,
) -> Tuple[pd.Series, pd.Series]:
```

### 2.2 Time Constants

```python
samples_baseline = int(1 * 60)   # 60 samples (last 1 second for baseline)
samples_start = int(2 * 60)      # 120 samples (start analysis at 2 seconds)
```

### 2.3 Processing Pipeline for iprgc_2hz_QI

For each unit, the following steps are applied:

#### Step 1: Load Trial Data

```python
trials_array = np.vstack(valid_trials)  # Shape: (n_trials, n_timepoints)
```

#### Step 2: Baseline Subtraction

The baseline is computed as the **mean of the last 1 second** (60 samples at 60 Hz):

$$\text{baseline}_i = \text{mean}(\text{trial}_i[-60:])$$
$$\text{trial}_i^{\text{corrected}} = \text{trial}_i - \text{baseline}_i$$

This zeros out the "dark" response at the end of the stimulus.

#### Step 3: Bessel Lowpass Filter (2 Hz)

A 5th-order Bessel lowpass filter with 2 Hz cutoff is applied:

```python
def bessel_lowpass_filter(trace, cutoff_freq=2.0, order=5, sampling_rate=60.0):
    nyquist = 0.5 * sampling_rate  # 30 Hz
    normalized_cutoff = cutoff_freq / nyquist  # 2/30 ≈ 0.0667
    
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False)
    filtered_trace = signal.filtfilt(b, a, trace)  # Zero-phase filtering
    return filtered_trace
```

The 2 Hz cutoff removes fast transients and retains the slow melanopsin-driven response characteristic of ipRGCs.

#### Step 4: Temporal Trimming

Data is trimmed from **2 seconds to the end** of the recording:

```python
trimmed_2hz = filtered_2hz[:, samples_start:]  # From index 120 onwards
```

The first 2 seconds are excluded to avoid onset artifacts.

#### Step 5: Quality Checks

Before computing QI, several validity checks are performed:

1. **Low firing rate check**: If any trial has mean firing rate < 1 Hz → QI = 0
   ```python
   has_low_firing_trial = any(np.mean(raw_trimmed_2hz[i]) < 1.0 for i in range(n_trials))
   ```

2. **Constant trial check**: If any trial has zero standard deviation → QI = 0
   ```python
   has_constant_trial = any(np.std(trimmed_2hz[i]) == 0 for i in range(n_trials))
   ```

3. **Near-zero check**: If max absolute value < 1.0 → QI = 0
   ```python
   all_near_zero = np.max(np.abs(trimmed_2hz)) < 1.0
   ```

#### Step 6: Pearson Correlation Quality Index

The QI is computed using `get_quality_index_pearson()`:

```python
def get_quality_index_pearson(array: np.ndarray) -> float:
    mean_trace = array.mean(axis=0)  # Average across all trials
    
    correlations = []
    for trial in array:
        if np.std(trial) > 0:  # Skip constant trials
            r, _ = stats.pearsonr(trial, mean_trace)
            if not np.isnan(r):
                correlations.append(r)
    
    return np.mean(correlations)
```

**Mathematical Definition**:

$$\text{QI}_{\text{2Hz}} = \frac{1}{N} \sum_{i=1}^{N} r(\text{trial}_i, \bar{\text{trace}})$$

Where:
- $N$ = number of valid trials
- $\bar{\text{trace}}$ = mean trace across all trials
- $r(x, y)$ = Pearson correlation coefficient

$$r(x, y) = \frac{\sum_t (x_t - \bar{x})(y_t - \bar{y})}{\sqrt{\sum_t (x_t - \bar{x})^2} \sqrt{\sum_t (y_t - \bar{y})^2}}$$

### 2.4 Interpretation

| QI Value | Interpretation |
|----------|----------------|
| ~1.0 | High consistency across trials (reliable ipRGC response) |
| ~0.5 | Moderate consistency |
| ~0 | Low consistency or low firing rate |
| NaN | Missing data or processing error |

---

## Complete Data Flow Summary

```
HDF5 File
    │
    ├── units/{unit_id}/spike_times_sectioned/iprgc_test/trials_spike_times
    │       ↓ (spike times in samples at 20 kHz)
    │
    └── get_sample_based_firing_rate()
            ↓ (binned at 60 Hz → ~7200 bins per trial)
            
    Parquet File (iprgc_test column)
            ↓ (2D array: n_trials × n_timepoints)
            
    compute_iprgc_qi()
        │
        ├── Baseline subtraction (mean of last 1 second)
        ├── Bessel lowpass filter (2 Hz cutoff, 5th order)
        ├── Trim (2 seconds to end)
        ├── Quality checks (low FR, constant, near-zero)
        │
        └── Pearson correlation QI
                ↓
                
    iprgc_2hz_QI (scalar per unit)
```

---

## Key Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| Acquisition rate | 20,000 Hz | Raw spike recording rate |
| Target rate | 60 Hz | Firing rate binning rate |
| Expected trial length | 7200 bins | 120 seconds × 60 Hz |
| Length tolerance | ±10% | Valid trial range: 6480-7920 bins |
| Baseline window | Last 1 second | 60 samples for baseline mean |
| Analysis start | 2 seconds | Skip first 120 samples |
| Lowpass cutoff | 2 Hz | Bessel filter cutoff |
| Filter order | 5 | Bessel filter order |
| Minimum firing rate | 1 Hz | Below this → QI = 0 |

---

## File References

- **Stage 1**: `dataframe_phase/load_traces/pipeline_firing_rate.py`
  - `get_sample_based_firing_rate()` - Lines 116-136
  - Trial validation - Lines 186-197, 306-354

- **Stage 2**: `dataframe_phase/extract_feature/extract_feature_step_iprgc.py`
  - `compute_iprgc_qi()` - Lines 221-346
  - `bessel_lowpass_filter()` - Lines 17-54
  - `get_quality_index_pearson()` - Lines 94-135
