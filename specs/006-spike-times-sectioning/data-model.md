# Data Model: Spike Times Unit Conversion and Stimulation Sectioning

**Date**: 2025-12-17  
**Spec**: [spec.md](./spec.md)  
**Plan**: [plan.md](./plan.md)

---

## Overview

This document defines the data model changes for spike times unit conversion and stimulation sectioning. Changes affect the `units` group structure in Zarr archives.

---

## Modified Entities

### 1. Unit spike_times (Modified)

The existing `spike_times` array changes its unit from nanoseconds (raw CMTR) to acquisition sample indices.

| Field | Type | Unit (Before) | Unit (After) | Description |
|-------|------|---------------|--------------|-------------|
| `spike_times` | uint64[] | ns (nanoseconds) | sample index | Spike timestamps |

**Location in Zarr**: `units/{unit_id}/spike_times`

**Conversion Formula**:
```python
sample_index = round(timestamp_ns * acquisition_rate / 1e9)
```

Where:
- `timestamp_ns`: Raw timestamp from CMTR in nanoseconds (10^-9 s)
- `acquisition_rate`: Sampling rate in Hz (typically 20000)
- Result is rounded to nearest integer

**Example**:
- Raw timestamp: 50,000,000 ns (50 ms)
- acquisition_rate: 20000 Hz
- Result: round(50,000,000 × 20000 / 10^9) = 1000 (sample index)

---

## New Entities

### 2. Sectioned Spike Times (New)

Spike timestamps extracted based on section_time boundaries, stored in TWO formats:
- **full_spike_times**: All spikes from all trials combined
- **trials_spike_times**: Spikes split per trial

**Location in Zarr**: `units/{unit_id}/spike_times_sectioned/{movie_name}/`

**Structure**:
```
units/{unit_id}/
├── spike_times                  # (N,) uint64 - full recording spike times
├── waveform                     # (T,) float32 - mean waveform
├── firing_rate_10hz             # (M,) float32 - binned firing rate
├── features/                    # Feature extraction outputs
└── spike_times_sectioned/       # NEW: Group for sectioned data
    ├── movie_A/
    │   ├── full_spike_times     # (M,) int64 - ALL spikes from all trials
    │   └── trials_spike_times/  # Group for per-trial data
    │       ├── 0                # (K0,) int64 - spikes in trial 0
    │       ├── 1                # (K1,) int64 - spikes in trial 1
    │       └── 2                # (K2,) int64 - spikes in trial 2
    └── movie_B/
        ├── full_spike_times
        └── trials_spike_times/
            └── ...
```

**Array Specifications**:

| Array | Shape | Dtype | Unit | Description |
|-------|-------|-------|------|-------------|
| `full_spike_times` | (M,) | int64 | samples (absolute) | All spike times from all trials combined |
| `trials_spike_times/{idx}` | (K,) | int64 | samples (absolute) | Spike times for trial idx |

**Spike Extraction Logic**:
```python
# Convert pad_margin (seconds) to samples
pre_samples = int(pad_margin[0] * acquisition_rate)   # e.g., 2.0 * 20000 = 40000 samples
post_samples = int(pad_margin[1] * acquisition_rate)  # e.g., 0.0 * 20000 = 0 samples

# For each movie, process up to trial_repeats trials
full_spikes = []
trials_spikes = {}

for trial_idx in range(min(trial_repeats, len(section_time[movie_name]))):
    trial_start, trial_end = section_time[movie_name][trial_idx]
    
    # Apply padding and clamp to valid range
    padded_start = max(0, trial_start - pre_samples)
    padded_end = trial_end + post_samples  # Clamped to max spike_time if needed
    
    mask = (spike_times >= padded_start) & (spike_times < padded_end)
    trial_spikes = spike_times[mask]
    
    # Store per-trial
    trials_spikes[trial_idx] = trial_spikes
    
    # Accumulate for full
    full_spikes.extend(trial_spikes)

# Store combined (sorted, unique)
full_spike_times = np.sort(np.unique(np.array(full_spikes)))
```

**Attributes** (per movie group):

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_trials` | int | Number of trials processed |
| `trial_repeats` | int | trial_repeats parameter used |
| `pad_margin` | Tuple[float, float] | Padding margins in seconds (pre, post) |
| `pre_samples` | int | Pre-margin in samples (pad_margin[0] × acquisition_rate) |
| `post_samples` | int | Post-margin in samples (pad_margin[1] × acquisition_rate) |
| `section_time_source` | str | Path to section_time used |
| `created_at` | str | ISO 8601 timestamp |

**Validation Rules**:
- Each array contains only spikes where: `(start - pre_samples) <= spike_time < (end + post_samples)`
- Empty trials/movies store empty arrays (shape (0,), dtype int64, NOT omitted)
- Trial indices are integers (0, 1, 2, ...) matching section_time row indices
- `full_spike_times` is sorted in ascending order

---

## Relationships

```
                                     spike_times_sectioned/{movie}/
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              full_spike_times    trials_spike_times/
                                    ▲                   │
                                    │               ┌───┴───┐
                                    │               0   1   2  ...
section_time/{movie_name}  ────────────► spike_times (sample indices)
   [start, end] pairs                         │
        │                                     │
   (N_trials, 2)                         (N_spikes,)
```

**Dependency Chain**:
1. `section_time/{movie}` must exist before sectioning
2. `spike_times` must be in sample units before sectioning
3. `trial_repeats` parameter controls how many trials to process

---

## Complete Zarr Schema (Post-Implementation)

```
{dataset_id}.zarr/
├── .zattrs                      # Root metadata
├── metadata/
│   ├── acquisition_rate         # float64 - samples per second
│   ├── sample_interval          # float64 - seconds per sample
│   ├── frame_timestamps         # uint64[] - sample indices of frames
│   └── ...
├── stimulus/
│   ├── light_reference/
│   │   ├── raw_ch1              # float32[] - full acquisition rate
│   │   └── raw_ch2              # float32[] - full acquisition rate
│   ├── section_time/
│   │   ├── movie_A              # int64[N, 2] - [start, end] samples
│   │   └── movie_B
│   └── light_template/
│       └── ...
└── units/
    └── unit_000/
        ├── .zattrs              # Unit metadata
        ├── spike_times          # uint64[N] - MODIFIED: now in sample units
        ├── waveform             # float32[T]
        ├── firing_rate_10hz     # float32[M]
        ├── features/            # Feature outputs
        └── spike_times_sectioned/  # NEW
            └── movie_A/
                ├── .zattrs      # Movie metadata (n_trials, trial_repeats, pad_margin)
                ├── full_spike_times    # int64[M] - all trials combined
                └── trials_spike_times/ # Per-trial group
                    ├── 0        # int64[K0] - trial 0 spikes
                    ├── 1        # int64[K1] - trial 1 spikes
                    └── 2        # int64[K2] - trial 2 spikes
```

---

## Time Coordinate System Alignment

After this feature, all time-related data uses acquisition sample indices:

| Data | Location | Unit | Rate |
|------|----------|------|------|
| `spike_times` | `units/{unit_id}/spike_times` | sample index | ~20 kHz |
| `section_time` | `stimulus/section_time/{movie}` | sample index | ~20 kHz |
| `frame_timestamps` | `metadata/frame_timestamps` | sample index | ~20 kHz |
| `full_spike_times` | `.../{movie}/full_spike_times` | sample index (absolute) | ~20 kHz |
| `trials_spike_times/{idx}` | `.../{movie}/trials_spike_times/{idx}` | sample index (absolute) | ~20 kHz |

---

## Migration Notes

- **No automatic migration**: Existing Zarr files retain their original spike_times units
- **Detection**: Old files have spike_times in ns (values ~10^12); new files have sample indices (values ~10^6-10^9)
- **Recommendation**: Re-run `load_recording()` with `force=True` to regenerate with new units
