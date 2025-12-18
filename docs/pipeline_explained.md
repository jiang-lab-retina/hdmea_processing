# HD-MEA Pipeline: Current Flow

**Last Updated**: 2025-12-16  
**Pipeline Version**: 0.1.0

## Overview

The HD-MEA Data Analysis Pipeline processes high-density multi-electrode array (HD-MEA) recordings
through a two-stage workflow: data loading and feature extraction. The pipeline produces self-describing
Zarr archives that contain all information needed for downstream analysis.

**Key Insight**: The pipeline works with multiple time coordinate systems (acquisition samples at ~20 kHz, 
display frames at ~45 Hz, and time in seconds). Understanding these coordinate systems and their conversions 
is essential for correct data analysis. See [Time Units and Coordinate Systems](#time-units-and-coordinate-systems) 
for details.

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL DATA                                  │
│                                                                          │
│   ┌──────────┐     ┌──────────┐     ┌──────────────┐   ┌─────────────┐  │
│   │  .cmcr   │     │  .cmtr   │     │ playlist.csv │   │movie_length │  │
│   │  (raw)   │     │ (spikes) │     │              │   │   .csv      │  │
│   └────┬─────┘     └────┬─────┘     └──────┬───────┘   └──────┬──────┘  │
│        │                │                   │                  │         │
└────────┼────────────────┼───────────────────┼──────────────────┼─────────┘
         │                │                   │                  │
         ▼                ▼                   │                  │
┌─────────────────────────────────────────────┼──────────────────┼─────────┐
│                    STAGE 1: LOAD RECORDING  │                  │         │
│                                             │                  │         │
│  ┌────────────────────────────────────────┐ │                  │         │
│  │         load_recording()               │ │                  │         │
│  │                                        │ │                  │         │
│  │  • Load CMCR (light reference)         │ │                  │         │
│  │  • Load CMTR (spike-sorted units)      │ │                  │         │
│  │  • Extract timing metadata             │ │                  │         │
│  │  • Detect frame timestamps             │ │                  │         │
│  │  • Compute firing rates (10Hz)         │ │                  │         │
│  │  • Convert spike_times to sample units │ │                  │         │
│  │  • Write to Zarr archive               │ │                  │         │
│  └───────────────────┬────────────────────┘ │                  │         │
│                      │                      │                  │         │
└──────────────────────┼──────────────────────┼──────────────────┼─────────┘
                       │                      │                  │
                       ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZARR ARCHIVE                                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  {dataset_id}.zarr                                               │    │
│  │                                                                  │    │
│  │  ├── units/                    # Spike-sorted units              │    │
│  │  │   ├── {unit_id}/                                              │    │
│  │  │   │   ├── spike_times       # Spike timestamps (sample idx)   │    │
│  │  │   │   ├── spike_times_sectioned/  # Sectioned by movie trial  │    │
│  │  │   │   ├── waveform          # Average waveform                │    │
│  │  │   │   ├── firing_rate_10hz  # Binned firing rate              │    │
│  │  │   │   └── features/         # Extracted features (Stage 2)    │    │
│  │  │                                                               │    │
│  │  ├── stimulus/                 # Stimulus information            │    │
│  │  │   ├── light_reference/      # Light sensor data               │    │
│  │  │   ├── frame_time/           # Frame timestamps                │    │
│  │  │   ├── section_time/         # Movie section boundaries        │    │
│  │  │   └── light_template/       # Averaged light templates        │    │
│  │  │                                                               │    │
│  │  └── metadata/                 # Recording metadata              │    │
│  │      ├── acquisition_rate      # Sampling rate (Hz)              │    │
│  │      ├── sample_interval       # Time per sample (s)             │    │
│  │      ├── frame_timestamps      # Frame indices                   │    │
│  │      └── sys_meta/             # Raw file metadata               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: FEATURE EXTRACTION                           │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │         extract_features()                                      │     │
│  │                                                                 │     │
│  │  For each registered feature extractor:                         │     │
│  │    1. Check cache (skip if already extracted)                   │     │
│  │    2. Validate required inputs exist                            │     │
│  │    3. Extract features for each unit                            │     │
│  │    4. Write results to units/{unit_id}/features/{feature}/      │     │
│  │                                                                 │     │
│  │  Currently Registered Features:                                 │     │
│  │    • frif - Full Recording Integrated Features                  │     │
│  │    • step_up - ON/OFF response indices                          │     │
│  │    • baseline_127 - Baseline activity features                  │     │
│  │    • chirp - Frequency response features                        │     │
│  │    • dense_noise - Receptive field features                     │     │
│  │    • moving_bar - Direction selectivity                         │     │
│  │    • green_blue - Chromatic response features                   │     │
│  │    • rgc_classifier - RGC type classification                   │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL: SECTION TIME LOADING                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │         add_section_time()  [Playlist-based]                    │     │
│  │                                                                 │     │
│  │  • Load playlist.csv and movie_length.csv                       │     │
│  │  • Compute frame boundaries for each movie                      │     │
│  │  • Extract and average light templates                          │     │
│  │  • Write to stimulus/section_time/ and stimulus/light_template/ │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │         add_section_time_analog()  [Peak detection]             │     │
│  │                                                                 │     │
│  │  • Detect peaks in raw_ch1 derivative (stimulus onsets)        │     │
│  │  • Store section times as acquisition sample indices           │     │
│  │  • Extract and average light templates across trials           │     │
│  │  • Write to stimulus/section_time/ and stimulus/light_template/│     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │         section_spike_times()  [Spike sectioning]              │     │
│  │                                                                 │     │
│  │  • Extract spikes within trial boundaries (with padding)       │     │
│  │  • Store full_spike_times (all trials combined)                │     │
│  │  • Store trials_spike_times (per-trial arrays)                 │     │
│  │  • Supports pad_margin (pre/post) for extended boundaries      │     │
│  │  • Write to units/{unit_id}/spike_times_sectioned/{movie}/     │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Entry Points

### Primary Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `load_recording()` | `hdmea.pipeline` | Stage 1: Load raw data into Zarr |
| `extract_features()` | `hdmea.pipeline` | Stage 2: Extract features from Zarr |
| `add_section_time()` | `hdmea.io.section_time` | Add movie section timing (playlist-based) |
| `add_section_time_analog()` | `hdmea.io.section_time` | Add movie section timing (peak detection) |
| `section_spike_times()` | `hdmea.io.spike_sectioning` | Section spike times by trial boundaries |
| `run_flow()` | `hdmea.pipeline` | Run a named flow (Stage 1 + Stage 2) |

### Example Usage

```python
from hdmea.pipeline import load_recording, extract_features
from hdmea.io.section_time import add_section_time, add_section_time_analog

# Stage 1: Load recording
result = load_recording(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
    dataset_id="REC_2023-12-07",
)

# Optional: Add section time metadata (before or after feature extraction)
# Method 1: Playlist-based (for predefined stimuli)
add_section_time(
    zarr_path=result.zarr_path,
    playlist_name="set6a",
    repeats=2,
)

# Method 2: Analog peak detection (for post-hoc timing)
# Note: Inspect np.diff(raw_ch1) to determine appropriate threshold
add_section_time_analog(
    zarr_path=result.zarr_path,
    threshold_value=1e6,  # Determined from signal inspection
    movie_name="iprgc_test",
    plot_duration=120.0,  # 2 minute windows (in seconds)
    repeat=2,  # Limit section_time to first 2 trials (light_template uses all)
)
# Output: section_time in acquisition sample indices, light_template averaged

# Stage 2: Extract features
extract_result = extract_features(
    zarr_path=result.zarr_path,
    features=["frif", "step_up", "chirp"],
)
```

## Data Flow Summary

1. **Input**: Raw `.cmcr` (sensor data) and `.cmtr` (spike-sorted) files
2. **Stage 1**: Load and convert to standardized Zarr format
3. **Stage 2**: Extract features for each unit, write back to Zarr
4. **Optional**: Add section timing metadata from playlist configuration
5. **Output**: Self-contained Zarr archive with all data and features

## Configuration

### Flow Configurations

Named flows are defined in `config/flows/{flow_name}.json` and specify:
- Which features to extract
- Default parameters for extractors
- Stage-specific settings

### Default Paths

| Path | Purpose |
|------|---------|
| `artifacts/` | Default output directory for Zarr archives |
| `config/flows/` | Flow configuration files |
| `//Jiangfs1/.../playlist.csv` | Default playlist configuration |
| `//Jiangfs1/.../movie_length.csv` | Default movie length configuration |

## Caching Behavior

- Stage 1 results are cached by `params_hash` - rerunning with same parameters skips loading
- Stage 2 features are cached per-unit - only extracts missing features
- Use `force=True` to override caching and recompute

## Time Units and Coordinate Systems

The pipeline works with multiple time coordinate systems. Understanding their relationships is critical for correct data analysis.

### Time Coordinate Systems

| System | Unit | Rate | Usage | Example |
|--------|------|------|-------|---------|
| **Acquisition Samples** | Sample index | ~20 kHz | Raw data, spike times, **section_time** | `spike_times`, `raw_ch1`, `section_time` |
| **Display Frames** | Frame index | ~45 Hz* | Frame sync, internal computations | `frame_timestamps`, `firing_rate_10hz` |
| **10 Hz Bins** | Bin index | 10 Hz | Legacy compatibility | Deprecated in current pipeline |
| **Time (seconds)** | Seconds | - | Duration parameters | `plot_duration`, `recording_duration_s` |

*Display frame rate varies by recording and is detected from `raw_ch2` signal

**Important**: As of 2025-12-17, all `section_time` arrays use acquisition sample indices for unified time representation.

### Key Metadata Arrays

| Array | Location | Shape | Purpose |
|-------|----------|-------|---------|
| `frame_timestamps` | `metadata/frame_timestamps` | (N_frames,) | Maps display frame index → acquisition sample index |
| `acquisition_rate` | `metadata/acquisition_rate` | scalar | Samples per second for raw data (typically 20000.0 Hz) |
| `sample_interval` | `metadata/sample_interval` | scalar | Seconds per sample (1 / acquisition_rate) |
| `frame_time` | `stimulus/frame_time/default` | (N_frames,) | Display frame timestamps in seconds |
| `section_time` | `stimulus/section_time/{movie}` | (N_trials, 2) | Trial boundaries as [start_sample, end_sample] |
| `light_template` | `stimulus/light_template/{movie}` | (M,) | Averaged light reference trace (M = duration in samples) |

### Time Conversions

#### Display Frame → Acquisition Sample

```python
# Using frame_timestamps lookup
frame_idx = 1000
sample_idx = frame_timestamps[frame_idx]
```

**Used by**: `add_section_time()` to extract light templates from `raw_ch1`

#### Acquisition Sample → Display Frame (Nearest)

```python
# Using _sample_to_nearest_frame() helper
from hdmea.io.section_time import _sample_to_nearest_frame

onset_sample = 450000  # Peak detected in raw_ch1
onset_frame = _sample_to_nearest_frame(onset_sample, frame_timestamps)
```

**Used by**: `add_section_time_analog()` for peak detection results

#### Seconds → Acquisition Samples

```python
# Using acquisition_rate
duration_seconds = 120.0  # 2 minutes
duration_samples = int(duration_seconds * acquisition_rate)
```

**Used by**: `add_section_time_analog()` for `plot_duration` parameter

#### Display Frames → Seconds

```python
# Using frame_time array
frame_idx = 1000
time_seconds = frame_time[frame_idx]
```

**Used by**: Plotting and visualization

### Important Distinctions

#### Legacy vs Current Pipeline

| Aspect | Legacy Code | Current Pipeline |
|--------|-------------|------------------|
| **Frame rate** | Assumed 10 Hz constant | Detected from `raw_ch2` (~45 Hz typical) |
| **Section time units** | 10 Hz bin indices | **Acquisition sample indices** (~20 kHz) |
| **Light reference** | Downsampled `10hz_ch1` | Full resolution `raw_ch1` at acquisition rate |
| **Timestamp detection** | Fixed 10 Hz | Peak detection in `raw_ch2` derivative |

#### Section Time Coordinate System

All `section_time` arrays use **acquisition sample indices** (at ~20 kHz), providing maximum temporal resolution:

```python
# section_time array format - values are acquisition sample indices
section_time = np.array([
    [start_sample, end_sample],  # Trial 1: ~200000, ~2600000 (10s to 130s at 20kHz)
    [start_sample, end_sample],  # Trial 2
    # ...
])

# To convert to time in seconds:
acquisition_rate = float(root["metadata"]["acquisition_rate"][()])
start_sample, end_sample = section_time[trial_idx]
start_time_s = start_sample / acquisition_rate
end_time_s = end_sample / acquisition_rate

# To slice raw light reference signal directly:
raw_ch1 = root["stimulus"]["light_reference"]["raw_ch1"]
trial_signal = raw_ch1[start_sample:end_sample]
```

**⚠️ Note**: Section time values are large integers (~millions) representing acquisition samples, not frame indices. To convert to display frames if needed, use `frame_timestamps` for lookup.

### Frame Rate Detection

The actual display frame rate is detected during `load_recording()`:

1. **Frame sync signal**: `raw_ch2` contains frame synchronization pulses
2. **Peak detection**: `get_frame_timestamps()` finds transitions in `raw_ch2` derivative
3. **Frame timestamps**: Peak locations stored as acquisition sample indices
4. **Frame rate**: Variable depending on display hardware and stimulus

**Example detection output**:
```
Detected 54,366 frame timestamps from light reference
Display rate: ~45.7 Hz
```

### Analog Section Time Detection

The `add_section_time_analog()` function stores acquisition sample indices directly:

```
1. Detect peaks in raw_ch1 derivative (acquisition samples)
   └─→ onset_samples [shape: (N_trials,)]

2. Calculate end times (seconds → samples)
   └─→ end_samples = onset_samples + int(plot_duration * acquisition_rate)
   └─→ Clip to signal length: end_samples = min(end_samples, len(raw_ch1) - 1)

3. Store section_time (acquisition sample indices directly)
   └─→ section_time = [[onset_samples[i], end_samples[i]] for i in trials]

4. Extract and average light templates from all detected trials
   └─→ light_template = mean(raw_ch1[onset:end] for each trial)
```

**Note**: `frame_timestamps` is NOT required for analog section time detection.
The function works directly with acquisition sample indices from `raw_ch1`.

### Best Practices

1. **Check acquisition_rate** - Don't hardcode 20 kHz, read from metadata
2. **Section time units** - All `section_time` arrays use acquisition sample indices
3. **Convert to time** - Use `time_s = sample_index / acquisition_rate` for time conversion
4. **Convert to frames** - Use `frame_timestamps` lookup if display frame indices are needed
5. **Inspect signals** - For analog detection, always inspect `np.diff(raw_ch1)` to choose threshold

### Common Edge Cases

#### Section Extends Beyond Signal Length

When `plot_duration` is long, sections may extend beyond the recorded signal:

```
raw_ch1 length: 23,794,000 samples (≈1190 seconds at 20 kHz)
plot_duration: 120 seconds → 2,400,000 samples per trial
Last onset: sample 21,357,695 → end would be 23,757,695 (within range ✓)
```

**Behavior**: `add_section_time_analog()` clips end samples to signal length with a warning:
```
2 section(s) truncated at signal boundary (end sample clipped to 23,793,999)
```

**Light template handling**: Truncated sections are included in template averaging using `np.nanmean` with `zip_longest` to handle variable lengths.

## Related Documentation

- [Pipeline Log](pipeline_log.md) - Changelog of major pipeline changes
- [Constitution](../.specify/memory/constitution.md) - Project principles and standards

