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
│  │  │   │   ├── spike_times       # Spike timestamps (µs)           │    │
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
│  │  • Detect peaks in raw_ch1 signal (stimulus onsets)            │     │
│  │  • Convert sample indices to display frame indices             │     │
│  │  • Apply plot_duration and repeat parameters                   │     │
│  │  • Write to stimulus/section_time/{movie_name}                 │     │
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
add_section_time_analog(
    zarr_path=result.zarr_path,
    threshold_value=3e6,  # Inspect signal to determine
    movie_name="iprgc_test",
    plot_duration=120.0,  # 2 minute windows
    repeat=3,  # Use first 3 trials only
)

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
| **Acquisition Samples** | Sample index | ~20 kHz | Raw sensor data, spike times | `spike_times`, `raw_ch1`, `raw_ch2` |
| **Display Frames** | Frame index | ~45 Hz* | Section boundaries, firing rates | `section_time`, `firing_rate_10hz` |
| **10 Hz Bins** | Bin index | 10 Hz | Legacy compatibility | Deprecated in current pipeline |
| **Time (seconds)** | Seconds | - | Duration parameters | `plot_duration`, `recording_duration_s` |

*Display frame rate varies by recording and is detected from `raw_ch2` signal

### Key Metadata Arrays

| Array | Location | Shape | Purpose |
|-------|----------|-------|---------|
| `frame_timestamps` | `metadata/frame_timestamps` | (N_frames,) | Maps display frame index → acquisition sample index |
| `acquisition_rate` | `metadata/acquisition_rate` | scalar | Samples per second for raw data (typically 20000.0 Hz) |
| `sample_interval` | `metadata/sample_interval` | scalar | Seconds per sample (1 / acquisition_rate) |
| `frame_time` | `stimulus/frame_time/default` | (N_frames,) | Display frame timestamps in seconds |

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
| **Section time units** | 10 Hz bin indices | Display frame indices (variable rate) |
| **Light reference** | Downsampled `10hz_ch1` | Full resolution `raw_ch1` at acquisition rate |
| **Timestamp detection** | Fixed 10 Hz | Peak detection in `raw_ch2` derivative |

#### Section Time Coordinate System

All `section_time` arrays use **display frame indices**, not 10 Hz bins:

```python
# section_time array format
section_time = np.array([
    [start_frame, end_frame],  # Trial 1
    [start_frame, end_frame],  # Trial 2
    # ...
])

# To extract firing rate for a movie section:
unit_firing_rate = root["units"]["unit_000"]["firing_rate_10hz"]
start, end = section_time[trial_idx]
trial_response = unit_firing_rate[start:end]
```

**⚠️ Warning**: Do NOT assume section_time values are at 10 Hz. They reference display frames at the recording's actual frame rate (~45 Hz typical).

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

The `add_section_time_analog()` function performs the following conversions:

```
1. Detect peaks in raw_ch1 (acquisition samples)
   └─→ onset_samples [shape: (N_trials,)]

2. Calculate end times (seconds → samples)
   └─→ end_samples = onset_samples + (plot_duration * acquisition_rate)

3. Convert to display frames (samples → frames)
   └─→ start_frames = _sample_to_nearest_frame(onset_samples, frame_timestamps)
   └─→ end_frames = _sample_to_nearest_frame(end_samples, frame_timestamps)

4. Store section_time (display frame indices)
   └─→ section_time = [[start_frames[i], end_frames[i]] for i in trials]
```

This ensures analog-detected sections are compatible with playlist-based sections.

### Best Practices

1. **Always use `frame_timestamps` for conversions** - Don't assume fixed frame rates
2. **Check acquisition_rate** - Don't hardcode 20 kHz, read from metadata
3. **Nearest frame rounding** - Use `_sample_to_nearest_frame()` for sample→frame conversion
4. **Section time units** - Always in display frame indices, not 10 Hz bins
5. **Inspect signals** - For analog detection, always inspect derivative statistics before choosing threshold

### Common Edge Cases

#### Raw Signal Extends Beyond Frame Timestamps

Sometimes `raw_ch1` extends beyond the coverage of `frame_timestamps`:

```
frame_timestamps: covers samples 0 to 18,846,811
raw_ch1: extends to sample 23,793,999
```

**Impact**: Peaks detected beyond `frame_timestamps[-1]` cannot be accurately converted to display frame indices.

**Behavior**: `add_section_time_analog()` filters out such peaks with a warning:
```
Filtered out 2 peaks beyond frame_timestamps coverage (sample > 18,846,811). 
Raw signal extends beyond frame sync range.
```

**Cause**: The frame sync signal (`raw_ch2`) may have stopped before the recording ended, or the recording includes pre/post-stimulus baseline without visual stimuli.

## Related Documentation

- [Pipeline Log](pipeline_log.md) - Changelog of major pipeline changes
- [Constitution](../.specify/memory/constitution.md) - Project principles and standards

