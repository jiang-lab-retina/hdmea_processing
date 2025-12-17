# HD-MEA Pipeline: Current Flow

**Last Updated**: 2025-12-16  
**Pipeline Version**: 0.1.0

## Overview

The HD-MEA Data Analysis Pipeline processes high-density multi-electrode array (HD-MEA) recordings
through a two-stage workflow: data loading and feature extraction. The pipeline produces self-describing
Zarr archives that contain all information needed for downstream analysis.

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
│  │         add_section_time()                                      │     │
│  │                                                                 │     │
│  │  • Load playlist.csv and movie_length.csv                       │     │
│  │  • Compute frame boundaries for each movie                      │     │
│  │  • Extract and average light templates                          │     │
│  │  • Write to stimulus/section_time/ and stimulus/light_template/ │     │
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
| `add_section_time()` | `hdmea.pipeline` | Add movie section timing metadata |
| `run_flow()` | `hdmea.pipeline` | Run a named flow (Stage 1 + Stage 2) |

### Example Usage

```python
from hdmea.pipeline import load_recording, extract_features, add_section_time

# Stage 1: Load recording
result = load_recording(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
    dataset_id="REC_2023-12-07",
)

# Optional: Add section time metadata (before or after feature extraction)
add_section_time(
    zarr_path=result.zarr_path,
    playlist_name="set6a",
    repeats=2,
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

## Related Documentation

- [Pipeline Log](pipeline_log.md) - Changelog of major pipeline changes
- [Constitution](../.specify/memory/constitution.md) - Project principles and standards

