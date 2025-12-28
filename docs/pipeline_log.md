# HD-MEA Pipeline Changelog

This document logs all major changes to the HD-MEA data analysis pipeline.
Entries are in reverse chronological order (newest first).

---

## [2025-12-27] DSGC Direction Sectioning

**Change**: Added `section_by_direction()` function to section spike times by moving bar direction for direction-selective ganglion cell (DSGC) analysis.

**Key Features**:
- Section spikes by 8 motion directions × 3 repetitions = 24 trials
- Per-pixel on/off timing from pre-computed dictionary
- Automatic cell center conversion from 15×15 STA grid to 300×300 stimulus space
- Configurable padding around on/off windows (default: 10 frames)
- Safe output with `force` parameter and `output_path` option for testing
- Filter by `unit_ids` parameter for selective processing

**Affected**:
- `hdmea.features.dsgc_direction` - new module with `section_by_direction()` function
- `hdmea.features.__init__` - exports `section_by_direction`, `DirectionSectionResult`, `DIRECTION_LIST`

**HDF5 Structure**:
```
units/{unit_id}/
└── spike_times_sectioned/
    └── moving_h_bar_s5_d8_3x/
        ├── full_spike_times      # UNCHANGED (source data)
        └── direction_section/
            └── {direction}/      # "0", "45", "90", ..., "315"
                ├── trials/
                │   ├── 0         # int64[] spike samples for rep 1
                │   ├── 1         # int64[] spike samples for rep 2
                │   └── 2         # int64[] spike samples for rep 3
                └── section_bounds  # int64[3,2] [start,end] per trial
```

**Usage**:
```python
from hdmea.features import section_by_direction

result = section_by_direction(
    "recording.h5",
    padding_frames=10,      # Frames before/after on/off window
    force=True,             # Overwrite existing data
    output_path="export/",  # Copy source first (for testing)
)
print(f"Processed {result.units_processed} units in {result.elapsed_seconds:.1f}s")
```

**Spec**: `specs/011-dsgc-direction-section/`

---

## [2025-12-18] Spike Triggered Average (STA) Computation

**Change**: Added `compute_sta()` function to compute Spike Triggered Average from noise movie stimuli for receptive field mapping.

**Key Features**:
- Automatic noise movie detection (case-insensitive search for "noise" in movie names)
- Spike time conversion from sampling indices to movie frame numbers
- Vectorized window extraction for high performance
- Multiprocessing support with 80% of CPU cores
- Progress bar during computation
- Edge effect handling (spikes near boundaries excluded)
- Retry logic for failed units (retry once, then skip)
- Configurable `cover_range` parameter (default: -60 to 0 frames)

**Affected**:
- `hdmea.features.sta` - new module with `compute_sta()` function
- `hdmea.features.__init__` - exports `compute_sta`, `STAResult`

**HDF5 Structure**:
```
units/{unit_id}/
└── features/
    └── {noise_movie_name}/
        └── sta                    # 3D array (time × height × width)
            └── .attrs: {n_spikes, n_spikes_excluded, cover_range, dtype_warning}
```

**Usage**:
```python
from hdmea.features import compute_sta

result = compute_sta(
    "artifacts/recording.h5",
    cover_range=(-60, 0),      # 60 frames before spike
    use_multiprocessing=True,  # Use 80% of CPU cores
    frame_rate=15.0,           # Movie frame rate
    force=True,                # Overwrite existing
)
print(f"Processed {result.units_processed} units in {result.elapsed_seconds:.1f}s")
```

**Stimuli Directory**: `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\{movie_name}.npy`

**PR/Branch**: `009-sta-computation`

---

## [2025-12-17] Spike Times Unit Conversion and Stimulation Sectioning

**Change**: Two-part feature implementing spike timestamp standardization and trial-based sectioning:

1. **Spike Times Unit Conversion**: Modified `load_recording()` to convert raw spike timestamps from nanoseconds (10^-9 s) to acquisition sample indices during loading. Formula: `sample_index = round(timestamp_ns × acquisition_rate / 10^9)`. All `spike_times` arrays now use a consistent unit (sample indices at ~20 kHz) with `unit="sample_index"` attribute.

2. **Spike Times Sectioning**: Added `section_spike_times()` function to extract spike timestamps within trial boundaries defined by `section_time`. Stores data in TWO formats:
   - `full_spike_times`: All spikes from all trials combined (sorted, unique)
   - `trials_spike_times/{idx}`: Spikes per individual trial

**Key Features**:
- `trial_repeats` parameter limits number of trials to process (default: 3)
- `pad_margin` tuple `(pre_s, post_s)` extends trial boundaries (default: 2s pre, 0s post)
- `force=False` raises `FileExistsError` if sectioned data exists
- Empty spike_times handled gracefully (stores empty arrays)
- Boundary clamping prevents negative sample indices

**Affected**:
- `hdmea.pipeline.runner.load_recording()` - spike_times now in sample indices
- `hdmea.pipeline.runner._convert_spike_times_to_samples()` - new conversion helper
- `hdmea.io.zarr_store.write_units()` - adds `unit="sample_index"` attribute
- `hdmea.io.spike_sectioning` - new module with `section_spike_times()` function
- `hdmea.io.__init__` - exports `section_spike_times`, `SectionResult`

**Zarr Structure**:
```
units/{unit_id}/
├── spike_times                    # Now in sample indices (was ns)
│   └── .zattrs: {"unit": "sample_index"}
└── spike_times_sectioned/
    └── {movie_name}/
        ├── full_spike_times       # All trials combined
        ├── trials_spike_times/
        │   ├── 0                  # Trial 0 spikes
        │   ├── 1                  # Trial 1 spikes
        │   └── ...
        └── .zattrs: {pad_margin, pre_samples, post_samples, trial_repeats, n_trials}
```

**Usage**:
```python
from hdmea.io import section_spike_times

result = section_spike_times(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    trial_repeats=3,
    pad_margin=(2.0, 0.0),  # 2s pre-margin, 0s post-margin
    force=False,
)
print(f"Processed {result.units_processed} units, {len(result.movies_processed)} movies")
```

**Migration**: 
- Existing Zarr files with spike_times in nanoseconds remain unchanged
- Re-running `load_recording(force=True)` will regenerate with sample indices
- Sectioned data is optional and additive

**PR/Branch**: `006-spike-times-sectioning`

---

## [2025-12-17] Unify Section Time to Acquisition Sample Indices

**Change**: Modified both `add_section_time_analog()` and `add_section_time()` to output section times in **acquisition sample indices** instead of display frame indices. This provides a unified unit across all section time data for consistent downstream processing.

**Key Changes**:
- `add_section_time_analog()`: Detects peaks in `raw_ch1` and stores sample indices directly (no frame_timestamps required)
- `add_section_time()`: Converts computed display frames to sample indices via `frame_timestamps` before storing
- All section_time arrays now use acquisition sample indices (int64)
- To convert to time: `time_seconds = sample_index / acquisition_rate`

**Why This Matters**:
- Consistent unit across analog and playlist-based section times
- Direct slicing of raw signals without frame conversion
- `frame_timestamps` no longer required for analog detection (not applicable during continuous stimulation)

**Affected**:
- `hdmea.io.section_time.add_section_time_analog()` - simplified, no frame_timestamps dependency
- `hdmea.io.section_time.add_section_time()` - outputs sample indices
- Tests updated to expect sample indices (~200000 instead of ~500 for 10s at 20kHz)
- Zarr structure: `stimulus/section_time/{movie_name}` values are now sample indices

**Usage**:
```python
from hdmea.io.section_time import add_section_time_analog

success = add_section_time_analog(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    threshold_value=1e5,  # Inspect np.diff(raw_ch1) to determine
    movie_name="iprgc_test",
    plot_duration=120.0,  # 2 minute windows (in seconds)
    repeat=3,  # Use first 3 trials only
)

# Section times are now in acquisition samples
# First trial at 10s: start_sample ≈ 200000 (at 20kHz)
```

**Migration**: Existing Zarr files with old section_time data (in display frame indices) remain unchanged. New section_time data will use acquisition sample indices.

**Backward Compatibility**: Downstream code should check `section_time.attrs.get("unit")` or assume sample indices for newly generated data.

**PR/Branch**: `005-analog-section-time`

---

## [2025-12-16] Add Section Time Loading

**Change**: Added `add_section_time()` function to load movie section timing from playlist and movie_length CSV configuration files. Computes frame boundaries for each movie and stores them in Zarr under `stimulus/section_time/`.

**Affected**:
- `hdmea.io.section_time` (new module)
- `hdmea.pipeline.__init__` (exports new function)
- Zarr structure: new `stimulus/section_time/` and `stimulus/light_template/` groups

**Migration**: No migration needed - this is a new optional feature.

**PR/Branch**: `004-load-section-time`

---

## [2025-12-16] Pipeline Documentation Requirements

**Change**: Added constitution requirement for pipeline documentation files (`pipeline_explained.md` and `pipeline_log.md`). All major pipeline changes MUST be logged.

**Affected**:
- `.specify/memory/constitution.md` (new Pipeline Documentation section)
- `docs/pipeline_explained.md` (new file)
- `docs/pipeline_log.md` (this file)

**Migration**: No migration needed - documentation is additive.

**PR/Branch**: `004-load-section-time`

---

## [2025-12-14] Initial Pipeline Implementation

**Change**: Initial implementation of the HD-MEA data analysis pipeline with two-stage architecture:
- Stage 1: `load_recording()` - Load CMCR/CMTR files into Zarr
- Stage 2: `extract_features()` - Extract registered features from Zarr

**Affected**:
- `hdmea.pipeline.runner` - Pipeline runner with caching
- `hdmea.io.cmcr` - CMCR file loading
- `hdmea.io.cmtr` - CMTR file loading
- `hdmea.io.zarr_store` - Zarr read/write operations
- `hdmea.features.registry` - Feature extractor registry

**Migration**: N/A - initial implementation.

**PR/Branch**: `001-hdmea-modular-pipeline`

---

## Template for New Entries

```markdown
## [YYYY-MM-DD] Brief Title

**Change**: Description of what changed

**Affected**: List of affected modules/components

**Migration**: Steps needed to update existing code (if applicable)

**PR/Branch**: Reference to PR or branch (if applicable)
```

