# HD-MEA Pipeline Changelog

This document logs all major changes to the HD-MEA data analysis pipeline.
Entries are in reverse chronological order (newest first).

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

