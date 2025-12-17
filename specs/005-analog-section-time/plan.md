# Implementation Plan: Analog Section Time Detection

**Branch**: `005-analog-section-time` | **Date**: 2025-12-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-analog-section-time/spec.md`

## Summary

Implement automatic detection of stimulus onset times from analog light reference signals (`raw_ch1`) and unify all section_time storage to use acquisition sample indices. This involves:
1. Modifying `add_section_time_analog()` to detect peaks in `raw_ch1` and store sample indices directly
2. Modifying `add_section_time()` (playlist-based) to convert display frames to sample indices
3. Updating tests to expect the new unified unit

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: zarr, numpy, scipy.signal.find_peaks  
**Storage**: Zarr (section_time arrays under `stimulus/section_time/{movie_name}`)  
**Testing**: pytest with synthetic fixtures  
**Target Platform**: Cross-platform (Windows/Linux)  
**Project Type**: Single Python package (`src/hdmea/`)  
**Performance Goals**: < 5 seconds for typical recordings (20M+ samples)  
**Constraints**: Must not break existing pipeline; existing data acceptable as-is  
**Scale/Scope**: Typical raw_ch1: 20-50M samples; section_time: 1-100 trials

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Implementation in `src/hdmea/io/section_time.py` |
| II. Modular Subpackage Layout | ✅ PASS | `io/` is correct location for data loading functions |
| III. Explicit I/O and Pure Functions | ✅ PASS | Functions take explicit parameters, return clear outputs |
| IV. Single Zarr Artifact | ✅ PASS | Section_time stored in existing Zarr recording artifact |
| V. Data Format Standards | ✅ PASS | Zarr for hierarchical data, int64 arrays |
| VI. No Hidden Global State | ✅ PASS | Uses logger, no global mutable state |
| VII. Independence from Legacy | ✅ PASS | Reimplementation, no legacy imports |
| Feature Registry Pattern | N/A | Not a feature extractor (IO function) |
| Testing Requirements | ✅ PASS | Unit tests with synthetic data required |

**Gate Status**: ✅ PASSED - No violations

## Project Structure

### Documentation (this feature)

```text
specs/005-analog-section-time/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contracts)
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/hdmea/
├── io/
│   ├── __init__.py          # Export add_section_time_analog
│   ├── section_time.py      # Main implementation (MODIFY)
│   └── zarr_store.py        # Zarr utilities (no changes needed)
├── utils/
│   └── exceptions.py        # MissingInputError (existing)
└── pipeline/
    └── runner.py            # May need update if it calls section_time

tests/
├── unit/
│   └── test_section_time.py # Update expected units (MODIFY)
├── fixtures/
│   └── section_time/
│       └── analog_fixture.py # Synthetic pulse generation (existing)
└── integration/
    └── test_pipeline.py     # May need update
```

**Structure Decision**: Single project structure. All changes are in existing `src/hdmea/io/` module with corresponding test updates.

## Complexity Tracking

No violations requiring justification.

---

## Phase 0: Research

### R1: Peak Detection Algorithm

**Question**: What peak detection approach works best for light reference signals?

**Decision**: Use `scipy.signal.find_peaks()` on the derivative of raw_ch1

**Rationale**: 
- Already proven in existing `_detect_analog_peaks()` implementation
- Works well for step-function light pulses (sharp transitions)
- Threshold parameter allows tuning for different signal amplitudes

**Alternatives Considered**:
- Template matching: More complex, not needed for simple step transitions
- Wavelet detection: Overkill for step functions
- Zero-crossing detection: Less robust to noise

### R2: Unit Conversion for Playlist Section Time

**Question**: How to convert playlist display frames to acquisition samples?

**Decision**: Use `frame_timestamps[frame_index]` lookup

**Rationale**:
- `frame_timestamps` already maps display frame indices to acquisition sample indices
- This is the existing mechanism used for light template extraction
- Simple, direct conversion with O(1) array indexing

**Formula**:
```python
# For each [start_frame, end_frame] pair:
start_sample = frame_timestamps[start_frame]
end_sample = frame_timestamps[end_frame]
```

### R3: Existing Implementation Analysis

**Question**: What changes are needed to existing `add_section_time_analog()`?

**Current State** (from code review):
- Uses `raw_ch1` for detection ✓
- Converts to display frame indices via `frame_timestamps` ✗ (needs removal)
- Requires `frame_timestamps` ✗ (should not be required)

**Decision**: Simplify by removing frame_timestamps conversion

**Changes Required**:
1. Remove frame_timestamps requirement from validation
2. Remove `_sample_to_nearest_frame()` conversion
3. Store raw onset samples directly
4. Compute end_sample = onset_sample + (plot_duration * acquisition_rate)

### R4: Test Impact Analysis

**Question**: What tests need updating?

**Decision**: Update test expectations to use sample indices

**Affected Tests** (from `test_section_time.py`):
- `test_raises_on_missing_frame_timestamps` - Remove this test (no longer required)
- `test_returns_false_peaks_beyond_frame_range` - Remove (no frame range filtering)
- All tests checking section_time values - Update expected units

---

## Phase 1: Design

See companion artifacts:
- [data-model.md](./data-model.md) - Entity definitions
- [contracts/api.md](./contracts/api.md) - API contracts
- [quickstart.md](./quickstart.md) - Usage examples
