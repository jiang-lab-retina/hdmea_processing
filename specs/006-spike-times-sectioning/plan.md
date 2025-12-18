# Implementation Plan: Spike Times Unit Conversion and Stimulation Sectioning

**Branch**: `006-spike-times-sectioning` | **Date**: 2025-12-17 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/006-spike-times-sectioning/spec.md`

## Summary

This feature implements two changes to spike times handling:

1. **Unit Conversion at Load Time**: Convert raw spike timestamps from nanoseconds (10^-9 s) to acquisition sample indices during CMTR data loading. The conversion formula is: `sample_index = timestamp_ns × acquisition_rate / 10^9`. This occurs in `load_recording()` before writing to Zarr.

2. **Spike Times Sectioning**: Add a new pipeline step `section_spike_times()` that extracts spike timestamps within each stimulation trial (defined by section_time) and stores them per-unit in **absolute sample indices** (not trial-relative).

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: numpy, zarr, McsPy (for CMTR loading)  
**Storage**: Zarr archives (hierarchical array storage)  
**Testing**: pytest with test dataset `artifacts/JIANG009_2025-04-10.zarr`  
**Target Platform**: Windows/Linux workstations  
**Project Type**: Single Python package (`src/hdmea/`)  
**Performance Goals**: 10 seconds for 1000 units with 100 trials  
**Constraints**: Must preserve backward compatibility with existing Zarr schema

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | All changes in `src/hdmea/` package |
| II. Modular Subpackage Layout | ✅ PASS | Changes to `io/` (load) and new module in `io/` (sectioning) |
| III. Explicit I/O and Pure Functions | ✅ PASS | Conversion is pure; sectioning has explicit I/O |
| IV. Single Zarr Artifact Per Recording | ✅ PASS | Modifies existing Zarr, no new files |
| V. Data Format Standards | ✅ PASS | Uses Zarr for spike data |
| VI. No Hidden Global State | ✅ PASS | No globals introduced |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |
| Feature Output Policy | ✅ PASS | FileExistsError when data exists, force=True to overwrite |
| Pipeline Documentation | 🔄 REQUIRED | Must update `docs/pipeline_explained.md` and `docs/pipeline_log.md` |

**No violations requiring justification.**

## Project Structure

### Documentation (this feature)

```text
specs/006-spike-times-sectioning/
├── plan.md              # This file
├── research.md          # Phase 0 output ✅
├── data-model.md        # Phase 1 output ✅
├── quickstart.md        # Phase 1 output ✅
├── contracts/           # Phase 1 output ✅
│   └── api.md           # API contracts ✅
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/hdmea/
├── io/
│   ├── cmtr.py              # MODIFIED: No changes needed (timestamps already raw)
│   ├── zarr_store.py        # MODIFIED: Add write_sectioned_spike_times()
│   └── spike_sectioning.py  # NEW: section_spike_times() function
├── pipeline/
│   └── runner.py            # MODIFIED: Convert spike_times in load_recording()
└── utils/
    └── exceptions.py        # Uses existing FileExistsError pattern

tests/
├── unit/
│   ├── test_spike_conversion.py    # NEW: Unit tests for conversion
│   └── test_spike_sectioning.py    # NEW: Unit tests for sectioning
└── integration/
    └── test_spike_pipeline.py      # NEW: End-to-end test with test zarr
```

**Structure Decision**: All changes follow existing modular layout. New sectioning logic in dedicated module `spike_sectioning.py` to maintain separation of concerns.

## Complexity Tracking

> No Constitution Check violations requiring justification.

---

## Phase 0: Research ✅ COMPLETE

See [research.md](./research.md) for full findings.

### Research Summary

| ID | Question | Decision |
|----|----------|----------|
| R1 | Raw timestamp unit from CMTR | **Nanoseconds** (10^-9 s) |
| R2 | Conversion location | `load_recording()` in runner.py |
| R3 | Sectioned storage structure | Group under `spike_times_sectioned/{movie}/` |
| R4 | Naming conflict resolution | Use `spike_times_sectioned/` prefix (not `spike_times/`) |
| R5 | Per-trial vs combined storage | **BOTH**: `full_spike_times` + `trials_spike_times/{idx}` |
| R6 | Overwrite protection pattern | FileExistsError with `force=True` override |
| R7 | Padding parameter design | Tuple `pad_margin=(pre_s, post_s)` default `(2.0, 0.0)` |

---

## Phase 1: Design ✅ COMPLETE

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      load_recording()                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │load_cmtr_data│───▶│ spike_times  │───▶│ write_units()    │  │
│  │   (ns raw)   │    │ conversion   │    │ (sample indices) │  │
│  └──────────────┘    │ ns → samples │    └──────────────────┘  │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   section_spike_times()                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Read units   │───▶│ Filter by    │───▶│ Write sectioned  │  │
│  │ spike_times  │    │ section_time │    │ (absolute times) │  │
│  └──────────────┘    │ + pad_margin │    └──────────────────┘  │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Load**: CMTR → `load_cmtr_data()` → raw ns timestamps
2. **Convert**: `load_recording()` → `spike_times_ns * acquisition_rate / 1e9` → sample indices
3. **Store**: `write_units()` → `units/{unit_id}/spike_times` (uint64 sample indices)
4. **Section**: `section_spike_times(trial_repeats=3, pad_margin=(2.0, 0.0))` → read section_time → filter spikes per trial →
   - `spike_times_sectioned/{movie}/full_spike_times` (all trials combined)
   - `spike_times_sectioned/{movie}/trials_spike_times/{idx}` (per-trial arrays)

### API Design

See [contracts/api.md](./contracts/api.md) for full API specification.

**Key Functions**:

| Function | Module | Type | Key Parameters |
|----------|--------|------|----------------|
| `load_recording()` | `hdmea.pipeline.runner` | Modified | (existing API, internal conversion added) |
| `section_spike_times()` | `hdmea.io.spike_sectioning` | New | `trial_repeats=3`, `pad_margin=(2.0, 0.0)`, `force=False` |

### Data Model

See [data-model.md](./data-model.md) for full schema.

**Changes**:
- `spike_times` dtype unchanged (uint64), unit changes from ns to sample index
- New group: `spike_times_sectioned/{movie}/` with:
  - `full_spike_times` - int64 array (all trials combined)
  - `trials_spike_times/{idx}` - int64 arrays (per-trial split)
- New parameters:
  - `trial_repeats=3` (default) for controlling trials processed
  - `pad_margin=(2.0, 0.0)` seconds (default) - tuple of (pre_margin, post_margin) for extending trial boundaries; converted to samples:
    - `pre_samples = int(pad_margin[0] * acquisition_rate)`
    - `post_samples = int(pad_margin[1] * acquisition_rate)`

---

## Phase 1 Deliverables Checklist

- [x] `research.md` - Research findings documented
- [x] `data-model.md` - Final schema with pad_margin tuple
- [x] `contracts/api.md` - API specifications with pad_margin tuple
- [x] `quickstart.md` - Usage examples with pad_margin tuple

---

## Post-Phase 1 Constitution Re-check

| Principle | Status |
|-----------|--------|
| Package-First Architecture | ✅ All logic in `src/hdmea/` |
| Modular Subpackage Layout | ✅ New module in `io/`, no circular imports |
| Explicit I/O | ✅ Conversion explicit in runner, sectioning has clear I/O |
| Single Zarr Per Recording | ✅ Modifies existing Zarr only |
| Feature Output Policy | ✅ FileExistsError for existing data |
| Pipeline Documentation | 🔄 Must update after implementation |

**Gate Status**: ✅ PASSED - Ready for Phase 2 task breakdown

---

## Implementation Notes

### Key Implementation Details

1. **Spike Times Conversion** (in `load_recording()`):
   ```python
   # Convert ns → sample indices
   spike_times_samples = np.round(
       spike_times_ns * acquisition_rate / 1e9
   ).astype(np.uint64)
   ```

2. **Padding Calculation** (in `section_spike_times()`):
   ```python
   pre_samples = int(pad_margin[0] * acquisition_rate)   # e.g., 2.0 * 20000 = 40000
   post_samples = int(pad_margin[1] * acquisition_rate)  # e.g., 0.0 * 20000 = 0
   
   # Per-trial boundary with padding (clamped)
   padded_start = max(0, trial_start - pre_samples)
   padded_end = trial_end + post_samples
   ```

3. **Storage Structure**:
   ```
   units/{unit_id}/spike_times_sectioned/{movie}/
   ├── full_spike_times              # All trials combined
   └── trials_spike_times/
       ├── 0                         # Trial 0
       ├── 1                         # Trial 1
       └── 2                         # Trial 2
   ```

### Test Data

- **Primary**: `artifacts/JIANG009_2025-04-10.zarr`
- Requires existing `section_time` data from spec 005
