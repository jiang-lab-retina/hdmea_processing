# Implementation Plan: DSGC Direction Sectioning

**Branch**: `011-dsgc-direction-section` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/011-dsgc-direction-section/spec.md`

## Summary

Section spike times by moving bar direction for direction-selective ganglion cell (DSGC) analysis. The system reads `full_spike_times` from HDF5, converts to movie-relative frames, uses a per-pixel on/off timing dictionary to determine when the bar crosses each cell's RF center, and organizes spikes by 8 directions × 3 repetitions.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: h5py, numpy, pickle (stdlib)  
**Storage**: HDF5 (read source, write in-place or to output copy)  
**Testing**: pytest with synthetic fixtures  
**Target Platform**: Windows (network paths: `\\Jiangfs1\...`)  
**Project Type**: Single package (`src/hdmea/`)  
**Performance Goals**: Process all units in a recording within 60 seconds  
**Constraints**: Must not modify source HDF5 during testing; use export folder  
**Scale/Scope**: ~50-200 units per recording, 24 trials per unit

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Code will reside in `src/hdmea/features/dsgc_direction.py` |
| II. Modular Subpackage Layout | ✅ PASS | Uses `features/` subpackage per constitution |
| III. Explicit I/O and Pure Functions | ✅ PASS | Explicit file paths, no global state |
| IV. Single HDF5 Artifact | ✅ PASS | Writes to same HDF5 under unit structure |
| IV.B. Deferred Save Mode | N/A | Not implementing session mode initially |
| V. Data Format Standards | ✅ PASS | HDF5 for hierarchical data, PKL for reference dict (pre-existing) |
| VI. No Hidden Global State | ✅ PASS | All config via parameters |
| VII. Independence from Legacy | ✅ PASS | No legacy imports |

**Gate Result**: PASS - All applicable principles satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/011-dsgc-direction-section/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (already created during specify)
├── quickstart.md        # Phase 1 output (already created during specify)
├── contracts/           # Phase 1 output
│   └── api.md           # API contract (already created during specify)
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/hdmea/
├── features/
│   ├── __init__.py           # Export section_by_direction
│   ├── dsgc_direction.py     # NEW: Main implementation
│   └── sta.py                # Existing STA module (reference for patterns)
└── io/
    └── section_time.py       # Existing: PRE_MARGIN_FRAME_NUM, convert_sample_index_to_frame

tests/
├── unit/
│   └── test_dsgc_direction.py    # NEW: Unit tests
└── fixtures/
    └── dsgc_direction_fixtures.py  # NEW: Synthetic test data

Projects/dsgc_section/
├── dsgc_section_data_organization.md  # Data documentation
├── dsgc_section.py                     # Test/development script
└── export/                             # Test output folder
```

**Structure Decision**: Single package structure. New module `dsgc_direction.py` in `features/` subpackage follows established patterns from `sta.py`.

## Phase 0: Research

### Research Tasks

1. **Existing Patterns**: How does `sta.py` handle frame conversion and HDF5 I/O?
2. **On/Off Dictionary**: Verify pickle file structure and loading
3. **Cell Center Conversion**: Confirm 15×15 to 300×300 scaling logic
4. **HDF5 Write Patterns**: How to create nested groups (`direction_section/{dir}/trials/{rep}`)

### Findings

See [research.md](./research.md) for detailed findings.

**Key Decisions**:

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| Use `searchsorted` for frame conversion | Matches existing `convert_sample_index_to_frame` | Linear search (too slow) |
| Integer scaling (×20) for center | Simple, matches grid ratios | Interpolation (unnecessary complexity) |
| String keys for directions | HDF5 group names must be strings | Integer keys require conversion |
| Copy-on-test pattern | Protects source data during development | In-place modification (risky for test data) |

## Phase 1: Design

### Data Model

See [data-model.md](./data-model.md) for complete data structures.

**Key Entities**:

| Entity | Type | Description |
|--------|------|-------------|
| `DirectionSectionResult` | dataclass | Return value with processing statistics |
| `DIRECTION_LIST` | List[int] | `[0, 45, 90, 135, 180, 225, 270, 315]` |
| `on_off_dict` | Dict[(row,col), Dict] | Per-pixel timing dictionary |

### API Contracts

See [contracts/api.md](./contracts/api.md) for complete API specification.

**Primary Function Signature**:

```python
def section_by_direction(
    hdf5_path: Union[str, Path],
    *,
    movie_name: str = "moving_h_bar_s5_d8_3x",
    on_off_dict_path: Optional[Union[str, Path]] = None,
    padding_frames: int = 10,
    force: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    unit_ids: Optional[List[str]] = None,
) -> DirectionSectionResult:
```

### Module Dependencies

```
hdmea.features.dsgc_direction
    ├── hdmea.io.section_time (PRE_MARGIN_FRAME_NUM, convert_sample_index_to_frame)
    ├── h5py
    ├── numpy
    ├── pickle (stdlib)
    └── logging (stdlib)
```

### Algorithm Outline

```
1. Load prerequisites
   ├── Load on/off dictionary (pickle)
   ├── Read frame_timestamps from HDF5
   └── Read section_time for movie

2. Compute movie start frame
   └── movie_start_frame = convert(section_time[0,0]) + PRE_MARGIN_FRAME_NUM

3. For each unit (or filtered unit_ids):
   a. Read cell center from sta_geometry
   b. Convert 15×15 → 300×300 (clip to [0,299])
   c. Read full_spike_times
   d. Convert spike samples → movie-relative frames
   e. For each trial (0-23):
      - Get on/off frames at cell center
      - Apply padding
      - Extract spikes in window
      - Map to direction + repetition
   f. Save to direction_section/{dir}/trials/{rep}

4. Return result statistics
```

## Complexity Tracking

No constitution violations requiring justification.

## Post-Design Constitution Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First | ✅ PASS | All logic in `src/hdmea/features/dsgc_direction.py` |
| II. Modular Layout | ✅ PASS | Follows dependency flow: `io → features` |
| III. Explicit I/O | ✅ PASS | All paths via parameters |
| IV. Single HDF5 | ✅ PASS | Writes to same recording artifact |
| VI. No Global State | ✅ PASS | Constants only, no mutable globals |
| VII. No Legacy | ✅ PASS | Fresh implementation |

**Final Gate**: PASS

