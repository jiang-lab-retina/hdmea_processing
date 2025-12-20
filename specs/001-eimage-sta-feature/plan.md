# Implementation Plan: Electrode Image STA (eimage_sta)

**Branch**: `001-eimage-sta-feature` | **Date**: 2025-12-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-eimage-sta-feature/spec.md`

## Summary

Compute Electrode Image Spike-Triggered Average (eimage_sta) for each unit by averaging high-pass filtered sensor data from the HD-MEA electrode array in a time window around each spike. The implementation uses vectorized filtering (scipy filtfilt with axis parameter), vectorized spike window extraction (NumPy fancy indexing), and HDF5 memory-mapped access for performance. Target: under 5 minutes for a typical recording (120s, ~100 units).

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: h5py (memory-mapped HDF5), scipy (Butterworth filter), numpy (vectorized operations), McsPy (CMCR sensor data access), tqdm (progress)  
**Storage**: HDF5 (.h5) for output, CMCR for sensor data input  
**Testing**: pytest with synthetic data fixtures  
**Target Platform**: Windows (primary), cross-platform compatible  
**Project Type**: Single Python package (hdmea)  
**Performance Goals**: <5 minutes for 120s recording with ~100 units  
**Constraints**: Memory-bounded via memory-mapped access, no GPU required  
**Scale/Scope**: 64×64 electrode array, 20kHz sampling, 120s duration, 10,000 spike limit per unit

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Implementation in `src/hdmea/features/eimage_sta/` |
| II. Modular Subpackage Layout | ✅ PASS | Feature in `features/`, IO in `io/`, filter utilities in `preprocess/` |
| III. Explicit I/O and Pure Functions | ✅ PASS | Pure compute functions, side effects isolated to write phase |
| IV. Single HDF5 Artifact Per Recording | ✅ PASS | eimage_sta stored in HDF5 features group |
| V. Data Format Standards | ✅ PASS | HDF5 for nested data, attributes for metadata |
| VI. No Hidden Global State | ✅ PASS | All config passed explicitly |
| VII. Independence from Legacy Code | ✅ PASS | Reimplementing legacy logic, not importing |
| Registry Pattern | ✅ PASS | Uses @FeatureRegistry.register decorator |
| Feature Extractor Requirements | ✅ PASS | Declares name, version, inputs, outputs, runtime_class |
| Logging Standards | ✅ PASS | Uses logging.getLogger(__name__) |

## Project Structure

### Documentation (this feature)

```text
specs/001-eimage-sta-feature/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 research findings
├── data-model.md        # Data structures and schemas
├── quickstart.md        # Quick start guide
├── checklists/          # Validation checklists
│   └── requirements.md
└── contracts/           # API contracts
    └── eimage_sta_api.md
```

### Source Code (repository root)

```text
src/hdmea/
├── features/
│   ├── eimage_sta/              # NEW: eimage_sta feature module
│   │   ├── __init__.py
│   │   ├── extractor.py         # EImageSTAExtractor class
│   │   └── compute.py           # Core computation functions
│   └── ... (existing features)
├── preprocess/
│   └── filtering.py             # EXTEND: Add vectorized high-pass filter
├── io/
│   ├── cmcr.py                  # EXTEND: Add sensor data loading
│   └── ... (existing)
└── ... (unchanged)

tests/
├── unit/
│   └── features/
│       └── test_eimage_sta.py   # Unit tests with synthetic data
├── integration/
│   └── test_eimage_sta_e2e.py   # End-to-end test with real data
└── fixtures/
    └── eimage_sta_fixtures.py   # Synthetic test data generators
```

**Structure Decision**: Single project structure following existing hdmea package layout. New feature module at `src/hdmea/features/eimage_sta/`.

## Test Data Paths

For integration testing, use the following real data files:

```python
cmcr_path = "O:\\20250410\\set6\\2025.04.10-11.12.57-Rec.cmcr"
cmtr_path = "O:\\20250410\\set6\\2025.04.10-11.12.57-Rec-.cmtr"
hdf5_path = "artifacts/2025.04.10-11.12.57-Rec.h5"
```

## Complexity Tracking

No constitution violations requiring justification.

## Implementation Phases

### Phase 1: Core Infrastructure

1. **Extend CMCR loader** (`io/cmcr.py`): Add `load_sensor_data()` function for memory-mapped sensor array access
2. **Add vectorized high-pass filter** (`preprocess/filtering.py`): Add `apply_highpass_filter_3d()` using scipy.signal.filtfilt with axis=0

### Phase 2: Feature Extractor

1. **Create eimage_sta module** (`features/eimage_sta/`):
   - `compute.py`: Core computation functions (vectorized window extraction, averaging)
   - `extractor.py`: EImageSTAExtractor class registered with FeatureRegistry

### Phase 3: Integration & Testing

1. **Unit tests**: Synthetic data tests for filter and STA computation
2. **Integration test**: End-to-end test with real CMCR/H5 data
3. **Performance validation**: Verify <5 minute target

### Phase 4: Optional Enhancements

1. **Filter caching**: Optional cache for filtered sensor data
2. **Progress reporting**: tqdm integration for long computations

