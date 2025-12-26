# Implementation Plan: Axon Tracking (AP Trace) for HDF5 Pipeline

**Branch**: `010-ap-trace-hdf5` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/010-ap-trace-hdf5/spec.md`

## Summary

This plan implements migration of the axon tracking (AP trace) analysis from legacy pkl-based processing to the new HDF5-based pipeline. The implementation adapts existing algorithms from `Legacy_code/Data_Processing_2025/Processing_2025/ap_trace_git/` to work with HDF5 data structures, following the pipeline constitution's package-first architecture and deferred-save patterns.

**Technical Approach**: Create a new module at `src/hdmea/features/ap_tracking/` that:
1. Reads STA data from HDF5 (`units/{unit_id}/features/eimage_sta/data`)
2. Applies existing detection algorithms (soma detection, AIS refinement)
3. Runs GPU-accelerated CNN model inference for axon pathway prediction
4. Calculates AP pathway intersections and soma polar coordinates
5. Parses DVNT positions from recording metadata
6. Writes results to `units/{unit_id}/features/ap_tracking/`

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: 
- h5py (HDF5 I/O)
- torch (CNN model inference, GPU acceleration)
- numpy (numerical operations)
- scipy (line fitting, statistics)

**Storage**: HDF5 files (read existing, write features)
**Testing**: pytest with synthetic STA data fixtures
**Target Platform**: Windows (primary), Linux server (remote processing)
**Project Type**: Single package (`src/hdmea/`)
**Performance Goals**: 
- <60 seconds per file (20 units) on GPU
- <5 minutes per file (20 units) on CPU

**Constraints**: 
- Must not import from Legacy_code/ (constitution VII)
- GPU memory: handle 8GB-32GB GPUs with adaptive batch sizing
- HDF5 single-writer access model

**Scale/Scope**: Batch processing 100+ HDF5 files, 10-100 units per file

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | All logic in `src/hdmea/features/ap_tracking/` |
| II. Modular Subpackage Layout | ✅ PASS | Placed under `features/` as feature extractor |
| III. Explicit I/O and Pure Functions | ✅ PASS | STA input → ap_trace output, explicit params |
| IV. Single HDF5 Artifact Per Recording | ✅ PASS | Features written to existing HDF5 file |
| IV.B. Deferred Save Mode | ✅ PASS | Support session parameter for deferred writes |
| V. Data Format Standards | ✅ PASS | HDF5 for nested data, no new PKL artifacts |
| VI. No Hidden Global State | ✅ PASS | Device selection passed as parameter |
| VII. Independence from Legacy Code | ✅ PASS | Algorithms reimplemented, no legacy imports |

**Constitution violations**: None

## Project Structure

### Documentation (this feature)

```text
specs/010-ap-trace-hdf5/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── contracts/           # Phase 1 output
    └── api.py           # Function signatures
```

### Source Code (repository root)

```text
src/hdmea/
├── features/
│   └── ap_tracking/           # NEW: AP trace feature module
│       ├── __init__.py        # Public API exports
│       ├── core.py            # Main compute_ap_tracking function
│       ├── soma_detector.py   # Soma detection from 3D STA
│       ├── ais_refiner.py     # Axon initial segment refinement
│       ├── model_inference.py # CNN model loading and inference
│       ├── postprocess.py     # Prediction post-processing
│       ├── pathway_analysis.py # AP pathway fitting and polar coords
│       └── dvnt_parser.py     # DVNT position parsing from metadata

tests/
├── unit/
│   └── features/
│       └── test_ap_tracking.py # Unit tests with synthetic data
├── integration/
│   └── test_ap_tracking_integration.py # End-to-end HDF5 test
└── fixtures/
    └── ap_tracking/
        ├── synthetic_sta.npy  # Synthetic STA data
        └── minimal_test.h5    # Minimal HDF5 with eimage_sta

Projects/ap_trace_hdf5/
├── model/                      # Trained model files (existing)
│   └── CNN_3d_with_velocity_model_from_all_process.pth
└── examples/
    └── run_ap_tracking.py      # Example usage script
```

**Structure Decision**: Single-package layout under `src/hdmea/features/ap_tracking/`. The module follows existing feature module patterns (e.g., `features/sta.py`) but uses a subdirectory due to multiple helper modules.

## Complexity Tracking

No constitution violations requiring justification.

## Design Decisions

### D1: Module Location

**Decision**: Place under `src/hdmea/features/ap_tracking/` as a subdirectory module rather than single file.

**Rationale**: The feature has 6 distinct algorithmic components (soma detection, AIS refinement, model inference, postprocessing, pathway analysis, DVNT parsing). A subdirectory keeps each component testable and maintainable while presenting a clean public API through `__init__.py`.

**Alternatives Rejected**:
- Single `ap_tracking.py` file: Would exceed 800+ lines, violating maintainability
- Under `analysis/`: Constitution places feature extractors under `features/`

### D2: Model File Location

**Decision**: Model file at `Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth` (per user specification).

**Rationale**: User explicitly specified this location. The path is passed as a parameter, allowing flexibility for different environments.

### D3: Output Feature Group Name

**Decision**: Write outputs to `features/ap_tracking/` (not `ap_trace`).

**Rationale**: User explicitly requested `features/ap_tracking` as the output group name in their final clarification.

### D4: DVNT Position Derivation

**Decision**: Parse `metadata/gsheet_row/Center_xy` at processing time using the legacy formula:
- Format: "L/R, VD_coord, NT_coord"
- `DV_position` = -VD_coord (positive = dorsal)
- `NT_position` = NT_coord (positive = nasal)
- `LR_position` = L/R string

**Rationale**: Follows the exact conversion logic from legacy `auto_label.py:label_DVNT_position()` as specified by user.

### D5: Overwrite Behavior

**Decision**: Always overwrite existing `ap_tracking` features when reprocessing.

**Rationale**: User explicitly chose option A (always overwrite) when asked about reprocessing behavior.

### D6: HDF5 Storage Strategy - Datasets Only

**Decision**: Store ALL values as explicit HDF5 datasets, not attributes.

**Rationale**: User explicitly requested "do not use attributes, save all data as explicit regular value". This provides:
- Consistent access patterns (`dataset[()]` for everything)
- Better HDF5 viewer compatibility
- Support for NaN in missing data
- Easier programmatic iteration

## Algorithm Adaptation Summary

| Legacy Function | New Location | Changes |
|-----------------|--------------|---------|
| `find_soma_from_3d_sta()` | `soma_detector.py` | No changes, pure NumPy |
| `soma_refiner()` | `ais_refiner.py` | No changes, pure NumPy |
| `AIS_refiner()` | `ais_refiner.py` | No changes, pure NumPy |
| `CNN3D_WithVelocity` | `model_inference.py` | Reimplemented model class |
| `run_predictions_gpu_optimized()` | `model_inference.py` | Adapted for HDF5 data |
| `process_predictions()` | `postprocess.py` | Adapted, no file I/O |
| `fit_line_to_projections()` | `pathway_analysis.py` | No changes |
| `calculate_soma_polar_coordinate()` | `pathway_analysis.py` | Adapted for HDF5 |
| `label_DVNT_position()` | `dvnt_parser.py` | Adapted for HDF5 metadata structure |

## Implementation Order

1. **Phase 1**: Core algorithms (soma, AIS) - no dependencies
2. **Phase 2**: Model inference - depends on Phase 1
3. **Phase 3**: Postprocessing - depends on Phase 2
4. **Phase 4**: Pathway analysis & polar coordinates - depends on Phase 3
5. **Phase 5**: DVNT parsing - independent, can parallelize
6. **Phase 6**: Integration (`core.py`) - depends on all above
7. **Phase 7**: Tests and documentation
