# Tasks: Axon Tracking (AP Trace) for HDF5 Pipeline

**Input**: Design documents from `/specs/010-ap-trace-hdf5/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested in specification - tests included in Polish phase for critical algorithms only.

**Organization**: Tasks grouped by user story for independent implementation and testing.

**⚠️ Testing Rule**: When running tests or validation, do NOT overwrite original HDF5 files. Copy input files to an export folder and process the copies. This preserves original data integrity.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- All file paths relative to repository root

---

## Phase 1: Setup (Module Structure)

**Purpose**: Create the ap_tracking module directory structure

- [x] T001 Create module directory structure at `src/hdmea/features/ap_tracking/`
- [x] T002 Create `src/hdmea/features/ap_tracking/__init__.py` with public API exports placeholder
- [x] T003 Verify torch is available in project dependencies in `pyproject.toml`
- [x] T004 Create export folder at `Projects/ap_trace_hdf5/export/` for test outputs (preserves originals)

---

## Phase 2: Foundational (Core Algorithms)

**Purpose**: Implement pure NumPy algorithms that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until these core algorithms are complete

- [x] T005 [P] Implement `find_soma_from_3d_sta()` in `src/hdmea/features/ap_tracking/soma_detector.py`
- [x] T006 [P] Implement `soma_refiner()` in `src/hdmea/features/ap_tracking/ais_refiner.py`
- [x] T007 [P] Implement `AIS_refiner()` in `src/hdmea/features/ap_tracking/ais_refiner.py`
- [x] T008 [P] Implement `parse_dvnt_from_center_xy()` in `src/hdmea/features/ap_tracking/dvnt_parser.py`
- [x] T009 Implement CNN3D_WithVelocity model class in `src/hdmea/features/ap_tracking/model_inference.py`
- [x] T010 Implement `load_cnn_model()` and `select_device()` in `src/hdmea/features/ap_tracking/model_inference.py`
- [x] T011 Implement `extract_all_cubes_from_sta()` in `src/hdmea/features/ap_tracking/model_inference.py`
- [x] T012 Implement `predict_batch_gpu_optimized()` in `src/hdmea/features/ap_tracking/model_inference.py`
- [x] T013 Implement `run_model_inference()` in `src/hdmea/features/ap_tracking/model_inference.py`
- [x] T014 [P] Implement `process_predictions()` and `improved_filter_noise_3d()` in `src/hdmea/features/ap_tracking/postprocess.py`
- [x] T015 [P] Implement `calculate_projections()` in `src/hdmea/features/ap_tracking/pathway_analysis.py`
- [x] T016 [P] Implement `fit_line_to_projections()` in `src/hdmea/features/ap_tracking/pathway_analysis.py`
- [x] T017 Implement `calculate_optimal_intersection()` in `src/hdmea/features/ap_tracking/pathway_analysis.py`
- [x] T018 Implement `calculate_soma_polar_coordinates()` in `src/hdmea/features/ap_tracking/pathway_analysis.py`

**Checkpoint**: All core algorithms ready - integration can now begin

---

## Phase 3: User Story 1 - Single File AP Trace Analysis (Priority: P1) 🎯 MVP

**Goal**: Process a single HDF5 file with AP tracking analysis and write results back

**Independent Test**: Copy source file to `Projects/ap_trace_hdf5/export/`, run `compute_ap_tracking("export/copy.h5", model_path)`, verify `ap_tracking` features written to each unit in the copy (preserves original)

### Implementation for User Story 1

- [x] T019 [US1] Implement HDF5 reading helpers for STA data in `src/hdmea/features/ap_tracking/core.py`
- [x] T020 [US1] Implement HDF5 reading for cell geometry (center_row, center_col) in `src/hdmea/features/ap_tracking/core.py`
- [x] T021 [US1] Implement DVNT metadata reading from `metadata/gsheet_row/Center_xy` in `src/hdmea/features/ap_tracking/core.py`
- [x] T022 [US1] Implement `write_ap_tracking_to_hdf5()` for writing results as datasets (not attributes) in `src/hdmea/features/ap_tracking/core.py`
- [x] T023 [US1] Implement `process_single_unit()` orchestration function in `src/hdmea/features/ap_tracking/core.py`
- [x] T024 [US1] Implement `compute_ap_tracking()` main entry point (immediate-save mode) in `src/hdmea/features/ap_tracking/core.py`
- [x] T025 [US1] Add error handling for missing eimage_sta data (skip unit with warning) in `src/hdmea/features/ap_tracking/core.py`
- [x] T026 [US1] Add error handling for missing model file in `src/hdmea/features/ap_tracking/core.py`
- [x] T027 [US1] Add logging throughout AP tracking pipeline in `src/hdmea/features/ap_tracking/core.py`
- [x] T028 [US1] Update `src/hdmea/features/ap_tracking/__init__.py` with public API exports

**Checkpoint**: User Story 1 complete - single file processing works independently

---

## Phase 4: User Story 2 - Session-Based Deferred Processing (Priority: P2)

**Goal**: Integrate AP tracking with PipelineSession for deferred save pattern

**Independent Test**: Create session, call `compute_ap_tracking(session=session)`, verify session contains ap_tracking data, call `session.save()`

### Implementation for User Story 2

- [ ] T029 [US2] Study PipelineSession interface in `src/hdmea/pipeline/` or existing feature implementations
- [ ] T030 [US2] Add `session` parameter to `compute_ap_tracking()` in `src/hdmea/features/ap_tracking/core.py`
- [ ] T031 [US2] Implement session-based data accumulation (no HDF5 write until save) in `src/hdmea/features/ap_tracking/core.py`
- [ ] T032 [US2] Implement session return pattern (return updated session) in `src/hdmea/features/ap_tracking/core.py`
- [ ] T033 [US2] Update `__init__.py` exports if needed in `src/hdmea/features/ap_tracking/__init__.py`

**Checkpoint**: User Story 2 complete - session-based processing works independently

---

## Phase 5: User Story 3 - Batch Processing Multiple Files (Priority: P3)

**Goal**: Process multiple HDF5 files with progress tracking and skip-existing functionality

**Independent Test**: Run `compute_ap_tracking_batch(file_list, model_path)` on folder with multiple files

### Implementation for User Story 3

- [ ] T034 [US3] Implement `compute_ap_tracking_batch()` function in `src/hdmea/features/ap_tracking/core.py`
- [ ] T035 [US3] Add `skip_existing` logic to check for existing ap_tracking features in `src/hdmea/features/ap_tracking/core.py`
- [ ] T036 [US3] Add progress callback support for batch processing in `src/hdmea/features/ap_tracking/core.py`
- [ ] T037 [US3] Add summary result dictionary (file → status mapping) in `src/hdmea/features/ap_tracking/core.py`
- [ ] T038 [US3] Add GPU memory cleanup between files in batch processing in `src/hdmea/features/ap_tracking/core.py`
- [ ] T039 [US3] Update `__init__.py` with `compute_ap_tracking_batch` export in `src/hdmea/features/ap_tracking/__init__.py`

**Checkpoint**: User Story 3 complete - batch processing works independently

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Testing, documentation, and validation

**⚠️ IMPORTANT**: All tests must copy input files to export folder before processing - never modify originals!

- [ ] T040 [P] Create synthetic STA fixture in `tests/fixtures/ap_tracking/synthetic_sta.npy`
- [ ] T041 [P] Create minimal test HDF5 file in `tests/fixtures/ap_tracking/minimal_test.h5`
- [ ] T042 [P] Unit test for soma_detector in `tests/unit/features/test_ap_tracking.py`
- [ ] T043 [P] Unit test for ais_refiner in `tests/unit/features/test_ap_tracking.py`
- [ ] T044 [P] Unit test for dvnt_parser in `tests/unit/features/test_ap_tracking.py`
- [ ] T045 Integration test (copy test file to export folder first) in `tests/integration/test_ap_tracking_integration.py`
- [x] T046 [P] Create example usage script in `Projects/ap_trace_hdf5/examples/run_ap_tracking.py`
- [ ] T047 Validate quickstart.md examples work with implementation
- [x] T048 Copy test file to `Projects/ap_trace_hdf5/export/` then run AP tracking on the copy
- [x] T049 Verify HDF5 output structure in export folder matches data-model.md specification

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001-T004 - No dependencies, can start immediately
- **Foundational (Phase 2)**: T005-T018 - Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: T019-T028 - Depends on Foundational phase completion - MVP delivery
- **User Story 2 (Phase 4)**: T029-T033 - Depends on User Story 1 (reuses core.py functions)
- **User Story 3 (Phase 5)**: T034-T039 - Depends on User Story 1 (reuses compute_ap_tracking)
- **Polish (Phase 6)**: T040-T049 - Depends on User Story 1 minimum, ideally all stories

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories ✅
- **User Story 2 (P2)**: Requires User Story 1 complete (extends compute_ap_tracking)
- **User Story 3 (P3)**: Requires User Story 1 complete (wraps compute_ap_tracking)

### Within Foundational Phase

```
Parallel Group A (no dependencies):
├── T005 soma_detector.py
├── T006-T007 ais_refiner.py
├── T008 dvnt_parser.py
├── T014 postprocess.py
├── T015-T016 pathway_analysis.py (projections, line fitting)

Sequential Group B (model inference):
├── T009 CNN model class
├── T010 load_cnn_model, select_device
├── T011 extract_all_cubes_from_sta
├── T012 predict_batch_gpu_optimized
└── T013 run_model_inference

Sequential Group C (depends on T016):
├── T017 calculate_optimal_intersection
└── T018 calculate_soma_polar_coordinates
```

### Within User Story 1

```
Sequential Order:
├── T019 HDF5 STA reading
├── T020 HDF5 geometry reading
├── T021 DVNT metadata reading
├── T022 write_ap_tracking_to_hdf5
├── T023 process_single_unit
├── T024 compute_ap_tracking (main)
├── T025-T026 Error handling
├── T027 Logging
└── T028 __init__.py exports
```

---

## Parallel Example: Foundational Phase

```bash
# Launch all independent algorithm tasks together:
Task T005: "Implement find_soma_from_3d_sta() in soma_detector.py"
Task T006: "Implement soma_refiner() in ais_refiner.py"
Task T007: "Implement AIS_refiner() in ais_refiner.py"
Task T008: "Implement parse_dvnt_from_center_xy() in dvnt_parser.py"
Task T014: "Implement process_predictions() in postprocess.py"
Task T015: "Implement calculate_projections() in pathway_analysis.py"
Task T016: "Implement fit_line_to_projections() in pathway_analysis.py"
```

---

## Parallel Example: Polish Phase

```bash
# Launch all test fixture and unit test tasks together:
Task T040: "Create synthetic STA fixture"
Task T041: "Create minimal test HDF5 file"
Task T042: "Unit test for soma_detector"
Task T043: "Unit test for ais_refiner"
Task T044: "Unit test for dvnt_parser"
Task T046: "Create example usage script"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T018)
3. Complete Phase 3: User Story 1 (T019-T028)
4. **STOP and VALIDATE**:
   - Create export folder: `Projects/ap_trace_hdf5/export/`
   - Copy test file to export folder (never modify original)
   - Run AP tracking on the copy
   - Verify HDF5 output structure matches data-model.md

### Incremental Delivery

1. Setup + Foundational → Algorithms ready
2. Add User Story 1 → Test independently → **MVP Complete!**
3. Add User Story 2 → Test session-based workflow
4. Add User Story 3 → Test batch processing
5. Polish phase → Tests, docs, validation

### Key Files Summary

| File | Tasks | Purpose |
|------|-------|---------|
| `soma_detector.py` | T005 | Soma detection from 3D STA |
| `ais_refiner.py` | T006-T007 | Soma and AIS refinement |
| `dvnt_parser.py` | T008 | DVNT position parsing |
| `model_inference.py` | T009-T013 | CNN model loading and inference |
| `postprocess.py` | T014 | Prediction post-processing |
| `pathway_analysis.py` | T015-T018 | AP pathway fitting and polar coords |
| `core.py` | T019-T038 | Main entry points and orchestration |
| `__init__.py` | T002, T028, T033, T039 | Public API exports |

---

## Notes

- All HDF5 values stored as explicit datasets (not attributes) per D6 decision
- Model file location: `Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth`
- Source test file: `Projects/load_gsheet/export_gsheet_20251225/2025.03.06-12.38.11-Rec.h5`
- **Export folder for testing**: `Projects/ap_trace_hdf5/export/` - copy files here before processing
- Output group: `units/{unit_id}/features/ap_tracking/`
- Always overwrite existing ap_tracking features on reprocessing
- **Never modify original source files** - always work on copies in export folder

