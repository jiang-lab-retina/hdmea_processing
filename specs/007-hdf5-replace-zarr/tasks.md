# Tasks: Replace Zarr Format with HDF5

**Input**: Design documents from `/specs/007-hdf5-replace-zarr/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Included as this is a foundational change affecting the entire pipeline.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/hdmea/` at repository root
- Tests in `tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency setup

- [x] T001 Add h5py>=3.0.0 to pyproject.toml dependencies
- [x] T002 [P] Create empty src/hdmea/io/hdf5_store.py with module docstring
- [x] T003 [P] Create empty tests/unit/test_hdf5_store.py with imports

---

## Phase 2: Foundational (Core HDF5 Module)

**Purpose**: Core hdf5_store.py functions that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement `create_recording_hdf5()` in src/hdmea/io/hdf5_store.py per contracts/api.md
- [x] T005 Implement `open_recording_hdf5()` with single-writer check in src/hdmea/io/hdf5_store.py
- [x] T006 Implement `_write_metadata_to_group()` helper in src/hdmea/io/hdf5_store.py
- [x] T007 Implement `write_source_files()` in src/hdmea/io/hdf5_store.py
- [x] T008 [P] Implement `mark_stage1_complete()` in src/hdmea/io/hdf5_store.py
- [x] T009 [P] Implement `get_stage1_status()` in src/hdmea/io/hdf5_store.py
- [x] T010 [P] Implement `list_units()` in src/hdmea/io/hdf5_store.py
- [x] T011 [P] Implement `list_features()` in src/hdmea/io/hdf5_store.py

**Checkpoint**: ✅ Core HDF5 functions ready - user story implementation can now begin

---

## Phase 3: User Story 1 & 2 - Create & Read HDF5 Data (Priority: P1) 🎯 MVP

**Goal**: Pipeline creates valid HDF5 files and reads data back correctly

**Independent Test**: Run Stage 1 pipeline on CMTR file → verify `.h5` output with correct structure → read data back and verify round-trip integrity

### Tests for User Stories 1 & 2

- [x] T012 [P] [US1] Unit test for create_recording_hdf5() in tests/unit/test_hdf5_store.py
- [x] T013 [P] [US1] Unit test for write_units() in tests/unit/test_hdf5_store.py
- [x] T014 [P] [US1] Unit test for write_stimulus() in tests/unit/test_hdf5_store.py
- [x] T015 [P] [US1] Unit test for write_metadata() in tests/unit/test_hdf5_store.py
- [x] T016 [P] [US2] Unit test for open_recording_hdf5() in tests/unit/test_hdf5_store.py
- [x] T017 [P] [US2] Unit test for data round-trip (write then read) in tests/unit/test_hdf5_store.py

### Implementation for User Story 1 (Write)

- [x] T018 [US1] Implement `write_units()` in src/hdmea/io/hdf5_store.py per contracts/api.md
- [x] T019 [US1] Implement `write_stimulus()` in src/hdmea/io/hdf5_store.py per contracts/api.md
- [x] T020 [US1] Implement `write_metadata()` in src/hdmea/io/hdf5_store.py per contracts/api.md
- [x] T021 [US1] Update src/hdmea/io/__init__.py to export hdf5_store functions
- [x] T022 [US1] Update src/hdmea/pipeline/runner.py to use hdf5_store.create_recording_hdf5()
- [x] T023 [US1] Update src/hdmea/pipeline/runner.py to call write_units() with HDF5 file
- [x] T024 [US1] Update src/hdmea/pipeline/runner.py to call write_stimulus() with HDF5 file
- [x] T025 [US1] Update src/hdmea/pipeline/runner.py to call write_metadata() with HDF5 file
- [x] T026 [US1] Update src/hdmea/pipeline/runner.py to call mark_stage1_complete()
- [x] T027 [US1] Update src/hdmea/pipeline/flows.py to use .h5 extension instead of .zarr

### Implementation for User Story 2 (Read)

- [x] T028 [US2] Update src/hdmea/io/section_time.py to use open_recording_hdf5() instead of zarr
- [x] T029 [US2] Update src/hdmea/io/section_time.py read operations for HDF5 dataset access
- [x] T030 [US2] Update src/hdmea/io/spike_sectioning.py to use open_recording_hdf5() instead of zarr
- [x] T031 [US2] Update src/hdmea/io/spike_sectioning.py write operations for HDF5 dataset creation

### Integration Test

- [x] T032 [US1] [US2] Integration test: full pipeline with HDF5 output in tests/integration/test_pipeline_hdf5.py

**Checkpoint**: ✅ User Stories 1 & 2 complete - pipeline creates and reads HDF5 files correctly

---

## Phase 4: User Story 3 - Write Extracted Features to HDF5 (Priority: P2)

**Goal**: Feature extractors write output to HDF5 files correctly

**Independent Test**: Run a feature extractor (e.g., baseline_127) on HDF5 recording → verify features written to `/units/{unit_id}/features/{feature_name}`

### Tests for User Story 3

- [x] T033 [P] [US3] Unit test for write_feature_to_unit() in tests/unit/test_hdf5_store.py
- [x] T034 [P] [US3] Unit test for nested feature data (scalars + arrays) in tests/unit/test_hdf5_store.py

### Implementation for User Story 3

- [x] T035 [US3] Implement `write_feature_to_unit()` in src/hdmea/io/hdf5_store.py per contracts/api.md
- [x] T036 [US3] Update src/hdmea/features/base.py to use hdf5_store for feature writing
- [x] T037 [P] [US3] Update src/hdmea/features/baseline/baseline_127.py to use HDF5 operations
- [x] T038 [P] [US3] Update src/hdmea/features/on_off/step_up.py to use HDF5 operations
- [x] T039 [P] [US3] Update src/hdmea/features/receptive_field/dense_noise.py to use HDF5 operations
- [x] T040 [P] [US3] Update src/hdmea/features/direction/moving_bar.py to use HDF5 operations
- [x] T041 [P] [US3] Update src/hdmea/features/frequency/chirp.py to use HDF5 operations
- [x] T042 [P] [US3] Update src/hdmea/features/chromatic/green_blue.py to use HDF5 operations
- [x] T043 [P] [US3] Update src/hdmea/features/frif/frif_extractor.py to use HDF5 operations
- [x] T044 [P] [US3] Update src/hdmea/features/cell_type/rgc_classifier.py to use HDF5 operations
- [x] T045 [P] [US3] Update src/hdmea/features/example/example_feature.py to use HDF5 operations

**Checkpoint**: ✅ User Story 3 complete - feature extraction writes to HDF5 correctly

---

## Phase 5: User Story 4 - Visualize HDF5 Data (Priority: P3)

**Goal**: Visualization GUI works with HDF5 files

**Independent Test**: Launch visualization GUI with HDF5 file → verify tree view shows correct structure → verify plots render correctly

### Implementation for User Story 4

- [x] T046 [US4] Update src/hdmea/viz/zarr_viz/utils.py to use h5py instead of zarr
- [x] T047 [US4] Update src/hdmea/viz/zarr_viz/tree.py to navigate HDF5 groups
- [x] T048 [US4] Update src/hdmea/viz/zarr_viz/metadata.py to read HDF5 attributes
- [x] T049 [US4] Update src/hdmea/viz/zarr_viz/plots.py to read HDF5 datasets
- [x] T050 [US4] Update src/hdmea/viz/zarr_viz/app.py to accept .h5 file paths
- [x] T051 [US4] Update src/hdmea/viz/zarr_viz/__main__.py to accept .h5 file paths
- [x] T052 [US4] Update src/hdmea/viz/__init__.py exports if needed

**Checkpoint**: ✅ User Story 4 complete - visualization works with HDF5 files

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Cleanup, deprecation, and documentation

- [x] T053 [P] Add deprecation warning to src/hdmea/io/zarr_store.py module docstring
- [x] T054 [P] Update docstrings in src/hdmea/io/hdf5_store.py for all functions
- [x] T055 Run all existing unit tests to verify no regressions
- [x] T056 Run quickstart.md validation with actual HDF5 file
- [x] T057 [P] Update pyproject.toml to remove zarr from required dependencies (move to optional)
- [x] T058 Verify single-writer error messages are clear and actionable

---

## ✅ IMPLEMENTATION COMPLETE

All tasks completed. The HD-MEA pipeline now uses HDF5 (`.h5`) as the primary data format:
- Core I/O in `hdf5_store.py` with full read/write support
- Pipeline runner creates `.h5` files instead of `.zarr` directories
- All feature extractors updated for HDF5 compatibility
- Visualization module supports both HDF5 and legacy Zarr files
- Zarr moved to optional dependencies with deprecation warning

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 - BLOCKS all user stories
- **US1 & US2 (Phase 3)**: Depends on Phase 2 completion
- **US3 (Phase 4)**: Depends on Phase 3 completion (needs working HDF5 I/O)
- **US4 (Phase 5)**: Depends on Phase 3 completion (needs working HDF5 files)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

```
Phase 2 (Foundational)
        │
        ▼
┌───────────────────┐
│  US1 & US2 (P1)   │  ← MVP: Write & Read HDF5
└───────────────────┘
        │
        ├───────────────────┐
        ▼                   ▼
┌───────────┐       ┌───────────┐
│  US3 (P2) │       │  US4 (P3) │  ← Can run in parallel
│ Features  │       │    Viz    │
└───────────┘       └───────────┘
```

### Within Each Phase

- Tests (if included) → Implementation
- Core functions → Dependent functions
- Module updates → Consumer updates (pipeline, features, viz)

### Parallel Opportunities

**Phase 1 (Setup)**: T002, T003 can run in parallel  
**Phase 2 (Foundational)**: T008-T011 can run in parallel (after T004-T007)  
**Phase 3 (US1/US2)**: T012-T017 (tests) can all run in parallel  
**Phase 4 (US3)**: T033-T034 (tests), T037-T045 (feature extractors) can all run in parallel  
**Phase 5 (US4)**: Tasks are sequential (dependencies within viz module)  
**Phase 6 (Polish)**: T053, T054, T057 can run in parallel

---

## Parallel Example: User Story 3

```bash
# Launch all feature extractor updates in parallel:
Task: "Update src/hdmea/features/baseline/baseline_127.py to use HDF5 operations"
Task: "Update src/hdmea/features/on_off/step_up.py to use HDF5 operations"
Task: "Update src/hdmea/features/receptive_field/dense_noise.py to use HDF5 operations"
Task: "Update src/hdmea/features/direction/moving_bar.py to use HDF5 operations"
Task: "Update src/hdmea/features/frequency/chirp.py to use HDF5 operations"
Task: "Update src/hdmea/features/chromatic/green_blue.py to use HDF5 operations"
Task: "Update src/hdmea/features/frif/frif_extractor.py to use HDF5 operations"
Task: "Update src/hdmea/features/cell_type/rgc_classifier.py to use HDF5 operations"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (core hdf5_store.py)
3. Complete Phase 3: User Stories 1 & 2 (write + read)
4. **STOP and VALIDATE**: Test with real CMTR file
5. Pipeline now creates and reads HDF5 files

### Incremental Delivery

1. **Setup + Foundational** → Core module ready
2. **US1 + US2** → Test with pipeline → **MVP Complete!**
3. **US3** → Test feature extraction → Features work
4. **US4** → Test visualization → Full feature parity
5. **Polish** → Deprecate Zarr → Clean release

### Key Validation Points

| Checkpoint | Validation |
|------------|------------|
| After T011 | All core hdf5_store functions exist |
| After T032 | Pipeline creates valid .h5 files |
| After T045 | Feature extraction works with HDF5 |
| After T052 | Visualization works with HDF5 |
| After T058 | Ready for release |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US1 and US2 are combined (both P1, tightly coupled write/read)
- Each checkpoint validates independent functionality
- Commit after each task or logical group
- Existing zarr_store.py kept for reference (deprecated, not deleted)

