# Tasks: Electrode Image STA (eimage_sta)

**Input**: Design documents from `/specs/001-eimage-sta-feature/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/eimage_sta_api.md ✅

**Tests**: Integration test included for performance validation (required by SC-001: <5 minute target).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Includes exact file paths in descriptions

## Path Conventions

- **Single project**: Following existing hdmea package layout
- Main package: `src/hdmea/`
- Tests: `tests/`

---

## Phase 1: Setup (Module Initialization)

**Purpose**: Create the eimage_sta feature module structure

- [x] T001 Create eimage_sta module directory at `src/hdmea/features/eimage_sta/`
- [x] T002 Create module `__init__.py` with public exports in `src/hdmea/features/eimage_sta/__init__.py`
- [x] T003 [P] Create test fixtures module at `tests/fixtures/eimage_sta_fixtures.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### IO Infrastructure

- [x] T004 Add `load_sensor_data()` function to `src/hdmea/io/cmcr.py` for CMCR sensor data access via McsPy
- [x] T005 Add sensor data validation and error handling in `src/hdmea/io/cmcr.py`

### Filtering Infrastructure

- [x] T006 [P] Add `apply_highpass_filter_3d()` function to `src/hdmea/preprocess/filtering.py` using vectorized scipy.signal.filtfilt with axis=0
- [x] T007 [P] Add filter parameter validation (cutoff_hz > 0, order >= 1) in `src/hdmea/preprocess/filtering.py`

### Data Classes

- [x] T008 [P] Create `EImageSTAConfig` dataclass in `src/hdmea/features/eimage_sta/compute.py`
- [x] T009 [P] Create `EImageSTAResult` dataclass in `src/hdmea/features/eimage_sta/compute.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Compute Electrode Image STA for Unit Analysis (Priority: P1) 🎯 MVP

**Goal**: Compute and store eimage_sta for all units in a recording, with metadata

**Independent Test**: Load HDF5 file, run compute_eimage_sta(), verify each unit has valid 3D array in features group

### Core Computation (US1)

- [x] T010 [US1] Implement `compute_sta_for_unit()` in `src/hdmea/features/eimage_sta/compute.py` using vectorized NumPy fancy indexing for window extraction
- [x] T011 [US1] Implement edge effect handling (exclude spikes too close to boundaries) in `compute_sta_for_unit()`
- [x] T012 [US1] Implement NaN result for units with zero valid spikes in `compute_sta_for_unit()`

### HDF5 Integration (US1)

- [x] T013 [US1] Implement `write_eimage_sta_to_hdf5()` in `src/hdmea/features/eimage_sta/compute.py` for storing result to HDF5
- [x] T014 [US1] Add metadata attributes (n_spikes, n_spikes_excluded, pre_samples, post_samples, version) to HDF5 write
- [x] T015 [US1] Implement skip-if-exists logic with force parameter in write function

### Main Entry Point (US1)

- [x] T016 [US1] Implement `compute_eimage_sta()` main function in `src/hdmea/features/eimage_sta/compute.py`
- [x] T017 [US1] Add tqdm progress bar for unit iteration in `compute_eimage_sta()`
- [x] T018 [US1] Add logging for timing information (filter time, total time) in `compute_eimage_sta()`

### Feature Extractor (US1)

- [x] T019 [US1] Create `EImageSTAExtractor` class in `src/hdmea/features/eimage_sta/extractor.py` with @FeatureRegistry.register decorator
- [x] T020 [US1] Implement extractor metadata (name, version, runtime_class, required_inputs, output_schema) in `extractor.py`
- [x] T021 [US1] Export extractor in `src/hdmea/features/eimage_sta/__init__.py`

**Checkpoint**: US1 complete - core eimage_sta computation works with default parameters

---

## Phase 4: User Story 2 - Configure Filter Parameters (Priority: P2)

**Goal**: Allow customization of high-pass filter cutoff frequency and filter order

**Independent Test**: Run computation with custom filter parameters (e.g., cutoff_hz=200), verify metadata reflects custom values

### Parameter Configuration (US2)

- [x] T022 [US2] Add cutoff_hz parameter to `compute_eimage_sta()` with default 100.0 in `src/hdmea/features/eimage_sta/compute.py`
- [x] T023 [US2] Add filter_order parameter to `compute_eimage_sta()` with default 2 in `src/hdmea/features/eimage_sta/compute.py`
- [x] T024 [US2] Store filter parameters (cutoff_hz, filter_order) in HDF5 metadata attributes

### Parameter Validation (US2)

- [x] T025 [US2] Add validation: cutoff_hz must be positive and less than Nyquist in `compute_eimage_sta()`
- [x] T026 [US2] Add validation: filter_order must be >= 1 in `compute_eimage_sta()`

**Checkpoint**: US2 complete - filter parameters are configurable and validated

---

## Phase 5: User Story 3 - Limit Computation Time with Spike Sampling (Priority: P3)

**Goal**: Allow limiting number of spikes per unit to control computation time

**Independent Test**: Run on high-spike unit with spike_limit=100, verify only 100 spikes used (check n_spikes in metadata)

### Spike Limit Implementation (US3)

- [x] T027 [US3] Add spike_limit parameter to `compute_eimage_sta()` with default 10000 in `src/hdmea/features/eimage_sta/compute.py`
- [x] T028 [US3] Implement spike limiting in `compute_sta_for_unit()` (take first N spikes) in `compute.py`
- [x] T029 [US3] Store spike_limit in HDF5 metadata attributes (-1 if no limit)

### Time Window Configuration (US3)

- [x] T030 [US3] Add pre_samples parameter with default 10 in `compute_eimage_sta()`
- [x] T031 [US3] Add post_samples parameter with default 40 in `compute_eimage_sta()`
- [x] T032 [US3] Add duration_s parameter with default 120.0 in `compute_eimage_sta()`

**Checkpoint**: US3 complete - spike limit and time windows are configurable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Performance validation, optional caching, and documentation

### Performance Validation

- [x] T033 Create integration test at `tests/integration/test_eimage_sta_e2e.py` with real test data
- [x] T034 Add performance assertion: total time < 5 minutes for test recording in integration test
- [x] T035 Add shape assertion: output shape matches (window_length, 64, 64) in integration test

### Optional Caching (PR-005)

- [x] T036 [P] Add use_cache parameter to `compute_eimage_sta()` in `src/hdmea/features/eimage_sta/compute.py`
- [x] T037 [P] Implement filtered data caching to HDF5 file in `compute.py`
- [x] T038 [P] Implement cache loading and validation (check filter params match) in `compute.py`

### Documentation

- [x] T039 [P] Add docstrings to all public functions in `src/hdmea/features/eimage_sta/compute.py`
- [x] T040 [P] Add module docstring to `src/hdmea/features/eimage_sta/__init__.py`

### Final Validation

- [x] T041 Run quickstart.md validation with test data paths
- [x] T042 Verify extractor appears in FeatureRegistry.list_all()

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 - Core MVP
- **Phase 4 (US2)**: Depends on US1 core functions being in place
- **Phase 5 (US3)**: Depends on US1 core functions being in place
- **Phase 6 (Polish)**: Depends on US1-US3 completion

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories - **MVP**
- **User Story 2 (P2)**: Extends US1 functions - Can be done in parallel with US3
- **User Story 3 (P3)**: Extends US1 functions - Can be done in parallel with US2

### Within Each User Story

- Core computation before HDF5 write
- HDF5 write before main entry point
- Main entry point before extractor class
- All components complete before checkpoint

### Parallel Opportunities

**Phase 1 (Setup):**
```
T002 (__init__.py) || T003 (fixtures)
```

**Phase 2 (Foundational):**
```
T004-T005 (IO)  ||  T006-T007 (Filtering)  ||  T008-T009 (Data classes)
```

**Phase 4+5 (US2 & US3 can run in parallel after US1):**
```
T022-T026 (US2: Filter params)  ||  T027-T032 (US3: Spike limits)
```

**Phase 6 (Polish):**
```
T036-T038 (Caching)  ||  T039-T040 (Docs)
```

---

## Parallel Example: Foundational Phase

```bash
# These can all run in parallel (different files):
T006: "Add apply_highpass_filter_3d() to src/hdmea/preprocess/filtering.py"
T008: "Create EImageSTAConfig dataclass in src/hdmea/features/eimage_sta/compute.py"
T009: "Create EImageSTAResult dataclass in src/hdmea/features/eimage_sta/compute.py"

# These must be sequential (same file):
T004 → T005: Both modify src/hdmea/io/cmcr.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (3 tasks)
2. Complete Phase 2: Foundational (6 tasks) - CRITICAL
3. Complete Phase 3: User Story 1 (12 tasks)
4. **STOP and VALIDATE**: Run on test data, verify output
5. Total MVP: 21 tasks

### Incremental Delivery

1. **MVP (US1)**: Core computation with defaults → Test with real data
2. **+US2**: Add filter parameter configuration → Test custom params
3. **+US3**: Add spike limiting → Test performance on large datasets
4. **Polish**: Add caching, docs, full test suite

### Test Data

```python
cmcr_path = "O:\\20250410\\set6\\2025.04.10-11.12.57-Rec.cmcr"
cmtr_path = "O:\\20250410\\set6\\2025.04.10-11.12.57-Rec-.cmtr"
hdf5_path = "artifacts/2025.04.10-11.12.57-Rec.h5"
```

---

## Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1: Setup | 3 | Module structure |
| Phase 2: Foundational | 6 | IO, filtering, data classes |
| Phase 3: US1 (P1) | 12 | Core STA computation |
| Phase 4: US2 (P2) | 5 | Filter parameters |
| Phase 5: US3 (P3) | 6 | Spike limits |
| Phase 6: Polish | 10 | Testing, caching, docs |
| **Total** | **42** | |

### MVP Scope

- **Tasks for MVP**: T001-T021 (21 tasks)
- **MVP delivers**: Full eimage_sta computation with default parameters
- **Post-MVP**: US2 (5 tasks), US3 (6 tasks), Polish (10 tasks)

---

## Notes

- [P] tasks = different files, no dependencies
- [US1/US2/US3] label maps task to specific user story
- Each user story is independently testable after completion
- Performance target: <5 minutes for 120s recording with ~100 units
- Commit after each logical group of tasks
- Stop at any checkpoint to validate independently

