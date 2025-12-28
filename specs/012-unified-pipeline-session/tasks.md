# Tasks: Unified Pipeline Session

**Input**: Design documents from `/specs/012-unified-pipeline-session/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Integration test included for validation against reference file (FR-013).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Package code**: `src/hdmea/pipeline/`
- **Step wrappers**: `Projects/unified_pipeline/steps/`
- **Example scripts**: `Projects/unified_pipeline/`
- **Tests**: `tests/integration/`

---

## Phase 1: Setup (Project Structure)

**Purpose**: Create directory structure and initialization files

- [x] T001 Create `Projects/unified_pipeline/` directory structure per plan.md
- [x] T002 Create `Projects/unified_pipeline/__init__.py` with package docstring
- [x] T003 [P] Create `Projects/unified_pipeline/steps/__init__.py` with step imports
- [x] T004 [P] Create `Projects/unified_pipeline/config.py` with configuration constants (paths, parameters)

---

## Phase 2: Foundational (Core Infrastructure)

**Purpose**: Core components that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Update `src/hdmea/pipeline/session.py` to change `save()` default from `overwrite=True` to `overwrite=False`
- [x] T006 Update `src/hdmea/pipeline/session.py` to add colorama import and red warning support for logging
- [x] T007 [P] Create `src/hdmea/pipeline/loader.py` with `load_session_from_hdf5()` function signature and docstring
- [x] T008 Implement `load_session_from_hdf5()` recursive HDF5 loading logic in `src/hdmea/pipeline/loader.py`
- [x] T009 Add `load_features` optional filter parameter to `load_session_from_hdf5()` in `src/hdmea/pipeline/loader.py`
- [x] T010 Update `src/hdmea/pipeline/__init__.py` to export `load_session_from_hdf5`
- [x] T011 Add tqdm progress bar wrapper utility to `Projects/unified_pipeline/config.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Process Recording from CMCR/CMTR (Priority: P1) 🎯 MVP

**Goal**: Complete pipeline from raw CMCR/CMTR files through all 11 steps to final HDF5 output

**Independent Test**: Run `run_single_from_cmcr.py` with test file `2024.08.08-10.40.20-Rec`, verify output matches reference file structure

### Step Wrappers for US1

- [x] T012 [P] [US1] Create `Projects/unified_pipeline/steps/load_recording.py` wrapping `load_recording_with_eimage_sta`
- [x] T013 [P] [US1] Create `Projects/unified_pipeline/steps/section_time.py` wrapping `add_section_time`, `section_spike_times`, `compute_sta`
- [x] T014 [P] [US1] Create `Projects/unified_pipeline/steps/metadata.py` wrapping `add_cmtr_unit_info`, `add_sys_meta_info`
- [x] T015 [P] [US1] Create `Projects/unified_pipeline/steps/geometry.py` wrapping `extract_eimage_sta_geometry`, `extract_rf_geometry_session`
- [x] T016 [P] [US1] Create `Projects/unified_pipeline/steps/gsheet.py` wrapping gsheet loading with red warning on failure
- [x] T017 [P] [US1] Create `Projects/unified_pipeline/steps/cell_type.py` wrapping manual cell type label loading
- [x] T018 [P] [US1] Create `Projects/unified_pipeline/steps/ap_tracking.py` wrapping `compute_ap_tracking`
- [x] T019 [P] [US1] Create `Projects/unified_pipeline/steps/dsgc.py` wrapping `section_by_direction`

### Example Script for US1

- [x] T020 [US1] Create `Projects/unified_pipeline/run_single_from_cmcr.py` with full 11-step pipeline (depends on T012-T019)
- [x] T021 [US1] Add configuration section to `run_single_from_cmcr.py` for test file paths
- [x] T022 [US1] Add progress logging (logger.info for steps, tqdm for units) to all step wrappers

### Validation for US1

- [x] T023 [US1] Create `tests/integration/test_unified_pipeline.py` with reference file comparison test
- [ ] T024 [US1] Run `run_single_from_cmcr.py` with `2024.08.08-10.40.20-Rec` and validate output structure matches reference

**Checkpoint**: User Story 1 complete - can process recordings from CMCR/CMTR through full pipeline

---

## Phase 4: User Story 2 - Resume Pipeline from Existing HDF5 (Priority: P2)

**Goal**: Load existing HDF5 file and continue processing from any intermediate state

**Independent Test**: Load intermediate HDF5, run remaining steps, verify all data preserved and new features added

### Implementation for US2

- [x] T025 [US2] Verify `load_session_from_hdf5()` correctly restores `completed_steps` from HDF5
- [x] T026 [US2] Add logic to step wrappers to skip if step already in `completed_steps` in `Projects/unified_pipeline/steps/__init__.py`
- [x] T027 [US2] Create `Projects/unified_pipeline/run_single_from_hdf5.py` example showing resume workflow

### Validation for US2

- [x] T028 [US2] Test loading intermediate HDF5 and running remaining steps in `tests/integration/test_unified_pipeline.py`
- [x] T029 [US2] Verify all original data preserved after additional processing

**Checkpoint**: User Story 2 complete - can resume processing from any checkpoint

---

## Phase 5: User Story 3 - Export with Flexible Options (Priority: P3)

**Goal**: Save results with choice of new file or overwrite, with testing safeguards

**Independent Test**: Save to new path (verify no overwrite), save with overwrite=True (verify update)

### Implementation for US3

- [x] T030 [US3] Add `FileExistsError` handling in example scripts with user-friendly message
- [x] T031 [US3] Add logging of blocked overwrite attempts (SC-006) in `src/hdmea/pipeline/session.py`
- [x] T032 [US3] Update `run_single_from_cmcr.py` to use output path in `Projects/unified_pipeline/test_output/`

### Validation for US3

- [x] T033 [US3] Test that `save()` raises `FileExistsError` when file exists and `overwrite=False`
- [x] T034 [US3] Test that `save(overwrite=True)` succeeds and logs warning

**Checkpoint**: User Story 3 complete - flexible, safe export options working

---

## Phase 6: User Story 4 - Extensible Step Pattern (Priority: P4)

**Goal**: Document and validate that new steps can be added without modifying core infrastructure

**Independent Test**: Create a minimal new step wrapper, verify it integrates with session workflow

### Implementation for US4

- [x] T035 [P] [US4] Create `Projects/unified_pipeline/steps/template.py` as a documented template for new steps
- [x] T036 [US4] Add docstring to `Projects/unified_pipeline/steps/__init__.py` explaining step pattern
- [x] T037 [US4] Verify new step can be debugged independently by importing and calling with minimal session

### Validation for US4

- [x] T038 [US4] Document step addition process in `Projects/unified_pipeline/README.md`

**Checkpoint**: User Story 4 complete - extensibility pattern documented and validated

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and documentation

- [x] T039 [P] Update `Projects/unified_pipeline/unified_pipeline_explained.md` with final pipeline flow
- [x] T040 [P] Add type hints to all new functions in `src/hdmea/pipeline/loader.py`
- [x] T041 [P] Add type hints to all step wrapper functions in `Projects/unified_pipeline/steps/`
- [ ] T042 Run full pipeline on test file and compare output to reference at `Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5`
- [ ] T043 Verify pipeline completes in < 10 minutes (SC-001)
- [x] T044 Verify example scripts have < 20 lines of core pipeline logic (SC-004)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup ────────────────────┐
                                   │
                                   ▼
Phase 2: Foundational ─────────────┤
         (BLOCKS all US)           │
                                   ▼
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
Phase 3: US1 (MVP)          Phase 4: US2              Phase 5: US3
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                            Phase 6: US4
                                   │
                                   ▼
                            Phase 7: Polish
```

### User Story Dependencies

| Story | Depends On | Notes |
|-------|------------|-------|
| US1 (P1) | Foundational only | MVP - can be completed first |
| US2 (P2) | Foundational + loader.py | Uses universal loader from Phase 2 |
| US3 (P3) | Foundational + session.py changes | Uses save() with new default |
| US4 (P4) | US1 complete | Needs working step pattern to document |

### Within Each Phase

- Tasks marked [P] can run in parallel
- Tasks without [P] have dependencies on earlier tasks in same phase

---

## Parallel Opportunities

### Phase 2 (Foundational)
```
Parallel Group A:
  T007: Create loader.py signature
  T011: Add tqdm utility
  
Sequential after T007:
  T008: Implement loading logic
  T009: Add load_features filter
```

### Phase 3 (US1 - Step Wrappers)
```
All step wrappers (T012-T019) can be created in parallel:
  T012: load_recording.py
  T013: section_time.py
  T014: metadata.py
  T015: geometry.py
  T016: gsheet.py
  T017: cell_type.py
  T018: ap_tracking.py
  T019: dsgc.py
```

### Phase 7 (Polish)
```
All documentation tasks (T039-T041) can run in parallel
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. ✅ Complete Phase 1: Setup (T001-T004)
2. ✅ Complete Phase 2: Foundational (T005-T011)
3. ✅ Complete Phase 3: User Story 1 (T012-T024)
4. **STOP and VALIDATE**: Run `run_single_from_cmcr.py` with test file
5. Verify output matches reference file structure

### Incremental Delivery

| Increment | Delivers | Value |
|-----------|----------|-------|
| MVP (US1) | Full pipeline from CMCR/CMTR | Core processing capability |
| +US2 | Resume from HDF5 | Incremental processing |
| +US3 | Safe export options | Data protection |
| +US4 | Extensibility docs | Developer experience |

### Recommended Execution Order

1. T001-T004 (Setup) - 4 tasks
2. T005-T011 (Foundational) - 7 tasks
3. T012-T024 (US1) - 13 tasks ← **MVP Complete**
4. T025-T029 (US2) - 5 tasks
5. T030-T034 (US3) - 5 tasks
6. T035-T038 (US4) - 4 tasks
7. T039-T044 (Polish) - 6 tasks

---

## Task Summary

| Phase | Tasks | Parallelizable |
|-------|-------|----------------|
| Phase 1: Setup | 4 | 2 |
| Phase 2: Foundational | 7 | 2 |
| Phase 3: US1 (MVP) | 13 | 8 |
| Phase 4: US2 | 5 | 0 |
| Phase 5: US3 | 5 | 0 |
| Phase 6: US4 | 4 | 1 |
| Phase 7: Polish | 6 | 3 |
| **Total** | **44** | **16** |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Test file: `2024.08.08-10.40.20-Rec`
- Reference output: `Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5`

