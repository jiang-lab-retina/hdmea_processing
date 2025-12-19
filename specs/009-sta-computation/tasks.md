# Tasks: STA Computation

**Input**: Design documents from `/specs/009-sta-computation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api.md, quickstart.md

**Tests**: Unit tests included per constitution requirements.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/hdmea/`, `tests/` at repository root
- Per constitution: Feature code in `src/hdmea/features/`

---

## Phase 1: Setup

**Purpose**: Module initialization and dependencies

- [x] T001 Create STA feature module at src/hdmea/features/sta.py with module docstring
- [x] T002 [P] Create STA test module at tests/unit/test_sta.py with imports
- [x] T003 [P] Create STA fixtures at tests/fixtures/sta_fixtures.py for synthetic data

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement `_load_stimulus_movie()` in src/hdmea/features/sta.py - loads .npy file, validates dtype, logs warning if not uint8
- [x] T005 [P] Implement `_convert_spikes_to_frames()` in src/hdmea/features/sta.py - converts sampling indices to frame numbers using rounding
- [x] T006 [P] Add STAResult dataclass in src/hdmea/features/sta.py per contracts/api.md
- [x] T007 Implement `_write_sta_to_hdf5()` in src/hdmea/features/sta.py - writes STA array to units/{id}/features/{movie}/sta
- [x] T008 [P] Create synthetic movie generator in tests/fixtures/sta_fixtures.py - small 10x10x100 uint8 array
- [x] T009 [P] Create synthetic spike generator in tests/fixtures/sta_fixtures.py - spike times compatible with test movie

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Compute STA for Noise Movie (Priority: P1) 🎯 MVP

**Goal**: Compute STA for all units using noise movie stimulus and save results to HDF5

**Independent Test**: Run `compute_sta("artifacts/2025.04.10-11.12.57-Rec.h5")` and verify STA arrays saved

### Tests for User Story 1

- [x] T010 [P] [US1] Unit test for `_compute_sta_for_unit()` with synthetic data in tests/unit/test_sta.py
- [x] T011 [P] [US1] Unit test for correct STA shape (time × height × width) in tests/unit/test_sta.py

### Implementation for User Story 1

- [x] T012 [US1] Implement `_compute_sta_for_unit()` in src/hdmea/features/sta.py - vectorized window extraction per research.md
- [x] T013 [US1] Implement `compute_sta()` main function in src/hdmea/features/sta.py - sequential version first
- [x] T014 [US1] Add spike time loading from HDF5 trials_spike_times/0 in src/hdmea/features/sta.py
- [x] T015 [US1] Integrate spike-to-frame conversion in compute_sta() flow in src/hdmea/features/sta.py
- [x] T016 [US1] Export compute_sta in src/hdmea/features/__init__.py

**Checkpoint**: User Story 1 complete - basic STA computation works without multiprocessing

---

## Phase 4: User Story 2 - Validate Noise Movie Detection (Priority: P1)

**Goal**: Automatically find noise movie and provide clear errors when detection fails

**Independent Test**: Test with 0, 1, and 2 noise movies - verify correct behavior for each case

### Tests for User Story 2

- [x] T017 [P] [US2] Unit test for `_find_noise_movie()` with exactly one match in tests/unit/test_sta.py
- [x] T018 [P] [US2] Unit test for `_find_noise_movie()` error when zero matches in tests/unit/test_sta.py
- [x] T019 [P] [US2] Unit test for `_find_noise_movie()` error when multiple matches in tests/unit/test_sta.py

### Implementation for User Story 2

- [x] T020 [US2] Implement `_find_noise_movie()` in src/hdmea/features/sta.py - case-insensitive search for "noise"
- [x] T021 [US2] Add clear ValueError messages listing found movies in src/hdmea/features/sta.py
- [x] T022 [US2] Integrate noise movie detection in compute_sta() in src/hdmea/features/sta.py

**Checkpoint**: User Story 2 complete - noise movie detection works with clear error messages

---

## Phase 5: User Story 3 - Parallel Processing for Performance (Priority: P2)

**Goal**: Process units in parallel using 80% of CPU cores with shared memory for stimulus

**Independent Test**: Compare timing with `use_multiprocessing=True` vs `False` - parallel should be faster

### Tests for User Story 3

- [x] T023 [P] [US3] Unit test for shared memory creation and cleanup in tests/unit/test_sta.py
- [x] T024 [P] [US3] Unit test for worker count calculation (80% of CPU) in tests/unit/test_sta.py

### Implementation for User Story 3

- [x] T025 [US3] Implement shared memory setup for stimulus array in src/hdmea/features/sta.py per research.md
- [x] T026 [US3] Implement worker function `_sta_worker()` that attaches to shared memory in src/hdmea/features/sta.py
- [x] T027 [US3] Implement multiprocessing pool with 80% CPU count in src/hdmea/features/sta.py
- [x] T028 [US3] Add `use_multiprocessing` parameter to compute_sta() in src/hdmea/features/sta.py
- [x] T029 [US3] Add progress bar with tqdm.imap integration in src/hdmea/features/sta.py
- [x] T030 [US3] Ensure shared memory cleanup in finally block in src/hdmea/features/sta.py

**Checkpoint**: User Story 3 complete - multiprocessing with progress bar works

---

## Phase 6: User Story 4 - Handle Edge Effects (Priority: P2)

**Goal**: Correctly exclude spikes near stimulus boundaries to prevent out-of-bounds errors

**Independent Test**: Verify spikes at frame 0 and frame N-1 are handled correctly

### Tests for User Story 4

- [x] T031 [P] [US4] Unit test for edge exclusion at start (spike < abs(cover_range[0])) in tests/unit/test_sta.py
- [x] T032 [P] [US4] Unit test for edge exclusion at end (spike > movie_length - cover_range[1]) in tests/unit/test_sta.py
- [x] T033 [P] [US4] Unit test for n_spikes_excluded count accuracy in tests/unit/test_sta.py

### Implementation for User Story 4

- [x] T034 [US4] Implement vectorized valid spike mask in `_compute_sta_for_unit()` in src/hdmea/features/sta.py
- [x] T035 [US4] Track and return n_spikes_excluded count in src/hdmea/features/sta.py
- [x] T036 [US4] Handle case of zero valid spikes - return NaN STA with warning in src/hdmea/features/sta.py
- [x] T037 [US4] Add cover_range validation (start < end) in compute_sta() in src/hdmea/features/sta.py

**Checkpoint**: User Story 4 complete - edge effects handled correctly

---

## Phase 7: Error Handling & Retry

**Purpose**: Robust error handling per FR-017

- [x] T038 Implement retry logic wrapper `_compute_with_retry()` in src/hdmea/features/sta.py
- [x] T039 [P] Unit test for retry on first failure in tests/unit/test_sta.py
- [x] T040 [P] Unit test for skip after second failure in tests/unit/test_sta.py
- [x] T041 Add failed units list to STAResult in src/hdmea/features/sta.py

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final integration and documentation

- [x] T042 Run full integration test with artifacts/2025.04.10-11.12.57-Rec.h5
- [x] T043 [P] Add type hints to all public functions in src/hdmea/features/sta.py
- [x] T044 [P] Add Google-style docstrings to all functions per constitution in src/hdmea/features/sta.py
- [x] T045 Update docs/pipeline_log.md with STA feature entry
- [x] T046 Run quickstart.md validation - verify all examples work

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational completion
  - US1 and US2 can proceed in parallel (both P1)
  - US3 and US4 can proceed in parallel (both P2)
  - US3/US4 logically depend on US1 being functional
- **Error Handling (Phase 7)**: Depends on US3 (multiprocessing)
- **Polish (Phase 8)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)**: Core STA computation - no dependencies on other stories
- **US2 (P1)**: Noise movie detection - no dependencies on other stories
- **US3 (P2)**: Multiprocessing - builds on US1 implementation
- **US4 (P2)**: Edge handling - builds on US1 implementation

### Parallel Opportunities

- T002, T003 can run in parallel (different test files)
- T005, T006, T008, T009 can run in parallel (independent helpers)
- All US1 tests (T010, T011) can run in parallel
- All US2 tests (T017, T018, T019) can run in parallel
- All US3 tests (T023, T024) can run in parallel
- All US4 tests (T031, T032, T033) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for US1 together:
Task: "Unit test for _compute_sta_for_unit() with synthetic data"
Task: "Unit test for correct STA shape"

# After tests fail, implement in order:
Task: "Implement _compute_sta_for_unit()"  # Core algorithm
Task: "Implement compute_sta() main function"  # Orchestration
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T009)
3. Complete Phase 3: User Story 1 (T010-T016)
4. **STOP and VALIDATE**: Test with real HDF5 file
5. Core STA computation works - can demo/use

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 → Basic STA works (MVP!)
3. Add US2 → Automatic noise movie detection
4. Add US3 → Performance with multiprocessing
5. Add US4 → Robust edge handling
6. Add Error Handling → Fault tolerance
7. Polish → Production ready

### Recommended Order for Single Developer

1. T001 → T003 (Setup)
2. T004 → T009 (Foundational - can parallelize T005/T006/T008/T009)
3. T010 → T016 (US1 - MVP)
4. T020 → T022 (US2 - detection)
5. T034 → T037 (US4 - edge handling)
6. T025 → T030 (US3 - multiprocessing)
7. T038 → T041 (Error handling)
8. T042 → T046 (Polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently testable
- Test file: artifacts/2025.04.10-11.12.57-Rec.h5
- Stimuli dir: M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\
- Cover range default: (-60, 0)

