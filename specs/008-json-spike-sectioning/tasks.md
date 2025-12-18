# Tasks: JSON-Based Spike Sectioning

**Input**: Design documents from `/specs/008-json-spike-sectioning/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested in spec. Test tasks omitted.

**Organization**: Tasks grouped by user story for independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/hdmea/`, `tests/` at repository root
- Config files in `config/stimuli/`

---

## Phase 1: Setup

**Purpose**: Project constants and type definitions

- [x] T001 Add `DEFAULT_CONFIG_DIR` constant in `src/hdmea/io/spike_sectioning.py`
- [x] T002 [P] Add `StimuliConfigDict` TypedDict for type hints in `src/hdmea/io/spike_sectioning.py`

---

## Phase 2: Foundational (Config Loading Infrastructure)

**Purpose**: Core config loading functions that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Implement `_load_stimuli_config()` helper function in `src/hdmea/io/spike_sectioning.py`
- [x] T004 Implement `_validate_all_configs()` for fail-fast validation in `src/hdmea/io/spike_sectioning.py`
- [x] T005 Add `config_dir` parameter to `section_spike_times()` function signature in `src/hdmea/io/spike_sectioning.py`

**Checkpoint**: Config loading infrastructure ready - user story implementation can begin ✅

---

## Phase 3: User Story 1+2 - JSON Config Sectioning with Frame Conversion (Priority: P1) 🎯 MVP

**Goal**: Load trial parameters from JSON config and accurately convert frame numbers to sample indices for spike sectioning.

**Independent Test**: Call `section_spike_times()` on HDF5 file with `moving_h_bar_s5_d8_3x` movie. Verify 3 trials created with correct boundaries from JSON config (`start_frame=60`, `trial_length_frame=4520`, `repeat=3`).

### Implementation for User Stories 1+2

- [x] T006 [US1] Implement `_calculate_trial_boundaries()` function using frame-to-sample conversion in `src/hdmea/io/spike_sectioning.py`
- [x] T007 [US1] Modify `_section_unit_spikes()` to accept `trial_boundaries` list instead of computing from section_time in `src/hdmea/io/spike_sectioning.py`
- [x] T008 [US1] Update `section_spike_times()` to load JSON configs before processing in `src/hdmea/io/spike_sectioning.py`
- [x] T009 [US1] Update main processing loop to use `_calculate_trial_boundaries()` per movie in `src/hdmea/io/spike_sectioning.py`
- [x] T010 [US1] Update `compute_unit_sections()` inner function to use new trial boundary format in `src/hdmea/io/spike_sectioning.py`

**Checkpoint**: Core JSON-based sectioning works for single movie type ✅

---

## Phase 4: User Story 3 - Strict Config Validation (Priority: P2)

**Goal**: Fail with comprehensive error listing all missing/invalid JSON configs before any processing begins.

**Independent Test**: Call `section_spike_times()` on HDF5 with movie "unknown_stimulus" (no JSON config). Verify `ValueError` raised listing the missing config.

### Implementation for User Story 3

- [x] T011 [US3] Add upfront validation call in `section_spike_times()` before any unit processing in `src/hdmea/io/spike_sectioning.py`
- [x] T012 [US3] Implement comprehensive error message formatting listing all missing configs in `src/hdmea/io/spike_sectioning.py`
- [x] T013 [US3] Add validation for required `section_kwargs` fields (start_frame, trial_length_frame, repeat) in `_load_stimuli_config()` in `src/hdmea/io/spike_sectioning.py`
- [x] T014 [US3] Add validation for field types and value constraints in `_load_stimuli_config()` in `src/hdmea/io/spike_sectioning.py`

**Checkpoint**: Invalid/missing configs detected upfront with clear errors ✅

---

## Phase 5: User Story 4 - Multiple Stimulus Types (Priority: P2)

**Goal**: Handle multiple movies in single recording, each with its own JSON config parameters.

**Independent Test**: Call `section_spike_times()` on HDF5 with both `step_up_5s_5i_3x` and `perfect_dense_noise_15x15_15hz_r42_3min`. Verify each movie sectioned with its specific parameters.

### Implementation for User Story 4

- [x] T015 [US4] Ensure config cache loads all movie configs at start of `section_spike_times()` in `src/hdmea/io/spike_sectioning.py`
- [x] T016 [US4] Update movie processing loop to retrieve correct config per movie in `src/hdmea/io/spike_sectioning.py`
- [x] T017 [US4] Add logging to show which config file used for each movie in `src/hdmea/io/spike_sectioning.py`

**Checkpoint**: Multiple stimulus types processed with correct individual configs ✅

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Deprecation, documentation, cleanup

- [x] T018 Add deprecation warning for `trial_repeats` parameter in `section_spike_times()` in `src/hdmea/io/spike_sectioning.py`
- [x] T019 [P] Update function docstrings to reflect JSON config usage in `src/hdmea/io/spike_sectioning.py`
- [x] T020 [P] Add info logging showing config directory and loaded configs in `src/hdmea/io/spike_sectioning.py`
- [x] T021 [P] Update module docstring at top of `src/hdmea/io/spike_sectioning.py`
- [ ] T022 Run quickstart.md validation scenarios manually

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories 1+2 (Phase 3)**: Depends on Foundational
- **User Story 3 (Phase 4)**: Can run after Foundational, parallel with Phase 3
- **User Story 4 (Phase 5)**: Depends on Phase 3 (needs core sectioning working)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Stories 1+2 (P1)**: Core functionality - no dependencies on other stories
- **User Story 3 (P2)**: Validation logic - can be implemented in parallel with US1+2
- **User Story 4 (P2)**: Multi-movie support - depends on US1+2 being functional

### Within Each Phase

- Setup: T001 → T002 (T002 can be parallel)
- Foundational: T003 → T004 → T005 (sequential - each builds on previous)
- US1+2: T006 → T007 → T008 → T009 → T010 (sequential - integration)
- US3: T011 → T012 → T013 → T014 (T013, T014 can be parallel after T012)
- US4: T015 → T016 → T017 (sequential)
- Polish: All [P] tasks can run in parallel

### Parallel Opportunities

```
Phase 1:
  T001 ─┬─> T002 [P]
        │
Phase 2:
  T003 ──> T004 ──> T005
        │
Phase 3 (US1+2):           Phase 4 (US3) - can run in parallel:
  T006 ──> T007 ──>          T011 ──> T012 ──> T013 [P]
  T008 ──> T009 ──>                         ──> T014 [P]
  T010
        │
Phase 5 (US4):
  T015 ──> T016 ──> T017
        │
Phase 6:
  T018, T019 [P], T020 [P], T021 [P] ──> T022
```

---

## Parallel Example: Setup Phase

```bash
# After T001 completes, T002 can run in parallel:
Task T001: "Add DEFAULT_CONFIG_DIR constant"
Task T002: "Add StimuliConfigDict TypedDict" [P]
```

## Parallel Example: Polish Phase

```bash
# All documentation tasks can run in parallel:
Task T019: "Update function docstrings" [P]
Task T020: "Add info logging" [P]
Task T021: "Update module docstring" [P]
```

---

## Implementation Strategy

### MVP First (User Stories 1+2 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T005)
3. Complete Phase 3: User Stories 1+2 (T006-T010)
4. **STOP and VALIDATE**: Test with single movie type
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Config loading infrastructure ready
2. Add US1+2 → Test with single movie → Working MVP!
3. Add US3 → Fail-fast validation → Robust error handling
4. Add US4 → Multi-movie support → Full feature complete
5. Polish → Documentation and deprecation warnings

### Single Developer Strategy

Execute phases sequentially:
1. Phase 1 → Phase 2 → Phase 3 (MVP checkpoint)
2. Phase 4 → Phase 5 → Phase 6

---

## Notes

- All changes confined to single file: `src/hdmea/io/spike_sectioning.py`
- Existing JSON configs in `config/stimuli/` - no new configs needed
- Frame conversion uses existing `_convert_frame_to_sample_index()` from `section_time.py`
- HDF5 output structure unchanged - backward compatible
- `trial_repeats` parameter deprecated but signature preserved

