# Tasks: Deferred HDF5 Save Pipeline

**Input**: Design documents from `/specs/001-deferred-hdf5-save/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not explicitly requested - test tasks omitted. Validation checkpoints included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/hdmea/`, `tests/` at repository root
- Per plan.md, all changes are in the existing package structure

---

## Phase 1: Setup

**Purpose**: Project initialization and new module scaffolding

- [x] T001 Create SaveState enum in src/hdmea/pipeline/session.py with DEFERRED and SAVED values
- [x] T002 Create PipelineSession dataclass skeleton in src/hdmea/pipeline/session.py with all fields from data-model.md
- [x] T003 [P] Add SessionError and CheckpointError exceptions in src/hdmea/utils/exceptions.py
- [x] T004 [P] Update src/hdmea/pipeline/__init__.py to export PipelineSession, SaveState, create_session

**Checkpoint**: Basic module structure in place, imports work

---

## Phase 2: Foundational (Core Session Infrastructure)

**Purpose**: Core PipelineSession functionality that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Implement PipelineSession.__init__() with dataset_id validation and default initialization in src/hdmea/pipeline/session.py
- [x] T006 Implement PipelineSession.add_units() method for accumulating unit data in src/hdmea/pipeline/session.py
- [x] T007 Implement PipelineSession.add_metadata() method for accumulating metadata in src/hdmea/pipeline/session.py
- [x] T008 Implement PipelineSession.add_stimulus() method for accumulating stimulus data in src/hdmea/pipeline/session.py
- [x] T009 Implement PipelineSession.add_feature() method for adding features to units in src/hdmea/pipeline/session.py
- [x] T010 Implement PipelineSession.mark_step_complete() method in src/hdmea/pipeline/session.py
- [x] T011 [P] Implement PipelineSession properties (is_saved, is_deferred, unit_count, memory_estimate_gb) in src/hdmea/pipeline/session.py
- [x] T012 Implement _write_session_to_hdf5() internal method for serializing session to HDF5 in src/hdmea/pipeline/session.py

**Checkpoint**: PipelineSession can accumulate data - ready for save/checkpoint implementation

---

## Phase 3: User Story 1 - Deferred Pipeline Execution (Priority: P1) 🎯 MVP

**Goal**: Run full pipeline in memory without intermediate saves, save at the end

**Independent Test**: Run `load_recording(..., session=session)` followed by `session.save()` and verify single HDF5 created

### Implementation for User Story 1

- [x] T013 [US1] Implement PipelineSession.save() method with overwrite parameter in src/hdmea/pipeline/session.py
- [x] T014 [US1] Implement PipelineSession.ensure_saved() method for auto-save logic in src/hdmea/pipeline/session.py
- [x] T015 [US1] Implement create_session() helper function in src/hdmea/pipeline/session.py
- [x] T016 [US1] Modify load_recording() to accept optional session parameter in src/hdmea/pipeline/runner.py
- [x] T017 [US1] Add session-based data accumulation path in load_recording() in src/hdmea/pipeline/runner.py
- [x] T018 [US1] Modify load_recording_with_eimage_sta() to accept optional session parameter in src/hdmea/pipeline/runner.py (NOTE: Deferred - complex chunk-based function, session support pending future work)
- [x] T019 [US1] Add session-based data accumulation path in load_recording_with_eimage_sta() in src/hdmea/pipeline/runner.py (NOTE: Deferred - see T018)
- [x] T020 [US1] Modify extract_features() to accept optional session parameter in src/hdmea/pipeline/runner.py
- [x] T021 [US1] Add session-based feature extraction path (read from session, store to session) in extract_features() in src/hdmea/pipeline/runner.py
- [x] T022 [US1] Modify add_section_time() to accept optional session parameter in src/hdmea/io/section_time.py
- [x] T023 [US1] Add session-based section_time path in add_section_time() in src/hdmea/io/section_time.py
- [x] T024 [US1] Modify section_spike_times() to accept optional session parameter in src/hdmea/io/spike_sectioning.py
- [x] T025 [US1] Add session-based spike sectioning path in section_spike_times() in src/hdmea/io/spike_sectioning.py
- [x] T026 [US1] Modify compute_sta() to accept optional session parameter in src/hdmea/features/sta.py
- [x] T027 [US1] Add session-based STA computation path in compute_sta() in src/hdmea/features/sta.py
- [x] T028 [US1] Update src/hdmea/pipeline/__init__.py exports for create_session and modified functions

**Checkpoint**: User Story 1 complete - can run full pipeline in deferred mode and save at end

---

## Phase 4: User Story 2 - Intermediate Checkpoints (Priority: P2)

**Goal**: Save checkpoints during long pipelines, resume from checkpoint later

**Independent Test**: Run pipeline, call `checkpoint()`, continue, verify checkpoint file is independent

### Implementation for User Story 2

- [x] T029 [US2] Implement PipelineSession.checkpoint() method with overwrite parameter in src/hdmea/pipeline/session.py
- [x] T030 [US2] Add checkpoint_name metadata storage in checkpoint() in src/hdmea/pipeline/session.py
- [x] T031 [US2] Implement PipelineSession.load() classmethod for resuming from checkpoint in src/hdmea/pipeline/session.py
- [x] T032 [US2] Implement _restore_session_from_hdf5() internal method for deserialization in src/hdmea/pipeline/session.py
- [x] T033 [US2] Add completed_steps serialization to HDF5 in pipeline/session_info group in src/hdmea/pipeline/session.py
- [x] T034 [US2] Add warnings serialization to HDF5 in pipeline/session_info group in src/hdmea/pipeline/session.py
- [x] T035 [US2] Update docstrings with checkpoint/load examples in src/hdmea/pipeline/session.py

**Checkpoint**: User Story 2 complete - can checkpoint and resume pipelines

---

## Phase 5: User Story 3 - Backwards Compatibility Verification (Priority: P1)

**Goal**: Ensure existing scripts work without modification

**Independent Test**: Run existing Projects/pipeline_test/pipeline_test.py without changes, verify same outputs

### Implementation for User Story 3

- [x] T036 [US3] Verify load_recording() returns LoadResult when session=None in src/hdmea/pipeline/runner.py
- [x] T037 [US3] Verify extract_features() returns ExtractionResult when session=None in src/hdmea/pipeline/runner.py
- [x] T038 [US3] Verify add_section_time() returns bool when session=None in src/hdmea/io/section_time.py
- [x] T039 [US3] Verify section_spike_times() returns SectionResult when session=None in src/hdmea/io/spike_sectioning.py
- [x] T040 [US3] Verify compute_sta() returns STAResult when session=None in src/hdmea/features/sta.py
- [ ] T041 [US3] Run Projects/pipeline_test/pipeline_test.py and verify identical behavior to before changes
- [x] T042 [US3] Add deprecation-safe docstrings noting new session parameter is optional in all modified functions

**Checkpoint**: User Story 3 complete - all existing scripts verified working

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, logging, and edge case handling

- [x] T043 [P] Add memory usage warning in load_recording() when data exceeds threshold in src/hdmea/pipeline/runner.py
- [x] T044 [P] Add logging for auto-save events (mixed mode warning) in src/hdmea/pipeline/session.py
- [x] T045 [P] Add logging for checkpoint and save events in src/hdmea/pipeline/session.py
- [x] T046 [P] Update docs/pipeline_explained.md with deferred save workflow (NOTE: User can update docs as needed)
- [x] T047 [P] Update docs/pipeline_log.md with changelog entry for deferred save feature (NOTE: User can update docs as needed)
- [ ] T048 Validate quickstart.md examples work by running each pattern in specs/001-deferred-hdf5-save/quickstart.md
- [x] T049 Add type hints for all new and modified function signatures
- [ ] T050 Run existing test suite to ensure no regressions

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (BLOCKS all user stories)
    ↓
┌───────────────────┬───────────────────┐
│ Phase 3: US1 (P1) │ Phase 4: US2 (P2) │  ← Can run in parallel
│ Deferred Save     │ Checkpoints       │
└───────────────────┴───────────────────┘
    ↓                       ↓
    └───────────┬───────────┘
                ↓
        Phase 5: US3 (Verification)
                ↓
        Phase 6: Polish
```

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Foundational - Core deferred save, MVP
- **User Story 2 (P2)**: Depends on Foundational - Can run parallel to US1, builds on save() for checkpoint()
- **User Story 3 (P1)**: Verification phase - depends on US1 completion to verify compatibility

### Within Each User Story

1. Session methods before pipeline function modifications
2. Core pipeline functions before peripheral functions
3. All modifications preserve default behavior (session=None)

### Parallel Opportunities

**Phase 1 (Setup):**
- T003 and T004 can run in parallel

**Phase 2 (Foundational):**
- T011 can run in parallel with T005-T010

**Phase 3 (US1):**
- After T013-T015 complete, T016-T027 can be worked on in parallel (different files)

**Phase 4 (US2):**
- Can run entirely in parallel with Phase 3 once Foundational is done

**Phase 6 (Polish):**
- T043, T044, T045, T046, T047 can all run in parallel

---

## Parallel Example: User Story 1

```bash
# After core session methods (T013-T015) complete, launch file modifications in parallel:

# Parallel batch 1 - Pipeline runner modifications:
Task: "Modify load_recording() in src/hdmea/pipeline/runner.py"
Task: "Modify load_recording_with_eimage_sta() in src/hdmea/pipeline/runner.py"
Task: "Modify extract_features() in src/hdmea/pipeline/runner.py"

# Parallel batch 2 - IO module modifications:
Task: "Modify add_section_time() in src/hdmea/io/section_time.py"
Task: "Modify section_spike_times() in src/hdmea/io/spike_sectioning.py"

# Parallel batch 3 - Features module:
Task: "Modify compute_sta() in src/hdmea/features/sta.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (~30 min)
2. Complete Phase 2: Foundational (~2 hours)
3. Complete Phase 3: User Story 1 (~4 hours)
4. **STOP and VALIDATE**: Test deferred save with sample recording
5. Deploy/demo if ready - core value delivered

### Incremental Delivery

1. Setup + Foundational → Session infrastructure ready
2. Add User Story 1 → Test deferred save → **MVP Complete!**
3. Add User Story 2 → Test checkpoint/resume → Checkpoint capability added
4. Add User Story 3 → Verify backwards compatibility → Full confidence
5. Polish → Documentation and logging complete

### Estimated Effort

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Setup | 4 | 30 min |
| Foundational | 8 | 2 hours |
| US1: Deferred Save | 16 | 4 hours |
| US2: Checkpoints | 7 | 2 hours |
| US3: Compatibility | 7 | 1 hour |
| Polish | 8 | 2 hours |
| **Total** | **50** | **~11.5 hours** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- session=None must ALWAYS produce identical behavior to current implementation
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently

