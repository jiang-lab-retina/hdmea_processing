# Tasks: Add frame_time and acquisition_rate to Metadata

**Input**: Design documents from `/specs/003-add-metadata-fields/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Unit tests included as they are documented in research.md and plan.md.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/hdmea/`, `tests/` at repository root

---

## Phase 1: Setup

**Purpose**: No new project setup required - feature extends existing codebase

- [x] T001 Review existing acquisition_rate extraction in src/hdmea/io/cmcr.py (no changes needed per research.md)
- [x] T002 Review existing write_metadata() in src/hdmea/io/zarr_store.py (no changes needed per research.md)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add CMTR extraction capability that US1 depends on

**⚠️ CRITICAL**: US1 requires CMTR fallback extraction before priority chain can be implemented

- [x] T003 Add acquisition_rate extraction to load_cmtr_data() in src/hdmea/io/cmtr.py
- [x] T004 Add validate_acquisition_rate() helper function in src/hdmea/pipeline/runner.py

**Checkpoint**: Foundation ready - CMTR can now provide acquisition_rate as fallback

---

## Phase 3: User Story 1 - Access Recording Timing Parameters (Priority: P1) 🎯 MVP

**Goal**: Researchers can access `acquisition_rate` and `frame_time` from Zarr metadata after loading a recording

**Independent Test**: Load a recording from CMCR/CMTR files, verify both `acquisition_rate` (Hz) and `frame_time` (seconds) are present in `/metadata` group attributes

### Tests for User Story 1

- [x] T005 [P] [US1] Create test fixtures for metadata extraction in tests/fixtures/synthetic_zarr.py
- [x] T006 [P] [US1] Create unit tests for acquisition_rate priority chain in tests/unit/test_metadata_fields.py
- [x] T007 [P] [US1] Create unit tests for frame_time computation in tests/unit/test_metadata_fields.py

### Implementation for User Story 1

- [x] T008 [US1] Implement acquisition_rate priority chain in load_recording() in src/hdmea/pipeline/runner.py
- [x] T009 [US1] Implement frame_time computation (1/acquisition_rate) in load_recording() in src/hdmea/pipeline/runner.py
- [x] T010 [US1] Ensure acquisition_rate and frame_time are added to metadata dict before write_metadata() in src/hdmea/pipeline/runner.py
- [x] T011 [US1] Add warning log when using default acquisition_rate (20000 Hz) in src/hdmea/pipeline/runner.py
- [x] T012 [US1] Add warning log when acquisition_rate outside typical range (1000-100000 Hz) in src/hdmea/pipeline/runner.py

**Checkpoint**: User Story 1 complete - metadata fields are stored in Zarr and accessible programmatically

---

## Phase 4: User Story 2 - View Timing Metadata in Zarr Viz GUI (Priority: P2)

**Goal**: Researchers can view `acquisition_rate` and `frame_time` in the zarr-viz GUI without code

**Independent Test**: Open a Zarr file with timing metadata in zarr-viz, navigate to `/metadata` group, verify both fields are displayed in the metadata panel

### Implementation for User Story 2

- [x] T013 [US2] Verify timing metadata displays correctly in metadata panel in src/hdmea/viz/zarr_viz/app.py (enhanced to show group attributes with timing metadata)
- [x] T014 [US2] Add manual test: open sample Zarr in zarr-viz, verify acquisition_rate and frame_time visible

**Checkpoint**: User Story 2 complete - timing metadata visible in GUI

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Validation and documentation updates

- [x] T015 Run all unit tests to verify no regressions: pytest tests/unit/test_metadata_fields.py -v
- [x] T016 [P] Update specs/003-add-metadata-fields/quickstart.md with actual test commands after verification
- [x] T017 Run quickstart.md validation - execute example code snippets
- [x] T018 Update specs/003-add-metadata-fields/checklists/requirements.md to mark implementation complete

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - review only, can start immediately
- **Foundational (Phase 2)**: No dependencies - BLOCKS User Story 1
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion (T003, T004)
- **User Story 2 (Phase 4)**: Depends on Phase 3 completion (needs Zarr with metadata)
- **Polish (Phase 5)**: Depends on Phase 3 and 4 completion

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational phase - No dependencies on US2
- **User Story 2 (P2)**: Requires US1 complete (needs Zarr files with timing metadata to test)

### Within Each User Story

- Tests written first (T005-T007)
- Core implementation (T008-T012)
- Validation and logging after core

### Parallel Opportunities

**Within Phase 2** (different files):
- T003 and T004 can run in parallel (cmtr.py vs runner.py)

**Within US1 Tests** (different test files/functions):
- T005, T006, T007 can run in parallel

**Between Stories** (after US1 complete):
- US2 can start immediately after US1

---

## Parallel Example: User Story 1 Tests

```bash
# Launch all tests for User Story 1 together:
Task T005: "Create test fixtures for metadata extraction in tests/fixtures/synthetic_zarr.py"
Task T006: "Create unit tests for acquisition_rate priority chain in tests/unit/test_metadata_fields.py"
Task T007: "Create unit tests for frame_time computation in tests/unit/test_metadata_fields.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002) - review existing code
2. Complete Phase 2: Foundational (T003-T004) - add CMTR extraction
3. Complete Phase 3: User Story 1 (T005-T012) - core metadata feature
4. **STOP and VALIDATE**: Run tests, verify metadata in Zarr
5. Deploy/demo if ready - researchers can now access timing metadata

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy (MVP!)
3. Add User Story 2 → Verify GUI display → Deploy (complete feature)
4. Polish phase → Documentation and cleanup

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 is independently completable and testable
- US2 depends on US1 for test data but is a verification-only phase
- Most changes are in runner.py (T004, T008-T012) - cannot parallelize within that file
- CMTR extraction (T003) is in separate file, can parallelize with runner.py tasks

