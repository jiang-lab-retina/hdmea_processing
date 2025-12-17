# Tasks: Load Section Time Metadata

**Input**: Design documents from `/specs/004-load-section-time/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Tests are included based on plan.md reference to pytest.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Create module structure and constants

- [x] T001 Create section_time.py module with docstring and imports in src/hdmea/io/section_time.py
- [x] T002 Define constants (PRE_MARGIN_FRAME_NUM, POST_MARGIN_FRAME_NUM, DEFAULT_PAD_FRAME, default paths) in src/hdmea/io/section_time.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core helper functions required by all user stories

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Implement _convert_frame_to_time() helper function in src/hdmea/io/section_time.py
- [x] T004 [P] Implement _load_playlist_csv() helper function in src/hdmea/io/section_time.py
- [x] T005 [P] Implement _load_movie_length_csv() helper function in src/hdmea/io/section_time.py
- [x] T006 Implement _get_movie_start_end_frame() core algorithm function in src/hdmea/io/section_time.py

**Checkpoint**: Foundation ready - helper functions implemented, user story implementation can begin

---

## Phase 3: User Story 1 - Load Section Times from Playlist (Priority: P1) 🎯 MVP

**Goal**: Enable researchers to automatically segment recordings by visual stimulation movies using a predefined playlist configuration.

**Independent Test**: Load a recording with known playlist and verify frame boundaries match expected values.

### Tests for User Story 1

- [x] T007 [P] [US1] Create test fixtures directory and sample CSV files in tests/fixtures/section_time/
- [x] T008 [P] [US1] Create minimal test Zarr fixture with frame_time and light_reference in tests/fixtures/section_time/

### Implementation for User Story 1

- [x] T009 [US1] Implement add_section_time() main function signature with all parameters in src/hdmea/io/section_time.py
- [x] T010 [US1] Implement Zarr opening and frame_time extraction logic in add_section_time() in src/hdmea/io/section_time.py
- [x] T011 [US1] Implement light reference extraction for template computation in add_section_time() in src/hdmea/io/section_time.py
- [x] T012 [US1] Implement section_time and light_template computation using _get_movie_start_end_frame() in src/hdmea/io/section_time.py
- [x] T013 [US1] Implement Zarr writing for section_time/{movie_name} datasets in src/hdmea/io/section_time.py
- [x] T014 [US1] Implement Zarr writing for light_template/{movie_name} datasets in src/hdmea/io/section_time.py
- [x] T015 [US1] Implement root attributes update (section_time_playlist, section_time_repeats) in src/hdmea/io/section_time.py
- [x] T016 [US1] Add success logging for section time addition in src/hdmea/io/section_time.py
- [x] T017 [US1] Write unit test for add_section_time() happy path in tests/unit/test_section_time.py

**Checkpoint**: Core section time loading works with valid inputs. MVP complete.

---

## Phase 4: User Story 2 - Handle Missing Configuration Files (Priority: P2)

**Goal**: Graceful error handling that prevents pipeline crashes and provides actionable feedback.

**Independent Test**: Call function with non-existent file paths and verify appropriate warnings are logged.

### Implementation for User Story 2

- [x] T018 [US2] Add playlist file not found handling with warning log in add_section_time() in src/hdmea/io/section_time.py
- [x] T019 [US2] Add movie_length file not found handling with warning log in add_section_time() in src/hdmea/io/section_time.py
- [x] T020 [US2] Add playlist name not found handling with available names in error log in add_section_time() in src/hdmea/io/section_time.py
- [x] T021 [US2] Add Zarr not found handling with error log in add_section_time() in src/hdmea/io/section_time.py
- [x] T022 [US2] Add frame_time/frame_timestamps missing handling with error log in add_section_time() in src/hdmea/io/section_time.py
- [x] T023 [US2] Add movie not in movie_length handling (skip with warning) in _get_movie_start_end_frame() in src/hdmea/io/section_time.py
- [x] T024 [US2] Add overwrite protection (raise FileExistsError unless force=True) in add_section_time() in src/hdmea/io/section_time.py
- [x] T025 [US2] Write unit tests for error handling scenarios in tests/unit/test_section_time.py

**Checkpoint**: All error cases handled gracefully with informative messages.

---

## Phase 5: User Story 3 - Custom Configuration Paths (Priority: P3)

**Goal**: Flexibility for different lab setups and local testing with custom file paths.

**Independent Test**: Provide custom paths and verify they are used correctly.

### Implementation for User Story 3

- [x] T026 [US3] Ensure playlist_file_path parameter is respected when provided in add_section_time() in src/hdmea/io/section_time.py
- [x] T027 [US3] Ensure movie_length_file_path parameter is respected when provided in add_section_time() in src/hdmea/io/section_time.py
- [x] T028 [US3] Ensure Path objects and strings are both handled correctly in src/hdmea/io/section_time.py
- [x] T029 [US3] Write unit tests for custom path scenarios in tests/unit/test_section_time.py

**Checkpoint**: Custom configuration paths work correctly.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Integration, documentation, and final touches

- [x] T030 [P] Export add_section_time from hdmea.pipeline in src/hdmea/pipeline/__init__.py
- [x] T031 [P] Update docs/pipeline_explained.md with section time loading flow
- [x] T032 [P] Add entry to docs/pipeline_log.md for section time feature
- [ ] T033 Run all tests and verify no regressions
- [ ] T034 Validate quickstart.md examples work correctly

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup ──────────────► Phase 2: Foundational ──────────────┐
                                                                    │
                               ┌────────────────────────────────────┤
                               │                                    │
                               ▼                                    ▼
                   Phase 3: US1 (P1) MVP            Phase 4: US2 (P2)
                               │                           │
                               ▼                           ▼
                   Phase 5: US3 (P3) ◄─────────────────────┘
                               │
                               ▼
                   Phase 6: Polish
```

### User Story Dependencies

- **User Story 1 (P1)**: Requires Phase 2 completion. Core implementation - no story dependencies.
- **User Story 2 (P2)**: Can start after Phase 2. Adds error handling to US1 code.
- **User Story 3 (P3)**: Can start after Phase 2. Validates path handling already in US1.

### Within Each User Story

1. Tests (fixtures) first
2. Core implementation
3. Edge cases and validation
4. Story-specific tests

### Parallel Opportunities

**Phase 2 (Foundational)**:
```
T004 [P] _load_playlist_csv
T005 [P] _load_movie_length_csv
```

**Phase 3 (US1) - Fixtures**:
```
T007 [P] CSV fixtures
T008 [P] Zarr fixture
```

**Phase 6 (Polish)**:
```
T030 [P] Pipeline export
T031 [P] pipeline_explained.md
T032 [P] pipeline_log.md
```

---

## Parallel Example: Foundational Phase

```bash
# Launch CSV loaders in parallel (different functions, no dependencies):
Task T004: "Implement _load_playlist_csv() in src/hdmea/io/section_time.py"
Task T005: "Implement _load_movie_length_csv() in src/hdmea/io/section_time.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T006)
3. Complete Phase 3: User Story 1 (T007-T017)
4. **STOP and VALIDATE**: Test with real recording
5. Can be used immediately for research

### Incremental Delivery

1. **Setup + Foundational** → Core helpers ready
2. **Add User Story 1** → MVP! Basic section time loading works
3. **Add User Story 2** → Robust error handling
4. **Add User Story 3** → Flexible configuration
5. **Polish** → Documentation and integration complete

### Single Developer Flow

Execute tasks sequentially by phase. Within each phase, parallelize where marked [P].

---

## Notes

- [P] tasks = different files or functions, no dependencies
- [US1/US2/US3] maps task to specific user story
- Each user story is independently completable and testable
- Commit after each phase completion
- US1 is the MVP - stop there if time-constrained
- All error handling (US2) can be added incrementally
- Custom paths (US3) are likely already working from US1 - just needs validation

---

## Implementation Status

**Completed**: 32/34 tasks (94%)

| Phase | Status |
|-------|--------|
| Phase 1: Setup | ✅ Complete (2/2) |
| Phase 2: Foundational | ✅ Complete (4/4) |
| Phase 3: US1 MVP | ✅ Complete (11/11) |
| Phase 4: US2 Error Handling | ✅ Complete (8/8) |
| Phase 5: US3 Custom Paths | ✅ Complete (4/4) |
| Phase 6: Polish | 🔄 In Progress (3/5) |

**Remaining**: T033 (run tests), T034 (validate quickstart)
