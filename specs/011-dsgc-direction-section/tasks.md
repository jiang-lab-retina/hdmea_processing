# Tasks: DSGC Direction Sectioning

**Input**: Design documents from `/specs/011-dsgc-direction-section/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/api.md ✅

**Tests**: Not explicitly requested in spec. Minimal validation included in Polish phase.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Package**: `src/hdmea/features/` (new module: `dsgc_direction.py`)
- **Tests**: `tests/unit/` (if added)
- **Development**: `Projects/dsgc_section/` (test scripts, export folder)

---

## Phase 1: Setup

**Purpose**: Create module structure and export configuration

- [x] T001 Create module file `src/hdmea/features/dsgc_direction.py` with docstring and imports
- [x] T002 Add `section_by_direction` to `src/hdmea/features/__init__.py` exports
- [x] T003 [P] Create export folder `Projects/dsgc_section/export/` (gitignored)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Constants, data classes, and helper functions needed by all user stories

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Define constants in `src/hdmea/features/dsgc_direction.py`:
  - `DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]`
  - `N_REPETITIONS = 3`
  - `N_TRIALS = 24`
  - `DEFAULT_PADDING_FRAMES = 10`
  - `COORDINATE_SCALE_FACTOR = 20`
  - `DEFAULT_STIMULI_DIR`
  - `STA_GEOMETRY_FEATURE = "sta_perfect_dense_noise_15x15_15hz_r42_3min"`

- [x] T005 [P] Create `DirectionSectionResult` dataclass in `src/hdmea/features/dsgc_direction.py`:
  - Fields: hdf5_path, movie_name, units_processed, units_skipped, padding_frames, elapsed_seconds, warnings, skipped_units

- [x] T006 [P] Implement `load_on_off_dict()` helper in `src/hdmea/features/dsgc_direction.py`:
  - Load pickle file from path
  - Validate structure (check for required keys)
  - Return Dict[(row,col), Dict]
  - Raise FileNotFoundError with clear message if not found

- [x] T007 [P] [US2] Implement `convert_center_15_to_300()` helper in `src/hdmea/features/dsgc_direction.py`:
  - Multiply by COORDINATE_SCALE_FACTOR (20)
  - Cast to int
  - Clip to [0, 299] range
  - Return (row, col) tuple

- [x] T008 [P] [US2] Implement `get_cell_center()` helper in `src/hdmea/features/dsgc_direction.py`:
  - Read from HDF5: `units/{unit_id}/features/{sta_feature}/sta_geometry/center_row` and `center_col`
  - Call `convert_center_15_to_300()`
  - Return Optional[Tuple[int, int]] (None if not found)
  - Log warning if clipping occurs

- [x] T009 Configure logging: `logger = logging.getLogger(__name__)` in `src/hdmea/features/dsgc_direction.py`

**Checkpoint**: Foundation ready - core implementation can now begin

---

## Phase 3: User Story 1 + 4 - Core Direction Sectioning (Priority: P1) 🎯 MVP

**Goal**: Section spike times by direction at cell center and save results without modifying source data

**Independent Test**: Run on test file, verify 8 direction groups × 3 trials created, original data unchanged

**Combines**:
- US1: Section Spikes by Direction at Cell Center
- US4: Save Results Without Overwriting Source

### Implementation for User Story 1 + 4

- [x] T010 [US1] Implement `section_unit_by_direction()` in `src/hdmea/features/dsgc_direction.py`:
  - Input: spike_frames (movie-relative), cell_center, on_off_dict, padding_frames
  - For each trial (0-23): extract spikes in [on - padding, off + padding] window
  - Map trial index to direction and repetition
  - Return Dict[direction, {trials: [...], bounds: [...]}]

- [x] T011 [US1] Implement spike-to-frame conversion logic in `section_by_direction()`:
  - Read frame_timestamps from `metadata/frame_timestamps`
  - Read section_time for movie
  - Compute movie_start_frame = convert_sample_to_frame(section_time[0,0]) + PRE_MARGIN_FRAME_NUM
  - Import `convert_sample_index_to_frame`, `PRE_MARGIN_FRAME_NUM` from `hdmea.io.section_time`

- [x] T012 [US1] Implement main loop in `section_by_direction()`:
  - Get list of units (or filter by unit_ids if provided)
  - For each unit: get cell center, read full_spike_times, convert to frames, call section_unit_by_direction
  - Track units_processed, units_skipped, warnings

- [x] T013 [US4] Implement output_path copy logic in `section_by_direction()`:
  - If output_path provided: copy source HDF5 to output_path using shutil.copy2
  - Open the target file (output_path or hdf5_path) for writing

- [x] T014 [US4] Implement force parameter handling in `section_by_direction()`:
  - Check if `direction_section` group exists
  - If exists and force=False: skip unit, log info
  - If exists and force=True: delete existing group first

- [x] T015 [US4] Implement HDF5 write logic in `section_by_direction()`:
  - Create `direction_section/{direction}/trials/{rep}` groups using require_group()
  - Save spike arrays as int64 datasets
  - Save section_bounds as int64[3,2] dataset
  - Add attributes: direction_list, n_directions, n_repetitions, padding_frames, cell_center_row, cell_center_col

- [x] T016 [US1] Implement unit_ids parameter filtering in `section_by_direction()`:
  - If unit_ids is None: process all units
  - If unit_ids provided: filter to only those units
  - Validate each specified unit exists and has required data

- [x] T017 [US1] Add error handling and edge cases:
  - Skip units missing STA geometry (log warning)
  - Skip units missing full_spike_times (log warning)
  - Handle empty spike arrays (save empty dataset)
  - Clip cell center if out of bounds (log warning)

- [x] T018 [US1] Return DirectionSectionResult with statistics:
  - Count units_processed, units_skipped
  - Collect warnings and skipped_units lists
  - Calculate elapsed_seconds

**Checkpoint**: Core sectioning functional - run on test file to verify

---

## Phase 4: User Story 3 - Configurable Padding (Priority: P2)

**Goal**: Apply configurable padding to trial windows

**Independent Test**: Compare spike counts with padding=0 vs padding=10

**Note**: Padding is already integrated in Phase 3 (T010). This phase adds validation and documentation.

### Implementation for User Story 3

- [x] T019 [US3] Add padding parameter validation in `section_by_direction()`:
  - Validate padding_frames >= 0
  - Log info message with padding value being used

- [x] T020 [US3] Document padding behavior in function docstring:
  - Explain start = on_time - padding, end = off_time + padding
  - Note that padding is applied symmetrically

**Checkpoint**: Padding configurable and validated

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Finalize, test, and document

- [x] T021 [P] Create development/test script `Projects/dsgc_section/dsgc_section.py`:
  - Copy test HDF5 to export folder
  - Run section_by_direction on copy
  - Print result statistics
  - Verify output structure

- [x] T022 [P] Add type hints to all public functions in `src/hdmea/features/dsgc_direction.py`

- [x] T023 [P] Add complete docstrings (Google style) to all public functions

- [x] T024 Run quickstart.md validation:
  - Execute all code examples in `specs/011-dsgc-direction-section/quickstart.md`
  - Verify they work as documented

- [x] T025 Update `docs/pipeline_log.md` with new feature entry:
  - Date, description, affected modules

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup ─────────────────────┐
                                    ▼
Phase 2: Foundational ──────────────┤
                                    ▼
Phase 3: US1+US4 (Core) ────────────┤ (MVP complete here)
                                    ▼
Phase 4: US3 (Padding) ─────────────┤
                                    ▼
Phase 5: Polish ────────────────────┘
```

### User Story Dependencies

- **User Story 2 (Center Conversion)**: Implemented in Foundational phase (T007, T008) - prerequisite for US1
- **User Story 1 (Core Sectioning)**: Depends on US2 - implemented in Phase 3
- **User Story 4 (Save Without Overwrite)**: Integrated with US1 in Phase 3
- **User Story 3 (Padding)**: Enhancement in Phase 4 - core padding logic already in T010

### Within Each Phase

- Constants before data classes
- Helper functions before main function
- Core logic before error handling
- Implementation before documentation

### Parallel Opportunities

**Phase 2** (after T004):
```
T005, T006, T007, T008 can run in parallel (different functions)
```

**Phase 5**:
```
T021, T022, T023 can run in parallel (different concerns)
```

---

## Parallel Example: Phase 2 (Foundational)

```bash
# After T004 (constants) completes, launch these in parallel:
Task: T005 "Create DirectionSectionResult dataclass"
Task: T006 "Implement load_on_off_dict() helper"
Task: T007 "Implement convert_center_15_to_300() helper"
Task: T008 "Implement get_cell_center() helper"
```

---

## Implementation Strategy

### MVP First (Phase 1-3)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T009)
3. Complete Phase 3: Core Sectioning (T010-T018)
4. **STOP and VALIDATE**: Run on test file
5. If works: MVP complete, can be used immediately

### Incremental Delivery

1. MVP complete → Core sectioning works with default padding
2. Add Phase 4 → Padding validation/documentation
3. Add Phase 5 → Polish, tests, documentation

### Single Developer Strategy

Execute tasks in order: T001 → T002 → ... → T025

Total estimated time: 4-6 hours

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tasks** | 25 |
| **Phase 1 (Setup)** | 3 tasks |
| **Phase 2 (Foundational)** | 6 tasks |
| **Phase 3 (US1+US4 Core)** | 9 tasks |
| **Phase 4 (US3 Padding)** | 2 tasks |
| **Phase 5 (Polish)** | 5 tasks |
| **Parallel Opportunities** | 8 tasks marked [P] |
| **MVP Scope** | T001-T018 (17 tasks) |

---

## Notes

- [P] tasks = different files or functions, no dependencies
- [Story] label maps task to specific user story for traceability
- Test file: `M:\Python_Project\Data_Processing_2027\Projects\ap_trace_hdf5\export_ap_tracking_20251226\2024.08.08-10.40.20-Rec.h5`
- Export folder: `M:\Python_Project\Data_Processing_2027\Projects\dsgc_section\export\`
- Always copy test file before modifying (output_path parameter)

