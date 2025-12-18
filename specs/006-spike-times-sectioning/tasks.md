# Tasks: Spike Times Unit Conversion and Stimulation Sectioning

**Input**: Design documents from `/specs/006-spike-times-sectioning/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/api.md ✅

**Tests**: Included for verification (minimal unit tests for core logic).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Package**: `src/hdmea/` (existing Python package)
- **Tests**: `tests/unit/`, `tests/integration/`
- **Test Data**: `artifacts/JIANG009_2025-04-10.zarr`

---

## Phase 1: Setup

**Purpose**: Verify prerequisites and create module structure

- [x] T001 Verify test dataset exists at `artifacts/JIANG009_2025-04-10.zarr`
- [x] T002 [P] Create new module file `src/hdmea/io/spike_sectioning.py` with module docstring and imports
- [x] T003 [P] Add `SectionResult` dataclass to `src/hdmea/io/spike_sectioning.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before user story implementation

**⚠️ CRITICAL**: User Story 2 depends on User Story 1 conversion being complete for full integration, but can be developed and tested with pre-converted data.

- [x] T004 Add `section_spike_times` to `src/hdmea/io/__init__.py` exports
- [x] T005 Review existing `load_recording()` in `src/hdmea/pipeline/runner.py` to identify conversion insertion point

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Load Spike Times in Sample Units (Priority: P1) 🎯 MVP

**Goal**: Convert raw spike timestamps from nanoseconds to acquisition sample indices during CMTR data loading

**Independent Test**: Load a CMTR file, verify stored spike_times are in sample units (ns × acquisition_rate / 10^9)

### Implementation for User Story 1

- [x] T006 [US1] Create helper function `_convert_spike_times_to_samples()` in `src/hdmea/pipeline/runner.py`
  - Input: spike_times array (ns), acquisition_rate (Hz)
  - Output: spike_times array (sample indices as uint64)
  - Formula: `np.round(spike_times_ns * acquisition_rate / 1e9).astype(np.uint64)`

- [x] T007 [US1] Modify `load_recording()` in `src/hdmea/pipeline/runner.py` to call conversion before `write_units()`
  - Add conversion step after loading CMTR data
  - Pass acquisition_rate from metadata
  - Convert each unit's spike_times before storage

- [x] T008 [US1] Add unit attribute `spike_times_unit="sample_index"` when writing to Zarr in `src/hdmea/io/zarr_store.py` (optional metadata)

- [x] T009 [US1] Verify conversion with test dataset `artifacts/JIANG009_2025-04-10.zarr`
  - Load with `load_recording(force=True)` to regenerate
  - Check spike_times values are ~10^6-10^9 (sample indices) not ~10^12 (nanoseconds)

**Checkpoint**: ✅ User Story 1 complete - spike_times now stored in sample units

---

## Phase 4: User Story 2 - Section Spike Times by Stimulation (Priority: P2)

**Goal**: Extract spike timestamps within each stimulation trial and store BOTH combined and per-trial formats

**Independent Test**: Given a zarr with spike_times (in sample units) and section_time data, verify sectioned data is created under `spike_times_sectioned/{movie}/`

**Dependencies**: Requires section_time data from spec 005; can be tested with pre-existing zarr

### Implementation for User Story 2

- [x] T010 [US2] Implement `_section_unit_spikes()` helper in `src/hdmea/io/spike_sectioning.py`
  - Input: spike_times (array), section_time (Nx2 array), trial_repeats, pre_samples, post_samples
  - Output: Tuple of (full_spike_times array, trials_spike_times dict)
  - Logic: For each trial, extract spikes within [start-pre_samples, end+post_samples], clamp boundaries

- [x] T011 [US2] Implement `_write_sectioned_spikes()` helper in `src/hdmea/io/spike_sectioning.py`
  - Input: unit_group (zarr.Group), movie_name, full_spike_times, trials_spike_times, metadata
  - Creates: `spike_times_sectioned/{movie}/full_spike_times` array
  - Creates: `spike_times_sectioned/{movie}/trials_spike_times/{idx}` arrays
  - Writes metadata attributes (n_trials, trial_repeats, pad_margin, pre_samples, post_samples)
  - Handles force=True/False for overwrite protection

- [x] T012 [US2] Implement main `section_spike_times()` function in `src/hdmea/io/spike_sectioning.py`
  - Parameters: zarr_path, movie_names (optional), trial_repeats=3, pad_margin=(2.0, 0.0), force=False
  - Read acquisition_rate from zarr metadata
  - Convert pad_margin to samples: pre_samples, post_samples
  - Iterate all movies in section_time (or specified movie_names)
  - Iterate all units, call _section_unit_spikes(), _write_sectioned_spikes()
  - Return SectionResult with success, units_processed, movies_processed, etc.

- [x] T013 [US2] Add logging for section_spike_times() progress in `src/hdmea/io/spike_sectioning.py`
  - Log: movies found, units processed, trials per movie
  - Use logger.info() for progress, logger.warning() for issues

- [x] T014 [US2] Verify sectioning with test dataset `artifacts/JIANG009_2025-04-10.zarr`
  - Ensure section_time exists (from spec 005)
  - Run section_spike_times() with default parameters
  - Verify full_spike_times and trials_spike_times created for each unit/movie

**Checkpoint**: ✅ User Story 2 complete - spike_times_sectioned populated

---

## Phase 5: User Story 3 - Handle Missing Data Gracefully (Priority: P3)

**Goal**: Robust handling of edge cases (missing section_time, empty spike_times, overwrite protection)

**Independent Test**: Provide zarr with missing section_time or empty units, verify warnings and graceful handling

### Implementation for User Story 3

- [x] T015 [US3] Add missing section_time handling in `section_spike_times()` in `src/hdmea/io/spike_sectioning.py`
  - If no section_time group exists: log warning, return early with success=True
  - If specific movie not in section_time: log warning, skip movie

- [x] T016 [US3] Add empty spike_times handling in `_section_unit_spikes()` in `src/hdmea/io/spike_sectioning.py`
  - If spike_times is empty: return empty arrays for full and per-trial
  - Store empty arrays (shape=(0,), dtype=int64) not omitted

- [x] T017 [US3] Add FileExistsError handling in `_write_sectioned_spikes()` in `src/hdmea/io/spike_sectioning.py`
  - If spike_times_sectioned/{movie} exists and force=False: raise FileExistsError
  - If force=True: delete existing group before writing

- [x] T018 [US3] Add boundary clamping in `_section_unit_spikes()` in `src/hdmea/io/spike_sectioning.py`
  - Clamp padded_start to max(0, trial_start - pre_samples)
  - Clamp padded_end to min(max_spike_time, trial_end + post_samples) if desired

- [x] T019 [US3] Add warnings collection in SectionResult for `section_spike_times()`
  - Collect warnings for: movies with no trials, units with no spikes, etc.
  - Return in SectionResult.warnings list

**Checkpoint**: ✅ User Story 3 complete - all edge cases handled gracefully

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and finalization

- [x] T020 [P] Update `docs/pipeline_explained.md` with new spike_times conversion step
- [x] T021 [P] Append entry to `docs/pipeline_log.md` for this feature
- [x] T022 Run `quickstart.md` validation - execute all code examples against test zarr
- [x] T023 [P] Add type hints and docstrings review for all new/modified functions
- [x] T024 Code cleanup - remove any debug prints, ensure logger usage

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion
- **User Story 1 (Phase 3)**: Depends on Foundational - implements spike_times conversion
- **User Story 2 (Phase 4)**: Depends on Foundational - can develop with pre-converted data, full integration after US1
- **User Story 3 (Phase 5)**: Depends on US1 and US2 implementation - adds error handling
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

| Story | Depends On | Can Start After |
|-------|------------|-----------------|
| US1 (P1) | Foundational | Phase 2 complete |
| US2 (P2) | Foundational | Phase 2 complete (can develop; full test after US1) |
| US3 (P3) | US1, US2 core | US1 and US2 core implementation |

### Within Each User Story

- Core logic before I/O operations
- Helpers before main functions
- Implementation before verification

### Parallel Opportunities

- T002, T003 can run in parallel (different files/sections)
- T020, T021, T023 can run in parallel (different docs/files)
- US1 and US2 core implementation can proceed in parallel (US2 uses test data with pre-converted spike_times)

---

## Parallel Example: Phase 1 Setup

```bash
# Launch setup tasks in parallel:
Task T002: "Create spike_sectioning.py module"
Task T003: "Add SectionResult dataclass"
```

## Parallel Example: User Story 2 Helpers

```bash
# Launch helper implementations in parallel:
Task T010: "_section_unit_spikes() helper"
Task T011: "_write_sectioned_spikes() helper"
# Then T012 depends on both
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (spike times conversion)
4. **STOP and VALIDATE**: Verify spike_times are in sample units
5. Deploy/demo if ready - data now in unified coordinate system

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 → Test → **MVP!** (spike_times in sample units)
3. Add User Story 2 → Test → **Value add** (sectioned spike_times available)
4. Add User Story 3 → Test → **Robust** (handles all edge cases)
5. Polish → Documentation complete

### Single Developer Strategy (Recommended)

Execute in order:
1. Phase 1 (T001-T003)
2. Phase 2 (T004-T005)
3. Phase 3 - User Story 1 (T006-T009) → Verify before continuing
4. Phase 4 - User Story 2 (T010-T014) → Verify before continuing
5. Phase 5 - User Story 3 (T015-T019)
6. Phase 6 - Polish (T020-T024)

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 24 |
| Phase 1 (Setup) | 3 tasks |
| Phase 2 (Foundational) | 2 tasks |
| Phase 3 (US1 - Conversion) | 4 tasks |
| Phase 4 (US2 - Sectioning) | 5 tasks |
| Phase 5 (US3 - Error Handling) | 5 tasks |
| Phase 6 (Polish) | 5 tasks |
| Parallel Opportunities | 7 task groups |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Test data: `artifacts/JIANG009_2025-04-10.zarr`
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
