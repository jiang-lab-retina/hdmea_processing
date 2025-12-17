# Tasks: Analog Section Time Detection

**Input**: Design documents from `/specs/005-analog-section-time/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/api.md, quickstart.md, research.md

**Tests**: Test updates included as modifications affect existing test expectations.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

**Test Data**: Use `artifacts/JIANG009_2025-04-10.zarr` for integration/acceptance testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No new project setup needed - modifications to existing package.

*No tasks - using existing `src/hdmea/io/section_time.py` and `tests/unit/test_section_time.py`*

---

## Phase 2: Foundational (Unit Unification)

**Purpose**: Core infrastructure change that MUST be complete before user story tasks.

**CRITICAL**: These tasks unify section_time units across both functions.

- [x] T001 Modify `add_section_time()` to output acquisition sample indices instead of display frame indices in `src/hdmea/io/section_time.py`. After computing `movie_start_end_frame` as frame indices, convert to samples via `start_sample = frame_timestamps[start_frame]`, `end_sample = frame_timestamps[end_frame]` before storing.

- [x] T002 Modify `add_section_time_analog()` to remove frame_timestamps dependency and store sample indices directly in `src/hdmea/io/section_time.py`. Remove lines that: (a) validate frame_timestamps existence, (b) filter peaks by frame_timestamps range, (c) convert onset_samples to display frames via `_sample_to_nearest_frame()`. Peak detection should use `raw_ch1` directly (fix reference to undefined `signal_1khz` at line 577), compute end_sample as `onset_sample + int(plot_duration * acquisition_rate)`, and store [start_sample, end_sample] pairs directly.

**Checkpoint**: Both functions now output acquisition sample indices.

---

## Phase 3: User Story 1 - Detect Stimulus Onsets from Light Signal (Priority: P1) MVP

**Goal**: Automatically detect stimulus onset times from `raw_ch1` analog signal and store as section times in acquisition sample indices.

**Independent Test**: Load `artifacts/JIANG009_2025-04-10.zarr`, run detection with appropriate threshold, verify detected times are stored as sample indices under `stimulus/section_time/{movie_name}`.

### Implementation for User Story 1

- [x] T003 [US1] Fix `_detect_analog_peaks()` usage in `add_section_time_analog()` in `src/hdmea/io/section_time.py`. Change line 577 from `_detect_analog_peaks(signal_1khz, threshold_value)` to `_detect_analog_peaks(raw_ch1, threshold_value)`.

- [x] T004 [US1] Remove frame_timestamps validation code block from `add_section_time_analog()` in `src/hdmea/io/section_time.py`. Delete the check for `metadata/frame_timestamps` and remove any loading of frame_timestamps variable.

- [x] T005 [US1] Update peak-to-section conversion in `add_section_time_analog()` in `src/hdmea/io/section_time.py`. Replace frame conversion logic with direct sample index storage: peaks from `_detect_analog_peaks(raw_ch1, threshold)` are already in acquisition sample indices, calculate `end_samples = onset_samples + int(plot_duration * acquisition_rate)`, clip end_samples to `len(raw_ch1) - 1`, build section_time as `np.column_stack([onset_samples, end_samples]).astype(np.int64)`, remove all frame conversion and frame_timestamps range filtering logic.

- [x] T006 [US1] Update test `test_basic_detection` in `tests/unit/test_section_time.py` to verify section_time values are in acquisition sample indices (not frame indices). Expected first trial start should be approximately `10 * 20000 = 200000` samples.

- [x] T007 [US1] Remove obsolete tests in `tests/unit/test_section_time.py`: delete `test_raises_on_missing_frame_timestamps` and `test_returns_false_peaks_beyond_frame_range`.

**Checkpoint**: `add_section_time_analog()` detects peaks in raw_ch1, stores acquisition sample indices.

---

## Phase 4: User Story 2 - Customize Detection Parameters (Priority: P2)

**Goal**: Allow users to adjust detection sensitivity via threshold_value and control section duration via plot_duration.

### Implementation for User Story 2

- [x] T008 [US2] Verify threshold_value validation in `add_section_time_analog()` in `src/hdmea/io/section_time.py`. Confirm ValueError is raised if threshold_value is None.

- [x] T009 [US2] Update `test_plot_duration_affects_section_length` in `tests/unit/test_section_time.py` to verify section spans are in sample units. Expected: `short_duration=1.0` gives 20000 samples, `long_duration=10.0` gives 200000 samples.

- [x] T010 [US2] Update `test_threshold_sensitivity` in `tests/unit/test_section_time.py` to verify higher threshold detects fewer peaks and output is in sample indices.

---

## Phase 5: User Story 3 - Handle Edge Cases Gracefully (Priority: P3)

**Goal**: Provide clear error messages when required data is missing or no stimuli are detected.

### Implementation for User Story 3

- [x] T011 [US3] Verify MissingInputError message for missing raw_ch1 in `src/hdmea/io/section_time.py`.

- [x] T012 [US3] Verify MissingInputError message for missing acquisition_rate in `src/hdmea/io/section_time.py`.

- [x] T013 [US3] Update `test_returns_false_no_peaks` in `tests/unit/test_section_time.py` to not require frame_timestamps.

- [x] T014 [US3] Update analog_zarr fixture in `tests/unit/test_section_time.py` to remove frame_timestamps creation.

- [x] T015 [US3] Add test `test_section_clipped_at_signal_boundary` in `tests/unit/test_section_time.py`.

---

## Phase 6: Polish

- [x] T016 [P] Update docstring for `add_section_time()` in `src/hdmea/io/section_time.py` to document output is in acquisition sample indices.

- [x] T017 [P] Update `docs/pipeline_log.md` to document the unit change.

- [x] T018 Run acceptance test with `artifacts/JIANG009_2025-04-10.zarr`.

---

## Implementation Strategy

### MVP First (Phase 2 + User Story 1)

1. Complete T001, T002 (Foundational)
2. Complete T003-T007 (US1)
3. Test with `artifacts/JIANG009_2025-04-10.zarr`

### Files Modified

| File | Tasks |
|------|-------|
| `src/hdmea/io/section_time.py` | T001-T005, T008, T011-T012, T016 |
| `tests/unit/test_section_time.py` | T006-T007, T009-T010, T013-T015 |
| `docs/pipeline_log.md` | T017 |

---

## Notes

- Test data: `artifacts/JIANG009_2025-04-10.zarr`
- Verify section_time values are ~millions (samples) not ~thousands (frames)
- Current code bug: line 577 references undefined `signal_1khz` - fixed in T003
