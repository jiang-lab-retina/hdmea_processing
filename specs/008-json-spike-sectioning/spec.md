# Feature Specification: JSON-Based Spike Sectioning

**Feature Branch**: `008-json-spike-sectioning`  
**Created**: 2024-12-18  
**Status**: Draft  
**Input**: User description: "Edit _section_unit_spikes function and other functions to use the JSON files in config/stimuli to do section. The JSON file section_kwargs has the frame number for start frame and trial_length_frame and repeat. Match the movie name to file name to find the correct kwargs. The frame number need to be converted to sample index for the correct shifting. The JSON file is with reference to the start of the movie as section_frame_start + PRE_MARGIN_FRAME_NUM. Get the sections by starting from start_frame in JSON file and keep adding trial_length_frame for several times (repeat in JSON)"

## Clarifications

### Session 2024-12-18

- Q: Missing JSON Config Fallback Behavior - should system skip unconfigured movies, use existing section_time as fallback, or require all movies have JSON config? → A: Require all movies have JSON config (fail if any missing)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Section Spike Times Using JSON Configuration (Priority: P1)

A researcher wants to section spike times into trials for a specific stimulus movie. Instead of relying on hardcoded trial boundaries, the system loads the trial parameters (start_frame, trial_length_frame, repeat) from the corresponding JSON configuration file in `config/stimuli/`. The movie name from the HDF5 section_time data is matched to the JSON filename to retrieve the correct sectioning parameters.

**Why this priority**: This is the core functionality required. Without the ability to load and use JSON configuration for sectioning, no other features can work.

**Independent Test**: Can be tested by sectioning spike times for a recording that includes a movie matching a JSON config file (e.g., `moving_h_bar_s5_d8_3x.json`), and verifying that trials are extracted at correct boundaries.

**Acceptance Scenarios**:

1. **Given** an HDF5 file with spike_times and section_time for movie "moving_h_bar_s5_d8_3x", **When** section_spike_times is called, **Then** the system loads `config/stimuli/moving_h_bar_s5_d8_3x.json` and uses `start_frame=60`, `trial_length_frame=4520`, `repeat=3` to create 3 trial sections.

2. **Given** section_kwargs with start_frame=60 and PRE_MARGIN_FRAME_NUM=60, **When** calculating trial boundaries, **Then** the first trial starts at frame `section_frame_start + 60` (accounting for movie start offset) converted to sample index.

3. **Given** JSON config with repeat=3 and trial_length_frame=600, **When** sectioning is performed, **Then** 3 trials are created with boundaries at: [start, start+600), [start+600, start+1200), [start+1200, start+1800).

---

### User Story 2 - Frame-to-Sample Index Conversion (Priority: P1)

Frame numbers specified in JSON config files must be accurately converted to acquisition sample indices using the frame_timestamps array. This enables precise alignment of spike times with stimulus boundaries.

**Why this priority**: Correct frame-to-sample conversion is essential for accurate trial segmentation. Incorrect conversion would produce meaningless results.

**Independent Test**: Can be tested by verifying that frame numbers from JSON config are converted to matching sample indices via frame_timestamps lookup.

**Acceptance Scenarios**:

1. **Given** start_frame=60 from JSON and frame_timestamps array, **When** converting to sample index, **Then** the result equals `frame_timestamps[section_frame_start + 60]` where section_frame_start is the movie start frame from section_time.

2. **Given** trial_length_frame=600, **When** calculating trial end boundaries, **Then** each trial boundary is computed as frame number then converted to sample index (not adding frames to sample indices directly).

---

### User Story 3 - Strict JSON Config Requirement (Priority: P2)

All movies in the HDF5 file must have a corresponding JSON configuration file in `config/stimuli/`. If any movie lacks a JSON config, the sectioning operation fails with a clear error message listing the missing configurations.

**Why this priority**: Ensures data integrity and explicit configuration for all stimuli. Prevents silent failures or inconsistent processing methods across movies.

**Independent Test**: Can be tested by attempting to process an HDF5 file containing a movie without a corresponding JSON config and verifying the operation fails with informative error.

**Acceptance Scenarios**:

1. **Given** movie "unknown_stimulus" with no JSON config file, **When** sectioning is attempted, **Then** the system raises an error indicating missing config file for "unknown_stimulus".

2. **Given** a recording with movies where one lacks JSON config, **When** sectioning is performed, **Then** the operation fails before processing any movies, listing all missing config files.

---

### User Story 4 - Support Multiple Stimulus Types (Priority: P2)

The system should handle various stimulus types with different sectioning parameters. Each JSON file defines its own start_frame, trial_length_frame, and repeat values appropriate for that stimulus.

**Why this priority**: Enables flexible use across different experimental paradigms but builds on the core P1 functionality.

**Independent Test**: Can be tested by sectioning a recording containing multiple stimulus types and verifying each uses its specific JSON parameters.

**Acceptance Scenarios**:

1. **Given** movies "step_up_5s_5i_3x" (start_frame=240, trial_length_frame=600, repeat=3) and "perfect_dense_noise_15x15_15hz_r42_3min" (start_frame=0, trial_length_frame=10800, repeat=1), **When** sectioning both, **Then** each movie is sectioned according to its own JSON parameters.

---

### Edge Cases

- What happens when JSON config exists but has missing or malformed `section_kwargs`? → Fail with validation error specifying the issue.
- How does system handle when `repeat` in JSON exceeds the recording duration? → Process available trials, warn if truncated.
- What happens when `start_frame` from JSON plus section_frame_start exceeds total frames? → Fail with error indicating invalid frame range.
- How does system handle dense noise with repeat=1 (single long section instead of multiple trials)? → Creates single trial of full trial_length_frame duration.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load stimulus-specific sectioning parameters from JSON files in `config/stimuli/` directory.
- **FR-002**: System MUST match movie names from HDF5 section_time to JSON filenames (e.g., movie "moving_h_bar_s5_d8_3x" matches `moving_h_bar_s5_d8_3x.json`).
- **FR-003**: System MUST extract `start_frame`, `trial_length_frame`, and `repeat` from the JSON file's `section_kwargs` object.
- **FR-004**: System MUST convert JSON frame numbers to sample indices using `frame_timestamps[frame_number]` lookup.
- **FR-005**: System MUST calculate trial boundaries relative to the movie start: first trial starts at `section_frame_start + PRE_MARGIN_FRAME_NUM + start_frame` (converted to sample index).
- **FR-006**: System MUST create `repeat` number of trials, each with length `trial_length_frame` (converted to samples).
- **FR-007**: System MUST support the existing JSON config format with `section_kwargs` containing `start_frame`, `trial_length_frame`, and `repeat` fields.
- **FR-008**: System MUST fail with an error listing all missing JSON config files when any movie in section_time lacks a corresponding config file.
- **FR-009**: System MUST preserve existing function signatures and HDF5 output format for backward compatibility.

### Key Entities

- **JSON Config File**: Located in `config/stimuli/{movie_name}.json`. Contains `section_kwargs` with `start_frame` (int), `trial_length_frame` (int), `repeat` (int).
- **section_time**: HDF5 dataset at `stimulus/section_time/{movie_name}` containing shape (N, 2) array with [start_sample, end_sample] for each movie occurrence.
- **frame_timestamps**: HDF5 dataset at `metadata/frame_timestamps` mapping display frame indices to acquisition sample indices.
- **PRE_MARGIN_FRAME_NUM**: Constant (60 frames) representing the margin before movie content begins.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All stimulus movies with corresponding JSON config files are sectioned using parameters from those files.
- **SC-002**: Trial boundaries calculated from JSON parameters match expected frame-to-sample conversions within 1 sample tolerance.
- **SC-003**: Sectioning produces the correct number of trials as specified by `repeat` in JSON config.
- **SC-004**: Each trial has duration matching `trial_length_frame` converted to samples (±1 sample tolerance for frame alignment).
- **SC-005**: Existing HDF5 output structure (`spike_times_sectioned/{movie_name}/full_spike_times` and `trials_spike_times/`) is preserved.
- **SC-006**: Missing JSON config files are detected upfront with clear error messages listing all missing files.

## Assumptions

- JSON config files in `config/stimuli/` follow the established format with `section_kwargs` object containing `start_frame`, `trial_length_frame`, and `repeat`.
- Movie names in HDF5 `section_time` group exactly match JSON filenames (without `.json` extension).
- `frame_timestamps` array is always available in HDF5 `metadata` group.
- `PRE_MARGIN_FRAME_NUM` constant (60) applies to all movies using this sectioning approach.
- JSON config `start_frame` is relative to the movie content start (after PRE_MARGIN padding).
