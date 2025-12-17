# Feature Specification: Analog Section Time Detection

**Feature Branch**: `005-analog-section-time`  
**Created**: 2025-12-17  
**Status**: Draft  
**Input**: User description: "Implement analog movie loading for automatic section time detection using 10hz light reference signal"

## Clarifications

### Session 2025-12-17

- Q: Should section_time store 10 Hz indices or display frame indices? → A: ~~Convert to display frame indices~~ **REVISED**: Store as acquisition sample indices (frame_timestamps doesn't cover analog sections)
- Q: What is the unit for plot_duration parameter? → A: Seconds (default 120 = 2 minute window per trial)
- Q: How to convert onset time to display frame index? → A: ~~Nearest frame~~ **REVISED**: No conversion needed - store raw sample indices
- Q: What if frame_timestamps is missing? → A: ~~Error out~~ **REVISED**: Not required - analog section_time uses acquisition sample indices directly
- Q: Should threshold have a default value? → A: No default - require explicit threshold (forces user inspection)
- Q: What unit for analog section_time storage? → A: Acquisition sample indices (native unit of raw_ch1, no conversion needed)
- Q: How to handle different units between analog and playlist section_time? → A: Unify all section_time to use acquisition sample indices (update playlist section_time to match)
- Q: Scope of unit unification? → A: This spec covers BOTH add_section_time_analog() AND add_section_time() - unified change for consistency
- Q: How to handle existing data with old unit (display frames)? → A: No migration needed - existing data acceptable as-is; new data uses samples

## Overview

This feature enables automatic detection of stimulus presentation times by analyzing the light reference analog signal. Unlike playlist-based section time loading (spec 004), this approach detects actual stimulus onset times from the recorded light sensor signal, making it suitable for experiments where exact timing must be determined post-hoc (e.g., ipRGC stimulation tests).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Detect Stimulus Onsets from Light Signal (Priority: P1)

A researcher has recorded an experiment with light pulse stimuli (e.g., ipRGC test). The exact onset times of each stimulus presentation need to be automatically detected from the analog light reference signal and stored for downstream analysis.

**Why this priority**: This is the core capability - without accurate stimulus detection, no further analysis is possible.

**Independent Test**: Can be fully tested by loading a zarr with known light pulses, running the detection, and verifying detected times match expected pulse positions.

**Acceptance Scenarios**:

1. **Given** a zarr file with `stimulus/light_reference/raw_ch1` signal containing light pulses, **When** the user runs analog section time detection, **Then** the system detects stimulus onset times (as acquisition sample indices) and stores them under `stimulus/section_time/{movie_name}`

2. **Given** a recording with multiple repeated stimuli, **When** detection runs with a specified repeat count, **Then** only the first N repeats are used for subsequent processing (where N = repeat parameter)

3. **Given** detection parameters (threshold, duration), **When** detection runs, **Then** peaks in the signal derivative above threshold are identified as stimulus onsets

---

### User Story 2 - Customize Detection Parameters (Priority: P2)

A researcher needs to adjust detection sensitivity for different experimental setups where signal characteristics vary.

**Why this priority**: Different experiments may have different signal-to-noise ratios or stimulus durations, requiring parameter tuning.

**Independent Test**: Can be tested by running detection with different threshold values and verifying appropriate sensitivity changes.

**Acceptance Scenarios**:

1. **Given** a configurable threshold value, **When** the threshold is increased, **Then** fewer peaks (only stronger signal transitions) are detected

2. **Given** a configurable plot_duration parameter, **When** the user specifies a custom duration, **Then** each section spans the specified duration from onset

---

### User Story 3 - Handle Edge Cases Gracefully (Priority: P3)

The system should handle recordings that may have missing data or no detectable stimuli.

**Why this priority**: Robustness prevents data loss and provides clear feedback when detection fails.

**Independent Test**: Can be tested by providing recordings without valid signals and verifying appropriate error messages.

**Acceptance Scenarios**:

1. **Given** a zarr file without `raw_ch1` signal, **When** detection runs, **Then** a clear error message is returned indicating the missing signal

2. **Given** a signal with no peaks above threshold, **When** detection runs, **Then** the system reports that no stimuli were detected

---

### Edge Cases

- What happens when the signal is flat (no light transitions)? → Return empty section_time with warning
- What happens when `raw_ch1` is missing? → Return clear error indicating signal not found
- What happens when only partial recordings exist? → Detect available peaks and warn if fewer than expected
- What happens when section_time already exists for this movie? → Require force=True to overwrite
- What happens when acquisition_rate is missing? → Return clear error (required for end time calculation)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST detect stimulus onset times by finding peaks in the derivative of the `raw_ch1` light reference signal (at acquisition rate ~20 kHz)
- **FR-002**: System MUST filter peaks using a required threshold_value parameter (no default - user must specify based on signal inspection)
- **FR-003**: System MUST store detected onset times as **acquisition sample indices** (no conversion to display frames - frame_timestamps doesn't cover analog sections)
- **FR-003a**: System MUST calculate section end time as `onset_sample + plot_duration_seconds × acquisition_rate` (default: 120 seconds)
- **FR-004**: System MUST store detected section boundaries as a 2D array under `stimulus/section_time/{movie_name}`
- **FR-005**: System MUST support a configurable movie_name parameter (default: "iprgc_test")
- **FR-006**: System MUST support a repeat parameter to limit the number of detected trials used
- **FR-007**: System MUST NOT extract or store unit response traces (unlike legacy code)
- **FR-008**: System MUST raise an error if `stimulus/light_reference/raw_ch1` does not exist
- **FR-008a**: System MUST require `metadata/acquisition_rate` to compute end times from plot_duration
- **FR-009**: System MUST prevent accidental overwriting of existing section_time data unless force=True

**Playlist-based Section Time (unified unit change):**
- **FR-010**: `add_section_time()` MUST output acquisition sample indices instead of display frame indices
- **FR-011**: `add_section_time()` MUST convert display frames to samples using: `sample_index = frame_timestamps[frame_index]`
- **FR-012**: Both functions MUST produce section_time arrays with the same unit (acquisition sample indices)

### Key Entities

- **Section Time Array**: 2D integer array of shape (N, 2) where N is the number of detected trials, containing [start_sample, end_sample] pairs as **acquisition sample indices** (unified unit for all section_time - requires updating playlist-based section_time to also use this unit)
- **Movie Name**: String identifier for the stimulus type (e.g., "iprgc_test") used as the key under section_time group
- **raw_ch1**: Light reference signal at acquisition rate (~20 kHz) used for peak detection
- **acquisition_rate**: Sampling rate in Hz, required to convert plot_duration (seconds) to sample count

### Changes to Existing Code

This spec includes modifications to ensure unified section_time unit across the pipeline:

- **`add_section_time()` (playlist-based)**: Change output from display frame indices to acquisition sample indices
  - Use `frame_timestamps` array to convert display frames → acquisition samples
  - Formula: `sample_index = frame_timestamps[frame_index]`
  
- **`add_section_time_analog()`**: Remove frame_timestamps conversion, store raw sample indices directly
  - Detect peaks in raw_ch1 (acquisition rate)
  - Store onset samples directly without conversion
  
- **Tests**: Update test expectations to match new unit (acquisition sample indices)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Detected stimulus onsets align with actual light pulse times within ±50 samples (~2.5ms at 20 kHz)
- **SC-002**: Processing completes within 5 seconds for typical recordings
- **SC-003**: All detected section times are accessible via standard zarr API after storage
- **SC-004**: System correctly handles recordings with 0 to 100+ stimulus presentations

## Assumptions

- The raw_ch1 signal has sharp transitions (step changes) at stimulus onset that create detectable peaks in the signal derivative
- The raw_ch1 signal is sampled at acquisition rate (~20 kHz); index N corresponds to time N / acquisition_rate seconds
- The zarr file contains `metadata/acquisition_rate` for computing end times from plot_duration
- The zarr file follows the established schema with `stimulus/light_reference/raw_ch1` array
- Signal amplitude is sufficient that a reasonable threshold value can distinguish true stimulus onsets from noise
- During analog/ipRGC stimulation, there are no display frames (frame_timestamps doesn't cover this period)

## Out of Scope

- Unit response extraction (explicitly excluded per user requirement)
- Light template generation from raw signal (handled separately if needed)
- Real-time stimulus detection during recording
- Cross-channel synchronization validation
- Migration of existing section_time data from display frames to acquisition samples (existing data remains as-is)
