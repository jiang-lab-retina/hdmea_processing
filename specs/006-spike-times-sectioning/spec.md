# Feature Specification: Spike Times Unit Conversion and Stimulation Sectioning

**Feature Branch**: `006-spike-times-sectioning`  
**Created**: 2025-12-17  
**Status**: Draft  
**Input**: User description: "1. change the code of unit spike_times in each unit(loaded raw data in 10^-9 second unit) to be the unit of per data acquisition interval. Make it happen when the data is created 2. Write new step in the pipeline to section each stimulation spike_times, and save it in a 'spike_times' dict of each unit"

## Clarifications

### Session 2025-12-17

- Q: What test dataset should be used for validation? → A: `artifacts/JIANG009_2025-04-10.zarr`
- Q: Behavior when sectioned data exists and force=False? → A: Raise FileExistsError (require explicit force=True to overwrite)
- Q: Should sectioned spike times be split per trial or combined? → A: **BOTH** - store `full_spike_times` (all trials combined) AND `trials_spike_times/{trial_idx}` (per-trial split using section_time boundaries)
- Q: New parameter for trial count? → A: Add `trial_repeats` parameter (default=3) to control number of trials to process
- Q: Should trial boundaries include padding? → A: Add `pad_margin` parameter as tuple `(pre_margin_s, post_margin_s)` with default=(2.0, 0.0) seconds. Pre-margin extends before trial start, post-margin extends after trial end. Converted to samples: `pre_samples = int(pre_margin * acquisition_rate)`, `post_samples = int(post_margin * acquisition_rate)`. Extends trial boundaries to `[start - pre_samples, end + post_samples]`

## Overview

This feature implements two related changes to the spike times handling in the HD-MEA pipeline:

1. **Unit Conversion at Load Time**: Convert raw spike timestamps from nanoseconds ($10^{-9}$ seconds) to acquisition sample indices during data loading (Stage 1), providing a unified time coordinate system aligned with section_time and other pipeline data.

2. **Spike Times Sectioning**: Add a new pipeline step that extracts spike timestamps falling within each stimulation period (defined by section_time) and stores them per-unit, enabling trial-aligned spike analysis.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load Spike Times in Sample Units (Priority: P1)

As a researcher, I want spike timestamps automatically converted to acquisition sample units when loading CMTR data, so that spike times use the same coordinate system as section_time boundaries and I don't need to perform manual unit conversions during analysis.

**Why this priority**: This is the foundational change—all downstream spike analysis depends on having consistent time units. Converting at load time ensures data consistency from the start.

**Independent Test**: Can be fully tested by loading a CMTR file with known spike timestamps (in nanoseconds), then verifying the stored spike_times are in acquisition sample units (nanoseconds × acquisition_rate / 10^9).

**Acceptance Scenarios**:

1. **Given** a CMTR file with spike timestamps in nanoseconds (10^-9 s),  
   **When** I run load_recording(),  
   **Then** the spike_times stored in `units/{unit_id}/spike_times` are in acquisition sample index units (integer sample indices at ~20 kHz)

2. **Given** a recording with acquisition_rate = 20000 Hz and a spike at timestamp 50,000,000 ns (50 ms),  
   **When** data is loaded,  
   **Then** the spike_times value is 1000 (= 50,000,000 × 20000 / 10^9 = 1000 samples)

3. **Given** metadata contains acquisition_rate extracted from CMCR/CMTR,  
   **When** spike_times are converted,  
   **Then** the conversion uses the actual acquisition_rate (not a hardcoded default)

---

### User Story 2 - Section Spike Times by Stimulation (Priority: P2)

As a researcher, I want to extract spike timestamps that fall within each stimulation period and store them per-unit, so that I can analyze neural responses aligned to stimulus presentations without manually filtering spike arrays.

**Why this priority**: This enables trial-aligned spike analysis which is essential for understanding stimulus-evoked responses. Depends on User Story 1 (unified time units) and existing section_time data.

**Independent Test**: Can be tested by loading a zarr with spike_times (in sample units) and section_time data, running the sectioning step, and verifying each unit has both `full_spike_times` (combined) and `trials_spike_times` (per-trial) under spike_times_sectioned.

**Acceptance Scenarios**:

1. **Given** a zarr file with `units/{unit_id}/spike_times` (in sample units) and `stimulus/section_time/{movie_name}` (trial boundaries),  
   **When** I run the spike times sectioning pipeline step with trial_repeats=3,  
   **Then** each unit contains:
   - `spike_times_sectioned/{movie_name}/full_spike_times` - ALL spikes from all trials combined
   - `spike_times_sectioned/{movie_name}/trials_spike_times/{trial_idx}` - spikes split per trial (0, 1, 2, ...)

2. **Given** section_time with 3 trials for "movie_A" with boundaries [[1000,2000], [4000,5000], [7000,8000]] and acquisition_rate=20000 Hz,  
   **When** sectioning runs with trial_repeats=3 and pad_margin=(2.0, 0.0) (= 40000 samples pre, 0 samples post),  
   **Then** trial boundaries are extended: [[0,2000], [0,5000], [0,8000]] (with pre-margin padding, clamped to 0)
   - Spikes within extended boundaries are included in each trial
   - `full_spike_times` contains all unique spikes from all padded trial windows
   - `trials_spike_times/{idx}` contains spikes within each padded trial window

3. **Given** a unit with no spikes falling within any trial period,  
   **When** sectioning runs,  
   **Then** empty arrays are stored for both full_spike_times and each trials_spike_times entry

4. **Given** section_time exists for multiple movies,  
   **When** sectioning runs,  
   **Then** spike_times_sectioned group contains subgroups for all movies present in section_time, each with full_spike_times and trials_spike_times

---

### User Story 3 - Handle Missing Data Gracefully (Priority: P3)

The pipeline should handle edge cases where section_time or spike_times data may be incomplete.

**Why this priority**: Robustness prevents pipeline failures and provides clear feedback for incomplete datasets.

**Independent Test**: Can be tested by providing zarr files with missing section_time or units with no spikes.

**Acceptance Scenarios**:

1. **Given** a zarr file without any section_time data,  
   **When** I run the spike times sectioning step,  
   **Then** the step completes with a warning indicating no section_time found, and no spike_times_sectioned group is created

2. **Given** a unit with an empty spike_times array,  
   **When** sectioning runs,  
   **Then** all movie arrays in spike_times_sectioned are empty arrays

---

### Edge Cases

- What happens when acquisition_rate is missing during load? → Use default (20000 Hz) with warning, same as existing behavior
- What happens when spike_times are already in sample units (re-processing)? → Sectioning step is idempotent; re-running produces same result
- What happens when section_time boundaries extend beyond spike_times range? → Only include spikes within valid range; no padding
- What happens when force=False and spike_times_sectioned group already exists? → Raise FileExistsError requiring explicit force=True to overwrite
- What happens when padded trial start is negative? → Clamp to 0 (no negative sample indices)
- What happens when padded trial end exceeds recording length? → Clamp to max spike_time (or recording duration if available)

## Requirements *(mandatory)*

### Functional Requirements

**Part 1: Unit Conversion at Load Time**

- **FR-001**: System MUST convert spike timestamps from nanoseconds to acquisition sample indices during CMTR data loading
- **FR-002**: System MUST use the formula: `sample_index = timestamp_ns × acquisition_rate / 10^9`
- **FR-003**: System MUST store converted spike_times as integer array (int64/uint64) in `units/{unit_id}/spike_times`
- **FR-004**: System MUST use the acquisition_rate extracted from CMCR/CMTR metadata (via existing priority chain)
- **FR-005**: System MUST round converted values to nearest integer sample index

**Part 2: Spike Times Sectioning Pipeline Step**

- **FR-006**: System MUST provide a pipeline step function `section_spike_times()` with parameters:
  - `trial_repeats` (default=3) - number of trials to process
  - `pad_margin` (default=(2.0, 0.0) seconds) - tuple of (pre_margin, post_margin) to extend trial boundaries
- **FR-007**: System MUST read section_time boundaries from `stimulus/section_time/{movie_name}` for each movie
- **FR-007a**: System MUST convert `pad_margin` from seconds to samples:
  - `pre_samples = int(pad_margin[0] * acquisition_rate)`
  - `post_samples = int(pad_margin[1] * acquisition_rate)`
- **FR-008**: System MUST extract spikes falling within padded trial boundaries: `(start_sample - pre_samples) <= spike_time < (end_sample + post_samples)`
- **FR-009**: System MUST store BOTH types of sectioned spike times:
  - `units/{unit_id}/spike_times_sectioned/{movie_name}/full_spike_times` - ALL spikes from all trials combined
  - `units/{unit_id}/spike_times_sectioned/{movie_name}/trials_spike_times/{trial_idx}` - spikes per trial (using section_time boundaries)
- **FR-010**: System MUST store absolute spike times (sample indices), NOT trial-relative offsets
- **FR-011**: System MUST use `trial_repeats` parameter to limit number of trials processed (uses first N trials from section_time)
- **FR-012**: System MUST raise FileExistsError if sectioned data exists for a unit and force=False; force=True overwrites existing data
- **FR-013**: System MUST process all units in the zarr archive

### Key Entities

- **spike_times (raw)**: Original timestamps from CMTR in nanoseconds (10^-9 seconds); read-only input
- **spike_times (converted)**: Timestamps converted to acquisition sample indices; stored in `units/{unit_id}/spike_times`
- **section_time**: Trial boundaries as [start_sample, end_sample] pairs per movie; already stored in acquisition sample units
- **full_spike_times**: All spikes from all trials combined; stored under `spike_times_sectioned/{movie}/full_spike_times`
- **trials_spike_times**: Spikes split per trial using section_time boundaries; stored under `spike_times_sectioned/{movie}/trials_spike_times/{trial_idx}`
- **trial_repeats**: Parameter controlling number of trials to process (default=3)
- **pad_margin**: Tuple of (pre_margin, post_margin) in seconds (default=(2.0, 0.0)); pre_margin extends before trial start, post_margin extends after trial end; converted to samples using acquisition_rate

### Data Model Changes

**Modified Arrays:**

| Array | Location | Shape | Dtype | Unit (Before) | Unit (After) |
|-------|----------|-------|-------|---------------|--------------|
| `spike_times` | `units/{unit_id}/spike_times` | (N,) | uint64 | μs (microseconds) | sample index |

**New Arrays:**

| Array | Location | Shape | Dtype | Unit | Description |
|-------|----------|-------|-------|------|-------------|
| `full_spike_times` | `units/{unit_id}/spike_times_sectioned/{movie}/full_spike_times` | (M,) | int64 | samples (absolute) | All spike times from all trials combined |
| `trials_spike_times/{idx}` | `units/{unit_id}/spike_times_sectioned/{movie}/trials_spike_times/{idx}` | (K,) | int64 | samples (absolute) | Spike times for trial idx (0, 1, 2, ...) |

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Spike times conversion produces values that, when multiplied by sample_interval, equal the original nanosecond timestamps within floating-point precision
- **SC-002**: Sectioned spike times for a trial contain exactly the spikes that fall within section_time boundaries (100% accuracy)
- **SC-003**: Processing completes within 10 seconds for recordings with up to 1000 units and 100 trials
- **SC-004**: All sectioned spike times are accessible via standard zarr API
- **SC-005**: Pipeline step is idempotent - re-running produces identical results

## Assumptions

- Raw spike timestamps from CMTR are in nanoseconds (10^-9 seconds) as per MaxWell/MaxTwo format specification
- acquisition_rate is available in metadata (via existing extraction from CMCR/CMTR)
- section_time data uses acquisition sample indices (per spec 005)
- Absolute spike times (sample indices) are stored, matching the full recording spike_times coordinate system
- Both combined (`full_spike_times`) and per-trial (`trials_spike_times`) storage are needed for different analysis workflows
- Default `trial_repeats=3` matches typical experimental design; can be overridden per-call
- Default `pad_margin=(2.0, 0.0)` seconds provides 2-second pre-stimulus baseline; no post-stimulus padding by default

## Test Data

- **Primary test dataset**: `artifacts/JIANG009_2025-04-10.zarr`
- This zarr archive should be used for validating unit conversion and sectioning functionality

## Out of Scope

- Waveform sectioning (only spike timestamps are sectioned)
- Spike rate computation per trial (handled by separate feature extractors)
- Cross-unit synchronization analysis
- Visualization of sectioned spikes
- Migration of existing spike_times data to new units (new recordings only)
