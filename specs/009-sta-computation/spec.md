# Feature Specification: Spike Triggered Average (STA) Computation

**Feature Branch**: `009-sta-computation`  
**Created**: 2025-12-18  
**Status**: Draft  
**Input**: User description: "Compute STA (spike triggered average) for noise movies using sectioned spike times, with configurable cover range and multiprocessing support"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compute STA for Noise Movie (Priority: P1)

A researcher wants to compute the Spike Triggered Average (STA) for all units in a recording using a noise stimulus movie. The system automatically identifies the noise movie from the available movies, loads the corresponding stimulus data, and computes STA for each unit using their sectioned spike times.

**Why this priority**: This is the core functionality that enables receptive field analysis - the primary purpose of STA computation.

**Independent Test**: Can be tested by running STA computation on a recording with a noise movie and verifying the output STA array has correct dimensions and values.

**Acceptance Scenarios**:

1. **Given** an HDF5 recording with sectioned spike times and exactly one movie containing "noise" in its name, **When** STA computation is requested, **Then** the system computes STA for each unit and saves results to the features group.

2. **Given** an HDF5 recording with sectioned spike times, **When** STA computation completes, **Then** the computed STA is saved as a 3D array under `features/{noise_movie_name}/sta`.

3. **Given** a unit with spike times in the noise movie trial, **When** STA is computed with default cover_range (-60, 0), **Then** the STA represents the average stimulus preceding each spike within the valid index range.

---

### User Story 2 - Validate Noise Movie Detection (Priority: P1)

A researcher needs clear feedback when the recording doesn't have exactly one noise movie, preventing computation with ambiguous or missing stimulus data.

**Why this priority**: Proper error handling prevents incorrect analysis and provides clear guidance to users.

**Independent Test**: Can be tested by providing recordings with 0, 1, or 2+ movies containing "noise" in the name.

**Acceptance Scenarios**:

1. **Given** an HDF5 recording with no movies containing "noise" in the name, **When** STA computation is requested, **Then** the system raises an error indicating no noise movie was found.

2. **Given** an HDF5 recording with multiple movies containing "noise" in their names, **When** STA computation is requested, **Then** the system raises an error listing the ambiguous movie names.

---

### User Story 3 - Parallel Processing for Performance (Priority: P2)

A researcher with many units needs STA computation to complete in reasonable time by utilizing multiple CPU cores. The researcher can optionally disable multiprocessing for debugging or resource-constrained environments.

**Why this priority**: Performance optimization improves user experience but is not strictly required for core functionality.

**Independent Test**: Can be tested by measuring computation time with multiprocessing enabled vs disabled.

**Acceptance Scenarios**:

1. **Given** a recording with many units, **When** STA computation runs with `use_multiprocessing=True`, **Then** computation utilizes 80% of available CPU cores and completes faster than single-threaded execution.

2. **Given** a recording with many units, **When** STA computation runs with `use_multiprocessing=False`, **Then** computation runs sequentially on a single thread.

3. **Given** STA computation is running, **When** units are being processed, **Then** a progress bar displays the current processing status.

---

### User Story 4 - Handle Edge Effects (Priority: P2)

A researcher needs edge cases handled properly when spike times are near the beginning or end of the stimulus, ensuring only valid averaging windows are used.

**Why this priority**: Data integrity is important but this is a refinement of the core algorithm.

**Independent Test**: Can be tested by verifying spikes near stimulus boundaries are excluded from averaging.

**Acceptance Scenarios**:

1. **Given** a spike time where `spike_time + cover_range[0] < 0`, **When** STA is computed, **Then** that spike is excluded from the average (no out-of-bounds access).

2. **Given** a spike time where `spike_time + cover_range[1] >= stimulus_length`, **When** STA is computed, **Then** that spike is excluded from the average.

---

### Edge Cases

- What happens when a unit has no valid spike times within the bounds for STA computation?
  - System should return an empty or NaN STA and log a warning.
- What happens when the stimulus .npy file is not found at the expected path?
  - System should raise a clear FileNotFoundError with the expected path.
- What happens when cover_range values are invalid (e.g., start > end)?
  - System should validate and raise an error before computation.

## Clarifications

### Session 2025-12-18

- Q: What is the exact path for stimulus movie files? → A: `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\{movie_name}.npy`
- Q: Should multiprocessing be optional? → A: Yes, provide a parameter to enable/disable multiprocessing.
- Q: Should progress be displayed during computation? → A: Yes, show a progress bar when processing STA.
- Q: How should multiprocessing be configured? → A: Process units in parallel using 80% of CPU count.
- Q: How should the stimulus movie be handled for multiprocessing? → A: Load movie once, use shared memory for all workers.
- Q: What extraction strategy for spike windows? → A: Vectorized with pre-computed valid indices and numpy fancy indexing.
- Q: What numeric precision for STA computation? → A: Match input dtype; raise warning if not uint8.
- Q: How to compute STA mean from windows? → A: Stack all windows into array, use np.mean(axis=0).
- Q: How to handle errors in parallel workers? → A: Retry failed units once, then continue with partial results.
- Q: What test file should be used? → A: `artifacts/2025.04.10-11.12.57-Rec.h5`
- Q: How are spike times related to movie frames? → A: Spike times are sampling indices, not frame numbers. Convert by rounding to closest frame number.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST identify the noise movie by searching for "noise" (case-insensitive) in movie names under `spike_times_sectioned`.

- **FR-002**: System MUST raise an error if zero or more than one movie contains "noise" in its name, listing the found movie names in the error message.

- **FR-003**: System MUST load the stimulus movie data from the absolute path: `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\{noise_movie_name}.npy`.

- **FR-004**: System MUST use the first trial's spike times (`trials_spike_times/0`) from each unit for STA computation.

- **FR-005**: System MUST compute STA using a configurable `cover_range` parameter with default value `(-60, 0)`.

- **FR-006**: System MUST only include spikes where the full window `[spike_time + cover_range[0], spike_time + cover_range[1])` is within valid stimulus array bounds.

- **FR-007**: System MUST compute STA as the mean of all valid stimulus windows aligned to spike times.

- **FR-008**: System MUST save the computed STA as a 3D array to `features/{noise_movie_name}/sta` for each unit.

- **FR-009**: System MUST provide a `use_multiprocessing` parameter (default: True) to enable or disable parallel processing.

- **FR-010**: System MUST validate that cover_range[0] < cover_range[1] before computation.

- **FR-011**: System MUST display a progress bar during STA computation showing processing status across units.

- **FR-012**: When multiprocessing is enabled, system MUST process units in parallel using 80% of available CPU cores (rounded down, minimum 1).

- **FR-013**: System MUST load the stimulus movie once and use shared memory to make it accessible to all worker processes, avoiding redundant I/O and reducing memory footprint.

- **FR-014**: System MUST use vectorized numpy operations for spike window extraction: pre-compute all valid spike indices first, then use numpy fancy indexing for batch extraction instead of Python loops.

- **FR-015**: System MUST preserve the input stimulus dtype during computation. If the stimulus dtype is not uint8, system MUST log a warning indicating unexpected dtype.

- **FR-016**: System MUST compute STA by stacking all valid spike-aligned windows into an array and using numpy's mean function with axis=0 for optimal performance.

- **FR-017**: When a unit fails during STA computation, system MUST retry that unit once. If it fails again, log a warning and continue processing remaining units, returning partial results.

- **FR-018**: System MUST convert spike times from sampling indices to movie frame numbers by rounding to the closest frame. This conversion uses the acquisition rate and frame rate metadata to compute the correct mapping.

### Key Entities

- **STA (Spike Triggered Average)**: A 3D array representing the average stimulus pattern preceding spikes. Dimensions correspond to the stimulus movie dimensions (e.g., height × width × time_window).

- **Noise Movie**: A visual stimulus movie used for receptive field mapping. Identified by "noise" substring in the movie name.

- **Cover Range**: A tuple (start_offset, end_offset) defining the temporal window relative to each spike time for averaging. Negative values indicate time before the spike.

- **Spike Time Conversion**: The process of converting spike times from sampling indices (acquisition rate, e.g., 20kHz) to movie frame numbers (display rate, e.g., 15Hz). Conversion uses rounding to the closest frame number.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: STA computation completes for all units in a typical recording (100+ units) within 5 minutes using multiprocessing.

- **SC-002**: Computed STA arrays have dimensions matching the stimulus movie spatial dimensions and the cover_range temporal extent.

- **SC-003**: Edge effect handling correctly excludes spikes near boundaries with 100% accuracy (no out-of-bounds errors).

- **SC-004**: Clear error messages are provided when noise movie detection fails (0 or >1 matches).

- **SC-005**: Progress bar accurately reflects processing progress (0% to 100%) during STA computation.

- **SC-006**: Multiprocessing utilizes exactly 80% of available CPU cores (rounded down, minimum 1 worker).

## Assumptions

- The stimulus movie .npy files are pre-existing and located at `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\`.
- The noise movie name in the HDF5 matches the .npy filename (without extension).
- Spike times in `trials_spike_times` are sampling indices (acquisition rate units), NOT movie frame numbers. They must be converted to frame numbers before use.
- The first trial (`trials_spike_times/0`) contains representative data for STA computation.
- The system has access to the tqdm library (or similar) for progress bar display.
- Test file for validation: `artifacts/2025.04.10-11.12.57-Rec.h5`
