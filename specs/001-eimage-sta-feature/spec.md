# Feature Specification: Electrode Image STA (eimage_sta)

**Feature Branch**: `001-eimage-sta-feature`  
**Created**: 2025-12-19  
**Status**: Draft  
**Input**: User description: "Compute eimage_sta (called network_sta in the legacy code) and add it as eimage_sta item in the features of each unit. Use the legacy code approach."

## Clarifications

### Session 2025-12-19

- Q: How should the high-pass filter be optimized? → A: Vectorize filtering - apply filter across entire 3D array using axis parameter (not per-electrode loops)
- Q: How should STA computation be parallelized? → A: Vectorized spike extraction - use NumPy fancy indexing to extract all windows at once per unit
- Q: How should large sensor data be loaded? → A: Memory-mapped access - use HDF5 memory mapping for on-demand data access
- Q: What is the acceptable computation time target? → A: Under 5 minutes for typical recording (120 seconds data, ~100 units)
- Q: Should filtered data be cached for reuse? → A: Optional cache - cache filtered data to temp/HDF5 if user enables

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compute Electrode Image STA for Unit Analysis (Priority: P1)

A researcher wants to understand the spatial electrical activity pattern across the HD-MEA electrode array triggered by a unit's spikes. By computing the eimage_sta, they can visualize how electrical signals propagate across the electrode grid in the time window surrounding each spike, revealing axonal footprints and network activity patterns.

**Why this priority**: This is the core feature - computing and storing the electrode image STA is the fundamental requirement that all other use cases depend on.

**Independent Test**: Can be fully tested by loading an HDF5 file with units, running the eimage_sta computation, and verifying that each unit has a valid 3D array (time × rows × columns) stored in its features.

**Acceptance Scenarios**:

1. **Given** an HDF5 file with units and raw sensor data, **When** the user runs the eimage_sta computation, **Then** each unit receives an `eimage_sta` entry in its features group containing a 3D array.
2. **Given** an HDF5 file with units, **When** the computation completes, **Then** metadata is stored including the number of spikes used, filter parameters, and time window settings.
3. **Given** a unit with fewer spikes than the spike limit, **When** the eimage_sta is computed, **Then** all available spikes are used and the count is recorded accurately.

---

### User Story 2 - Configure Filter Parameters (Priority: P2)

A researcher needs to adjust the high-pass filter settings to optimize signal quality for their specific recording conditions. They want to customize the cutoff frequency and filter order.

**Why this priority**: While sensible defaults exist, different recording conditions may require parameter tuning for optimal results.

**Independent Test**: Can be tested by running the computation with different filter parameters and verifying the output reflects those settings.

**Acceptance Scenarios**:

1. **Given** custom filter parameters (cutoff frequency, filter order), **When** the computation runs, **Then** those parameters are applied to the data preprocessing.
2. **Given** no custom parameters specified, **When** the computation runs, **Then** default values are used (100 Hz cutoff, order 2).

---

### User Story 3 - Limit Computation Time with Spike Sampling (Priority: P3)

A researcher has units with very high spike counts and wants to limit computation time by using a subset of spikes. They configure a maximum spike limit to control processing duration.

**Why this priority**: Performance optimization for large datasets; the core feature works without this but it improves usability for high-firing units.

**Independent Test**: Can be tested by running on a high-spike-count unit with a low spike limit and verifying only that many spikes are used.

**Acceptance Scenarios**:

1. **Given** a spike limit of N, **When** a unit has more than N spikes, **Then** only the first N spikes are used for averaging.
2. **Given** a spike limit of N, **When** a unit has fewer than N spikes, **Then** all spikes are used and no error occurs.

---

### Edge Cases

- What happens when a unit has zero spikes? → Store an array filled with NaN and record 0 spikes used.
- What happens when spike times fall too close to recording boundaries? → Exclude those spikes from averaging (similar to existing STA behavior).
- What happens when sensor data is not available? → Raise a clear error indicating missing sensor data path.
- What happens when the recording is shorter than expected? → Process available data and log a warning about truncated duration.
- What happens when feature already exists? → Skip unless force=True is specified (consistent with existing STA pattern).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST compute electrode image STA by averaging raw sensor data in a time window around each spike for every unit in the HDF5 file.
- **FR-002**: System MUST apply a high-pass Butterworth filter to the sensor data before computing the average, using configurable cutoff frequency and filter order.
- **FR-003**: System MUST store the computed eimage_sta as a 3D array (time_samples × electrode_rows × electrode_columns) in each unit's features group.
- **FR-004**: System MUST record metadata with the result, including: number of spikes used, number of spikes excluded, filter parameters, and time window settings.
- **FR-005**: System MUST support configurable time windows (pre-spike and post-spike sample counts).
- **FR-006**: System MUST support an optional spike limit to cap the number of spikes used per unit.
- **FR-007**: System MUST skip units with existing eimage_sta unless force overwrite is enabled.
- **FR-008**: System MUST handle edge effects by excluding spikes whose time windows would extend beyond the sensor data boundaries.
- **FR-009**: System MUST return NaN-filled arrays for units with zero valid spikes.
- **FR-010**: System MUST provide progress feedback during long-running computations.
- **FR-011**: System MUST support optional caching of filtered sensor data for reuse across multiple analysis runs.

### Performance Requirements

- **PR-001**: System MUST apply high-pass filtering using vectorized operations across the entire 3D sensor array (using scipy's axis parameter), not per-electrode loops.
- **PR-002**: System MUST extract spike windows using vectorized NumPy fancy indexing to gather all windows for a unit in a single operation.
- **PR-003**: System MUST access sensor data via HDF5 memory-mapped I/O to handle large files without loading entirely into RAM.
- **PR-004**: System MUST complete computation for a typical recording (120 seconds of data, ~100 units) in under 5 minutes.
- **PR-005**: System SHOULD provide optional filtered data caching to accelerate repeated analysis runs on the same recording.

### Key Entities

- **eimage_sta**: A 3D array representing the average electrode activity pattern around spike times. Dimensions: (time_window_length, electrode_rows, electrode_columns). Stored as float32 for memory efficiency.
- **Sensor Data**: Raw voltage recordings from the HD-MEA electrode array. 3D array: (time_samples, rows, columns). Accessed via memory-mapped I/O.
- **Spike Times**: Timestamps of detected action potentials for a unit, converted to sample indices.
- **Filtered Data Cache**: Optional cached version of high-pass filtered sensor data stored in temp or HDF5 for reuse.

## Assumptions

- Sensor data is available in the source HDF5/H5 file in a standard location (consistent with existing data loading patterns).
- Acquisition rate metadata is available for timestamp-to-sample conversion.
- The electrode array has a 2D grid structure (rows × columns).
- Default parameters from legacy code: 100 Hz cutoff frequency, filter order 2, 10 samples pre-spike, 40 samples post-spike, 10,000 spike limit.
- First 120 seconds of sensor data is used (consistent with legacy approach).
- HDF5 files support memory-mapped access for efficient large data handling.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System computes and stores eimage_sta for a typical recording (120 seconds, ~100 units) in under 5 minutes.
- **SC-002**: Computed eimage_sta arrays have the expected shape: (pre_spike_samples + post_spike_samples, electrode_rows, electrode_columns).
- **SC-003**: Results are reproducible: running the same computation twice on identical data produces identical results.
- **SC-004**: Memory usage remains bounded during computation through memory-mapped data access.
- **SC-005**: Users can verify computation success by checking that the eimage_sta feature exists in each unit's features group with valid metadata attributes.
- **SC-006**: Vectorized filtering achieves at least 10x speedup compared to per-electrode loop approach.
- **SC-007**: Repeated runs with cached filtered data complete at least 2x faster than initial run.
