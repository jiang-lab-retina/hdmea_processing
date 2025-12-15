# Feature Specification: Add frame_time and acquisition_rate to Metadata

**Feature Branch**: `003-add-metadata-fields`  
**Created**: 2024-12-15  
**Status**: Draft  
**Input**: User description: "add frame_time, acquisition_rate to meta_data"

## Clarifications

### Session 2024-12-15

- Q: What is the source priority for extracting acquisition_rate? → A: CMCR primary, CMTR fallback, then default (20kHz)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Recording Timing Parameters (Priority: P1)

A researcher loads a recording into the pipeline and needs to know the timing characteristics (sampling rate and frame duration) for downstream analysis such as spike rate calculations, stimulus alignment, or time-domain signal processing.

**Why this priority**: Timing metadata is fundamental for all temporal analyses. Without knowing acquisition_rate and frame_time, users cannot correctly interpret spike timestamps, align stimuli, or calculate proper time axes.

**Independent Test**: Can be fully tested by loading a recording and verifying that `acquisition_rate` (Hz) and `frame_time` (seconds per sample) are accessible in the Zarr metadata group.

**Acceptance Scenarios**:

1. **Given** a recording is loaded from CMCR/CMTR files, **When** the user accesses the Zarr metadata group, **Then** `acquisition_rate` is present as a numeric value in Hz (e.g., 20000).
2. **Given** a recording is loaded from CMCR/CMTR files, **When** the user accesses the Zarr metadata group, **Then** `frame_time` is present as a numeric value in seconds (e.g., 0.00005 for 20kHz).
3. **Given** a CMCR file with acquisition rate of 20000 Hz, **When** the recording is loaded, **Then** `frame_time` equals `1/acquisition_rate` (i.e., 0.00005 seconds).

---

### User Story 2 - View Timing Metadata in Zarr Viz GUI (Priority: P2)

A researcher using the zarr-viz GUI wants to quickly inspect the timing parameters of a recording without writing code.

**Why this priority**: The GUI provides a convenient way to verify data integrity and understand recording parameters. Displaying timing metadata improves usability.

**Independent Test**: Can be tested by opening a Zarr file in zarr-viz and observing that `acquisition_rate` and `frame_time` appear in the metadata panel.

**Acceptance Scenarios**:

1. **Given** a Zarr file with timing metadata, **When** the user views the metadata group in zarr-viz, **Then** both `acquisition_rate` and `frame_time` are displayed with their values.

---

### Edge Cases

- What happens when CMCR file is not provided (CMTR only)?
  - System attempts to extract acquisition_rate from CMTR metadata. If unavailable, uses default (20000 Hz) and logs a warning.
- What happens when acquisition_rate cannot be determined from CMCR?
  - System falls back to CMTR metadata extraction. If still unavailable, uses default (20000 Hz) and logs a warning.
- What happens if acquisition_rate is zero or negative in source data?
  - System rejects invalid values and proceeds to next source in priority chain (CMCR → CMTR → default).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST store `acquisition_rate` in the Zarr metadata group as a numeric value representing samples per second (Hz).
- **FR-002**: System MUST compute and store `frame_time` in the Zarr metadata group as `1/acquisition_rate`, representing seconds per sample.
- **FR-003**: System MUST extract `acquisition_rate` using this priority chain: (1) CMCR file metadata/estimation, (2) CMTR file metadata, (3) default value of 20000 Hz.
- **FR-004**: System MUST validate that acquisition_rate is a positive number before storing.
- **FR-005**: System MUST log a warning when using default acquisition rate instead of extracted value.
- **FR-006**: System MUST attempt extraction from CMTR when CMCR extraction fails or CMCR is not provided.

### Key Entities

- **acquisition_rate**: The sampling rate of the recording in Hz (samples per second). Extracted using priority chain: CMCR file (primary) → CMTR file (fallback) → default 20000 Hz (last resort).
- **frame_time**: The duration of a single sample in seconds. Calculated as `1/acquisition_rate`. Also known as sample interval or sample period.
- **metadata group**: The Zarr group at `/metadata` that stores recording-level parameters as attributes or datasets.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of Zarr files produced by the pipeline contain both `acquisition_rate` and `frame_time` in the metadata group.
- **SC-002**: `frame_time` is mathematically consistent with `acquisition_rate` (i.e., `frame_time * acquisition_rate == 1.0` within floating-point tolerance).
- **SC-003**: Users can access timing metadata programmatically with a single attribute lookup (no computation required).
- **SC-004**: Timing metadata is visible in zarr-viz GUI without additional configuration.
- **SC-005**: When CMCR provides valid acquisition_rate, system uses it (verified by comparing against source file).
