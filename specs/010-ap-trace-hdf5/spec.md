# Feature Specification: Axon Tracking (AP Trace) for HDF5 Pipeline

**Feature Branch**: `010-ap-trace-hdf5`  
**Created**: 2025-12-25  
**Status**: Draft  
**Input**: User description: "Migrate axon tracking analysis from pkl-based processing to hdf5-based processing for the current pipeline workflow"

## Overview

This feature migrates the axon tracking (AP trace) analysis from the legacy pkl-based data processing workflow to the new HDF5-based pipeline. The analysis detects axon pathways from Spike-Triggered Average (STA) data, predicts axon signal propagation using a trained CNN model, and calculates soma polar coordinates relative to the optic disc.

### Data Mapping: PKL to HDF5

| PKL Path (Legacy) | HDF5 Path (New) |
|-------------------|-----------------|
| `units_data[unit_id]["network_sta"]` | `units/{unit_id}/features/eimage_sta/data` |
| Cell location (soma center) | `units/{unit_id}/features/eimage_sta/geometry/center_row` (Y) and `center_col` (X) |
| DVNT position source | `metadata/gsheet_row/Center_xy` (format: "L/R, VD_coord, NT_coord") |
| **Output**: `units_data[unit_id]["ap_trace"]` | `units/{unit_id}/features/ap_tracking/` |
| **Output**: `units_data[unit_id]["soma_polar_coordinates"]` | `units/{unit_id}/features/ap_tracking/soma_polar_coordinates/` |
| **Output**: DV/NT/LR positions | `units/{unit_id}/features/ap_tracking/DV_position`, `NT_position`, `LR_position` |

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single File AP Trace Analysis (Priority: P1)

A researcher wants to run axon tracking analysis on a single HDF5 recording file that has already been processed with STA computation (eimage_sta). The system should detect soma location, predict axon pathways using the trained CNN model, and store the results back into the HDF5 file.

**Why this priority**: This is the core functionality - processing a single file is the foundation for all other use cases.

**Independent Test**: Can be fully tested by running `apply_ap_trace_to_hdf5("path/to/recording.h5")` and verifying that `ap_trace` features are written to each unit.

**Acceptance Scenarios**:

1. **Given** an HDF5 file with `eimage_sta` data computed for all units, **When** the user runs the AP trace analysis, **Then** the system writes `ap_trace` features (soma position, AIS, prediction data, pathway fit) to each unit's features group.

2. **Given** an HDF5 file with no `eimage_sta` data, **When** the user runs the AP trace analysis, **Then** the system raises an informative error indicating that STA computation is required first.

3. **Given** an HDF5 file where some units lack `eimage_sta` data, **When** the user runs the AP trace analysis, **Then** the system processes available units and logs warnings for skipped units.

---

### User Story 2 - Session-Based Deferred Processing (Priority: P2)

A researcher wants to integrate AP trace analysis into a multi-step pipeline session, where data is accumulated in memory and saved once at the end. This follows the existing pipeline design pattern using `PipelineSession`.

**Why this priority**: Session-based processing is the standard pattern in the new pipeline and enables efficient multi-step workflows.

**Independent Test**: Can be tested by creating a session, running AP trace with `session=session` parameter, and calling `session.save()`.

**Acceptance Scenarios**:

1. **Given** a PipelineSession with units containing `eimage_sta` features, **When** the user calls `compute_ap_trace(session=session)`, **Then** the session is updated with `ap_trace` features and returned for further processing.

2. **Given** a session with AP trace results, **When** `session.save()` is called, **Then** all AP trace features are persisted to the HDF5 file.

---

### User Story 3 - Batch Processing Multiple Files (Priority: P3)

A researcher wants to process multiple HDF5 files in a batch, with progress tracking, skip-existing functionality, and summary logging.

**Why this priority**: Batch processing is essential for large-scale analysis but depends on single-file processing working correctly.

**Independent Test**: Can be tested by running batch processor on a folder with multiple HDF5 files and verifying all outputs.

**Acceptance Scenarios**:

1. **Given** a folder with multiple HDF5 files, **When** the batch processor runs, **Then** each file is processed and a summary log is generated.

2. **Given** a folder where some files have already been processed, **When** the batch processor runs, **Then** existing files are skipped and processing continues with remaining files.

---

### Edge Cases

- What happens when STA data has unexpected dimensions (not 50x65x65)?
- How does the system handle units with very few spikes (noisy STA)?
- What happens if the trained model file is missing or corrupted?
- How are GPU memory errors handled during batch processing?
- What happens if `metadata/gsheet_row/Center_xy` is missing or empty? → Proceed without DVNT positions, set them to None/NaN

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST read STA data from `units/{unit_id}/features/eimage_sta/data` in HDF5 files
- **FR-001b**: System MUST read cell location from `units/{unit_id}/features/eimage_sta/geometry/center_row` (Y coordinate) and `center_col` (X coordinate) for polar coordinate calculations
- **FR-002**: System MUST detect soma location from 3D STA data using existing `find_soma_from_3d_sta` algorithm
- **FR-003**: System MUST refine soma position using existing `soma_refiner` algorithm
- **FR-004**: System MUST detect axon initial segment (AIS) using existing `AIS_refiner` algorithm
- **FR-005**: System MUST apply the trained CNN model (`CNN_3d_with_velocity_model_from_all_process.pth`) to predict axon signals
- **FR-006**: System MUST post-process predictions to extract axon centroids
- **FR-007**: System MUST fit lines to axon projections and calculate AP pathway intersection
- **FR-008**: System MUST calculate soma polar coordinates relative to the calculated intersection point
- **FR-009**: System MUST write all AP trace results to `units/{unit_id}/features/ap_tracking/` in HDF5
- **FR-009b**: System MUST parse `metadata/gsheet_row/Center_xy` to extract DVNT positions using this conversion:
  - Format: "L/R, VD_coord, NT_coord" (e.g., "L, 1.5, -0.8")
  - `DV_position` = -VD_coord (positive = dorsal, negative = ventral)
  - `NT_position` = NT_coord (positive = nasal, negative = temporal)
  - `LR_position` = L/R string
- **FR-009c**: System MUST save `DV_position`, `NT_position`, `LR_position` to each unit's `features/ap_tracking/` group
- **FR-010**: System MUST support GPU acceleration when available (CUDA)
- **FR-011**: System MUST support CPU fallback when GPU is unavailable
- **FR-012**: System MUST support both immediate-save and deferred-save (session) modes
- **FR-013**: System MUST preserve existing HDF5 data when adding AP trace features
- **FR-014**: System MUST always overwrite existing `ap_tracking` features when reprocessing a unit

### Key Entities

- **Unit**: A detected neuron with spike times, waveform, and computed features. Key attributes: unit_id, spike_times, features dict.
- **STA (Spike-Triggered Average)**: 3D array (time, x, y) representing the average stimulus pattern preceding spikes. Located at `features/eimage_sta/data`.
- **Soma**: Detected center of the cell body. Attributes: refined_soma_txy (t, x, y coordinates).
- **Axon Initial Segment (AIS)**: First segment of axon near soma. Attributes: ais_txy coordinates.
- **AP Trace Prediction**: CNN model output predicting axon signal probability at each STA voxel. Shape matches STA.
- **Axon Centroid**: Post-processed center of axon signal at each time point.
- **AP Pathway**: Fitted line to axon projections with slope, intercept, and R² quality metric.
- **Soma Polar Coordinates**: Soma position in polar coordinates (radius, angle) relative to optic disc intersection.

### Assumptions

- STA data is computed before AP trace analysis (dependency on `compute_sta` or equivalent)
- The trained CNN model file is located at `Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth`
- Anatomical orientation (DV/NT position) may not be available for all recordings; the system will proceed without it when missing
- Default processing parameters match legacy implementation unless specified otherwise
- Test file for development: `Projects/load_gsheet/export_gsheet_20251225/2025.03.06-12.38.11-Rec.h5`

## Clarifications

### Session 2025-12-25

- Q: Where are the trained model files located? → A: `Projects/ap_trace_hdf5/model/`
- Q: Reprocessing behavior when ap_trace features already exist? → A: Always overwrite existing results
- Q: Where to get cell location for polar coordinate calculations? → A: `features/eimage_sta/geometry/center_row` and `center_col`
- Q: Coordinate mapping for center_row and center_col? → A: `center_row` = Y coordinate, `center_col` = X coordinate (standard NumPy array indexing)
- Q: Test file for development? → A: `Projects/load_gsheet/export_gsheet_20251225/2025.03.06-12.38.11-Rec.h5`
- Q: DVNT position source and output location? → A: Parse `metadata/gsheet_row/Center_xy` using legacy `label_DVNT_position` logic; output to `features/ap_tracking/` with DV_position, NT_position, LR_position

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System processes a single HDF5 file with 20 units in under 60 seconds on GPU
- **SC-002**: System processes a single HDF5 file with 20 units in under 5 minutes on CPU
- **SC-003**: 100% of units with valid STA data receive AP trace analysis results
- **SC-004**: Output AP trace features are readable and correctly structured in HDF5 viewer tools
- **SC-005**: Batch processing correctly handles 100+ files with resume capability
- **SC-006**: System produces equivalent results to legacy pkl-based processing for the same input data
