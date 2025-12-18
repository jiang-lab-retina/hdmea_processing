# Feature Specification: Replace Zarr Format with HDF5

**Feature Branch**: `007-hdf5-replace-zarr`  
**Created**: 2025-12-17  
**Status**: Draft  
**Input**: User description: "use HDF5 format to replace zarr"

## Clarifications

### Session 2025-12-17

- Q: What concurrent access model should HDF5 files use? → A: Single-writer only: raise error if file is already open for writing
- Q: Should HDF5 datasets use compression? → A: No compression: prioritize fastest I/O over file size
- Q: What file extension should output files use? → A: Use `.h5` extension for all output files
- Q: Should the implementation ensure MATLAB compatibility? → A: Basic compatibility using standard HDF5 types; no special MATLAB testing required

## Overview

This feature replaces the Zarr storage format with HDF5 throughout the HD-MEA data analysis pipeline. Currently, the pipeline stores all recording data (spike times, waveforms, stimulus information, metadata, and extracted features) in Zarr archives (`.zarr` directories). This change migrates to HDF5 single-file archives (`.h5` files), providing a more widely-supported format with better tooling compatibility.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create Recording in HDF5 Format (Priority: P1)

As a researcher, I want the pipeline to create recording data in HDF5 format instead of Zarr, so that I can use standard HDF5 tools (HDFView, h5py, MATLAB) to inspect and share my data without requiring specialized Zarr libraries.

**Why this priority**: This is the foundational change—all other functionality depends on correctly creating and writing to HDF5 files. The pipeline must create valid, readable HDF5 archives before any other features can work.

**Independent Test**: Can be fully tested by running the Stage 1 pipeline on a CMTR file and verifying the output is a valid `.h5` file readable by h5py with the expected group structure.

**Acceptance Scenarios**:

1. **Given** a CMTR recording file,  
   **When** I run the Stage 1 pipeline,  
   **Then** output is a single `.h5` file (not a `.zarr` directory) containing all recording data

2. **Given** a newly created HDF5 archive,  
   **When** I open it with h5py or HDFView,  
   **Then** I can navigate the group structure: `/units`, `/stimulus`, `/metadata`

3. **Given** metadata with acquisition_rate = 20000 Hz,  
   **When** data is written to HDF5,  
   **Then** the value is stored in `/metadata/acquisition_rate` and readable as a scalar

---

### User Story 2 - Read Existing Data from HDF5 (Priority: P1)

As a researcher, I want to read spike times, waveforms, and stimulus data from HDF5 files using the pipeline's existing API, so that I can perform analysis on my recordings with minimal code changes.

**Why this priority**: Reading data is equally fundamental—features, visualization, and all downstream analysis depend on reliable data access. This must work alongside creation for the pipeline to function.

**Independent Test**: Can be tested by creating an HDF5 file with known data, then using the pipeline's open/read functions to verify all data is correctly retrieved.

**Acceptance Scenarios**:

1. **Given** an HDF5 file with spike_times stored in `/units/{unit_id}/spike_times`,  
   **When** I call `open_recording_hdf5()` and access unit spike times,  
   **Then** I receive a numpy array identical to the original data

2. **Given** an HDF5 file with stimulus section_time data,  
   **When** I read `/stimulus/section_time/{movie_name}`,  
   **Then** I receive the correct trial boundary arrays

3. **Given** an HDF5 file created by the pipeline,  
   **When** I read metadata from the file,  
   **Then** all attributes (dataset_id, acquisition_rate, created_at) are preserved

---

### User Story 3 - Write Extracted Features to HDF5 (Priority: P2)

As a researcher, I want extracted features (step-up response, receptive fields, etc.) to be stored in HDF5 format, so that all my analysis results are in one unified file format.

**Why this priority**: Feature extraction is the core analytical output of the pipeline. This depends on Stories 1-2 for basic I/O but represents the pipeline's value-add.

**Independent Test**: Can be tested by running a feature extractor on an HDF5 recording and verifying the feature data is correctly written to `/units/{unit_id}/features/{feature_name}`.

**Acceptance Scenarios**:

1. **Given** a recording HDF5 with spike_times and stimulus data,  
   **When** I run a feature extractor (e.g., step_up),  
   **Then** feature values are written to `/units/{unit_id}/features/{feature_name}` in the same HDF5 file

2. **Given** extracted features with scalar and array values,  
   **When** data is written,  
   **Then** scalars are stored as HDF5 attributes, arrays as datasets

---

### User Story 4 - Visualize HDF5 Data with Existing GUI (Priority: P3)

As a researcher, I want the existing Zarr visualization GUI to work with HDF5 files, so that I can interactively explore my data without code changes.

**Why this priority**: Visualization is important for data exploration but not critical for the core pipeline functionality. Depends on Stories 1-2.

**Independent Test**: Can be tested by launching the visualization app with an HDF5 file path and verifying the tree view and plots render correctly.

**Acceptance Scenarios**:

1. **Given** an HDF5 recording file,  
   **When** I launch the visualization GUI with this file,  
   **Then** I see the hierarchical tree structure (units, stimulus, metadata)

2. **Given** the visualization GUI showing an HDF5 file,  
   **When** I select a unit's spike_times,  
   **Then** I can view plots of the spike data

---

### Edge Cases

- What happens when an HDF5 file is corrupted or incomplete? → Return clear error message indicating corruption; do not silently fail
- What happens when trying to write to a read-only HDF5 file? → Raise appropriate permission error with clear message
- What happens when HDF5 file path doesn't end with `.h5`? → Accept any path but log warning if extension is not `.h5` or `.hdf5`
- What happens when disk is full during write? → Allow h5py's native exception to propagate; pipeline catches and logs appropriately
- What happens when another process is already writing to the HDF5 file? → Raise error immediately with clear message indicating file is locked; single-writer access model enforced

## Requirements *(mandatory)*

### Functional Requirements

**Core I/O Operations**

- **FR-001**: System MUST provide `create_recording_hdf5()` function that creates a new HDF5 file with the standard group hierarchy (`/units`, `/stimulus`, `/metadata`)
- **FR-002**: System MUST provide `open_recording_hdf5()` function that opens an existing HDF5 file with specified access mode (read, read-write, append)
- **FR-003**: System MUST use `h5py` library for all HDF5 operations
- **FR-004**: System MUST create single-file HDF5 archives (not directories like Zarr)
- **FR-004a**: System MUST enforce single-writer access model: raise error if file is already open for writing by another process

**Data Writing**

- **FR-005**: System MUST provide `write_units()` function that writes spike_times, waveforms, and unit metadata to `/units/{unit_id}/`
- **FR-006**: System MUST provide `write_stimulus()` function that writes light_reference, frame_time, and section_time to `/stimulus/`
- **FR-007**: System MUST provide `write_metadata()` function that writes recording metadata to `/metadata/`
- **FR-008**: System MUST store numpy arrays as HDF5 datasets
- **FR-009**: System MUST store scalar values and simple types as HDF5 attributes on parent groups

**Data Reading**

- **FR-010**: System MUST support lazy loading where HDF5 datasets are not loaded into memory until accessed
- **FR-011**: System MUST preserve exact data types (uint64 for spike_times, float32 for waveforms)

**Feature Storage**

- **FR-012**: System MUST provide `write_feature_to_unit()` function for storing extracted features
- **FR-013**: Feature arrays MUST be stored as datasets under `/units/{unit_id}/features/{feature_name}/`

**Migration Compatibility**

- **FR-014**: All existing functions in `zarr_store.py` MUST have equivalent functions in new HDF5 module with same signatures
- **FR-015**: System MUST maintain the same logical data structure (groups, datasets, attributes) as current Zarr implementation

**Visualization**

- **FR-016**: Existing visualization module MUST be updated to read from HDF5 files
- **FR-017**: Tree view MUST display HDF5 groups and datasets in same format as Zarr

### Key Entities

- **Recording HDF5 File**: Single `.h5` file containing all data for one recording session; replaces `.zarr` directory
- **Unit Group**: HDF5 group at `/units/{unit_id}/` containing spike_times dataset, waveform dataset, and features subgroup
- **Stimulus Group**: HDF5 group at `/stimulus/` containing light_reference, frame_time, and section_time subgroups
- **Metadata Group**: HDF5 group at `/metadata/` containing recording parameters as datasets and attributes
- **Feature Group**: HDF5 group at `/units/{unit_id}/features/{feature_name}/` containing extracted feature values

### Data Model

**File Naming Convention**: Output files use `.h5` extension (e.g., `JIANG009_2025-04-10.h5`)

**File Structure (HDF5 equivalent of current Zarr structure)**:

```
{dataset_id}.h5
├── / (root)
│   ├── [attrs] dataset_id, hdmea_pipeline_version, created_at, updated_at, stage1_completed
│   │
│   ├── units/
│   │   └── {unit_id}/
│   │       ├── [attrs] row, col, global_id, spike_count
│   │       ├── spike_times          # dataset: (N,) uint64
│   │       ├── waveform             # dataset: (M,) float32
│   │       ├── firing_rate_10hz     # dataset: (T,) float32
│   │       ├── spike_times_sectioned/
│   │       │   └── {movie}/
│   │       │       ├── full_spike_times  # dataset: (K,) int64
│   │       │       └── trials_spike_times/
│   │       │           └── {trial_idx}   # dataset: (J,) int64
│   │       └── features/
│   │           └── {feature_name}/
│   │               └── {values}     # datasets/attrs for feature data
│   │
│   ├── stimulus/
│   │   ├── light_reference/
│   │   │   └── {rate_name}          # dataset: (S,) float32
│   │   ├── frame_time/
│   │   │   └── {movie_name}         # dataset: (F,) uint64
│   │   └── section_time/
│   │       └── {movie_name}         # dataset: (R, 2) uint64
│   │
│   └── metadata/
│       ├── acquisition_rate         # dataset or attr
│       ├── frame_time               # dataset or attr
│       └── sys_meta/                # subgroup for system metadata
```

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All pipeline stages (load, section, feature extraction) complete successfully with HDF5 files
- **SC-002**: HDF5 files are readable by standard tools (h5py, HDFView) without custom code
- **SC-003**: Data round-trip (write then read) preserves all values with exact precision
- **SC-004**: File I/O performance is within 20% of current Zarr performance for typical operations
- **SC-005**: All existing unit tests pass after migration to HDF5
- **SC-006**: Visualization GUI displays HDF5 data correctly

## Assumptions

- The `h5py` library is available and compatible with the current Python environment
- HDF5 single-file format is acceptable for all use cases (no need for distributed/cloud storage features of Zarr)
- Basic MATLAB compatibility is achieved through standard HDF5 types (uint64, float32); no special MATLAB testing required
- Existing Zarr archives will not be automatically migrated; users can re-run pipeline to generate HDF5 versions
- The logical data structure (hierarchy of groups and datasets) remains unchanged; only the storage format changes
- HDF5 datasets are stored without compression to maximize I/O speed; file size is acceptable tradeoff
- Chunking settings can use h5py defaults initially; optimization is out of scope

## Out of Scope

- Automatic migration tool for existing `.zarr` archives to `.h5`
- Support for both Zarr and HDF5 simultaneously (HDF5 completely replaces Zarr)
- Advanced HDF5 features (virtual datasets, external links, SWMR)
- Distributed/cloud storage (Zarr's primary advantage over HDF5)
- Performance optimization beyond basic functionality
