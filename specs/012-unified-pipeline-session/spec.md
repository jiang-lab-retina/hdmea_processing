# Feature Specification: Unified Pipeline Session

**Feature Branch**: `012-unified-pipeline-session`  
**Created**: 2025-12-28  
**Status**: Draft  
**Input**: User description: "Unified pipeline processing from cmcr/cmtr through every step using PipelineSession class"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Process Recording from CMCR/CMTR Files (Priority: P1)

A researcher wants to process a new recording from raw CMCR/CMTR files through the complete data processing pipeline, extracting all features and saving the results to an HDF5 file.

**Why this priority**: This is the primary use case - processing new recordings from their original source format through the entire analysis pipeline. It represents the most common workflow for data processing.

**Independent Test**: Can be fully tested by running the pipeline example script with a single CMCR/CMTR pair (2024.08.08-10.40.20-Rec) and verifying the output HDF5 contains all expected data groups matching the reference file.

**Acceptance Scenarios**:

1. **Given** a valid CMCR/CMTR file pair, **When** running the pipeline from start to finish, **Then** the output HDF5 file contains units, metadata, stimulus, and all extracted features
2. **Given** a valid CMCR/CMTR file pair, **When** running the pipeline with deferred saving, **Then** intermediate results are kept in memory and only written once at the end
3. **Given** invalid or missing source files, **When** attempting to start the pipeline, **Then** an appropriate error message is displayed before any processing begins

---

### User Story 2 - Resume Pipeline from Existing HDF5 (Priority: P2)

A researcher wants to continue processing from an existing HDF5 file (e.g., add new features or reprocess specific steps) without re-running the entire pipeline from raw files.

**Why this priority**: This enables incremental processing and reduces redundant computation. Essential for adding new analysis steps to existing data.

**Independent Test**: Can be tested by loading an intermediate HDF5 file, running additional pipeline steps, and verifying the output contains both original and new data.

**Acceptance Scenarios**:

1. **Given** an existing HDF5 file with partial processing, **When** loading into a session and running additional steps, **Then** all existing data is preserved and new features are added
2. **Given** an HDF5 file, **When** using the universal loader, **Then** all data from the file (units, metadata, stimulus, features) is loaded into the session
3. **Given** a corrupted or invalid HDF5 file, **When** attempting to load, **Then** an appropriate error message is displayed

---

### User Story 3 - Export Results with Flexible Output Options (Priority: P3)

A researcher wants to save processing results with the choice of creating a new file or updating an existing file, with safeguards to prevent accidental data loss during testing.

**Why this priority**: Flexible output options are essential for both production (updating existing files) and development/testing (creating new files to preserve originals).

**Independent Test**: Can be tested by saving to new file path and verifying original is unchanged, then saving with overwrite option and verifying file is updated.

**Acceptance Scenarios**:

1. **Given** a completed pipeline session, **When** saving with `export_mode="new_file"`, **Then** a new file is created without modifying any existing file
2. **Given** a completed pipeline session, **When** saving with `export_mode="overwrite"`, **Then** the existing file is replaced with new data
3. **Given** testing mode is enabled, **When** attempting to save, **Then** overwriting existing files is blocked

---

### User Story 4 - Add New Feature Extraction Step (Priority: P4)

A developer wants to add a new feature extraction step to the pipeline without modifying the core pipeline infrastructure.

**Why this priority**: Extensibility is important for long-term maintainability and allows the pipeline to evolve with new analysis methods.

**Independent Test**: Can be tested by creating a new feature extraction function following the pipeline pattern and verifying it integrates with the session workflow.

**Acceptance Scenarios**:

1. **Given** a new feature extraction function following the pipeline pattern, **When** integrating into the pipeline, **Then** it can be called with the session and returns the updated session
2. **Given** a feature extraction function in a project subfolder, **When** importing and calling it, **Then** it works independently and can be debugged without running the full pipeline

---

### Edge Cases

- What happens when the CMCR file exists but the corresponding CMTR file is missing or vice versa?
- How does the system handle units with missing spike times or invalid STA data?
- What happens when processing is interrupted mid-pipeline (power loss, error)? → User must restart from last checkpoint; use `session.checkpoint()` to save intermediate state for recovery
- How are memory constraints handled for recordings with very large numbers of units?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support starting the pipeline from CMCR/CMTR source files
- **FR-002**: System MUST support starting the pipeline from an existing HDF5 file
- **FR-003**: System MUST provide a universal HDF5 loader that loads all data from an HDF5 file into the session (units, metadata, stimulus, features) by default, with optional filters to selectively load specific features
- **FR-004**: Pipeline code MUST follow a concise, chainable pattern where each step returns the updated session
- **FR-005**: System MUST process data in memory without saving at each step (deferred mode) for performance
- **FR-006**: System MUST allow explicit saving at any pipeline step via `session.save()` or `session.checkpoint()`
- **FR-007**: System MUST support exporting results to a new file path (export as new)
- **FR-008**: System MUST support overwriting existing output files (with confirmation/flag)
- **FR-009**: System MUST default to `overwrite=False` on save operations; overwriting existing files requires explicit `overwrite=True` parameter
- **FR-010**: New feature extraction functions MUST be implementable in project subfolders and callable independently for debugging
- **FR-011**: System MUST provide two example files: one starting from CMCR/CMTR and one starting from HDF5
- **FR-012**: Pipeline MUST execute the following processing steps in order:
  1. Load recording with eimage_sta
  2. Add section time using playlist
  3. Section spike times
  4. Compute STA
  5. Add CMTR/CMCR metadata (unit_meta, sys_meta)
  6. Extract soma geometry
  7. Extract RF-STA geometry (Gaussian, DoG, ON/OFF fits)
  8. Load Google Sheet metadata
  9. Add manual cell type labels
  10. Compute AP tracking
  11. Section by direction (DSGC)
- **FR-013**: Test file 2024.08.08-10.40.20-Rec MUST produce output matching the reference file at Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5
- **FR-014**: System MUST log step completion messages via Python logging and display progress bars (tqdm) when iterating over units
- **FR-015**: If an optional external dependency (e.g., Google Sheet) is unavailable, system MUST skip that step with a visible warning (red text) and continue processing

### Key Entities

- **PipelineSession**: Central container holding all pipeline data in memory; manages deferred saving and tracks completed steps
- **Unit**: Individual neural unit with spike times, STA data, and extracted features
- **Metadata**: Recording metadata including acquisition parameters, source file info, and Google Sheet data
- **Stimulus**: Light reference, frame times, and section timing data
- **Features**: Extracted analysis results including eimage_sta geometry, RF geometry, AP tracking, and DSGC sectioning

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Processing a single recording from CMCR/CMTR through all 11 steps completes within 10 minutes on standard hardware
- **SC-002**: The test file (2024.08.08-10.40.20-Rec) output matches the reference file structure with all expected data groups present
- **SC-003**: Loading an HDF5 file into session restores 100% of the data that was saved
- **SC-004**: Pipeline code in example files requires no more than 20 lines for the core processing logic (excluding imports and configuration)
- **SC-005**: A new feature extraction step can be added by creating a single function file without modifying core pipeline code
- **SC-006**: Testing mode successfully blocks all file overwrites and logs attempted overwrite operations

## Clarifications

### Session 2025-12-28

- Q: How should interrupted processing be handled? → A: Manual checkpoints - user explicitly calls `checkpoint()` to save intermediate state
- Q: Should the universal HDF5 loader support selective loading? → A: Default load all, with optional filters (e.g., `load_features=["sta", "rf"]`)
- Q: How should testing mode be activated to prevent overwrites? → A: Parameter on save - `overwrite=False` is the default; must pass `overwrite=True` explicitly
- Q: How should progress be reported during pipeline execution? → A: Logger-based for step completion messages + progress bar (tqdm) for iterating units
- Q: What should happen if Google Sheet is unavailable? → A: Skip step with warning (displayed in red), continue pipeline without gsheet data

## Assumptions

- The user has access to the required source files (CMCR/CMTR) and supporting data (playlists, on/off dictionaries, Google Sheet credentials)
- The hdmea package and its dependencies are properly installed and configured
- Sufficient memory is available to hold a complete recording session in memory (typically < 8GB per recording)
- The existing function implementations in project subfolders (e.g., `compute_sta`, `extract_rf_geometry_session`) are correct and will be reused
- GPU availability is optional for AP tracking; CPU fallback is acceptable with longer processing time
