# Feature Specification: HD-MEA Data Analysis Pipeline v1

**Feature Branch**: `001-hdmea-modular-pipeline`  
**Created**: 2025-12-14  
**Updated**: 2025-12-14  
**Status**: Draft  
**Input**: Build a modular Python pipeline for HD-MEA recordings (.cmcr/.cmtr), extracting ~100 physiological features with extensible architecture

---

## Purpose and Scope

### What This System Does

The HD-MEA Data Analysis Pipeline v1 processes high-density multi-electrode array (HD-MEA) recordings to:

1. **Stage 1 - Data Loading**: Load raw recordings from external `.cmcr` and/or `.cmtr` files → produce a single Zarr artifact per recording (explicit file save)
2. **Stage 2 - Feature Extraction**: Read Zarr artifact → extract physiological features → write features back to the SAME Zarr artifact under each cell_id (explicit file save)
3. **Optional Export**: Generate Parquet summary tables from Zarr for cross-recording analysis
4. **Support** extensible downstream analyses and visualizations

### Two-Stage Pipeline Architecture (NON-NEGOTIABLE)

The pipeline has two distinct stages with explicit file saves between them:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Loading                                                  │
│  ─────────────────────                                                  │
│  Input:  External .cmcr/.cmtr files (paths provided by user)            │
│  Output: Single Zarr archive per recording (saved to disk)              │
│  Action: Load raw data, extract spikes/stimulus/metadata → Zarr        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (explicit file save)
                                    
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Feature Extraction                                            │
│  ──────────────────────────                                             │
│  Input:  Zarr archive from Stage 1                                      │
│  Output: SAME Zarr archive with features added under each cell_id       │
│  Action: Read Zarr → compute features → write to units/{id}/features/  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (explicit file save)
                                    
┌─────────────────────────────────────────────────────────────────────────┐
│  OPTIONAL: Export to Parquet                                            │
│  ──────────────────────────                                             │
│  Input:  Zarr archive with features                                     │
│  Output: Parquet table(s) for cross-recording analysis                  │
│  Action: Flatten per-unit features to tabular format                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### What is IN Scope (v1)

- Loading `.cmcr` and `.cmtr` files from external paths (files NOT stored in project folder)
- Stage 1: Producing single Zarr artifact per recording with raw data
- Stage 2: Adding extracted features to Zarr under each unit/cell_id
- Core feature extractors replicating legacy "set6a" workflow behavior:
  - ON/OFF response detection and quantification
  - Receptive field estimation via spike-triggered average (STA)
  - Direction selectivity index (DSI) and preferred direction
  - Orientation selectivity index (OSI)
  - Frequency response characteristics
  - Color (chromatic) response features
  - Baseline firing statistics
  - Response quality metrics
  - Cell type classification (RGC vs unknown based on axon propagation)
- Feature registry system for extensibility
- Optional Parquet export for cross-recording analysis
- JSON configuration for pipeline settings
- At least one end-to-end flow equivalent to legacy "set6a" workflow

### What is OUT OF Scope (v1)

- GUI or web interface
- Real-time processing
- Spike sorting (input is pre-sorted `.cmtr` files)
- Machine learning model training (analysis only)
- Cloud deployment or distributed processing
- Direct modification of any code in `./Legacy_code/`
- Copying or storing raw `.cmcr`/`.cmtr` files in the project folder

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load Recording to Zarr (Stage 1) (Priority: P1)

As a researcher, I want to load a paired `.cmcr` and `.cmtr` recording from an external path and produce a single Zarr archive containing all data needed for feature extraction, so that I can cache expensive I/O operations and work with a self-contained artifact.

**Why this priority**: This is the foundational step—no downstream analysis can occur without loaded data. The "one Zarr per recording" requirement is NON-NEGOTIABLE per project constitution.

**Independent Test**: Can be fully tested by providing external paths to a sample `.cmcr`/`.cmtr` pair and verifying the output Zarr contains expected data groups.

**Acceptance Scenarios**:

1. **Given** valid external paths to a `.cmcr` and `.cmtr` file pair for the same recording session,  
   **When** I run the data loading step with a dataset_id,  
   **Then** the system produces exactly ONE Zarr archive saved to `artifacts/{dataset_id}.zarr/` containing:
   - Per-unit spike timestamps (in microseconds) under `units/{unit_id}/spike_times`
   - Per-unit spike waveforms (mean cutout) under `units/{unit_id}/waveform`
   - Per-unit spatial location (row, column) in unit metadata
   - Light reference / stimulus timing data under `stimulus/`
   - Frame timing information under `stimulus/frame_time`
   - Recording metadata under `metadata/`
   - Source file paths recorded in `.zattrs` (for provenance, NOT copied)

2. **Given** an external path to a valid `.cmtr` file without a matching `.cmcr` file,  
   **When** I run the data loading step,  
   **Then** the system produces a Zarr archive with available data and logs a warning that raw sensor data is unavailable.

3. **Given** an external path to a valid `.cmcr` file without a matching `.cmtr` file,  
   **When** I run the data loading step,  
   **Then** the system produces a Zarr archive with raw light reference and metadata, logs a warning that spike-sorted units are unavailable.

4. **Given** a data loading step that has already produced a Zarr for a dataset_id with identical input file paths and parameters,  
   **When** I run data loading again with the same inputs,  
   **Then** the system skips processing and returns the existing artifact path (caching behavior).

5. **Given** the data loading step completes,  
   **Then** the Zarr file is explicitly saved to disk and readable by any standard Zarr-compatible tool.

---

### User Story 2 - Extract Features to Zarr (Stage 2) (Priority: P1)

As a researcher, I want to run feature extraction on a loaded Zarr archive and have the features written back into the SAME Zarr file under each cell/unit, so that all data for a recording remains in one self-contained artifact.

**Why this priority**: Feature extraction is the primary value delivered by this pipeline. Keeping features in the Zarr under each cell_id maintains data locality and simplifies access.

**Independent Test**: Can be tested by providing a Stage-1 Zarr and verifying features appear under `units/{unit_id}/features/` after extraction.

**Acceptance Scenarios**:

1. **Given** a valid Zarr archive from Stage 1 for a recording with "step_up" stimulus,  
   **When** I run feature extraction for ON/OFF response features,  
   **Then** the system writes features to the Zarr under each unit:
   ```
   units/{unit_id}/features/step_up_5s_5i_3x/
   ├── on_response_flag       # scalar or array
   ├── off_response_flag
   ├── on_peak_value
   ├── off_peak_value
   ├── on_sustained_response
   ├── off_sustained_response
   ├── response_quality
   ├── filtered_trace         # array data
   └── .zattrs                 # feature metadata (version, params_hash)
   ```

2. **Given** a valid Zarr archive for a recording with "dense_noise" stimulus,  
   **When** I run feature extraction for receptive field (STA) features,  
   **Then** the system writes to each unit:
   ```
   units/{unit_id}/features/perfect_dense_noise_15x15_15hz_r42_3min/
   ├── sta/                    # group containing STA arrays
   │   ├── sta_array           # 3D array (time × y × x)
   │   ├── center_coordinate   # [time, y, x]
   │   └── spike_as_frame_num  # array
   ├── gaussian_fit/
   │   ├── parameters_max
   │   ├── parameters_min
   │   └── ...
   └── .zattrs                 # feature metadata
   ```

3. **Given** a valid Zarr archive for a recording with "moving_bar" stimulus,  
   **When** I run feature extraction for direction selectivity features,  
   **Then** the system writes to each unit:
   ```
   units/{unit_id}/features/moving_h_bar_s5_d8_3x/
   ├── dsi_on
   ├── dsi_off
   ├── preferred_direction_on
   ├── preferred_direction_off
   ├── osi_on
   ├── osi_off
   ├── p_value/
   │   ├── on_p_value
   │   ├── off_p_value
   │   └── ...
   └── .zattrs                 # feature metadata
   ```

4. **Given** features already exist for a unit under the same feature name and version,  
   **When** I run feature extraction without `force=True`,  
   **Then** the system skips that feature (cache hit) and logs that it was skipped.

5. **Given** features already exist,  
   **When** I run feature extraction with `force=True`,  
   **Then** the system overwrites the existing features and logs a warning.

6. **Given** the feature extraction step completes,  
   **Then** the Zarr file is explicitly saved to disk with all new features.

---

### User Story 3 - Export Features to Parquet (Priority: P3)

As a researcher, I want to export features from multiple Zarr files to a combined Parquet table, so that I can perform cross-recording analysis and use standard data science tools.

**Why this priority**: Export is optional and useful for aggregate analysis, but the primary artifact (Zarr with embedded features) must work first.

**Independent Test**: Can be tested by running export on a Zarr with features and verifying the Parquet contains flattened feature columns.

**Acceptance Scenarios**:

1. **Given** one or more Zarr archives with extracted features,  
   **When** I run the Parquet export step,  
   **Then** the system produces a Parquet table with one row per unit containing:
   - `dataset_id`, `unit_id`
   - Flattened feature columns (e.g., `step_up_5s_5i_3x__on_peak_value`)
   - `_export_version`, `_export_timestamp`

2. **Given** a Parquet export request for specific feature sets only,  
   **When** I specify which features to export,  
   **Then** only those feature columns are included in the output.

---

### User Story 4 - Run End-to-End Pipeline Flow (Priority: P2)

As a researcher, I want to run a complete pipeline flow (Stage 1 + Stage 2 with multiple feature sets) with a single command or configuration, so that I can process new recordings with minimal manual intervention.

**Why this priority**: Combining steps into flows improves usability, but individual stages must work first.

**Independent Test**: Can be tested by running a named flow (e.g., "set6a_full") on external file paths and verifying the Zarr contains all expected features.

**Acceptance Scenarios**:

1. **Given** a pipeline configuration specifying flow "set6a_full" with external paths to `.cmcr`/`.cmtr` files,  
   **When** I run the pipeline,  
   **Then** the system:
   - Runs Stage 1: Creates Zarr with raw data (explicit save)
   - Runs Stage 2: Adds all feature sets to Zarr (explicit save after each feature set or at end)
   - Logs progress and completion status

2. **Given** a partially completed flow (Stage 1 done, some features extracted),  
   **When** I re-run the flow,  
   **Then** the system skips already-completed steps and resumes from the first incomplete feature set.

---

### User Story 5 - Add New Feature Extractor (Priority: P3)

As a developer, I want to add a new feature extractor by creating a single Python module with a decorated class, without editing any existing code files, so that the system remains maintainable as we add ~100 features.

**Why this priority**: Extensibility is critical for long-term maintainability but requires the core system to exist first.

**Independent Test**: Can be tested by adding a minimal "example_feature" extractor and verifying it appears in the registry and writes to the correct Zarr location.

**Acceptance Scenarios**:

1. **Given** I create a new feature extractor registered with `@FeatureRegistry.register("my_new_feature")`,  
   **When** I run that extractor on a Zarr,  
   **Then** results are written to `units/{unit_id}/features/my_new_feature/` in the Zarr.

2. **Given** a registered feature extractor with declared `required_inputs`,  
   **When** I run that extractor on a Zarr missing a required input,  
   **Then** the system raises a descriptive error indicating the missing input.

---

### Edge Cases

- **Missing `.cmtr` for a recording**: System processes available `.cmcr` data, logs warning, marks artifact as incomplete for spike-based features.
- **Missing `.cmcr` for a recording**: System processes `.cmtr` data, logs warning about missing raw sensor data.
- **Mismatched `.cmcr`/`.cmtr` pair**: System detects metadata mismatch, raises error, refuses to proceed.
- **Partial metadata in recording files**: System extracts what is available, logs warnings for missing fields.
- **Very large recordings (>1 hour, >500 units)**: System handles via chunked processing.
- **Stimulus not in playlist database**: System logs warning, skips stimulus-aligned features, continues with other features.
- **Corrupt or unreadable input file**: System raises descriptive error with file path and error details.
- **External file path does not exist**: System raises clear error indicating the file was not found at the provided path.
- **External file path becomes inaccessible after Stage 1**: Stage 2 continues (only needs Zarr), but re-running Stage 1 will fail with clear error.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Input Handling

- **FR-001**: System MUST accept external file paths to `.cmcr` and/or `.cmtr` files (files are NOT stored in project folder).
- **FR-002**: System MUST accept a `dataset_id` parameter or derive it deterministically from file name/path using documented rules.
- **FR-003**: System MUST validate that paired `.cmcr` and `.cmtr` files represent the same recording session before processing.
- **FR-004**: System MUST record source file paths in Zarr metadata for provenance (but NOT copy the files).

#### Stage 1: Data Loading (cmcr/cmtr → Zarr)

- **FR-010**: System MUST produce exactly ONE Zarr archive per recording during data loading (NON-NEGOTIABLE).
- **FR-011**: The Zarr archive MUST contain per-unit spike timestamps under `units/{unit_id}/spike_times`.
- **FR-012**: The Zarr archive MUST contain stimulus timing and light reference data under `stimulus/`.
- **FR-013**: The Zarr archive MUST contain recording metadata under `metadata/`.
- **FR-014**: Per-unit QC features (e.g., quality_index) MUST be stored under `units/{unit_id}/features/` like other features.
- **FR-015**: Stage 1 MUST explicitly save the Zarr to disk before Stage 2 can proceed.
- **FR-016**: Stage 1 output MUST be a valid, self-contained Zarr readable without the original raw files.

#### Stage 2: Feature Extraction (Zarr → Zarr with features)

- **FR-020**: Feature extraction MUST read from and write back to the SAME Zarr archive.
- **FR-021**: Features MUST be stored under each unit at `units/{unit_id}/features/{feature_name}/`.
- **FR-022**: Each feature extractor MUST declare: unique name, version, required inputs, output schema.
- **FR-023**: Feature metadata (version, params_hash, timestamp) MUST be stored in feature group `.zattrs`.
- **FR-024**: Stage 2 MUST explicitly save the Zarr to disk after feature extraction.
- **FR-025**: Feature extraction MUST NOT silently overwrite prior features; explicit `force=True` required.
- **FR-026**: Feature extractors MUST be selectable by configuration (feature sets / flows).
- **FR-027**: Stage 2 MUST NOT require access to original `.cmcr`/`.cmtr` files.
- **FR-028**: Stimulus types MUST be configurable via configuration files, not hardcoded—users can add/modify stimulus definitions without code changes.

#### Optional Parquet Export

- **FR-030**: System SHOULD provide optional export of features from Zarr to Parquet format.
- **FR-031**: Parquet export MUST flatten per-unit features to tabular format for cross-recording analysis.
- **FR-032**: Parquet files are supplementary; the Zarr is the primary artifact.

#### Reproducibility and Caching

- **FR-040**: Pipeline runs MUST be reproducible: same inputs + same config produce identical outputs.
- **FR-041**: Artifacts MUST include metadata: dataset_id, params_hash, code_version, timestamp.
- **FR-042**: Pipeline MUST skip already-completed steps when appropriate (cache hit).
- **FR-043**: Random operations (e.g., shuffle tests for DSI significance) MUST use explicit seeds from config.

#### Artifact Formats (NON-NEGOTIABLE)

- **FR-050**: Nested/hierarchical recording data and features MUST use Zarr format.
- **FR-051**: Optional tabular exports for cross-recording analysis MUST use Parquet format.
- **FR-052**: Configuration files MUST use JSON format.
- **FR-053**: PKL format MUST NOT be used as a primary artifact format.

#### Independence from Legacy

- **FR-060**: New pipeline MUST NOT import from `./Legacy_code/`.
- **FR-061**: New pipeline MUST NOT modify any files in `./Legacy_code/`.
- **FR-062**: New pipeline MUST replicate the functional behavior of legacy workflows.

### Key Entities

- **Recording**: A session captured by HD-MEA, represented by external `.cmcr` and/or `.cmtr` files, identified by `dataset_id`.
- **Unit/Cell**: A single sorted neuron/cell identified within a recording, with spike times, waveform, and electrode location. All data for a unit lives under `units/{unit_id}/` in the Zarr.
- **Stimulus**: A visual stimulus presented during recording, with timing information.
- **Feature**: A quantified property of a unit's response, stored under `units/{unit_id}/features/{feature_name}/`.
- **Zarr Artifact**: The primary output—a single file containing all data and features for a recording.
- **Flow**: A named sequence of pipeline stages (Stage 1 + Stage 2 feature sets) that can be run together.

---

## Artifact Directory Conventions

### Directory Structure

```
artifacts/                          # Gitignored (local artifact storage)
└── {dataset_id}.zarr/              # ONE Zarr per recording
    ├── .zattrs                     # Root metadata (version, hash, timestamp, source paths)
    ├── units/                      # Per-unit data and features
    │   └── {unit_id}/              # One group per cell/unit
    │       ├── spike_times         # Array: spike timestamps (μs)
    │       ├── waveform            # Array: mean spike waveform
    │       ├── firing_rate_10hz    # Array: binned firing rate at 10Hz
    │       ├── .zattrs             # Unit metadata (row, col, globalID)
    │       └── features/           # All features for this unit
    │           ├── step_up_5s_5i_3x/
    │           │   ├── on_response_flag
    │           │   ├── off_response_flag
    │           │   ├── on_peak_value
    │           │   ├── ...
    │           │   └── .zattrs     # Feature metadata (version, params)
    │           ├── perfect_dense_noise_15x15_15hz_r42_3min/
    │           │   ├── sta/
    │           │   ├── gaussian_fit/
    │           │   └── .zattrs
    │           └── moving_h_bar_s5_d8_3x/
    │               ├── dsi_on
    │               ├── preferred_direction_on
    │               ├── p_value/
    │               └── .zattrs
    ├── stimulus/                   # Stimulus timing and light reference
    │   ├── light_reference/        # Light sensor traces at various rates
    │   ├── frame_time/             # Frame → timestamp mapping
    │   ├── section_time/           # Per-movie timing info
    │   └── template_auto/          # Averaged light templates
    └── metadata/                   # Recording-level metadata
        ├── acquisition_rate
        ├── recording_duration
        └── electrode_geometry

exports/                            # Optional Parquet exports (gitignored)
└── {dataset_id}_features.parquet   # Flattened feature table

config/                             # Pipeline configuration (tracked in git)
├── flows/
│   └── set6a_full.json            # Flow configuration
├── stimuli/                        # Stimulus type definitions (configurable)
│   ├── step_up_5s_5i_3x.json      # Timing, parameters for step_up stimulus
│   ├── moving_h_bar_s5_d8_3x.json # Moving bar stimulus definition
│   └── ...                         # Add new stimuli here without code changes
└── defaults.json                   # Default parameters
```

### Artifact Metadata Schema

Root `.zattrs` for Zarr:

```json
{
  "dataset_id": "JIANG009_2024-01-15",
  "hdmea_pipeline_version": "0.1.0",
  "created_at": "2025-12-14T15:30:00Z",
  "updated_at": "2025-12-14T16:45:00Z",
  "stage1_completed": true,
  "stage1_params_hash": "sha256:a1b2c3d4...",
  "source_files": {
    "cmcr_path": "//server/data/recordings/JIANG009.cmcr",
    "cmtr_path": "//server/data/recordings/JIANG009.cmtr",
    "cmcr_exists": true,
    "cmtr_exists": true
  },
  "features_extracted": ["step_up_5s_5i_3x", "perfect_dense_noise_15x15_15hz_r42_3min"],
  "config": { ... }
}
```

Feature group `.zattrs`:

```json
{
  "feature_name": "step_up_5s_5i_3x",
  "extractor_version": "1.0.0",
  "params_hash": "sha256:e5f6g7h8...",
  "extracted_at": "2025-12-14T16:45:00Z",
  "config": { ... }
}
```

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can run Stage 1 (data loading) for a typical recording (<30 min, <200 units) in under 10 minutes on standard hardware.
- **SC-002**: Users can run Stage 2 (all core features) in under 5 minutes on a loaded Zarr.
- **SC-003**: A new feature extractor can be added by a developer in under 30 minutes (coding + testing).
- **SC-004**: Pipeline produces identical Zarr outputs when run twice with same inputs and config (100% reproducibility).
- **SC-005**: Feature values match legacy workflow outputs within acceptable tolerance (e.g., DSI values differ by <0.01).
- **SC-006**: End-to-end "set6a_full" flow completes successfully on 90% of recordings that succeeded with legacy workflow.
- **SC-007**: Zarr files are readable and usable without running any pipeline code (self-describing via metadata).
- **SC-008**: No imports from `./Legacy_code/` exist in the new codebase.
- **SC-009**: Stage 2 can complete without access to the original `.cmcr`/`.cmtr` files.
- **SC-010**: Feature data for any unit can be accessed in under 1 second from the Zarr.

---

## Assumptions

1. **McsPy library available**: The `McsPy.McsCMOSMEA` library is available for reading `.cmcr`/`.cmtr` files.
2. **Stimulus database available**: A playlist/movie-length database exists for stimulus timing alignment.
3. **Single-machine processing**: v1 targets single-machine processing; distributed processing is out of scope.
4. **Python 3.10+**: The pipeline targets Python 3.10 or later.
5. **Reasonable recording sizes**: Most recordings are under 1 hour with under 500 units.
6. **External file access**: Users have read access to external `.cmcr`/`.cmtr` file paths they provide.
7. **Network paths acceptable**: External paths may be UNC paths (e.g., `//server/share/file.cmcr`).

---

## Non-Goals

1. **Do not modify legacy code**: `./Legacy_code/` is READ-ONLY reference material.
2. **Do not import from legacy code**: The new pipeline is fully independent.
3. **Do not use PKL as primary format**: PKL is forbidden for artifacts (per constitution).
4. **Do not build a GUI**: Command-line and programmatic access only in v1.
5. **Do not perform spike sorting**: Input is pre-sorted `.cmtr` files.
6. **Do not copy raw files**: Raw `.cmcr`/`.cmtr` files are not copied to project folder; only paths are recorded.
7. **Do not require raw files after Stage 1**: Once Zarr is created, raw files are not needed.

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| McsPy library incompatibility | High | Medium | Pin library version; create abstraction layer |
| Feature parity verification difficulty | Medium | High | Comprehensive test suite comparing legacy vs new outputs |
| Performance degradation on large recordings | Medium | Medium | Chunked processing; progress indicators |
| External file path inaccessible | Medium | Medium | Clear error messages; document that Stage 1 requires access |
| Zarr file corruption during write | Medium | Low | Atomic writes where possible; validation after write |
| Feature schema evolution | Low | Medium | Version all feature schemas; document migration |

---

## Compatibility with Legacy Workflows

The v1 pipeline MUST replicate the functional behavior of these legacy flows:

### Primary Reference: `set6a` workflow (Data_Processing_2025)

**Stimulus types supported** (configurable—these are examples, not hardcoded):

*Note: Stimulus types are defined in configuration files (e.g., `config/stimuli/`). New stimulus types can be added without code changes by providing timing parameters and feature mappings.*

- `baseline_127` - baseline firing statistics
- `step_up_5s_5i_3x` / `step_up_5s_5i_b0_3x` - ON/OFF response
- `green_blue_3s_3i_3x` - chromatic response
- `freq_step_5st_3x` - frequency response (chirp)
- `gratings_all_L600_S1_D8_3x` - direction/orientation selectivity (gratings)
- `moving_h_bar_s5_d8_3x` - direction selectivity (moving bar)
- `perfect_dense_noise_15x15_15hz_r42_3min` - receptive field (STA)

**Feature categories** (stored under `units/{unit_id}/features/`):
- QC features: quality index
- Baseline features: mean firing rate, std
- ON/OFF response: response flags, peak values, sustained response, response quality
- Chromatic: green/blue ON/OFF extreme values
- Frequency: amplitude, phase, R² at each test frequency
- Direction selectivity: DSI, OSI, preferred direction, p-values
- Receptive field: STA, center coordinates, Gaussian fit, LNL fit
- Cell type: RGC vs unknown (from AP propagation in network STA)

The new pipeline MUST produce equivalent feature values when given the same inputs.

---

## Clarifications

### Session 2025-12-14

- Q: How should stimulus types be handled? → A: Stimulus types must be configurable—users should be able to add/modify stimulus definitions via configuration files without code changes.
- Q: How should QC features be represented in the directory structure? → A: QC features (like quality_index) are stored like any other feature under `units/{unit_id}/features/`—no special treatment needed in examples.
