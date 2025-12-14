# Tasks: HD-MEA Data Analysis Pipeline v1

**Input**: Design documents from `/specs/001-hdmea-modular-pipeline/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Minimal tests included per constitution (smoke tests, unit tests for feature extractors, integration test).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

---

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create project structure and install dependencies

- [x] T001 Create project structure with `src/hdmea/`, `tests/`, `config/`, `notebooks/` directories
- [x] T002 Create `pyproject.toml` with package metadata, dependencies (zarr, pyarrow, pandas, numpy, scipy, scikit-image, pydantic), and dev extras (pytest)
- [x] T003 [P] Create `.gitignore` with `artifacts/`, `exports/`, `notebooks/_scratch/`, `*.pyc`, `__pycache__/`
- [x] T004 [P] Create `src/hdmea/__init__.py` with package version `__version__ = "0.1.0"`
- [x] T005 [P] Create empty `__init__.py` files in all subpackages: `io/`, `preprocess/`, `features/`, `analysis/`, `viz/`, `pipeline/`, `utils/`
- [x] T006 [P] Create `tests/conftest.py` with pytest configuration
- [x] T007 [P] Create `config/defaults.json` with default pipeline parameters

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Utilities Module (leaf dependency)

- [x] T008 [P] Implement logging setup in `src/hdmea/utils/logging.py` (per constitution: `logging.getLogger(__name__)`)
- [x] T009 [P] Implement config hashing in `src/hdmea/utils/hashing.py` (SHA256 of JSON-serialized config)
- [x] T010 [P] Implement input validation in `src/hdmea/utils/validation.py` (path validation, dataset_id regex)

### Exceptions Module

- [x] T011 Implement exception hierarchy in `src/hdmea/utils/exceptions.py` (HDMEAError, DataLoadError, FeatureExtractionError, MissingInputError, ConfigurationError, CacheConflictError)

### Feature Registry (required for US2, US4, US5)

- [x] T012 Implement `FeatureExtractor` base class in `src/hdmea/features/base.py` (name, version, required_inputs, output_schema, runtime_class, extract method)
- [x] T013 Implement `FeatureRegistry` class in `src/hdmea/features/registry.py` (register decorator, get, list_all, get_metadata)

### Configuration Loading

- [x] T014 Implement config loading with Pydantic validation in `src/hdmea/pipeline/config.py` (FlowConfig, StimulusConfig, load_flow_config, load_stimulus_config)

### Smoke Tests

- [x] T015 [P] Create smoke tests for all modules in `tests/smoke/test_imports.py` (verify all imports work)

**Checkpoint**: Foundation ready - user story implementation can begin

---

## Phase 3: User Story 1 - Load Recording to Zarr (Priority: P1) 🎯 MVP

**Goal**: Load external `.cmcr`/`.cmtr` files and produce single Zarr artifact per recording

**Independent Test**: Provide external paths to sample files, verify Zarr contains units/, stimulus/, metadata/

### Implementation for User Story 1

- [x] T016 [P] [US1] Implement CMTR file reading via McsPy in `src/hdmea/io/cmtr.py` (load_cmtr_data function)
- [x] T017 [P] [US1] Implement CMCR file reading via McsPy in `src/hdmea/io/cmcr.py` (load_cmcr_data, extract light reference)
- [x] T018 [US1] Implement Zarr store operations in `src/hdmea/io/zarr_store.py` (create_recording_zarr, write_units, write_stimulus, write_metadata, read_recording_zarr)
- [x] T019 [US1] Implement data loading pipeline in `src/hdmea/pipeline/runner.py` (load_recording function per contracts/pipeline_api.py)
- [x] T020 [US1] Implement caching logic for Stage 1 in `src/hdmea/pipeline/runner.py` (check existing Zarr, skip if params match)
- [x] T021 [US1] Add validation for CMCR/CMTR pair matching in `src/hdmea/io/cmtr.py` (raise MismatchError if metadata differs)
- [x] T022 [US1] Implement firing rate calculation (10Hz binning) in `src/hdmea/preprocess/filtering.py`
- [x] T023 [US1] Create sample stimulus config in `config/stimuli/step_up_5s_5i_3x.json`

### Tests for User Story 1

- [x] T024 [P] [US1] Create synthetic spike data fixtures in `tests/fixtures/synthetic_spikes.py`
- [x] T025 [P] [US1] Create minimal Zarr fixture in `tests/fixtures/minimal_zarr/`
- [x] T026 [US1] Unit test for CMTR loading in `tests/unit/test_cmtr.py` (mock McsPy)
- [x] T027 [US1] Unit test for Zarr store operations in `tests/unit/test_zarr_store.py`
- [x] T028 [US1] Integration test for load_recording in `tests/integration/test_load_recording.py`

**Checkpoint**: Stage 1 (Data Loading) fully functional and independently testable

---

## Phase 4: User Story 2 - Extract Features to Zarr (Priority: P1)

**Goal**: Run feature extraction on Zarr, write features back under each unit

**Independent Test**: Provide Stage-1 Zarr, verify features appear under units/{unit_id}/features/

### Core Feature Extraction Infrastructure

- [x] T029 [US2] Implement feature extraction runner in `src/hdmea/pipeline/runner.py` (extract_features function per contracts/pipeline_api.py)
- [x] T030 [US2] Implement feature writing to Zarr in `src/hdmea/io/zarr_store.py` (write_feature_to_unit, write_feature_metadata)
- [x] T031 [US2] Implement feature caching logic in `src/hdmea/pipeline/runner.py` (check existing feature, skip if version matches)

### ON/OFF Response Feature Extractor (step_up stimulus)

- [x] T032 [P] [US2] Create `src/hdmea/features/on_off/__init__.py`
- [x] T033 [US2] Implement ON/OFF extractor in `src/hdmea/features/on_off/step_up.py` (@FeatureRegistry.register("step_up_5s_5i_3x"))
- [x] T034 [US2] Unit test for ON/OFF extractor in `tests/unit/test_on_off.py` (synthetic spikes with known response)

### Baseline Feature Extractor

- [x] T035 [P] [US2] Create `src/hdmea/features/baseline/__init__.py`
- [x] T036 [US2] Implement baseline extractor in `src/hdmea/features/baseline/baseline_127.py` (@FeatureRegistry.register("baseline_127"))
- [x] T037 [US2] Unit test for baseline extractor in `tests/unit/test_baseline.py`

### Direction Selectivity Feature Extractor (moving bar stimulus)

- [x] T038 [P] [US2] Create `src/hdmea/features/direction/__init__.py`
- [x] T039 [US2] Implement DSI/OSI extractor in `src/hdmea/features/direction/moving_bar.py` (@FeatureRegistry.register("moving_h_bar_s5_d8_3x"))
- [x] T040 [US2] Add shuffle test for DSI significance (explicit random seed) in `src/hdmea/features/direction/moving_bar.py`
- [x] T041 [US2] Unit test for DSI extractor in `tests/unit/test_direction.py`
- [x] T042 [P] [US2] Create stimulus config `config/stimuli/moving_h_bar_s5_d8_3x.json`

### Receptive Field Feature Extractor (dense noise stimulus)

- [x] T043 [P] [US2] Create `src/hdmea/features/receptive_field/__init__.py`
- [x] T044 [US2] Implement STA extractor in `src/hdmea/features/receptive_field/dense_noise.py` (@FeatureRegistry.register("perfect_dense_noise_15x15_15hz_r42_3min"))
- [x] T045 [US2] Implement Gaussian fitting for RF in `src/hdmea/features/receptive_field/gaussian_fit.py`
- [x] T046 [US2] Unit test for STA extractor in `tests/unit/test_receptive_field.py`
- [x] T047 [P] [US2] Create stimulus config `config/stimuli/perfect_dense_noise_15x15_15hz_r42_3min.json`

### Additional Feature Extractors (core set6a workflow)

- [x] T048 [P] [US2] Create `src/hdmea/features/chromatic/__init__.py`
- [x] T049 [US2] Implement chromatic extractor in `src/hdmea/features/chromatic/green_blue.py` (@FeatureRegistry.register("green_blue_3s_3i_3x"))
- [x] T050 [P] [US2] Create stimulus config `config/stimuli/green_blue_3s_3i_3x.json`

- [x] T051 [P] [US2] Create `src/hdmea/features/frequency/__init__.py`
- [x] T052 [US2] Implement frequency response extractor in `src/hdmea/features/frequency/chirp.py` (@FeatureRegistry.register("freq_step_5st_3x"))
- [x] T053 [P] [US2] Create stimulus config `config/stimuli/freq_step_5st_3x.json`

- [x] T054 [P] [US2] Create `src/hdmea/features/cell_type/__init__.py`
- [x] T055 [US2] Implement cell type classifier in `src/hdmea/features/cell_type/rgc_classifier.py` (RGC vs unknown based on network STA)

### Test for Feature Registry

- [x] T056 [US2] Unit test for FeatureRegistry in `tests/unit/test_registry.py` (register, get, list_all, duplicate registration error)

**Checkpoint**: Stage 2 (Feature Extraction) fully functional with core extractors

---

## Phase 5: User Story 4 - Run End-to-End Pipeline Flow (Priority: P2)

**Goal**: Run complete flow (Stage 1 + Stage 2) with single command via configuration

**Independent Test**: Run "set6a_full" flow on external paths, verify Zarr contains all expected features

### Implementation for User Story 4

- [x] T057 [US4] Implement flow runner in `src/hdmea/pipeline/flows.py` (load flow config, execute stages in order)
- [x] T058 [US4] Implement run_flow function in `src/hdmea/pipeline/runner.py` (per contracts/pipeline_api.py)
- [x] T059 [US4] Implement incremental resume logic in `src/hdmea/pipeline/runner.py` (skip completed stages/features)
- [x] T060 [US4] Create set6a_full flow config in `config/flows/set6a_full.json`

### Test for User Story 4

- [x] T061 [US4] Integration test for run_flow in `tests/integration/test_pipeline.py` (end-to-end with synthetic data)

**Checkpoint**: Full pipeline flows functional

---

## Phase 6: User Story 3 - Export Features to Parquet (Priority: P3)

**Goal**: Export features from Zarr to Parquet for cross-recording analysis

**Independent Test**: Export from Zarr with features, verify Parquet contains flattened columns

### Implementation for User Story 3

- [x] T062 [US3] Implement Parquet export in `src/hdmea/io/parquet_export.py` (export_features_to_parquet per contracts/pipeline_api.py)
- [x] T063 [US3] Implement feature flattening logic (Zarr nested → Parquet columns) in `src/hdmea/io/parquet_export.py`
- [x] T064 [US3] Add export metadata (_export_version, _export_timestamp) in `src/hdmea/io/parquet_export.py`

### Test for User Story 3

- [x] T065 [US3] Unit test for Parquet export in `tests/unit/test_parquet_export.py`

**Checkpoint**: Parquet export functional

---

## Phase 7: User Story 5 - Add New Feature Extractor (Priority: P3)

**Goal**: Enable adding new extractors without editing existing code

**Independent Test**: Add example_feature extractor, verify it appears in registry and writes to correct location

### Implementation for User Story 5

- [x] T066 [US5] Create example feature extractor in `src/hdmea/features/example/example_feature.py` (template for new extractors)
- [x] T067 [US5] Document extractor creation process in `src/hdmea/features/README.md`
- [x] T068 [US5] Implement auto-discovery of extractors in `src/hdmea/features/__init__.py` (import all subpackages to trigger registration)

### Test for User Story 5

- [x] T069 [US5] Test new extractor registration in `tests/unit/test_extensibility.py`

**Checkpoint**: Extensibility validated

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple user stories

- [x] T070 [P] Create notebook directory structure in `notebooks/` (dev/, validation/, reports/, templates/, flows/, _scratch/)
- [x] T071 [P] Configure Jupytext pairing in `pyproject.toml` (formats = "ipynb,py:percent")
- [x] T072 [P] Create sample validation notebook in `notebooks/validation/verify_stage1.py`
- [x] T073 [P] Create sample validation notebook in `notebooks/validation/verify_stage2.py`
- [x] T074 [P] Add CLI entry point in `pyproject.toml` [project.scripts] section (optional)
- [x] T075 Run quickstart.md validation (manual verification of getting started guide)
- [x] T076 Verify no imports from Legacy_code/ in codebase (SC-008)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
     │
     ▼
Phase 2 (Foundational) ─── BLOCKS ALL USER STORIES
     │
     ├─────────────────────────┬───────────────────────────┐
     ▼                         ▼                           ▼
Phase 3 (US1: Load)     Phase 4 (US2: Extract)        (can parallel)
     │                         │
     │                         │
     ▼                         ▼
Phase 5 (US4: Flow) ◄──────────┤ (depends on US1 + US2)
     │                         │
     ├─────────────────────────┤
     ▼                         ▼
Phase 6 (US3: Export)    Phase 7 (US5: Extend)
     │                         │
     └─────────────────────────┘
                │
                ▼
        Phase 8 (Polish)
```

### User Story Dependencies

| Story | Depends On | Can Start After |
|-------|------------|-----------------|
| US1 (Load) | Foundational | Phase 2 complete |
| US2 (Extract) | Foundational + FeatureRegistry | Phase 2 complete (can parallel with US1) |
| US4 (Flow) | US1 + US2 | Both US1 and US2 complete |
| US3 (Export) | US2 (needs features in Zarr) | US2 complete |
| US5 (Extend) | US2 (needs registry working) | US2 complete |

### Within Each User Story

1. Infrastructure/utilities first
2. Core implementation
3. Tests
4. Validation

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005, T006, T007 can all run in parallel

**Phase 2 (Foundational)**:
- T008, T009, T010 can run in parallel
- T015 can run in parallel after T004, T005

**Phase 3 (US1)**:
- T016, T017 can run in parallel (different files)
- T024, T025 can run in parallel

**Phase 4 (US2)**:
- Feature subpackage creation (T032, T035, T038, T043, T048, T051, T054) can all run in parallel
- Stimulus configs (T023, T042, T047, T050, T053) can all run in parallel

---

## Parallel Example: User Story 2 Feature Extractors

```bash
# Launch all feature subpackage __init__.py creations together:
T032: src/hdmea/features/on_off/__init__.py
T035: src/hdmea/features/baseline/__init__.py
T038: src/hdmea/features/direction/__init__.py
T043: src/hdmea/features/receptive_field/__init__.py
T048: src/hdmea/features/chromatic/__init__.py
T051: src/hdmea/features/frequency/__init__.py
T054: src/hdmea/features/cell_type/__init__.py

# Launch all stimulus configs together:
T023: config/stimuli/step_up_5s_5i_3x.json
T042: config/stimuli/moving_h_bar_s5_d8_3x.json
T047: config/stimuli/perfect_dense_noise_15x15_15hz_r42_3min.json
T050: config/stimuli/green_blue_3s_3i_3x.json
T053: config/stimuli/freq_step_5st_3x.json
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup ✓
2. Complete Phase 2: Foundational ✓
3. Complete Phase 3: User Story 1 (Load Recording) ✓
4. **VALIDATE**: Test Stage 1 independently
5. Complete Phase 4: User Story 2 (Extract Features) ✓
6. **VALIDATE**: Test Stage 2 independently
7. **MVP COMPLETE**: Can load recordings and extract core features

### Incremental Delivery

| Milestone | Stories | Value Delivered |
|-----------|---------|-----------------|
| MVP | US1 + US2 | Load data, extract features to Zarr |
| v1.0 | + US4 | End-to-end flows with single command |
| v1.1 | + US3 | Cross-recording Parquet export |
| v1.2 | + US5 | Documented extensibility for new features |

### Parallel Team Strategy

With multiple developers after Phase 2:

- **Developer A**: US1 (Load Recording)
- **Developer B**: US2 core (Registry, Runner)
- **Developer C**: US2 extractors (ON/OFF, DSI, STA)

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 76 |
| Phase 1 (Setup) | 7 tasks |
| Phase 2 (Foundational) | 8 tasks |
| Phase 3 (US1: Load) | 13 tasks |
| Phase 4 (US2: Extract) | 28 tasks |
| Phase 5 (US4: Flow) | 5 tasks |
| Phase 6 (US3: Export) | 4 tasks |
| Phase 7 (US5: Extend) | 4 tasks |
| Phase 8 (Polish) | 7 tasks |
| Parallelizable | 35 tasks marked [P] |
| MVP Scope | US1 + US2 (48 tasks) |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [USn] label maps task to specific user story for traceability
- Feature extractors use legacy code as REFERENCE ONLY (no imports)
- All tests use synthetic data, not real recordings
- Commit after each task or logical group
- Stop at any checkpoint to validate independently

