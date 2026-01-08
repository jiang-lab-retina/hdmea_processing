# Tasks: Baden-Method RGC Clustering Pipeline

**Input**: Design documents from `/specs/001-baden-rgc-clustering/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Tests**: Not explicitly requested in specification - skipping test tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Output directory**: `dataframe_phase/classification_v2/Baden_method/`
- **All source files**: Within the output directory above

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create directory structure and initialize Python package

- [X] T001 Create directory structure: `dataframe_phase/classification_v2/Baden_method/` with subdirs `models/`, `plots/`, `results/`
- [X] T002 Create package marker `dataframe_phase/classification_v2/Baden_method/__init__.py` with version and module exports
- [X] T003 [P] Create configuration module `dataframe_phase/classification_v2/Baden_method/config.py` with all constants from research.md

### Config Module Contents (T003)
- Default paths (INPUT_PATH, OUTPUT_DIR)
- Column names (CHIRP_COL, COLOR_COL, BAR_COLS, RF_COL, QI_COL, DS_PVAL_COL, AXON_COL)
- Filter thresholds (QI_THRESHOLD=0.5, DS_P_THRESHOLD=0.05)
- Signal processing (SAMPLING_RATE=60, LOWPASS_CUTOFF=10, BASELINE_SAMPLES=8)
- Sparse PCA specs (component counts and top-k per stimulus)
- GMM parameters (N_INIT=20, REG_COVAR=1e-5, K_MAX_DS=30, K_MAX_NDS=60)
- Bootstrap parameters (N_ITERATIONS=20, SAMPLE_FRACTION=0.9)

**Checkpoint**: Project structure ready

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Preprocessing module that ALL subsequent phases depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement `load_data(path)` function in `dataframe_phase/classification_v2/Baden_method/preprocessing.py`
- [X] T005 Implement `filter_rows(df, qi_threshold, required_columns)` for NaN, QI, axon_type filtering in `preprocessing.py`
- [X] T006 Implement `split_ds_nds(df, p_threshold)` to split by ds_p_value in `preprocessing.py`
- [X] T007 Implement `lowpass_filter(trace, fs, cutoff, order)` using scipy sosfiltfilt in `preprocessing.py`
- [X] T008 Implement `baseline_subtract(trace, n_samples)` in `preprocessing.py`
- [X] T009 Implement `normalize_trace(trace, eps)` with max-absolute normalization in `preprocessing.py`
- [X] T010 Implement `preprocess_traces(df)` that applies filter, baseline, normalize to all traces in `preprocessing.py`
- [X] T011 Add logging with `logging.getLogger(__name__)` throughout `preprocessing.py`

**Checkpoint**: Foundation ready - feature extraction can now begin

---

## Phase 3: User Story 3 - Feature Extraction with Sparse PCA (Priority: P2)

**Goal**: Extract standardized 40-dimensional feature vector from stimulus responses using sparse PCA with enforced sparsity constraints

**Independent Test**: Verify feature matrix has shape (N, 40) and each sparse PCA component has exactly the specified number of non-zero weights

### Implementation for User Story 3

- [X] T012 [P] [US3] Implement `fit_sparse_pca(X, n_components, top_k, alpha)` with hard sparsity enforcement in `dataframe_phase/classification_v2/Baden_method/features.py`
- [X] T013 [P] [US3] Implement helper `_enforce_hard_sparsity(components, top_k)` to threshold and renormalize components in `features.py`
- [X] T014 [US3] Implement `extract_chirp_features(df)` returning 20 features with 10 non-zero bins each in `features.py`
- [X] T015 [US3] Implement `extract_color_features(df)` returning 6 features with 10 non-zero bins each in `features.py`
- [X] T016 [US3] Implement `extract_bar_svd(cell_traces_dict)` for SVD time course extraction from 8-direction matrix in `features.py`
- [X] T017 [US3] Implement `extract_bar_features(df)` returning 8 TC features (5 bins) + 4 derivative features (6 bins) in `features.py`
- [X] T018 [US3] Implement `extract_rf_features(df)` returning 2 regular PCA features from sta_time_course in `features.py`
- [X] T019 [US3] Implement `build_feature_matrix(df, return_models)` concatenating all features into 40D matrix in `features.py`
- [X] T020 [US3] Implement `standardize_features(X)` returning z-scored features and fitted StandardScaler in `features.py`
- [X] T021 [US3] Add logging and validation (verify 40 features, no NaN) in `features.py`

**Checkpoint**: Feature extraction module complete - clustering can begin

---

## Phase 4: User Story 4 - Automatic Cluster Number Selection via BIC (Priority: P2)

**Goal**: Automatically determine optimal number of clusters using BIC, with diagonal GMM and multiple restarts

**Independent Test**: Verify BIC values computed for k=1 to k_max, optimal k is minimum BIC, log Bayes factors computed

### Implementation for User Story 4

- [X] T022 [P] [US4] Implement `fit_gmm(X, k, n_init, reg_covar, random_state)` fitting diagonal GMM in `dataframe_phase/classification_v2/Baden_method/clustering.py`
- [X] T023 [US4] Implement `fit_gmm_bic(X, k_grid, n_init, reg_covar, random_state)` returning BIC table for all k values in `clustering.py`
- [X] T024 [US4] Implement `select_optimal_k(bic_table)` finding k with minimum BIC in `clustering.py`
- [X] T025 [US4] Implement `compute_log_bayes_factors(bic_table)` computing LBF between adjacent k values in `clustering.py`
- [X] T026 [US4] Implement `fit_final_gmm(X, k, n_init, reg_covar, random_state)` for final model fitting in `clustering.py`
- [X] T027 [US4] Implement `predict_clusters(gmm, X)` returning labels and posterior probabilities in `clustering.py`
- [X] T028 [US4] Add progress logging with tqdm for BIC grid search in `clustering.py`

**Checkpoint**: Clustering module complete - evaluation can begin

---

## Phase 5: User Story 5 - Cluster Quality Evaluation (Priority: P3)

**Goal**: Evaluate cluster quality using posterior probability separability curves and bootstrap stability analysis

**Independent Test**: Verify posterior curves generated per cluster, bootstrap returns median correlation ≥ 0.90 for stable clusters

### Implementation for User Story 5

- [X] T029 [P] [US5] Implement `compute_posterior_curves(labels, posteriors)` returning rank-ordered curves per cluster in `dataframe_phase/classification_v2/Baden_method/evaluation.py`
- [X] T030 [US5] Implement `match_cluster_centers(original_means, bootstrap_means)` using maximum correlation matching in `evaluation.py`
- [X] T031 [US5] Implement `bootstrap_stability(X, k, n_iter, frac, random_state)` with resample-refit-match-correlate logic in `evaluation.py`
- [X] T032 [US5] Add stability threshold check and warning if median_corr < 0.90 in `evaluation.py`
- [X] T033 [US5] Add progress logging with tqdm for bootstrap iterations in `evaluation.py`

**Checkpoint**: Evaluation module complete - visualization can begin

---

## Phase 6: Visualization Module

**Goal**: Generate and save all required plots (BIC curves, posterior curves, UMAP projections)

**Independent Test**: Verify all 6 plot files are generated (2 populations × 3 plot types)

### Implementation for Visualization

- [X] T034 [P] Implement `plot_bic_curve(bic_table, population_name, save_path, ax)` in `dataframe_phase/classification_v2/Baden_method/visualization.py`
- [X] T035 [P] Implement `plot_posterior_curves(curves, population_name, save_path, ax)` in `visualization.py`
- [X] T036 [P] Implement `compute_umap_projection(X, n_neighbors, min_dist, random_state)` in `visualization.py`
- [X] T037 Implement `plot_umap_clusters(X, labels, population_name, save_path, ax)` in `visualization.py`
- [X] T038 Configure matplotlib style defaults for publication-quality plots in `visualization.py`

**Checkpoint**: Visualization module complete - pipeline orchestration can begin

---

## Phase 7: User Story 1 + 2 - End-to-End Pipeline with DS/Non-DS Split (Priority: P1) 🎯 MVP

**Goal**: Complete pipeline that loads data, processes DS and non-DS populations independently, and saves all outputs

**Independent Test**: Run `run_baden_pipeline()` on input parquet, verify outputs: clustering_results.parquet, BIC tables, models, plots, stability_metrics.json

### Implementation for User Story 1 & 2

- [X] T039 [US1] Create pipeline module skeleton with `run_baden_pipeline()` signature in `dataframe_phase/classification_v2/Baden_method/pipeline.py`
- [X] T040 [US1] Implement data loading and preprocessing orchestration in `pipeline.py`
- [X] T041 [US2] Implement DS/non-DS population split and separate processing loops in `pipeline.py`
- [X] T042 [US1] Implement feature extraction calls for both populations in `pipeline.py`
- [X] T043 [US1] Implement clustering with BIC selection for both populations in `pipeline.py`
- [X] T044 [US1] Implement quality evaluation for both populations in `pipeline.py`
- [X] T045 [US1] Implement visualization generation for both populations in `pipeline.py`
- [X] T046 [US1] Implement results aggregation into output DataFrame in `pipeline.py`
- [X] T047 [US1] Implement `save_results(results_df, output_dir)` saving clustering_results.parquet in `pipeline.py`
- [X] T048 [US1] Implement `save_models(models_dict, output_dir)` saving all GMM/PCA/scaler models with joblib in `pipeline.py`
- [X] T049 [US1] Implement `save_bic_tables(bic_ds, bic_nds, output_dir)` saving BIC data in `pipeline.py`
- [X] T050 [US1] Implement `save_stability_metrics(metrics_dict, output_dir)` saving JSON with stability results in `pipeline.py`
- [X] T051 [US1] Add comprehensive logging with progress updates throughout pipeline in `pipeline.py`
- [X] T052 [US1] Add error handling with informative messages for edge cases in `pipeline.py`
- [X] T053 [US1] Add random seed propagation for reproducibility in `pipeline.py`

**Checkpoint**: Full pipeline functional - both US1 and US2 complete

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, edge case handling, and documentation

- [ ] T054 [P] Run end-to-end pipeline on actual input data and verify all outputs generated
- [ ] T055 [P] Verify output parquet has expected columns per data-model.md schema
- [ ] T056 [P] Verify all 6 plots saved to plots/ directory
- [ ] T057 Verify reproducibility: run pipeline twice with same seed, compare outputs
- [X] T058 Add docstrings to all public functions following Google style
- [X] T059 [P] Update `__init__.py` with clean public API exports
- [ ] T060 Run quickstart.md examples and verify they work

### Validation Scripts Created
- `validate_pipeline.py`: Standalone validation (no imports, tests data loading and filtering)
- `test_run.py`: Full pipeline test with progress tracking

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundational - preprocessing.py)
    ↓
    ├── Phase 3 (US3 - features.py) ──────────────────┐
    │       ↓                                          │
    │   Phase 4 (US4 - clustering.py)                  │
    │       ↓                                          │
    │   Phase 5 (US5 - evaluation.py)                  │
    │       ↓                                          │
    └── Phase 6 (visualization.py) ←──────────────────┘
            ↓
    Phase 7 (US1+US2 - pipeline.py)
            ↓
    Phase 8 (Polish)
```

### User Story Dependencies

| Story | Depends On | Files |
|-------|------------|-------|
| US3 (Features) | Phase 2 (preprocessing) | features.py |
| US4 (Clustering) | US3 (features) | clustering.py |
| US5 (Evaluation) | US4 (clustering) | evaluation.py |
| US1/US2 (Pipeline) | US3, US4, US5, visualization | pipeline.py |

### Within Each Phase

- Functions can be implemented in parallel if marked [P]
- Functions depending on other functions must wait

### Parallel Opportunities

**Phase 3 (Features)**:
```bash
# Can run in parallel:
T012: fit_sparse_pca
T013: _enforce_hard_sparsity
# Then sequential: T014-T021
```

**Phase 6 (Visualization)**:
```bash
# Can run in parallel:
T034: plot_bic_curve
T035: plot_posterior_curves  
T036: compute_umap_projection
# Then: T037, T038
```

**Phase 8 (Polish)**:
```bash
# Can run in parallel:
T054: Run pipeline
T055: Verify parquet
T056: Verify plots
```

---

## Parallel Example: Feature Extraction (Phase 3)

```bash
# Launch sparse PCA helpers in parallel:
Task T012: "Implement fit_sparse_pca in features.py"
Task T013: "Implement _enforce_hard_sparsity in features.py"

# Then feature extraction functions (sequential - depend on helpers):
Task T014: "Implement extract_chirp_features in features.py"
Task T015: "Implement extract_color_features in features.py"
# etc.
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational preprocessing (T004-T011)
3. Complete Phase 3: Feature extraction (T012-T021)
4. Complete Phase 4: Clustering (T022-T028)
5. Complete Phase 5: Evaluation (T029-T033)
6. Complete Phase 6: Visualization (T034-T038)
7. Complete Phase 7: Pipeline orchestration (T039-T053)
8. **STOP and VALIDATE**: Run full pipeline, verify outputs
9. Complete Phase 8: Polish (T054-T060)

### Incremental Delivery

Each phase after Foundational can be validated independently:
- After Phase 3: Can extract 40D features from preprocessed data
- After Phase 4: Can cluster and select optimal k
- After Phase 5: Can evaluate cluster quality
- After Phase 6: Can generate visualizations
- After Phase 7: Full end-to-end pipeline works

### Single Developer Strategy

1. Work through phases sequentially
2. Validate each module before moving on
3. Integration in Phase 7 ties everything together

---

## Task Summary

| Phase | Tasks | Parallel | Story |
|-------|-------|----------|-------|
| 1. Setup | T001-T003 | 1 | - |
| 2. Foundational | T004-T011 | 0 | - |
| 3. Feature Extraction | T012-T021 | 2 | US3 |
| 4. Clustering | T022-T028 | 1 | US4 |
| 5. Evaluation | T029-T033 | 1 | US5 |
| 6. Visualization | T034-T038 | 3 | - |
| 7. Pipeline | T039-T053 | 0 | US1, US2 |
| 8. Polish | T054-T060 | 3 | - |
| **Total** | **60 tasks** | **11 parallel** | **5 stories** |

---

## Notes

- [P] tasks = different files or independent functions, no dependencies
- [Story] label maps task to specific user story for traceability
- No test tasks generated (tests not explicitly requested in spec)
- US1 and US2 combined in Phase 7 as they share the pipeline module
- Commit after each task or logical group
- Stop at any checkpoint to validate incrementally

