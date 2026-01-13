# Tasks: Weakly Supervised RGC Clustering with Autoencoder Features

**Input**: Design documents from `/specs/013-ae-rgc-clustering/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

**Tests**: Not explicitly requested in specification - test tasks omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- All paths relative to `dataframe_phase/classification_v2/Autoencoder_method/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project structure and configuration

- [x] T001 Create folder structure: `Autoencoder_method/`, `models/`, `models_saved/`, `results/`, `plots/`
- [x] T002 Create `Autoencoder_method/__init__.py` with package marker and version
- [x] T003 [P] Create `Autoencoder_method/models/__init__.py` exposing model classes
- [x] T004 Create `Autoencoder_method/config.py` with all configurable parameters per research.md

**Checkpoint**: Project structure ready

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Implement `data_loader.py`: `load_and_filter_data()` - load parquet, apply QI/baseline/batch filters per FR-001 to FR-004
- [x] T006 [P] Implement `data_loader.py`: `extract_trace_arrays()` - extract traces from DataFrame, compute trial means
- [x] T007 Implement `preprocessing.py`: `preprocess_segment()` - low-pass filter + downsample per segment per FR-005 to FR-009
- [x] T008 [P] Implement `preprocessing.py`: `preprocess_all_segments()` - apply preprocessing to all 10 segments
- [x] T009 [P] Implement `preprocessing.py`: `build_segment_map()` - compute start/end indices per segment
- [x] T010 Implement `grouping.py`: `assign_coarse_groups()` - precedence-based disjoint group assignment per FR-010 to FR-012
- [x] T011 [P] Implement `grouping.py`: `get_group_mask()` - boolean mask helper for group filtering

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Train Autoencoder (Priority: P1) 🎯 MVP

**Goal**: Train autoencoder to produce 49D embeddings with fixed per-segment dimensions

**Independent Test**: Train AE on data, verify embeddings are 49D with correct per-segment breakdown (4×5 + 6 + 12 + 3 + 4 + 4)

### Implementation for User Story 1

- [x] T012 [P] [US1] Implement `models/encoders.py`: `SegmentEncoder` class - 1D CNN encoder per segment
- [x] T013 [P] [US1] Implement `models/decoders.py`: `SegmentDecoder` class - 1D CNN decoder per segment
- [x] T014 [US1] Implement `models/autoencoder.py`: `MultiSegmentAutoencoder` class - compose 10 encoder/decoder pairs (depends on T012, T013)
- [x] T015 [P] [US1] Implement `models/losses.py`: `WeightedReconstructionLoss` - inverse-length weighted MSE
- [x] T016 [P] [US1] Implement `models/losses.py`: `SupervisedContrastiveLoss` - SupCon loss for group labels per FR-017
- [x] T017 [US1] Implement `models/losses.py`: `CombinedAELoss` - combine reconstruction + β×supervision (depends on T015, T016)
- [x] T018 [US1] Implement `train.py`: `train_autoencoder()` - training loop with checkpointing per FR-018, FR-019 (depends on T014, T017)
- [x] T019 [US1] Implement `embed.py`: `extract_embeddings()` - extract 49D embeddings from trained AE
- [x] T020 [US1] Implement `embed.py`: `standardize_embeddings()` - z-score standardization per FR-020
- [x] T021 [US1] Implement `evaluation.py`: `compute_silhouette_score()` - verify group separation in embeddings

**Checkpoint**: User Story 1 complete - can train AE and extract standardized 49D embeddings

---

## Phase 4: User Story 2 - Baden-Style Clustering (Priority: P1)

**Goal**: Cluster embeddings using diagonal GMM with BIC selection, constrained to groups

**Independent Test**: Provide embeddings + groups, verify per-group clustering produces group-pure clusters with BIC curves

### Implementation for User Story 2

- [x] T022 [P] [US2] Implement `clustering.py`: `fit_gmm_bic()` - fit GMMs for k range, compute BIC per FR-020, FR-023
- [x] T023 [US2] Implement `clustering.py`: `select_k_by_logbf()` - select k* via log Bayes factor threshold per FR-021
- [x] T024 [US2] Implement `clustering.py`: `cluster_per_group()` - cluster each group separately per FR-022 (depends on T022, T023)
- [x] T025 [US2] Implement `evaluation.py`: `validate_group_purity()` - verify no cluster crosses group boundaries per SC-008
- [x] T026 [US2] Implement `evaluation.py`: `get_cluster_sizes()` - compute cluster size distribution per SC-004

**Checkpoint**: User Stories 1 AND 2 complete - full AE→clustering pipeline works

---

## Phase 5: User Story 3 - Cross-Validation (Priority: P2)

**Goal**: Validate generalization via omitted-label CV turns measuring purity

**Independent Test**: Run 3 CV turns (omit each label), verify purity table with aggregate CVScore

### Implementation for User Story 3

- [x] T027 [P] [US3] Implement `evaluation.py`: `compute_purity()` - cluster purity metric per FR-025
- [x] T028 [US3] Implement `crossval.py`: `run_cv_turns()` - orchestrate 3 CV turns per FR-024, FR-026 (depends on T018, T024, T027)
- [x] T029 [US3] Implement `crossval.py`: `compute_cvscore()` - aggregate purity across turns per SC-002
- [x] T030 [US3] Implement `crossval.py`: `save_cv_results()` - save CV purity table to results/cv_purity.csv

**Checkpoint**: User Story 3 complete - CV validation produces CVScore

---

## Phase 6: User Story 4 - Bootstrap Stability (Priority: P2)

**Goal**: Assess cluster stability via 90% subsampling bootstraps

**Independent Test**: Run 20 bootstrap iterations, verify mean±SD of median correlations per group

### Implementation for User Story 4

- [x] T031 [P] [US4] Implement `stability.py`: `match_clusters_by_correlation()` - match bootstrap clusters to reference per FR-028
- [x] T032 [US4] Implement `stability.py`: `run_bootstrap_stability()` - full bootstrap testing per FR-027, FR-029 (depends on T024, T031)
- [x] T033 [US4] Implement `stability.py`: `summarize_stability()` - compute mean±SD, flag unstable per SC-003
- [x] T034 [US4] Implement `stability.py`: `save_stability_results()` - save to results/stability_metrics.json

**Checkpoint**: User Story 4 complete - stability assessment works

---

## Phase 7: User Story 5 - Visualization & Reports (Priority: P3)

**Goal**: Generate publication-ready outputs: UMAP, BIC curves, response prototypes

**Independent Test**: Run full pipeline, verify plots/ contains all expected outputs

### Implementation for User Story 5

- [x] T035 [P] [US5] Implement `visualization.py`: `plot_umap_embeddings()` - UMAP colored by group/cluster per FR-030
- [x] T036 [P] [US5] Implement `visualization.py`: `plot_bic_curve()` - BIC vs k with selected k* marked per FR-023
- [x] T037 [P] [US5] Implement `visualization.py`: `plot_response_prototypes()` - mean±SEM per cluster per segment per FR-031
- [x] T038 [P] [US5] Implement `visualization.py`: `plot_cluster_sizes()` - histogram of cluster sizes
- [x] T039 [US5] Implement `visualization.py`: `generate_all_plots()` - orchestrate all visualization outputs (depends on T035-T038)
- [x] T040 [US5] Implement `evaluation.py`: `save_embeddings()` - save to results/embeddings.parquet per FR-032
- [x] T041 [US5] Implement `evaluation.py`: `save_cluster_assignments()` - save to results/cluster_assignments.parquet per FR-032

**Checkpoint**: User Story 5 complete - all visualizations and reports generated

---

## Phase 8: Integration & Entry Point

**Purpose**: Create unified pipeline entry point

- [x] T042 Implement `run_pipeline.py`: `main()` - orchestrate full pipeline per FR-035 (depends on all US tasks)
- [x] T043 Implement `run_pipeline.py`: CLI argument parsing for input_path, output_dir overrides
- [x] T044 Implement `run_pipeline.py`: Add logging setup and progress reporting

**Checkpoint**: Full pipeline executable via single command

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T045 [P] Add docstrings to all public functions per contracts/pipeline_api.md
- [x] T046 [P] Add type hints to all functions per Python 3.11+ conventions
- [x] T047 Validate config.py parameters match research.md defaults
- [ ] T048 Run full pipeline on test data, verify all outputs generated
- [x] T049 [P] Update specs/013-ae-rgc-clustering/quickstart.md with actual usage examples
- [x] T050 [P] Create README.md in Autoencoder_method/ with installation and usage

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
     │
     ▼
Phase 2 (Foundational) ─────────────────┐
     │                                   │
     ├──────────────┬───────────────┐   │
     ▼              ▼               ▼   │
Phase 3 (US1)  Phase 4 (US2)   [wait]  │
     │              │                   │
     └──────┬───────┘                   │
            │                           │
     ┌──────┴──────┐                    │
     ▼             ▼                    │
Phase 5 (US3) Phase 6 (US4)            │
     │             │                    │
     └──────┬──────┘                    │
            ▼                           │
     Phase 7 (US5)                      │
            │                           │
            ▼                           │
     Phase 8 (Integration) ◄────────────┘
            │
            ▼
     Phase 9 (Polish)
```

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Depends on US1 (needs trained AE for embeddings)
- **User Story 3 (P2)**: Depends on US1 + US2 (needs training + clustering)
- **User Story 4 (P2)**: Depends on US2 (needs clustering results)
- **User Story 5 (P3)**: Depends on US1 + US2 (needs embeddings + clusters)

### Within Each User Story

- Models before services
- Encoders/decoders before autoencoder
- Losses before training
- Training before embedding extraction
- Clustering before validation

### Parallel Opportunities

**Phase 2 (Foundational)**:
- T006, T008, T009, T011 can run in parallel

**Phase 3 (US1)**:
- T012, T013 can run in parallel (encoder/decoder)
- T015, T016 can run in parallel (loss components)

**Phase 4-6**:
- T022, T027, T031 can run in parallel (independent utilities)

**Phase 7 (US5)**:
- T035, T036, T037, T038 can all run in parallel (independent plots)

---

## Parallel Example: User Story 1

```bash
# Launch encoders and decoders together:
Task: T012 "Implement SegmentEncoder in models/encoders.py"
Task: T013 "Implement SegmentDecoder in models/decoders.py"

# Then autoencoder (depends on above):
Task: T014 "Implement MultiSegmentAutoencoder in models/autoencoder.py"

# Launch loss components together:
Task: T015 "Implement WeightedReconstructionLoss in models/losses.py"
Task: T016 "Implement SupervisedContrastiveLoss in models/losses.py"

# Then combined loss and training:
Task: T017 "Implement CombinedAELoss in models/losses.py"
Task: T018 "Implement train_autoencoder in train.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (AE training + embedding)
4. Complete Phase 4: User Story 2 (Clustering)
5. **STOP and VALIDATE**: Test AE→clustering pipeline independently
6. Can produce cluster assignments at this point

### Incremental Delivery

1. **MVP (US1 + US2)**: Train AE → Extract embeddings → Cluster → Output assignments
2. **Add US3**: CV validation → CVScore metric for hyperparameter tuning
3. **Add US4**: Stability testing → Confidence in clusters
4. **Add US5**: Visualizations → Publication-ready outputs
5. Each story adds value without breaking previous stories

### Single Developer Strategy

Execute phases sequentially:
1. Setup + Foundational (T001-T011)
2. US1 (T012-T021) → Verify embeddings
3. US2 (T022-T026) → Verify clustering
4. US3 (T027-T030) → Verify CV
5. US4 (T031-T034) → Verify stability
6. US5 (T035-T041) → Verify plots
7. Integration (T042-T044)
8. Polish (T045-T050)

---

## Task Summary

| Phase | Story | Task Count | Parallel Tasks |
|-------|-------|------------|----------------|
| Phase 1: Setup | - | 4 | 1 |
| Phase 2: Foundational | - | 7 | 4 |
| Phase 3: US1 | US1 | 10 | 4 |
| Phase 4: US2 | US2 | 5 | 1 |
| Phase 5: US3 | US3 | 4 | 1 |
| Phase 6: US4 | US4 | 4 | 1 |
| Phase 7: US5 | US5 | 7 | 4 |
| Phase 8: Integration | - | 3 | 0 |
| Phase 9: Polish | - | 6 | 4 |
| **Total** | | **50** | **20** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All file paths relative to `dataframe_phase/classification_v2/Autoencoder_method/`
