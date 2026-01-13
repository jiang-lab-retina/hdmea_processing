# Tasks: DEC-Refined RGC Subtype Clustering

**Input**: Design documents from `/specs/014-dec-rgc-clustering/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested. Minimal validation included in Polish phase.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Source**: `dataframe_phase/classification_v2/divide_conquer_method/`
- **Reference**: `dataframe_phase/classification_v2/Autoencoder_method/` (for code reuse)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and package structure

- [ ] T001 Create package directory structure in `dataframe_phase/classification_v2/divide_conquer_method/`
- [ ] T002 Create `__init__.py` with package exports in `divide_conquer_method/__init__.py`
- [ ] T003 [P] Create `models/__init__.py` subpackage init
- [ ] T004 [P] Create `clustering/__init__.py` subpackage init
- [ ] T005 [P] Create `validation/__init__.py` subpackage init
- [ ] T006 Create output directories: `results/`, `plots/`, `models_saved/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Configuration

- [ ] T007 Create `config.py` with all parameters (DS/OS thresholds, K_MAX, DEC settings) in `divide_conquer_method/config.py`

### Data Loading (Reuse + Modify)

- [ ] T008 Copy and adapt `data_loader.py` from Autoencoder_method to `divide_conquer_method/data_loader.py` - add RGC-only filter
- [ ] T009 Create `grouping.py` with DS > OS priority assignment in `divide_conquer_method/grouping.py`

### Preprocessing (Full Reuse)

- [ ] T010 Copy `preprocessing.py` from Autoencoder_method to `divide_conquer_method/preprocessing.py` - no modifications needed

### Model Architecture (Full Reuse)

- [ ] T011 [P] Copy `models/encoders.py` from Autoencoder_method to `divide_conquer_method/models/encoders.py`
- [ ] T012 [P] Copy `models/decoders.py` from Autoencoder_method to `divide_conquer_method/models/decoders.py`
- [ ] T013 [P] Copy and simplify `models/autoencoder.py` from Autoencoder_method to `divide_conquer_method/models/autoencoder.py` - remove SupCon/purity integration

### Training (Simplify)

- [ ] T014 Create simplified `train.py` with reconstruction-only loss in `divide_conquer_method/train.py`

### Embedding Extraction (Reuse)

- [ ] T015 Copy `embed.py` from Autoencoder_method to `divide_conquer_method/embed.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Run Complete Pipeline for One Group (Priority: P1) 🎯 MVP

**Goal**: Run full pipeline on one group (e.g., DSGC) with AE training, GMM k-selection, DEC refinement, and ipRGC validation

**Independent Test**: `python -m divide_conquer_method.run_pipeline --group DSGC --subset 500`

### GMM/BIC Clustering

- [ ] T016 [P] [US1] Copy `models/gpu_gmm.py` from Autoencoder_method to `divide_conquer_method/models/gpu_gmm.py`
- [ ] T017 [US1] Create `clustering/gmm_bic.py` with fit_gmm_bic() and select_k_min_bic() in `divide_conquer_method/clustering/gmm_bic.py`

### DEC Implementation (New)

- [ ] T018 [US1] Create `models/dec.py` with DECLayer (Student-t kernel, soft assignments) in `divide_conquer_method/models/dec.py`
- [ ] T019 [US1] Create `clustering/dec_refine.py` with refine_with_dec() training loop in `divide_conquer_method/clustering/dec_refine.py`

### ipRGC Validation (New)

- [ ] T020 [US1] Create `validation/iprgc_metrics.py` with compute_iprgc_metrics(), compute_enrichment(), compute_purity() in `divide_conquer_method/validation/iprgc_metrics.py`

### Output Saving

- [ ] T021 [US1] Create `evaluation.py` with save_embeddings(), save_cluster_assignments(), save_k_selection() in `divide_conquer_method/evaluation.py`

### Main Pipeline

- [ ] T022 [US1] Create `run_pipeline.py` with main() function for single-group execution in `divide_conquer_method/run_pipeline.py`
- [ ] T023 [US1] Add CLI argument parsing (--group, --input, --output, --subset) in `divide_conquer_method/run_pipeline.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Compare GMM-Only vs DEC-Refined Clusters (Priority: P1)

**Goal**: Generate comparison metrics and visualizations for GMM vs DEC clustering

**Independent Test**: After running pipeline, check `results/{group}/iprgc_validation.json` for both metric sets

### Comparison Metrics

- [ ] T024 [US2] Add comparison table generation to `evaluation.py` - save_comparison_table() in `divide_conquer_method/evaluation.py`

### Visualization

- [ ] T025 [P] [US2] Create `visualization.py` with plot_bic_curve() in `divide_conquer_method/visualization.py`
- [ ] T026 [P] [US2] Add plot_umap_comparison() showing GMM vs DEC clusters in `divide_conquer_method/visualization.py`
- [ ] T027 [US2] Add plot_iprgc_enrichment() bar chart in `divide_conquer_method/visualization.py`
- [ ] T028 [US2] Add plot_cluster_prototypes() for mean±SEM traces in `divide_conquer_method/visualization.py`

### Integration

- [ ] T029 [US2] Integrate visualization calls into run_pipeline.py in `divide_conquer_method/run_pipeline.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Batch Run All Three Groups (Priority: P2)

**Goal**: Process DSGC, OSGC, and Other groups in one command with consolidated report

**Independent Test**: `python -m divide_conquer_method.run_pipeline --all-groups`

### Batch Processing

- [ ] T030 [US3] Add --all-groups CLI flag handling in `divide_conquer_method/run_pipeline.py`
- [ ] T031 [US3] Implement loop over ['DSGC', 'OSGC', 'Other'] groups in `divide_conquer_method/run_pipeline.py`
- [ ] T032 [US3] Add per-group output subdirectories (results/{group}/, plots/{group}/) in `divide_conquer_method/run_pipeline.py`

### Consolidated Report

- [ ] T033 [US3] Create generate_consolidated_report() in `divide_conquer_method/evaluation.py`
- [ ] T034 [US3] Add summary table with cluster counts and ipRGC metrics across groups in `divide_conquer_method/evaluation.py`

**Checkpoint**: All three groups can be processed in one command

---

## Phase 6: User Story 4 - Resume from Checkpoints (Priority: P2)

**Goal**: Allow resuming pipeline from saved checkpoints (--skip-training, --skip-gmm, --skip-dec)

**Independent Test**: Run with --skip-training after initial run completes

### Checkpoint Loading

- [ ] T035 [US4] Add load_model() function to load saved autoencoder in `divide_conquer_method/train.py`
- [ ] T036 [US4] Add load_gmm_results() to load cached k* and GMM model in `divide_conquer_method/clustering/gmm_bic.py`
- [ ] T037 [US4] Add load_dec_results() to load cached DEC assignments in `divide_conquer_method/clustering/dec_refine.py`

### CLI Flags

- [ ] T038 [US4] Add --skip-training, --skip-gmm, --skip-dec flags in `divide_conquer_method/run_pipeline.py`
- [ ] T039 [US4] Implement checkpoint detection and loading logic in main() in `divide_conquer_method/run_pipeline.py`

**Checkpoint**: Pipeline can resume from any saved stage

---

## Phase 7: User Story 5 - Visualize Results Without Rerunning (Priority: P3)

**Goal**: Regenerate all plots from saved artifacts without computation

**Independent Test**: `python -m divide_conquer_method.run_pipeline --visualize-only --group DSGC`

### Visualization-Only Mode

- [ ] T040 [US5] Add --visualize-only flag in `divide_conquer_method/run_pipeline.py`
- [ ] T041 [US5] Create load_artifacts_for_visualization() to load embeddings, assignments from parquet/json in `divide_conquer_method/evaluation.py`
- [ ] T042 [US5] Implement generate_all_plots() standalone function in `divide_conquer_method/visualization.py`
- [ ] T043 [US5] Wire --visualize-only to skip training/clustering and call generate_all_plots() in `divide_conquer_method/run_pipeline.py`

**Checkpoint**: All plots can be regenerated from saved files in < 30 seconds

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T044 [P] Add logging throughout pipeline (use logging.getLogger(__name__)) in all modules
- [ ] T045 [P] Add docstrings to all public functions (Google-style)
- [ ] T046 Add edge case handling: small groups (< 50 cells), NaN traces, DEC non-convergence in `divide_conquer_method/run_pipeline.py`
- [ ] T047 Add reject-reason counter for data filtering audit in `divide_conquer_method/data_loader.py`
- [ ] T048 Validate pipeline end-to-end with quickstart.md examples
- [ ] T049 Run full pipeline on actual data and verify outputs match expectations

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can proceed in priority order (P1 → P2 → P3)
  - US1 and US2 are both P1, can be done in sequence (US2 builds on US1 outputs)
- **Polish (Phase 8)**: Depends on core user stories (US1, US2) being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Depends on US1 completion (needs pipeline output to visualize)
- **User Story 3 (P2)**: Can start after US1 - Adds batch mode
- **User Story 4 (P2)**: Can start after US1 - Adds checkpoint resume
- **User Story 5 (P3)**: Depends on US2 (visualization code) - Adds visualization-only mode

### Within Each User Story

- Models/utilities before main integration
- Core implementation before CLI flags
- Story complete before moving to next priority

### Parallel Opportunities

- T003, T004, T005 (subpackage inits) can run in parallel
- T011, T012, T013 (model files) can run in parallel
- T016 (gpu_gmm copy) parallel with T017 (gmm_bic)
- T025, T026 (visualization functions) can run in parallel
- T044, T045 (logging, docstrings) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch GMM components together:
Task: "Copy models/gpu_gmm.py from Autoencoder_method"
Task: "Create clustering/gmm_bic.py with fit_gmm_bic()"

# These can run in parallel since they're different files with no dependencies
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test with `--group DSGC --subset 500`
5. Verify outputs: BIC curve, cluster assignments, ipRGC metrics

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → **MVP Ready!**
3. Add User Story 2 → Test visualizations → Comparison plots working
4. Add User Story 3 → Test --all-groups → Batch mode working
5. Add User Story 4 → Test checkpoints → Resume mode working
6. Add User Story 5 → Test --visualize-only → Publication figure workflow

### Estimated Time

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Setup | T001-T006 | 15 min |
| Foundational | T007-T015 | 1 hr (mostly copy+adapt) |
| User Story 1 | T016-T023 | 2-3 hrs (new DEC code) |
| User Story 2 | T024-T029 | 1-2 hrs (visualization) |
| User Story 3 | T030-T034 | 30 min |
| User Story 4 | T035-T039 | 45 min |
| User Story 5 | T040-T043 | 30 min |
| Polish | T044-T049 | 1 hr |

**Total**: ~7-9 hours to complete all stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Code reuse from Autoencoder_method significantly reduces implementation time
- Key new code: `models/dec.py`, `clustering/dec_refine.py`, `validation/iprgc_metrics.py`, `grouping.py`
