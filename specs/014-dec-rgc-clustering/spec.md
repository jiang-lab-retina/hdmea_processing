# Feature Specification: DEC-Refined RGC Subtype Clustering

**Feature Branch**: `014-dec-rgc-clustering`  
**Created**: 2026-01-13  
**Status**: Draft  
**Input**: Divide-and-conquer clustering pipeline for RGC subtypes using unsupervised CNN autoencoder, Baden-style GMM+BIC k-selection, and DEC refinement with ipRGC validation.

---

## Overview

This pipeline clusters **Retinal Ganglion Cells (RGCs)** into functional subtypes using a two-phase approach:

1. **Phase 1 (Unsupervised)**: Train a CNN autoencoder with reconstruction-only loss, then select optimal cluster count k* using Baden-style diagonal GMM + BIC.

2. **Phase 2 (Refinement)**: Initialize Deep Embedded Clustering (DEC) with k* and GMM centers, refine embeddings and cluster assignments.

3. **Validation**: Evaluate biological relevance via ipRGC enrichment analysis.

Key differences from the existing Autoencoder_method:
- **No weak supervision** during initial clustering (pure unsupervised)
- **DEC refinement** stage for cluster sharpening
- **ipRGC enrichment** as primary validation metric
- **DS > OS priority** grouping rule (not AC > ipRGC > DS)

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Complete Pipeline for One Group (Priority: P1)

A researcher wants to cluster RGCs from one functional group (e.g., DS-RGCs) into subtypes, automatically selecting the optimal number of clusters based on BIC, and obtain DEC-refined assignments.

**Why this priority**: Core functionality that demonstrates the complete pipeline workflow on a single group.

**Independent Test**: Run pipeline on DSGC group from a parquet file; verify outputs include BIC curve, initial GMM labels, DEC-refined labels, and ipRGC enrichment report.

**Acceptance Scenarios**:

1. **Given** a parquet file with RGC traces and metadata, **When** pipeline is run for the DSGC group, **Then** the system outputs:
   - BIC curve showing model selection
   - Optimal k* based on minimum BIC
   - Initial cluster assignments from GMM
   - Refined cluster assignments from DEC
   - ipRGC purity and enrichment metrics

2. **Given** the pipeline completes, **When** user inspects output directory, **Then** all artifacts exist without rerunning any computation.

---

### User Story 2 - Compare GMM-Only vs DEC-Refined Clusters (Priority: P1)

A researcher wants to evaluate whether DEC refinement improves cluster quality compared to the initial GMM clustering.

**Why this priority**: Essential for validating that DEC provides value over simple GMM clustering.

**Independent Test**: Compare ipRGC purity metrics between initial GMM and final DEC assignments for the same run.

**Acceptance Scenarios**:

1. **Given** a completed pipeline run, **When** comparing initial vs refined clusters, **Then** both sets of metrics are saved side-by-side in a comparison table.

2. **Given** the output files, **When** generating visualizations, **Then** UMAP plots show both initial and refined cluster boundaries for comparison.

---

### User Story 3 - Batch Run All Three Groups (Priority: P2)

A researcher wants to run the pipeline on all three groups (DSGC, OSGC, Other) in one command and get consolidated results.

**Why this priority**: Convenience for full dataset analysis; builds on single-group functionality.

**Independent Test**: Run `--all-groups` mode and verify outputs exist for each group in separate subdirectories.

**Acceptance Scenarios**:

1. **Given** the input data contains cells from all three groups, **When** pipeline runs with `--all-groups`, **Then** each group is processed independently with its own k* selection.

2. **Given** all groups complete, **When** user views summary, **Then** a consolidated report shows cluster counts and ipRGC metrics across all groups.

---

### User Story 4 - Resume from Checkpoints (Priority: P2)

A researcher wants to resume a crashed run from the last completed stage (e.g., skip autoencoder training if already complete).

**Why this priority**: Practical for long-running pipelines with expensive training stages.

**Independent Test**: Run pipeline, interrupt after AE training, restart with `--skip-training`, verify it loads saved model and continues.

**Acceptance Scenarios**:

1. **Given** a saved autoencoder checkpoint exists, **When** running with `--skip-training`, **Then** pipeline loads the model and proceeds to GMM/DEC stages.

2. **Given** GMM results are cached, **When** running with `--skip-gmm`, **Then** pipeline uses cached k* and proceeds to DEC stage.

---

### User Story 5 - Visualize Results Without Rerunning (Priority: P3)

A researcher wants to regenerate plots (UMAP, BIC curves, prototypes) from saved artifacts without rerunning any computation.

**Why this priority**: Efficient iteration on figures for publications.

**Independent Test**: Run visualization-only mode on existing results directory.

**Acceptance Scenarios**:

1. **Given** a completed results directory, **When** running `--visualize-only`, **Then** all plots are regenerated from saved embeddings and assignments.

---

### Edge Cases

- What happens when a group has fewer cells than the minimum cluster count? → Assign all to single cluster, log warning.
- How does system handle NaN values in required trace columns? → Drop cells with NaNs, log reject count by reason.
- What if DEC fails to converge? → Stop after max iterations, save current state with warning flag.
- What if all cells in a group are classified as ipRGC? → Enrichment is undefined; report as special case.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Data Selection

- **FR-001**: System MUST filter to `axon_type == "rgc"` cells only
- **FR-002**: System MUST assign groups using DS > OS priority:
  - DSGC: `ds_p_value < 0.05`
  - OSGC: `os_p_value < 0.05` AND NOT DSGC
  - Other: remaining RGCs
- **FR-003**: System MUST log overlap count (cells meeting both DS and OS thresholds) and confirm they go to DSGC
- **FR-004**: System MUST drop rows with NaN in required columns: `axon_type`, `ds_p_value`, `os_p_value`, `iprgc_2hz_QI`
- **FR-005**: System MUST maintain a reject-reason counter for audit

#### Preprocessing

- **FR-006**: System MUST apply segment-specific preprocessing:
  - Frequency sections (0.5-4 Hz): low-pass 10 Hz, resample 60→10 Hz
  - Frequency section 10 Hz: no low-pass, preserve at 60 Hz with edge cropping
  - Color, step-up: low-pass 10 Hz, resample 60→10 Hz
  - Moving bar (8 directions): low-pass 10 Hz, resample 60→10 Hz, concatenate
  - RF time course: keep full 60 samples unchanged
  - iprgc_test: low-pass 4 Hz, resample to 2 Hz with proper anti-aliasing
- **FR-007**: System MUST enforce fixed expected lengths per segment; drop cells that do not match

#### Autoencoder

- **FR-008**: System MUST train CNN autoencoder with reconstruction-only loss (no weak supervision)
- **FR-009**: System MUST use segment-wise encoders with fixed latent budget (total 49D):
  - freq_section_* (×5): 4D each
  - green_blue: 6D
  - bar_concat: 12D
  - sta_time_course: 3D
  - iprgc_test: 4D
  - step_up: 4D
- **FR-010**: System MUST normalize reconstruction loss by segment length
- **FR-011**: System MUST implement early stopping on validation reconstruction loss
- **FR-012**: System MUST save model checkpoints

#### GMM/BIC Clustering

- **FR-013**: System MUST fit diagonal-covariance GMM for each k in candidate grid
- **FR-014**: System MUST use multiple restarts (≥20) and covariance regularization
- **FR-015**: System MUST select k* as argmin(BIC)
- **FR-016**: System MUST save full BIC curve data
- **FR-017**: System MUST save posterior probabilities for all cells

#### DEC Refinement

- **FR-018**: System MUST initialize DEC cluster centers from GMM means at k*
- **FR-019**: System MUST implement DEC objective: KL(P||Q) with target distribution sharpening
- **FR-020**: System MUST update target distribution periodically (every m iterations)
- **FR-021**: System MUST include reconstruction term (IDEC-style) to prevent collapse
- **FR-022**: System MUST stop when assignment changes < threshold or max iterations reached
- **FR-023**: System MUST save both initial GMM and final DEC labels/assignments

#### ipRGC Validation

- **FR-024**: System MUST define ipRGC label as `iprgc_2hz_QI > 0.8`
- **FR-025**: System MUST compute cluster-wise ipRGC fraction: f_c = P(ipRGC | cluster=c)
- **FR-026**: System MUST compute overall ipRGC purity metric
- **FR-027**: System MUST compute enrichment: E_c = f_c / P(ipRGC)
- **FR-028**: System MUST report top 3 enriched clusters with sizes and fractions
- **FR-029**: System MUST compute metrics for both initial GMM and final DEC clustering

#### Output Artifacts

- **FR-030**: System MUST save preprocessing metadata (segment names, lengths, filter settings, cell IDs)
- **FR-031**: System MUST save embeddings (before and after DEC if updated)
- **FR-032**: System MUST save UMAP 2D coordinates for visualization
- **FR-033**: System MUST save cluster prototypes (mean±SEM traces per cluster per segment)
- **FR-034**: System MUST save comparison tables (GMM-only vs DEC-refined metrics)

### Key Entities

- **Cell**: Individual RGC with trace data across 10 stimulus segments and metadata (ds_p_value, os_p_value, iprgc_2hz_QI)
- **Group**: Functional category (DSGC, OSGC, Other) based on DS/OS priority rule
- **Embedding**: 49-dimensional latent representation from autoencoder
- **Cluster**: Subtype assignment from GMM (initial) or DEC (refined)
- **Prototype**: Mean response profile for a cluster, computed per segment

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Pipeline processes 2000+ cells per group without memory errors
- **SC-002**: BIC curve shows clear minimum (k* is interpretable, not at grid boundary)
- **SC-003**: DEC refinement converges within 200 iterations for typical datasets
- **SC-004**: ipRGC purity after DEC is ≥ initial GMM purity (no degradation)
- **SC-005**: At least one cluster shows ipRGC enrichment > 2× baseline prevalence
- **SC-006**: All visualization artifacts can be regenerated from saved files in < 30 seconds
- **SC-007**: Complete pipeline execution (one group) finishes in < 30 minutes on GPU

### Biological Validation Criteria

- **SC-008**: Cluster separation reflects known functional distinctions (DS tuning, ipRGC responses)
- **SC-009**: ipRGC-enriched clusters show characteristic slow, sustained responses in iprgc_test traces
- **SC-010**: Direction-selective clusters show tuning in bar_concat prototypes

---

## Clarifications

### Session 2026-01-13

- Q: What is the default input file path? → A: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet`

---

## Assumptions

1. **Default input file**: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet` (same as Autoencoder_method)
2. Input parquet file follows the same schema as `Autoencoder_method` (column names, trace formats)
3. All cells have complete trace data for all 10 segments
4. GPU (CUDA) is available for training; CPU fallback is supported but slower
5. Minimum viable group size is 50 cells (same as Autoencoder_method)
6. Expected cluster count range: 1-40 for most groups (configurable per group)
7. DEC uses Student-t kernel for soft assignments (standard DEC formulation)

---

## Dependencies

- **Autoencoder_method**: Shared preprocessing functions, config patterns, and visualization utilities
- **PyTorch**: Deep learning framework for autoencoder and DEC
- **scikit-learn**: GMM fitting and BIC computation
- **UMAP**: Dimensionality reduction for visualization

---

## Out of Scope

- Weak supervision during training (SupCon loss, purity loss) - this is the key difference from Autoencoder_method
- Amacrine cells (AC) - this pipeline is RGC-only
- ipRGC as a separate group - ipRGC is used for validation, not grouping
- Bootstrap stability testing (can be added later)
- Full cross-validation framework (simplified to ipRGC validation)
