# Feature Specification: Weakly Supervised RGC Subtype Clustering with Autoencoder Features

**Feature Branch**: `013-ae-rgc-clustering`  
**Created**: 2026-01-11  
**Status**: Draft  
**Input**: User description: Weakly supervised RGC subtype clustering pipeline with autoencoder-derived features, cross-validation via omitted label purity, and Baden-style clustering

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Autoencoder to Learn Fixed-Budget Latent Representations (Priority: P1)

A researcher wants to train an autoencoder that transforms preprocessed electrophysiology traces into a 49-dimensional latent feature vector, where each stimulus segment contributes a fixed number of features (e.g., 4 features from each frequency section, 12 from moving bar, etc.). The autoencoder should incorporate weak supervision from coarse group labels (AC vs RGC, DS vs non-DS, ipRGC vs non-ipRGC) without requiring subtype labels.

**Why this priority**: The autoencoder is the foundation of the entire pipeline—without trained latent representations, no downstream clustering or validation can occur.

**Independent Test**: Can be fully tested by training the autoencoder on preprocessed data and verifying that output embeddings have correct dimensionality (49D) with correct per-segment breakdown.

**Acceptance Scenarios**:

1. **Given** a preprocessed parquet file with all required trace columns and group labels, **When** the autoencoder is trained, **Then** each cell produces a 49D embedding with the specified per-segment dimensions (freq sections: 4×5=20, color: 6, bar: 12, STA: 3, ipRGC: 4, step-up: 4).

2. **Given** training data with coarse group labels (axon_type, ds_p_value, iprgc_2hz_QI), **When** the autoencoder is trained with weak supervision loss enabled, **Then** the embeddings show measurable separation by group (verified via silhouette score or similar metric).

3. **Given** a trained autoencoder, **When** embeddings are extracted for all cells, **Then** no embedding contains NaN values and all embeddings are finite.

---

### User Story 2 - Cluster Embeddings Using Baden-Style GMM with Group Constraints (Priority: P1)

A researcher wants to cluster the 49D autoencoder embeddings using diagonal-covariance GMM with BIC/log-Bayes-factor model selection, ensuring that each discovered cluster contains cells from only one coarse group (AC, ipRGC, DS-RGC, or nonDS-RGC).

**Why this priority**: Clustering is the primary analytical output—subtypes must be discovered and constrained to biologically meaningful groups.

**Independent Test**: Can be tested by providing pre-computed embeddings and verifying that clustering produces group-pure clusters with valid BIC curves.

**Acceptance Scenarios**:

1. **Given** a matrix of 49D embeddings with associated group labels, **When** Baden-style clustering is performed, **Then** separate GMM models are fit for each disjoint group with BIC curves showing optimal k selection.

2. **Given** clustering results, **When** cluster assignments are inspected, **Then** no cluster contains cells from multiple coarse groups.

3. **Given** multiple candidate k values, **When** BIC/log-Bayes-factor selection is performed, **Then** the selected k* is the value that maximizes model evidence within the defined threshold.

---

### User Story 3 - Validate Generalization via Omitted-Label Cross-Validation (Priority: P2)

A researcher wants to assess whether the learned representations capture biologically meaningful structure beyond the explicitly supervised labels. This is done by "omitting" one coarse label during training and measuring how well the resulting clusters align with that omitted label (purity score).

**Why this priority**: Cross-validation prevents overfitting to supervision signals and ensures the representation learns genuine biological structure.

**Independent Test**: Can be tested by running three CV turns (omit each label once) and computing purity scores for each omitted label.

**Acceptance Scenarios**:

1. **Given** a configured CV turn that omits the "ipRGC" label, **When** the autoencoder is trained without ipRGC in the supervision loss and clustering is done without ipRGC group splitting, **Then** the discovered clusters are evaluated for purity with respect to the ipRGC label.

2. **Given** three CV turns (omit axon_type, omit ds_cell, omit ipRGC), **When** all turns complete, **Then** a summary table shows purity scores for each omitted label and an aggregate CVScore.

3. **Given** CV purity results, **When** the researcher compares configurations, **Then** configurations with higher CVScore indicate better generalization.

---

### User Story 4 - Assess Cluster Stability via Bootstrap Subsampling (Priority: P2)

A researcher wants to verify that discovered clusters are stable under perturbation by performing Baden-style 90% subsampling bootstraps and measuring cluster reproducibility via correlation of cluster centroids.

**Why this priority**: Stability ensures clusters represent robust biological structure rather than sampling artifacts.

**Independent Test**: Can be tested by running bootstrap iterations on a single group's clustering and computing median correlation scores.

**Acceptance Scenarios**:

1. **Given** a fitted GMM with k* clusters on a group, **When** 20 bootstrap iterations are performed (90% subsampling), **Then** each bootstrap produces matched clusters with correlation scores to reference centroids.

2. **Given** bootstrap results, **When** stability is summarized, **Then** mean ± SD of median correlations is reported per group.

3. **Given** stability thresholds, **When** any group has mean median correlation below threshold, **Then** that group/run is flagged as potentially unstable.

---

### User Story 5 - Generate Comprehensive Analysis Reports and Visualizations (Priority: P3)

A researcher wants publication-ready outputs including UMAP embeddings colored by group/cluster, BIC curves per group, cluster response prototypes (mean±SEM traces), and cross-validation summary tables.

**Why this priority**: Visualizations enable interpretation and manuscript preparation, but are downstream of core analytical functionality.

**Independent Test**: Can be tested by running the full pipeline and verifying that all expected output files are generated in the results folder.

**Acceptance Scenarios**:

1. **Given** completed clustering, **When** visualizations are generated, **Then** UMAP plots show embeddings colored by axon_type, DS status, ipRGC status, and final cluster assignments.

2. **Given** each subtype cluster, **When** response prototypes are computed, **Then** mean±SEM traces are shown for each stimulus segment (chirp sections, color, bar, RF, ipRGC, step-up).

3. **Given** all pipeline outputs, **When** the results folder is inspected, **Then** it contains: BIC curves, cluster size histograms, CV purity tables, stability reports, and UMAP visualizations.

---

### Edge Cases

- What happens when a cell has NaN in `iprgc_2hz_QI`? → Treated as "not ipRGC" for group assignment.
- What happens when a coarse group has too few cells for reliable GMM fitting? → Minimum cell count per group is enforced; groups below threshold are flagged.
- What happens when no k value produces acceptable BIC improvement? → The pipeline falls back to k=1 (single cluster) for that group with a warning.
- What happens when the ipRGC stimulus data is missing for some cells? → Those cells are excluded from analysis (required column filtering).
- What happens when bootstrap iterations produce degenerate clusters? → Degenerate iterations are excluded from stability calculation with a warning.

## Requirements *(mandatory)*

### Functional Requirements

#### Data Loading and Filtering

- **FR-001**: System MUST load data from parquet files containing all required trace columns and metadata.
- **FR-002**: System MUST filter cells using the same criteria as existing Baden pipeline (QI threshold, axon type, baseline threshold, batch size minimum).
- **FR-003**: System MUST extend required columns to include: all `freq_section_*`, `green_blue_3s_3i_3x`, all 8 `corrected_moving_h_bar_*` columns, `sta_time_course`, `step_up_5s_5i_b0_3x`, `iprgc_test`, and `iprgc_2hz_QI`.
- **FR-004**: System MUST treat NaN values in `iprgc_2hz_QI` as "not ipRGC" rather than excluding those cells.

#### Preprocessing

- **FR-005**: System MUST apply the same preprocessing chain as existing pipeline (trial mean → low-pass filter → downsample) for all stimulus traces except where explicitly specified otherwise.
- **FR-006**: System MUST apply 10 Hz low-pass filtering and 60→10 Hz downsampling for frequency sections 0.5/1/2/4 Hz.
- **FR-007**: System MUST preserve the 10 Hz frequency section without filtering/downsampling (slice edges only).
- **FR-008**: System MUST apply 2 Hz low-pass filtering for `iprgc_test` traces and downsample to 2 Hz target rate.
- **FR-009**: System MUST concatenate all 8 moving bar directions in fixed order (0°, 45°, ..., 315°) into a single segment.

#### Group Label Definitions

- **FR-010**: System MUST compute disjoint group assignments using configurable precedence (default: AC first, then ipRGC, then DS-RGC, else nonDS-RGC).
- **FR-011**: System MUST use thresholds: `iprgc_2hz_QI > 0.8` for ipRGC, `ds_p_value < 0.05` for DS, `axon_type == "ac"` for amacrine cells.
- **FR-012**: System MUST allow precedence order to be configurable.

#### Autoencoder Architecture

- **FR-013**: System MUST implement segment-wise encoding where each stimulus segment has its own encoder producing a fixed-size latent vector.
- **FR-014**: System MUST produce latent vectors with exactly these dimensions per segment: freq sections (4 each × 5 = 20), color (6), bar concat (12), STA (3), ipRGC (4), step-up (4) = 49 total.
- **FR-015**: System MUST implement corresponding segment-wise decoders for reconstruction.
- **FR-016**: System MUST compute reconstruction loss as weighted sum of per-segment losses, with weights inversely proportional to segment length.

#### Weak Supervision

- **FR-017**: System MUST implement supervised contrastive loss as the default weak supervision component, using only coarse group labels.
- **FR-018**: System MUST allow the weak supervision weight (β) to be configured.
- **FR-019**: System MUST allow specific group labels to be excluded from supervision (for CV turns).

#### Clustering

- **FR-020**: System MUST perform diagonal-covariance GMM clustering on standardized (z-scored) embeddings.
- **FR-021**: System MUST select optimal k using BIC or log-Bayes-factor threshold.
- **FR-022**: System MUST cluster each disjoint group separately (clusters cannot cross group boundaries).
- **FR-023**: System MUST generate BIC curves showing model selection rationale.

#### Cross-Validation

- **FR-024**: System MUST implement omitted-label CV turns where one coarse label is excluded from both AE supervision and group splitting.
- **FR-025**: System MUST compute cluster purity with respect to the omitted label.
- **FR-026**: System MUST produce a CV summary table with purity for each omitted label and aggregate CVScore.

#### Stability Assessment

- **FR-027**: System MUST implement 90% subsampling bootstrap stability testing.
- **FR-028**: System MUST match bootstrap clusters to reference clusters via maximum correlation of centroids.
- **FR-029**: System MUST report mean ± SD of median correlations across bootstraps.

#### Output and Reporting

- **FR-030**: System MUST generate UMAP visualizations of embeddings colored by group labels and cluster assignments.
- **FR-031**: System MUST generate cluster response prototype plots (mean ± SEM per stimulus segment).
- **FR-032**: System MUST save all trained models, embeddings, and cluster assignments for reproducibility.

#### Code Organization

- **FR-033**: System MUST contain all new code within `dataframe_phase/classification_v2/Autoencoder_method/` folder.
- **FR-034**: System MUST expose all configurable parameters in a centralized `config.py` file.
- **FR-035**: System MUST provide a single entry-point script (`run_pipeline.py`) that executes the full pipeline.
- **FR-036**: System MUST implement GMM clustering, BIC selection, and bootstrap stability independently (no imports from `Baden_method/`).

### Key Entities

- **Cell**: Individual neural recording unit with unique ID, associated traces, metadata, and group labels.
- **Trace Segment**: A preprocessed 1D time series from a specific stimulus (e.g., freq_section_1hz, green_blue).
- **Embedding**: 49D latent vector derived from autoencoder encoding of all trace segments for a cell.
- **Coarse Group**: Disjoint category assigned to each cell (AC, ipRGC, DS-RGC, nonDS-RGC) based on label precedence.
- **Subtype Cluster**: Fine-grained cluster discovered within a coarse group via GMM.
- **CV Turn**: One cross-validation iteration where a specific coarse label is omitted from supervision and splitting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Embeddings preserve biological signal: UMAP visualizations show clear separation between coarse groups (AC vs RGC, DS vs non-DS).
- **SC-002**: Cluster purity for omitted labels exceeds 70% on average across CV turns (CVScore ≥ 0.7).
- **SC-003**: Cluster stability: mean median correlation ≥ 0.8 across 90% bootstrap iterations for each group.
- **SC-004**: All discovered clusters have ≥ 10 cells (no degenerate micro-clusters).
- **SC-005**: BIC curves show clear model selection with at least 2-point improvement over adjacent k values.
- **SC-006**: Full pipeline (preprocessing → training → clustering → validation) completes in under 4 hours for datasets up to 10,000 cells.
- **SC-007**: Response prototypes for each cluster show interpretable stimulus-response patterns consistent with known RGC functional types.
- **SC-008**: Zero cells assigned to clusters that violate group constraints (100% group purity by construction).

## Clarifications

### Session 2026-01-11

- Q: Where should all new code be located? → A: `dataframe_phase/classification_v2/Autoencoder_method/`
- Q: Code reuse strategy for clustering? → A: Implement clustering independently within `Autoencoder_method/`
- Q: Default weak supervision loss type? → A: Supervised contrastive loss (primary approach)

## Assumptions

- The input parquet file contains all required trace columns in the expected format (nested arrays for trial data, 1D arrays for already-averaged traces).
- The existing Baden preprocessing pipeline documentation accurately describes the current signal processing chain.
- `iprgc_test` column exists in the input data with shape compatible with trial averaging and filtering.
- Hardware resources (GPU or sufficient CPU/RAM) are available for autoencoder training.
- The fixed latent dimensions (49D total) are appropriate for capturing stimulus-relevant variance.
- Group precedence (AC > ipRGC > DS > nonDS) reflects meaningful biological hierarchy for the analysis goals.
