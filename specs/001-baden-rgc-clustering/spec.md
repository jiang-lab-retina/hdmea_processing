# Feature Specification: Baden-Method RGC Clustering Pipeline

**Feature Branch**: `001-baden-rgc-clustering`  
**Created**: 2026-01-06  
**Status**: Draft  
**Input**: User description: "Implement Baden-aligned unsupervised RGC clustering pipeline with sparse PCA features, diagonal GMM, and BIC model selection"
**Input Dataset**: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet`  
**Output Directory**: `dataframe_phase/classification_v2/Baden_method`

## Overview

Implement an unsupervised retinal ganglion cell (RGC) classification pipeline that faithfully reproduces the methodology described in Baden et al. The pipeline extracts a standardized 40-dimensional feature vector from four visual stimulus responses, clusters cells using Gaussian Mixture Models with BIC-based model selection, and evaluates cluster quality through posterior probability analysis and bootstrap stability testing.

The pipeline processes direction-selective (DS) and non-DS cells independently, as per the original methodology.

## Clarifications

### Session 2026-01-06

- Q: Which quality index column should be used for filtering? → A: Use `step_up_QI` (the column that exists in the data)
- Q: Should the pipeline generate visualization plots? → A: Yes, generate BIC curves, posterior curves, and 2D cluster projections (UMAP or PCA)
- Q: What should the default k_max values be for DS and non-DS populations? → A: DS: k_max=30, non-DS: k_max=60
- Q: What is the input dataset? → A: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet`
- Q: How should results be exported? → A: Export cluster labels and related data as a parquet file

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Complete End-to-End Clustering (Priority: P1)

A researcher loads a parquet dataframe containing MEA recording data with averaged stimulus response traces and runs the full Baden clustering pipeline to classify RGCs into functional subtypes.

**Why this priority**: This is the core functionality that delivers the primary scientific value—automatically classifying RGCs into biologically meaningful subtypes using a validated methodology.

**Independent Test**: Can be tested by running the pipeline on the input parquet file and verifying it produces labeled clusters with quality metrics.

**Acceptance Scenarios**:

1. **Given** a parquet dataframe with required trace columns and metadata, **When** the pipeline is executed, **Then** the system produces cluster labels for each cell, saves the trained models, and outputs quality evaluation metrics.

2. **Given** a parquet dataframe with some cells missing required traces, **When** the pipeline is executed, **Then** cells with missing data are excluded from clustering, documented in the output, and remaining valid cells are clustered successfully.

3. **Given** cells with step quality index (QI) below threshold or non-RGC axon types, **When** preprocessing runs, **Then** these cells are filtered out before feature extraction with counts logged.

---

### User Story 2 - DS vs Non-DS Independent Clustering (Priority: P1)

A researcher needs direction-selective and non-direction-selective cells clustered independently, as the original Baden methodology dictates separate clustering for these populations.

**Why this priority**: This matches the biological reality that DS cells form a distinct population requiring separate analysis, and is essential for reproducing Baden's methodology.

**Independent Test**: Can verify by checking that DS cells (p < 0.05) and non-DS cells are processed through completely separate feature extraction and clustering pipelines.

**Acceptance Scenarios**:

1. **Given** a dataset containing both DS and non-DS cells, **When** clustering is performed, **Then** two independent GMM models are fitted—one for DS cells and one for non-DS cells.

2. **Given** ds_p_value column in the input data, **When** the DS/non-DS split is applied, **Then** cells with ds_p_value < 0.05 are classified as DS, and all others as non-DS.

---

### User Story 3 - Feature Extraction with Sparse PCA (Priority: P2)

A researcher needs to extract the standardized 40-dimensional feature vector from stimulus responses using sparse PCA with enforced sparsity constraints matching Baden's specifications.

**Why this priority**: Accurate feature extraction is foundational—incorrect features will produce meaningless clusters regardless of clustering quality.

**Independent Test**: Can verify feature matrix dimensions (N × 40) and sparsity constraints on learned components.

**Acceptance Scenarios**:

1. **Given** preprocessed stimulus traces, **When** sparse PCA is applied to chirp responses, **Then** 20 features are extracted using components with exactly 10 non-zero time bins each.

2. **Given** preprocessed color stimulus traces, **When** sparse PCA is applied, **Then** 6 features are extracted using components with exactly 10 non-zero time bins each.

3. **Given** 8-direction moving bar responses, **When** SVD is applied per cell, **Then** a temporal component (time course) and tuning curve are extracted from the first singular vector.

4. **Given** moving bar time courses, **When** sparse PCA is applied, **Then** 8 features with 5 non-zero bins (from time course) and 4 features with 6 non-zero bins (from derivative) are extracted.

5. **Given** RF time course (STA) data, **When** regular PCA is applied, **Then** 2 features are extracted.

---

### User Story 4 - Automatic Cluster Number Selection via BIC (Priority: P2)

A researcher needs the system to automatically determine the optimal number of clusters using Bayesian Information Criterion (BIC), rather than manually specifying cluster count.

**Why this priority**: Automatic model selection is critical for reproducibility and removes subjective decisions about cluster count.

**Independent Test**: Can verify by checking that BIC values are computed for a range of k values and the minimum is selected.

**Acceptance Scenarios**:

1. **Given** a standardized feature matrix, **When** GMM clustering is performed, **Then** BIC values are computed for cluster counts from 1 to a configurable maximum.

2. **Given** BIC values for multiple k values, **When** model selection completes, **Then** the k with minimum BIC is selected as optimal.

3. **Given** BIC values, **When** log Bayes factors are computed, **Then** values greater than 6 indicate strong evidence for additional splitting.

---

### User Story 5 - Cluster Quality Evaluation (Priority: P3)

A researcher needs to evaluate cluster quality using posterior probability separability curves and bootstrap stability analysis to validate the clustering results.

**Why this priority**: Quality metrics are essential for publication and determining if clusters are biologically meaningful versus statistical artifacts.

**Independent Test**: Can verify by generating posterior curves and stability correlation values.

**Acceptance Scenarios**:

1. **Given** a fitted GMM model, **When** posterior analysis is performed, **Then** rank-ordered posterior probability curves are generated for each cluster showing assignment confidence.

2. **Given** a fitted GMM with optimal k, **When** bootstrap stability is evaluated, **Then** 20 resampling iterations at 90% sample size are performed and median correlation of matched cluster centers is reported.

3. **Given** bootstrap stability results, **When** median correlation exceeds 0.95, **Then** clusters are considered stable.

---

### Edge Cases

- What happens when a cell has all-zero traces after baseline subtraction? → Cell is excluded with warning logged.
- What happens when fewer than k cells remain after filtering? → Pipeline raises informative error about insufficient data.
- How does system handle traces with different lengths than expected? → Traces are validated against expected frame counts and non-conforming cells are excluded.
- What happens when BIC continues decreasing up to k_max? → Warning is raised suggesting k_max may need to be increased.

## Requirements *(mandatory)*

### Functional Requirements

#### Data Input & Filtering
- **FR-001**: System MUST load data from parquet format with configurable input path (default: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet`).
- **FR-002**: System MUST exclude cells with NaN values in any required column (chirp trace, color trace, 8 moving bar traces, RF time course, step_up_QI, ds_p_value, axon_type).
- **FR-003**: System MUST exclude cells containing NaN values within any trace array.
- **FR-004**: System MUST filter to retain only cells with step_up_QI > 0.5.
- **FR-005**: System MUST filter to retain only cells with axon_type == "rgc".
- **FR-006**: System MUST split cells into DS (ds_p_value < 0.05) and non-DS (ds_p_value >= 0.05) groups.

#### Signal Preprocessing
- **FR-007**: System MUST apply a 10 Hz low-pass zero-phase filter to chirp, color, and moving bar traces.
- **FR-008**: System MUST NOT apply low-pass filtering to RF time course (sta_time_course).
- **FR-009**: System MUST subtract baseline (median of first 8 samples) from each trace.
- **FR-010**: System MUST normalize traces after baseline subtraction using max-absolute value with small epsilon for numerical stability.

#### Feature Extraction
- **FR-011**: System MUST extract 20 sparse PCA features from chirp responses with 10 non-zero time bins per component.
- **FR-012**: System MUST extract 6 sparse PCA features from color responses with 10 non-zero time bins per component.
- **FR-013**: System MUST perform SVD on each cell's 8-direction moving bar matrix to extract temporal component.
- **FR-014**: System MUST extract 8 sparse PCA features from bar time courses with 5 non-zero bins per component.
- **FR-015**: System MUST compute temporal derivative of bar time course and extract 4 sparse PCA features with 6 non-zero bins per component.
- **FR-016**: System MUST extract 2 regular PCA features from RF time course.
- **FR-017**: System MUST concatenate all features into a 40-dimensional feature vector per cell.
- **FR-018**: System MUST z-score standardize features within each population (DS and non-DS separately).
- **FR-019**: System MUST enforce hard sparsity constraints by keeping only top-k weights per component after sparse PCA fitting.

#### Clustering
- **FR-020**: System MUST fit diagonal covariance GMM models.
- **FR-021**: System MUST add regularization (1e-5) to covariance diagonal.
- **FR-022**: System MUST restart GMM fitting 20 times per candidate k and retain best likelihood solution.
- **FR-023**: System MUST compute BIC for each candidate k from 1 to configurable maximum (default k_max=30 for DS, k_max=60 for non-DS).
- **FR-024**: System MUST select optimal k as the value minimizing BIC.
- **FR-025**: System MUST compute log Bayes factors between adjacent k values.

#### Quality Evaluation
- **FR-026**: System MUST compute posterior probabilities for cluster assignments.
- **FR-027**: System MUST generate rank-ordered posterior probability curves per cluster.
- **FR-028**: System MUST perform bootstrap stability analysis with 20 iterations at 90% sample size.
- **FR-029**: System MUST match bootstrap cluster centers to original centers by maximum correlation.
- **FR-030**: System MUST report median correlation across matched clusters.

#### Output & Persistence
- **FR-031**: System MUST save all outputs to the specified output directory (dataframe_phase/classification_v2/Baden_method).
- **FR-032**: System MUST save trained GMM models for both DS and non-DS populations.
- **FR-033**: System MUST save cluster labels for each cell as a parquet file containing: cell identifiers, cluster labels (DS and non-DS), posterior probabilities, and population assignment (DS/non-DS).
- **FR-034**: System MUST save BIC curves and selected k values.
- **FR-035**: System MUST save evaluation metrics (posterior curves, stability correlations).
- **FR-036**: System MUST save sparse PCA and PCA transformation models for reproducibility.

#### Visualization Outputs
- **FR-037**: System MUST generate and save BIC curve plots showing BIC values vs cluster count k for both DS and non-DS populations.
- **FR-038**: System MUST generate and save posterior probability separability curve plots for each population.
- **FR-039**: System MUST generate and save 2D cluster projection plots using UMAP or PCA, colored by cluster assignment, for both DS and non-DS populations.
- **FR-040**: System MUST save all plots to a `plots/` subdirectory within the output directory.

### Key Entities

- **Cell**: Individual RGC unit with unique identifier, stimulus response traces, and extracted features.
- **Stimulus Response Traces**: Time-series firing rate data for chirp, color, moving bar (8 directions), and RF time course stimuli.
- **Feature Vector**: 40-dimensional standardized vector combining sparse PCA and PCA features from all stimuli.
- **Cluster**: Group of functionally similar cells identified by GMM, characterized by mean and covariance in feature space.
- **GMM Model**: Fitted Gaussian mixture model with cluster parameters, trained separately for DS and non-DS populations.
- **Results Parquet**: Output file containing cell identifiers, cluster labels, posterior probabilities, population flags, and feature vectors for downstream analysis.

## Assumptions

- Input parquet file (`firing_rate_with_all_features_loaded_extracted20260104.parquet`) contains the required columns as specified in the feature description.
- Sampling rate for traces is 60 Hz (consistent with existing codebase).
- Trace lengths are consistent across cells for each stimulus type.
- The step_up_QI threshold of 0.5 is appropriate for filtering high-quality responses.
- The ds_p_value column contains valid statistical test results for direction selectivity.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Pipeline successfully processes the full input dataset and produces cluster assignments for all valid cells within 30 minutes on standard workstation hardware.
- **SC-002**: Feature extraction produces exactly 40 features per cell as specified (20+6+8+4+2).
- **SC-003**: Sparse PCA components respect exact sparsity constraints (10, 10, 5, 6 non-zero bins respectively).
- **SC-004**: BIC curve shows clear minimum for both DS and non-DS populations, or appropriate warning is raised.
- **SC-005**: Bootstrap stability analysis achieves median cluster correlation ≥ 0.90 for stable clustering.
- **SC-006**: Posterior probability separability curves show sigmoid-like transitions indicating good cluster separation.
- **SC-007**: All output files are saved successfully and can be reloaded for further analysis; results parquet file is loadable with pandas and contains expected columns.
- **SC-008**: Pipeline produces reproducible results when run multiple times with same random seed.
- **SC-009**: All visualization plots are saved as image files and clearly display the relevant metrics (BIC values, posterior probabilities, cluster assignments).
