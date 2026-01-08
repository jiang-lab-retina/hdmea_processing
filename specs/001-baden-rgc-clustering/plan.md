# Implementation Plan: Baden-Method RGC Clustering Pipeline

**Branch**: `001-baden-rgc-clustering` | **Date**: 2026-01-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-baden-rgc-clustering/spec.md`

## Summary

Implement an unsupervised RGC classification pipeline reproducing the Baden et al. methodology. The pipeline extracts a 40-dimensional feature vector from four visual stimuli (chirp, color, moving bar, RF time course) using sparse PCA with enforced sparsity constraints, clusters cells using diagonal-covariance GMMs with BIC-based model selection, and evaluates cluster quality through posterior probability analysis and bootstrap stability testing. DS and non-DS cells are processed independently.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: 
- pandas, numpy (data handling)
- scipy (signal processing: Butterworth filter, SVD)
- scikit-learn (SparsePCA, PCA, GaussianMixture)
- umap-learn (2D projections for visualization)
- matplotlib (plotting)
- joblib (model persistence)

**Storage**: Parquet (tabular data), pickle/joblib (trained models)  
**Testing**: pytest with synthetic data fixtures  
**Target Platform**: Windows/Linux workstation  
**Project Type**: Single standalone pipeline (not part of hdmea package)  
**Performance Goals**: Full pipeline completes within 30 minutes for ~47k cells  
**Constraints**: Memory-efficient processing, reproducible with random seed  
**Scale/Scope**: ~47,000 cells input, ~40,000 after filtering, 40 features each

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ COMPLIANT | Standalone pipeline in `dataframe_phase/classification_v2/Baden_method/` - acceptable for downstream analysis scripts |
| II. Modular Subpackage Layout | ✅ COMPLIANT | Will use modular file structure within output directory |
| III. Explicit I/O and Pure Functions | ✅ COMPLIANT | All functions will have explicit inputs/outputs, config via parameters |
| IV. Single HDF5 Artifact | N/A | Consumes parquet, outputs parquet - not raw recording processing |
| V. Data Format Standards | ✅ COMPLIANT | Parquet for tabular data, JSON for config |
| VI. No Hidden Global State | ✅ COMPLIANT | No module-level mutable state, logger via `logging.getLogger(__name__)` |
| VII. Independence from Legacy Code | ✅ COMPLIANT | No imports from `Legacy_code/` |

**Gate Status**: ✅ PASSED - No violations requiring justification

## Project Structure

### Documentation (this feature)

```text
specs/001-baden-rgc-clustering/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0: Research decisions
├── data-model.md        # Phase 1: Data structures
├── quickstart.md        # Phase 1: Usage guide
└── checklists/
    └── requirements.md  # Quality checklist
```

### Source Code

```text
dataframe_phase/classification_v2/Baden_method/
├── __init__.py              # Package marker
├── config.py                # Configuration constants and defaults
├── preprocessing.py         # FR-001 to FR-010: Data loading, filtering, signal conditioning
├── features.py              # FR-011 to FR-019: Sparse PCA, PCA, feature extraction
├── clustering.py            # FR-020 to FR-025: GMM fitting, BIC selection
├── evaluation.py            # FR-026 to FR-030: Posterior curves, bootstrap stability
├── visualization.py         # FR-037 to FR-040: Plotting functions
├── pipeline.py              # Main orchestration: run_baden_pipeline()
├── models/                  # Saved model artifacts
│   ├── ds_gmm.pkl
│   ├── nds_gmm.pkl
│   ├── ds_spca_*.pkl
│   └── nds_spca_*.pkl
├── plots/                   # Generated visualizations
│   ├── bic_curves_ds.png
│   ├── bic_curves_nds.png
│   ├── posterior_curves_ds.png
│   ├── posterior_curves_nds.png
│   ├── umap_clusters_ds.png
│   └── umap_clusters_nds.png
└── results/                 # Output data
    ├── clustering_results.parquet
    ├── bic_table.parquet
    └── stability_metrics.json
```

**Structure Decision**: Single project with modular Python files. Each module handles a specific pipeline stage, with `pipeline.py` orchestrating the full workflow. This follows the constitution's preference for explicit I/O and pure functions while keeping the implementation self-contained.

## Module Responsibilities

### `config.py`
- Default paths (input parquet, output directory)
- Filter thresholds (step_up_QI > 0.5, axon_type == "rgc")
- DS split threshold (ds_p_value < 0.05)
- Signal processing parameters (10 Hz cutoff, 60 Hz sampling rate, baseline samples = 8)
- Sparse PCA specifications (n_components, top-k sparsity per stimulus)
- GMM parameters (n_init=20, reg_covar=1e-5, k_max_ds=30, k_max_nds=60)
- Bootstrap parameters (n_iterations=20, sample_fraction=0.9)

### `preprocessing.py`
- `load_data(path)`: Load parquet, return DataFrame
- `filter_rows(df)`: Apply NaN, QI, axon_type filters
- `split_ds_nds(df)`: Split by ds_p_value threshold
- `lowpass_filter(trace, fs, cutoff)`: Zero-phase Butterworth filter
- `baseline_subtract(trace, n_samples)`: Subtract median of first n samples
- `normalize_trace(trace, eps)`: Max-absolute normalization
- `preprocess_traces(df)`: Apply all signal conditioning

### `features.py`
- `fit_sparse_pca(X, n_components, top_k)`: Fit sparse PCA with hard sparsity
- `extract_chirp_features(df)`: 20 features from freq_step_5st_3x
- `extract_color_features(df)`: 6 features from green_blue_3s_3i_3x
- `extract_bar_svd(df)`: SVD on 8-direction matrix → time course
- `extract_bar_features(df)`: 8 features from time course, 4 from derivative
- `extract_rf_features(df)`: 2 PCA features from sta_time_course
- `build_feature_matrix(df)`: Concatenate all → 40D matrix
- `standardize_features(X)`: Z-score normalization

### `clustering.py`
- `fit_gmm_bic(X, k_grid, n_init, reg_covar)`: Fit GMMs, return BIC table
- `select_optimal_k(bic_table)`: Find k with minimum BIC
- `compute_log_bayes_factors(bic_table)`: Compute LBF between adjacent k
- `fit_final_gmm(X, k, n_init, reg_covar)`: Fit final model with selected k
- `predict_clusters(gmm, X)`: Return labels and posteriors

### `evaluation.py`
- `compute_posterior_curves(labels, posteriors)`: Rank-ordered curves per cluster
- `bootstrap_stability(X, k, n_iter, frac)`: Resample, refit, match, correlate
- `match_cluster_centers(original, bootstrap)`: Maximum correlation matching

### `visualization.py`
- `plot_bic_curve(bic_table, population_name, save_path)`
- `plot_posterior_curves(curves, population_name, save_path)`
- `plot_umap_clusters(X, labels, population_name, save_path)`

### `pipeline.py`
- `run_baden_pipeline(input_path, output_dir, random_seed)`: Full orchestration
- Logging with progress updates
- Error handling with informative messages
- Results aggregation and saving

## Complexity Tracking

> No violations requiring justification. Constitution gates passed.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Standalone vs Package | Standalone in dataframe_phase/ | Downstream analysis, not core hdmea functionality |
| Feature extraction | Per-stimulus modules | Clear separation, easier testing |
| Model persistence | joblib pickle | scikit-learn standard, efficient for sklearn objects |

