# Quickstart: Baden-Method RGC Clustering Pipeline

**Branch**: `001-baden-rgc-clustering` | **Date**: 2026-01-06

## Overview

This guide shows how to run the Baden-method RGC clustering pipeline to classify retinal ganglion cells into functional subtypes.

## Prerequisites

### Required Python Packages

```bash
pip install pandas numpy scipy scikit-learn umap-learn matplotlib joblib tqdm
```

### Required Input Data

The pipeline requires a parquet file with the following columns:
- `freq_step_5st_3x` (chirp traces)
- `green_blue_3s_3i_3x` (color traces)
- `corrected_moving_h_bar_s5_d8_3x_*` (8 moving bar direction traces)
- `sta_time_course` (RF time course)
- `step_up_QI` (quality index)
- `ds_p_value` (direction selectivity p-value)
- `axon_type` (cell type)

Default input: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet`

---

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```python
from dataframe_phase.classification_v2.Baden_method.pipeline import run_baden_pipeline

# Run with defaults
results = run_baden_pipeline(
    random_seed=42
)

print(f"DS clusters: {results['ds']['optimal_k']}")
print(f"Non-DS clusters: {results['nds']['optimal_k']}")
```

### Option 2: Run with Custom Parameters

```python
from dataframe_phase.classification_v2.Baden_method.pipeline import run_baden_pipeline

results = run_baden_pipeline(
    input_path="path/to/your/data.parquet",
    output_dir="path/to/output/",
    k_max_ds=40,           # Max clusters for DS cells
    k_max_nds=80,          # Max clusters for non-DS cells
    qi_threshold=0.5,      # Quality index threshold
    ds_p_threshold=0.05,   # DS p-value threshold
    random_seed=42,
)
```

---

## Step-by-Step Usage

### 1. Load and Preprocess Data

```python
from dataframe_phase.classification_v2.Baden_method import preprocessing

# Load data
df = preprocessing.load_data(
    "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet"
)
print(f"Loaded {len(df)} cells")

# Filter rows (NaN, QI, axon_type)
df_filtered = preprocessing.filter_rows(df, qi_threshold=0.5)
print(f"After filtering: {len(df_filtered)} cells")

# Split DS vs non-DS
df_ds, df_nds = preprocessing.split_ds_nds(df_filtered, p_threshold=0.05)
print(f"DS cells: {len(df_ds)}, non-DS cells: {len(df_nds)}")

# Preprocess traces (low-pass filter, baseline subtract, normalize)
df_ds = preprocessing.preprocess_traces(df_ds)
df_nds = preprocessing.preprocess_traces(df_nds)
```

### 2. Extract Features

```python
from dataframe_phase.classification_v2.Baden_method import features

# Extract 40D feature matrix for DS cells
X_ds, feature_names, models_ds = features.build_feature_matrix(
    df_ds,
    return_models=True
)
print(f"DS feature matrix shape: {X_ds.shape}")  # (N_ds, 40)

# Standardize features
X_ds_std, scaler_ds = features.standardize_features(X_ds)

# Same for non-DS
X_nds, _, models_nds = features.build_feature_matrix(df_nds, return_models=True)
X_nds_std, scaler_nds = features.standardize_features(X_nds)
```

### 3. Cluster with GMM + BIC Selection

```python
from dataframe_phase.classification_v2.Baden_method import clustering

# Fit GMMs and select optimal k for DS
bic_table_ds, optimal_k_ds = clustering.select_k_by_bic(
    X_ds_std,
    k_grid=range(1, 31),  # k_max=30 for DS
    n_init=20,
    reg_covar=1e-5,
    random_state=42,
)
print(f"DS optimal k: {optimal_k_ds}")

# Fit final model and get predictions
gmm_ds = clustering.fit_final_gmm(X_ds_std, optimal_k_ds)
labels_ds = gmm_ds.predict(X_ds_std)
posteriors_ds = gmm_ds.predict_proba(X_ds_std)
```

### 4. Evaluate Cluster Quality

```python
from dataframe_phase.classification_v2.Baden_method import evaluation

# Compute posterior curves
curves_ds = evaluation.compute_posterior_curves(labels_ds, posteriors_ds)

# Bootstrap stability analysis
stability_ds, all_corrs_ds = evaluation.bootstrap_stability(
    X_ds_std,
    k=optimal_k_ds,
    n_iter=20,
    frac=0.9,
    random_state=42,
)
print(f"DS bootstrap stability: {stability_ds:.3f}")
```

### 5. Visualize Results

```python
from dataframe_phase.classification_v2.Baden_method import visualization

# Plot BIC curve
visualization.plot_bic_curve(
    bic_table_ds,
    population_name="DS",
    save_path="plots/bic_curves_ds.png"
)

# Plot posterior curves
visualization.plot_posterior_curves(
    curves_ds,
    population_name="DS",
    save_path="plots/posterior_curves_ds.png"
)

# Plot UMAP clusters
visualization.plot_umap_clusters(
    X_ds_std,
    labels_ds,
    population_name="DS",
    save_path="plots/umap_clusters_ds.png"
)
```

### 6. Save Results

```python
import pandas as pd
import joblib

# Create results DataFrame
results_df = pd.DataFrame({
    'cell_id': df_ds.index,
    'population': 'DS',
    'cluster_label': labels_ds,
    'posterior_probability': posteriors_ds.max(axis=1),
})

# Save to parquet
results_df.to_parquet("results/clustering_results.parquet")

# Save models
joblib.dump(gmm_ds, "models/ds_gmm.pkl")
joblib.dump(scaler_ds, "models/ds_scaler.pkl")
```

---

## Output Files

After running the pipeline, you'll find:

```
dataframe_phase/classification_v2/Baden_method/
├── results/
│   ├── clustering_results.parquet   # Cell labels and posteriors
│   ├── bic_table.parquet            # BIC values for all k
│   └── stability_metrics.json       # Bootstrap stability results
├── models/
│   ├── ds_gmm.pkl                   # DS GMM model
│   ├── nds_gmm.pkl                  # Non-DS GMM model
│   └── ...                          # sPCA/PCA/scaler models
└── plots/
    ├── bic_curves_ds.png
    ├── bic_curves_nds.png
    ├── posterior_curves_ds.png
    ├── posterior_curves_nds.png
    ├── umap_clusters_ds.png
    └── umap_clusters_nds.png
```

---

## Configuration Reference

### Default Parameters (in `config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_PATH` | `dataframe_phase/extract_feature/...parquet` | Input data path |
| `OUTPUT_DIR` | `dataframe_phase/classification_v2/Baden_method` | Output directory |
| `SAMPLING_RATE` | 60 | Hz, trace sampling rate |
| `LOWPASS_CUTOFF` | 10 | Hz, filter cutoff frequency |
| `BASELINE_SAMPLES` | 8 | Samples for baseline median |
| `QI_THRESHOLD` | 0.5 | Minimum step_up_QI |
| `DS_P_THRESHOLD` | 0.05 | DS classification threshold |
| `K_MAX_DS` | 30 | Max clusters for DS |
| `K_MAX_NDS` | 60 | Max clusters for non-DS |
| `GMM_N_INIT` | 20 | GMM restart count |
| `GMM_REG_COVAR` | 1e-5 | Covariance regularization |
| `BOOTSTRAP_N_ITER` | 20 | Stability iterations |
| `BOOTSTRAP_FRAC` | 0.9 | Subsample fraction |

### Sparse PCA Configuration

| Stimulus | n_components | top_k (non-zero bins) |
|----------|--------------|----------------------|
| Chirp | 20 | 10 |
| Color | 6 | 10 |
| Bar time course | 8 | 5 |
| Bar derivative | 4 | 6 |
| RF time course | 2 (regular PCA) | N/A |

---

## Troubleshooting

### Common Issues

**"ValueError: No valid cells after filtering"**
- Check that input parquet has required columns
- Verify QI threshold isn't too strict
- Ensure some cells have `axon_type == "rgc"`

**"BIC keeps decreasing up to k_max"**
- Increase k_max parameter
- Check if data has natural cluster structure
- Consider if population is too small

**"Bootstrap stability < 0.90"**
- Clusters may be unstable - consider using fewer clusters
- Check if optimal k is too high for sample size
- Review posterior curves for overlapping clusters

### Performance Tips

- For large datasets (>50k cells), consider using `MiniBatchSparsePCA`
- UMAP computation can be slow - use `n_jobs=-1` for parallelization
- Bootstrap stability is the slowest step - reduce `n_iter` for quick testing

