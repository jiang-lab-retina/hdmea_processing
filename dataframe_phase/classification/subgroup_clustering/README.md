# RGC Subgroup Clustering Pipeline

This pipeline clusters retinal ganglion cells (RGCs) into biologically meaningful subtypes using convolutional autoencoders and K-Means clustering.

## Overview

The pipeline:
1. Separates RGCs into 4 subgroups: **ipRGC**, **DSGC**, **OSGC**, **Other**
2. Uses subgroup-specific movie traces (ipRGC includes `iprgc_test`, others exclude it)
3. Trains separate autoencoders for each subgroup
4. Optimizes hyperparameters using Optuna (Bayesian optimization)
5. Clusters using K-Means within biologically expected k ranges
6. Optionally applies **Contrastive Learning** to improve cluster separation
7. Generates UMAP visualizations

### Approaches Available

| Approach | Description |
|----------|-------------|
| **Optimized AE** | Standard autoencoder with optimized hyperparameters |
| **Contrastive AE** | Uses supervised contrastive loss to improve cluster separation |

The contrastive approach uses initial cluster labels (from Optimized AE) to train a contrastive autoencoder that pulls same-cluster samples together and pushes different-cluster samples apart in the latent space.

---

## Quick Start

### Run Everything (Recommended for First Time)

```powershell
# Full optimization (~8 hours total, 2h per subgroup)
python -m dataframe_phase.classification.subgroup_clustering.run_optimized

# Then visualize
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

---

## Commands Reference

### 1. Full Hyperparameter Optimization

**When to use:** First time running, or when you want to find the best hyperparameters.

```powershell
python -m dataframe_phase.classification.subgroup_clustering.run_optimized
```

| Parameter | Value |
|-----------|-------|
| Trials per subgroup | 50 |
| Timeout per subgroup | 2 hours |
| Total time | ~8 hours |

### 2. Quick Optimization (Testing)

**When to use:** Testing changes, quick experiments.

```powershell
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --quick
```

| Parameter | Value |
|-----------|-------|
| Trials per subgroup | 15 |
| Timeout per subgroup | 30 minutes |
| Total time | ~2 hours |

### 3. Optimize Single Subgroup

**When to use:** Re-running optimization for one specific subgroup.

```powershell
# Examples
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup ipRGC
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup DSGC --quick
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup OSGC
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup Other --quick
```

**Note:** Results are automatically merged with existing results from other subgroups.

### 4. Use Saved Parameters (Skip Optimization)

**When to use:** You already have optimized parameters in `config.py` and just want to re-train/cluster.

```powershell
python -m dataframe_phase.classification.subgroup_clustering.run_with_saved_params
```

| Parameter | Value |
|-----------|-------|
| Uses | `OPTIMIZED_PARAMS` from `config.py` |
| Total time | ~10 minutes |

### 5. Skip Optimization (Use Existing Optuna Results)

**When to use:** You ran optimization before and want to re-train with those parameters.

```powershell
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --skip-optimization
```

This loads parameters from `optuna_studies/{subgroup}_best_params.json`.

### 6. Generate Visualizations Only

**When to use:** After running optimization, to create/update plots.

```powershell
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

### 7. Run with Contrastive Learning

**When to use:** To improve cluster separation using contrastive learning.

```powershell
# Run both Optimized AE and Contrastive AE
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --skip-optimization

# Run contrastive only (requires existing optimized results)
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --skip-optimization --contrastive-only
```

### 8. Run without Contrastive Learning

**When to use:** Skip contrastive approach for faster execution.

```powershell
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --skip-optimization --no-contrastive
```

---

## Common Scenarios

### Scenario 1: Fresh Start

```powershell
# Clear old results (optional)
Remove-Item dataframe_phase/classification/subgroup_clustering/output/*.pkl -ErrorAction SilentlyContinue
Remove-Item dataframe_phase/classification/subgroup_clustering/output/*.json -ErrorAction SilentlyContinue

# Run full optimization
python -m dataframe_phase.classification.subgroup_clustering.run_optimized

# Visualize
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

### Scenario 2: Re-optimize One Subgroup

```powershell
# Only re-optimize DSGC (keeps other subgroups' results)
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup DSGC

# Visualize all subgroups
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

### Scenario 3: Quick Test with New Settings

```powershell
# Edit config.py with new settings, then:
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --quick

# Check results
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

### Scenario 4: Reproducible Run with Known Parameters

```powershell
# After editing OPTIMIZED_PARAMS in config.py:
python -m dataframe_phase.classification.subgroup_clustering.run_with_saved_params

# Visualize
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

### Scenario 5: Continue Interrupted Optimization

```powershell
# If optimization was interrupted, run remaining subgroups individually:
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup OSGC
python -m dataframe_phase.classification.subgroup_clustering.run_optimized --subgroup Other

# Results automatically merge with existing
python -m dataframe_phase.classification.subgroup_clustering.validation.visualize_optimized
```

---

## Output Files

### Results Directory: `output/`

| File | Description |
|------|-------------|
| `optimized_results.pkl` | All results (latents, labels, metrics) for all subgroups |
| `optimized_metrics_summary.json` | Summary metrics (silhouette, CH, DB scores) |
| `{subgroup}_optimized_results.parquet` | Per-subgroup DataFrame with cluster labels |

### Models Directory: `models/`

| File | Description |
|------|-------------|
| `{subgroup}_optimized_ae.pth` | Trained autoencoder model weights |
| `{subgroup}_contrastive_ae.pth` | Trained contrastive autoencoder model weights |

### Optuna Directory: `optuna_studies/`

| File | Description |
|------|-------------|
| `{subgroup}_best_params.json` | Best hyperparameters from optimization |
| `{subgroup}_study.pkl` | Full Optuna study object |
| `optimization_summary.json` | Summary of all optimizations |

### Plots Directory: `validation/plots/`

| File | Description |
|------|-------------|
| `{subgroup}_optimized_clusters.png` | UMAP plot for Optimized AE |
| `{subgroup}_contrastive_clusters.png` | UMAP plot for Contrastive AE |
| `{subgroup}_approach_comparison.png` | Side-by-side comparison of approaches |
| `{subgroup}_supervised_umap.png` | Supervised UMAP visualization |
| `{subgroup}_umap_comparison.png` | Unsupervised vs Supervised UMAP |
| `optimized_comparison_grid.png` | 2x2 grid comparing all subgroups |
| `all_approaches_comparison.png` | Grid comparing all subgroups x approaches |
| `optimized_metrics_summary.png` | Bar charts of clustering metrics |
| `optimized_hyperparameters.png` | Table of optimized parameters |

---

## Configuration

### Expected Cluster Ranges (`config.py`)

```python
EXPECTED_K_RANGES = {
    "ipRGC": (3, 10),   # ipRGC subtypes
    "DSGC": (4, 12),    # Direction-selective types
    "OSGC": (4, 12),    # Orientation-selective types
    "Other": (8, 24),   # Diverse population
}
```

### Subgroup-Specific Movie Columns

- **ipRGC**: Uses 13 movies (includes `iprgc_test`) → 11,403 features
- **DSGC/OSGC/Other**: Uses 12 movies (excludes `iprgc_test`) → 4,204 features

---

## Troubleshooting

### GPU Not Being Used

Check GPU availability:
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"
```

### Results Not Showing All Subgroups

Results are now merged automatically. If you see missing subgroups:
1. Check that all subgroups were run
2. Re-run missing subgroups with `--subgroup` flag
3. Re-generate visualizations

### Clear All Results and Start Fresh

```powershell
Remove-Item dataframe_phase/classification/subgroup_clustering/output/* -Recurse -ErrorAction SilentlyContinue
Remove-Item dataframe_phase/classification/subgroup_clustering/models/* -Recurse -ErrorAction SilentlyContinue
Remove-Item dataframe_phase/classification/subgroup_clustering/optuna_studies/* -Recurse -ErrorAction SilentlyContinue
Remove-Item dataframe_phase/classification/subgroup_clustering/validation/plots/* -Recurse -ErrorAction SilentlyContinue
```

---

## Metrics Interpretation

| Metric | Good Value | Description |
|--------|------------|-------------|
| Silhouette | > 0.4 | Cluster separation quality (-1 to 1) |
| Calinski-Harabasz | Higher is better | Ratio of between/within cluster dispersion |
| Davies-Bouldin | < 1.5 | Average cluster similarity (lower is better) |

