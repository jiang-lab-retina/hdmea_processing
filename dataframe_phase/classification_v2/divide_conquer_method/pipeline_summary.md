# DEC-Refined RGC Subtype Clustering Pipeline

## Overview

This pipeline classifies retinal ganglion cells (RGCs) into subtypes using a divide-and-conquer strategy with Deep Embedding Clustering (DEC) refinement. Cells are first separated into three functional groups based on direction/orientation selectivity, then clustered independently within each group.

## Pipeline Architecture

```
Input Parquet
     |
     v
[Step 1] Load & Filter Data
     |
     v
[Step 2] Assign Groups (DSGC / OSGC / Other)
     |
     +---> Per Group:
           |
           v
     [Step 3] Preprocess Traces
           |
           v
     [Step 4] Train Autoencoder
           |
           v
     [Step 5] Extract Embeddings
           |
           v
     [Step 6] GMM/BIC k-Selection
           |
           v
     [Step 7] DEC Refinement
           |
           v
     [Step 8] Evaluation & Plots
```

## Input

- **File**: `firing_rate_with_all_features_loaded_extracted_area20260205.parquet`
- **Raw cells**: 46,992 units across 322 batches

## Step 1: Load and Filter Data

**Module**: `data_loader.py`

Applies sequential quality filters:

| Filter | Criterion | Removed |
|--------|-----------|---------|
| RGC filter | `axon_type` must be RGC | 29,574 |
| NaN ds_p_value | `ds_p_value` must not be NaN | 446 |
| None traces | `iprgc_test` must not be None | 5 |
| Low QI | `step_up_QI > 0.5` | 1,271 |
| High baseline | `baseline < 200 Hz` | 1 |
| **Total remaining** | | **15,695 cells** |

## Step 2: Assign Groups

**Module**: `grouping.py`

Cells are divided into three functional groups using DS > OS priority:

| Group | Criterion | Cells | % |
|-------|-----------|-------|---|
| DSGC | `ds_p_value < 0.05` | 4,581 | 29.2% |
| OSGC | `os_p_value < 0.05` (and not DS) | 2,735 | 17.4% |
| Other | Neither DS nor OS | 8,379 | 53.4% |

- 420 cells are both DS and OS; these are assigned to DSGC (DS > OS priority)

## Step 3: Preprocess Traces

**Module**: `preprocessing.py`

Per-segment preprocessing for each trace type:

- **Lowpass filtering**: Default 240 Hz cutoff, ipRGC traces at 10 Hz
- **Downsampling**: Default 60 Hz target rate, ipRGC at 2 Hz
- **Edge cropping**: Removes filter artifacts from segment boundaries
- **Normalization**: Z-score normalization per segment
- **Moving bar concatenation**: 8 direction traces concatenated into single segment

## Step 4: Train Autoencoder

**Module**: `train.py`

Multi-segment temporal convolutional network (TCN) autoencoder:

| Parameter | Value |
|-----------|-------|
| Encoder type | TCN |
| Total latent dim | 49 |
| Epochs | 150 |
| Batch size | 128 |
| Learning rate | 1e-4 |
| Loss | MSE reconstruction |
| Early stopping | Yes (patience-based) |

Each trace segment has its own encoder/decoder branch with segment-specific latent dimensions. The latent vectors are concatenated to form the final embedding.

## Step 5: Extract Embeddings

**Module**: `embed.py`

- Extracts latent embeddings from trained autoencoder
- Standardizes embeddings (zero mean, unit variance)
- Saves both initial and DEC-refined embeddings

## Step 6: GMM/BIC k-Selection

**Module**: `clustering/gmm_bic.py`

Selects optimal number of clusters using Gaussian Mixture Models:

| Parameter | Value |
|-----------|-------|
| Covariance type | Diagonal |
| k range | 2 to K_MAX (group-dependent) |
| n_init | 20 |
| Regularization | 1e-1 |
| Selection method | Elbow (BIC curve) |

The elbow method identifies where BIC improvement becomes marginal (threshold: 1%).

## Step 7: DEC Refinement

**Module**: `clustering/dec_refine.py`

Iteratively refines both embeddings and cluster assignments using IDEC-style training:

| Parameter | Value |
|-----------|-------|
| Max iterations | 50 |
| Update interval | 10 |
| Convergence threshold | 0.001 |
| Reconstruction weight | 0.01 |
| Alpha (Student-t df) | 1.0 |

**Training loop**:

1. Compute soft assignments using Student-t kernel
2. Derive target distribution P (sharpened soft assignments)
3. Minimize KL divergence (clustering loss) + MSE (reconstruction loss)
4. Check convergence (< 0.1% label changes between updates)

## Step 8: Evaluation and Validation

### ipRGC Purity Validation

**Module**: `validation/iprgc_metrics.py`

Uses ipRGC cells (identified by `iprgc_2hz_QI > 0.8`) as internal validation. A good clustering should concentrate ipRGC cells into a small number of clusters.

- **Purity score**: Measures how well clusters separate ipRGC vs non-ipRGC cells
- **Enrichment score**: Per-cluster ipRGC enrichment relative to baseline prevalence

### Mosaic Validation

**Module**: `validation/mosaic_validation.py`

Validates that each subtype has sufficient receptive field (RF) coverage to tile the retina:

$$\text{rf\_coverage} = \frac{\text{median\_rf\_area} \times n_{\text{cells}} / \text{conversion\_factor}}{\text{total\_area\_mm}^2}$$

$$\text{conversion\_factor} = \frac{\text{recorded\_RGCs}}{\text{total\_area\_mm}^2 \times 3000}$$

| Constant | Value | Description |
|----------|-------|-------------|
| NORMAL_RGC_PER_MM2 | 3,000 | Expected RGC density |
| ELECTRODE_PIXELS_PER_MM | 65 | Electrode array resolution |
| NOISE_PIXELS_PER_MM | 15 | Visual stimulus resolution |
| RF_COVERAGE_THRESHOLD | 0.8 | Minimum coverage to pass |

**Current results**: 37/41 subtypes pass mosaic validation (90.2%)

## Output Structure

```
divide_conquer_method/
  results/
    {GROUP}/
      cluster_assignments.parquet    # Final cluster labels
      k_selection.json               # BIC values and k*
      iprgc_validation.json          # ipRGC purity metrics
      comparison.json                 # GMM vs DEC comparison
      embeddings_initial.npy         # Pre-DEC embeddings
      embeddings_dec_refined.npy     # Post-DEC embeddings
  models/
    {GROUP}/
      autoencoder_best.pt            # Best AE checkpoint
  plots/
    {GROUP}/
      ...                             # UMAP, BIC, prototype plots
  validation/
    result/
      mosaic_validation_results.parquet
      mosaic_summary.csv
      nan_validation_report.csv
    figure/
      rf_coverage_by_subtype.png
      validation_heatmap.png
```

## Usage

```bash
# Run all groups
python -m divide_conquer_method.run_pipeline --all-groups

# Run single group
python -m divide_conquer_method.run_pipeline --group DSGC

# Skip training (load checkpoint)
python -m divide_conquer_method.run_pipeline --skip-training --all-groups

# Regenerate plots only
python -m divide_conquer_method.run_pipeline --visualize-only

# Test with subset
python -m divide_conquer_method.run_pipeline --group DSGC --subset 500
```

## Module Dependency Graph

```
config.py
  |
  +-- data_loader.py
  |     |
  +-- grouping.py
  |     |
  +-- preprocessing.py
  |     |
  +-- models/ (autoencoder, encoders, decoders, dec)
  |     |
  +-- train.py
  |     |
  +-- embed.py
  |     |
  +-- clustering/
  |     +-- gmm_bic.py
  |     +-- dec_refine.py
  |     |
  +-- validation/
  |     +-- iprgc_metrics.py
  |     +-- mosaic_validation.py
  |     |
  +-- evaluation.py
  |     |
  +-- visualization.py
  |
  +-- run_pipeline.py  (orchestrator)
```
