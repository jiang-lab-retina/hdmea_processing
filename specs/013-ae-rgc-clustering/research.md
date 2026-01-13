# Research & Design Decisions: AE-RGC Clustering

**Feature**: 013-ae-rgc-clustering  
**Date**: 2026-01-11

This document captures research findings and design decisions made during planning.

---

## 1. Autoencoder Architecture

### Decision: Segment-Wise Multi-Encoder Architecture

**Chosen**: Independent encoder/decoder pairs for each stimulus segment

**Rationale**:
- Each stimulus has different temporal dynamics and biological meaning
- Fixed latent dimension per segment ensures balanced feature budget
- Different segment lengths don't require padding/masking hacks
- Per-segment reconstruction loss prevents long segments from dominating

**Alternatives Considered**:

| Alternative | Why Rejected |
|-------------|--------------|
| Single concatenated encoder | Long segments dominate, fixed dim budget impossible |
| Shared encoder with segment tokens | Overly complex, harder to interpret |
| Variational autoencoder (VAE) | Added complexity, regularization fights weak supervision |

### Encoder Architecture Per Segment

```python
# Default architecture for each segment encoder
Conv1D(in_channels=1, out_channels=32, kernel_size=7, stride=2)
BatchNorm1D + ReLU
Conv1D(32, 64, kernel_size=5, stride=2)
BatchNorm1D + ReLU
Conv1D(64, 128, kernel_size=3, stride=2)
BatchNorm1D + ReLU
AdaptiveAvgPool1D(output_size=4)
Flatten
Linear(512, latent_dim)  # latent_dim varies per segment
```

**Note**: Architecture can be simplified for short segments (STA, step-up).

---

## 2. Weak Supervision Strategy

### Decision: Supervised Contrastive Loss

**Chosen**: SupCon loss with coarse group labels as positives

**Rationale**:
- Creates well-separated embedding manifolds
- Prevents mode collapse (all embeddings converging)
- Compatible with partial label omission for CV turns
- Well-studied in representation learning literature

**Implementation**:

```python
def supcon_loss(embeddings, labels, temperature=0.1):
    """
    embeddings: (N, D) normalized embeddings
    labels: (N,) integer group labels
    """
    similarity = embeddings @ embeddings.T / temperature
    # Mask: same label = positive, different = negative
    # Cross-entropy over positive pairs
    ...
```

**Alternatives Considered**:

| Alternative | Why Rejected |
|-------------|--------------|
| Center loss | Simpler but prone to collapse, weaker separation |
| Triplet loss | Requires careful mining, slower training |
| Unsupervised contrastive | Doesn't use available group labels |

---

## 3. Segment Preprocessing

### Decision: Stimulus-Specific Filter Parameters

| Segment | Low-pass | Target Rate | Notes |
|---------|----------|-------------|-------|
| freq_section_0p5hz | 10 Hz | 10 Hz | Standard Baden preprocessing |
| freq_section_1hz | 10 Hz | 10 Hz | Standard Baden preprocessing |
| freq_section_2hz | 10 Hz | 10 Hz | Standard Baden preprocessing |
| freq_section_4hz | 10 Hz | 10 Hz | Standard Baden preprocessing |
| freq_section_10hz | None | 60 Hz | Preserve high-freq dynamics |
| green_blue | 10 Hz | 10 Hz | Standard Baden preprocessing |
| bar_concat | 10 Hz | 10 Hz | 8 dirs concatenated in order |
| sta_time_course | None | 10 Hz | Already smooth, downsample only |
| iprgc_test | 2 Hz | 2 Hz | Slow dynamics, aggressive downsample |
| step_up | 10 Hz | 10 Hz | Standard Baden preprocessing |

### Segment Length Estimates (after preprocessing)

| Segment | Original Shape | After Preprocessing | Latent Dim |
|---------|----------------|---------------------|------------|
| freq_section_* (×5) | (240,) | ~40 samples @ 10Hz | 4 each |
| green_blue | (~600,) | ~100 samples @ 10Hz | 6 |
| bar_concat | 8×(~300,) | ~400 samples @ 10Hz | 12 |
| sta_time_course | (60,) | ~10 samples @ 10Hz | 3 |
| iprgc_test | (~7200,) | ~240 samples @ 2Hz | 4 |
| step_up | (~600,) | ~100 samples @ 10Hz | 4 |

---

## 4. Coarse Group Definition

### Decision: Precedence-Based Disjoint Groups

**Chosen**: Hierarchical precedence to ensure each cell belongs to exactly one group

**Default Precedence Order**:
1. `axon_type == "ac"` → **AC** (amacrine cells)
2. `iprgc_2hz_QI > 0.8` → **ipRGC** (intrinsically photosensitive)
3. `ds_p_value < 0.05` → **DS-RGC** (direction-selective RGC)
4. else → **nonDS-RGC** (non-direction-selective RGC)

**Rationale**:
- AC/RGC distinction is primary (different cell types)
- ipRGC is rare, functionally distinct
- DS/nonDS is the classic Baden split
- Configurable for alternative analyses

**Implementation**:

```python
def assign_group(row, precedence_order=None):
    if precedence_order is None:
        precedence_order = ["ac", "iprgc", "ds", "nonds"]
    
    for group in precedence_order:
        if group == "ac" and row["axon_type"] == "ac":
            return "AC"
        if group == "iprgc" and row.get("iprgc_2hz_QI", 0) > 0.8:
            return "ipRGC"
        if group == "ds" and row["ds_p_value"] < 0.05:
            return "DS-RGC"
    return "nonDS-RGC"
```

---

## 5. Clustering Strategy

### Decision: Per-Group Diagonal GMM with BIC Selection

**Chosen**: Cluster each group independently, diagonal covariance, BIC/logBF selection

**Rationale**:
- Matches Baden methodology for comparability
- Diagonal covariance prevents overfitting on 49D
- Per-group clustering guarantees group purity
- BIC provides principled model selection

**Parameters** (from Baden config):

| Parameter | Value | Notes |
|-----------|-------|-------|
| GMM covariance | diagonal | Reduced parameters |
| n_init | 20 | Multiple restarts |
| reg_covar | 1e-3 | Regularization |
| k_max (DS) | 40 | Maximum clusters |
| k_max (nonDS) | 80 | Maximum clusters |
| logBF threshold | 6.0 | Strong evidence |

---

## 6. Cross-Validation Design

### Decision: Omitted-Label Purity Turns

**Chosen**: Three CV turns, each omitting one coarse label from supervision

**CV Turn Structure**:

| Turn | Omitted Label | Active Labels | Purity Measured On |
|------|---------------|---------------|-------------------|
| 1 | axon_type | ds, iprgc | axon_type |
| 2 | ds_cell | axon_type, iprgc | ds_cell |
| 3 | iprgc | axon_type, ds | iprgc |

**Purity Calculation**:

```python
def cluster_purity(cluster_labels, true_labels):
    """
    For each cluster, find the majority true label.
    Purity = sum of majority counts / total samples
    """
    total = len(cluster_labels)
    purity_sum = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        values, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    return purity_sum / total
```

**CVScore**:

$$\text{CVScore} = \frac{1}{3} \sum_{t=1}^{3} \text{Purity}(L_t)$$

---

## 7. Bootstrap Stability

### Decision: 90% Subsampling with Centroid Correlation

**Chosen**: 20 iterations of 90% subsampling, match clusters via correlation

**Algorithm**:

```python
for b in range(n_bootstraps):
    # Sample 90% of cells
    sample_idx = np.random.choice(n_cells, size=int(0.9*n_cells), replace=False)
    
    # Refit GMM with same k*
    gmm_boot = fit_gmm(embeddings[sample_idx], k=k_star)
    
    # Compute bootstrap centroids
    centroids_boot = compute_centroids(embeddings[sample_idx], gmm_boot.labels_)
    
    # Match to reference centroids via max correlation
    correlations = []
    for ref_centroid in reference_centroids:
        best_corr = max(pearsonr(ref_centroid, boot_centroid)[0] 
                        for boot_centroid in centroids_boot)
        correlations.append(best_corr)
    
    stability_scores[b] = np.median(correlations)

# Report: mean ± std of stability_scores
```

---

## 8. Loss Weighting Strategy

### Decision: Inverse-Length Weighting for Reconstruction

**Chosen**: Weight each segment's reconstruction loss inversely by input length

**Rationale**:
- Prevents long segments (bar_concat, iprgc_test) from dominating
- Ensures all stimulus types contribute equally to representation

**Implementation**:

```python
segment_weights = {}
total_inv_len = sum(1/len(seg) for seg in segments.values())
for name, seg in segments.items():
    segment_weights[name] = (1/len(seg)) / total_inv_len

# Normalized reconstruction loss
L_rec = sum(w * mse(seg, recon) for (name, seg), (_, recon), w 
            in zip(segments.items(), reconstructions.items(), segment_weights.values()))
```

---

## 9. Hyperparameter Defaults

### Autoencoder Training

| Parameter | Default | Range for Tuning |
|-----------|---------|------------------|
| learning_rate | 1e-3 | [1e-4, 1e-2] |
| batch_size | 128 | [64, 256] |
| epochs | 100 | [50, 200] |
| β (supcon weight) | 0.1 | [0.01, 1.0] |
| τ (temperature) | 0.1 | [0.05, 0.5] |
| weight_decay | 1e-5 | [0, 1e-4] |
| dropout | 0.1 | [0, 0.3] |

### Clustering

| Parameter | Default | Notes |
|-----------|---------|-------|
| k_max | 40/80 | DS/nonDS |
| logBF_threshold | 6.0 | Strong evidence |
| n_init | 20 | GMM restarts |
| reg_covar | 1e-3 | Regularization |

### Stability

| Parameter | Default |
|-----------|---------|
| n_bootstraps | 20 |
| sample_fraction | 0.9 |
| stability_threshold | 0.8 |

---

## 10. Output Artifacts

| Artifact | Format | Location |
|----------|--------|----------|
| Trained AE model | .pt | models_saved/autoencoder_best.pt |
| Cell embeddings | .parquet | results/embeddings.parquet |
| Cluster assignments | .parquet | results/cluster_assignments.parquet |
| BIC tables | .parquet | results/bic_*.parquet |
| CV purity table | .csv | results/cv_purity.csv |
| Stability metrics | .json | results/stability_metrics.json |
| UMAP plots | .png | plots/umap_*.png |
| BIC curves | .png | plots/bic_*.png |
| Response prototypes | .png | plots/prototypes_*.png |

---

## References

1. Baden et al. (2016) - Original RGC clustering methodology
2. Khosla et al. (2020) - Supervised Contrastive Learning
3. scikit-learn GMM documentation
4. PyTorch autoencoder best practices
