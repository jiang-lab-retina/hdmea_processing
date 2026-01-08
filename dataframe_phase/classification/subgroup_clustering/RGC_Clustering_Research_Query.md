# Research Query: Improving Unsupervised Clustering of Retinal Ganglion Cell Subtypes

## 1. Background

### 1.1 Scientific Context

We are analyzing **retinal ganglion cells (RGCs)** from multi-electrode array (MEA) recordings. RGCs are the output neurons of the retina that transmit visual information to the brain. There are approximately **40-50 known RGC subtypes** in the mammalian retina, each with distinct functional properties.

The goal is to **automatically cluster RGCs into biologically meaningful subtypes** based on their electrophysiological response patterns to visual stimuli.

### 1.2 Current Workflow

1. **Pre-classification**: RGCs are first separated into 4 broad functional groups based on established criteria:
   - **ipRGC** (intrinsically photosensitive RGCs): Identified by ipRGC Quality Index > 0.8
   - **DSGC** (Direction-Selective GCs): Identified by Direction Selectivity Index p-value < 0.05
   - **OSGC** (Orientation-Selective GCs): Identified by Orientation Selectivity Index p-value < 0.05
   - **Other**: Remaining RGCs that don't fit above criteria

2. **Sub-clustering**: Within each group, we want to find distinct subtypes (e.g., 6-10 ipRGC subtypes, 8-12 DSGC subtypes, etc.)

### 1.3 Biological Expectations

| Subgroup | Sample Size | Expected Clusters | Biological Rationale |
|----------|-------------|-------------------|----------------------|
| ipRGC | 871 | 6-10 | Known subtypes: M1-M6 + additional types |
| DSGC | 1,849 | 8-12 | ON, ON-OFF, OFF types × 4 preferred directions |
| OSGC | 1,801 | 4-12 | ON, OFF types × multiple orientations |
| Other | 3,523 | 12-24 | Diverse population including alpha, beta, gamma types |

---

## 2. Data Structure

### 2.1 Input Features

Each RGC unit has a **concatenated response trace** of length **11,403 time points** (firing rate over time). This trace is formed by concatenating the mean responses to 13 different visual stimuli:

```
Stimuli concatenated:
1. freq_step_5st_3x          - Frequency step stimulus
2. green_blue_3s_3i_3x       - Color (chromatic) stimulus
3. step_up_5s_5i_b0_3x       - Step-up light stimulus (low intensity)
4. step_up_5s_5i_b0_30x      - Step-up light stimulus (high intensity)
5-12. moving_h_bar_s5_d8_3x_{0,45,90,135,180,225,270,315}  
                              - Moving bar in 8 directions
13. iprgc_test               - Sustained light for ipRGC identification
```

### 2.2 Data Dimensions

| Subgroup | Samples (N) | Features (L) | Data Shape |
|----------|-------------|--------------|------------|
| ipRGC | 871 | 11,403 | (871, 11403) |
| DSGC | 1,849 | 11,403 | (1849, 11403) |
| OSGC | 1,801 | 11,403 | (1801, 11403) |
| Other | 3,523 | 11,403 | (3523, 11403) |

### 2.3 Data Characteristics

- **High dimensionality**: 11,403 features per sample
- **Temporal structure**: Data is a time series (firing rate vs time)
- **Multi-stimulus**: Responses to different stimuli are concatenated
- **Noise**: Biological variability + recording noise
- **Class imbalance within clusters**: Some subtypes are rare

---

## 3. Approaches Tried

### 3.1 Approach 1: Standard Autoencoder + GMM

**Method:**
- 1D Convolutional Autoencoder (3 layers: 1→16→32→64 channels)
- Latent dimension: 100
- Training: MSE reconstruction loss, Adam optimizer
- Clustering: Gaussian Mixture Model with BIC for k selection

**Result:** Poor cluster separation, GMM often failed with singular covariance matrices.

### 3.2 Approach 2: Supervised Contrastive Autoencoder

**Method:**
- Added projection head after encoder
- Combined loss: MSE reconstruction + contrastive loss (using 4 pre-defined subgroup labels)
- Temperature-scaled contrastive loss

**Result:** Improved separation between 4 main groups, but did not help with sub-clustering within groups.

### 3.3 Approach 3: Variational Autoencoder (VAE) + K-Means

**Method:**
- VAE with KL divergence regularization
- Latent space should be more structured
- K-Means clustering on latent codes

**Result:** Similar performance to standard AE.

### 3.4 Approach 4: Deep Embedded Clustering (DEC)

**Method:**
- Pre-train autoencoder
- Initialize cluster centers with K-Means
- Fine-tune with combined reconstruction + KL divergence clustering loss
- Soft cluster assignments using Student's t-distribution

**Result:** Did not significantly improve over simpler methods.

### 3.5 Approach 5: Optuna Hyperparameter Optimization (Current Best)

**Method:**
- Bayesian optimization over hyperparameter space:
  - Latent dimension: 32-256
  - Learning rate: 1e-5 to 1e-3
  - Network depth: 2-4 conv layers
  - Base channels: 16, 32, 64
  - Dropout: 0.0-0.5
  - Weight decay: 1e-6 to 1e-3
- Objective: Maximize silhouette score
- 50 trials per subgroup, 2-hour timeout

**Optimized Parameters Found:**

| Subgroup | Latent Dim | LR | Layers | Channels | Dropout | k |
|----------|------------|-------------|--------|----------|---------|---|
| ipRGC | 256 | 4.0e-05 | 3 | 32 | 0.39 | 9 |
| DSGC | 32 | 1.3e-04 | 3 | 16 | 0.23 | 8 |
| OSGC | 64 | 1.2e-05 | 2 | 16 | 0.26 | 7 |
| Other | 32 | 1.6e-04 | 3 | 16 | 0.43 | 12 |

**Current Results:**

| Subgroup | k Found | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|----------|---------|------------|-------------------|----------------|
| ipRGC | 9 | **0.131** | 85.5 | 1.92 |
| DSGC | 8 | **0.166** | 320.7 | 1.84 |
| OSGC | 7 | **0.291** | 740.9 | 1.22 |
| Other | 12 | **0.172** | 566.2 | 1.69 |

---

## 4. The Problem

### 4.1 Primary Issue: Low Silhouette Scores

The silhouette scores are **too low** for confident cluster assignments:

- **Silhouette interpretation:**
  - 0.7-1.0: Strong cluster structure
  - 0.5-0.7: Reasonable structure
  - 0.25-0.5: Weak structure, clusters may be artificial
  - **< 0.25: No substantial structure** ← Most of our results

- **Current scores:**
  - ipRGC: 0.131 (very weak)
  - DSGC: 0.166 (very weak)
  - OSGC: 0.291 (weak but best)
  - Other: 0.172 (very weak)

### 4.2 Visual Observation

UMAP visualizations show:
- Clusters are **overlapping** rather than separated
- No clear boundaries between clusters
- Points form a **continuous manifold** rather than discrete groups

### 4.3 Possible Causes

1. **Continuous biological variation**: RGC subtypes may exist on a continuum rather than discrete clusters
2. **Feature representation inadequacy**: Concatenated traces may not capture discriminative features
3. **Noise overwhelming signal**: Recording noise may mask subtype differences
4. **Wrong stimuli combination**: Some stimuli may not be informative for certain subtypes
5. **Inappropriate dimensionality reduction**: Information loss during encoding
6. **Temporal alignment issues**: Responses may have variable latencies

---

## 5. What We Want to Achieve

### 5.1 Target Metrics

- **Silhouette score > 0.4** for all subgroups
- **Clear visual separation** in UMAP/t-SNE plots
- **Biologically interpretable clusters** that correspond to known RGC types

### 5.2 Constraints

- Must work in **unsupervised** or **semi-supervised** setting (no ground truth labels for subtypes)
- Must handle **high-dimensional temporal data** (11,403 features)
- Must produce **discrete cluster assignments**
- Should be **reproducible**

---

## 6. Questions for Research

1. **Feature Engineering**: What preprocessing or feature extraction methods work best for temporal neural response data?
   - Should we use wavelet transforms, spectrograms, or hand-crafted features?
   - Should different stimuli be weighted differently?

2. **Architecture**: What neural network architectures are best for clustering time-series data?
   - Temporal Convolutional Networks (TCN)?
   - Transformer-based autoencoders?
   - Recurrent architectures (LSTM/GRU)?

3. **Clustering Approach**: What clustering methods work best for biological data with potential continuous variation?
   - Spectral clustering?
   - HDBSCAN with density-based approach?
   - Hierarchical clustering?
   - Mixture of factor analyzers?

4. **Loss Functions**: What training objectives encourage better cluster separation?
   - Deep clustering losses (DEC, IDEC, DCN)?
   - Contrastive learning without labels (SimCLR, MoCo)?
   - Information-theoretic losses?

5. **Data Augmentation**: How can we augment temporal neural data to improve representations?
   - Time warping?
   - Adding noise?
   - Cropping/masking?

6. **Evaluation**: Given the lack of ground truth, how do we validate clusters are biologically meaningful?
   - Cross-validation strategies?
   - Biological prior constraints?

---

## 7. Technical Environment

- **Language**: Python 3.11
- **Deep Learning**: PyTorch 2.x with CUDA (RTX 5090 GPU)
- **Optimization**: Optuna for hyperparameter search
- **Visualization**: UMAP, t-SNE, matplotlib
- **Clustering**: scikit-learn (K-Means, GMM)

---

## 8. Code Architecture

```
dataframe_phase/classification/subgroup_clustering/
├── config.py                    # Hyperparameters and settings
├── data_loader.py               # Load and preprocess data
├── models.py                    # AE, VAE, DEC architectures
├── trainers.py                  # Training loops
├── clustering.py                # K-Means, GMM clustering
├── hyperparameter_optimization.py  # Optuna optimization
├── run_optimized.py             # Main pipeline
└── validation/
    └── visualize_optimized.py   # UMAP plots
```

---

## 9. Summary

We are trying to cluster retinal ganglion cells into biologically meaningful subtypes based on their electrophysiological responses. Despite trying multiple approaches (AE, VAE, DEC, supervised contrastive learning, hyperparameter optimization), we achieve silhouette scores of only 0.13-0.29, indicating poor cluster separation.

**We need guidance on:**
1. Better feature representations for temporal neural data
2. More appropriate clustering methods for biological data
3. Alternative deep learning architectures
4. Strategies to handle potentially continuous biological variation

The ideal solution should produce well-separated clusters (silhouette > 0.4) that can be validated against known RGC subtype biology.

