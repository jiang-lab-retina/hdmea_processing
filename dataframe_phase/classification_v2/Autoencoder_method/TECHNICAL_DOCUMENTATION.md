# Autoencoder-Based Retinal Cell Type Clustering Pipeline

## Technical Documentation

**Version:** 1.0  
**Last Updated:** 2026-01-13

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Flow and Pipeline Steps](#3-data-flow-and-pipeline-steps)
4. [Configuration Parameters](#4-configuration-parameters)
5. [Model Architecture](#5-model-architecture)
6. [Loss Functions](#6-loss-functions)
7. [Clustering Algorithm](#7-clustering-algorithm)
8. [Cross-Validation](#8-cross-validation)
9. [Implementation Details](#9-implementation-details)
10. [Output Files](#10-output-files)

---

## 1. Overview

### 1.1 Purpose

This pipeline performs **unsupervised clustering of retinal cells** (Retinal Ganglion Cells and Amacrine Cells) based on their electrophysiological response profiles. The goal is to discover functional cell types from multi-segment time series data recorded during various visual stimuli.

### 1.2 Key Features

- **Multi-segment autoencoder**: Separate encoder-decoder pairs for each stimulus type
- **49-dimensional latent space**: Fixed embedding size for downstream analysis
- **Weakly supervised contrastive learning**: Uses coarse labels to guide representation learning
- **Cluster purity loss**: Directly optimizes for pure clusters during training
- **Baden-style GMM clustering**: BIC-based model selection with log Bayes factor thresholds
- **Per-group clustering**: Ensures no cluster crosses major cell type boundaries
- **GPU acceleration**: Supports CUDA for training and GMM fitting

### 1.3 Input Data

| Data Type | Description |
|-----------|-------------|
| **Parquet file** | Contains electrophysiological recordings with trace columns and metadata |
| **Trace segments** | 10 different stimulus response time series per cell |
| **Metadata** | Axon type (RGC/AC), direction selectivity p-value, ipRGC quality index |

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA                                        │
│  Parquet file with 10 trace columns per cell + metadata                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                        │
│  • Low-pass filtering (10 Hz for most, 2 Hz for ipRGC)                     │
│  • Downsampling (60 Hz → 10 Hz or 2 Hz)                                    │
│  • Segment-specific handling (10 Hz section: edge slicing only)            │
│  • NaN handling                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-SEGMENT AUTOENCODER                                │
│  ┌─────────────┐   ┌─────────────┐        ┌─────────────┐                  │
│  │ Segment 1   │   │ Segment 2   │  ...   │ Segment 10  │                  │
│  │ Encoder     │   │ Encoder     │        │ Encoder     │                  │
│  │ (1D CNN)    │   │ (1D CNN)    │        │ (MLP/CNN)   │                  │
│  └──────┬──────┘   └──────┬──────┘        └──────┬──────┘                  │
│         │                 │                      │                          │
│         ▼                 ▼                      ▼                          │
│       [z₁]             [z₂]                   [z₁₀]                         │
│         │                 │                      │                          │
│         └─────────────────┼──────────────────────┘                          │
│                           │                                                 │
│                           ▼                                                 │
│                   ┌───────────────┐                                         │
│                   │ Concatenate   │                                         │
│                   │   z ∈ ℝ⁴⁹     │                                         │
│                   └───────┬───────┘                                         │
│                           │                                                 │
│         ┌─────────────────┼──────────────────────┐                          │
│         │                 │                      │                          │
│         ▼                 ▼                      ▼                          │
│  ┌─────────────┐   ┌─────────────┐        ┌─────────────┐                  │
│  │ Segment 1   │   │ Segment 2   │  ...   │ Segment 10  │                  │
│  │ Decoder     │   │ Decoder     │        │ Decoder     │                  │
│  └─────────────┘   └─────────────┘        └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GMM CLUSTERING                                       │
│  Per-group (AC, ipRGC, DS-RGC, nonDS-RGC)                                  │
│  BIC-based k selection with log Bayes factor threshold                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUTS                                           │
│  • Embeddings (49D per cell)                                               │
│  • Cluster assignments (subtype labels)                                    │
│  • UMAP visualizations                                                      │
│  • Purity metrics                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
Autoencoder_method/
├── config.py              # All configuration parameters
├── run_pipeline.py        # Main entry point
├── data_loader.py         # Data loading and filtering
├── preprocessing.py       # Signal processing
├── grouping.py            # Coarse group assignment
├── train.py               # Training loop
├── embed.py               # Embedding extraction
├── clustering.py          # GMM clustering
├── crossval.py            # Cross-validation
├── evaluation.py          # Metrics and saving
├── stability.py           # Bootstrap stability
├── visualization.py       # Plot generation
├── models/
│   ├── autoencoder.py     # MultiSegmentAutoencoder
│   ├── encoders.py        # SegmentEncoder, SegmentEncoderMLP
│   ├── decoders.py        # SegmentDecoder, SegmentDecoderMLP
│   ├── losses.py          # All loss functions
│   ├── soft_clustering.py # SoftKMeans for purity loss
│   └── gpu_gmm.py         # GPU-accelerated GMM
├── results/               # Output data files
├── plots/                 # Generated visualizations
└── models_saved/          # Trained model checkpoints
```

---

## 3. Data Flow and Pipeline Steps

### Step 1: Data Loading and Filtering

**Module:** `data_loader.py`

```python
df = load_and_filter_data(input_path)
```

**Filters applied:**
- Quality index (QI) ≥ 0.7
- Valid axon type (RGC or AC)
- Baseline firing rate < 200 Hz
- Batch has ≥ 25 good cells
- All required trace columns present

### Step 2: Preprocessing

**Module:** `preprocessing.py`

| Segment | Filter | Sampling Rate | Special Handling |
|---------|--------|---------------|------------------|
| freq_section_0.5-4Hz | 10 Hz LP | 60 → 10 Hz | Standard |
| freq_section_10Hz | None | 60 Hz | Edge slicing (±1s) |
| green_blue_3s_3i_3x | 10 Hz LP | 60 → 10 Hz | Standard |
| bar_concat (8 dirs) | 10 Hz LP | 60 → 10 Hz | Concatenated |
| sta_time_course | None | Original | No processing |
| iprgc_test | 2 Hz LP | 60 → 2 Hz | Slow dynamics |
| step_up_5s_5i_b0_3x | 10 Hz LP | 60 → 10 Hz | Standard |

**Signal processing:**
- Butterworth filter (order 4)
- Zero-phase filtering (`sosfiltfilt`)
- NaN replacement with 0

### Step 3: Coarse Group Assignment

**Module:** `grouping.py`

Priority-based mutually exclusive groups:

```
1. AC        → axon_type == "ac"
2. ipRGC     → iprgc_2hz_QI > 0.8
3. DS-RGC    → ds_p_value < 0.05
4. nonDS-RGC → remaining RGCs
```

### Step 4: Autoencoder Training

**Module:** `train.py`

**Training objective:**
$$L_{\text{total}} = L_{\text{rec}} + \beta \cdot L_{\text{supcon}} + \alpha \cdot L_{\text{purity}}$$

Where:
- $L_{\text{rec}}$: Weighted reconstruction loss (inverse length weighting)
- $L_{\text{supcon}}$: Supervised contrastive loss on coarse groups
- $L_{\text{purity}}$: Cluster purity loss (conditional entropy)

### Step 5: Embedding Extraction

**Module:** `embed.py`

```python
embeddings = extract_embeddings(model, segments)
embeddings_std = standardize_embeddings(embeddings)  # z-score
```

### Step 6: GMM Clustering

**Module:** `clustering.py`

Per-group clustering with BIC-based k selection:

1. Fit GMM for k = 1, 2, ..., k_max
2. Compute BIC for each k
3. Select k using log Bayes factor threshold:
   $$\log \text{BF} = \frac{\text{BIC}_{k-1} - \text{BIC}_k}{2}$$
4. Stop when $\log \text{BF} < 6$ (weak evidence for more clusters)

### Step 7: Evaluation and Visualization

**Modules:** `evaluation.py`, `visualization.py`

- Silhouette score
- Cross-validation purity
- UMAP embeddings
- BIC curves
- Prototype plots

---

## 4. Configuration Parameters

### 4.1 Column Names

```python
# Trace columns
FREQ_SECTION_COLS = ["freq_section_0p5hz", "freq_section_1hz", 
                     "freq_section_2hz", "freq_section_4hz", "freq_section_10hz"]
COLOR_COL = "green_blue_3s_3i_3x"
BAR_COLS = ["corrected_moving_h_bar_s5_d8_3x_000", ..., "_315"]  # 8 directions
RF_COL = "sta_time_course"
IPRGC_COL = "iprgc_test"
STEP_UP_COL = "step_up_5s_5i_b0_3x"

# Metadata columns
DS_PVAL_COL = "ds_p_value"
AXON_COL = "axon_type"
IPRGC_QI_COL = "iprgc_2hz_QI"
QI_COL = "step_up_QI"
```

### 4.2 Segment Latent Dimensions

| Segment | Latent Dim | Rationale |
|---------|------------|-----------|
| freq_section_* (×5) | 4 each | Temporal frequency response |
| green_blue_3s_3i_3x | 6 | Color opponency |
| bar_concat | 12 | Direction tuning (8 dirs) |
| sta_time_course | 3 | RF temporal kernel |
| iprgc_test | 4 | Intrinsic photosensitivity |
| step_up_5s_5i_b0_3x | 4 | ON/OFF/sustained response |
| **Total** | **49** | |

### 4.3 Filter Thresholds

```python
QI_THRESHOLD = 0.7              # Min quality index
DS_P_THRESHOLD = 0.05           # p-value for DS classification
IPRGC_QI_THRESHOLD = 0.8        # QI for ipRGC classification
BASELINE_MAX_THRESHOLD = 200.0  # Max baseline firing rate
MIN_BATCH_GOOD_CELLS = 25       # Min cells per recording batch
MIN_CELLS_PER_GROUP = 50        # Min cells for GMM fitting
```

### 4.4 Signal Processing

```python
SAMPLING_RATE = 60.0            # Original Hz
TARGET_SAMPLING_RATE = 10.0     # After downsampling
LOWPASS_CUTOFF = 10.0           # Hz (default)
FILTER_ORDER = 4                # Butterworth order

# ipRGC-specific
IPRGC_LOWPASS_CUTOFF = 2.0      # Hz
IPRGC_TARGET_RATE = 2.0         # Hz
```

### 4.5 Autoencoder Training

```python
AE_HIDDEN_DIMS = [32, 64, 128]  # Conv layer channels
AE_DROPOUT = 0.1                # Dropout probability
AE_EPOCHS = 150                 # Max training epochs
AE_BATCH_SIZE = 128             # Mini-batch size
AE_LEARNING_RATE = 1e-4         # Adam learning rate
AE_WEIGHT_DECAY = 1e-5          # L2 regularization

# Loss weights
SUPCON_WEIGHT = 1.0             # β (contrastive loss)
SUPCON_TEMPERATURE = 0.05       # τ (softmax sharpness)

# Purity loss
USE_PURITY_LOSS = True
PURITY_LOSS_WEIGHT = 1.0        # α
PURITY_N_CLUSTERS = 100         # Soft clusters
PURITY_TEMPERATURE = 1.0        # Assignment sharpness

# Training control
EARLY_STOPPING_PATIENCE = 15    # Epochs without improvement
DEVICE = "cuda"                 # Training device
```

### 4.6 GMM Clustering

```python
GMM_N_INIT = 20                 # Random restarts
GMM_REG_COVAR = 1e-3           # Covariance regularization
GMM_USE_GPU = True             # Use PyTorch GMM

# Max clusters per group
K_MAX = {
    "AC": 40,
    "ipRGC": 10,
    "DS-RGC": 40,
    "nonDS-RGC": 80,
}

LOG_BF_THRESHOLD = 6.0         # Evidence threshold
MIN_CLUSTER_SIZE = 10          # Min cells per cluster
```

### 4.7 Bootstrap Stability

```python
RUN_BOOTSTRAP = True
BOOTSTRAP_N_ITERATIONS = 20
BOOTSTRAP_SAMPLE_FRACTION = 0.9
STABILITY_THRESHOLD = 0.8
BOOTSTRAP_RANDOM_SEED = 42
```

### 4.8 Cross-Validation

```python
RUN_CV_TURNS = True
CV_TURNS = [
    {"omit": "axon_type", "active": ["ds_cell", "iprgc"]},
    {"omit": "ds_cell", "active": ["axon_type", "iprgc"]},
    {"omit": "iprgc", "active": ["axon_type", "ds_cell"]},
]
CVSCORE_THRESHOLD = 0.7
```

### 4.9 UMAP Visualization

```python
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
```

---

## 5. Model Architecture

### 5.1 MultiSegmentAutoencoder

**File:** `models/autoencoder.py`

```python
class MultiSegmentAutoencoder(nn.Module):
    """
    Forward pass:
        segments: Dict[str, (batch, length)] 
        → {
            'embeddings': Dict[str, (batch, latent_dim)],
            'full_embedding': (batch, 49),
            'reconstructions': Dict[str, (batch, length)],
        }
    """
```

The model maintains separate encoder-decoder pairs for each segment, allowing segment-specific architectures while producing a unified 49D embedding.

### 5.2 SegmentEncoder (1D CNN)

**File:** `models/encoders.py`

For segments with length ≥ 20:

```
Input: (batch, length)
    ↓ unsqueeze to (batch, 1, length)
    ↓ Conv1d(1 → 32, k=7, s=2) + BN + ReLU + Dropout
    ↓ Conv1d(32 → 64, k=5, s=2) + BN + ReLU + Dropout
    ↓ Conv1d(64 → 128, k=3, s=2) + BN + ReLU + Dropout
    ↓ AdaptiveAvgPool1d(4)
    ↓ Flatten
    ↓ Linear(128×4 → latent_dim)
Output: (batch, latent_dim)
```

### 5.3 SegmentEncoderMLP

For short segments (length < 20):

```
Input: (batch, length)
    ↓ Linear(length → 64) + BN + ReLU + Dropout
    ↓ Linear(64 → 32) + BN + ReLU + Dropout
    ↓ Linear(32 → latent_dim)
Output: (batch, latent_dim)
```

### 5.4 Decoders

Mirror architecture of encoders using transposed convolutions (CNN) or MLPs.

---

## 6. Loss Functions

### 6.1 Weighted Reconstruction Loss

**File:** `models/losses.py` → `WeightedReconstructionLoss`

Inverse-length weighting prevents long segments from dominating:

$$L_{\text{rec}} = \sum_{s \in \text{segments}} w_s \cdot \text{MSE}(x_s, \hat{x}_s)$$

Where:
$$w_s = \frac{1/|s|}{\sum_{s'} 1/|s'|}$$

### 6.2 Supervised Contrastive Loss (SupCon)

**File:** `models/losses.py` → `SupervisedContrastiveLoss`

Pulls together embeddings with the same coarse group label:

$$L_{\text{supcon}} = -\frac{1}{|P_i|} \sum_{p \in P_i} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}$$

Where:
- $P_i$ = set of samples with same label as $i$
- $\tau$ = temperature (0.05 default, lower = sharper separation)
- Embeddings are L2-normalized before computing similarity

**Effect:** Cells in the same coarse group cluster together in embedding space.

### 6.3 Cluster Purity Loss

**File:** `models/losses.py` → `ClusterPurityLoss`

Minimizes conditional entropy $H(Y|C)$ to encourage pure clusters:

$$H(Y|C) = -\sum_c p(c) \sum_y p(y|c) \log p(y|c)$$

**Components:**
1. **SoftKMeans:** Learnable cluster centers with soft assignments
   $$q_{ic} = \frac{\exp(-\|z_i - \mu_c\|^2 / \tau)}{\sum_{c'} \exp(-\|z_i - \mu_{c'}\|^2 / \tau)}$$

2. **Per-label entropy:** For each binary label (axon_type, ds_cell, iprgc)
   $$p(y=1|c) = \frac{\sum_i q_{ic} \cdot y_i}{\sum_i q_{ic}}$$

3. **Cluster-weighted entropy:**
   $$L_{\text{purity}} = \frac{1}{|Y|} \sum_{y \in Y} \sum_c p(c) \cdot H(y|c)$$

**Effect:** Each cluster becomes dominated by one label value.

### 6.4 Combined Loss

$$L_{\text{total}} = L_{\text{rec}} + \beta \cdot L_{\text{supcon}} + \alpha \cdot L_{\text{purity}}$$

Default: $\beta = 1.0$, $\alpha = 1.0$

---

## 7. Clustering Algorithm

### 7.1 Per-Group GMM Fitting

**File:** `clustering.py` → `cluster_per_group`

```python
for group in ['AC', 'ipRGC', 'DS-RGC', 'nonDS-RGC']:
    embeddings_group = embeddings[groups == group]
    embeddings_group = StandardScaler().fit_transform(embeddings_group)
    
    # Fit GMMs for k = 1 to k_max
    models, bic_values = fit_gmm_bic(embeddings_group, k_max=K_MAX[group])
    
    # Select k by log Bayes factor
    k_selected = select_k_by_logbf(bic_values)
    
    # Get cluster assignments
    labels = models[k_selected].predict(embeddings_group)
```

### 7.2 BIC-Based Model Selection

**Bayesian Information Criterion:**
$$\text{BIC} = -2 \log L + k \cdot \log n$$

Where $k$ = number of parameters, $n$ = number of samples.

**Log Bayes Factor:**
$$\log \text{BF}(k \text{ vs } k-1) = \frac{\text{BIC}_{k-1} - \text{BIC}_k}{2}$$

**Selection rule:** Choose smallest $k$ where $\log \text{BF} < 6$ (weak evidence for more clusters).

| Log BF | Interpretation |
|--------|----------------|
| < 1 | Not worth mentioning |
| 1-3 | Weak evidence |
| 3-10 | Moderate evidence |
| > 10 | Strong evidence |

### 7.3 GPU-Accelerated GMM

**File:** `models/gpu_gmm.py` → `GaussianMixtureGPU`

PyTorch implementation of diagonal-covariance GMM:
- EM algorithm with configurable iterations
- Multiple random initializations
- BIC computation
- Compatible with scikit-learn API

---

## 8. Cross-Validation

### 8.1 Post-hoc Purity Analysis

**File:** `evaluation.py` → `compute_cv_purity_posthoc`

Measures cluster purity against **original cell labels** (not derived from groups):

| Label | Source | Description |
|-------|--------|-------------|
| axon_type | `df['axon_type']` | 'ac' vs 'rgc' |
| ds_cell | `ds_p_value < 0.05` | Direction-selective |
| iprgc | `iprgc_2hz_QI > 0.8` | Intrinsically photosensitive |

**Purity formula:**
$$\text{Purity} = \frac{\sum_c \max_y |C_c \cap Y_y|}{n}$$

### 8.2 True Cross-Validation (Leave-One-Label-Out)

**File:** `crossval.py` → `run_full_cv`

For each label to omit:
1. Create group labels using only active labels
2. Retrain autoencoder without supervision from omitted label
3. Cluster with GMM
4. Measure purity against omitted label

**CV Score:** Average purity across all omitted labels.

---

## 9. Implementation Details

### 9.1 Training Loop

```python
for epoch in range(epochs):
    for batch_indices in dataloader:
        # Get batch data
        batch_segments = {name: segments[name][batch_indices] for name in segments}
        batch_labels = group_labels[batch_indices]
        batch_purity = purity_labels[batch_indices]
        
        # Forward pass
        output = model(batch_segments)
        
        # Compute losses
        losses = loss_fn(
            originals=batch_segments,
            reconstructions=output['reconstructions'],
            embeddings=output['full_embedding'],
            group_labels=batch_labels,
            purity_labels=batch_purity,
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Learning rate scheduling
    scheduler.step(epoch_loss)
    
    # Early stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break
```

### 9.2 Segment Normalization

During training, each segment is normalized globally:
```python
for name, arr in segments.items():
    mean = np.mean(arr)
    std = np.std(arr) + 1e-8
    arr_normalized = (arr - mean) / std
```

### 9.3 Embedding Standardization

After training, embeddings are z-score standardized:
```python
scaler = StandardScaler()
embeddings_std = scaler.fit_transform(embeddings)
```

### 9.4 UMAP Projection

```python
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)
umap_coords = reducer.fit_transform(embeddings_std)
```

---

## 10. Output Files

### 10.1 Results Directory

| File | Format | Description |
|------|--------|-------------|
| `embeddings.parquet` | Parquet | 49D embeddings per cell |
| `cluster_assignments.parquet` | Parquet | Cluster IDs and subtype labels |
| `k_selection.json` | JSON | BIC curves and k selection details |
| `k_selection.csv` | CSV | Summary of selected k per group |
| `cv_purity.json` | JSON | Cross-validation purity results |
| `stability_metrics.json` | JSON | Bootstrap stability scores |

### 10.2 Plots Directory

| File | Description |
|------|-------------|
| `umap_by_group.png` | UMAP colored by coarse group |
| `umap_by_cluster.png` | UMAP colored by cluster ID |
| `bic_curves_combined.png` | BIC vs k for all groups |
| `cv_purity.png` | Purity bar chart per label |
| `prototypes/*.png` | Mean response per cluster |

### 10.3 Models Directory

| File | Description |
|------|-------------|
| `autoencoder_best.pt` | Best model checkpoint |

### 10.4 Embedding Parquet Schema

```
cell_id: int
coarse_group: str
model_version: str
z_0 ... z_48: float  (49 embedding dimensions)
```

### 10.5 Cluster Assignment Parquet Schema

```
cell_id: int
coarse_group: str
cluster_id: int
subtype_label: str  (e.g., "DS-RGC::cluster_03")
posterior_prob: float
```

---

## Usage

### Command Line

```bash
# Full pipeline
python -m Autoencoder_method.run_pipeline

# With custom paths
python -m Autoencoder_method.run_pipeline --input data.parquet --output results/

# Quick test (subset of cells)
python -m Autoencoder_method.run_pipeline --subset 500

# Skip training (use existing model)
python -m Autoencoder_method.run_pipeline --skip-training --model models/autoencoder_best.pt

# Run full cross-validation
python -m Autoencoder_method.run_pipeline --run-cv
```

### Python API

```python
from Autoencoder_method.run_pipeline import main

results = main(
    input_path="data.parquet",
    output_dir="results/",
    skip_training=False,
    skip_stability=False,
    run_cv=True,
)

print(f"Silhouette score: {results['silhouette_score']:.4f}")
print(f"CV Purity: {results['cv_purity']['cv_score']:.4f}")
```

---

## References

1. **Baden et al. (2016)** - Functional diversity of RGCs in mouse retina
2. **Xie et al. (2016)** - Deep Embedded Clustering (DEC)
3. **Khosla et al. (2020)** - Supervised Contrastive Learning
4. **Schwarz (1978)** - Bayesian Information Criterion

---

*Document generated for the Retinal Cell Type Clustering Pipeline v1.0*
