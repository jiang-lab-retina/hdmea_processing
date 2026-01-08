# Baden-Method Pipeline Modifications for Electrophysiology Data

This document explains the modifications made to the Baden-method RGC clustering pipeline, originally designed for calcium imaging data, to work with multi-electrode array (MEA) electrophysiology spike rate data.

**Reference:** Baden et al. (2016). "The functional diversity of retinal ganglion cells in the mouse." *Nature*, 529(7586), 345-350.

---

## 1. Introduction and Overview

### Original Baden Paper Approach

The Baden paper classified mouse retinal ganglion cells (RGCs) using two-photon calcium imaging data with the following characteristics:

- **Sampling rate:** ~15.6 Hz (calcium indicator dynamics)
- **Signal type:** Relative fluorescence changes (ΔF/F)
- **Temporal resolution:** Limited by calcium indicator kinetics
- **Features:** Chirp response, color opponency, direction selectivity, receptive field structure

### Modified Pipeline Purpose

This pipeline adapts the Baden method for **electrophysiology spike rate data** with:

- **Sampling rate:** 60 Hz (spike rate estimation)
- **Signal type:** Instantaneous firing rate (Hz)
- **Temporal resolution:** Higher than calcium imaging
- **Challenge:** Higher frequency content requires filtering to match Baden's temporal scale

### Key Data Characteristics Differences

| Aspect | Calcium Imaging (Baden) | Electrophysiology (This Pipeline) |
|--------|------------------------|-----------------------------------|
| Sampling rate | ~15.6 Hz | 60 Hz |
| Signal dynamics | Slow (Ca²⁺ kinetics) | Fast (spike timing) |
| Noise characteristics | Shot noise, bleaching | Spike count variability |
| Baseline | Fluorescence background | Spontaneous firing rate |

---

## 2. Preprocessing Modifications for Electrophysiology

### Current Configuration

The preprocessing pipeline includes several optional steps controlled by `config.py`:

```python
APPLY_BASELINE_ZEROING = False      # Disabled, but available
APPLY_MAX_ABS_NORMALIZATION = False # Disabled, but available
```

### Signal Processing Chain

| Step | Original Baden | Modified Pipeline | Current Status |
|------|---------------|-------------------|----------------|
| Sampling | ~15.6 Hz calcium | 60 Hz spike rate | **Active** |
| Low-pass filtering | None needed | 10 Hz Butterworth (order 4) | **Active** |
| Downsampling | None | 6× (60 Hz → 10 Hz) | **Active** |
| Baseline subtraction | Per-stimulus | Global per-cell | **Disabled** (configurable) |
| Max-abs normalization | Applied | Available | **Disabled** (configurable) |

### Preprocessing Flowchart

```
Raw Trace (60 Hz)
       │
       ▼
┌──────────────────┐
│  Average Trials  │  ◄── Multiple repetitions averaged
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Low-pass Filter  │  ◄── 10 Hz cutoff, 4th-order Butterworth
│   (zero-phase)   │      Removes high-frequency spike noise
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   Downsample 6×  │  ◄── 60 Hz → 10 Hz (matches Baden's temporal scale)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│    [OPTIONAL]    │  ◄── APPLY_BASELINE_ZEROING = False
│ Baseline Subtract│      Subtracts median of first 5 samples (0.5s)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│    [OPTIONAL]    │  ◄── APPLY_MAX_ABS_NORMALIZATION = False
│ Max-abs Normalize│      Scales trace to [-1, 1] range
└──────────────────┘
       │
       ▼
  Preprocessed Trace (10 Hz)
```

### Why Low-pass Filtering?

Electrophysiology spike rate data at 60 Hz contains high-frequency components that:
1. Are not present in calcium imaging (filtered by indicator kinetics)
2. May introduce noise into feature extraction
3. Would be aliased during downsampling without anti-aliasing filter

The 10 Hz low-pass filter mimics the temporal smoothing inherent in calcium imaging while preserving stimulus-relevant dynamics.

---

## 3. SVD Mathematical Mechanism and Interpretation

### Purpose

Singular Value Decomposition (SVD) extracts the dominant **temporal pattern** from 8-direction moving bar responses, separating temporal dynamics from directional tuning.

### Matrix Construction

For each cell, responses to 8 directions are stacked into a matrix:

$$X \in \mathbb{R}^{T \times 8}$$

where:
- $T$ = number of time points (after preprocessing)
- 8 = number of directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)

### SVD Decomposition

The matrix is decomposed as:

$$X = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{T \times r}$ — left singular vectors (temporal patterns)
- $\Sigma \in \mathbb{R}^{r \times r}$ — diagonal matrix of singular values
- $V^T \in \mathbb{R}^{r \times 8}$ — right singular vectors (direction weights)
- $r = \min(T, 8)$ — rank

### Time Course Extraction

The **first temporal component** captures the dominant response shape:

$$\mathbf{tc} = \mathbf{u}_1 \cdot \sigma_1$$

where:
- $\mathbf{u}_1$ is the first column of $U$ (primary temporal pattern)
- $\sigma_1$ is the largest singular value (amplitude scaling)

### Why Time Course, Not Direction Tuning?

The pipeline uses the **time course** ($U[:, 0] \cdot S[0]$) rather than the **tuning curve** ($V[0, :]$) because:

1. **Temporal dynamics** differentiate RGC types (transient vs. sustained, ON vs. OFF)
2. **Direction selectivity** is handled separately via p-value classification (DS vs. non-DS)
3. Using tuning curve would conflate temporal type with directional preference

### Per-Cell Normalization

Before SVD, each cell's direction matrix is normalized by its maximum absolute value:

$$X_{\text{norm}} = \frac{X}{\max(|X|) + \epsilon}$$

This **preserves relative amplitude differences across directions** (important for DSGCs) while allowing comparison across cells with different overall response magnitudes.

---

## 4. GMM Mathematical Mechanism

### Model Specification

The pipeline uses a **Gaussian Mixture Model (GMM)** with diagonal covariance:

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2))$$

where:
- $K$ = number of clusters
- $\pi_k$ = mixing coefficient (prior probability of cluster $k$)
- $\boldsymbol{\mu}_k$ = mean vector for cluster $k$
- $\boldsymbol{\sigma}_k^2$ = variance vector for cluster $k$ (diagonal covariance)

### Diagonal Covariance Assumption

Using diagonal covariance (vs. full covariance) means:
- Features are assumed **conditionally independent** given cluster assignment
- Reduces parameters from $O(d^2)$ to $O(d)$ per cluster
- More robust with limited samples per cluster
- Matches Baden's original implementation

### Log-Likelihood

For $n$ samples, the log-likelihood is:

$$\log L = \sum_{i=1}^{n} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2)) \right)$$

### Bayesian Information Criterion (BIC)

BIC balances model fit against complexity:

$$\text{BIC} = -2 \log L + p \cdot \log(n)$$

where:
- $\log L$ = maximized log-likelihood
- $p$ = number of free parameters: $K \cdot (1 + 2d) - 1$ for diagonal GMM
- $n$ = number of samples

**Lower BIC indicates better model** (better fit relative to complexity).

### Current GMM Settings

```python
GMM_N_INIT = 20       # Multiple random initializations
GMM_REG_COVAR = 1e-3  # Regularization on diagonal
USE_GPU = True        # GPU acceleration via PyTorch
```

### Multiple Initialization Strategy

GMM fitting is sensitive to initialization. The pipeline:
1. Runs K-means initialization
2. Performs 20 random restarts (`GMM_N_INIT = 20`)
3. Selects the model with highest log-likelihood

---

## 5. GMM_REG_COVAR and LOG_BF_THRESHOLD

### Covariance Regularization (`GMM_REG_COVAR`)

**Current setting:** `GMM_REG_COVAR = 1e-3`

Regularization $\lambda$ is added to the diagonal of each cluster's covariance matrix:

$$\Sigma_k \leftarrow \Sigma_k + \lambda \cdot I$$

**Purpose:**
- Prevents **singular covariance** when cluster has fewer samples than features
- Controls **cluster tightness** — higher $\lambda$ = looser clusters
- Prevents **overfitting** to noise

**Effect on clustering:**
| `GMM_REG_COVAR` | Effect |
|-----------------|--------|
| 1e-5 (Baden) | Tight clusters, may overfit |
| 1e-3 (current) | Moderate regularization, more robust |
| 1e-2 | Loose clusters, fewer total clusters |

### Log Bayes Factor Threshold (`LOG_BF_THRESHOLD`)

**Current setting:** `LOG_BF_THRESHOLD = 6.0`

The Log Bayes Factor compares adjacent models:

$$\log \text{BF}(k+1, k) \approx -\frac{1}{2} \left( \text{BIC}_{k+1} - \text{BIC}_k \right)$$

**Interpretation (Kass & Raftery, 1995):**

| Log BF | Evidence for $k+1$ over $k$ |
|--------|----------------------------|
| < 0 | Negative (favor $k$) |
| 0–2 | Weak |
| 2–6 | Moderate |
| 6–10 | Strong |
| > 10 | Very strong |

**Stopping criterion:**
The pipeline stops adding clusters when $\log \text{BF} < 6.0$, meaning there is no longer "strong evidence" for adding another cluster.

**Why not just use minimum BIC?**
BIC often continues decreasing slowly without a clear minimum. The Log Bayes Factor provides a principled stopping criterion based on statistical evidence.

---

## 6. Frequency Section Data Preprocessing

### Stimulus Structure

The frequency step stimulus (`freq_step_5st_3x`) contains 5 frequency sections presented sequentially:

| Section | Frequency | Frame Range (60 Hz) | Duration |
|---------|-----------|---------------------|----------|
| freq_section_0p5hz | 0.5 Hz | 30–270 | 4.0 s |
| freq_section_1hz | 1 Hz | 330–570 | 4.0 s |
| freq_section_2hz | 2 Hz | 630–870 | 4.0 s |
| freq_section_4hz | 4 Hz | 930–1170 | 4.0 s |
| freq_section_10hz | 10 Hz | 1230–1470 | 4.0 s |

### Per-Section Processing

Each frequency section requires different preprocessing based on its temporal characteristics:

**Current configuration (`FREQ_SECTION_FILTER` in config.py):**

```python
FREQ_SECTION_FILTER = {
    "freq_section_0p5hz": True,   # Low-pass filter + downsample
    "freq_section_1hz": True,     # Low-pass filter + downsample
    "freq_section_2hz": True,     # Low-pass filter + downsample
    "freq_section_4hz": True,     # Low-pass filter + downsample
    "freq_section_10hz": False,   # NO filtering, NO downsampling
}
```

### Why No Filtering for 10 Hz Section?

The 10 Hz modulation frequency would be **attenuated by the 10 Hz low-pass filter**. Therefore:
- `freq_section_10hz` is kept at **60 Hz** (no filtering or downsampling)
- A **slice `[60:-60]`** removes the first and last second to exclude edge artifacts

**Configuration:**
```python
FREQ_10HZ_START_OFFSET = 60   # Skip first 60 frames (1s at 60Hz)
FREQ_10HZ_END_OFFSET = -60    # Skip last 60 frames (1s at 60Hz)
```

### Feature Extraction Summary

| Section | Filtering | Downsampling | Final Rate | Sparse PCA Components |
|---------|-----------|--------------|------------|----------------------|
| 0.5 Hz | 10 Hz LP | 6× | 10 Hz | 4 (4 non-zero bins each) |
| 1 Hz | 10 Hz LP | 6× | 10 Hz | 4 |
| 2 Hz | 10 Hz LP | 6× | 10 Hz | 4 |
| 4 Hz | 10 Hz LP | 6× | 10 Hz | 4 |
| 10 Hz | None | None | 60 Hz | 4 |

**Total frequency section features:** 5 sections × 4 components = **20 features**

---

## 7. Data Cleaning Thresholds

### Current Filter Settings

All thresholds are configured in `config.py`:

| Parameter | Current Value | Purpose |
|-----------|---------------|---------|
| `QI_THRESHOLD` | **0.7** | Minimum step-up quality index |
| `MIN_BATCH_GOOD_CELLS` | **25** | Minimum cells per recording batch |
| `BASELINE_MAX_THRESHOLD` | **200.0** | Maximum baseline firing rate (Hz) |
| `DS_P_THRESHOLD` | **0.05** | Direction selectivity p-value threshold |
| `VALID_AXON_TYPES` | **["rgc", "ac"]** | Included cell types |

### Quality Index (`QI_THRESHOLD = 0.7`)

The step-up quality index measures response reliability:

$$\text{QI} = \frac{\text{Var}(\bar{r})}{\overline{\text{Var}(r_i)}}$$

where:
- $\bar{r}$ = mean response across trials
- $r_i$ = individual trial responses

**Higher QI** indicates consistent, stimulus-locked responses. Threshold of 0.7 is more stringent than typical (0.5).

### Minimum Batch Size (`MIN_BATCH_GOOD_CELLS = 25`)

**Purpose:** Exclude recording sessions with too few quality cells.

**Rationale:**
- Small batches may have systematic differences (recording quality, animal health)
- Ensures each batch contributes meaningfully to clustering
- Prevents bias from outlier recordings

**Implementation:** Cells are grouped by filename prefix (recording session). Batches with fewer than 25 cells (after QI filtering) are excluded entirely.

### Maximum Baseline Firing Rate (`BASELINE_MAX_THRESHOLD = 200.0`)

**Purpose:** Exclude cells with abnormally high spontaneous activity.

**Rationale:**
- Very high baseline firing (>200 Hz) may indicate:
  - Recording artifacts
  - Multi-unit contamination
  - Injured or dying cells
- Normal RGC baseline typically < 50 Hz

**Calculation:** Median of first 5 samples (0.5s) of filtered step-up trace.

### Direction Selectivity Classification (`DS_P_THRESHOLD = 0.05`)

**Purpose:** Split population into DS (direction-selective) and non-DS cells.

**Test:** Rayleigh test or similar circular statistics test on direction tuning.

**Classification:**
- $p < 0.05$ → **DS cell** (direction-selective)
- $p \geq 0.05$ → **non-DS cell**

DS and non-DS populations are clustered **separately** because direction selectivity is a major functional division.

### Valid Axon Types (`VALID_AXON_TYPES = ["rgc", "ac"]`)

**Purpose:** Filter by morphological classification.

**Options:**
- `"rgc"` — Retinal ganglion cells
- `"ac"` — Amacrine cells

Both cell types can be included or either can be analyzed separately by modifying this list.

---

## Summary of Pipeline Modifications

| Aspect | Original Baden | This Implementation |
|--------|---------------|---------------------|
| **Data source** | Calcium imaging | MEA electrophysiology |
| **Sampling** | ~15.6 Hz | 60 Hz → filtered/downsampled to 10 Hz |
| **Chirp stimulus** | Full chirp | Sectioned frequency steps |
| **10 Hz response** | Attenuated by Ca²⁺ | Preserved (no filtering) |
| **Baseline handling** | Per-stimulus | Global per-cell (optional) |
| **Normalization** | Max-abs | Optional (disabled by default) |
| **BIC stopping** | Minimum BIC | Log Bayes Factor threshold |
| **Regularization** | 1e-5 | 1e-3 (more conservative) |
| **GPU acceleration** | Not mentioned | PyTorch CUDA support |

---

## References

1. Baden T, Berens P, Franke K, Román Rosón M, Bethge M, Euler T. (2016). The functional diversity of retinal ganglion cells in the mouse. *Nature*, 529(7586):345-350.

2. Kass RE, Raftery AE. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430):773-795.

3. Schwarz G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2):461-464.
