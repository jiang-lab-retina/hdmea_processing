# Research: Baden-Method RGC Clustering Pipeline

**Branch**: `001-baden-rgc-clustering` | **Date**: 2026-01-06

## Research Summary

This document captures technical decisions and research findings for implementing the Baden et al. RGC clustering methodology.

---

## 1. Sparse PCA Implementation

### Decision
Use `sklearn.decomposition.SparsePCA` with post-hoc hard sparsity enforcement via top-k thresholding.

### Rationale
- scikit-learn's SparsePCA uses L1 regularization which produces soft sparsity (many small non-zero weights)
- Baden requires exact sparsity: "10 non-zero time bins" per component
- Solution: Fit SparsePCA with moderate alpha, then threshold each component to keep only top-k weights by absolute value

### Implementation Pattern
```python
from sklearn.decomposition import SparsePCA
import numpy as np

def fit_sparse_pca_with_hard_sparsity(X, n_components, top_k, alpha=1.0):
    """Fit sparse PCA and enforce hard top-k sparsity per component."""
    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    spca.fit(X)
    
    # Enforce hard sparsity
    components = spca.components_.copy()
    for i in range(n_components):
        comp = components[i]
        # Keep only top-k by absolute value
        threshold_idx = np.argsort(np.abs(comp))[-top_k:]
        mask = np.zeros_like(comp, dtype=bool)
        mask[threshold_idx] = True
        components[i] = np.where(mask, comp, 0.0)
        # Renormalize
        norm = np.linalg.norm(components[i])
        if norm > 0:
            components[i] /= norm
    
    spca.components_ = components
    return spca
```

### Alternatives Considered
- `MiniBatchSparsePCA`: Faster but less accurate for small datasets
- Custom sPCA implementation: Too complex, unnecessary

---

## 2. Zero-Phase Filtering

### Decision
Use `scipy.signal.sosfiltfilt` with Butterworth filter for 10 Hz low-pass.

### Rationale
- Zero-phase filtering avoids time shifts that could misalign features
- `sosfiltfilt` is numerically stable compared to `filtfilt` with ba coefficients
- Butterworth provides flat passband response

### Implementation Pattern
```python
from scipy.signal import butter, sosfiltfilt

def lowpass_filter(trace, fs=60, cutoff=10, order=4):
    """Apply zero-phase Butterworth low-pass filter."""
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, trace)
```

### Parameters
- Sampling rate: 60 Hz (from existing codebase)
- Cutoff frequency: 10 Hz (user requirement)
- Filter order: 4 (standard choice)

---

## 3. GMM with Diagonal Covariance

### Decision
Use `sklearn.mixture.GaussianMixture` with `covariance_type='diag'` and `reg_covar=1e-5`.

### Rationale
- Baden explicitly uses diagonal covariance GMM
- `reg_covar=1e-5` matches Baden's "add 10^-5 to the diagonal" for numerical stability
- `n_init=20` matches Baden's "restart EM 20 times"

### Implementation Pattern
```python
from sklearn.mixture import GaussianMixture

def fit_gmm(X, k, n_init=20, reg_covar=1e-5, random_state=42):
    """Fit diagonal GMM with multiple restarts."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='diag',
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
    )
    gmm.fit(X)
    return gmm
```

---

## 4. BIC-Based Model Selection

### Decision
Compute BIC for k=1 to k_max, select k with minimum BIC, compute log Bayes factors for confirmation.

### Rationale
- BIC is directly available via `gmm.bic(X)` in scikit-learn
- Lower BIC indicates better model (penalizes complexity)
- Log Bayes factor approximation: $\log \text{BF}_{k+1,k} \approx -\frac{1}{2}(\text{BIC}_{k+1} - \text{BIC}_k)$
- Values > 6 indicate strong evidence for additional cluster

### Implementation Pattern
```python
def select_k_by_bic(X, k_grid, **gmm_kwargs):
    """Fit GMMs for k values, return BIC table and optimal k."""
    results = []
    for k in k_grid:
        gmm = fit_gmm(X, k, **gmm_kwargs)
        results.append({'k': k, 'bic': gmm.bic(X), 'log_likelihood': gmm.score(X) * len(X)})
    
    bic_table = pd.DataFrame(results)
    optimal_k = bic_table.loc[bic_table['bic'].idxmin(), 'k']
    
    # Compute log Bayes factors
    bic_table['log_bf_next'] = -0.5 * bic_table['bic'].diff(-1)
    
    return bic_table, optimal_k
```

---

## 5. Bootstrap Stability Analysis

### Decision
20 iterations at 90% sampling, match clusters by maximum correlation of means.

### Rationale
- Baden reports ~0.96-0.97 median correlation for stable clusters
- 90% sampling provides sufficient variation while retaining most data
- Maximum correlation matching is straightforward and robust

### Implementation Pattern
```python
def bootstrap_stability(X, k, n_iter=20, frac=0.9, random_state=42):
    """Assess cluster stability via bootstrap resampling."""
    rng = np.random.RandomState(random_state)
    original_gmm = fit_gmm(X, k)
    original_means = original_gmm.means_
    
    correlations = []
    for i in range(n_iter):
        # Subsample
        n_samples = int(len(X) * frac)
        idx = rng.choice(len(X), n_samples, replace=False)
        X_boot = X[idx]
        
        # Fit
        boot_gmm = fit_gmm(X_boot, k, random_state=random_state + i)
        boot_means = boot_gmm.means_
        
        # Match by max correlation
        corr_matrix = np.corrcoef(original_means, boot_means)[:k, k:]
        matched_corrs = []
        for j in range(k):
            best_match = np.argmax(corr_matrix[j])
            matched_corrs.append(corr_matrix[j, best_match])
        
        correlations.append(np.median(matched_corrs))
    
    return np.median(correlations), correlations
```

---

## 6. Moving Bar SVD for Time Course Extraction

### Decision
Stack 8-direction traces into T×8 matrix, perform SVD, use first left singular vector as time course.

### Rationale
- Baden: "SVD on a T by D normalised mean response matrix"
- First singular vector captures dominant temporal pattern across all directions
- Provides direction-independent time course for sparse PCA

### Implementation Pattern
```python
def extract_bar_time_course(cell_traces_dict):
    """Extract temporal component from 8-direction moving bar responses."""
    # Stack traces: (T, 8)
    directions = ['000', '045', '090', '135', '180', '225', '270', '315']
    traces = np.column_stack([cell_traces_dict[d] for d in directions])
    
    # Normalize each direction
    traces = traces / (np.abs(traces).max(axis=0, keepdims=True) + 1e-8)
    
    # SVD
    U, S, Vt = np.linalg.svd(traces, full_matrices=False)
    
    # First temporal component (scaled by singular value)
    time_course = U[:, 0] * S[0]
    tuning_curve = Vt[0, :]
    
    return time_course, tuning_curve
```

---

## 7. UMAP for Visualization

### Decision
Use UMAP for 2D cluster projections in visualization.

### Rationale
- UMAP better preserves global structure than t-SNE for cluster visualization
- Faster computation than t-SNE for large datasets
- Well-suited for showing cluster separation in feature space

### Implementation Pattern
```python
import umap

def compute_umap_projection(X, n_neighbors=15, min_dist=0.1, random_state=42):
    """Compute 2D UMAP projection of feature matrix."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)
```

---

## 8. Column Name Mapping

### Decision
Map Baden stimulus types to actual parquet column names.

| Baden Term | Column Name | Notes |
|------------|-------------|-------|
| Chirp | `freq_step_5st_3x` | List column, trial-averaged |
| Color | `green_blue_3s_3i_3x` | List column, trial-averaged |
| Moving bar | `corrected_moving_h_bar_s5_d8_3x_*` | 8 columns (000-315) |
| RF time course | `sta_time_course` | List column |
| Quality index | `step_up_QI` | Scalar, threshold > 0.5 |
| DS p-value | `ds_p_value` | Scalar, threshold < 0.05 |
| Cell type | `axon_type` | String, filter for "rgc" |

---

## Dependencies Summary

```text
# Core
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3

# Visualization
matplotlib>=3.7
umap-learn>=0.5

# Model persistence
joblib>=1.3

# Progress
tqdm>=4.65
```

