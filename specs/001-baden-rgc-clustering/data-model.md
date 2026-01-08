# Data Model: Baden-Method RGC Clustering Pipeline

**Branch**: `001-baden-rgc-clustering` | **Date**: 2026-01-06

## Input Schema

### Source: `firing_rate_with_all_features_loaded_extracted20260104.parquet`

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `freq_step_5st_3x` | list[float] | Chirp response trace (trial-averaged) | Chirp features (20 sPCA) |
| `green_blue_3s_3i_3x` | list[float] | Color response trace (trial-averaged) | Color features (6 sPCA) |
| `corrected_moving_h_bar_s5_d8_3x_000` | list[float] | Moving bar 0° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_045` | list[float] | Moving bar 45° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_090` | list[float] | Moving bar 90° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_135` | list[float] | Moving bar 135° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_180` | list[float] | Moving bar 180° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_225` | list[float] | Moving bar 225° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_270` | list[float] | Moving bar 270° direction | Bar SVD → time course |
| `corrected_moving_h_bar_s5_d8_3x_315` | list[float] | Moving bar 315° direction | Bar SVD → time course |
| `sta_time_course` | list[float] | RF time course from STA | RF features (2 PCA) |
| `step_up_QI` | float | Step response quality index | Filtering (> 0.5) |
| `ds_p_value` | float | Direction selectivity p-value | DS/non-DS split (< 0.05) |
| `axon_type` | str | Cell type classification | Filtering (== "rgc") |

### Cell Identity (Index)
The DataFrame index serves as the unique cell identifier. This is preserved throughout the pipeline.

---

## Intermediate Data Structures

### PreprocessedData
After filtering and signal conditioning.

```python
@dataclass
class PreprocessedData:
    """Container for preprocessed cell data."""
    cell_ids: np.ndarray           # (N,) Cell identifiers
    chirp_traces: np.ndarray       # (N, T_chirp) Preprocessed chirp
    color_traces: np.ndarray       # (N, T_color) Preprocessed color
    bar_traces: np.ndarray         # (N, 8, T_bar) Preprocessed moving bar (8 dirs)
    rf_traces: np.ndarray          # (N, T_rf) RF time courses (not filtered)
    is_ds: np.ndarray              # (N,) Boolean mask for DS cells
    
    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)
    
    @property
    def ds_indices(self) -> np.ndarray:
        return np.where(self.is_ds)[0]
    
    @property
    def nds_indices(self) -> np.ndarray:
        return np.where(~self.is_ds)[0]
```

### FeatureMatrix
40-dimensional feature representation.

```python
@dataclass
class FeatureMatrix:
    """40D feature matrix for a population (DS or non-DS)."""
    cell_ids: np.ndarray           # (N,) Cell identifiers
    features: np.ndarray           # (N, 40) Standardized feature matrix
    feature_names: list[str]       # (40,) Feature column names
    scaler: StandardScaler         # Fitted scaler for inverse transform
    
    # Component breakdown
    chirp_cols: slice = slice(0, 20)      # Features 0-19
    color_cols: slice = slice(20, 26)     # Features 20-25
    bar_tc_cols: slice = slice(26, 34)    # Features 26-33
    bar_deriv_cols: slice = slice(34, 38) # Features 34-37
    rf_cols: slice = slice(38, 40)        # Features 38-39
```

### ClusteringResult
GMM clustering output.

```python
@dataclass
class ClusteringResult:
    """Clustering results for a population."""
    population: str                # "DS" or "non-DS"
    cell_ids: np.ndarray           # (N,) Cell identifiers
    labels: np.ndarray             # (N,) Cluster assignments (0 to k-1)
    posteriors: np.ndarray         # (N, k) Posterior probabilities
    optimal_k: int                 # Selected number of clusters
    bic_table: pd.DataFrame        # BIC values for all k
    gmm: GaussianMixture           # Fitted model
    log_bayes_factors: np.ndarray  # LBF between adjacent k values
```

### EvaluationMetrics
Quality assessment outputs.

```python
@dataclass
class EvaluationMetrics:
    """Cluster quality evaluation metrics."""
    population: str                     # "DS" or "non-DS"
    posterior_curves: dict[int, np.ndarray]  # Cluster ID → rank-ordered posteriors
    bootstrap_median_corr: float        # Median correlation across clusters
    bootstrap_all_corrs: list[float]    # All bootstrap iteration correlations
    is_stable: bool                     # median_corr >= 0.90
```

---

## Output Schema

### Results Parquet: `clustering_results.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | object | Original cell identifier (from DataFrame index) |
| `population` | str | "DS" or "non-DS" |
| `cluster_label` | int | Cluster assignment within population |
| `global_cluster_id` | str | Unique ID: "{population}_{cluster_label}" |
| `posterior_probability` | float | Max posterior probability for assigned cluster |
| `posterior_all` | list[float] | Full posterior vector (all clusters) |
| `feature_0` to `feature_39` | float | Standardized feature values |

### BIC Table: `bic_table.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `population` | str | "DS" or "non-DS" |
| `k` | int | Number of clusters |
| `bic` | float | BIC value |
| `log_likelihood` | float | Log-likelihood |
| `log_bf_next` | float | Log Bayes factor vs k+1 |
| `is_optimal` | bool | Whether this k was selected |

### Stability Metrics: `stability_metrics.json`

```json
{
  "DS": {
    "optimal_k": 12,
    "bootstrap_median_correlation": 0.94,
    "bootstrap_iterations": 20,
    "sample_fraction": 0.9,
    "is_stable": true,
    "all_correlations": [0.92, 0.95, ...]
  },
  "non-DS": {
    "optimal_k": 32,
    "bootstrap_median_correlation": 0.96,
    "bootstrap_iterations": 20,
    "sample_fraction": 0.9,
    "is_stable": true,
    "all_correlations": [0.94, 0.97, ...]
  }
}
```

---

## Model Artifacts

### Saved Models (joblib pickle format)

| File | Contents | Used For |
|------|----------|----------|
| `models/ds_gmm.pkl` | Fitted GaussianMixture for DS | Prediction on new DS cells |
| `models/nds_gmm.pkl` | Fitted GaussianMixture for non-DS | Prediction on new non-DS cells |
| `models/ds_spca_chirp.pkl` | SparsePCA for DS chirp | Feature extraction |
| `models/ds_spca_color.pkl` | SparsePCA for DS color | Feature extraction |
| `models/ds_spca_bar_tc.pkl` | SparsePCA for DS bar time course | Feature extraction |
| `models/ds_spca_bar_deriv.pkl` | SparsePCA for DS bar derivative | Feature extraction |
| `models/ds_pca_rf.pkl` | PCA for DS RF time course | Feature extraction |
| `models/ds_scaler.pkl` | StandardScaler for DS features | Standardization |
| `models/nds_spca_chirp.pkl` | SparsePCA for non-DS chirp | Feature extraction |
| `models/nds_spca_color.pkl` | SparsePCA for non-DS color | Feature extraction |
| `models/nds_spca_bar_tc.pkl` | SparsePCA for non-DS bar time course | Feature extraction |
| `models/nds_spca_bar_deriv.pkl` | SparsePCA for non-DS bar derivative | Feature extraction |
| `models/nds_pca_rf.pkl` | PCA for non-DS RF time course | Feature extraction |
| `models/nds_scaler.pkl` | StandardScaler for non-DS features | Standardization |

---

## Visualization Outputs

| File | Description | Contents |
|------|-------------|----------|
| `plots/bic_curves_ds.png` | BIC vs k for DS | Line plot with optimal k marked |
| `plots/bic_curves_nds.png` | BIC vs k for non-DS | Line plot with optimal k marked |
| `plots/posterior_curves_ds.png` | Posterior separability for DS | Multi-line plot, one per cluster |
| `plots/posterior_curves_nds.png` | Posterior separability for non-DS | Multi-line plot, one per cluster |
| `plots/umap_clusters_ds.png` | UMAP projection for DS | Scatter, colored by cluster |
| `plots/umap_clusters_nds.png` | UMAP projection for non-DS | Scatter, colored by cluster |

---

## Validation Rules

### Input Validation
- All required columns must exist and have expected types
- List columns must have consistent lengths within each stimulus type
- `step_up_QI` must be numeric, no NaN in required rows
- `ds_p_value` must be numeric, no NaN in required rows
- `axon_type` must be string

### Feature Validation
- Feature matrix must have exactly 40 columns
- No NaN or Inf values in feature matrix after extraction
- Sparse PCA components must have exactly specified non-zero weights

### Clustering Validation
- Optimal k must be in valid range (1 to k_max)
- All cells must have cluster assignment
- Posterior probabilities must sum to 1.0 per cell

### Output Validation
- Results parquet must have one row per valid input cell
- All cluster labels must be valid (0 to k-1)
- Model files must be loadable with joblib

