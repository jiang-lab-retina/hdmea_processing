# Pipeline API Contracts

**Branch**: `001-baden-rgc-clustering` | **Date**: 2026-01-06

## Main Entry Point

### `run_baden_pipeline()`

```python
def run_baden_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    k_max_ds: int = 30,
    k_max_nds: int = 60,
    qi_threshold: float = 0.5,
    ds_p_threshold: float = 0.05,
    random_seed: int = 42,
) -> dict:
    """
    Run the full Baden-method RGC clustering pipeline.
    
    Args:
        input_path: Path to input parquet file. Defaults to config.INPUT_PATH.
        output_dir: Output directory. Defaults to config.OUTPUT_DIR.
        k_max_ds: Maximum clusters to evaluate for DS population.
        k_max_nds: Maximum clusters to evaluate for non-DS population.
        qi_threshold: Minimum step_up_QI for cell inclusion.
        ds_p_threshold: P-value threshold for DS classification.
        random_seed: Random seed for reproducibility.
    
    Returns:
        dict with keys:
            - 'ds': ClusteringResult for DS population
            - 'nds': ClusteringResult for non-DS population
            - 'n_input_cells': Total cells in input
            - 'n_filtered_cells': Cells after filtering
            - 'output_paths': dict of saved file paths
    
    Raises:
        FileNotFoundError: If input_path doesn't exist.
        ValueError: If no valid cells remain after filtering.
        ValueError: If required columns missing from input.
    """
```

---

## Module APIs

### preprocessing.py

```python
def load_data(path: str | Path) -> pd.DataFrame:
    """Load parquet file, return DataFrame."""
    
def filter_rows(
    df: pd.DataFrame,
    qi_threshold: float = 0.5,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter rows by NaN, QI threshold, and axon_type."""

def split_ds_nds(
    df: pd.DataFrame,
    p_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into DS and non-DS subsets."""

def preprocess_traces(
    df: pd.DataFrame,
    lowpass_cutoff: float = 10.0,
    sampling_rate: float = 60.0,
    baseline_samples: int = 8,
) -> pd.DataFrame:
    """Apply low-pass filter, baseline subtraction, and normalization."""
```

### features.py

```python
def build_feature_matrix(
    df: pd.DataFrame,
    return_models: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str], dict]:
    """
    Extract 40D feature matrix from preprocessed traces.
    
    Returns:
        If return_models=False: (N, 40) feature matrix
        If return_models=True: (features, feature_names, models_dict)
    """

def standardize_features(
    X: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """Z-score standardize features, return transformed and fitted scaler."""
```

### clustering.py

```python
def select_k_by_bic(
    X: np.ndarray,
    k_grid: range | list[int],
    n_init: int = 20,
    reg_covar: float = 1e-5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Fit GMMs for k values, return BIC table and optimal k."""

def fit_final_gmm(
    X: np.ndarray,
    k: int,
    n_init: int = 20,
    reg_covar: float = 1e-5,
    random_state: int = 42,
) -> GaussianMixture:
    """Fit final GMM with selected k, return fitted model."""
```

### evaluation.py

```python
def compute_posterior_curves(
    labels: np.ndarray,
    posteriors: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute rank-ordered posterior curves per cluster."""

def bootstrap_stability(
    X: np.ndarray,
    k: int,
    n_iter: int = 20,
    frac: float = 0.9,
    random_state: int = 42,
) -> tuple[float, list[float]]:
    """
    Compute bootstrap stability.
    
    Returns:
        (median_correlation, list_of_all_correlations)
    """
```

### visualization.py

```python
def plot_bic_curve(
    bic_table: pd.DataFrame,
    population_name: str,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot BIC vs k curve."""

def plot_posterior_curves(
    curves: dict[int, np.ndarray],
    population_name: str,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot posterior separability curves."""

def plot_umap_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    population_name: str,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot UMAP 2D projection colored by cluster."""
```

---

## Data Contracts

### Input Parquet Schema

```json
{
  "type": "object",
  "required": [
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x",
    "corrected_moving_h_bar_s5_d8_3x_000",
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
    "sta_time_course",
    "step_up_QI",
    "ds_p_value",
    "axon_type"
  ],
  "properties": {
    "freq_step_5st_3x": {"type": "array", "items": {"type": "number"}},
    "green_blue_3s_3i_3x": {"type": "array", "items": {"type": "number"}},
    "sta_time_course": {"type": "array", "items": {"type": "number"}},
    "step_up_QI": {"type": "number"},
    "ds_p_value": {"type": "number"},
    "axon_type": {"type": "string", "enum": ["rgc", "ac", "unknown"]}
  }
}
```

### Output Parquet Schema

```json
{
  "type": "object",
  "required": [
    "cell_id",
    "population",
    "cluster_label",
    "global_cluster_id",
    "posterior_probability"
  ],
  "properties": {
    "cell_id": {"type": "string"},
    "population": {"type": "string", "enum": ["DS", "non-DS"]},
    "cluster_label": {"type": "integer", "minimum": 0},
    "global_cluster_id": {"type": "string", "pattern": "^(DS|non-DS)_[0-9]+$"},
    "posterior_probability": {"type": "number", "minimum": 0, "maximum": 1}
  }
}
```

