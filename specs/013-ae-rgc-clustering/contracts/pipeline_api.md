# Pipeline API Contracts: AE-RGC Clustering

**Feature**: 013-ae-rgc-clustering  
**Date**: 2026-01-11

This document defines the internal Python API contracts for the autoencoder-based RGC clustering pipeline.

---

## Module Overview

```
Autoencoder_method/
├── config.py          # Configuration constants
├── data_loader.py     # Data loading and filtering
├── preprocessing.py   # Signal conditioning
├── grouping.py        # Coarse group assignment
├── models/            # Neural network components
│   ├── encoders.py
│   ├── decoders.py
│   ├── autoencoder.py
│   └── losses.py
├── train.py           # Training loop
├── embed.py           # Embedding extraction
├── clustering.py      # GMM clustering
├── stability.py       # Bootstrap stability
├── crossval.py        # Cross-validation
├── visualization.py   # Plotting
├── evaluation.py      # Metrics
└── run_pipeline.py    # Entry point
```

---

## 1. data_loader.py

### `load_and_filter_data`

```python
def load_and_filter_data(
    input_path: Path | str,
    qi_threshold: float = 0.7,
    baseline_max: float = 200.0,
    min_batch_cells: int = 25,
    valid_axon_types: list[str] = ["rgc", "ac"],
) -> pd.DataFrame:
    """
    Load parquet file and apply cell filtering.
    
    Args:
        input_path: Path to input parquet file
        qi_threshold: Minimum quality index for inclusion
        baseline_max: Maximum baseline firing rate (Hz)
        min_batch_cells: Minimum cells per batch after filtering
        valid_axon_types: Allowed axon type values
    
    Returns:
        DataFrame with filtered cells, indexed by cell_id
        Columns include all trace columns and metadata
    
    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If required columns are missing
    """
```

### `extract_trace_arrays`

```python
def extract_trace_arrays(
    df: pd.DataFrame,
    trace_columns: list[str],
) -> dict[str, np.ndarray]:
    """
    Extract trace data from DataFrame into numpy arrays.
    
    Args:
        df: DataFrame with trace columns
        trace_columns: List of column names to extract
    
    Returns:
        Dict mapping column name to (n_cells, trace_length) array
        Handles nested arrays (trial data) by computing trial mean
    
    Raises:
        KeyError: If any trace_column not in df
    """
```

---

## 2. preprocessing.py

### `preprocess_segment`

```python
def preprocess_segment(
    traces: np.ndarray,
    segment_name: str,
    sampling_rate: float = 60.0,
    lowpass_cutoff: float | None = 10.0,
    target_rate: float | None = 10.0,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Preprocess a single segment's traces for all cells.
    
    Args:
        traces: (n_cells, trace_length) raw trace data
        segment_name: Segment identifier for segment-specific logic
        sampling_rate: Original sampling rate (Hz)
        lowpass_cutoff: Low-pass filter cutoff (Hz), None to skip
        target_rate: Target sampling rate after downsampling, None to skip
        filter_order: Butterworth filter order
    
    Returns:
        (n_cells, new_length) preprocessed traces
    
    Notes:
        - For freq_section_10hz: no filtering, no downsampling, slice edges
        - For iprgc_test: 2 Hz lowpass, 2 Hz target rate
        - For others: 10 Hz lowpass, 10 Hz target rate
    """
```

### `preprocess_all_segments`

```python
def preprocess_all_segments(
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Preprocess all trace segments for the entire dataset.
    
    Args:
        df: DataFrame with all trace columns
    
    Returns:
        Dict mapping segment_name to (n_cells, segment_length) arrays
        Keys: freq_section_*, green_blue, bar_concat, sta, iprgc, step_up
    
    Notes:
        - bar_concat is concatenation of 8 directions in fixed order
    """
```

### `build_segment_map`

```python
def build_segment_map(
    segments: dict[str, np.ndarray],
) -> dict[str, tuple[int, int]]:
    """
    Build index map for segment positions in concatenated vector.
    
    Args:
        segments: Dict of preprocessed segment arrays
    
    Returns:
        Dict mapping segment_name to (start_idx, end_idx) in concat vector
    """
```

---

## 3. grouping.py

### `assign_coarse_groups`

```python
def assign_coarse_groups(
    df: pd.DataFrame,
    precedence: list[str] = ["ac", "iprgc", "ds", "nonds"],
    iprgc_threshold: float = 0.8,
    ds_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Assign disjoint coarse group labels to cells.
    
    Args:
        df: DataFrame with axon_type, ds_p_value, iprgc_2hz_QI columns
        precedence: Order of group precedence (first match wins)
        iprgc_threshold: iprgc_2hz_QI threshold for ipRGC group
        ds_threshold: ds_p_value threshold for DS group
    
    Returns:
        DataFrame with added 'coarse_group' column
        Values: "AC", "ipRGC", "DS-RGC", "nonDS-RGC"
    
    Notes:
        - NaN in iprgc_2hz_QI treated as "not ipRGC"
        - Each cell assigned to exactly one group
    """
```

### `get_group_mask`

```python
def get_group_mask(
    df: pd.DataFrame,
    group: str,
) -> np.ndarray:
    """
    Get boolean mask for cells in a specific group.
    
    Args:
        df: DataFrame with coarse_group column
        group: Group name ("AC", "ipRGC", "DS-RGC", "nonDS-RGC")
    
    Returns:
        Boolean array of shape (n_cells,)
    """
```

---

## 4. models/autoencoder.py

### `SegmentEncoder`

```python
class SegmentEncoder(nn.Module):
    """
    1D CNN encoder for a single trace segment.
    
    Args:
        input_length: Length of input trace
        latent_dim: Output embedding dimension
        hidden_dims: List of hidden channel dimensions
        dropout: Dropout probability
    
    Forward:
        x: (batch, input_length) → z: (batch, latent_dim)
    """
```

### `SegmentDecoder`

```python
class SegmentDecoder(nn.Module):
    """
    1D CNN decoder for a single trace segment.
    
    Args:
        latent_dim: Input embedding dimension
        output_length: Target output trace length
        hidden_dims: List of hidden channel dimensions
        dropout: Dropout probability
    
    Forward:
        z: (batch, latent_dim) → x_hat: (batch, output_length)
    """
```

### `MultiSegmentAutoencoder`

```python
class MultiSegmentAutoencoder(nn.Module):
    """
    Multi-segment autoencoder with per-segment encoders/decoders.
    
    Args:
        segment_configs: Dict mapping segment_name to {
            'input_length': int,
            'latent_dim': int,
        }
        hidden_dims: List of hidden dimensions for all encoders
        dropout: Dropout probability
    
    Forward:
        segments: Dict[str, Tensor(batch, length)] → {
            'embeddings': Dict[str, Tensor(batch, latent_dim)],
            'full_embedding': Tensor(batch, total_latent_dim),
            'reconstructions': Dict[str, Tensor(batch, length)],
        }
    
    Properties:
        total_latent_dim: Sum of all segment latent dims (49)
    """
```

---

## 5. models/losses.py

### `WeightedReconstructionLoss`

```python
class WeightedReconstructionLoss(nn.Module):
    """
    Weighted MSE loss across segments.
    
    Args:
        segment_lengths: Dict mapping segment_name to input length
        loss_type: "mse" or "huber"
    
    Forward:
        originals: Dict[str, Tensor]
        reconstructions: Dict[str, Tensor]
        → scalar loss (weighted by inverse segment length)
    """
```

### `SupervisedContrastiveLoss`

```python
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss using group labels.
    
    Args:
        temperature: Softmax temperature (default 0.1)
    
    Forward:
        embeddings: Tensor(batch, dim) - L2 normalized
        labels: Tensor(batch,) - integer group labels
        → scalar loss
    
    Notes:
        - Pulls together embeddings with same label
        - Pushes apart embeddings with different labels
    """
```

### `CombinedAELoss`

```python
class CombinedAELoss(nn.Module):
    """
    Combined reconstruction + weak supervision loss.
    
    Args:
        segment_lengths: Dict for reconstruction weighting
        beta: Weight for supervision loss
        temperature: SupCon temperature
    
    Forward:
        originals: Dict[str, Tensor]
        reconstructions: Dict[str, Tensor]
        embeddings: Tensor(batch, dim)
        labels: Tensor(batch,)
        → {
            'total': scalar,
            'reconstruction': scalar,
            'supervision': scalar,
        }
    """
```

---

## 6. train.py

### `train_autoencoder`

```python
def train_autoencoder(
    segments: dict[str, np.ndarray],
    group_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    beta: float = 0.1,
    temperature: float = 0.1,
    device: str = "cuda",
    checkpoint_dir: Path | None = None,
    excluded_labels: list[str] | None = None,
) -> tuple[MultiSegmentAutoencoder, dict]:
    """
    Train the multi-segment autoencoder.
    
    Args:
        segments: Preprocessed segment arrays per cell
        group_labels: Coarse group label per cell
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization
        beta: Supervised contrastive loss weight
        temperature: SupCon temperature
        device: "cuda" or "cpu"
        checkpoint_dir: Directory to save checkpoints
        excluded_labels: Labels to exclude from supervision (for CV)
    
    Returns:
        (trained_model, training_history)
        history contains: loss, rec_loss, sup_loss per epoch
    
    Notes:
        - Saves best model based on total loss
        - Uses early stopping if loss plateaus
    """
```

---

## 7. embed.py

### `extract_embeddings`

```python
def extract_embeddings(
    model: MultiSegmentAutoencoder,
    segments: dict[str, np.ndarray],
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract embeddings for all cells.
    
    Args:
        model: Trained autoencoder
        segments: Preprocessed segment arrays
        batch_size: Inference batch size
        device: "cuda" or "cpu"
    
    Returns:
        (n_cells, 49) embedding array
    
    Notes:
        - Model set to eval mode
        - No gradients computed
    """
```

### `standardize_embeddings`

```python
def standardize_embeddings(
    embeddings: np.ndarray,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Z-score standardize embeddings.
    
    Args:
        embeddings: (n_cells, 49) raw embeddings
        groups: If provided, standardize within each group separately
    
    Returns:
        (standardized_embeddings, scaler_params)
        scaler_params contains mean/std per feature (or per group)
    """
```

---

## 8. clustering.py

### `fit_gmm_bic`

```python
def fit_gmm_bic(
    embeddings: np.ndarray,
    k_range: range,
    n_init: int = 20,
    reg_covar: float = 1e-3,
) -> pd.DataFrame:
    """
    Fit GMMs for range of k and compute BIC.
    
    Args:
        embeddings: (n_cells, dim) standardized embeddings
        k_range: Range of cluster counts to try
        n_init: Number of GMM initialization attempts
        reg_covar: Covariance regularization
    
    Returns:
        DataFrame with columns: k, bic, log_likelihood, n_params
    """
```

### `select_k_by_logbf`

```python
def select_k_by_logbf(
    bic_df: pd.DataFrame,
    threshold: float = 6.0,
) -> int:
    """
    Select optimal k using log Bayes factor threshold.
    
    Args:
        bic_df: DataFrame from fit_gmm_bic
        threshold: Log BF threshold for stopping
    
    Returns:
        Optimal k value
    
    Notes:
        - Starts from k=1, increases until improvement < threshold
    """
```

### `cluster_per_group`

```python
def cluster_per_group(
    embeddings: np.ndarray,
    groups: np.ndarray,
    cell_ids: np.ndarray,
    k_max: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Cluster each group separately using Baden-style GMM.
    
    Args:
        embeddings: (n_cells, 49) standardized embeddings
        groups: Coarse group label per cell
        cell_ids: Cell identifiers
        k_max: Max k per group, defaults from config
    
    Returns:
        DataFrame with columns:
        - cell_id, coarse_group, cluster_id, subtype_label, posterior_prob
    
    Notes:
        - Clusters numbered 0 to k*-1 within each group
        - subtype_label format: "{group}::cluster_{id:02d}"
    """
```

---

## 9. stability.py

### `run_bootstrap_stability`

```python
def run_bootstrap_stability(
    embeddings: np.ndarray,
    groups: np.ndarray,
    cluster_results: pd.DataFrame,
    n_iterations: int = 20,
    sample_fraction: float = 0.9,
    random_seed: int = 42,
) -> dict:
    """
    Run bootstrap stability testing.
    
    Args:
        embeddings: (n_cells, 49) embeddings
        groups: Coarse group per cell
        cluster_results: Results from cluster_per_group
        n_iterations: Number of bootstrap iterations
        sample_fraction: Fraction of cells to sample
        random_seed: For reproducibility
    
    Returns:
        {
            "groups": {
                group_name: {
                    "k_star": int,
                    "median_correlations": list[float],
                    "mean_stability": float,
                    "std_stability": float,
                    "is_stable": bool,
                }
            }
        }
    """
```

---

## 10. crossval.py

### `run_cv_turns`

```python
def run_cv_turns(
    df: pd.DataFrame,
    segments: dict[str, np.ndarray],
    turns: list[dict] | None = None,
    **train_kwargs,
) -> dict:
    """
    Run cross-validation with omitted labels.
    
    Args:
        df: Full DataFrame with metadata
        segments: Preprocessed segment arrays
        turns: List of CV turn configs, each with:
            - "omit": label to omit
            - "active": labels to use for supervision
        train_kwargs: Additional args for train_autoencoder
    
    Returns:
        {
            "turns": [
                {
                    "turn_id": int,
                    "omitted_label": str,
                    "purity_score": float,
                    "model_path": str,
                    ...
                }
            ],
            "cvscore": float,  # Mean purity
        }
    
    Default turns:
        1. Omit axon_type, active [ds_cell, iprgc]
        2. Omit ds_cell, active [axon_type, iprgc]
        3. Omit iprgc, active [axon_type, ds_cell]
    """
```

### `compute_purity`

```python
def compute_purity(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute cluster purity with respect to true labels.
    
    Args:
        cluster_labels: Predicted cluster assignments
        true_labels: Ground truth labels
    
    Returns:
        Purity score in [0, 1]
    
    Formula:
        purity = sum(max count per cluster) / total
    """
```

---

## 11. visualization.py

### `plot_umap_embeddings`

```python
def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    output_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Generate UMAP visualization of embeddings.
    
    Args:
        embeddings: (n_cells, 49) embeddings
        labels: Labels for coloring
        label_name: Name for legend
        output_path: Where to save figure
        n_neighbors, min_dist: UMAP parameters
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
```

### `plot_bic_curve`

```python
def plot_bic_curve(
    bic_df: pd.DataFrame,
    k_selected: int,
    group_name: str,
    output_path: Path,
) -> plt.Figure:
    """
    Plot BIC curve with selected k marked.
    """
```

### `plot_response_prototypes`

```python
def plot_response_prototypes(
    df: pd.DataFrame,
    segments: dict[str, np.ndarray],
    cluster_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate mean ± SEM response prototype plots per cluster.
    
    Creates one multi-panel figure per cluster showing all segments.
    """
```

### `generate_all_plots`

```python
def generate_all_plots(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    segments: dict[str, np.ndarray],
    cluster_results: pd.DataFrame,
    bic_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Generate all visualization outputs.
    
    Creates:
    - UMAP by group
    - UMAP by cluster
    - BIC curves per group
    - Response prototypes per cluster
    """
```

---

## 12. run_pipeline.py

### `main`

```python
def main(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    run_cv: bool = True,
    run_stability: bool = True,
    **config_overrides,
) -> dict:
    """
    Run the complete AE-RGC clustering pipeline.
    
    Args:
        input_path: Override config.INPUT_PATH
        output_dir: Override config.OUTPUT_DIR
        run_cv: Whether to run CV turns
        run_stability: Whether to run bootstrap stability
        config_overrides: Override any config parameter
    
    Returns:
        {
            "n_cells": int,
            "n_clusters": int,
            "groups": {...},
            "cvscore": float | None,
            "stability": {...} | None,
        }
    
    Side effects:
        - Saves model to models_saved/
        - Saves results to results/
        - Saves plots to plots/
    """
```

---

## Error Handling

All public functions should raise:

| Exception | When |
|-----------|------|
| `FileNotFoundError` | Input file doesn't exist |
| `ValueError` | Invalid parameter values |
| `KeyError` | Missing required columns |
| `RuntimeError` | Training/inference failures |

---

## Type Hints

All functions use Python 3.11+ type hints:

```python
from typing import Literal
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
```

---

## Logging

All modules use standard logging:

```python
import logging
logger = logging.getLogger(__name__)

# Levels:
# DEBUG: Detailed trace info
# INFO: Progress updates
# WARNING: Recoverable issues
# ERROR: Failures
```
