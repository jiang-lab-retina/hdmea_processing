# Pipeline API Contract: DEC-Refined RGC Subtype Clustering

## Command Line Interface

### Main Pipeline

```bash
# Run pipeline for a single group
python -m divide_conquer_method.run_pipeline --group DSGC

# Run pipeline for all groups
python -m divide_conquer_method.run_pipeline --all-groups

# With custom input/output
python -m divide_conquer_method.run_pipeline \
    --input data.parquet \
    --output results/ \
    --group DSGC

# Resume from checkpoint
python -m divide_conquer_method.run_pipeline \
    --skip-training \
    --skip-gmm \
    --group DSGC

# Visualization only
python -m divide_conquer_method.run_pipeline \
    --visualize-only \
    --group DSGC
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input`, `-i` | Path | config.INPUT_PATH | Input parquet file |
| `--output`, `-o` | Path | config.OUTPUT_DIR | Output directory |
| `--group` | str | None | Group to process: DSGC, OSGC, Other |
| `--all-groups` | flag | False | Process all three groups |
| `--skip-training` | flag | False | Skip AE training, load checkpoint |
| `--skip-gmm` | flag | False | Skip GMM, load cached k* |
| `--skip-dec` | flag | False | Skip DEC refinement |
| `--visualize-only` | flag | False | Only regenerate plots |
| `--subset` | int | None | Use subset of N cells for testing |
| `--verbose`, `-v` | flag | False | Enable debug logging |

---

## Python API

### Main Entry Point

```python
from divide_conquer_method.run_pipeline import main

results = main(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    group: str | None = None,          # "DSGC", "OSGC", "Other"
    all_groups: bool = False,
    skip_training: bool = False,
    skip_gmm: bool = False,
    skip_dec: bool = False,
    skip_plots: bool = False,
    subset_n: int | None = None,
) -> dict
```

**Returns**: Dictionary with:
```python
{
    'input_path': str,
    'output_dir': str,
    'groups_processed': list[str],
    'per_group': {
        'DSGC': {
            'n_cells': int,
            'k_selected': int,
            'bic_min': float,
            'gmm_metrics': {...},
            'dec_metrics': {...},
            'iprgc_enrichment': {...},
        },
        ...
    },
    'duration_seconds': float,
}
```

### Data Loading

```python
from divide_conquer_method.data_loader import load_and_filter_data

df = load_and_filter_data(
    input_path: Path | str,
    require_complete_traces: bool = True,
) -> pd.DataFrame
```

**Returns**: Filtered DataFrame with RGC cells only.

**Reject reasons logged**:
- `not_rgc`: axon_type != "rgc"
- `nan_metadata`: NaN in required columns
- `nan_trace`: NaN in trace arrays
- `length_mismatch`: Trace length doesn't match expected

### Group Assignment

```python
from divide_conquer_method.grouping import assign_groups

df = assign_groups(
    df: pd.DataFrame,
    ds_threshold: float = 0.05,
    os_threshold: float = 0.05,
) -> pd.DataFrame
```

**Adds columns**:
- `group`: "DSGC", "OSGC", or "Other"
- `is_ds`: bool
- `is_os`: bool

**Logs**: Overlap count (cells meeting both DS and OS thresholds).

### Autoencoder Training

```python
from divide_conquer_method.train import train_autoencoder

model, history = train_autoencoder(
    segments: dict[str, np.ndarray],
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: Path | None = None,
) -> tuple[MultiSegmentAutoencoder, dict]
```

**Returns**:
- `model`: Trained autoencoder
- `history`: Training loss per epoch

### GMM/BIC Clustering

```python
from divide_conquer_method.clustering.gmm_bic import fit_gmm_bic, select_k_min_bic

models, bic_values = fit_gmm_bic(
    embeddings: np.ndarray,
    k_range: range | list[int],
    n_init: int = 20,
    reg_covar: float = 1e-3,
) -> tuple[list[GaussianMixture], np.ndarray]

k_selected, best_model = select_k_min_bic(
    models: list[GaussianMixture],
    bic_values: np.ndarray,
    k_range: list[int],
) -> tuple[int, GaussianMixture]
```

### DEC Refinement

```python
from divide_conquer_method.clustering.dec_refine import refine_with_dec

refined_labels, refined_embeddings, dec_history = refine_with_dec(
    model: MultiSegmentAutoencoder,
    segments: dict[str, np.ndarray],
    initial_centers: np.ndarray,
    k: int,
    max_iterations: int = 200,
    update_interval: int = 10,
    convergence_threshold: float = 0.001,
    reconstruction_weight: float = 0.1,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict]
```

**Returns**:
- `refined_labels`: (n_cells,) cluster assignments
- `refined_embeddings`: (n_cells, 49) updated embeddings
- `dec_history`: Training log with convergence metrics

### ipRGC Validation

```python
from divide_conquer_method.validation.iprgc_metrics import (
    compute_iprgc_metrics,
    compute_enrichment,
    compute_purity,
)

metrics = compute_iprgc_metrics(
    cluster_labels: np.ndarray,
    iprgc_labels: np.ndarray,
) -> dict

# Returns:
{
    'purity': float,
    'baseline_prevalence': float,
    'per_cluster': {
        cluster_id: {
            'n_cells': int,
            'n_iprgc': int,
            'fraction': float,
            'enrichment': float,
        },
        ...
    },
    'top_enriched': list[dict],  # Top 3 by enrichment
}
```

---

## Configuration Schema

```python
# config.py

# Paths
INPUT_PATH = Path("dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet")
OUTPUT_DIR = Path("dataframe_phase/classification_v2/divide_conquer_method")

# Group thresholds
DS_P_THRESHOLD = 0.05
OS_P_THRESHOLD = 0.05
IPRGC_QI_THRESHOLD = 0.8
GROUP_PRIORITY = ["DS", "OS", "OTHER"]

# Preprocessing (same as Autoencoder_method)
SAMPLING_RATE = 60.0
TARGET_RATE_DEFAULT = 10.0
TARGET_RATE_IPRGC = 2.0
LOWPASS_DEFAULT = 10.0
LOWPASS_IPRGC = 4.0

# Segment latent dimensions (total = 49)
SEGMENT_LATENT_DIMS = {
    "freq_section_0p5hz": 4,
    "freq_section_1hz": 4,
    "freq_section_2hz": 4,
    "freq_section_4hz": 4,
    "freq_section_10hz": 4,
    "green_blue_3s_3i_3x": 6,
    "bar_concat": 12,
    "sta_time_course": 3,
    "iprgc_test": 4,
    "step_up_5s_5i_b0_3x": 4,
}

# Autoencoder
AE_HIDDEN_DIMS = [32, 64, 128]
AE_DROPOUT = 0.1
AE_EPOCHS = 150
AE_BATCH_SIZE = 128
AE_LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15

# GMM/BIC
K_MAX = {
    "DSGC": 40,
    "OSGC": 20,
    "Other": 80,
}
GMM_N_INIT = 20
GMM_REG_COVAR = 1e-3

# DEC
DEC_UPDATE_INTERVAL = 10
DEC_MAX_ITERATIONS = 200
DEC_CONVERGENCE_THRESHOLD = 0.001
DEC_RECONSTRUCTION_WEIGHT = 0.1
DEC_ALPHA = 1.0  # Student-t degrees of freedom

# Device
DEVICE = "cuda"
```

---

## Output File Schema

### embeddings.parquet

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | int | Cell identifier |
| `group` | str | DSGC, OSGC, or Other |
| `z_0` ... `z_48` | float | 49D embedding |
| `embedding_type` | str | "initial" or "dec_refined" |

### cluster_assignments.parquet

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | int | Cell identifier |
| `group` | str | DSGC, OSGC, or Other |
| `gmm_cluster` | int | Initial GMM assignment |
| `gmm_posterior` | float | Max posterior from GMM |
| `dec_cluster` | int | Final DEC assignment |
| `dec_soft_max` | float | Max soft assignment from DEC |

### k_selection.json

```json
{
  "group": "DSGC",
  "k_range": [1, 2, 3, ...],
  "bic_values": [1234.5, 1200.3, ...],
  "k_selected": 12,
  "selection_method": "min_bic"
}
```

### iprgc_validation.json

```json
{
  "group": "DSGC",
  "baseline_prevalence": 0.05,
  "initial_gmm": {
    "purity": 0.87,
    "top_enriched": [...]
  },
  "dec_refined": {
    "purity": 0.91,
    "top_enriched": [...]
  }
}
```

---

## Error Handling

| Error | Handling |
|-------|----------|
| Group has < 50 cells | Assign all to single cluster, log warning |
| NaN in required columns | Drop row, increment reject counter |
| DEC doesn't converge | Stop at max iterations, save with warning flag |
| GPU out of memory | Fall back to CPU, log warning |
| Missing checkpoint file | Raise FileNotFoundError with helpful message |
| k* at grid boundary | Log warning, suggest increasing K_MAX |
