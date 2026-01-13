# Autoencoder-based RGC Subtype Clustering Pipeline

A weakly supervised clustering pipeline that uses autoencoder-derived embeddings with Baden-style GMM clustering for retinal ganglion cell (RGC) subtype classification.

## Features

- **Segment-wise Autoencoders**: 10 separate encoder/decoder pairs for different stimulus response traces
- **Fixed Latent Budget**: 49D total embedding (4×5 + 6 + 12 + 3 + 4 + 4)
- **Supervised Contrastive Loss**: Weak supervision using coarse group labels
- **Per-Group Clustering**: Diagonal GMM with BIC selection, clusters cannot cross group boundaries
- **Cross-Validation**: Omitted-label purity validation
- **Bootstrap Stability**: 90% subsampling to assess cluster robustness
- **Publication-Ready Outputs**: UMAP plots, BIC curves, response prototypes

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib seaborn umap-learn tqdm
```

## Quick Start

### Basic Usage

```python
from Autoencoder_method import run_pipeline

# Run with default settings
results = run_pipeline.main()

# Outputs are saved to:
# - models_saved/autoencoder_best.pt
# - results/embeddings.parquet
# - results/cluster_assignments.parquet
# - plots/*.png
```

### Command Line

```bash
# From the classification_v2 directory:
python -m Autoencoder_method.run_pipeline

# With custom paths:
python -m Autoencoder_method.run_pipeline --input data.parquet --output results/

# Skip training (use existing model):
python -m Autoencoder_method.run_pipeline --skip-training --model models/autoencoder_best.pt

# Quick run without stability testing or plots:
python -m Autoencoder_method.run_pipeline --skip-stability --skip-plots
```

## Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ Load Data   │ ──> │ Preprocess   │ ──> │ Assign Groups │
│ & Filter    │     │ Segments     │     │ (AC/ipRGC/DS) │
└─────────────┘     └──────────────┘     └───────────────┘
                                                  │
                                                  v
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ Generate    │ <── │ Cluster      │ <── │ Train AE &    │
│ Plots       │     │ per Group    │     │ Extract 49D   │
└─────────────┘     └──────────────┘     └───────────────┘
```

## Configuration

All parameters are in `config.py`. Key settings:

```python
# Segment latent dimensions (must sum to 49)
SEGMENT_LATENT_DIMS = {
    "freq_section_0p5hz": 4,
    "freq_section_1hz": 4,
    # ... etc
}

# Training
AE_EPOCHS = 100
AE_BATCH_SIZE = 128
SUPCON_WEIGHT = 0.1  # β weight for supervised contrastive loss

# Clustering
K_MAX = {"AC": 20, "ipRGC": 10, "DS-RGC": 40, "nonDS-RGC": 80}
LOG_BF_THRESHOLD = 6.0

# Stability
BOOTSTRAP_N_ITERATIONS = 20
BOOTSTRAP_SAMPLE_FRACTION = 0.9
```

## Output Files

| File | Description |
|------|-------------|
| `models_saved/autoencoder_best.pt` | Trained model checkpoint |
| `results/embeddings.parquet` | 49D embeddings per cell |
| `results/cluster_assignments.parquet` | Cluster IDs and subtype labels |
| `results/cv_purity.csv` | Cross-validation purity results |
| `results/stability_metrics.json` | Bootstrap stability metrics |
| `plots/umap_by_group.png` | UMAP colored by coarse group |
| `plots/umap_by_cluster.png` | UMAP colored by subtype cluster |
| `plots/bic_*.png` | BIC curves per group |
| `plots/cluster_sizes.png` | Cluster size distributions |
| `plots/prototypes_*.png` | Response prototypes per group |

## Module Structure

```
Autoencoder_method/
├── __init__.py          # Package initialization
├── config.py            # All configuration parameters
├── data_loader.py       # Data loading and filtering
├── preprocessing.py     # Signal conditioning per segment
├── grouping.py          # Coarse group assignment
├── models/
│   ├── encoders.py      # 1D CNN segment encoders
│   ├── decoders.py      # 1D CNN segment decoders
│   ├── autoencoder.py   # Multi-segment autoencoder
│   └── losses.py        # Reconstruction + SupCon losses
├── train.py             # Training loop
├── embed.py             # Embedding extraction
├── clustering.py        # GMM clustering with BIC
├── crossval.py          # Cross-validation
├── stability.py         # Bootstrap stability
├── evaluation.py        # Metrics and saving
├── visualization.py     # Plot generation
└── run_pipeline.py      # Main entry point
```

## Coarse Groups

Cells are assigned to disjoint groups using precedence:

1. **AC**: `axon_type == "ac"`
2. **ipRGC**: `iprgc_2hz_QI > 0.8`
3. **DS-RGC**: `ds_p_value < 0.05`
4. **nonDS-RGC**: All remaining RGCs

## Cross-Validation

Three CV turns are performed:

| Turn | Omitted | Active Labels | What it Tests |
|------|---------|---------------|---------------|
| 1 | axon_type | ds_cell, iprgc | Can clusters distinguish AC vs RGC? |
| 2 | ds_cell | axon_type, iprgc | Can clusters distinguish DS vs non-DS? |
| 3 | iprgc | axon_type, ds_cell | Can clusters distinguish ipRGC vs others? |

The **CVScore** is the mean purity across all turns.

## References

- Baden et al. (2016) - Original RGC classification methodology
- Khosla et al. (2020) - Supervised Contrastive Learning
