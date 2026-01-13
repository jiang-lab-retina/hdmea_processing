# Quickstart Guide: AE-RGC Clustering Pipeline

**Feature**: 013-ae-rgc-clustering  
**Date**: 2026-01-11

This guide shows how to use the autoencoder-based RGC clustering pipeline.

---

## Prerequisites

### Python Environment

```bash
# Create and activate environment
conda create -n ae-rgc python=3.11
conda activate ae-rgc

# Install PyTorch (with CUDA if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install scikit-learn scipy pandas numpy pyarrow umap-learn matplotlib seaborn tqdm
```

### Input Data

The pipeline expects a parquet file with the following columns:
- **Trace columns**: `freq_section_*`, `green_blue_3s_3i_3x`, `corrected_moving_h_bar_*`, `sta_time_course`, `step_up_5s_5i_b0_3x`, `iprgc_test`
- **Metadata columns**: `axon_type`, `ds_p_value`, `iprgc_2hz_QI`, `step_up_QI`

---

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```bash
cd dataframe_phase/classification_v2

# Run with default config
python -m Autoencoder_method.run_pipeline

# Or specify input file
python -m Autoencoder_method.run_pipeline --input /path/to/data.parquet

# Skip stability testing and plots for faster runs
python -m Autoencoder_method.run_pipeline --skip-stability --skip-plots
```

### Option 2: Step-by-Step Execution

```python
from Autoencoder_method import config
from Autoencoder_method.data_loader import load_and_filter_data
from Autoencoder_method.preprocessing import preprocess_all_segments
from Autoencoder_method.grouping import assign_coarse_groups, encode_group_labels
from Autoencoder_method.train import train_autoencoder
from Autoencoder_method.embed import extract_embeddings, standardize_embeddings
from Autoencoder_method.clustering import cluster_per_group
from Autoencoder_method.stability import run_bootstrap_stability, summarize_stability
from Autoencoder_method.visualization import generate_all_plots

# Step 1: Load and filter data
df = load_and_filter_data(config.INPUT_PATH)
print(f"Loaded {len(df)} cells after filtering")

# Step 2: Assign coarse groups
df = assign_coarse_groups(df)
print(df['coarse_group'].value_counts())

# Step 3: Preprocess traces
segments = preprocess_all_segments(df)
print(f"Preprocessed {len(segments)} segments")

# Step 4: Train autoencoder
group_labels, group_map = encode_group_labels(df)
model, history = train_autoencoder(
    segments=segments,
    group_labels=group_labels,
)
print(f"Final loss: {history['loss'][-1]:.4f}")

# Step 5: Extract and standardize embeddings
embeddings = extract_embeddings(model, segments)
embeddings_std, _ = standardize_embeddings(embeddings)
print(f"Embeddings shape: {embeddings_std.shape}")  # (n_cells, 49)

# Step 6: Cluster per group
groups = df['coarse_group'].values
cluster_ids, posterior_probs, group_results = cluster_per_group(
    embeddings=embeddings_std,
    groups=groups,
)
print(f"Clusters per group: {[(g, r['k_selected']) for g, r in group_results.items()]}")

# Step 7: Stability testing
stability_summary, _ = run_bootstrap_stability(
    embeddings=embeddings_std,
    groups=groups,
)
overall = summarize_stability(stability_summary)
print(f"Mean stability: {overall['overall_mean_correlation']:.3f}")

# Step 8: Generate visualizations
generate_all_plots(
    embeddings=embeddings_std,
    segments=segments,
    groups=groups,
    cluster_labels=cluster_ids,
    group_results=group_results,
    output_dir=config.PLOTS_DIR,
)
```

---

## Configuration

All parameters are in `config.py`. Key settings:

### Input/Output Paths

```python
INPUT_PATH = "path/to/data.parquet"
OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / "models_saved"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
```

### Preprocessing

```python
# Sampling and filtering
SAMPLING_RATE = 60.0          # Original Hz
TARGET_SAMPLING_RATE = 10.0   # After downsample
LOWPASS_CUTOFF = 10.0         # Hz

# Cell filtering
QI_THRESHOLD = 0.7
DS_P_THRESHOLD = 0.05
BASELINE_MAX_THRESHOLD = 200.0
MIN_BATCH_GOOD_CELLS = 25

# ipRGC-specific
IPRGC_LOWPASS_CUTOFF = 2.0    # Hz
IPRGC_TARGET_RATE = 2.0       # Hz
IPRGC_QI_THRESHOLD = 0.8
```

### Autoencoder

```python
# Latent dimensions (fixed)
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

# Training
AE_EPOCHS = 100
AE_BATCH_SIZE = 128
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_DROPOUT = 0.1

# Weak supervision
SUPCON_WEIGHT = 0.1           # β in loss
SUPCON_TEMPERATURE = 0.1      # τ
```

### Clustering

```python
# GMM settings
GMM_N_INIT = 20
GMM_REG_COVAR = 1e-3
K_MAX_DS = 40
K_MAX_NDS = 80
K_MAX_AC = 20
K_MAX_IPRGC = 10
LOG_BF_THRESHOLD = 6.0
```

### Stability

```python
RUN_BOOTSTRAP = True
BOOTSTRAP_N_ITERATIONS = 20
BOOTSTRAP_SAMPLE_FRACTION = 0.9
STABILITY_THRESHOLD = 0.8
```

---

## Cross-Validation

Run CV turns to validate generalization:

```python
from Autoencoder_method.crossval import run_cv_turns

cv_results = run_cv_turns(
    df=df,
    segments=segments,
    turns=[
        {"omit": "axon_type", "active": ["ds_cell", "iprgc"]},
        {"omit": "ds_cell", "active": ["axon_type", "iprgc"]},
        {"omit": "iprgc", "active": ["axon_type", "ds_cell"]},
    ]
)

print("CV Purity Scores:")
for turn in cv_results:
    print(f"  Omit {turn['omitted_label']}: purity = {turn['purity']:.3f}")

print(f"CVScore: {cv_results['cvscore']:.3f}")
```

---

## Output Files

After running the pipeline:

```
Autoencoder_method/
├── models_saved/
│   ├── autoencoder_best.pt      # Best trained model
│   ├── ae_turn1.pt              # CV turn 1 model
│   ├── ae_turn2.pt              # CV turn 2 model
│   └── ae_turn3.pt              # CV turn 3 model
├── results/
│   ├── embeddings.parquet       # 49D embeddings per cell
│   ├── cluster_assignments.parquet
│   ├── bic_DS-RGC.parquet       # BIC curve data
│   ├── bic_nonDS-RGC.parquet
│   ├── cv_purity.csv            # CV summary
│   └── stability_metrics.json
└── plots/
    ├── umap_by_group.png
    ├── umap_by_cluster.png
    ├── bic_DS-RGC.png
    ├── bic_nonDS-RGC.png
    ├── prototypes_DS-RGC_*.png
    └── prototypes_nonDS-RGC_*.png
```

---

## Common Tasks

### Change Group Precedence

```python
# In config.py or at runtime
GROUP_PRECEDENCE = ["ac", "ds", "iprgc", "nonds"]  # DS before ipRGC
```

### Disable CV (Faster Runs)

```python
# In config.py
RUN_CV_TURNS = False
```

### Use CPU Only

```python
# In config.py
DEVICE = "cpu"  # Instead of "cuda"
```

### Adjust Weak Supervision Strength

```python
# More supervision (tighter group clusters)
SUPCON_WEIGHT = 0.5

# Less supervision (more data-driven)
SUPCON_WEIGHT = 0.01
```

---

## Troubleshooting

### "CUDA out of memory"

Reduce batch size:
```python
AE_BATCH_SIZE = 64  # or 32
```

### "Too few cells in group X"

Lower the minimum threshold or exclude that group:
```python
MIN_CELLS_PER_GROUP = 50  # Default is 100
EXCLUDE_GROUPS = ["ipRGC"]  # If too few ipRGCs
```

### "Clusters are unstable"

Try:
1. Increase training epochs
2. Reduce k_max
3. Increase regularization: `GMM_REG_COVAR = 1e-2`

### "Low CV purity scores"

The representation may not capture the omitted label well. Try:
1. Increase β (more supervision)
2. Check if omitted label has biological signal in the data
3. Review preprocessing for the relevant traces

---

## Next Steps

1. **Analyze clusters**: Examine response prototypes for biological interpretation
2. **Compare with Baden**: Run same cells through Baden_method for comparison
3. **Tune hyperparameters**: Use CVScore to optimize β, architecture
4. **Export for publication**: Use generated plots and tables
