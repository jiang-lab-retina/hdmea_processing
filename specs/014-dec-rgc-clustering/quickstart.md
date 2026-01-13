# Quickstart: DEC-Refined RGC Subtype Clustering

## Installation

The pipeline is located in `dataframe_phase/classification_v2/divide_conquer_method/`.

```bash
# Ensure you're in the project root with the hdmea environment
cd M:\Python_Project\Data_Processing_2027
conda activate hdmea

# Dependencies (same as Autoencoder_method)
pip install torch scikit-learn umap-learn tqdm pandas pyarrow
```

---

## Quick Start

### Run Full Pipeline (Single Group)

```bash
cd dataframe_phase/classification_v2
python -m divide_conquer_method.run_pipeline --group DSGC
```

This will:
1. Load and filter RGC cells from the default parquet file
2. Assign cells to DSGC, OSGC, Other groups
3. Train autoencoder on DSGC group (reconstruction-only)
4. Select k* via GMM + BIC
5. Refine clusters with DEC
6. Compute ipRGC enrichment
7. Generate UMAP plots and comparison metrics

### Run All Groups

```bash
python -m divide_conquer_method.run_pipeline --all-groups
```

### Quick Test (Subset)

```bash
python -m divide_conquer_method.run_pipeline --group DSGC --subset 500
```

---

## Python API Usage

```python
from divide_conquer_method.run_pipeline import main

# Run complete pipeline
results = main(group="DSGC")

# View results
print(f"Selected k: {results['per_group']['DSGC']['k_selected']}")
print(f"ipRGC purity (GMM): {results['per_group']['DSGC']['gmm_metrics']['purity']:.3f}")
print(f"ipRGC purity (DEC): {results['per_group']['DSGC']['dec_metrics']['purity']:.3f}")
```

### Step-by-Step API

```python
import numpy as np
from pathlib import Path

from divide_conquer_method.data_loader import load_and_filter_data
from divide_conquer_method.grouping import assign_groups
from divide_conquer_method.preprocessing import preprocess_all_segments
from divide_conquer_method.train import train_autoencoder
from divide_conquer_method.embed import extract_embeddings, standardize_embeddings
from divide_conquer_method.clustering.gmm_bic import fit_gmm_bic, select_k_min_bic
from divide_conquer_method.clustering.dec_refine import refine_with_dec
from divide_conquer_method.validation.iprgc_metrics import compute_iprgc_metrics

# Step 1: Load data
df = load_and_filter_data("path/to/data.parquet")
df = assign_groups(df)

# Step 2: Filter to one group
group_df = df[df['group'] == 'DSGC'].reset_index(drop=True)

# Step 3: Preprocess traces
segments = preprocess_all_segments(group_df)

# Step 4: Train autoencoder
model, history = train_autoencoder(segments)

# Step 5: Extract embeddings
embeddings = extract_embeddings(model, segments)
embeddings_std = standardize_embeddings(embeddings)

# Step 6: GMM clustering + BIC selection
models, bic_values = fit_gmm_bic(embeddings_std, k_range=range(1, 41))
k_selected, best_model = select_k_min_bic(models, bic_values, list(range(1, 41)))
gmm_labels = best_model.predict(embeddings_std)
gmm_centers = best_model.means_

# Step 7: DEC refinement
dec_labels, dec_embeddings, dec_history = refine_with_dec(
    model=model,
    segments=segments,
    initial_centers=gmm_centers,
    k=k_selected,
)

# Step 8: Validate with ipRGC
iprgc_labels = (group_df['iprgc_2hz_QI'] > 0.8).values
gmm_metrics = compute_iprgc_metrics(gmm_labels, iprgc_labels)
dec_metrics = compute_iprgc_metrics(dec_labels, iprgc_labels)

print(f"GMM purity: {gmm_metrics['purity']:.3f}")
print(f"DEC purity: {dec_metrics['purity']:.3f}")
print(f"Top enriched cluster: {dec_metrics['top_enriched'][0]}")
```

---

## Output Structure

After running the pipeline, outputs are saved to:

```
divide_conquer_method/
├── results/
│   ├── DSGC/
│   │   ├── embeddings.parquet
│   │   ├── cluster_assignments.parquet
│   │   ├── k_selection.json
│   │   └── iprgc_validation.json
│   ├── OSGC/
│   │   └── ...
│   └── Other/
│       └── ...
├── plots/
│   ├── DSGC/
│   │   ├── bic_curve.png
│   │   ├── umap_gmm.png
│   │   ├── umap_dec.png
│   │   ├── umap_comparison.png
│   │   ├── iprgc_enrichment.png
│   │   └── prototypes/
│   │       └── cluster_*.png
│   └── ...
└── models_saved/
    ├── DSGC/
    │   ├── autoencoder_best.pt
    │   └── dec_refined.pt
    └── ...
```

---

## Key Configuration Options

Edit `config.py` to customize:

```python
# Group thresholds
DS_P_THRESHOLD = 0.05   # p-value for DS classification
OS_P_THRESHOLD = 0.05   # p-value for OS classification
IPRGC_QI_THRESHOLD = 0.8  # QI for ipRGC validation

# Cluster count limits per group
K_MAX = {
    "DSGC": 40,
    "OSGC": 20,
    "Other": 80,
}

# DEC parameters
DEC_UPDATE_INTERVAL = 10        # Update target every N iterations
DEC_MAX_ITERATIONS = 200        # Stop if not converged
DEC_CONVERGENCE_THRESHOLD = 0.001  # Assignment change threshold
DEC_RECONSTRUCTION_WEIGHT = 0.1    # IDEC reconstruction term
```

---

## Common Tasks

### Resume from Checkpoint

```bash
# Skip training if model exists
python -m divide_conquer_method.run_pipeline --group DSGC --skip-training

# Skip both training and GMM
python -m divide_conquer_method.run_pipeline --group DSGC --skip-training --skip-gmm

# Skip DEC, just compare GMM results
python -m divide_conquer_method.run_pipeline --group DSGC --skip-dec
```

### Regenerate Plots Only

```bash
python -m divide_conquer_method.run_pipeline --group DSGC --visualize-only
```

### Compare GMM vs DEC

After pipeline runs, view comparison:

```python
import json

with open("results/DSGC/iprgc_validation.json") as f:
    metrics = json.load(f)

print(f"GMM purity: {metrics['initial_gmm']['purity']:.3f}")
print(f"DEC purity: {metrics['dec_refined']['purity']:.3f}")
print(f"Improvement: {metrics['dec_refined']['purity'] - metrics['initial_gmm']['purity']:.3f}")
```

---

## Troubleshooting

### GPU Out of Memory

```python
# In config.py, reduce batch size
AE_BATCH_SIZE = 64  # Default 128

# Or use CPU (slower)
DEVICE = "cpu"
```

### DEC Not Converging

```python
# Increase max iterations
DEC_MAX_ITERATIONS = 500

# Lower convergence threshold
DEC_CONVERGENCE_THRESHOLD = 0.01
```

### k* at Grid Boundary

If k* equals K_MAX, the true optimum may be higher:

```python
# Increase K_MAX for the affected group
K_MAX["DSGC"] = 60
```

---

## Next Steps

1. **Analyze prototypes**: Examine cluster mean traces in `plots/{group}/prototypes/`
2. **Compare methods**: Run `Autoencoder_method` on same data for comparison
3. **Refine thresholds**: Adjust DS/OS thresholds based on domain knowledge
4. **Add stability testing**: Implement bootstrap analysis (future work)
