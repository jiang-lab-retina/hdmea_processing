# Data Model: DEC-Refined RGC Subtype Clustering

## Input Data Schema

### Source: Parquet File

**Default input file**: `dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet`

Same schema as `Autoencoder_method`, with these key columns:

#### Required Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `axon_type` | string | Cell type: "rgc" or "ac" (only "rgc" used) |
| `ds_p_value` | float | Direction selectivity p-value |
| `os_p_value` | float | Orientation selectivity p-value |
| `iprgc_2hz_QI` | float | ipRGC quality index (for validation) |
| `step_up_QI` | float | Step-up quality index (for filtering) |

#### Required Trace Columns (10 Segments)

| Column | Samples | Rate | Description |
|--------|---------|------|-------------|
| `freq_section_0p5hz` | varies | 60 Hz raw | Frequency response 0.5 Hz |
| `freq_section_1hz` | varies | 60 Hz raw | Frequency response 1 Hz |
| `freq_section_2hz` | varies | 60 Hz raw | Frequency response 2 Hz |
| `freq_section_4hz` | varies | 60 Hz raw | Frequency response 4 Hz |
| `freq_section_10hz` | varies | 60 Hz raw | Frequency response 10 Hz |
| `green_blue_3s_3i_3x` | varies | 60 Hz raw | Color stimulus response |
| `corrected_moving_h_bar_s5_d8_3x_XXX` | varies | 60 Hz raw | Bar responses (8 directions) |
| `sta_time_course` | 60 | N/A | RF temporal kernel |
| `iprgc_test` | varies | 60 Hz raw | ipRGC test stimulus |
| `step_up_5s_5i_b0_3x` | varies | 60 Hz raw | Step-up response |

---

## Group Assignment Logic

```
Priority: DS > OS

if ds_p_value < 0.05:
    group = "DSGC"
elif os_p_value < 0.05:
    group = "OSGC"
else:
    group = "Other"
```

---

## Preprocessed Segment Lengths

After segment-specific filtering and resampling:

| Segment | Latent Dim | Processing | Expected Length |
|---------|------------|------------|-----------------|
| `freq_section_0p5hz` | 4 | LP 10Hz, resample 10Hz | TBD |
| `freq_section_1hz` | 4 | LP 10Hz, resample 10Hz | TBD |
| `freq_section_2hz` | 4 | LP 10Hz, resample 10Hz | TBD |
| `freq_section_4hz` | 4 | LP 10Hz, resample 10Hz | TBD |
| `freq_section_10hz` | 4 | Edge crop only, keep 60Hz | TBD |
| `green_blue_3s_3i_3x` | 6 | LP 10Hz, resample 10Hz | TBD |
| `bar_concat` | 12 | LP 10Hz, resample 10Hz, concat 8 dirs | TBD |
| `sta_time_course` | 3 | No processing | 60 |
| `iprgc_test` | 4 | LP 4Hz, resample 2Hz | TBD |
| `step_up_5s_5i_b0_3x` | 4 | LP 10Hz, resample 10Hz | TBD |
| **Total** | **49** | | |

---

## Output Data Structures

### 1. Embeddings (embeddings.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | int | Cell identifier |
| `group` | string | DSGC, OSGC, or Other |
| `z_0` ... `z_48` | float | 49D embedding |
| `embedding_type` | string | "initial" or "dec_refined" |

### 2. Cluster Assignments (cluster_assignments.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | int | Cell identifier |
| `group` | string | DSGC, OSGC, or Other |
| `gmm_cluster` | int | Initial GMM assignment |
| `gmm_posterior` | float | Max posterior from GMM |
| `dec_cluster` | int | Final DEC assignment |
| `dec_soft_max` | float | Max soft assignment from DEC |

### 3. BIC Curve (k_selection.json)

```json
{
  "group": "DSGC",
  "k_range": [1, 2, 3, ..., 40],
  "bic_values": [1234.5, 1200.3, ...],
  "k_selected": 12,
  "selection_method": "min_bic"
}
```

### 4. ipRGC Metrics (iprgc_validation.json)

```json
{
  "group": "DSGC",
  "baseline_prevalence": 0.05,
  "initial_gmm": {
    "purity": 0.87,
    "top_enriched": [
      {"cluster": 3, "fraction": 0.45, "enrichment": 9.0, "size": 20},
      {"cluster": 7, "fraction": 0.30, "enrichment": 6.0, "size": 33}
    ]
  },
  "dec_refined": {
    "purity": 0.91,
    "top_enriched": [
      {"cluster": 3, "fraction": 0.52, "enrichment": 10.4, "size": 19},
      {"cluster": 7, "fraction": 0.35, "enrichment": 7.0, "size": 31}
    ]
  }
}
```

### 5. Cluster Prototypes (prototypes/{group}/cluster_{id}.json)

```json
{
  "cluster_id": 3,
  "group": "DSGC",
  "n_cells": 20,
  "segments": {
    "freq_section_0p5hz": {
      "mean": [...],
      "sem": [...]
    },
    ...
  }
}
```

### 6. UMAP Coordinates (umap_coords.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | int | Cell identifier |
| `group` | string | DSGC, OSGC, or Other |
| `umap_x_initial` | float | UMAP x (initial embedding) |
| `umap_y_initial` | float | UMAP y (initial embedding) |
| `umap_x_dec` | float | UMAP x (DEC embedding) |
| `umap_y_dec` | float | UMAP y (DEC embedding) |

---

## Configuration Structure

### config.py Key Parameters

```python
# Group thresholds
DS_P_THRESHOLD = 0.05
OS_P_THRESHOLD = 0.05
IPRGC_QI_THRESHOLD = 0.8
GROUP_PRIORITY = ["DS", "OS", "OTHER"]

# Preprocessing
SAMPLING_RATE = 60.0
TARGET_RATE_DEFAULT = 10.0
TARGET_RATE_IPRGC = 2.0
LOWPASS_DEFAULT = 10.0
LOWPASS_IPRGC = 4.0

# Segment latent dimensions
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

# GMM/BIC
K_MAX = {
    "DSGC": 40,
    "OSGC": 20,
    "Other": 80,
}
GMM_N_INIT = 20
GMM_REG_COVAR = 1e-3

# DEC
DEC_UPDATE_INTERVAL = 10  # Update target every N iterations
DEC_MAX_ITERATIONS = 200
DEC_CONVERGENCE_THRESHOLD = 0.001
DEC_RECONSTRUCTION_WEIGHT = 0.1  # IDEC-style

# AE Training
AE_EPOCHS = 150
AE_BATCH_SIZE = 128
AE_LEARNING_RATE = 1e-4
```

---

## Directory Structure

```
divide_conquer_method/
├── config.py              # Configuration parameters
├── run_pipeline.py        # Main entry point
├── data_loader.py         # Load and filter data
├── preprocessing.py       # Segment-specific processing
├── grouping.py            # DS/OS group assignment
├── models/
│   ├── autoencoder.py     # CNN autoencoder (reconstruction only)
│   ├── encoders.py        # Segment encoders
│   ├── decoders.py        # Segment decoders
│   └── dec.py             # DEC implementation
├── clustering/
│   ├── gmm_bic.py         # GMM fitting and BIC selection
│   └── dec_refine.py      # DEC refinement loop
├── validation/
│   └── iprgc_metrics.py   # ipRGC purity and enrichment
├── visualization.py       # UMAP, BIC curves, prototypes
├── evaluation.py          # Comparison metrics
├── results/               # Output data files
├── plots/                 # Generated visualizations
└── models_saved/          # Trained model checkpoints
```
