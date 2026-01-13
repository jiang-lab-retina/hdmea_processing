# Data Model: AE-RGC Clustering

**Feature**: 013-ae-rgc-clustering  
**Date**: 2026-01-11

This document defines the data entities, schemas, and relationships for the autoencoder-based RGC clustering pipeline.

---

## Entity Relationship Diagram

```
┌─────────────────────┐       ┌─────────────────────┐
│       Cell          │       │    TraceSegment     │
├─────────────────────┤       ├─────────────────────┤
│ cell_id (PK)        │──────<│ cell_id (FK)        │
│ filename            │       │ segment_name        │
│ unit_id             │       │ raw_trace           │
│ axon_type           │       │ preprocessed_trace  │
│ ds_p_value          │       │ segment_length      │
│ iprgc_2hz_QI        │       └─────────────────────┘
│ step_up_QI          │
│ coarse_group        │       ┌─────────────────────┐
│ baseline_value      │       │     Embedding       │
└─────────────────────┘       ├─────────────────────┤
         │                    │ cell_id (FK)        │
         │                    │ embedding_vector    │
         │                    │ segment_embeddings  │
         ▼                    │ model_version       │
┌─────────────────────┐       └─────────────────────┘
│   ClusterAssignment │
├─────────────────────┤       ┌─────────────────────┐
│ cell_id (FK)        │       │     CVTurn          │
│ coarse_group        │       ├─────────────────────┤
│ cluster_id          │       │ turn_id             │
│ subtype_label       │       │ omitted_label       │
│ posterior_prob      │       │ active_labels       │
└─────────────────────┘       │ purity_score        │
                              │ model_path          │
                              └─────────────────────┘
```

---

## Entity Schemas

### 1. Cell

The primary unit of analysis - one neural recording unit.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `cell_id` | string | Unique identifier | PK, format: `{filename}_{unit_id}` |
| `filename` | string | Source recording file | Not null |
| `unit_id` | int | Unit number within recording | Not null |
| `axon_type` | enum | Cell type classification | `{"rgc", "ac"}` |
| `ds_p_value` | float | Direction selectivity p-value | [0, 1] |
| `iprgc_2hz_QI` | float | ipRGC quality index | [0, 1] or NaN |
| `step_up_QI` | float | Response quality index | [0, 1] |
| `coarse_group` | enum | Assigned group label | `{"AC", "ipRGC", "DS-RGC", "nonDS-RGC"}` |
| `baseline_value` | float | Baseline firing rate (Hz) | ≥ 0 |
| `batch_id` | string | Recording batch identifier | Derived from filename |

**Validation Rules**:
- `step_up_QI > 0.7` for inclusion
- `baseline_value < 200.0` for inclusion
- `axon_type` in `{"rgc", "ac"}`
- Batch must have ≥ 25 cells after filtering

---

### 2. TraceSegment

Preprocessed stimulus response trace for one segment of one cell.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `cell_id` | string | Parent cell | FK → Cell |
| `segment_name` | enum | Stimulus segment identifier | See segment list |
| `raw_trace` | ndarray | Original trace data | 1D float array |
| `preprocessed_trace` | ndarray | After filter/downsample | 1D float array |
| `segment_length` | int | Length of preprocessed trace | > 0 |
| `sampling_rate` | float | Sampling rate (Hz) | Segment-specific |

**Segment Names**:
```python
SEGMENT_NAMES = [
    "freq_section_0p5hz",
    "freq_section_1hz", 
    "freq_section_2hz",
    "freq_section_4hz",
    "freq_section_10hz",
    "green_blue_3s_3i_3x",
    "bar_concat",  # 8 directions concatenated
    "sta_time_course",
    "iprgc_test",
    "step_up_5s_5i_b0_3x",
]
```

---

### 3. Embedding

49-dimensional latent representation of a cell.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `cell_id` | string | Parent cell | FK → Cell |
| `embedding_vector` | ndarray | Full 49D embedding | Shape (49,) |
| `segment_embeddings` | dict | Per-segment latent vectors | Keys = segment names |
| `model_version` | string | AE model checkpoint ID | Not null |
| `extraction_timestamp` | datetime | When embedding was computed | ISO 8601 |

**Segment Embedding Dimensions**:
```python
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
# Total: 4*5 + 6 + 12 + 3 + 4 + 4 = 49
```

---

### 4. ClusterAssignment

Final cluster assignment for a cell.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `cell_id` | string | Parent cell | FK → Cell |
| `coarse_group` | enum | Cell's group | FK → Cell.coarse_group |
| `cluster_id` | int | Cluster number within group | 0 to k*-1 |
| `subtype_label` | string | Full subtype identifier | `{group}::cluster_{id}` |
| `posterior_prob` | float | GMM posterior probability | [0, 1] |
| `k_star` | int | Optimal k for this group | > 0 |

**Subtype Label Format**:
```
Examples:
- "DS-RGC::cluster_05"
- "nonDS-RGC::cluster_12"
- "AC::cluster_03"
- "ipRGC::cluster_01"
```

---

### 5. CVTurn

Cross-validation turn configuration and results.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `turn_id` | int | Turn number | 1, 2, or 3 |
| `omitted_label` | string | Label excluded from supervision | `{"axon_type", "ds_cell", "iprgc"}` |
| `active_labels` | list[str] | Labels used for supervision | Complement of omitted |
| `purity_score` | float | Purity on omitted label | [0, 1] |
| `model_path` | string | Path to trained AE for this turn | File path |
| `cluster_results_path` | string | Path to clustering results | File path |
| `training_epochs` | int | Epochs trained | > 0 |
| `final_loss` | float | Training loss at end | ≥ 0 |

---

### 6. StabilityResult

Bootstrap stability assessment for a group.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `coarse_group` | enum | Group being assessed | FK |
| `k_star` | int | Number of clusters | > 0 |
| `n_bootstraps` | int | Bootstrap iterations | Default 20 |
| `sample_fraction` | float | Subsampling fraction | Default 0.9 |
| `median_correlations` | list[float] | Per-bootstrap median correlations | Length = n_bootstraps |
| `mean_stability` | float | Mean of median correlations | [0, 1] |
| `std_stability` | float | Std of median correlations | ≥ 0 |
| `is_stable` | bool | Whether group passes threshold | mean ≥ 0.8 |

---

### 7. BICResult

BIC selection results for a group.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `coarse_group` | enum | Group being clustered | Not null |
| `k` | int | Number of clusters | 1 to k_max |
| `bic` | float | Bayesian Information Criterion | Real number |
| `log_likelihood` | float | Log-likelihood | Real number |
| `n_parameters` | int | GMM parameter count | > 0 |
| `n_samples` | int | Cells in group | > 0 |
| `is_selected` | bool | Whether this k was chosen | At most one per group |

---

## File Schemas (Parquet/CSV)

### embeddings.parquet

```python
schema = {
    "cell_id": str,           # Primary key
    "coarse_group": str,      # Group label
    # 49 embedding columns:
    "z_0": float, "z_1": float, ..., "z_48": float,
    # Or as nested array:
    "embedding": list[float],  # Length 49
    "model_version": str,
}
```

### cluster_assignments.parquet

```python
schema = {
    "cell_id": str,           # Primary key
    "coarse_group": str,      # Group label
    "cluster_id": int,        # Cluster within group
    "subtype_label": str,     # Full label
    "posterior_prob": float,  # GMM posterior
}
```

### cv_purity.csv

```csv
turn_id,omitted_label,purity_score,model_path,n_clusters,training_loss
1,axon_type,0.85,models_saved/ae_turn1.pt,45,0.023
2,ds_cell,0.72,models_saved/ae_turn2.pt,52,0.025
3,iprgc,0.91,models_saved/ae_turn3.pt,48,0.022
```

### stability_metrics.json

```json
{
  "timestamp": "2026-01-11T10:30:00Z",
  "groups": {
    "DS-RGC": {
      "k_star": 15,
      "mean_stability": 0.87,
      "std_stability": 0.04,
      "is_stable": true,
      "median_correlations": [0.85, 0.88, ...]
    },
    "nonDS-RGC": {
      "k_star": 28,
      "mean_stability": 0.82,
      "std_stability": 0.06,
      "is_stable": true,
      "median_correlations": [0.80, 0.84, ...]
    }
  }
}
```

---

## State Transitions

### Cell Processing States

```
RAW → FILTERED → PREPROCESSED → EMBEDDED → CLUSTERED
  │       │           │            │           │
  │       │           │            │           └── ClusterAssignment created
  │       │           │            └── Embedding created
  │       │           └── TraceSegments created
  │       └── Passes QI/baseline/batch filters
  └── Loaded from parquet
```

### Pipeline Execution States

```
INIT → DATA_LOADED → PREPROCESSED → AE_TRAINED → EMBEDDED → CLUSTERED → VALIDATED
                                         │                        │
                                         └── (CV turns branch) ───┘
```

---

## Indexing and Relationships

### Primary Keys
- `Cell.cell_id` - Unique cell identifier
- `CVTurn.turn_id` - CV iteration number

### Foreign Keys
- `TraceSegment.cell_id` → `Cell.cell_id`
- `Embedding.cell_id` → `Cell.cell_id`
- `ClusterAssignment.cell_id` → `Cell.cell_id`

### Indexes (for parquet/query efficiency)
- `cell_id` - All tables
- `coarse_group` - For group-wise operations
- `batch_id` - For batch-level filtering

---

## Data Volume Estimates

| Entity | Est. Count | Storage |
|--------|------------|---------|
| Cell | 5,000-10,000 | ~1 MB (metadata) |
| TraceSegment | 50,000-100,000 | ~500 MB (traces) |
| Embedding | 5,000-10,000 | ~2 MB (49D vectors) |
| ClusterAssignment | 5,000-10,000 | ~0.5 MB |
| CVTurn | 3 | ~10 KB |
| StabilityResult | 4 | ~50 KB |
| BICResult | 200-400 | ~100 KB |

**Total estimated storage**: ~600 MB per full run
