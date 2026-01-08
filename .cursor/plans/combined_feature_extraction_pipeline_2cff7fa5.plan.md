---
name: Combined Feature Extraction Pipeline
overview: Create a unified pipeline script that combines all 5 feature extraction modules using high-level functions. Modify extract_step_features and extract_gb_features to support optional QC filtering.
todos:
  - id: modify-step
    content: Add skip_filtering parameter to extract_step_features() in extract_feature_step.py
    status: completed
  - id: modify-gb
    content: Add skip_filtering parameter to extract_gb_features() in extract_feature_gb.py
    status: completed
    dependencies:
      - modify-step
  - id: create-pipeline
    content: Create extract_all_features.py with imports and unified pipeline
    status: completed
    dependencies:
      - modify-gb
---

# Combined Feature Extraction Pipeline

## Overview

1. Modify two existing files to add `skip_filtering` parameter
2. Create a new unified pipeline file that imports and calls all high-level functions

## Pipeline Flow

```mermaid
flowchart TD
    Input["Load parquet"] --> StepIPRGC["1. Step/ipRGC QI"]
    StepIPRGC --> DSGC["2. DSGC Features"]
    DSGC --> Step["3. Step Response Features"]
    Step --> GB["4. Green-Blue Features"]
    GB --> Freq["5. Frequency Features"]
    Freq --> Output["Save parquet"]
```



## Step 1: Modify Existing Files

### [`extract_feature_step.py`](dataframe_phase/extract_feature/extract_feature_step.py)

Add `skip_filtering: bool = False` parameter to `extract_step_features()`. When `True`, skip the QC filtering block (lines 403-421).

### [`extract_feature_gb.py`](dataframe_phase/extract_feature/extract_feature_gb.py)

Add `skip_filtering: bool = False` parameter to `extract_gb_features()`. When `True`, skip the QC filtering block (lines 378-401).

## Step 2: Create Combined Pipeline

Create [`dataframe_phase/extract_feature/extract_all_features.py`](dataframe_phase/extract_feature/extract_all_features.py)

### Imports

```python
from extract_feature_step_iprgc import compute_step_up_qi, compute_iprgc_qi, add_good_cell_counts
from extract_feature_dsgc import remap_direction_columns, process_unit, DIRECTION_ANGLES, CORRECTED_DIRECTION_COLUMNS, N_PERMUTATIONS, N_TRIALS
from extract_feature_step import extract_step_features
from extract_feature_gb import extract_gb_features
from extract_feature_freq import extract_freq_step_features
```



### Processing Steps

1. **Step/ipRGC QI**: Call `compute_step_up_qi()`, `compute_iprgc_qi()`, `add_good_cell_counts()`
2. **DSGC**: Loop with `remap_direction_columns()` and `process_unit()` (same as original main)
3. **Step Response**: Call `extract_step_features(df, skip_filtering=True)`