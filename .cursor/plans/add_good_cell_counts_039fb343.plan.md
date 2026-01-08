---
name: Add Good Cell Counts
overview: Add two new columns (`good_count` and `good_rgc_count`) to the DataFrame, computed per recording file based on `step_up_QI > 0.5` and `axon_type == 'rgc'` criteria.
todos:
  - id: add-function
    content: Add add_good_cell_counts() function to extract_feature_step_iprgc.py
    status: completed
  - id: update-main
    content: Update main() to use new input file and call the new function
    status: completed
    dependencies:
      - add-function
---

# Add Good Cell Count Columns

## Overview

Modify [`dataframe_phase/extract_feature/extract_feature_step_iprgc.py`](dataframe_phase/extract_feature/extract_feature_step_iprgc.py) to add two new columns to the DataFrame:

- `good_count`: Total number of cells with `step_up_QI > 0.5` per recording file
- `good_rgc_count`: Total number of RGC cells with `step_up_QI > 0.5` AND `axon_type == 'rgc'` per recording file

## Data Structure

The index format is: `{filename}_unit_{id}` (e.g., `2024.02.26-10.53.19-Rec_unit_001`)

- Extract filename by splitting on `_unit_` and taking the first part
- Group by filename to compute per-recording counts

## Implementation

1. Add a new function `add_good_cell_counts()` that:

- Extracts the filename from each index entry by splitting on `_unit_`
- Groups by filename and counts cells meeting criteria
- Maps counts back to each row

2. Update `main()` to:

- Load from the new input file: `dataframe_phase/load_feature/firing_rate_with_dsgc_features_typed20251230.parquet`
- Call `add_good_cell_counts()` after loading
- Save with the new columns included

## Key Code Changes

```python
def add_good_cell_counts(df: pd.DataFrame) -> pd.DataFrame:
    # Extract filename from index (format: "filename_unit_id")
    df['_filename'] = df.index.str.rsplit('_unit_', n=1).str[0]
    
    # Count good cells per file (step_up_QI > 0.5)
    good_mask = df['step_up_QI'] > 0.5
    good_counts = df[good_mask].groupby('_filename').size()
    
    # Count good RGC cells per file
    good_rgc_mask = good_mask & (df['axon_type'] == 'rgc')
    good_rgc_counts = df[good_rgc_mask].groupby('_filename').size()
    
    # Map counts back to each row
    df['good_count'] = df['_filename'].map(good_counts).fillna(0).astype(int)
    df['good_rgc_count'] = df['_filename'].map(good_rgc_counts).fillna(0).astype(int)
    
    # Remove helper column
    df = df.drop(columns=['_filename'])
    return df





```