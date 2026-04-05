"""Check alignment H5 files for matched unit mapping."""
import h5py
import os

align_dir = "Projects/unified_special_pipeline/blocker_alignment_analysis/output/aligned"
if os.path.exists(align_dir):
    files = os.listdir(align_dir)
    print("aligned/ contents:", files[:10])
else:
    print("No aligned/ subfolder")

# Check pipeline_compare.py for mapping logic
# But first let's check a sample alignment result H5 
# The alignment.py creates matched unit pairs - let's see how
# Let's look at the compared dataframe for clues about unit mapping
import pandas as pd
df = pd.read_parquet("dataframe_compare/output/compared_dataframe_v2_labeled.parquet")

# Check if there are columns that store the unit IDs from before/after recordings
before_unit_cols = [c for c in df.columns if "unit" in c.lower()]
print("Unit columns:", before_unit_cols)

# Check trace columns to understand unit mapping
trace_cols = [c for c in df.columns if "trace" in c.lower() or "firing" in c.lower()]
print("Trace columns sample:", trace_cols[:5])

# The original pipeline_compare.py should show us how data is indexed
# Let's check column types of non-numeric columns
for c in df.columns:
    if df[c].dtype == "object":
        print(f"  {c}: dtype=object, sample={df[c].iloc[0]}")
