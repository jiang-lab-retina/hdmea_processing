"""Debug script to investigate trace column contents."""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
project_root = Path(__file__).resolve().parents[3]
input_path = project_root / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet"

print(f"Loading: {input_path}")
df = pd.read_parquet(input_path)

# Filter to RGCs first
df = df.dropna(subset=["step_up_QI", "ds_p_value", "axon_type"])
df = df[df["step_up_QI"] > 0.5]
df = df[df["axon_type"] == "rgc"]
print(f"RGC cells after basic filters: {len(df)}")

# Investigate the chirp column
col = "freq_step_5st_3x"
print(f"\n=== Investigating '{col}' ===")

# Check column dtype
print(f"Column dtype: {df[col].dtype}")

# Check a few samples
print(f"\nFirst 5 samples:")
for i, (idx, val) in enumerate(df[col].head().items()):
    print(f"  [{i}] Index {idx}:")
    print(f"      Type: {type(val)}")
    print(f"      Value repr: {repr(val)[:100]}...")
    if val is not None:
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val)
            print(f"      Array dtype: {arr.dtype}")
            print(f"      Array shape: {arr.shape}")
            print(f"      First 5 values: {arr[:5] if len(arr) >= 5 else arr}")
            print(f"      Has NaN: {np.any(pd.isna(arr))}")
            # Try float64 conversion
            try:
                arr_f = np.asarray(val, dtype=np.float64)
                print(f"      Float64 conversion: SUCCESS")
                print(f"      Float64 NaN check: {np.any(np.isnan(arr_f))}")
            except Exception as e:
                print(f"      Float64 conversion: FAILED - {e}")
        else:
            print(f"      Not array-like")

# Check how many are None/NaN at the cell level
null_count = df[col].isna().sum()
print(f"\nNull/NaN at cell level: {null_count} / {len(df)}")

# Check how many have internal NaN
def check_internal_nan(x):
    """Check various conditions that might indicate invalid data."""
    if x is None:
        return "None"
    if isinstance(x, float) and np.isnan(x):
        return "float_nan"
    if not isinstance(x, (list, np.ndarray)):
        return f"unexpected_type:{type(x).__name__}"
    
    arr = np.asarray(x)
    if arr.dtype == object:
        return f"object_dtype"
    
    try:
        arr_f = np.asarray(x, dtype=np.float64)
        if np.any(np.isnan(arr_f)):
            return "has_nan"
        return "valid"
    except Exception as e:
        return f"conversion_error:{str(e)[:30]}"

results = df[col].apply(check_internal_nan)
print(f"\nTrace status distribution:")
for status, count in results.value_counts().items():
    print(f"  {status}: {count} ({100*count/len(df):.1f}%)")

# If all have internal NaN, check WHY
if (results == "has_nan").sum() == len(df):
    print("\n=== All traces have NaN - investigating further ===")
    sample = df[col].iloc[0]
    arr = np.asarray(sample, dtype=np.float64)
    nan_positions = np.where(np.isnan(arr))[0]
    print(f"NaN positions in first trace: {nan_positions[:20]}...")
    print(f"Total NaN in first trace: {len(nan_positions)} / {len(arr)}")
    print(f"Percentage NaN: {100*len(nan_positions)/len(arr):.1f}%")

