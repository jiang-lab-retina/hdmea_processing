import pandas as pd
import numpy as np

df = pd.read_parquet(r"m:/Python_Project/Data_Processing_2027/dataframe_phase/classification_v2/divide_conquer_method/results/labeled_dataframe.parquet")

print("=" * 80)
print("1. SHAPE AND BASIC INFO")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n" + "=" * 80)
print("2. ALL COLUMN NAMES")
print("=" * 80)
for i, col in enumerate(df.columns.tolist(), 1):
    print(f"{i:3d}. {col}")

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("4. SPATIAL COORDINATES (x, y positions)")
print("=" * 80)
spatial_cols = [c for c in df.columns if any(x in c.lower() for x in ["x", "y", "coord", "position", "centroid"])]
numeric_spatial = [c for c in spatial_cols if pd.api.types.is_numeric_dtype(df[c])]
if numeric_spatial:
    print(f"Found numeric spatial columns: {numeric_spatial}")
    for col in numeric_spatial:
        print(f"\nSample values from {col}:")
        print(df[col].head(10).tolist())
        print(f"  Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")

print("\n" + "=" * 80)
print("5. CELL TYPE/SUBTYPE/LABEL COLUMNS")
print("=" * 80)
label_cols = [c for c in df.columns if any(x in c.lower() for x in ["type", "subtype", "label", "class", "category"])]
if label_cols:
    print(f"Found label/subtype columns: {label_cols}")
    for col in label_cols:
        try:
            print(f"\nUnique values in {col} ({df[col].nunique()} unique):")
            print(df[col].value_counts())
        except (TypeError, ValueError):
            print(f"\n{col} contains arrays or unhashable types - cannot count unique values")

print("\n" + "=" * 80)
print("6. RECEPTIVE FIELD AND FEATURE COLUMNS")
print("=" * 80)
rf_cols = [c for c in df.columns if any(x in c.lower() for x in ["rf", "receptive", "field", "feature"])]
if rf_cols:
    print(f"Found RF/feature columns: {rf_cols}")
    for col in rf_cols:
        print(f"\n{col} - dtype: {df[col].dtype}, sample values:")
        print(df[col].head(5).tolist())
else:
    print("No obvious RF/feature columns found")

print("\n" + "=" * 80)
print("7. ALL OBJECT/CATEGORY COLUMNS (potential labels)")
print("=" * 80)
for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        try:
            print(f"\nUnique values in {col} ({df[col].nunique()} unique):")
            print(df[col].value_counts())
        except (TypeError, ValueError):
            print(f"\n{col} - contains arrays or unhashable types, skipping unique count")

print("\n" + "=" * 80)
print("8. FIRST 3 ROWS (full data)")
print("=" * 80)
print(df.head(3).to_string())