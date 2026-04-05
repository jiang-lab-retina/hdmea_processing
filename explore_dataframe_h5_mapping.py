import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(".")
PARQUET_FILE = PROJECT_ROOT / "dataframe_phase/classification_v2/divide_conquer_method/results/labeled_dataframe.parquet"
HDF5_DIR = PROJECT_ROOT / "Projects/unified_pipeline/export_dsgc_sta_updated"

print("=" * 80)
print("EXPLORING LABELED DATAFRAME STRUCTURE")
print("=" * 80)

df = pd.read_parquet(PARQUET_FILE)
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {list(df.columns)}")

print("\n" + "-" * 80)
print("INDEX STRUCTURE ANALYSIS")
print("-" * 80)
print(f"Index type: {type(df.index)}")
print(f"Is MultiIndex: {isinstance(df.index, pd.MultiIndex)}")
print(f"Index names: {df.index.names}")
print(f"Index nlevels: {df.index.nlevels}")
print(f"Index dtype: {df.index.dtype}")

print("\nFirst 10 index values:")
for i, idx_val in enumerate(df.index[:10], 1):
    print(f"  {i:2d}. {idx_val}")

print("\nSample index values (first 5):")
for idx_val in df.index[:5]:
    print(f"  - {idx_val}")

def parse_index(index_value: str) -> tuple:
    parts = index_value.rsplit("_unit_", 1)
    if len(parts) == 2:
        dataset_id = parts[0]
        unit_id = f"unit_{parts[1]}"
        return dataset_id, unit_id
    else:
        raise ValueError(f"Cannot parse index: {index_value}")

print("\n" + "-" * 80)
print("INDEX PARSING ANALYSIS")
print("-" * 80)

print("\nParsing first 5 index values:")
for idx_val in df.index[:5]:
    try:
        dataset_id, unit_id = parse_index(idx_val)
        print(f"  Index: {idx_val}")
        print(f"    -> dataset_id: {dataset_id}")
        print(f"    -> unit_id: {unit_id}")
    except Exception as e:
        print(f"  ERROR parsing {idx_val}: {e}")

print("\n" + "-" * 80)
print("H5 FILE MAPPING ANALYSIS")
print("-" * 80)

grouped = defaultdict(list)
for idx in df.index:
    try:
        dataset_id, unit_id = parse_index(idx)
        grouped[dataset_id].append((idx, unit_id))
    except Exception as e:
        print(f"ERROR parsing index {idx}: {e}")

print(f"\nTotal unique dataset_ids (H5 files): {len(grouped)}")
print(f"Total dataframe rows: {len(df)}")

print("\nDataset ID to H5 file mapping (first 10):")
for i, (dataset_id, unit_list) in enumerate(list(grouped.items())[:10], 1):
    h5_path = HDF5_DIR / f"{dataset_id}.h5"
    h5_exists = h5_path.exists()
    print(f"\n  {i}. Dataset ID: {dataset_id}")
    print(f"     H5 file path: {h5_path}")
    print(f"     H5 file exists: {h5_exists}")
    print(f"     Number of units in this dataset: {len(unit_list)}")
    unit_ids = [uid for _, uid in unit_list[:5]]
    print(f"     Unit IDs: {unit_ids}{'...' if len(unit_list) > 5 else ''}")

print("\n" + "-" * 80)
print("H5 DIRECTORY CHECK")
print("-" * 80)
print(f"HDF5_DIR: {HDF5_DIR}")
print(f"HDF5_DIR exists: {HDF5_DIR.exists()}")

if HDF5_DIR.exists():
    h5_files = list(HDF5_DIR.glob("*.h5"))
    print(f"Number of .h5 files in directory: {len(h5_files)}")
    if h5_files:
        print("\nSample H5 files (first 5):")
        for h5_file in h5_files[:5]:
            print(f"  - {h5_file.name}")

print("\n" + "-" * 80)
print("H5 FILE AVAILABILITY CHECK")
print("-" * 80)

missing_files = []
existing_files = []
for dataset_id in grouped.keys():
    h5_path = HDF5_DIR / f"{dataset_id}.h5"
    if h5_path.exists():
        existing_files.append(dataset_id)
    else:
        missing_files.append(dataset_id)

print(f"Dataset IDs with existing H5 files: {len(existing_files)}")
print(f"Dataset IDs with missing H5 files: {len(missing_files)}")

if missing_files:
    print("\nMissing H5 files (first 10):")
    for dataset_id in missing_files[:10]:
        print(f"  - {dataset_id}.h5")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
