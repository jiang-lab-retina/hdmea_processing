# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Extract AP Tracking Data & Compute Legacy Transformed Coordinates
#
# **Step 1**: Read H5 files and extract per-unit columns:
#   - ap_slope, ap_intercept, ap_r_value (pathway fit)
#   - soma_row, soma_col (refined soma position)
#   - axon_centroids (raw centroid array as list of [t,row,col])
#   - center_xy (per-recording DVNT string)
#
# **Step 2**: Compute legacy ONH per recording from all pathways in that
#   recording, then derive legacy_transformed_x / legacy_transformed_y.
#
# Saves enriched dataframe as parquet (no H5 access needed afterwards).

# %%
import sys
import os
import importlib.util
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py

print("Imports done", flush=True)

# %%
# ---------- project paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SPATIAL_DIR.parent.parent

# Direct-import pathway_analysis and dvnt_parser without triggering
# the package __init__.py (which pulls in PyTorch via core.py).
def _import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pa = _import_module_from_file(
    "pathway_analysis",
    PROJECT_ROOT / "src" / "hdmea" / "features" / "ap_tracking" / "pathway_analysis.py",
)
_dvnt = _import_module_from_file(
    "dvnt_parser",
    PROJECT_ROOT / "src" / "hdmea" / "features" / "ap_tracking" / "dvnt_parser.py",
)

APPathway = _pa.APPathway
APIntersection = _pa.APIntersection
calculate_optimal_intersection = _pa.calculate_optimal_intersection
calculate_soma_polar_coordinates = _pa.calculate_soma_polar_coordinates

DVNTPosition = _dvnt.DVNTPosition
parse_dvnt_from_center_xy = _dvnt.parse_dvnt_from_center_xy

print("Loaded pathway_analysis + dvnt_parser", flush=True)

# %%
# ---------- I/O paths ----------
INPUT_PARQUET = (
    PROJECT_ROOT / "dataframe_phase" / "classification_v2"
    / "divide_conquer_method" / "results" / "labeled_dataframe.parquet"
)
HDF5_DIR = PROJECT_ROOT / "Projects" / "unified_pipeline" / "export_dsgc_sta_updated"

RESULTS_DIR = SPATIAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_PARQUET = RESULTS_DIR / "labeled_dataframe_enriched.parquet"
FINAL_PARQUET = RESULTS_DIR / "labeled_dataframe_with_legacy_coords.parquet"


# ==========================================================================
# Helpers
# ==========================================================================

# %%
def parse_index(index_value):
    parts = index_value.rsplit("_unit_", 1)
    if len(parts) == 2:
        return parts[0], f"unit_{parts[1]}"
    raise ValueError(f"Cannot parse index: {index_value}")


def group_indices_by_file(df):
    grouped = defaultdict(list)
    for idx in df.index:
        try:
            dataset_id, unit_id = parse_index(idx)
            grouped[dataset_id].append((idx, unit_id))
        except ValueError:
            continue
    return grouped


def _read_scalar(f, path, default=np.nan):
    if path not in f:
        return default
    val = f[path][()]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode()
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return default
    return val


# ==========================================================================
# Step 1: Extract H5 data into dataframe columns
# ==========================================================================

# %%
def extract_h5_data(dataset_id, units, h5_dir):
    """
    Read AP pathway, soma, centroid, and DVNT data from one H5 file.

    Returns dict {df_index: {col: value, ...}} for each unit.
    """
    nan_row = {
        "ap_slope": np.nan,
        "ap_intercept": np.nan,
        "ap_r_value": np.nan,
        "soma_row": np.nan,
        "soma_col": np.nan,
        "axon_centroids": None,
        "center_xy": "",
    }
    result = {idx: dict(nan_row) for idx, _ in units}

    h5_path = h5_dir / f"{dataset_id}.h5"
    if not h5_path.exists():
        return result

    with h5py.File(str(h5_path), "r") as f:
        # Recording-level: DVNT
        center_xy = _read_scalar(f, "metadata/gsheet_row/Center_xy", default="")
        if not isinstance(center_xy, str):
            center_xy = ""

        for df_idx, unit_id in units:
            result[df_idx]["center_xy"] = center_xy

            ap_base = f"units/{unit_id}/features/ap_tracking"

            # Pathway fit
            pw = f"{ap_base}/ap_pathway"
            slope = _read_scalar(f, f"{pw}/slope")
            intercept = _read_scalar(f, f"{pw}/intercept")
            r_value = _read_scalar(f, f"{pw}/r_value")
            result[df_idx]["ap_slope"] = float(slope) if not (isinstance(slope, float) and np.isnan(slope)) else np.nan
            result[df_idx]["ap_intercept"] = float(intercept) if not (isinstance(intercept, float) and np.isnan(intercept)) else np.nan
            result[df_idx]["ap_r_value"] = float(r_value) if not (isinstance(r_value, float) and np.isnan(r_value)) else np.nan

            # Refined soma
            sm = f"{ap_base}/refined_soma"
            soma_x = _read_scalar(f, f"{sm}/x")
            soma_y = _read_scalar(f, f"{sm}/y")
            result[df_idx]["soma_row"] = float(soma_x) if not (isinstance(soma_x, float) and np.isnan(soma_x)) else np.nan
            result[df_idx]["soma_col"] = float(soma_y) if not (isinstance(soma_y, float) and np.isnan(soma_y)) else np.nan

            # Axon centroids (stored as JSON string for parquet compatibility)
            centroid_path = f"{ap_base}/post_processed_data/axon_centroids"
            if centroid_path in f:
                arr = f[centroid_path][()]
                if arr.size > 0:
                    result[df_idx]["axon_centroids"] = json.dumps(arr.tolist())

    return result


# ==========================================================================
# Step 2: Compute legacy transformed coordinates from dataframe
# ==========================================================================

# %%
def compute_legacy_coords(df):
    """
    Group by recording, compute legacy ONH per recording, then derive
    legacy_transformed_x/y for each unit.  Operates purely on dataframe
    columns (no H5 access).
    """
    legacy_tx = pd.Series(np.nan, index=df.index, dtype=float)
    legacy_ty = pd.Series(np.nan, index=df.index, dtype=float)

    grouped = group_indices_by_file(df)

    for rec_i, (dataset_id, units) in enumerate(sorted(grouped.items())):
        # Collect pathways from ALL units in this recording that have valid fits
        pathways = {}
        for df_idx, unit_id in units:
            slope = df.at[df_idx, "ap_slope"]
            intercept = df.at[df_idx, "ap_intercept"]
            r_value = df.at[df_idx, "ap_r_value"]
            if pd.isna(slope) or pd.isna(intercept) or pd.isna(r_value):
                continue
            pathways[unit_id] = APPathway(
                slope=float(slope),
                intercept=float(intercept),
                r_value=float(r_value),
                p_value=0.0,
                std_err=0.0,
                num_points=0,
            )

        # Legacy ONH
        legacy_onh = calculate_optimal_intersection(pathways)
        if legacy_onh is None:
            continue

        # DVNT (same for all units in recording - take from first unit)
        center_xy = df.at[units[0][0], "center_xy"]
        if isinstance(center_xy, str) and center_xy.strip():
            dvnt = parse_dvnt_from_center_xy(center_xy)
        else:
            dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

        # Per-unit transformed coordinates
        for df_idx, unit_id in units:
            soma_row = df.at[df_idx, "soma_row"]
            soma_col = df.at[df_idx, "soma_col"]
            if pd.isna(soma_row) or pd.isna(soma_col):
                continue

            polar = calculate_soma_polar_coordinates(
                soma_xy=(int(soma_row), int(soma_col)),
                intersection=legacy_onh,
                dv_position=dvnt.dv_position,
                nt_position=dvnt.nt_position,
            )
            legacy_tx.at[df_idx] = polar.transformed_x
            legacy_ty.at[df_idx] = polar.transformed_y

        if (rec_i + 1) % 50 == 0:
            print(f"  [{rec_i+1}/{len(grouped)}] computed", flush=True)

    print(f"  [{len(grouped)}/{len(grouped)}] done", flush=True)
    return legacy_tx, legacy_ty


# ==========================================================================
# Main
# ==========================================================================

# %%
print("\n=== Step 1: Extract H5 data into dataframe ===", flush=True)
df = pd.read_parquet(INPUT_PARQUET)
print(f"  Loaded dataframe: {df.shape}", flush=True)

grouped = group_indices_by_file(df)
print(f"  Recordings: {len(grouped)}", flush=True)

# Pre-allocate new columns
df["ap_slope"] = np.nan
df["ap_intercept"] = np.nan
df["ap_r_value"] = np.nan
df["soma_row"] = np.nan
df["soma_col"] = np.nan
df["axon_centroids"] = None  # JSON strings
df["center_xy"] = ""

n_total = len(grouped)
for i, (dataset_id, units) in enumerate(sorted(grouped.items())):
    rec_data = extract_h5_data(dataset_id, units, HDF5_DIR)
    for df_idx, cols in rec_data.items():
        for col, val in cols.items():
            df.at[df_idx, col] = val
    if (i + 1) % 50 == 0 or (i + 1) == n_total:
        print(f"  [{i+1}/{n_total}] {dataset_id}", flush=True)

# Summary of extraction
valid_slope = df["ap_slope"].notna().sum()
valid_soma = df["soma_row"].notna().sum()
valid_centroids = df["axon_centroids"].notna().sum()
valid_dvnt = (df["center_xy"].str.strip() != "").sum()
print(f"\n  Extraction summary:", flush=True)
print(f"    Valid ap_slope:       {valid_slope}", flush=True)
print(f"    Valid soma_row:       {valid_soma}", flush=True)
print(f"    Valid axon_centroids: {valid_centroids}", flush=True)
print(f"    Valid center_xy:      {valid_dvnt}", flush=True)

# %%
print(f"\n  Saving enriched dataframe -> {ENRICHED_PARQUET}", flush=True)
df.to_parquet(ENRICHED_PARQUET, index=True)
print(f"  Saved.", flush=True)

# %%
print("\n=== Step 2: Compute legacy transformed coordinates ===", flush=True)
legacy_tx, legacy_ty = compute_legacy_coords(df)

df["legacy_transformed_x"] = legacy_tx
df["legacy_transformed_y"] = legacy_ty

valid = df["legacy_transformed_x"].notna().sum()
print(f"\n  Results:", flush=True)
print(f"    Total units:     {len(df)}", flush=True)
print(f"    Valid legacy tx: {valid} ({100*valid/len(df):.1f}%)", flush=True)
print(f"    NaN legacy tx:   {len(df) - valid}", flush=True)

# Spot check
print(f"\n  Sample (current vs legacy):", flush=True)
cols = ["transformed_x", "transformed_y", "legacy_transformed_x", "legacy_transformed_y"]
print(df[cols].dropna(subset=["legacy_transformed_x"]).head(10).to_string(), flush=True)

# %%
print(f"\n  Saving final dataframe -> {FINAL_PARQUET}", flush=True)
df.to_parquet(FINAL_PARQUET, index=True)
print(f"\nDone.", flush=True)
