"""
Improved ONH v6: ONLY change ONH detection. Everything else is
identical to the legacy pipeline -- same angle correction, same
calculate_soma_polar_coordinates, no sign flips, no rotation.
"""
import math, importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SPATIAL_DIR.parent.parent

def _import_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pa = _import_mod("pathway_analysis",
    PROJECT_ROOT / "src" / "hdmea" / "features" / "ap_tracking" / "pathway_analysis.py")
_dvnt = _import_mod("dvnt_parser",
    PROJECT_ROOT / "src" / "hdmea" / "features" / "ap_tracking" / "dvnt_parser.py")

APPathway = _pa.APPathway
APIntersection = _pa.APIntersection
calculate_optimal_intersection = _pa.calculate_optimal_intersection
calculate_soma_polar_coordinates = _pa.calculate_soma_polar_coordinates

DVNTPosition = _dvnt.DVNTPosition
parse_dvnt_from_center_xy = _dvnt.parse_dvnt_from_center_xy

print("Imports done", flush=True)

ENRICHED_PARQUET = SPATIAL_DIR / "results" / "labeled_dataframe_enriched.parquet"
FREQ_PARQUET = SPATIAL_DIR / "results" / "labeled_dataframe_with_legacy_coords_freq.parquet"
RESULTS_DIR = SPATIAL_DIR / "improved_legacy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COORD_SCALE = 16
COORD_LIMIT = 100


def parse_index(idx):
    parts = idx.rsplit("_unit_", 1)
    if len(parts) == 2:
        return parts[0], f"unit_{parts[1]}"
    raise ValueError(idx)


# =====================================================================
# ONH variants
# =====================================================================

def robust_onh(pathways, r2_min=0.7, mad_factor=3.0, max_dist=80.0):
    """Median + MAD outlier rejection, high R^2 threshold."""
    valid = {k: v for k, v in pathways.items()
             if v is not None and v.r_value**2 >= r2_min}
    if len(valid) < 2:
        return None
    ids = list(valid.keys())
    pts = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            f1, f2 = valid[ids[i]], valid[ids[j]]
            dm = f1.slope - f2.slope
            if abs(dm) < 0.05:
                continue
            xi = (f2.intercept - f1.intercept) / dm
            yi = f1.slope * xi + f1.intercept
            if abs(xi - 33) > max_dist or abs(yi - 33) > max_dist:
                continue
            pts.append((xi, yi))
    if len(pts) == 0:
        return None
    arr = np.array(pts)
    med_x, med_y = np.median(arr[:,0]), np.median(arr[:,1])
    if len(pts) >= 5:
        dists = np.sqrt((arr[:,0] - med_x)**2 + (arr[:,1] - med_y)**2)
        mad = np.median(dists)
        if mad > 0:
            keep = dists < mad_factor * mad
            if keep.sum() >= 3:
                arr = arr[keep]
    fx, fy = float(np.median(arr[:,0])), float(np.median(arr[:,1]))
    errors = [((fit.slope*fx - fy + fit.intercept)**2 / (1 + fit.slope**2))
              for fit in valid.values()]
    return APIntersection(x=fx, y=fy, mse=float(np.mean(errors)))


def robust_onh_with_fallback(pathways):
    onh = robust_onh(pathways, r2_min=0.7)
    if onh is None:
        onh = robust_onh(pathways, r2_min=0.5)
    return onh


# =====================================================================
# Shared transform function (exact legacy pipeline)
# =====================================================================

def compute_coords(df_enr, grouped, onh_func):
    """Use exactly calculate_soma_polar_coordinates with the given ONH func."""
    tx = pd.Series(np.nan, index=df_enr.index, dtype=float)
    ty = pd.Series(np.nan, index=df_enr.index, dtype=float)
    for rec, units in sorted(grouped.items()):
        pathways = {}
        for didx, uid in units:
            s = df_enr.at[didx, "ap_slope"]
            i = df_enr.at[didx, "ap_intercept"]
            r = df_enr.at[didx, "ap_r_value"]
            if pd.isna(s) or pd.isna(i) or pd.isna(r):
                continue
            pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                      r_value=float(r), p_value=0,
                                      std_err=0, num_points=0)
        onh = onh_func(pathways)
        if onh is None:
            continue
        cxy = df_enr.at[units[0][0], "center_xy"]
        dvnt = (parse_dvnt_from_center_xy(cxy)
                if isinstance(cxy, str) and cxy.strip()
                else DVNTPosition(dv_position=None, nt_position=None,
                                  lr_position=None))
        for didx, uid in units:
            sr = df_enr.at[didx, "soma_row"]
            sc = df_enr.at[didx, "soma_col"]
            if pd.isna(sr) or pd.isna(sc):
                continue
            polar = calculate_soma_polar_coordinates(
                soma_xy=(int(sr), int(sc)),
                intersection=onh,
                dv_position=dvnt.dv_position,
                nt_position=dvnt.nt_position,
            )
            tx.at[didx] = polar.transformed_x
            ty.at[didx] = polar.transformed_y
    return tx, ty


def evaluate_gradient(df, xcol, ycol, feat="green_blue_on_ratio"):
    sub = df[[xcol, ycol, feat]].replace([np.inf, -np.inf], np.nan).dropna()
    mask = (sub[xcol].abs() < COORD_LIMIT) & (sub[ycol].abs() < COORD_LIMIT)
    sub = sub[mask]
    if len(sub) < 30:
        return np.nan, np.nan, np.nan, np.nan, len(sub)
    y = sub[ycol].values * COORD_SCALE
    x = sub[xcol].values * COORD_SCALE
    c = sub[feat].values
    ry, py = pearsonr(y, c)
    rx, px = pearsonr(x, c)
    return ry, py, rx, px, len(sub)


# =====================================================================
# Load data
# =====================================================================
print("\nLoading data ...", flush=True)
df_enr = pd.read_parquet(ENRICHED_PARQUET)
df_freq = pd.read_parquet(FREQ_PARQUET)

if "green_blue_on_ratio" not in df_enr.columns:
    df_enr["green_blue_on_ratio"] = df_freq["green_blue_on_ratio"]

grouped = defaultdict(list)
for idx in df_enr.index:
    try:
        rec, uid = parse_index(idx)
        grouped[rec].append((idx, uid))
    except ValueError:
        continue

print(f"  Recordings: {len(grouped)}", flush=True)

# =====================================================================
# Run
# =====================================================================
print("\n=== Legacy ONH (calculate_optimal_intersection) ===", flush=True)
tx_leg, ty_leg = compute_coords(df_enr, grouped, calculate_optimal_intersection)
df_enr["leg_tx"], df_enr["leg_ty"] = tx_leg, ty_leg
ry_leg, py_leg, rx_leg, px_leg, n_leg = evaluate_gradient(df_enr, "leg_tx", "leg_ty")
print(f"  r(gb,Y) = {ry_leg:.6f}, p = {py_leg:.2e}, r(gb,X) = {rx_leg:.6f}, n = {n_leg}", flush=True)

print("\n=== Robust ONH (R^2>0.7, median+MAD) ===", flush=True)
tx_rob, ty_rob = compute_coords(df_enr, grouped, robust_onh_with_fallback)
df_enr["rob_tx"], df_enr["rob_ty"] = tx_rob, ty_rob
ry_rob, py_rob, rx_rob, px_rob, n_rob = evaluate_gradient(df_enr, "rob_tx", "rob_ty")
print(f"  r(gb,Y) = {ry_rob:.6f}, p = {py_rob:.2e}, r(gb,X) = {rx_rob:.6f}, n = {n_rob}", flush=True)

# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*70, flush=True)
print("SUMMARY: green_blue_on_ratio vs Y", flush=True)
print("="*70, flush=True)
print(f"  Legacy ONH:  r = {ry_leg:.6f}  p = {py_leg:.2e}  n = {n_leg}", flush=True)
print(f"  Robust ONH:  r = {ry_rob:.6f}  p = {py_rob:.2e}  n = {n_rob}", flush=True)

best = max([(ry_leg, "leg_tx", "leg_ty", "Legacy", n_leg),
            (ry_rob, "rob_tx", "rob_ty", "Robust", n_rob)],
           key=lambda t: t[0])
print(f"\n  Best: {best[3]} (r = {best[0]:.6f}, n = {best[4]})", flush=True)

df_out = pd.read_parquet(FREQ_PARQUET)
df_out["improved_tx"] = df_enr[best[1]]
df_out["improved_ty"] = df_enr[best[2]]

out_path = RESULTS_DIR / "labeled_dataframe_improved_coords.parquet"
df_out.to_parquet(out_path, index=True)
print(f"  Saved -> {out_path}", flush=True)
print("\nDone.", flush=True)
