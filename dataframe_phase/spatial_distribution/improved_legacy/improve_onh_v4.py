"""
Improved ONH v4: Preserve 4-leaf pattern from DVNT angle correction.

The 4-leaf pattern comes from DVNT-anchored rotation at (33,33).
We must NOT change per-recording angles -- only improve ONH position
and optionally apply a single global rotation.

Steps:
  A: Legacy ONH + legacy transform (reproduce baseline)
  B: Robust ONH + legacy transform
  C: Legacy ONH + legacy transform + global rotation
  D: Robust ONH + legacy transform + global rotation
"""
import sys, math, importlib.util
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


def robust_onh(pathways, r2_min=0.7, mad_factor=3.0, max_dist=80.0):
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


def compute_coords(df_enr, grouped, onh_func, label):
    """
    Compute transformed coords using the exact legacy pipeline.
    onh_func: callable(pathways) -> APIntersection or None
    """
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
                                      r_value=float(r), p_value=0, std_err=0,
                                      num_points=0)

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


def apply_global_rotation(tx_series, ty_series, gb_series, label):
    """Find and apply the best global rotation to maximize r(gb, Y)."""
    valid = tx_series.notna() & ty_series.notna() & gb_series.notna()
    cmask = (tx_series.abs() < COORD_LIMIT) & (ty_series.abs() < COORD_LIMIT)
    sel = valid & cmask
    xv = tx_series[sel].values
    yv = ty_series[sel].values
    gv = gb_series[sel].values

    rv = np.sqrt(xv**2 + yv**2)
    thv = np.arctan2(yv, xv)

    best_deg, best_r = 0.0, -2.0
    for dd in np.arange(-180, 180, 1.0):
        dr = math.radians(dd)
        yr = rv * np.sin(thv + dr) * COORD_SCALE
        rr = np.corrcoef(yr, gv)[0, 1]
        if not np.isnan(rr) and rr > best_r:
            best_r = rr
            best_deg = dd

    # Fine-tune
    for dd in np.arange(best_deg - 1, best_deg + 1.01, 0.1):
        dr = math.radians(dd)
        yr = rv * np.sin(thv + dr) * COORD_SCALE
        rr = np.corrcoef(yr, gv)[0, 1]
        if not np.isnan(rr) and rr > best_r:
            best_r = rr
            best_deg = dd

    print(f"  {label}: global rotation = {best_deg:.1f} deg", flush=True)

    grad = math.radians(best_deg)
    all_x = tx_series.values.copy()
    all_y = ty_series.values.copy()
    r_all = np.sqrt(all_x**2 + all_y**2)
    th_all = np.arctan2(all_y, all_x)
    new_tx = pd.Series(r_all * np.cos(th_all + grad), index=tx_series.index)
    new_ty = pd.Series(r_all * np.sin(th_all + grad), index=ty_series.index)
    return new_tx, new_ty, best_deg


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
# Step A: Legacy ONH + legacy transform
# =====================================================================
print("\n=== Step A: Legacy ONH + legacy transform ===", flush=True)
tx_A, ty_A = compute_coords(df_enr, grouped, calculate_optimal_intersection, "A")
df_enr["A_tx"], df_enr["A_ty"] = tx_A, ty_A
ryA, pyA, rxA, pxA, nA = evaluate_gradient(df_enr, "A_tx", "A_ty")
print(f"  r(gb,Y) = {ryA:.6f}, p = {pyA:.2e}, r(gb,X) = {rxA:.6f}, n = {nA}", flush=True)

# =====================================================================
# Step B: Robust ONH + legacy transform
# =====================================================================
print("\n=== Step B: Robust ONH + legacy transform ===", flush=True)

def robust_onh_with_fallback(pathways):
    onh = robust_onh(pathways, r2_min=0.7)
    if onh is None:
        onh = robust_onh(pathways, r2_min=0.5)
    return onh

tx_B, ty_B = compute_coords(df_enr, grouped, robust_onh_with_fallback, "B")
df_enr["B_tx"], df_enr["B_ty"] = tx_B, ty_B
ryB, pyB, rxB, pxB, nB = evaluate_gradient(df_enr, "B_tx", "B_ty")
print(f"  r(gb,Y) = {ryB:.6f}, p = {pyB:.2e}, r(gb,X) = {rxB:.6f}, n = {nB}", flush=True)

# =====================================================================
# Step C: Legacy ONH + legacy transform + global rotation
# =====================================================================
print("\n=== Step C: Legacy ONH + global rotation ===", flush=True)
tx_C, ty_C, rot_C = apply_global_rotation(
    tx_A, ty_A, df_enr["green_blue_on_ratio"], "C")
df_enr["C_tx"], df_enr["C_ty"] = tx_C, ty_C
ryC, pyC, rxC, pxC, nC = evaluate_gradient(df_enr, "C_tx", "C_ty")
print(f"  r(gb,Y) = {ryC:.6f}, p = {pyC:.2e}, r(gb,X) = {rxC:.6f}, n = {nC}", flush=True)

# =====================================================================
# Step D: Robust ONH + legacy transform + global rotation
# =====================================================================
print("\n=== Step D: Robust ONH + global rotation ===", flush=True)
tx_D, ty_D, rot_D = apply_global_rotation(
    tx_B, ty_B, df_enr["green_blue_on_ratio"], "D")
df_enr["D_tx"], df_enr["D_ty"] = tx_D, ty_D
ryD, pyD, rxD, pxD, nD = evaluate_gradient(df_enr, "D_tx", "D_ty")
print(f"  r(gb,Y) = {ryD:.6f}, p = {pyD:.2e}, r(gb,X) = {rxD:.6f}, n = {nD}", flush=True)

# =====================================================================
# Summary & save
# =====================================================================
print("\n" + "="*70, flush=True)
print("SUMMARY: green_blue_on_ratio vs Y (dorsal-ventral)", flush=True)
print("="*70, flush=True)
print(f"  A: Legacy ONH + legacy transform:          r = {ryA:.6f}  p = {pyA:.2e}  n = {nA}", flush=True)
print(f"  B: Robust ONH + legacy transform:          r = {ryB:.6f}  p = {pyB:.2e}  n = {nB}", flush=True)
print(f"  C: Legacy ONH + global rot ({rot_C:+.1f} deg):     r = {ryC:.6f}  p = {pyC:.2e}  n = {nC}", flush=True)
print(f"  D: Robust ONH + global rot ({rot_D:+.1f} deg):     r = {ryD:.6f}  p = {pyD:.2e}  n = {nD}", flush=True)

results = [
    (ryA, "A_tx", "A_ty", "Step_A", nA),
    (ryB, "B_tx", "B_ty", "Step_B", nB),
    (ryC, "C_tx", "C_ty", "Step_C", nC),
    (ryD, "D_tx", "D_ty", "Step_D", nD),
]
best = max(results, key=lambda t: t[0])
print(f"\n  Best: {best[3]} (r = {best[0]:.6f}, n = {best[4]})", flush=True)

df_out = pd.read_parquet(FREQ_PARQUET)
df_out["improved_tx"] = df_enr[best[1]]
df_out["improved_ty"] = df_enr[best[2]]

out_path = RESULTS_DIR / "labeled_dataframe_improved_coords.parquet"
df_out.to_parquet(out_path, index=True)
print(f"  Saved -> {out_path}", flush=True)
print("\nDone.", flush=True)
