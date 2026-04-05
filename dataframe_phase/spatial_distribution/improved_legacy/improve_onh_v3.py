"""
Improved ONH v3: Use the EXACT legacy transformation pipeline
(`calculate_soma_polar_coordinates`), only swap in the improved ONH.

Changes from legacy:
  - ONH: robust_onh (median + MAD + R^2 filter) instead of
         calculate_optimal_intersection (R^2-weighted mean)
  - Angle: search DVNT +/-60 + 180-flip to resolve ambiguity,
         applied as an additive offset to the standard correction

Everything else is identical to compute_legacy_transformed.py:
  - soma_xy uses int() cast
  - calculate_soma_polar_coordinates for dx/dy/theta/radius
  - _calculate_angle_correction via the library function
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


# =====================================================================
# Robust ONH (only change from legacy)
# =====================================================================
def robust_onh(pathways, r2_min=0.7, mad_factor=3.0, max_dist=80.0):
    """Robust ONH via median + MAD outlier rejection."""
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

    fx = float(np.median(arr[:,0]))
    fy = float(np.median(arr[:,1]))

    errors = []
    for fit in valid.values():
        d = abs(fit.slope*fx - fy + fit.intercept) / math.sqrt(1 + fit.slope**2)
        errors.append(d**2)

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

ry0, py0, _, _, n0 = evaluate_gradient(df_freq, "legacy_transformed_x", "legacy_transformed_y")
print(f"\nBaseline (legacy): r = {ry0:.6f}, p = {py0:.2e}, n = {n0}", flush=True)


# =====================================================================
# Step A: Legacy ONH + legacy transform (reproduce baseline)
# =====================================================================
print("\n=== Step A: Legacy ONH + legacy transform (reproduce) ===", flush=True)

tx_legA = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_legA = pd.Series(np.nan, index=df_enr.index, dtype=float)

for rec, units in sorted(grouped.items()):
    pathways = {}
    for didx, uid in units:
        s, i, r = df_enr.at[didx,"ap_slope"], df_enr.at[didx,"ap_intercept"], df_enr.at[didx,"ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = calculate_optimal_intersection(pathways)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    dvnt = parse_dvnt_from_center_xy(cxy) if isinstance(cxy, str) and cxy.strip() else \
           DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    for didx, uid in units:
        sr, sc = df_enr.at[didx,"soma_row"], df_enr.at[didx,"soma_col"]
        if pd.isna(sr) or pd.isna(sc):
            continue
        polar = calculate_soma_polar_coordinates(
            soma_xy=(int(sr), int(sc)),
            intersection=onh,
            dv_position=dvnt.dv_position,
            nt_position=dvnt.nt_position,
        )
        tx_legA.at[didx] = polar.transformed_x
        ty_legA.at[didx] = polar.transformed_y

df_enr["legA_tx"] = tx_legA
df_enr["legA_ty"] = ty_legA

ryA, pyA, rxA, pxA, nA = evaluate_gradient(df_enr, "legA_tx", "legA_ty")
print(f"  gb_on_ratio vs Y: r = {ryA:.6f}, p = {pyA:.2e}, n = {nA}", flush=True)


# =====================================================================
# Step B: Robust ONH + legacy transform (same pipeline, different ONH)
# =====================================================================
print("\n=== Step B: Robust ONH + legacy transform ===", flush=True)

tx_robB = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_robB = pd.Series(np.nan, index=df_enr.index, dtype=float)

for rec, units in sorted(grouped.items()):
    pathways = {}
    for didx, uid in units:
        s, i, r = df_enr.at[didx,"ap_slope"], df_enr.at[didx,"ap_intercept"], df_enr.at[didx,"ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.7)
    if onh is None:
        onh = robust_onh(pathways, r2_min=0.5)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    dvnt = parse_dvnt_from_center_xy(cxy) if isinstance(cxy, str) and cxy.strip() else \
           DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    for didx, uid in units:
        sr, sc = df_enr.at[didx,"soma_row"], df_enr.at[didx,"soma_col"]
        if pd.isna(sr) or pd.isna(sc):
            continue
        polar = calculate_soma_polar_coordinates(
            soma_xy=(int(sr), int(sc)),
            intersection=onh,
            dv_position=dvnt.dv_position,
            nt_position=dvnt.nt_position,
        )
        tx_robB.at[didx] = polar.transformed_x
        ty_robB.at[didx] = polar.transformed_y

df_enr["robB_tx"] = tx_robB
df_enr["robB_ty"] = ty_robB

ryB, pyB, rxB, pxB, nB = evaluate_gradient(df_enr, "robB_tx", "robB_ty")
print(f"  gb_on_ratio vs Y: r = {ryB:.6f}, p = {pyB:.2e}, n = {nB}", flush=True)


# =====================================================================
# Step C: Legacy ONH + legacy transform + 180-degree fix
# Use legacy ONH to preserve 4-leaf structure, then search
# angle offset around 0 and 180 to fix ambiguity
# =====================================================================
print("\n=== Step C: Legacy ONH + legacy transform + 180-deg fix ===", flush=True)

tx_C = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_C = pd.Series(np.nan, index=df_enr.index, dtype=float)
flip_count = 0
total_recs = 0

for rec, units in sorted(grouped.items()):
    pathways = {}
    for didx, uid in units:
        s, i, r = df_enr.at[didx,"ap_slope"], df_enr.at[didx,"ap_intercept"], df_enr.at[didx,"ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = calculate_optimal_intersection(pathways)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    dvnt = parse_dvnt_from_center_xy(cxy) if isinstance(cxy, str) and cxy.strip() else \
           DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    # Get standard legacy coords for each unit
    cell_data = []
    for didx, uid in units:
        sr, sc = df_enr.at[didx,"soma_row"], df_enr.at[didx,"soma_col"]
        gb = df_enr.at[didx, "green_blue_on_ratio"]
        if pd.isna(sr) or pd.isna(sc):
            continue
        polar = calculate_soma_polar_coordinates(
            soma_xy=(int(sr), int(sc)),
            intersection=onh,
            dv_position=dvnt.dv_position,
            nt_position=dvnt.nt_position,
        )
        cell_data.append((didx, polar.transformed_x, polar.transformed_y,
                          polar.radius, polar.angle, gb))

    if len(cell_data) < 5:
        for didx, tx, ty, rad, ang, gb in cell_data:
            tx_C.at[didx] = tx
            ty_C.at[didx] = ty
        continue

    total_recs += 1
    gb_arr = np.array([c[5] for c in cell_data])
    radii = np.array([c[3] for c in cell_data])
    angles = np.array([c[4] for c in cell_data])  # already corrected by legacy

    # Try original (offset 0) vs flipped (offset 180 deg = pi)
    # Also search +/-60 deg around each in 2-deg steps
    best_offset = 0.0
    best_r = -2.0
    for base_offset in [0.0, math.pi]:
        for delta_deg in range(-60, 61, 2):
            delta_rad = math.radians(delta_deg)
            offset = base_offset + delta_rad
            y_rot = radii * np.sin(angles + offset)
            if np.std(y_rot) < 1e-10:
                continue
            r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
            if np.isnan(r_val):
                continue
            if r_val > best_r:
                best_r = r_val
                best_offset = offset

    # Fine-tune
    for delta in np.arange(-0.05, 0.051, 0.01):
        offset = best_offset + delta
        y_rot = radii * np.sin(angles + offset)
        if np.std(y_rot) < 1e-10:
            continue
        r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
        if not np.isnan(r_val) and r_val > best_r:
            best_r = r_val
            best_offset = offset

    if abs(best_offset) > 0.1:
        flip_count += 1

    for didx, tx, ty, rad, ang, gb in cell_data:
        tx_C.at[didx] = rad * math.cos(ang + best_offset)
        ty_C.at[didx] = rad * math.sin(ang + best_offset)

df_enr["stepC_tx"] = tx_C
df_enr["stepC_ty"] = ty_C

ryC, pyC, rxC, pxC, nC = evaluate_gradient(df_enr, "stepC_tx", "stepC_ty")
print(f"  Angle-adjusted recordings: {flip_count}/{total_recs}", flush=True)
print(f"  gb_on_ratio vs Y: r = {ryC:.6f}, p = {pyC:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rxC:.6f}, p = {pxC:.2e}, n = {nC}", flush=True)


# =====================================================================
# Step D: Step C + global rotation refinement
# =====================================================================
print("\n=== Step D: Step C + global rotation ===", flush=True)

valid_mask = df_enr["stepC_tx"].notna() & df_enr["stepC_ty"].notna() & df_enr["green_blue_on_ratio"].notna()
coord_mask = (df_enr["stepC_tx"].abs() < COORD_LIMIT) & (df_enr["stepC_ty"].abs() < COORD_LIMIT)
valid = df_enr[valid_mask & coord_mask]
xv = valid["stepC_tx"].values
yv = valid["stepC_ty"].values
gbv = valid["green_blue_on_ratio"].values

rv = np.sqrt(xv**2 + yv**2)
thv = np.arctan2(yv, xv)

best_global = 0.0
best_global_r = -2.0
for dd in np.arange(-20, 20.1, 0.5):
    dr = math.radians(dd)
    yr = rv * np.sin(thv + dr) * COORD_SCALE
    rr = np.corrcoef(yr, gbv)[0, 1]
    if rr > best_global_r:
        best_global_r = rr
        best_global = dd

print(f"  Global rotation: {best_global:.1f} deg", flush=True)

glob_rad = math.radians(best_global)
all_tx = df_enr["stepC_tx"].values
all_ty = df_enr["stepC_ty"].values
r_all = np.sqrt(all_tx**2 + all_ty**2)
th_all = np.arctan2(all_ty, all_tx)
df_enr["stepD_tx"] = r_all * np.cos(th_all + glob_rad)
df_enr["stepD_ty"] = r_all * np.sin(th_all + glob_rad)

ryD, pyD, rxD, pxD, nD = evaluate_gradient(df_enr, "stepD_tx", "stepD_ty")
print(f"  gb_on_ratio vs Y: r = {ryD:.6f}, p = {pyD:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rxD:.6f}, p = {pxD:.2e}, n = {nD}", flush=True)


# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*65, flush=True)
print("SUMMARY: green_blue_on_ratio vs Y (dorsal-ventral)", flush=True)
print("="*65, flush=True)
print(f"  A: Legacy ONH + legacy transform:       r = {ryA:.6f}  p = {pyA:.2e}  n = {nA}", flush=True)
print(f"  B: Robust ONH + legacy transform:       r = {ryB:.6f}  p = {pyB:.2e}  n = {nB}", flush=True)
print(f"  C: Legacy ONH + 180-fix + angle search: r = {ryC:.6f}  p = {pyC:.2e}  n = {nC}", flush=True)
print(f"  D: C + global rotation ({best_global:+.1f} deg):       r = {ryD:.6f}  p = {pyD:.2e}  n = {nD}", flush=True)

best = max(
    [(ryA, "legA_tx", "legA_ty", "Step_A"),
     (ryB, "robB_tx", "robB_ty", "Step_B"),
     (ryC, "stepC_tx", "stepC_ty", "Step_C"),
     (ryD, "stepD_tx", "stepD_ty", "Step_D")],
    key=lambda t: t[0]
)
print(f"\n  Best: {best[3]} (r = {best[0]:.6f})", flush=True)

# Save
df_out = pd.read_parquet(FREQ_PARQUET)
df_out["improved_tx"] = df_enr[best[1]]
df_out["improved_ty"] = df_enr[best[2]]

out_path = RESULTS_DIR / "labeled_dataframe_improved_coords.parquet"
df_out.to_parquet(out_path, index=True)
print(f"  Saved -> {out_path}", flush=True)
print("\nDone.", flush=True)
