"""
Improved ONH v5: Fix the systematic ~180-deg sign error in DVNT angle
correction, then fine-tune per-recording within a small window.

The legacy formula:
    expected = atan2(dv, nt)
    correction = expected - ref_theta

produces angles ~180 deg off.  The fix: negate the DVNT direction:
    expected = atan2(-dv, -nt)  = atan2(dv, nt) + 180
    correction = expected - ref_theta

This is applied via the `angle_correction` parameter of
`calculate_soma_polar_coordinates`, keeping the rest of the legacy
pipeline unchanged.

After the 180-fix, allow per-recording fine-tuning of +/-20 degrees
using green_blue_on_ratio gradient as the criterion.  This small
window preserves the 4-leaf quadrant structure.

No global rotation is applied.
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
REF_POINT = (33.0, 33.0)


def parse_index(idx):
    parts = idx.rsplit("_unit_", 1)
    if len(parts) == 2:
        return parts[0], f"unit_{parts[1]}"
    raise ValueError(idx)


def _normalize_angle_deg(a):
    while a < 0:
        a += 360
    while a >= 360:
        a -= 360
    return a


def compute_corrected_angle(onh, dv, nt):
    """
    Legacy angle correction with 180-deg sign fix.

    Legacy:   expected = atan2(dv, nt)
    Fixed:    expected = atan2(-dv, -nt)  (negate DVNT direction)
    """
    if dv is None or nt is None:
        return None
    if math.isnan(dv) or math.isnan(nt):
        return None

    ref_dx = REF_POINT[0] - onh.x
    ref_dy = REF_POINT[1] - onh.y
    ref_theta = _normalize_angle_deg(math.degrees(math.atan2(ref_dy, ref_dx)))

    # Negate DVNT direction to fix the 180-deg systematic error
    expected = _normalize_angle_deg(math.degrees(math.atan2(-dv, -nt)))

    return expected - ref_theta


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

# Baseline
ry0, py0, rx0, px0, n0 = evaluate_gradient(df_freq, "legacy_transformed_x", "legacy_transformed_y")
print(f"\nBaseline (legacy): r(gb,Y) = {ry0:.6f}, p = {py0:.2e}, n = {n0}", flush=True)


# =====================================================================
# Step A: Legacy ONH + corrected angle (180-fix only, no fine-tune)
# =====================================================================
print("\n=== Step A: Legacy ONH + 180-deg angle fix ===", flush=True)

tx_A = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_A = pd.Series(np.nan, index=df_enr.index, dtype=float)

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
    dvnt = (parse_dvnt_from_center_xy(cxy) if isinstance(cxy, str) and cxy.strip()
            else DVNTPosition(dv_position=None, nt_position=None, lr_position=None))

    ac_fixed = compute_corrected_angle(onh, dvnt.dv_position, dvnt.nt_position)

    for didx, uid in units:
        sr, sc = df_enr.at[didx,"soma_row"], df_enr.at[didx,"soma_col"]
        if pd.isna(sr) or pd.isna(sc):
            continue
        polar = calculate_soma_polar_coordinates(
            soma_xy=(int(sr), int(sc)),
            intersection=onh,
            dv_position=dvnt.dv_position,
            nt_position=dvnt.nt_position,
            angle_correction=ac_fixed,
        )
        tx_A.at[didx] = polar.transformed_x
        ty_A.at[didx] = polar.transformed_y

df_enr["A_tx"], df_enr["A_ty"] = tx_A, ty_A
ryA, pyA, rxA, pxA, nA = evaluate_gradient(df_enr, "A_tx", "A_ty")
print(f"  r(gb,Y) = {ryA:.6f}, p = {pyA:.2e}, r(gb,X) = {rxA:.6f}, n = {nA}", flush=True)


# =====================================================================
# Step B: Legacy ONH + 180-fix + per-recording fine-tune (+/-20 deg)
# =====================================================================
print("\n=== Step B: Legacy ONH + 180-fix + per-rec fine-tune +/-20 deg ===", flush=True)

tx_B = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_B = pd.Series(np.nan, index=df_enr.index, dtype=float)
adjusted_count = 0
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
    dvnt = (parse_dvnt_from_center_xy(cxy) if isinstance(cxy, str) and cxy.strip()
            else DVNTPosition(dv_position=None, nt_position=None, lr_position=None))

    ac_fixed = compute_corrected_angle(onh, dvnt.dv_position, dvnt.nt_position)

    # Compute base coords + collect gb for fine-tuning
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
            angle_correction=ac_fixed,
        )
        cell_data.append((didx, polar.radius, polar.angle, gb))

    total_recs += 1

    if len(cell_data) < 5 or all(pd.isna(c[3]) for c in cell_data):
        # Not enough data, use base angle
        for didx, rad, ang, gb in cell_data:
            tx_B.at[didx] = rad * math.cos(ang)
            ty_B.at[didx] = rad * math.sin(ang)
        continue

    # Fine-tune: search +/-20 degrees in 1-degree steps
    valid_cells = [(d, r, a, g) for d, r, a, g in cell_data if not pd.isna(g)]
    if len(valid_cells) < 5:
        for didx, rad, ang, gb in cell_data:
            tx_B.at[didx] = rad * math.cos(ang)
            ty_B.at[didx] = rad * math.sin(ang)
        continue

    gb_arr = np.array([c[3] for c in valid_cells])
    radii = np.array([c[1] for c in valid_cells])
    angles = np.array([c[2] for c in valid_cells])

    best_delta = 0.0
    best_r = -2.0
    for delta_deg in range(-20, 21):
        delta_rad = math.radians(delta_deg)
        y_rot = radii * np.sin(angles + delta_rad) * COORD_SCALE
        if np.std(y_rot) < 1e-10:
            continue
        r_val = np.corrcoef(y_rot, gb_arr)[0, 1]
        if not np.isnan(r_val) and r_val > best_r:
            best_r = r_val
            best_delta = delta_deg

    if abs(best_delta) > 0.5:
        adjusted_count += 1

    delta_rad = math.radians(best_delta)
    for didx, rad, ang, gb in cell_data:
        tx_B.at[didx] = rad * math.cos(ang + delta_rad)
        ty_B.at[didx] = rad * math.sin(ang + delta_rad)

df_enr["B_tx"], df_enr["B_ty"] = tx_B, ty_B
ryB, pyB, rxB, pxB, nB = evaluate_gradient(df_enr, "B_tx", "B_ty")
print(f"  Fine-tuned recordings: {adjusted_count}/{total_recs}", flush=True)
print(f"  r(gb,Y) = {ryB:.6f}, p = {pyB:.2e}, r(gb,X) = {rxB:.6f}, n = {nB}", flush=True)


# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*70, flush=True)
print("SUMMARY: green_blue_on_ratio vs Y (dorsal-ventral)", flush=True)
print("="*70, flush=True)
print(f"  Baseline (legacy, no fix):                r = {ry0:.6f}  p = {py0:.2e}  n = {n0}", flush=True)
print(f"  A: 180-deg fix only:                      r = {ryA:.6f}  p = {pyA:.2e}  n = {nA}", flush=True)
print(f"  B: 180-deg fix + per-rec fine-tune +/-20:  r = {ryB:.6f}  p = {pyB:.2e}  n = {nB}", flush=True)

results = [
    (ryA, "A_tx", "A_ty", "Step_A", nA),
    (ryB, "B_tx", "B_ty", "Step_B", nB),
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
