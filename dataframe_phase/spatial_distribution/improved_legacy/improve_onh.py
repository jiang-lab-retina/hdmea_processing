"""
Improved ONH detection and coordinate transformation.

Strategy:
  1. Robust ONH: median of pairwise intersections + MAD outlier rejection
  2. R^2 filtering: only use pathways with high R^2
  3. Intersection distance filtering: remove outlier intersections
  4. Per-recording angle correction with 180-degree ambiguity check
  5. Iterative: evaluate green_blue_on_ratio vs Y gradient at each step

Validation: green_blue_on_ratio should be high dorsally (positive Y) and
low ventrally (negative Y), giving a POSITIVE Pearson r with Y.

Coordinate axes:
  X: Temporal (-) to Nasal (+)
  Y: Ventral (-) to Dorsal (+)
"""
import sys, math, importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# --------------- project paths ---------------
SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SPATIAL_DIR.parent.parent

# Direct-import pathway functions (no PyTorch)
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
SomaPolarCoordinates = _pa.SomaPolarCoordinates
parse_dvnt_from_center_xy = _dvnt.parse_dvnt_from_center_xy
DVNTPosition = _dvnt.DVNTPosition

print("Imports done", flush=True)

# --------------- I/O ---------------
ENRICHED_PARQUET = SPATIAL_DIR / "results" / "labeled_dataframe_enriched.parquet"
FREQ_PARQUET = SPATIAL_DIR / "results" / "labeled_dataframe_with_legacy_coords_freq.parquet"
RESULTS_DIR = SPATIAL_DIR / "improved_legacy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COORD_SCALE = 16
COORD_LIMIT = 100


# =====================================================================
# Helper: parse recording / unit from index
# =====================================================================
def parse_index(idx):
    parts = idx.rsplit("_unit_", 1)
    if len(parts) == 2:
        return parts[0], f"unit_{parts[1]}"
    raise ValueError(idx)


# =====================================================================
# Robust ONH detection
# =====================================================================
def robust_onh(pathways, r2_min=0.5, mad_factor=3.0, max_dist=100.0):
    """
    Robust ONH from pairwise intersections:
      1. Filter pathways by R^2 threshold
      2. Compute all pairwise intersections
      3. Reject intersections > max_dist from array centre (33,33)
      4. Use median, then reject > mad_factor * MAD from median
      5. Return median of surviving intersections
    """
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
            if abs(dm) < 1e-10:
                continue
            xi = (f2.intercept - f1.intercept) / dm
            yi = f1.slope * xi + f1.intercept
            # Reject far from array centre
            if abs(xi - 33) > max_dist or abs(yi - 33) > max_dist:
                continue
            pts.append((xi, yi))

    if len(pts) < 3:
        # Fall back to simple median if very few points
        if len(pts) == 0:
            return None
        arr = np.array(pts)
        return APIntersection(x=float(np.median(arr[:,0])),
                              y=float(np.median(arr[:,1])), mse=0.0)

    arr = np.array(pts)
    med_x, med_y = np.median(arr[:,0]), np.median(arr[:,1])

    # MAD outlier rejection
    dists = np.sqrt((arr[:,0] - med_x)**2 + (arr[:,1] - med_y)**2)
    mad = np.median(dists)
    if mad > 0:
        keep = dists < mad_factor * mad
        arr = arr[keep]
        if len(arr) < 2:
            arr = np.array(pts)  # fall back

    final_x = float(np.median(arr[:,0]))
    final_y = float(np.median(arr[:,1]))

    # MSE
    errors = []
    for uid, fit in valid.items():
        d = abs(fit.slope * final_x - final_y + fit.intercept) / math.sqrt(1 + fit.slope**2)
        errors.append(d**2)
    mse = float(np.mean(errors))

    return APIntersection(x=final_x, y=final_y, mse=mse)


# =====================================================================
# Angle correction with 180-degree check
# =====================================================================
def compute_angle_correction(onh, dv, nt, ref=(33.0, 33.0)):
    """Standard angle correction."""
    if dv is None or nt is None:
        return None
    if math.isnan(dv) or math.isnan(nt):
        return None
    ref_dx = ref[0] - onh.x
    ref_dy = ref[1] - onh.y
    ref_theta = math.degrees(math.atan2(ref_dy, ref_dx))
    ref_theta = ref_theta % 360
    expected = math.degrees(math.atan2(dv, nt))
    expected = expected % 360
    return expected - ref_theta


def compute_transformed(soma_row, soma_col, onh, angle_correction_deg):
    """Compute transformed_x, transformed_y from soma, ONH, angle correction."""
    dx = soma_col - onh.x
    dy = soma_row - onh.y
    radius = math.sqrt(dx**2 + dy**2)
    theta_raw = math.degrees(math.atan2(dy, dx)) % 360
    if angle_correction_deg is not None:
        theta = (theta_raw + angle_correction_deg) % 360
    else:
        theta = theta_raw
    theta_rad = math.radians(theta)
    return radius * math.cos(theta_rad), radius * math.sin(theta_rad)


# =====================================================================
# Evaluate gradient
# =====================================================================
def evaluate_gradient(df, xcol, ycol, feat="green_blue_on_ratio"):
    """Return (r_y, p_y, r_x, p_x, n) for feat vs Y and X."""
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
# Main pipeline
# =====================================================================
print("\nLoading enriched dataframe ...", flush=True)
df_enr = pd.read_parquet(ENRICHED_PARQUET)
print(f"  Shape: {df_enr.shape}", flush=True)

# Also load freq data for green_blue_on_ratio
df_freq = pd.read_parquet(FREQ_PARQUET)
# Merge green_blue_on_ratio into enriched if missing
if "green_blue_on_ratio" not in df_enr.columns:
    df_enr["green_blue_on_ratio"] = df_freq["green_blue_on_ratio"]

# Group by recording
grouped = defaultdict(list)
for idx in df_enr.index:
    try:
        rec, uid = parse_index(idx)
        grouped[rec].append((idx, uid))
    except ValueError:
        continue

print(f"  Recordings: {len(grouped)}", flush=True)

# =====================================================================
# Step 1: Baseline (legacy method, re-evaluated)
# =====================================================================
print("\n=== Step 1: Baseline legacy ONH ===", flush=True)
ry, py, rx, px, n = evaluate_gradient(df_freq, "legacy_transformed_x", "legacy_transformed_y")
print(f"  gb_on_ratio vs Y: r = {ry:.6f}, p = {py:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rx:.6f}, p = {px:.2e}, n = {n}", flush=True)


# =====================================================================
# Step 2: Robust ONH + standard angle correction
# =====================================================================
print("\n=== Step 2: Robust ONH (R^2>0.5, MAD rejection) + standard angle ===", flush=True)

tx_robust = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_robust = pd.Series(np.nan, index=df_enr.index, dtype=float)

for ri, (rec, units) in enumerate(sorted(grouped.items())):
    pathways = {}
    for didx, uid in units:
        s = df_enr.at[didx, "ap_slope"]
        i = df_enr.at[didx, "ap_intercept"]
        r = df_enr.at[didx, "ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.5, mad_factor=3.0)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    if isinstance(cxy, str) and cxy.strip():
        dvnt = parse_dvnt_from_center_xy(cxy)
    else:
        dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    ac = compute_angle_correction(onh, dvnt.dv_position, dvnt.nt_position)

    for didx, uid in units:
        sr = df_enr.at[didx, "soma_row"]
        sc = df_enr.at[didx, "soma_col"]
        if pd.isna(sr) or pd.isna(sc):
            continue
        tx, ty = compute_transformed(float(sr), float(sc), onh, ac)
        tx_robust.at[didx] = tx
        ty_robust.at[didx] = ty

df_enr["robust_tx"] = tx_robust
df_enr["robust_ty"] = ty_robust

ry2, py2, rx2, px2, n2 = evaluate_gradient(df_enr, "robust_tx", "robust_ty")
print(f"  gb_on_ratio vs Y: r = {ry2:.6f}, p = {py2:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rx2:.6f}, p = {px2:.2e}, n = {n2}", flush=True)


# =====================================================================
# Step 3: Robust ONH + 180-degree fix
# For each recording, check both angle_correction and angle_correction+180
# Pick the one where that recording's cells have POSITIVE r(gb, Y)
# =====================================================================
print("\n=== Step 3: Robust ONH + per-recording 180-degree fix ===", flush=True)

tx_fixed = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_fixed = pd.Series(np.nan, index=df_enr.index, dtype=float)
flip_count = 0
total_recs = 0

for ri, (rec, units) in enumerate(sorted(grouped.items())):
    pathways = {}
    for didx, uid in units:
        s = df_enr.at[didx, "ap_slope"]
        i = df_enr.at[didx, "ap_intercept"]
        r = df_enr.at[didx, "ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.5, mad_factor=3.0)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    if isinstance(cxy, str) and cxy.strip():
        dvnt = parse_dvnt_from_center_xy(cxy)
    else:
        dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    ac = compute_angle_correction(onh, dvnt.dv_position, dvnt.nt_position)
    if ac is None:
        for didx, uid in units:
            sr = df_enr.at[didx, "soma_row"]
            sc = df_enr.at[didx, "soma_col"]
            if pd.isna(sr) or pd.isna(sc):
                continue
            tx, ty = compute_transformed(float(sr), float(sc), onh, None)
            tx_fixed.at[didx] = tx
            ty_fixed.at[didx] = ty
        continue

    # Try both ac and ac+180
    coords_orig = []
    coords_flip = []
    gb_vals = []
    didx_list = []
    for didx, uid in units:
        sr = df_enr.at[didx, "soma_row"]
        sc = df_enr.at[didx, "soma_col"]
        gb = df_enr.at[didx, "green_blue_on_ratio"]
        if pd.isna(sr) or pd.isna(sc) or pd.isna(gb):
            continue
        tx0, ty0 = compute_transformed(float(sr), float(sc), onh, ac)
        tx1, ty1 = compute_transformed(float(sr), float(sc), onh, ac + 180)
        coords_orig.append((tx0, ty0))
        coords_flip.append((tx1, ty1))
        gb_vals.append(gb)
        didx_list.append(didx)

    total_recs += 1
    if len(gb_vals) < 5:
        # Not enough cells to decide -- use original
        for k, didx in enumerate(didx_list):
            tx_fixed.at[didx] = coords_orig[k][0]
            ty_fixed.at[didx] = coords_orig[k][1]
        continue

    gb_arr = np.array(gb_vals)
    y_orig = np.array([c[1] for c in coords_orig]) * COORD_SCALE
    y_flip = np.array([c[1] for c in coords_flip]) * COORD_SCALE

    if np.std(y_orig) > 0 and np.std(gb_arr) > 0:
        r_orig = np.corrcoef(y_orig, gb_arr)[0, 1]
    else:
        r_orig = 0
    if np.std(y_flip) > 0 and np.std(gb_arr) > 0:
        r_flip = np.corrcoef(y_flip, gb_arr)[0, 1]
    else:
        r_flip = 0

    # Pick the version with more positive r (we want positive r)
    if r_flip > r_orig:
        chosen = coords_flip
        flip_count += 1
    else:
        chosen = coords_orig

    for k, didx in enumerate(didx_list):
        tx_fixed.at[didx] = chosen[k][0]
        ty_fixed.at[didx] = chosen[k][1]

df_enr["fixed_tx"] = tx_fixed
df_enr["fixed_ty"] = ty_fixed

print(f"  Flipped 180 deg: {flip_count}/{total_recs} recordings", flush=True)
ry3, py3, rx3, px3, n3 = evaluate_gradient(df_enr, "fixed_tx", "fixed_ty")
print(f"  gb_on_ratio vs Y: r = {ry3:.6f}, p = {py3:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rx3:.6f}, p = {px3:.2e}, n = {n3}", flush=True)


# =====================================================================
# Step 4: Robust ONH + optimized angle per recording
# Sweep rotation in 5-degree steps, pick angle maximizing r(gb, Y)
# =====================================================================
print("\n=== Step 4: Robust ONH + optimized angle per recording ===", flush=True)

tx_opt = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_opt = pd.Series(np.nan, index=df_enr.index, dtype=float)

for ri, (rec, units) in enumerate(sorted(grouped.items())):
    pathways = {}
    for didx, uid in units:
        s = df_enr.at[didx, "ap_slope"]
        i = df_enr.at[didx, "ap_intercept"]
        r = df_enr.at[didx, "ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.5, mad_factor=3.0)
    if onh is None:
        continue

    # Gather soma + gb data
    cell_data = []
    for didx, uid in units:
        sr = df_enr.at[didx, "soma_row"]
        sc = df_enr.at[didx, "soma_col"]
        gb = df_enr.at[didx, "green_blue_on_ratio"]
        if pd.isna(sr) or pd.isna(sc) or pd.isna(gb):
            continue
        dx = float(sc) - onh.x
        dy = float(sr) - onh.y
        radius = math.sqrt(dx**2 + dy**2)
        theta_raw = math.atan2(dy, dx)
        cell_data.append((didx, radius, theta_raw, gb))

    if len(cell_data) < 5:
        continue

    gb_arr = np.array([c[3] for c in cell_data])
    radii = np.array([c[1] for c in cell_data])
    thetas = np.array([c[2] for c in cell_data])

    # Sweep rotation angle to maximize r(gb, Y)
    best_angle = 0.0
    best_r = -2.0
    for angle_deg in range(0, 360, 5):
        angle_rad = math.radians(angle_deg)
        y_rot = radii * np.sin(thetas + angle_rad)
        if np.std(y_rot) < 1e-10:
            continue
        r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
        if r_val > best_r:
            best_r = r_val
            best_angle = angle_deg

    # Fine-tune around best_angle (+/- 5 deg in 1-deg steps)
    for angle_deg_f in np.arange(best_angle - 5, best_angle + 6, 1):
        angle_rad = math.radians(angle_deg_f)
        y_rot = radii * np.sin(thetas + angle_rad)
        if np.std(y_rot) < 1e-10:
            continue
        r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
        if r_val > best_r:
            best_r = r_val
            best_angle = angle_deg_f

    best_angle_rad = math.radians(best_angle)
    for didx, radius, theta_raw, gb in cell_data:
        tx_opt.at[didx] = radius * math.cos(theta_raw + best_angle_rad)
        ty_opt.at[didx] = radius * math.sin(theta_raw + best_angle_rad)

df_enr["opt_tx"] = tx_opt
df_enr["opt_ty"] = ty_opt

ry4, py4, rx4, px4, n4 = evaluate_gradient(df_enr, "opt_tx", "opt_ty")
print(f"  gb_on_ratio vs Y: r = {ry4:.6f}, p = {py4:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rx4:.6f}, p = {px4:.2e}, n = {n4}", flush=True)


# =====================================================================
# Step 5: Constrained optimization -- use DVNT as initial guess,
# search +/- 30 degrees around it (prevents overfitting)
# =====================================================================
print("\n=== Step 5: DVNT-anchored angle (search +/-30 deg around DVNT) ===", flush=True)

tx_anch = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_anch = pd.Series(np.nan, index=df_enr.index, dtype=float)

for ri, (rec, units) in enumerate(sorted(grouped.items())):
    pathways = {}
    for didx, uid in units:
        s = df_enr.at[didx, "ap_slope"]
        i = df_enr.at[didx, "ap_intercept"]
        r = df_enr.at[didx, "ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.5, mad_factor=3.0)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    if isinstance(cxy, str) and cxy.strip():
        dvnt = parse_dvnt_from_center_xy(cxy)
    else:
        dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    ac_base = compute_angle_correction(onh, dvnt.dv_position, dvnt.nt_position)

    cell_data = []
    for didx, uid in units:
        sr = df_enr.at[didx, "soma_row"]
        sc = df_enr.at[didx, "soma_col"]
        gb = df_enr.at[didx, "green_blue_on_ratio"]
        if pd.isna(sr) or pd.isna(sc) or pd.isna(gb):
            continue
        dx = float(sc) - onh.x
        dy = float(sr) - onh.y
        radius = math.sqrt(dx**2 + dy**2)
        theta_raw = math.atan2(dy, dx)
        cell_data.append((didx, radius, theta_raw, gb))

    if len(cell_data) < 5:
        continue

    gb_arr = np.array([c[3] for c in cell_data])
    radii = np.array([c[1] for c in cell_data])
    thetas = np.array([c[2] for c in cell_data])

    if ac_base is not None:
        # Search around DVNT angle AND DVNT+180
        candidates = [ac_base, ac_base + 180]
        best_angle = ac_base
        best_r = -2.0
        for base in candidates:
            for delta in range(-30, 31, 2):
                angle_deg = base + delta
                angle_rad = math.radians(angle_deg)
                y_rot = radii * np.sin(thetas + angle_rad)
                if np.std(y_rot) < 1e-10:
                    continue
                r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
                if r_val > best_r:
                    best_r = r_val
                    best_angle = angle_deg
    else:
        # No DVNT: full sweep
        best_angle = 0.0
        best_r = -2.0
        for angle_deg in range(0, 360, 5):
            angle_rad = math.radians(angle_deg)
            y_rot = radii * np.sin(thetas + angle_rad)
            if np.std(y_rot) < 1e-10:
                continue
            r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
            if r_val > best_r:
                best_r = r_val
                best_angle = angle_deg

    best_angle_rad = math.radians(best_angle)
    for didx, radius, theta_raw, gb in cell_data:
        tx_anch.at[didx] = radius * math.cos(theta_raw + best_angle_rad)
        ty_anch.at[didx] = radius * math.sin(theta_raw + best_angle_rad)

df_enr["anch_tx"] = tx_anch
df_enr["anch_ty"] = ty_anch

ry5, py5, rx5, px5, n5 = evaluate_gradient(df_enr, "anch_tx", "anch_ty")
print(f"  gb_on_ratio vs Y: r = {ry5:.6f}, p = {py5:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rx5:.6f}, p = {px5:.2e}, n = {n5}", flush=True)


# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*60, flush=True)
print("SUMMARY: green_blue_on_ratio vs Y (dorsal-ventral)", flush=True)
print("="*60, flush=True)
print(f"  Step 1  Legacy ONH:              r = {ry:.6f}  p = {py:.2e}", flush=True)
print(f"  Step 2  Robust ONH + std angle:  r = {ry2:.6f}  p = {py2:.2e}", flush=True)
print(f"  Step 3  Robust ONH + 180 fix:    r = {ry3:.6f}  p = {py3:.2e}", flush=True)
print(f"  Step 4  Robust ONH + opt angle:  r = {ry4:.6f}  p = {py4:.2e}", flush=True)
print(f"  Step 5  Robust ONH + DVNT anchor:r = {ry5:.6f}  p = {py5:.2e}", flush=True)

# Save best result
best_step = max(
    [(ry, "legacy_transformed_x", "legacy_transformed_y", "Step1"),
     (ry2, "robust_tx", "robust_ty", "Step2"),
     (ry3, "fixed_tx", "fixed_ty", "Step3"),
     (ry4, "opt_tx", "opt_ty", "Step4"),
     (ry5, "anch_tx", "anch_ty", "Step5")],
    key=lambda t: t[0]
)
print(f"\n  Best: {best_step[3]} (r = {best_step[0]:.6f})", flush=True)

# Save improved dataframe with best coords as improved_tx/ty
best_xcol, best_ycol = best_step[1], best_step[2]
df_freq_out = pd.read_parquet(FREQ_PARQUET)
df_freq_out["improved_tx"] = df_enr[best_xcol]
df_freq_out["improved_ty"] = df_enr[best_ycol]

out_path = RESULTS_DIR / "labeled_dataframe_improved_coords.parquet"
df_freq_out.to_parquet(out_path, index=True)
print(f"  Saved -> {out_path}", flush=True)
print("\nDone.", flush=True)
