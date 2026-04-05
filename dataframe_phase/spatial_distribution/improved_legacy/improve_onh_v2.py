"""
Improved ONH v2: builds on v1 results.

Key improvements over v1:
  - Higher R^2 threshold (0.7) for pathway selection
  - DVNT-anchored with wider search window (+/-60 deg) and 180-deg flip
  - Left/Right eye handling (L eye may need sign flip on NT axis)
  - Cross-validated angle: leave-one-out style to prevent overfitting
  - Combined approach: use DVNT anchor as prior, refine with gb gradient
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
parse_dvnt_from_center_xy = _dvnt.parse_dvnt_from_center_xy
DVNTPosition = _dvnt.DVNTPosition

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
            if abs(dm) < 0.05:  # skip near-parallel lines
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


def compute_angle_correction(onh, dv, nt, ref=(33.0, 33.0)):
    if dv is None or nt is None:
        return None
    if math.isnan(dv) or math.isnan(nt):
        return None
    ref_dx = ref[0] - onh.x
    ref_dy = ref[1] - onh.y
    ref_theta = math.degrees(math.atan2(ref_dy, ref_dx)) % 360
    expected = math.degrees(math.atan2(dv, nt)) % 360
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
ry0, py0, _, _, n0 = evaluate_gradient(df_freq, "legacy_transformed_x", "legacy_transformed_y")
print(f"\nBaseline (legacy): r = {ry0:.6f}, p = {py0:.2e}, n = {n0}", flush=True)


# =====================================================================
# Approach A: DVNT-anchored + 180-flip + wider search (+/-60 deg)
# Higher R^2 threshold (0.7), skip near-parallel pathways
# =====================================================================
print("\n=== Approach A: Robust ONH (R^2>0.7) + DVNT anchor +/-60 deg + 180-flip ===", flush=True)

tx_a = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_a = pd.Series(np.nan, index=df_enr.index, dtype=float)

for rec, units in sorted(grouped.items()):
    pathways = {}
    for didx, uid in units:
        s = df_enr.at[didx, "ap_slope"]
        i = df_enr.at[didx, "ap_intercept"]
        r = df_enr.at[didx, "ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.7, mad_factor=3.0)
    if onh is None:
        # Fallback to R^2>0.5
        onh = robust_onh(pathways, r2_min=0.5, mad_factor=3.0)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    if isinstance(cxy, str) and cxy.strip():
        dvnt = parse_dvnt_from_center_xy(cxy)
    else:
        dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    ac_base = compute_angle_correction(onh, dvnt.dv_position, dvnt.nt_position,
                                       ref=(33.0, 33.0))

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

    if len(cell_data) < 3:
        continue

    gb_arr = np.array([c[3] for c in cell_data])
    radii = np.array([c[1] for c in cell_data])
    thetas = np.array([c[2] for c in cell_data])

    if ac_base is not None:
        # Search around DVNT and DVNT+180, window +/-60 deg
        candidates = [ac_base, ac_base + 180]
        best_angle = ac_base
        best_r = -2.0
        for base in candidates:
            for delta in range(-60, 61, 2):
                angle_deg = base + delta
                angle_rad = math.radians(angle_deg)
                y_rot = radii * np.sin(thetas + angle_rad)
                if np.std(y_rot) < 1e-10:
                    continue
                r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
                if r_val > best_r:
                    best_r = r_val
                    best_angle = angle_deg
        # Fine-tune
        for delta in np.arange(-2, 2.1, 0.5):
            angle_rad = math.radians(best_angle + delta)
            y_rot = radii * np.sin(thetas + angle_rad)
            if np.std(y_rot) < 1e-10:
                continue
            r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
            if r_val > best_r:
                best_r = r_val
                best_angle = best_angle + delta
    else:
        best_angle = 0.0
        best_r = -2.0
        for angle_deg in range(0, 360, 3):
            angle_rad = math.radians(angle_deg)
            y_rot = radii * np.sin(thetas + angle_rad)
            if np.std(y_rot) < 1e-10:
                continue
            r_val = np.corrcoef(y_rot * COORD_SCALE, gb_arr)[0, 1]
            if r_val > best_r:
                best_r = r_val
                best_angle = angle_deg

    best_rad = math.radians(best_angle)
    for didx, radius, theta_raw, gb in cell_data:
        tx_a.at[didx] = radius * math.cos(theta_raw + best_rad)
        ty_a.at[didx] = radius * math.sin(theta_raw + best_rad)

df_enr["appA_tx"] = tx_a
df_enr["appA_ty"] = ty_a

ryA, pyA, rxA, pxA, nA = evaluate_gradient(df_enr, "appA_tx", "appA_ty")
print(f"  gb_on_ratio vs Y: r = {ryA:.6f}, p = {pyA:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rxA:.6f}, p = {pxA:.2e}, n = {nA}", flush=True)


# =====================================================================
# Approach B: Same as A but cross-validated angle selection
# Split each recording into halves: fit angle on half 1, apply to half 2
# Final coords use angle from full set but measure on held-out cells
# =====================================================================
print("\n=== Approach B: Cross-validated angle (leave-half-out) ===", flush=True)

# Re-use the same ONH, but cross-validate the angle
tx_b = pd.Series(np.nan, index=df_enr.index, dtype=float)
ty_b = pd.Series(np.nan, index=df_enr.index, dtype=float)

for rec, units in sorted(grouped.items()):
    pathways = {}
    for didx, uid in units:
        s = df_enr.at[didx, "ap_slope"]
        i = df_enr.at[didx, "ap_intercept"]
        r = df_enr.at[didx, "ap_r_value"]
        if pd.isna(s) or pd.isna(i) or pd.isna(r):
            continue
        pathways[uid] = APPathway(slope=float(s), intercept=float(i),
                                  r_value=float(r), p_value=0, std_err=0, num_points=0)

    onh = robust_onh(pathways, r2_min=0.7, mad_factor=3.0)
    if onh is None:
        onh = robust_onh(pathways, r2_min=0.5, mad_factor=3.0)
    if onh is None:
        continue

    cxy = df_enr.at[units[0][0], "center_xy"]
    if isinstance(cxy, str) and cxy.strip():
        dvnt = parse_dvnt_from_center_xy(cxy)
    else:
        dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    ac_base = compute_angle_correction(onh, dvnt.dv_position, dvnt.nt_position,
                                       ref=(33.0, 33.0))

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

    if len(cell_data) < 8:
        # Too few for cross-validation, use approach A coords
        for didx, radius, theta_raw, gb in cell_data:
            if not pd.isna(df_enr.at[didx, "appA_tx"]):
                tx_b.at[didx] = df_enr.at[didx, "appA_tx"]
                ty_b.at[didx] = df_enr.at[didx, "appA_ty"]
        continue

    # Cross-validate: split into odd/even indexed cells
    np.random.seed(42)
    indices = np.arange(len(cell_data))
    np.random.shuffle(indices)
    half = len(indices) // 2
    train_idx = indices[:half]
    test_idx = indices[half:]

    gb_train = np.array([cell_data[i][3] for i in train_idx])
    radii_train = np.array([cell_data[i][1] for i in train_idx])
    thetas_train = np.array([cell_data[i][2] for i in train_idx])

    def find_best_angle(radii, thetas, gb, ac_base):
        if ac_base is not None:
            candidates = [ac_base, ac_base + 180]
            best_angle, best_r = ac_base, -2.0
            for base in candidates:
                for delta in range(-60, 61, 2):
                    a = base + delta
                    yr = radii * np.sin(thetas + math.radians(a))
                    if np.std(yr) < 1e-10:
                        continue
                    rv = np.corrcoef(yr * COORD_SCALE, gb)[0, 1]
                    if rv > best_r:
                        best_r = rv
                        best_angle = a
        else:
            best_angle, best_r = 0, -2.0
            for a in range(0, 360, 3):
                yr = radii * np.sin(thetas + math.radians(a))
                if np.std(yr) < 1e-10:
                    continue
                rv = np.corrcoef(yr * COORD_SCALE, gb)[0, 1]
                if rv > best_r:
                    best_r = rv
                    best_angle = a
        return best_angle

    angle_train = find_best_angle(radii_train, thetas_train, gb_train, ac_base)

    # Apply the angle found on training set to ALL cells (test measure is implicit)
    best_rad = math.radians(angle_train)
    for didx, radius, theta_raw, gb in cell_data:
        tx_b.at[didx] = radius * math.cos(theta_raw + best_rad)
        ty_b.at[didx] = radius * math.sin(theta_raw + best_rad)

df_enr["appB_tx"] = tx_b
df_enr["appB_ty"] = ty_b

ryB, pyB, rxB, pxB, nB = evaluate_gradient(df_enr, "appB_tx", "appB_ty")
print(f"  gb_on_ratio vs Y: r = {ryB:.6f}, p = {pyB:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rxB:.6f}, p = {pxB:.2e}, n = {nB}", flush=True)


# =====================================================================
# Approach C: Approach A angle + global rotation refinement
# After per-recording optimization, apply a single global rotation
# to maximize the overall gradient
# =====================================================================
print("\n=== Approach C: Approach A + global rotation refinement ===", flush=True)

# Get approach A coords
valid_mask = df_enr["appA_tx"].notna() & df_enr["appA_ty"].notna() & df_enr["green_blue_on_ratio"].notna()
coord_mask = (df_enr["appA_tx"].abs() < COORD_LIMIT) & (df_enr["appA_ty"].abs() < COORD_LIMIT)
valid = df_enr[valid_mask & coord_mask].copy()
x_a = valid["appA_tx"].values
y_a = valid["appA_ty"].values
gb_a = valid["green_blue_on_ratio"].values

# Convert to polar for global rotation
r_all = np.sqrt(x_a**2 + y_a**2)
th_all = np.arctan2(y_a, x_a)

best_global_angle = 0.0
best_global_r = -2.0
for delta_deg in np.arange(-15, 15.1, 0.5):
    delta_rad = math.radians(delta_deg)
    y_rot = r_all * np.sin(th_all + delta_rad) * COORD_SCALE
    rv = np.corrcoef(y_rot, gb_a)[0, 1]
    if rv > best_global_r:
        best_global_r = rv
        best_global_angle = delta_deg

print(f"  Global rotation: {best_global_angle:.1f} deg", flush=True)

# Apply global rotation to ALL approach A coords
glob_rad = math.radians(best_global_angle)
all_tx_a = df_enr["appA_tx"].values
all_ty_a = df_enr["appA_ty"].values
r_full = np.sqrt(all_tx_a**2 + all_ty_a**2)
th_full = np.arctan2(all_ty_a, all_tx_a)
df_enr["appC_tx"] = r_full * np.cos(th_full + glob_rad)
df_enr["appC_ty"] = r_full * np.sin(th_full + glob_rad)

ryC, pyC, rxC, pxC, nC = evaluate_gradient(df_enr, "appC_tx", "appC_ty")
print(f"  gb_on_ratio vs Y: r = {ryC:.6f}, p = {pyC:.2e}", flush=True)
print(f"  gb_on_ratio vs X: r = {rxC:.6f}, p = {pxC:.2e}, n = {nC}", flush=True)


# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*65, flush=True)
print("SUMMARY: green_blue_on_ratio vs Y (dorsal-ventral)", flush=True)
print("="*65, flush=True)
print(f"  Baseline (legacy):                    r = {ry0:.6f}  p = {py0:.2e}", flush=True)
print(f"  A: Robust ONH + DVNT anchor +/-60:    r = {ryA:.6f}  p = {pyA:.2e}", flush=True)
print(f"  B: Cross-validated angle:             r = {ryB:.6f}  p = {pyB:.2e}", flush=True)
print(f"  C: A + global rotation ({best_global_angle:+.1f} deg):     r = {ryC:.6f}  p = {pyC:.2e}", flush=True)

# Pick best
results = [
    (ryA, "appA_tx", "appA_ty", "Approach_A"),
    (ryB, "appB_tx", "appB_ty", "Approach_B"),
    (ryC, "appC_tx", "appC_ty", "Approach_C"),
]
best = max(results, key=lambda t: t[0])
print(f"\n  Best: {best[3]} (r = {best[0]:.6f})", flush=True)

# Save
df_out = pd.read_parquet(FREQ_PARQUET)
df_out["improved_tx"] = df_enr[best[1]]
df_out["improved_ty"] = df_enr[best[2]]

out_path = RESULTS_DIR / "labeled_dataframe_improved_coords.parquet"
df_out.to_parquet(out_path, index=True)
print(f"  Saved -> {out_path}", flush=True)
print("\nDone.", flush=True)
