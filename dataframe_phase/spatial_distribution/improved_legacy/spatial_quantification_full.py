"""
Comprehensive Spatial Quantification of 2-D Retinal Feature Topographies
=========================================================================
Uses saved hexbin data (raw_mean + gam_pred) to compute a full set of
complementary spatial statistics for each feature (all-cells & per-subtype):

  1. Global gradient  (plane fit, WLS weighted by bin count)
  2. GAM structure     (deviance explained, improvement over plane,
                        dynamic range, extremum location, hotspot area)
  3. Spatial autocorrelation (global & local Moran's I, Getis-Ord Gi*)
  4. Unevenness        (hexbin CV, Gini coefficient)
  5. Radial / angular  (radial correlation + bootstrap CI, quadrant ANOVA)
  6. Subtype consistency (circular dispersion, vector strength, % sig.)
  7. Significance       (permutation nulls for plane R^2 & Moran's I, FDR)
  8. Phase-feature handling (circular mean/variance, cos/sin decomposition)

Input
-----
  results/hexbin_data_all_cells.parquet
  results/hexbin_data_per_subtype.parquet

Output
------
  results/spatial_quant_all_cells.parquet
  results/spatial_quant_per_subtype.parquet
  results/spatial_quant_subtype_consistency.parquet
  results/spatial_quantification_full.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, f_oneway
from scipy.spatial import cKDTree
import warnings, time

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEX_ALL = RESULTS_DIR / "hexbin_data_all_cells.parquet"
HEX_SUB = RESULTS_DIR / "hexbin_data_per_subtype.parquet"

N_PERM = 999          # permutations for null tests
N_BOOT = 999          # bootstrap replicates for CIs
K_NEIGHBORS = 6       # for spatial weights
FDR_ALPHA = 0.05
HOTSPOT_QUANTILE = 0.90
RNG = np.random.default_rng(42)

PHASE_FEATURES = set()  # populated after loading data


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

# --- 1. Plane fit (WLS) ---
def weighted_plane_fit(x, y, z, w):
    """WLS: z = b0 + bx*x + by*y, weighted by w.
    Returns bx, by, b0, R^2."""
    n = len(z)
    if n < 4:
        return dict(bx=np.nan, by=np.nan, b0=np.nan, grad_mag=np.nan,
                    grad_dir_deg=np.nan, plane_r2=np.nan)
    W = np.diag(w)
    A = np.column_stack([np.ones(n), x, y])
    try:
        AtWA = A.T @ W @ A
        AtWz = A.T @ W @ z
        coeffs = np.linalg.solve(AtWA, AtWz)
    except np.linalg.LinAlgError:
        return dict(bx=np.nan, by=np.nan, b0=np.nan, grad_mag=np.nan,
                    grad_dir_deg=np.nan, plane_r2=np.nan)
    b0, bx, by = coeffs
    pred = A @ coeffs
    ss_res = np.sum(w * (z - pred)**2)
    z_mean = np.average(z, weights=w)
    ss_tot = np.sum(w * (z - z_mean)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mag = float(np.sqrt(bx**2 + by**2))
    theta = float(np.degrees(np.arctan2(by, bx)))
    return dict(bx=float(bx), by=float(by), b0=float(b0),
                grad_mag=mag, grad_dir_deg=theta, plane_r2=float(r2))


# --- 2. GAM structure ---
def gam_metrics(raw_mean, gam_pred, bin_x, bin_y, plane_r2):
    """Compute GAM-specific metrics from hexbin-level data."""
    mask = np.isfinite(gam_pred) & np.isfinite(raw_mean)
    out = dict(gam_r2=np.nan, gam_plane_delta_r2=np.nan,
               gam_dynamic_range=np.nan, gam_pct_range=np.nan,
               gam_max_x=np.nan, gam_max_y=np.nan,
               gam_min_x=np.nan, gam_min_y=np.nan,
               gam_max_quadrant="", gam_min_quadrant="",
               hotspot_area_frac=np.nan)
    if mask.sum() < 5:
        return out
    rm, gp = raw_mean[mask], gam_pred[mask]
    bx, by = bin_x[mask], bin_y[mask]

    # R^2 of GAM at hexbin level
    z_mean = np.mean(rm)
    ss_tot = np.sum((rm - z_mean)**2)
    ss_res = np.sum((rm - gp)**2)
    gam_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    out["gam_r2"] = float(gam_r2)
    out["gam_plane_delta_r2"] = float(gam_r2 - plane_r2) if np.isfinite(plane_r2) else np.nan

    # Dynamic range
    dr = float(np.max(gp) - np.min(gp))
    out["gam_dynamic_range"] = dr
    out["gam_pct_range"] = float(dr / abs(z_mean) * 100) if z_mean != 0 else np.nan

    # Extremum locations
    imax = np.argmax(gp)
    imin = np.argmin(gp)
    out["gam_max_x"] = float(bx[imax])
    out["gam_max_y"] = float(by[imax])
    out["gam_min_x"] = float(bx[imin])
    out["gam_min_y"] = float(by[imin])
    out["gam_max_quadrant"] = _quadrant(bx[imax], by[imax])
    out["gam_min_quadrant"] = _quadrant(bx[imin], by[imin])

    # Hotspot area (fraction above 90th pctl)
    q90 = np.quantile(gp, HOTSPOT_QUANTILE)
    out["hotspot_area_frac"] = float(np.mean(gp >= q90))
    return out


def _quadrant(x, y):
    if x >= 0 and y >= 0:
        return "DN"  # Dorsal-Nasal
    if x < 0 and y >= 0:
        return "DT"  # Dorsal-Temporal
    if x >= 0 and y < 0:
        return "VN"  # Ventral-Nasal
    return "VT"       # Ventral-Temporal


# --- 3. Spatial autocorrelation ---
def build_knn_weights(xy, k=K_NEIGHBORS):
    """Build binary KNN weight matrix. Returns (idx, n)."""
    n = len(xy)
    if n < k + 1:
        return None, n
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=min(k + 1, n))
    return idx, n


def moran_i_global(z, knn_idx, n):
    """Global Moran's I using pre-built KNN index."""
    if knn_idx is None or n < K_NEIGHBORS + 1:
        return np.nan
    z_dev = z - np.mean(z)
    denom = np.sum(z_dev**2)
    if denom == 0:
        return np.nan
    numer, W = 0.0, 0.0
    for i in range(n):
        for j_pos in range(1, knn_idx.shape[1]):
            j = knn_idx[i, j_pos]
            numer += z_dev[i] * z_dev[j]
            W += 1.0
    return float((n / W) * (numer / denom)) if W > 0 else np.nan


def local_moran(z, knn_idx, n):
    """Local Moran's I_i for each bin.  Returns array of I_i."""
    if knn_idx is None or n < K_NEIGHBORS + 1:
        return np.full(n, np.nan)
    z_dev = z - np.mean(z)
    m2 = np.mean(z_dev**2)
    if m2 == 0:
        return np.full(n, np.nan)
    I_i = np.zeros(n)
    for i in range(n):
        lag = 0.0
        for j_pos in range(1, knn_idx.shape[1]):
            lag += z_dev[knn_idx[i, j_pos]]
        I_i[i] = (z_dev[i] / m2) * lag
    return I_i


def getis_ord_gi_star(z, knn_idx, n):
    """Getis-Ord Gi* z-scores for each bin."""
    if knn_idx is None or n < K_NEIGHBORS + 1:
        return np.full(n, np.nan)
    x_bar = np.mean(z)
    S = np.std(z)
    if S == 0:
        return np.full(n, np.nan)
    k = knn_idx.shape[1] - 1
    gi = np.zeros(n)
    for i in range(n):
        w_sum_x = z[i]  # self-weight = 1 for Gi*
        w_cnt = 1.0
        for j_pos in range(1, knn_idx.shape[1]):
            w_sum_x += z[knn_idx[i, j_pos]]
            w_cnt += 1.0
        numer = w_sum_x - x_bar * w_cnt
        denom_val = S * np.sqrt((n * w_cnt - w_cnt**2) / (n - 1))
        gi[i] = numer / denom_val if denom_val > 0 else 0.0
    return gi


def hotspot_summary(local_i, gi_star, z):
    """Summarize local hotspot statistics."""
    out = {}
    if np.all(np.isnan(local_i)):
        out["n_hot_local_moran"] = 0
        out["n_cold_local_moran"] = 0
        out["n_hot_gi_star"] = 0
        out["n_cold_gi_star"] = 0
        return out
    # Local Moran: significant positive I_i with z_i > mean = hot cluster
    z_dev = z - np.mean(z)
    # Use |I_i| > 1.96 (approx) as significance proxy
    sig_pos = (local_i > 0) & (np.abs(local_i) > 1.96)
    out["n_hot_local_moran"] = int(np.sum(sig_pos & (z_dev > 0)))
    out["n_cold_local_moran"] = int(np.sum(sig_pos & (z_dev < 0)))
    # Gi*: |z-score| > 1.96 => significant
    out["n_hot_gi_star"] = int(np.sum(gi_star > 1.96))
    out["n_cold_gi_star"] = int(np.sum(gi_star < -1.96))
    return out


# --- 4. Unevenness ---
def gini_coefficient(x):
    """Gini coefficient of array x."""
    x = np.sort(x)
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


# --- 5. Radial / angular ---
def radial_angular_analysis(bx, by, z, w, n_boot=N_BOOT):
    """Radial correlation + bootstrap CI, quadrant means + F-stat."""
    out = {}
    r = np.sqrt(bx**2 + by**2)
    if np.std(r) < 1e-12 or np.std(z) < 1e-12 or len(z) < 10:
        out.update(dict(radial_r=np.nan, radial_p=np.nan,
                        radial_slope=np.nan, radial_r_ci_lo=np.nan,
                        radial_r_ci_hi=np.nan))
        for q in ["DN", "DT", "VN", "VT"]:
            out[f"quad_mean_{q}"] = np.nan
        out["quad_F"] = np.nan
        out["quad_p"] = np.nan
        return out

    rho, pval = pearsonr(r, z)
    # slope
    A = np.column_stack([r, np.ones_like(r)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    out["radial_r"] = float(rho)
    out["radial_p"] = float(pval)
    out["radial_slope"] = float(coeffs[0])

    # Bootstrap CI for radial_r
    boot_r = np.empty(n_boot)
    n = len(z)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if np.std(r[idx]) < 1e-12 or np.std(z[idx]) < 1e-12:
            boot_r[b] = np.nan
        else:
            boot_r[b], _ = pearsonr(r[idx], z[idx])
    boot_r = boot_r[np.isfinite(boot_r)]
    if len(boot_r) > 10:
        out["radial_r_ci_lo"] = float(np.percentile(boot_r, 2.5))
        out["radial_r_ci_hi"] = float(np.percentile(boot_r, 97.5))
    else:
        out["radial_r_ci_lo"] = np.nan
        out["radial_r_ci_hi"] = np.nan

    # Quadrant means
    quads = {"DN": (bx >= 0) & (by >= 0),
             "DT": (bx < 0) & (by >= 0),
             "VN": (bx >= 0) & (by < 0),
             "VT": (bx < 0) & (by < 0)}
    groups = []
    for q, mask in quads.items():
        vals = z[mask]
        out[f"quad_mean_{q}"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        if len(vals) > 0:
            groups.append(vals)

    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        try:
            F, p = f_oneway(*groups)
            out["quad_F"] = float(F)
            out["quad_p"] = float(p)
        except Exception:
            out["quad_F"] = np.nan
            out["quad_p"] = np.nan
    else:
        out["quad_F"] = np.nan
        out["quad_p"] = np.nan

    return out


# --- 7. Permutation tests ---
def permutation_pvalue(z, bx, by, w, knn_idx, n, n_perm=N_PERM):
    """Permutation null for plane R^2 and Moran's I."""
    obs_plane = _plane_fit_simple(bx, by, z, w)
    obs_r2 = obs_plane["plane_r2"]
    obs_moran = moran_i_global(z, knn_idx, n)

    count_r2 = 0
    count_moran = 0
    for _ in range(n_perm):
        z_perm = RNG.permutation(z)
        pf = _plane_fit_simple(bx, by, z_perm, w)
        if np.isfinite(pf["plane_r2"]) and np.isfinite(obs_r2):
            if pf["plane_r2"] >= obs_r2:
                count_r2 += 1
        mi = moran_i_global(z_perm, knn_idx, n)
        if np.isfinite(mi) and np.isfinite(obs_moran):
            if mi >= obs_moran:
                count_moran += 1

    perm_p_r2 = (count_r2 + 1) / (n_perm + 1)
    perm_p_moran = (count_moran + 1) / (n_perm + 1)
    return float(perm_p_r2), float(perm_p_moran)


# --- 8. Phase (circular) features ---
def is_phase_feature(feat):
    return "phase_deg" in feat


def circular_stats(z_deg):
    """Circular mean, variance, resultant length for degree data."""
    theta = np.radians(z_deg)
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C**2 + S**2)
    circ_mean_deg = float(np.degrees(np.arctan2(S, C)))
    circ_var = float(1 - R)
    return dict(circ_mean_deg=circ_mean_deg, circ_var=circ_var,
                circ_resultant_length=float(R))


def decompose_phase(z_deg):
    """Return (cos, sin) components for spatial modeling."""
    theta = np.radians(z_deg)
    return np.cos(theta), np.sin(theta)


# --- FDR ---
def fdr_correct(pvalues):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.array(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    valid = np.isfinite(pvals)
    adj = np.full(n, np.nan)
    if valid.sum() == 0:
        return adj
    idx_valid = np.where(valid)[0]
    p_valid = pvals[idx_valid]
    order = np.argsort(p_valid)
    ranked = np.empty_like(p_valid)
    ranked[order] = np.arange(1, len(p_valid) + 1)
    adj_valid = p_valid * len(p_valid) / ranked
    # Enforce monotonicity
    adj_valid[order] = np.minimum.accumulate(adj_valid[order][::-1])[::-1]
    adj_valid = np.clip(adj_valid, 0, 1)
    adj[idx_valid] = adj_valid
    return adj


# =====================================================================
# SINGLE-FEATURE ANALYSIS
# =====================================================================
def analyse_feature(bx, by, raw_mean, gam_pred, counts, feature_name):
    """Full analysis for one feature scope. Returns dict of metrics."""
    m = {"feature": feature_name}
    n = len(raw_mean)
    m["n_bins"] = n

    if n < 5:
        return m

    z = raw_mean
    w = counts.astype(float)
    w_norm = w / w.sum()  # normalize for WLS

    # --- 1. Plane fit (WLS) ---
    pf = _plane_fit_simple(bx, by, z, w)
    m.update(pf)

    # --- Phase handling ---
    if is_phase_feature(feature_name):
        cs = circular_stats(z)
        m.update(cs)
        # Also fit plane on cos/sin components
        cos_z, sin_z = decompose_phase(z)
        pf_cos = _plane_fit_simple(bx, by, cos_z, w)
        pf_sin = _plane_fit_simple(bx, by, sin_z, w)
        m["phase_cos_plane_r2"] = pf_cos["plane_r2"]
        m["phase_sin_plane_r2"] = pf_sin["plane_r2"]
        m["phase_cos_grad_mag"] = pf_cos["grad_mag"]
        m["phase_sin_grad_mag"] = pf_sin["grad_mag"]

    # --- 2. GAM structure ---
    gm = gam_metrics(z, gam_pred, bx, by, pf["plane_r2"])
    m.update(gm)

    # --- 3. Spatial autocorrelation ---
    xy = np.column_stack([bx, by])
    knn_idx, nn = build_knn_weights(xy)
    m["moran_i"] = moran_i_global(z, knn_idx, nn)

    li = local_moran(z, knn_idx, nn)
    gi = getis_ord_gi_star(z, knn_idx, nn)
    hs = hotspot_summary(li, gi, z)
    m.update(hs)

    # --- 4. Unevenness ---
    z_mean = np.mean(z)
    m["hexbin_cv"] = float(np.std(z) / abs(z_mean)) if z_mean != 0 else np.nan
    m["gini"] = gini_coefficient(z)

    # --- 5. Radial / angular ---
    ra = radial_angular_analysis(bx, by, z, w)
    m.update(ra)

    # --- 7. Permutation tests ---
    perm_p_r2, perm_p_moran = permutation_pvalue(z, bx, by, w, knn_idx, nn)
    m["perm_p_plane_r2"] = perm_p_r2
    m["perm_p_moran_i"] = perm_p_moran

    return m


def _plane_fit_simple(x, y, z, w):
    """Simple WLS plane fit without building full diagonal matrix."""
    n = len(z)
    if n < 4:
        return dict(bx=np.nan, by=np.nan, b0=np.nan, grad_mag=np.nan,
                    grad_dir_deg=np.nan, plane_r2=np.nan)
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(n), x, y]) * sw[:, None]
    b = z * sw
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return dict(bx=np.nan, by=np.nan, b0=np.nan, grad_mag=np.nan,
                    grad_dir_deg=np.nan, plane_r2=np.nan)
    b0, bx, by = coeffs
    pred = np.column_stack([np.ones(n), x, y]) @ coeffs
    ss_res = np.sum(w * (z - pred)**2)
    z_mean = np.average(z, weights=w)
    ss_tot = np.sum(w * (z - z_mean)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mag = float(np.sqrt(bx**2 + by**2))
    theta = float(np.degrees(np.arctan2(by, bx)))
    return dict(bx=float(bx), by=float(by), b0=float(b0),
                grad_mag=mag, grad_dir_deg=theta, plane_r2=float(r2))


# =====================================================================
# MAIN
# =====================================================================
print("Loading hexbin data ...", flush=True)
df_all = pd.read_parquet(HEX_ALL)
df_sub = pd.read_parquet(HEX_SUB)
features = sorted(df_all["feature"].unique())
subtypes = sorted(df_sub["scope"].unique())
print(f"  All-cells: {len(features)} features, {len(df_all)} bins", flush=True)
print(f"  Per-subtype: {len(subtypes)} subtypes, {len(df_sub)} bins", flush=True)

# Identify phase features
PHASE_FEATURES = {f for f in features if is_phase_feature(f)}
print(f"  Phase features ({len(PHASE_FEATURES)}): {sorted(PHASE_FEATURES)}", flush=True)


# =====================================================================
# PHASE A: All-cells
# =====================================================================
print("\n=== Phase A: All-cells analysis ===", flush=True)
rows_all = []
t0 = time.time()

for fi, feat in enumerate(features):
    sub = df_all[df_all["feature"] == feat]
    bx = sub["bin_x"].to_numpy()
    by = sub["bin_y"].to_numpy()
    rm = sub["raw_mean"].to_numpy()
    gp = sub["gam_pred"].to_numpy()
    cnt = sub["count"].to_numpy().astype(float)

    m = analyse_feature(bx, by, rm, gp, cnt, feat)
    m["scope"] = "all_cells"
    rows_all.append(m)

    if (fi + 1) % 10 == 0 or (fi + 1) == len(features):
        print(f"  [{fi+1}/{len(features)}] {feat}  ({time.time()-t0:.0f}s)", flush=True)

df_quant_all = pd.DataFrame(rows_all)
print(f"  All-cells rows: {len(df_quant_all)}", flush=True)


# =====================================================================
# PHASE B: Per-subtype (no GAM, lighter permutation)
# =====================================================================
print("\n=== Phase B: Per-subtype analysis ===", flush=True)
rows_sub = []
t0 = time.time()

for fi, feat in enumerate(features):
    feat_data = df_sub[df_sub["feature"] == feat]
    for stype in subtypes:
        sdata = feat_data[feat_data["scope"] == stype]
        if len(sdata) < 5:
            continue
        bx = sdata["bin_x"].to_numpy()
        by = sdata["bin_y"].to_numpy()
        rm = sdata["raw_mean"].to_numpy()
        gp = np.full_like(rm, np.nan)  # no GAM for per-subtype
        cnt = sdata["count"].to_numpy().astype(float)

        m = analyse_feature(bx, by, rm, gp, cnt, feat)
        m["scope"] = stype
        rows_sub.append(m)

    if (fi + 1) % 10 == 0 or (fi + 1) == len(features):
        print(f"  [{fi+1}/{len(features)}] {feat}  ({time.time()-t0:.0f}s)", flush=True)

df_quant_sub = pd.DataFrame(rows_sub)
print(f"  Per-subtype rows: {len(df_quant_sub)}", flush=True)


# =====================================================================
# PHASE C: Subtype consistency (gradient direction across subtypes)
# =====================================================================
print("\n=== Phase C: Subtype consistency ===", flush=True)
rows_cons = []

for feat in features:
    feat_sub = df_quant_sub[df_quant_sub["feature"] == feat].copy()
    if len(feat_sub) < 3:
        continue

    # Filter to subtypes with significant plane (perm_p < 0.05)
    sig_sub = feat_sub[feat_sub["perm_p_plane_r2"] < 0.05]
    all_dirs = feat_sub["grad_dir_deg"].dropna().to_numpy()
    sig_dirs = sig_sub["grad_dir_deg"].dropna().to_numpy()
    all_mags = feat_sub["grad_mag"].dropna().to_numpy()
    sig_mags = sig_sub["grad_mag"].dropna().to_numpy()

    c = {"feature": feat}
    c["n_subtypes_total"] = len(feat_sub)
    c["n_subtypes_sig"] = len(sig_sub)
    c["pct_subtypes_sig"] = float(len(sig_sub) / len(feat_sub) * 100)

    # Circular mean & SD of gradient direction (all subtypes)
    if len(all_dirs) >= 2:
        theta = np.radians(all_dirs)
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        R = np.sqrt(C**2 + S**2)
        c["mean_dir_deg"] = float(np.degrees(np.arctan2(S, C)))
        c["circ_sd_deg"] = float(np.degrees(np.sqrt(-2 * np.log(R)))) if R > 0 else 180.0
        c["circ_resultant"] = float(R)
    else:
        c["mean_dir_deg"] = np.nan
        c["circ_sd_deg"] = np.nan
        c["circ_resultant"] = np.nan

    # Vector strength (magnitude-weighted)
    if len(all_mags) >= 2 and np.sum(all_mags) > 0:
        theta = np.radians(all_dirs[:len(all_mags)])
        wx = all_mags * np.cos(theta)
        wy = all_mags * np.sin(theta)
        c["vector_strength"] = float(np.sqrt(np.sum(wx)**2 + np.sum(wy)**2) / np.sum(all_mags))
    else:
        c["vector_strength"] = np.nan

    # Same for significant subtypes only
    if len(sig_dirs) >= 2:
        theta = np.radians(sig_dirs)
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        R = np.sqrt(C**2 + S**2)
        c["sig_mean_dir_deg"] = float(np.degrees(np.arctan2(S, C)))
        c["sig_circ_sd_deg"] = float(np.degrees(np.sqrt(-2 * np.log(R)))) if R > 0 else 180.0
        c["sig_vector_strength"] = float(R)
    else:
        c["sig_mean_dir_deg"] = np.nan
        c["sig_circ_sd_deg"] = np.nan
        c["sig_vector_strength"] = np.nan

    rows_cons.append(c)

df_consistency = pd.DataFrame(rows_cons)
print(f"  Consistency rows: {len(df_consistency)}", flush=True)


# =====================================================================
# PHASE D: FDR correction
# =====================================================================
print("\n=== Phase D: FDR correction ===", flush=True)
for df_q, label in [(df_quant_all, "all_cells"), (df_quant_sub, "per_subtype")]:
    for col in ["perm_p_plane_r2", "perm_p_moran_i"]:
        if col in df_q.columns:
            adj = fdr_correct(df_q[col].to_numpy())
            df_q[col.replace("perm_p_", "fdr_q_")] = adj
    if "radial_p" in df_q.columns:
        adj = fdr_correct(df_q["radial_p"].to_numpy())
        df_q["fdr_q_radial"] = adj
    if "quad_p" in df_q.columns:
        adj = fdr_correct(df_q["quad_p"].to_numpy())
        df_q["fdr_q_quad"] = adj
    print(f"  FDR applied to {label}", flush=True)


# =====================================================================
# SAVE
# =====================================================================
print("\n=== Saving results ===", flush=True)
out1 = RESULTS_DIR / "spatial_quant_all_cells.parquet"
out2 = RESULTS_DIR / "spatial_quant_per_subtype.parquet"
out3 = RESULTS_DIR / "spatial_quant_subtype_consistency.parquet"

df_quant_all.to_parquet(out1, index=False)
df_quant_sub.to_parquet(out2, index=False)
df_consistency.to_parquet(out3, index=False)
print(f"  {out1.name}: {df_quant_all.shape}", flush=True)
print(f"  {out2.name}: {df_quant_sub.shape}", flush=True)
print(f"  {out3.name}: {df_consistency.shape}", flush=True)


# =====================================================================
# MD SUMMARY
# =====================================================================
print("\n=== Writing MD summary ===", flush=True)

ac = df_quant_all.set_index("feature")
L = []

L.append("# Comprehensive Spatial Quantification of Retinal Feature Topographies\n")
L.append("## Data & Coordinate System\n")
L.append(f"- **Cells analysed**: {int(ac['n_bins'].iloc[0])} hexbins (gridsize=40, all cells)")
L.append(f"- **X axis**: Temporal (neg) --> Nasal (pos)")
L.append(f"- **Y axis**: Ventral (neg) --> Dorsal (pos)")
L.append(f"- **Features**: {len(features)} ({len(PHASE_FEATURES)} circular/phase)")
L.append(f"- **Subtypes**: {len(subtypes)}")
L.append(f"- **Permutation tests**: {N_PERM} permutations, FDR at q < {FDR_ALPHA}")
L.append(f"- **Bootstrap CIs**: {N_BOOT} resamples (radial correlation)")
L.append("")

# ---- 1. GLOBAL GRADIENT ----
L.append("---\n## 1. Global Gradient (Weighted Plane Fit)\n")
L.append("$z_i = \\beta_0 + \\beta_x x_i + \\beta_y y_i$, weighted by bin count.\n")
sig_plane = ac[ac["fdr_q_plane_r2"] < FDR_ALPHA]
L.append(f"**{len(sig_plane)}/{len(ac)} features** have significant plane gradient "
         f"(permutation FDR q < {FDR_ALPHA}).\n")
L.append("### Top 15 by gradient magnitude\n")
L.append("| Feature | beta_x | beta_y | Mag | Dir (deg) | R^2 | perm_p | FDR q |")
L.append("|---------|--------|--------|-----|-----------|-----|--------|-------|")
for feat in ac.nlargest(15, "grad_mag").index:
    r = ac.loc[feat]
    L.append(f"| {feat} | {r['bx']:.6f} | {r['by']:.6f} | {r['grad_mag']:.6f} "
             f"| {r['grad_dir_deg']:.1f} | {r['plane_r2']:.4f} "
             f"| {r['perm_p_plane_r2']:.4f} | {r.get('fdr_q_plane_r2', np.nan):.4f} |")
L.append("")

# Direction interpretation
L.append("### Gradient direction interpretation\n")
L.append("| Angle range | Direction |")
L.append("|-------------|-----------|")
L.append("| -22 to +22 | Nasal |")
L.append("| +22 to +68 | Dorsal-Nasal |")
L.append("| +68 to +112 | Dorsal |")
L.append("| +112 to +158 | Dorsal-Temporal |")
L.append("| +/-158 to +/-180 | Temporal |")
L.append("| -158 to -112 | Ventral-Temporal |")
L.append("| -112 to -68 | Ventral |")
L.append("| -68 to -22 | Ventral-Nasal |")
L.append("")

# ---- 2. GAM STRUCTURE ----
L.append("---\n## 2. Nonlinear Surface (GAM Structure)\n")
L.append("$z_i = \\beta_0 + f(x_i, y_i)$, tensor-product spline.\n")
L.append("### Top 15 by GAM improvement over plane\n")
col_delta = "gam_plane_delta_r2"
top_gam = ac.nlargest(15, col_delta)
L.append("| Feature | Plane R^2 | GAM R^2 | Delta R^2 | Dyn Range | % Range "
         "| Max loc | Max Q | Min loc | Min Q | A_0.9 |")
L.append("|---------|----------|---------|----------|-----------|--------"
         "|---------|---------|---------|---------|----|")
for feat in top_gam.index:
    r = ac.loc[feat]
    L.append(
        f"| {feat} | {r['plane_r2']:.4f} | {r['gam_r2']:.4f} "
        f"| {r[col_delta]:.4f} | {r['gam_dynamic_range']:.4f} "
        f"| {r['gam_pct_range']:.1f}% "
        f"| ({r['gam_max_x']:.0f},{r['gam_max_y']:.0f}) | {r['gam_max_quadrant']} "
        f"| ({r['gam_min_x']:.0f},{r['gam_min_y']:.0f}) | {r['gam_min_quadrant']} "
        f"| {r['hotspot_area_frac']:.3f} |"
    )
L.append("")

# ---- 3. SPATIAL AUTOCORRELATION ----
L.append("---\n## 3. Spatial Autocorrelation\n")
sig_moran = ac[ac["fdr_q_moran_i"] < FDR_ALPHA]
L.append(f"**{len(sig_moran)}/{len(ac)} features** have significant Moran's I "
         f"(permutation FDR q < {FDR_ALPHA}).\n")
L.append("### Top 15 by Moran's I\n")
L.append("| Feature | Moran's I | perm_p | FDR q | Hot bins (Gi*) | Cold bins (Gi*) "
         "| Hot (local I) | Cold (local I) |")
L.append("|---------|----------|--------|-------|----------------|---------------"
         "|---------------|----------------|")
for feat in ac.nlargest(15, "moran_i").index:
    r = ac.loc[feat]
    L.append(
        f"| {feat} | {r['moran_i']:.4f} | {r['perm_p_moran_i']:.4f} "
        f"| {r.get('fdr_q_moran_i', np.nan):.4f} "
        f"| {r.get('n_hot_gi_star', 0):.0f} | {r.get('n_cold_gi_star', 0):.0f} "
        f"| {r.get('n_hot_local_moran', 0):.0f} | {r.get('n_cold_local_moran', 0):.0f} |"
    )
L.append("")

# ---- 4. UNEVENNESS ----
L.append("---\n## 4. Unevenness\n")
L.append("### Top 15 by Hexbin CV\n")
L.append("| Feature | Hexbin CV | Gini |")
L.append("|---------|----------|------|")
for feat in ac.nlargest(15, "hexbin_cv").index:
    r = ac.loc[feat]
    L.append(f"| {feat} | {r['hexbin_cv']:.4f} | {r['gini']:.4f} |")
L.append("")

# ---- 5. RADIAL / ANGULAR ----
L.append("---\n## 5. Radial and Angular Structure\n")
L.append("Radial: $z \\sim \\alpha_0 + \\alpha_r r$ from origin (0,0).\n")
sig_rad = ac[ac["fdr_q_radial"] < FDR_ALPHA]
L.append(f"**{len(sig_rad)}/{len(ac)} features** have significant radial trend "
         f"(FDR q < {FDR_ALPHA}).\n")
L.append("### Top 15 by |radial r|\n")
L.append("| Feature | Radial r | 95% CI | Slope | p | FDR q |")
L.append("|---------|---------|--------|-------|---|-------|")
top_rad = ac.reindex(ac["radial_r"].abs().nlargest(15).index)
for feat in top_rad.index:
    r = ac.loc[feat]
    ci_lo = r.get("radial_r_ci_lo", np.nan)
    ci_hi = r.get("radial_r_ci_hi", np.nan)
    ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else "N/A"
    L.append(
        f"| {feat} | {r['radial_r']:.4f} | {ci_str} | {r['radial_slope']:.6f} "
        f"| {r['radial_p']:.2e} | {r.get('fdr_q_radial', np.nan):.4f} |"
    )
L.append("")

L.append("### Quadrant analysis (top 15 by F-statistic)\n")
L.append("| Feature | DN | DT | VN | VT | F | p | FDR q |")
L.append("|---------|-----|-----|-----|-----|---|---|-------|")
top_quad = ac.nlargest(15, "quad_F")
for feat in top_quad.index:
    r = ac.loc[feat]
    L.append(
        f"| {feat} "
        f"| {r.get('quad_mean_DN', np.nan):.4f} | {r.get('quad_mean_DT', np.nan):.4f} "
        f"| {r.get('quad_mean_VN', np.nan):.4f} | {r.get('quad_mean_VT', np.nan):.4f} "
        f"| {r['quad_F']:.2f} | {r['quad_p']:.2e} "
        f"| {r.get('fdr_q_quad', np.nan):.4f} |"
    )
L.append("")

# ---- 6. SUBTYPE CONSISTENCY ----
L.append("---\n## 6. Subtype Consistency\n")
if len(df_consistency) > 0:
    cons = df_consistency.set_index("feature")
    L.append("### Features with highest directional consistency across subtypes\n")
    L.append("| Feature | n_sig/total | % sig | Mean dir | Circ SD | Vec Strength | Sig Vec Str |")
    L.append("|---------|------------|-------|----------|---------|-------------|-------------|")
    top_vs = cons.nlargest(20, "vector_strength")
    for feat in top_vs.index:
        r = cons.loc[feat]
        L.append(
            f"| {feat} | {r['n_subtypes_sig']:.0f}/{r['n_subtypes_total']:.0f} "
            f"| {r['pct_subtypes_sig']:.0f}% "
            f"| {r['mean_dir_deg']:.1f} | {r['circ_sd_deg']:.1f} "
            f"| {r['vector_strength']:.4f} | {r.get('sig_vector_strength', np.nan):.4f} |"
        )
    L.append("")

    L.append("### Features with lowest consistency (most scattered directions)\n")
    L.append("| Feature | n_sig/total | Circ SD | Vec Strength |")
    L.append("|---------|------------|---------|-------------|")
    bottom_vs = cons[cons["n_subtypes_sig"] >= 3].nsmallest(10, "vector_strength")
    for feat in bottom_vs.index:
        r = cons.loc[feat]
        L.append(
            f"| {feat} | {r['n_subtypes_sig']:.0f}/{r['n_subtypes_total']:.0f} "
            f"| {r['circ_sd_deg']:.1f} | {r['vector_strength']:.4f} |"
        )
    L.append("")

# ---- 7. PERMUTATION SUMMARY ----
L.append("---\n## 7. Significance Summary\n")
n_sig_plane = (ac["fdr_q_plane_r2"] < FDR_ALPHA).sum()
n_sig_moran = (ac["fdr_q_moran_i"] < FDR_ALPHA).sum()
n_sig_radial = (ac["fdr_q_radial"] < FDR_ALPHA).sum() if "fdr_q_radial" in ac.columns else 0
n_sig_quad = (ac["fdr_q_quad"] < FDR_ALPHA).sum() if "fdr_q_quad" in ac.columns else 0

L.append("| Test | # sig (FDR q<0.05) | Total | % |")
L.append("|------|-------------------|-------|---|")
L.append(f"| Plane gradient (perm) | {n_sig_plane} | {len(ac)} | {n_sig_plane/len(ac)*100:.0f}% |")
L.append(f"| Moran's I (perm) | {n_sig_moran} | {len(ac)} | {n_sig_moran/len(ac)*100:.0f}% |")
L.append(f"| Radial correlation | {n_sig_radial} | {len(ac)} | {n_sig_radial/len(ac)*100:.0f}% |")
L.append(f"| Quadrant ANOVA | {n_sig_quad} | {len(ac)} | {n_sig_quad/len(ac)*100:.0f}% |")
L.append("")

# ---- 8. PHASE FEATURES ----
phase_rows = ac.loc[ac.index.isin(PHASE_FEATURES)]
if len(phase_rows) > 0:
    L.append("---\n## 8. Phase Features (Circular Statistics)\n")
    L.append("Phase features use circular mean/variance and cos/sin decomposition.\n")
    L.append("| Feature | Circ Mean (deg) | Circ Var | Resultant | cos R^2 | sin R^2 "
             "| Moran I | perm_p |")
    L.append("|---------|----------------|---------|-----------|---------|--------"
             "|---------|--------|")
    for feat in sorted(phase_rows.index):
        r = phase_rows.loc[feat]
        L.append(
            f"| {feat} | {r.get('circ_mean_deg', np.nan):.1f} "
            f"| {r.get('circ_var', np.nan):.4f} "
            f"| {r.get('circ_resultant_length', np.nan):.4f} "
            f"| {r.get('phase_cos_plane_r2', np.nan):.4f} "
            f"| {r.get('phase_sin_plane_r2', np.nan):.4f} "
            f"| {r['moran_i']:.4f} | {r['perm_p_moran_i']:.4f} |"
        )
    L.append("")

# ---- FULL TABLE ----
L.append("---\n## Full Feature Table (All Cells)\n")
cols_report = ["plane_r2", "perm_p_plane_r2", "fdr_q_plane_r2", "grad_mag", "grad_dir_deg",
               "gam_r2", "gam_plane_delta_r2", "gam_dynamic_range", "gam_pct_range",
               "moran_i", "perm_p_moran_i", "fdr_q_moran_i",
               "hexbin_cv", "gini",
               "radial_r", "radial_r_ci_lo", "radial_r_ci_hi", "radial_p",
               "quad_F", "quad_p",
               "n_hot_gi_star", "n_cold_gi_star"]
hdr = "| Feature | " + " | ".join(cols_report) + " |"
sep = "|" + "|".join(["---"] * (len(cols_report) + 1)) + "|"
L.append(hdr)
L.append(sep)
for feat in sorted(ac.index):
    r = ac.loc[feat]
    vals = []
    for c in cols_report:
        v = r.get(c, np.nan)
        if isinstance(v, float):
            if abs(v) < 0.001 and v != 0:
                vals.append(f"{v:.2e}")
            else:
                vals.append(f"{v:.4f}")
        else:
            vals.append(str(v))
    L.append(f"| {feat} | " + " | ".join(vals) + " |")
L.append("")

# Output file list
L.append("---\n## Output Files\n")
L.append(f"- `{out1.name}` ({df_quant_all.shape[0]} rows, {df_quant_all.shape[1]} cols)")
L.append(f"- `{out2.name}` ({df_quant_sub.shape[0]} rows, {df_quant_sub.shape[1]} cols)")
L.append(f"- `{out3.name}` ({df_consistency.shape[0]} rows, {df_consistency.shape[1]} cols)")
L.append(f"- `spatial_quantification_full.md` (this file)")

md_path = RESULTS_DIR / "spatial_quantification_full.md"
md_path.write_text("\n".join(L), encoding="utf-8")
print(f"  {md_path.name}", flush=True)
print(f"\nDone. ({time.time()-t0:.0f}s total)", flush=True)
