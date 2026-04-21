"""
Step 3: Spatial Quantification (Single-Condition)
=================================================
Full spatial quantification on hexbin data for combined GB control.
No before/after/delta split -- single condition analysis.
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway
from scipy.spatial import cKDTree

from config import OUTPUT_DIR, FIG_DIR_BASE, short

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 499
N_BOOT = 499
K_NEIGHBORS = 6
FDR_ALPHA = 0.05
HOTSPOT_QUANTILE = 0.90
RNG = np.random.default_rng(42)


# =====================================================================
# Helper functions
# =====================================================================

def weighted_plane_fit(x, y, z, w):
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
    b0, bx_, by_ = coeffs
    pred = A @ coeffs
    ss_res = w @ (z - pred) ** 2
    ss_tot = w @ (z - np.average(z, weights=w)) ** 2
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mag = float(np.sqrt(bx_ ** 2 + by_ ** 2))
    direction = float(np.degrees(np.arctan2(by_, bx_)))
    return dict(bx=float(bx_), by=float(by_), b0=float(b0),
                grad_mag=mag, grad_dir_deg=direction, plane_r2=r2)


def gam_metrics(raw_mean, gam_pred, bin_x, bin_y, plane_r2):
    m = {}
    if gam_pred is None or len(gam_pred) == 0 or np.all(np.isnan(gam_pred)):
        for k in ["gam_r2", "gam_delta_r2", "gam_dynamic_range",
                   "gam_max_x", "gam_max_y", "gam_min_x", "gam_min_y",
                   "hotspot_area_frac"]:
            m[k] = np.nan
        return m
    ss_res = np.sum((raw_mean - gam_pred) ** 2)
    ss_tot = np.sum((raw_mean - np.mean(raw_mean)) ** 2)
    gam_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    m["gam_r2"] = gam_r2
    m["gam_delta_r2"] = gam_r2 - plane_r2 if not np.isnan(plane_r2) else np.nan
    m["gam_dynamic_range"] = float(np.ptp(gam_pred))
    idx_max = np.argmax(gam_pred)
    idx_min = np.argmin(gam_pred)
    m["gam_max_x"] = float(bin_x[idx_max])
    m["gam_max_y"] = float(bin_y[idx_max])
    m["gam_min_x"] = float(bin_x[idx_min])
    m["gam_min_y"] = float(bin_y[idx_min])
    threshold = np.quantile(gam_pred, HOTSPOT_QUANTILE)
    m["hotspot_area_frac"] = float(np.mean(gam_pred >= threshold))
    return m


def build_knn_weights(xy, k=6):
    n = len(xy)
    if n < k + 1:
        return None, None
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=min(k + 1, n))
    return idx, n


def moran_i_global(z, knn_idx, n):
    if knn_idx is None or n < 4:
        return np.nan
    zm = z - np.mean(z)
    denom = np.sum(zm ** 2)
    if denom == 0:
        return np.nan
    numer, W = 0.0, 0.0
    for i in range(n):
        for jp in range(1, knn_idx.shape[1]):
            j = knn_idx[i, jp]
            numer += zm[i] * zm[j]
            W += 1
    return float((n / W) * (numer / denom)) if W > 0 else np.nan


def radial_angular_analysis(bx, by, z, w, n_boot=499):
    m = {}
    r = np.sqrt(bx ** 2 + by ** 2)
    if np.std(r) < 1e-12 or np.std(z) < 1e-12 or len(z) < 10:
        for k in ["radial_r", "radial_p", "radial_r_lo", "radial_r_hi"]:
            m[k] = np.nan
        m["quadrant_F"] = np.nan
        m["quadrant_p"] = np.nan
        return m
    rho, pval = pearsonr(r, z)
    m["radial_r"] = float(rho)
    m["radial_p"] = float(pval)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.choice(len(z), size=len(z), replace=True)
        try:
            boots[b], _ = pearsonr(r[idx], z[idx])
        except Exception:
            boots[b] = np.nan
    m["radial_r_lo"] = float(np.nanpercentile(boots, 2.5))
    m["radial_r_hi"] = float(np.nanpercentile(boots, 97.5))
    angles = np.degrees(np.arctan2(by, bx)) % 360
    q_labels = (angles // 90).astype(int)
    groups_vals = [z[q_labels == q] for q in range(4) if (q_labels == q).sum() > 1]
    if len(groups_vals) >= 2:
        try:
            F, p = f_oneway(*groups_vals)
            m["quadrant_F"] = float(F)
            m["quadrant_p"] = float(p)
        except Exception:
            m["quadrant_F"] = m["quadrant_p"] = np.nan
    else:
        m["quadrant_F"] = m["quadrant_p"] = np.nan
    return m


def permutation_pvalue(z, bx, by, w, knn_idx, n, n_perm=499):
    plane = weighted_plane_fit(bx, by, z, w)
    obs_r2 = plane["plane_r2"]
    obs_moran = moran_i_global(z, knn_idx, n)
    r2_null = np.empty(n_perm)
    moran_null = np.empty(n_perm)
    for p in range(n_perm):
        zp = RNG.permutation(z)
        r2_null[p] = weighted_plane_fit(bx, by, zp, w)["plane_r2"]
        moran_null[p] = moran_i_global(zp, knn_idx, n)
    p_r2 = float(np.mean(r2_null >= obs_r2)) if not np.isnan(obs_r2) else np.nan
    p_moran = float(np.mean(moran_null >= obs_moran)) if not np.isnan(obs_moran) else np.nan
    return p_r2, p_moran


def fdr_correct(pvalues):
    pv = np.array(pvalues, dtype=float)
    n = len(pv)
    valid = ~np.isnan(pv)
    q = np.full(n, np.nan)
    if valid.sum() == 0:
        return q
    idx_valid = np.where(valid)[0]
    pv_valid = pv[idx_valid]
    order = np.argsort(pv_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    q_valid = pv_valid * len(pv_valid) / ranks
    q_valid = np.minimum.accumulate(q_valid[np.argsort(-ranks)])
    q_valid = q_valid[np.argsort(np.argsort(ranks))]
    q[idx_valid] = np.minimum(q_valid, 1.0)
    return q


def analyse_feature(bx, by, raw_mean, gam_pred, counts, feature_name):
    w = counts / counts.sum() if counts.sum() > 0 else np.ones(len(counts))
    knn_idx, n = build_knn_weights(np.column_stack([bx, by]), K_NEIGHBORS)

    result = {"feature": feature_name, "n_bins": len(bx)}
    result["overall_mean"] = float(np.average(raw_mean, weights=w))
    result["overall_std"] = float(np.sqrt(np.average((raw_mean - result["overall_mean"]) ** 2, weights=w)))

    plane = weighted_plane_fit(bx, by, raw_mean, w)
    result.update(plane)

    gam_m = gam_metrics(raw_mean, gam_pred, bx, by, plane["plane_r2"])
    result.update(gam_m)

    result["moran_i"] = moran_i_global(raw_mean, knn_idx, n)

    rad = radial_angular_analysis(bx, by, raw_mean, w, N_BOOT)
    result.update(rad)

    if n is not None and n >= 10:
        p_r2, p_moran = permutation_pvalue(raw_mean, bx, by, w, knn_idx, n, N_PERM)
        result["perm_p_plane_r2"] = p_r2
        result["perm_p_moran"] = p_moran
    else:
        result["perm_p_plane_r2"] = np.nan
        result["perm_p_moran"] = np.nan

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    hex_files = {
        "all": OUTPUT_DIR / "hexbin_all_cells.parquet",
        "pergroup": OUTPUT_DIR / "hexbin_per_group.parquet",
    }

    all_results = []

    for scope_type, fpath in hex_files.items():
        if not fpath.exists():
            logger.warning(f"  Missing: {fpath}")
            continue

        logger.info(f"\n=== {scope_type} ===")
        df_hex = pd.read_parquet(fpath)
        features = sorted(df_hex["feature"].unique())
        scopes = sorted(df_hex["scope"].unique())
        logger.info(f"  Features: {len(features)}, Scopes: {len(scopes)}")

        t0 = time.time()
        for fi, feat in enumerate(features):
            for scope in scopes:
                sub = df_hex[(df_hex["feature"] == feat) & (df_hex["scope"] == scope)]
                if len(sub) < 5:
                    continue
                bx = sub["bin_x"].to_numpy()
                by = sub["bin_y"].to_numpy()
                raw_mean = sub["raw_mean"].to_numpy()
                gam_pred = sub["gam_pred"].to_numpy()
                counts = sub["count"].to_numpy().astype(float)

                mask = np.isfinite(raw_mean)
                if mask.sum() < 5:
                    continue

                gp = gam_pred[mask] if np.any(np.isfinite(gam_pred[mask])) else None

                result = analyse_feature(
                    bx[mask], by[mask], raw_mean[mask], gp,
                    counts[mask], feat,
                )
                result["scope"] = scope
                all_results.append(result)

            if (fi + 1) % 5 == 0 or (fi + 1) == len(features):
                elapsed = time.time() - t0
                logger.info(f"  [{fi+1}/{len(features)}] {feat} ({elapsed:.0f}s)")

    if not all_results:
        logger.warning("No results to save")
        return

    df_all = pd.DataFrame(all_results)

    # FDR correction per scope group
    for scope, grp in df_all.groupby("scope"):
        for pcol, qcol in [
            ("perm_p_plane_r2", "fdr_q_plane_r2"),
            ("perm_p_moran", "fdr_q_moran"),
        ]:
            if pcol in grp.columns:
                q = fdr_correct(grp[pcol].to_numpy())
                df_all.loc[grp.index, qcol] = q

    # Save split
    all_cells = df_all[df_all["scope"] == "all_cells"]
    per_group = df_all[df_all["scope"] != "all_cells"]

    if len(all_cells) > 0:
        out_path = OUTPUT_DIR / "spatial_quant_all.parquet"
        all_cells.to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(all_cells)} rows")

    if len(per_group) > 0:
        out_path = OUTPUT_DIR / "spatial_quant_per_group.parquet"
        per_group.to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(per_group)} rows")

    # Combined
    combined_path = OUTPUT_DIR / "spatial_quant_combined.parquet"
    df_all.to_parquet(combined_path, index=False)
    logger.info(f"  {combined_path.name}: {len(df_all)} rows")

    _write_summary(df_all)
    logger.info("Done.")


def _write_summary(df_all):
    lines = []
    lines.append("# Spatial Quantification: GB Control (Combined Before-Blocker)\n")

    ac = df_all[df_all["scope"] == "all_cells"]
    if len(ac) == 0:
        return

    ac = ac.set_index("feature")

    lines.append("\n## All Cells\n")
    lines.append("### Strongest Spatial Gradients\n")
    lines.append("| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r | FDR q (R2) |")
    lines.append("|---------|----------|-----------|----------|---------|----------|------------|")
    for feat, row in ac.nlargest(15, "grad_mag").iterrows():
        fdr = row.get("fdr_q_plane_r2", np.nan)
        fdr_str = f"{fdr:.4f}" if not np.isnan(fdr) else "n/a"
        lines.append(
            f"| {short(feat)} | {row['grad_mag']:.6f} | {row['grad_dir_deg']:.1f} "
            f"| {row['plane_r2']:.4f} | {row.get('moran_i', np.nan):.4f} "
            f"| {row.get('radial_r', np.nan):.4f} | {fdr_str} |"
        )

    lines.append("\n### Strongest Spatial Clustering (Moran's I)\n")
    lines.append("| Feature | Moran I | Plane R2 | Radial r | FDR q (Moran) |")
    lines.append("|---------|---------|----------|----------|---------------|")
    for feat, row in ac.nlargest(15, "moran_i").iterrows():
        fdr = row.get("fdr_q_moran", np.nan)
        fdr_str = f"{fdr:.4f}" if not np.isnan(fdr) else "n/a"
        lines.append(
            f"| {short(feat)} | {row['moran_i']:.4f} | {row['plane_r2']:.4f} "
            f"| {row.get('radial_r', np.nan):.4f} | {fdr_str} |"
        )

    lines.append("")

    md_path = FIG_DIR_BASE / "spatial" / "spatial_quantification_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  Summary: {md_path.name}")


if __name__ == "__main__":
    main()
