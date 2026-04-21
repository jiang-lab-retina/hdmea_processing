"""
Step 4: Radial Center Analysis (Single-Condition)
=================================================
Searches for optimal radial centers on hexbin data for combined
GB control dataset.
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import minimize

from config import OUTPUT_DIR, FIG_DIR_BASE, short

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COARSE_STEP = 200
FINE_RADIUS = 300
FINE_STEP = 50
SEARCH_LIMIT = 1200
MIN_BINS = 15


# =====================================================================
# Core functions
# =====================================================================

def radial_corr(cx, cy, bx, by, vals):
    r = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
    if np.std(r) < 1e-12 or np.std(vals) < 1e-12:
        return 0.0, 1.0, 0.0
    try:
        rho, pval = pearsonr(r, vals)
    except Exception:
        return 0.0, 1.0, 0.0
    A = np.column_stack([r, np.ones_like(r)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
        slope = coeffs[0]
    except Exception:
        slope = 0.0
    return float(rho), float(pval), float(slope)


def search_radial_center(bx, by, vals):
    r0, p0, s0 = radial_corr(0, 0, bx, by, vals)

    xs = np.arange(-SEARCH_LIMIT, SEARCH_LIMIT + 1, COARSE_STEP)
    ys = np.arange(-SEARCH_LIMIT, SEARCH_LIMIT + 1, COARSE_STEP)
    best_abs_r, best_cx, best_cy = abs(r0), 0.0, 0.0
    for cx in xs:
        for cy in ys:
            rho, _, _ = radial_corr(cx, cy, bx, by, vals)
            if abs(rho) > best_abs_r:
                best_abs_r = abs(rho)
                best_cx, best_cy = cx, cy

    xs2 = np.arange(best_cx - FINE_RADIUS, best_cx + FINE_RADIUS + 1, FINE_STEP)
    ys2 = np.arange(best_cy - FINE_RADIUS, best_cy + FINE_RADIUS + 1, FINE_STEP)
    for cx in xs2:
        for cy in ys2:
            rho, _, _ = radial_corr(cx, cy, bx, by, vals)
            if abs(rho) > best_abs_r:
                best_abs_r = abs(rho)
                best_cx, best_cy = cx, cy

    MAX_CENTER = SEARCH_LIMIT * 1.5

    def neg_abs_r(params):
        if abs(params[0]) > MAX_CENTER or abs(params[1]) > MAX_CENTER:
            return 0.0
        rho, _, _ = radial_corr(params[0], params[1], bx, by, vals)
        return -abs(rho)

    try:
        res = minimize(neg_abs_r, [best_cx, best_cy], method="Nelder-Mead",
                       options={"xatol": 10, "fatol": 1e-6, "maxiter": 500})
        cx_cand, cy_cand = float(res.x[0]), float(res.x[1])
        if (abs(cx_cand) <= MAX_CENTER and abs(cy_cand) <= MAX_CENTER
                and (res.success or -res.fun > best_abs_r)):
            best_cx, best_cy = cx_cand, cy_cand
    except Exception:
        pass

    r_best, p_best, s_best = radial_corr(best_cx, best_cy, bx, by, vals)

    return {
        "origin_r": r0, "origin_p": p0, "origin_slope": s0,
        "best_center_x": best_cx, "best_center_y": best_cy,
        "best_r": r_best, "best_p": p_best, "best_slope": s_best,
        "abs_r_improvement": abs(r_best) - abs(r0),
    }


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

        logger.info(f"\n=== {scope_type} radial center search ===")
        df_hex = pd.read_parquet(fpath)
        features = sorted(df_hex["feature"].unique())
        scopes = sorted(df_hex["scope"].unique())
        logger.info(f"  Features: {len(features)}, Scopes: {len(scopes)}")

        t0 = time.time()
        for fi, feat in enumerate(features):
            for scope in scopes:
                sub = df_hex[(df_hex["feature"] == feat) & (df_hex["scope"] == scope)]
                if len(sub) < MIN_BINS:
                    continue
                bx = sub["bin_x"].to_numpy()
                by = sub["bin_y"].to_numpy()

                for data_type in ["raw_mean", "gam_pred"]:
                    vals = sub[data_type].to_numpy()
                    mask = np.isfinite(vals)
                    if mask.sum() < MIN_BINS:
                        continue
                    res = search_radial_center(bx[mask], by[mask], vals[mask])
                    res["scope"] = scope
                    res["feature"] = feat
                    res["data_type"] = data_type
                    res["n_bins"] = int(mask.sum())
                    all_results.append(res)

            if (fi + 1) % 5 == 0 or (fi + 1) == len(features):
                elapsed = time.time() - t0
                logger.info(f"  [{fi+1}/{len(features)}] {feat} ({elapsed:.0f}s)")

    if not all_results:
        logger.warning("No results")
        return

    df_all = pd.DataFrame(all_results)

    # Save split
    all_cells = df_all[df_all["scope"] == "all_cells"]
    per_group = df_all[df_all["scope"] != "all_cells"]

    if len(all_cells) > 0:
        out_path = OUTPUT_DIR / "radial_center_all.parquet"
        all_cells.to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(all_cells)} rows")

    if len(per_group) > 0:
        out_path = OUTPUT_DIR / "radial_center_per_group.parquet"
        per_group.to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(per_group)} rows")

    combined_path = OUTPUT_DIR / "radial_center_combined.parquet"
    df_all.to_parquet(combined_path, index=False)
    logger.info(f"  {combined_path.name}: {len(df_all)} rows")

    _write_summary(df_all)
    logger.info("Done.")


def _write_summary(df_all):
    lines = []
    lines.append("# Radial Center Analysis: GB Control (Combined Before-Blocker)\n")

    ac_raw = df_all[
        (df_all["scope"] == "all_cells") &
        (df_all["data_type"] == "raw_mean")
    ]
    if len(ac_raw) == 0:
        return

    lines.append("\n## All Cells (Raw Mean)\n")
    lines.append("### Strongest radial trends (by |best_r|)\n")
    lines.append("| Feature | Best Cx | Best Cy | best_r | best_p | origin_r | Improvement |")
    lines.append("|---------|---------|---------|--------|--------|----------|-------------|")
    top = ac_raw.reindex(ac_raw["best_r"].abs().nlargest(15).index)
    for _, row in top.iterrows():
        lines.append(
            f"| {short(row['feature'])} | {row['best_center_x']:.0f} | {row['best_center_y']:.0f} "
            f"| {row['best_r']:.4f} | {row['best_p']:.2e} "
            f"| {row['origin_r']:.4f} | {row['abs_r_improvement']:.4f} |"
        )

    lines.append("\n### Largest improvement from center search\n")
    lines.append("| Feature | Improvement | origin_r | best_r | Best Cx | Best Cy |")
    lines.append("|---------|-------------|----------|--------|---------|---------|")
    top_imp = ac_raw.nlargest(15, "abs_r_improvement")
    for _, row in top_imp.iterrows():
        lines.append(
            f"| {short(row['feature'])} | {row['abs_r_improvement']:.4f} "
            f"| {row['origin_r']:.4f} | {row['best_r']:.4f} "
            f"| {row['best_center_x']:.0f} | {row['best_center_y']:.0f} |"
        )

    lines.append("")

    md_path = FIG_DIR_BASE / "spatial" / "radial_center_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  Summary: {md_path.name}")


if __name__ == "__main__":
    main()
