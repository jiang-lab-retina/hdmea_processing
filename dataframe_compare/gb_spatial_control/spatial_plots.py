"""
Step 2: Spatial Hexbin + GAM Heatmaps (Single-Condition)
========================================================
Creates hexbin heatmaps (raw + GAM) for each GB feature, at all-cells
and per-group levels.  No before/after comparison -- single condition only.
"""

import logging
import warnings
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from pygam import LinearGAM, LogisticGAM, PoissonGAM, te

from config import (
    OUTPUT_DIR, FIG_DIR_BASE, ALL_GB_FEATURES,
    X_COL, Y_COL, COORD_SCALE, XY_RANGE,
    GRIDSIZE_ALL, GRIDSIZE_GRP, MINCNT_ALL, MINCNT_GRP,
    CMAP, N_SPLINES_ALL, N_SPLINES_GRP, short,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "spatial"
FIG_ALL_DIR = FIG_DIR / "all_cells"
FIG_GRP_DIR = FIG_DIR / "per_group"

for d in [OUTPUT_DIR, FIG_ALL_DIR, FIG_GRP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Reused helper functions
# =====================================================================

def _choose_gam_family(y):
    uniq = np.unique(y[np.isfinite(y)])
    if set(uniq).issubset({0, 1}):
        return LogisticGAM
    if np.all(y >= 0) and np.allclose(y, np.round(y)) and y.max() > 5:
        return PoissonGAM
    return LinearGAM


def _fit_gam(x, y, c, n_splines):
    GamClass = _choose_gam_family(c)
    X_train = np.column_stack([x, y])
    gam = GamClass(te(0, 1, n_splines=[n_splines, n_splines]))
    try:
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            gam = gam.gridsearch(X_train, c)
        return gam
    except Exception:
        try:
            gam.fit(X_train, c)
            return gam
        except Exception:
            return None


def extract_hexbin_data(x, y, c, gridsize, mincnt):
    fig, ax = plt.subplots(figsize=(4, 4))
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP,
    )
    centers = hb.get_offsets().copy()
    means = hb.get_array().copy()
    ax.cla()
    hb2 = ax.hexbin(
        x, y, gridsize=gridsize, extent=(*XY_RANGE, *XY_RANGE),
        mincnt=mincnt, cmap=CMAP,
    )
    counts = hb2.get_array().copy()
    plt.close(fig)
    n = min(len(centers), len(means), len(counts))
    return centers[:n], means[:n], counts[:n]


def compute_moran_i(bin_centers, bin_values, k=6):
    n = len(bin_values)
    if n < k + 1:
        return np.nan
    z = bin_values - np.mean(bin_values)
    denom = np.sum(z ** 2)
    if denom == 0:
        return np.nan
    tree = cKDTree(bin_centers)
    _, idx = tree.query(bin_centers, k=min(k + 1, n))
    numer, W = 0.0, 0.0
    for i in range(n):
        for j_pos in range(1, idx.shape[1]):
            j = idx[i, j_pos]
            numer += z[i] * z[j]
            W += 1.0
    return float((n / W) * (numer / denom)) if W > 0 else np.nan


def compute_metrics(x_um, y_um, c, bin_centers, bin_means):
    m = {}
    m["n_valid"] = len(c)
    m["n_bins"] = len(bin_means)
    m["overall_mean"] = float(np.mean(c))
    m["overall_std"] = float(np.std(c))
    if len(bin_means) > 2:
        bm_mean = np.mean(bin_means)
        m["hexbin_cv"] = float(np.std(bin_means) / abs(bm_mean)) if bm_mean != 0 else np.nan
    else:
        m["hexbin_cv"] = np.nan
    if len(c) >= 10:
        A = np.column_stack([x_um, y_um, np.ones(len(x_um))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, c, rcond=None)
            gx, gy = coeffs[0], coeffs[1]
            pred = A @ coeffs
            ss_res = np.sum((c - pred) ** 2)
            ss_tot = np.sum((c - np.mean(c)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            m["gradient_mag"] = float(np.sqrt(gx ** 2 + gy ** 2))
            m["gradient_dir_deg"] = float(np.degrees(np.arctan2(gy, gx)))
            m["gradient_r2"] = float(r2)
        except Exception:
            m["gradient_mag"] = m["gradient_dir_deg"] = m["gradient_r2"] = np.nan
    else:
        m["gradient_mag"] = m["gradient_dir_deg"] = m["gradient_r2"] = np.nan
    radius = np.sqrt(x_um ** 2 + y_um ** 2)
    if len(c) >= 10 and np.std(radius) > 0 and np.std(c) > 0:
        try:
            r_val, p_val = pearsonr(radius, c)
            m["radial_r"] = float(r_val)
            m["radial_p"] = float(p_val)
        except Exception:
            m["radial_r"] = m["radial_p"] = np.nan
    else:
        m["radial_r"] = m["radial_p"] = np.nan
    if len(bin_means) >= 7:
        m["moran_i"] = compute_moran_i(bin_centers, bin_means)
    else:
        m["moran_i"] = np.nan
    return m


# =====================================================================
# Single-condition heatmap (2 panels: raw + GAM)
# =====================================================================

def plot_heatmap(x, y, c, feature, gridsize, mincnt, n_splines,
                 save_path, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.25, right=0.92)

    c_mean = np.mean(c)
    c_range = 0.5 * abs(c_mean) if c_mean != 0 else np.std(c)
    if c_range == 0:
        c_range = 1.0
    vmin, vmax = c_mean - c_range, c_mean + c_range

    ax_raw = axes[0]
    hb = ax_raw.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP,
        vmin=vmin, vmax=vmax,
    )
    ax_raw.set_aspect("equal")
    ax_raw.set_xlim(XY_RANGE)
    ax_raw.set_ylim(XY_RANGE)
    ax_raw.set_title(f"Raw Hexbin (n={len(c)})", fontsize=10)
    fig.colorbar(hb, ax=ax_raw, shrink=0.6, pad=0.02)

    ax_gam = axes[1]
    gam = _fit_gam(x, y, c, n_splines)
    if gam is not None:
        hb_g = ax_gam.hexbin(
            x, y, gridsize=gridsize,
            extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP,
        )
        offsets = hb_g.get_offsets()
        if len(offsets) > 0:
            preds = gam.predict(offsets)
            hb_g.set_array(preds)
            gam_mean = float(np.mean(preds))
            gam_std = float(np.std(preds))
            gam_std = max(gam_std, 1e-6)
            hb_g.set_clim(vmin=gam_mean - 5 * gam_std,
                          vmax=gam_mean + 5 * gam_std)
        fig.colorbar(hb_g, ax=ax_gam, shrink=0.6, pad=0.02)
    ax_gam.set_aspect("equal")
    ax_gam.set_xlim(XY_RANGE)
    ax_gam.set_ylim(XY_RANGE)
    ax_gam.set_title("GAM Smoothed", fontsize=10)

    fig.suptitle(f"{title_prefix}{feature}", fontsize=13, y=0.98)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    input_path = OUTPUT_DIR / "combined_gb_control.parquet"
    logger.info(f"Loading {input_path.name} ...")
    df = pd.read_parquet(input_path)
    logger.info(f"  Shape before filter: {df.shape}")

    # Response filter: positive ON peaks in both channels, either >= 50 Hz
    _pos = (df["green_on_peak_extreme"] > 0) & (df["blue_on_peak_extreme"] > 0)
    _thr = (df["green_on_peak_extreme"] >= 50) | (df["blue_on_peak_extreme"] >= 50)
    df = df[_pos & _thr].reset_index(drop=True)
    logger.info(f"  Shape after response filter: {df.shape}")

    available_features = [f for f in ALL_GB_FEATURES if f in df.columns]
    logger.info(f"  Available features: {len(available_features)}")

    # Phase 1: All-cells heatmaps + hexbin data + metrics
    logger.info("\n=== Phase 1: All-cells heatmaps ===")
    hexbin_rows = []
    metrics_rows = []

    for fi, feat in enumerate(available_features):
        cols = [X_COL, Y_COL, feat]
        data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 10:
            logger.info(f"  [{fi+1}/{len(available_features)}] {feat} -- skipped (n={len(data)})")
            continue

        x = data[X_COL].to_numpy() * COORD_SCALE
        y = data[Y_COL].to_numpy() * COORD_SCALE
        c = data[feat].to_numpy()

        centers, raw_means, counts = extract_hexbin_data(
            x, y, c, GRIDSIZE_ALL, MINCNT_ALL,
        )
        gam = _fit_gam(x, y, c, N_SPLINES_ALL)
        gam_preds = gam.predict(centers) if gam is not None and len(centers) > 0 else None

        for bi in range(len(centers)):
            hexbin_rows.append({
                "scope": "all_cells",
                "feature": feat,
                "bin_x": centers[bi, 0],
                "bin_y": centers[bi, 1],
                "count": int(counts[bi]),
                "raw_mean": float(raw_means[bi]),
                "gam_pred": float(gam_preds[bi]) if gam_preds is not None else np.nan,
            })

        m = compute_metrics(x, y, c, centers, raw_means)
        m["scope"] = "all_cells"
        m["feature"] = feat
        metrics_rows.append(m)

        save_path = FIG_ALL_DIR / f"Hexbin_{feat}.png"
        plot_heatmap(x, y, c, feat, GRIDSIZE_ALL, MINCNT_ALL,
                     N_SPLINES_ALL, save_path)

        if (fi + 1) % 5 == 0 or (fi + 1) == len(available_features):
            logger.info(f"  [{fi+1}/{len(available_features)}] {feat}")

    # Phase 2: Per-group heatmaps
    logger.info("\n=== Phase 2: Per-group heatmaps ===")
    groups = sorted(g for g in df["group"].dropna().unique() if g != "")
    logger.info(f"  Groups: {groups}")

    for gi, group in enumerate(groups):
        grp_df = df[df["group"] == group]
        logger.info(f"  {group}: {len(grp_df)} cells")

        for fi, feat in enumerate(available_features):
            cols = [X_COL, Y_COL, feat]
            data = grp_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 5:
                continue

            x = data[X_COL].to_numpy() * COORD_SCALE
            y = data[Y_COL].to_numpy() * COORD_SCALE
            c = data[feat].to_numpy()

            centers, raw_means, counts = extract_hexbin_data(
                x, y, c, GRIDSIZE_GRP, MINCNT_GRP,
            )

            for bi in range(len(centers)):
                hexbin_rows.append({
                    "scope": group,
                    "feature": feat,
                    "bin_x": centers[bi, 0],
                    "bin_y": centers[bi, 1],
                    "count": int(counts[bi]),
                    "raw_mean": float(raw_means[bi]),
                    "gam_pred": np.nan,
                })

            m = compute_metrics(x, y, c, centers, raw_means)
            m["scope"] = group
            m["feature"] = feat
            metrics_rows.append(m)

            save_path = FIG_GRP_DIR / f"Hexbin_{group}_{feat}.png"
            plot_heatmap(x, y, c, feat, GRIDSIZE_GRP, MINCNT_GRP,
                         N_SPLINES_GRP, save_path, title_prefix=f"[{group}] ")

        logger.info(f"  Completed {group}")

    # Phase 3: Save parquets
    logger.info("\n=== Phase 3: Saving results ===")

    all_cells_rows = [r for r in hexbin_rows if r["scope"] == "all_cells"]
    per_group_rows = [r for r in hexbin_rows if r["scope"] != "all_cells"]

    if all_cells_rows:
        out_path = OUTPUT_DIR / "hexbin_all_cells.parquet"
        pd.DataFrame(all_cells_rows).to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(all_cells_rows)} rows")

    if per_group_rows:
        out_path = OUTPUT_DIR / "hexbin_per_group.parquet"
        pd.DataFrame(per_group_rows).to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(per_group_rows)} rows")

    if metrics_rows:
        met_path = OUTPUT_DIR / "spatial_metrics.parquet"
        pd.DataFrame(metrics_rows).to_parquet(met_path, index=False)
        logger.info(f"  {met_path.name}: {len(metrics_rows)} rows")

    logger.info("Done.")


if __name__ == "__main__":
    main()
