"""
Step 2: Spatial Hexbin + GAM Comparison Plots
=============================================
Creates triptych plots (Before | After | Delta) for curated features,
at all-cells and per-group levels. Saves hexbin parquets and metrics.

Reuses logic from spatial_plots_improved_v2.py.
"""

import logging
import math
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

from compare_config import OUTPUT_DIR, FIG_DIR_BASE

INPUT_PARQUET = OUTPUT_DIR / "compared_dataframe_v2_labeled_spatial.parquet"
FIG_DIR = FIG_DIR_BASE / "spatial"
FIG_ALL_DIR = FIG_DIR / "all_cells"
FIG_GRP_DIR = FIG_DIR / "per_group"

for d in [OUTPUT_DIR, FIG_ALL_DIR, FIG_GRP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
X_COL = "improved_tx"
Y_COL = "improved_ty"
COORD_SCALE = 16
COORD_LIMIT = 100
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)

GRIDSIZE_ALL = 40
GRIDSIZE_GRP = 15
MINCNT_ALL = 2
MINCNT_GRP = 1
CMAP = "coolwarm"
CMAP_DELTA = "RdBu_r"
N_SPLINES_ALL = 30
N_SPLINES_GRP = 15

CURATED_FEATURES = [
    "green_blue_on_ratio",
    "green_blue_off_ratio",
    "green_blue_on_ratio_high",
    "green_blue_off_ratio_high",
    "on_off_ratio",
    "on_off_sus_ratio",
    "step_up_QI",
    "dsi",
    "osi",
    "on_peak_extreme",
    "off_peak_extreme",
    "on_sustained",
    "off_sustained",
    "on_trans_sus_ratio",
    "off_trans_sus_ratio",
    "iprgc_2hz_QI",
    "gb_base_mean",
    "gb_base_mean_high",
]


# =====================================================================
# Functions reused from spatial_plots_improved_v2.py
# =====================================================================

def _choose_gam_family(y):
    uniq = np.unique(y[np.isfinite(y)])
    if set(uniq).issubset({0, 1}):
        return LogisticGAM
    if np.all(y >= 0) and np.allclose(y, np.round(y)) and y.max() > 5:
        return PoissonGAM
    return LinearGAM


def _fit_gam(x, y, c, n_splines):
    """Fit GAM, return model or None."""
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
    """Returns (centers, raw_means, counts)."""
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
# Triptych plotting
# =====================================================================

def plot_triptych(x, y, c_before, c_after, c_delta, feature, gridsize,
                  mincnt, n_splines, save_path, title_prefix=""):
    """
    Create a 2-row x 3-col figure:
      Row 1: Raw hexbin (Before | After | Delta)
      Row 2: GAM smoothed (Before | After | Delta)
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.subplots_adjust(wspace=0.25, hspace=0.30, right=0.92)

    conditions = [
        (c_before, "Before", CMAP),
        (c_after, "After", CMAP),
        (c_delta, "Delta", CMAP_DELTA),
    ]

    # Compute shared color scale for before/after
    ba_mean = np.mean(np.concatenate([c_before, c_after]))
    ba_range = 0.5 * abs(ba_mean) if ba_mean != 0 else max(np.std(c_before), np.std(c_after))
    ba_vmin, ba_vmax = ba_mean - ba_range, ba_mean + ba_range

    # Delta color scale centered at 0
    d_abs_max = max(abs(np.percentile(c_delta, 5)), abs(np.percentile(c_delta, 95)))
    if d_abs_max == 0:
        d_abs_max = max(abs(c_delta.min()), abs(c_delta.max()))
    if d_abs_max == 0:
        d_abs_max = 1.0
    d_vmin, d_vmax = -d_abs_max, d_abs_max

    for col_idx, (c_vals, label, cmap) in enumerate(conditions):
        if col_idx < 2:
            vmin, vmax = ba_vmin, ba_vmax
        else:
            vmin, vmax = d_vmin, d_vmax

        ax_raw = axes[0, col_idx]
        ax_gam = axes[1, col_idx]

        # Raw hexbin
        hb = ax_raw.hexbin(
            x, y, C=c_vals, reduce_C_function=np.mean, gridsize=gridsize,
            extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        ax_raw.set_aspect("equal")
        ax_raw.set_xlim(XY_RANGE)
        ax_raw.set_ylim(XY_RANGE)
        ax_raw.set_title(f"{label} - Raw (n={len(c_vals)})", fontsize=10)
        fig.colorbar(hb, ax=ax_raw, shrink=0.6, pad=0.02)

        # GAM
        gam = _fit_gam(x, y, c_vals, n_splines)
        if gam is not None:
            hb_g = ax_gam.hexbin(
                x, y, gridsize=gridsize,
                extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=cmap,
            )
            offsets = hb_g.get_offsets()
            if len(offsets) > 0:
                preds = gam.predict(offsets)
                hb_g.set_array(preds)
                hb_g.set_clim(vmin=vmin, vmax=vmax)
            fig.colorbar(hb_g, ax=ax_gam, shrink=0.6, pad=0.02)
        ax_gam.set_aspect("equal")
        ax_gam.set_xlim(XY_RANGE)
        ax_gam.set_ylim(XY_RANGE)
        ax_gam.set_title(f"{label} - GAM", fontsize=10)

    fig.suptitle(f"{title_prefix}{feature}", fontsize=13, y=0.98)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    logger.info("Loading data ...")
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info(f"  Shape: {df.shape}")

    # Filter: need valid improved coordinates
    df = df.dropna(subset=[X_COL, Y_COL])
    mask = (df[X_COL].abs() < COORD_LIMIT) & (df[Y_COL].abs() < COORD_LIMIT)
    df = df[mask].copy()
    logger.info(f"  After coord filter: {df.shape}")

    # Identify available features
    available_features = []
    for feat in CURATED_FEATURES:
        bc = f"before_{feat}"
        ac = f"after_{feat}"
        if bc in df.columns and ac in df.columns:
            available_features.append(feat)
    logger.info(f"  Available curated features: {len(available_features)}")

    # Phase 1: All-cells triptych + hexbin data + metrics
    logger.info("\n=== Phase 1: All-cells triptych plots ===")
    hexbin_rows_before = []
    hexbin_rows_after = []
    hexbin_rows_delta = []
    metrics_rows = []

    for fi, feat in enumerate(available_features):
        bc = f"before_{feat}"
        ac = f"after_{feat}"

        cols = [X_COL, Y_COL, bc, ac]
        data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 10:
            logger.info(f"  [{fi+1}/{len(available_features)}] {feat} -- skipped (n={len(data)})")
            continue

        x = data[X_COL].to_numpy() * COORD_SCALE
        y = data[Y_COL].to_numpy() * COORD_SCALE
        c_before = data[bc].to_numpy()
        c_after = data[ac].to_numpy()
        c_delta = c_after - c_before

        # Hexbin data for all three conditions
        for condition, c_vals, rows_list in [
            ("before", c_before, hexbin_rows_before),
            ("after", c_after, hexbin_rows_after),
            ("delta", c_delta, hexbin_rows_delta),
        ]:
            centers, raw_means, counts = extract_hexbin_data(
                x, y, c_vals, GRIDSIZE_ALL, MINCNT_ALL,
            )
            gam = _fit_gam(x, y, c_vals, N_SPLINES_ALL)
            gam_preds = gam.predict(centers) if gam is not None and len(centers) > 0 else None

            for bi in range(len(centers)):
                rows_list.append({
                    "scope": "all_cells",
                    "feature": feat,
                    "bin_x": centers[bi, 0],
                    "bin_y": centers[bi, 1],
                    "count": int(counts[bi]),
                    "raw_mean": float(raw_means[bi]),
                    "gam_pred": float(gam_preds[bi]) if gam_preds is not None else np.nan,
                })

            m = compute_metrics(x, y, c_vals, centers, raw_means)
            m["scope"] = "all_cells"
            m["feature"] = feat
            m["condition"] = condition
            metrics_rows.append(m)

        # Plot triptych
        save_path = FIG_ALL_DIR / f"Triptych_{feat}.png"
        plot_triptych(
            x, y, c_before, c_after, c_delta, feat,
            GRIDSIZE_ALL, MINCNT_ALL, N_SPLINES_ALL, save_path,
        )

        if (fi + 1) % 5 == 0 or (fi + 1) == len(available_features):
            logger.info(f"  [{fi+1}/{len(available_features)}] {feat}")

    # Phase 2: Per-group triptych plots
    logger.info("\n=== Phase 2: Per-group triptych plots ===")
    groups = sorted(df["group"].dropna().unique())
    groups = [g for g in groups if g != ""]
    logger.info(f"  Groups: {groups}")

    for gi, group in enumerate(groups):
        grp_df = df[df["group"] == group]
        logger.info(f"  {group}: {len(grp_df)} cells")

        for fi, feat in enumerate(available_features):
            bc = f"before_{feat}"
            ac = f"after_{feat}"

            cols = [X_COL, Y_COL, bc, ac]
            data = grp_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 5:
                continue

            x = data[X_COL].to_numpy() * COORD_SCALE
            y = data[Y_COL].to_numpy() * COORD_SCALE
            c_before = data[bc].to_numpy()
            c_after = data[ac].to_numpy()
            c_delta = c_after - c_before

            # Per-group hexbin data
            for condition, c_vals, rows_list in [
                ("before", c_before, hexbin_rows_before),
                ("after", c_after, hexbin_rows_after),
                ("delta", c_delta, hexbin_rows_delta),
            ]:
                centers, raw_means, counts = extract_hexbin_data(
                    x, y, c_vals, GRIDSIZE_GRP, MINCNT_GRP,
                )
                for bi in range(len(centers)):
                    rows_list.append({
                        "scope": group,
                        "feature": feat,
                        "bin_x": centers[bi, 0],
                        "bin_y": centers[bi, 1],
                        "count": int(counts[bi]),
                        "raw_mean": float(raw_means[bi]),
                        "gam_pred": np.nan,
                    })

                m = compute_metrics(x, y, c_vals, centers, raw_means)
                m["scope"] = group
                m["feature"] = feat
                m["condition"] = condition
                metrics_rows.append(m)

            # Plot triptych
            save_path = FIG_GRP_DIR / f"Triptych_{group}_{feat}.png"
            plot_triptych(
                x, y, c_before, c_after, c_delta, feat,
                GRIDSIZE_GRP, MINCNT_GRP, N_SPLINES_GRP, save_path,
                title_prefix=f"[{group}] ",
            )

        logger.info(f"  Completed {group}")

    # Phase 3: Save parquets
    logger.info("\n=== Phase 3: Saving results ===")

    for name, rows in [
        ("hexbin_before_all", [r for r in hexbin_rows_before if r["scope"] == "all_cells"]),
        ("hexbin_after_all", [r for r in hexbin_rows_after if r["scope"] == "all_cells"]),
        ("hexbin_delta_all", [r for r in hexbin_rows_delta if r["scope"] == "all_cells"]),
        ("hexbin_before_pergroup", [r for r in hexbin_rows_before if r["scope"] != "all_cells"]),
        ("hexbin_after_pergroup", [r for r in hexbin_rows_after if r["scope"] != "all_cells"]),
        ("hexbin_delta_pergroup", [r for r in hexbin_rows_delta if r["scope"] != "all_cells"]),
    ]:
        if rows:
            out_path = OUTPUT_DIR / f"{name}.parquet"
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            logger.info(f"  {out_path.name}: {len(rows)} rows")

    if metrics_rows:
        met_path = OUTPUT_DIR / "spatial_metrics_compare.parquet"
        pd.DataFrame(metrics_rows).to_parquet(met_path, index=False)
        logger.info(f"  {met_path.name}: {len(metrics_rows)} rows")

    logger.info("Done.")


if __name__ == "__main__":
    main()
