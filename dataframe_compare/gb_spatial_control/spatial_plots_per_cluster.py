"""
Step 2b: Per-Cluster Hexbin + GAM Heatmaps with Step-Up Traces
==============================================================
For each RGC cluster (subtype), generates a figure per GB feature with:
  Row 1: Raw hexbin | GAM-smoothed heatmap
  Row 2: Averaged step-up response (mean +/- std across cells)
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
    OUTPUT_DIR, FIG_DIR_BASE, ALL_GB_FEATURES, SOURCE_PARQUETS,
    X_COL, Y_COL, COORD_SCALE, XY_RANGE,
    GRIDSIZE_CLUSTER, MINCNT_CLUSTER, N_SPLINES_CLUSTER,
    MIN_CELLS_CLUSTER, CMAP, STEP_TRACE_COL, SAMPLING_RATE,
    GROUP_COLORS, short,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "spatial" / "per_cluster"


# =====================================================================
# GAM / hexbin helpers (same as spatial_plots.py)
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
# Step-up trace helpers (adapted from plot_step_up_validation.py)
# =====================================================================

def _trials_to_mean_trace(cell_data):
    """Average across trials for a single cell -> 1-D array."""
    if cell_data is None:
        return None
    if isinstance(cell_data, np.ndarray) and cell_data.dtype == object:
        arrs = [np.asarray(a, dtype=np.float64) for a in cell_data if a is not None]
    elif isinstance(cell_data, list):
        arrs = [np.asarray(a, dtype=np.float64) for a in cell_data if a is not None]
    else:
        return np.asarray(cell_data, dtype=np.float64)
    if len(arrs) == 0:
        return None
    min_len = min(len(a) for a in arrs)
    stacked = np.vstack([a[:min_len] for a in arrs])
    return stacked.mean(axis=0)


def _collect_cluster_traces(trace_series):
    """Stack trial-averaged traces for a group of cells -> (n_cells, T)."""
    traces = []
    for val in trace_series:
        mt = _trials_to_mean_trace(val)
        if mt is not None and len(mt) > 0:
            traces.append(mt)
    if len(traces) == 0:
        return None
    min_len = min(len(t) for t in traces)
    return np.vstack([t[:min_len] for t in traces])


def _load_step_traces():
    """Load step-up trace column from all 3 source parquets, return combined df."""
    frames = []
    for exp, path in SOURCE_PARQUETS.items():
        if not path.exists():
            logger.warning(f"  Missing source parquet: {path}")
            continue
        logger.info(f"  Loading traces from {path.name} ...")
        cols_to_load = [STEP_TRACE_COL, "before_dataset_id", "subtype"]
        cols_to_load = [c for c in cols_to_load if c != ""]
        df = pd.read_parquet(path, columns=cols_to_load)
        df["source_experiment"] = exp
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _precompute_cluster_traces(trace_df, valid_ids):
    """Pre-compute mean/std traces per cluster for cells in valid_ids.

    Returns dict: {subtype: (mean_1d, std_1d, n_cells)} or None entries.
    """
    sub = trace_df[trace_df["before_dataset_id"].isin(valid_ids)].copy()
    result = {}
    for st in sub["subtype"].dropna().unique():
        if not st or st == "":
            continue
        cluster_rows = sub[sub["subtype"] == st]
        stacked = _collect_cluster_traces(cluster_rows[STEP_TRACE_COL])
        if stacked is not None and stacked.shape[0] >= 1:
            result[st] = (
                stacked.mean(axis=0),
                stacked.std(axis=0),
                stacked.shape[0],
            )
    return result


def _group_from_subtype(subtype):
    """Extract group prefix from subtype name (e.g. 'DSGC_3' -> 'DSGC')."""
    return subtype.rsplit("_", 1)[0]


# =====================================================================
# Per-cluster heatmap with step-up trace (2 rows)
# =====================================================================

def plot_cluster_heatmap(x, y, c, feature, gridsize, mincnt, n_splines,
                         trace_info, subtype, save_path):
    """Create a 2-row figure: hexbin+GAM on top, step-up trace on bottom."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8],
                          hspace=0.35, wspace=0.25,
                          left=0.06, right=0.94, top=0.93, bottom=0.06)

    c_mean = np.mean(c)
    c_range = 0.5 * abs(c_mean) if c_mean != 0 else np.std(c)
    if c_range == 0:
        c_range = 1.0
    vmin, vmax = c_mean - c_range, c_mean + c_range

    # Row 1 col 0: Raw hexbin
    ax_raw = fig.add_subplot(gs[0, 0])
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

    # Row 1 col 1: GAM smoothed
    ax_gam = fig.add_subplot(gs[0, 1])
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

    # Row 2: Step-up trace spanning both columns
    ax_trace = fig.add_subplot(gs[1, :])
    group = _group_from_subtype(subtype)
    line_color = GROUP_COLORS.get(group, "#333333")

    if trace_info is not None:
        mean_t, std_t, n_trace = trace_info
        t = np.arange(len(mean_t)) / SAMPLING_RATE
        ax_trace.fill_between(t, mean_t - std_t, mean_t + std_t,
                              color="gray", alpha=0.2)
        ax_trace.plot(t, mean_t, color=line_color, linewidth=1.2)
        ax_trace.set_title(f"Step-Up Response (n={n_trace} cells)", fontsize=10)
    else:
        ax_trace.text(0.5, 0.5, "no trace data", transform=ax_trace.transAxes,
                      ha="center", va="center", fontsize=11, color="gray")
        ax_trace.set_title("Step-Up Response", fontsize=10)

    ax_trace.set_xlabel("Time (s)", fontsize=9)
    ax_trace.set_ylabel("Response (Hz)", fontsize=9)
    ax_trace.tick_params(labelsize=8)

    fig.suptitle(f"[{subtype}] {feature}", fontsize=13, y=0.98)
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

    # Same response filter as spatial_plots.py
    _pos = (df["green_on_peak_extreme"] > 0) & (df["blue_on_peak_extreme"] > 0)
    _thr = (df["green_on_peak_extreme"] >= 50) | (df["blue_on_peak_extreme"] >= 50)
    df = df[_pos & _thr].reset_index(drop=True)
    logger.info(f"  Shape after response filter: {df.shape}")

    available_features = [f for f in ALL_GB_FEATURES if f in df.columns]
    logger.info(f"  Available features: {len(available_features)}")

    # Load step-up traces from source parquets
    logger.info("\n=== Loading step-up traces ===")
    trace_df = _load_step_traces()
    valid_ids = set(df["before_dataset_id"].dropna().unique())
    logger.info(f"  Valid dataset IDs after filter: {len(valid_ids)}")

    cluster_traces = _precompute_cluster_traces(trace_df, valid_ids)
    logger.info(f"  Pre-computed traces for {len(cluster_traces)} clusters")
    for st in sorted(cluster_traces):
        _, _, n = cluster_traces[st]
        logger.info(f"    {st}: {n} cells with traces")

    del trace_df

    # Identify clusters with enough cells
    subtypes = sorted(
        s for s in df["subtype"].dropna().unique()
        if s and s != ""
    )
    logger.info(f"\n=== Per-cluster heatmaps ({len(subtypes)} subtypes) ===")

    hexbin_rows = []
    metrics_rows = []

    for si, subtype in enumerate(subtypes):
        clust_df = df[df["subtype"] == subtype]
        n_cells = len(clust_df)
        if n_cells < MIN_CELLS_CLUSTER:
            logger.info(f"  [{si+1}/{len(subtypes)}] {subtype}: "
                         f"skipped (n={n_cells} < {MIN_CELLS_CLUSTER})")
            continue

        clust_fig_dir = FIG_DIR / subtype
        clust_fig_dir.mkdir(parents=True, exist_ok=True)

        trace_info = cluster_traces.get(subtype)
        logger.info(f"  [{si+1}/{len(subtypes)}] {subtype}: "
                     f"{n_cells} cells, trace={'yes' if trace_info else 'no'}")

        for fi, feat in enumerate(available_features):
            cols = [X_COL, Y_COL, feat]
            data = clust_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < MIN_CELLS_CLUSTER:
                continue

            x = data[X_COL].to_numpy() * COORD_SCALE
            y = data[Y_COL].to_numpy() * COORD_SCALE
            c = data[feat].to_numpy()

            centers, raw_means, counts = extract_hexbin_data(
                x, y, c, GRIDSIZE_CLUSTER, MINCNT_CLUSTER,
            )

            for bi in range(len(centers)):
                hexbin_rows.append({
                    "scope": subtype,
                    "feature": feat,
                    "bin_x": centers[bi, 0],
                    "bin_y": centers[bi, 1],
                    "count": int(counts[bi]),
                    "raw_mean": float(raw_means[bi]),
                    "gam_pred": np.nan,
                })

            m = compute_metrics(x, y, c, centers, raw_means)
            m["scope"] = subtype
            m["feature"] = feat
            metrics_rows.append(m)

            save_path = clust_fig_dir / f"Hexbin_{feat}.png"
            plot_cluster_heatmap(
                x, y, c, feat,
                GRIDSIZE_CLUSTER, MINCNT_CLUSTER, N_SPLINES_CLUSTER,
                trace_info, subtype, save_path,
            )

        logger.info(f"    Completed {subtype}")

    # Save parquets
    logger.info("\n=== Saving results ===")

    if hexbin_rows:
        out_path = OUTPUT_DIR / "hexbin_per_cluster.parquet"
        pd.DataFrame(hexbin_rows).to_parquet(out_path, index=False)
        logger.info(f"  {out_path.name}: {len(hexbin_rows)} rows")

    if metrics_rows:
        met_path = OUTPUT_DIR / "spatial_metrics_per_cluster.parquet"
        pd.DataFrame(metrics_rows).to_parquet(met_path, index=False)
        logger.info(f"  {met_path.name}: {len(metrics_rows)} rows")

    logger.info("Done.")


if __name__ == "__main__":
    main()
