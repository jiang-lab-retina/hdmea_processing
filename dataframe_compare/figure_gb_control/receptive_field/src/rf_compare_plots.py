"""
RF Compare Hexbin + GAM Heatmaps (4 conditions)
================================================
For each scalar RF feature, generates multi-panel comparison figures across
four conditions: before (control), STR, PTX, STR_PTX.

Two scopes:
  1. **all_cells** -- 5 rows x 4 cols GridSpec.
     Rows 0-3: raw hexbin (cols 0-1) + GAM (cols 2-3) per condition.
     Row 4: step-up averaged trace, one subplot per condition.

  2. **per_subtype** -- 5 rows x 4 cols GridSpec per cluster.
     Rows 0-3: raw hexbin | GAM (subtype) | GAM projected onto all-cells
               grid | cell-density hexbin per condition.
     Row 4: step-up averaged trace per condition (within the subtype).

Step traces are loaded directly from source parquets (not stored in the
combined RF parquet) following the pattern in
``gb_spatial_control/spatial_plots_per_cluster.py``.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import (
    SOURCE_PARQUETS, OUTPUT_DIR, FIG_DIR_BASE,
    X_COL, Y_COL, COORD_SCALE, XY_RANGE,
    GRIDSIZE_ALL, GRIDSIZE_GRP, MINCNT_ALL, MINCNT_GRP,
    CMAP, N_SPLINES_ALL, N_SPLINES_GRP,
    SAMPLING_RATE,
)
from spatial_plots import (
    extract_hexbin_data, compute_metrics, _fit_gam,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS = ["before", "STR", "PTX", "STR_PTX"]

EXP_TO_CONDITION = {
    "_str": "STR",
    "_ptx": "PTX",
    "_ptx_str": "STR_PTX",
}

RF_SCALAR_FEATURES = [
    "gaussian_sigma_x", "gaussian_sigma_y", "gaussian_amp", "gaussian_r2",
    "dog_sigma_exc", "dog_sigma_inh", "dog_amp_exc", "dog_amp_inh", "dog_r2",
    "lnl_a_norm", "lnl_bits_per_spike", "lnl_r_squared",
    "lnl_rectification_index", "lnl_nonlinearity_index", "lnl_threshold_g",
]

STEP_TRACE_NAME = "step_up_5s_5i_b0_3x"

FIG_DIR = FIG_DIR_BASE / "receptive_field" / "compare_all"
FIG_ALL_DIR = FIG_DIR / "all_cells"
FIG_SUB_DIR = FIG_DIR / "per_subtype"

MIN_CELLS_PLOT = 5


# =====================================================================
# Step-trace helpers (adapted from spatial_plots_per_cluster.py)
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


def _collect_traces(trace_series):
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
    """Load both before_ and after_ step traces from all 3 source parquets.

    Returns a DataFrame with columns:
        before_dataset_id, subtype, source_experiment,
        before_{STEP_TRACE_NAME}, after_{STEP_TRACE_NAME}
    """
    before_col = f"before_{STEP_TRACE_NAME}"
    after_col = f"after_{STEP_TRACE_NAME}"
    frames = []
    for exp, path in SOURCE_PARQUETS.items():
        if not path.exists():
            logger.warning("  Missing source parquet: %s", path)
            continue
        logger.info("  Loading traces from %s ...", path.name)
        want = [before_col, after_col, "before_dataset_id", "subtype"]
        schema_names = set(pq.read_schema(path).names)
        cols_to_load = [c for c in want if c in schema_names]
        df = pd.read_parquet(path, columns=cols_to_load)
        df["source_experiment"] = exp
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _precompute_traces(trace_df, valid_ids=None):
    """Pre-compute per-condition per-subtype mean/std traces.

    Returns nested dict:  {condition: {subtype: (mean_1d, std_1d, n_cells)}}
    Also returns an "all_cells" key per condition (pooled across subtypes).
    """
    before_col = f"before_{STEP_TRACE_NAME}"
    after_col = f"after_{STEP_TRACE_NAME}"

    if valid_ids is not None and "before_dataset_id" in trace_df.columns:
        trace_df = trace_df[trace_df["before_dataset_id"].isin(valid_ids)].copy()

    result = {}

    for cond in CONDITIONS:
        result[cond] = {}
        if cond == "before":
            col = before_col
            sub = trace_df
        else:
            exp = [e for e, c in EXP_TO_CONDITION.items() if c == cond][0]
            col = after_col
            sub = trace_df[trace_df["source_experiment"] == exp]

        if col not in sub.columns:
            continue

        stacked_all = _collect_traces(sub[col])
        if stacked_all is not None and stacked_all.shape[0] >= 1:
            result[cond]["all_cells"] = (
                stacked_all.mean(axis=0),
                stacked_all.std(axis=0),
                stacked_all.shape[0],
            )

        for st in sub["subtype"].dropna().unique():
            if not st or st == "":
                continue
            st_rows = sub[sub["subtype"] == st]
            stacked = _collect_traces(st_rows[col])
            if stacked is not None and stacked.shape[0] >= 1:
                result[cond][st] = (
                    stacked.mean(axis=0),
                    stacked.std(axis=0),
                    stacked.shape[0],
                )
    return result


# =====================================================================
# Panel plotting helpers
# =====================================================================

def _plot_raw_hexbin(ax, x, y, c, gridsize, mincnt, vmin=None, vmax=None):
    """Raw mean-value hexbin panel. Returns the PolyCollection."""
    if len(c) < 2:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_aspect("equal")
        ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
        return None
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP,
        vmin=vmin, vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
    return hb


def _plot_gam_hexbin(ax, x, y, c, gridsize, mincnt, n_splines):
    """GAM-smoothed hexbin panel. Returns (gam_model, offsets)."""
    if len(c) < 10:
        ax.text(0.5, 0.5, "Too few cells", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_aspect("equal")
        ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
        return None, None
    gam = _fit_gam(x, y, c, n_splines)
    if gam is not None:
        hb_g = ax.hexbin(
            x, y, gridsize=gridsize,
            extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP,
        )
        offsets = hb_g.get_offsets()
        if len(offsets) > 0:
            preds = gam.predict(offsets)
            hb_g.set_array(preds)
            p_mean = float(np.mean(preds))
            p_std = max(float(np.std(preds)), 1e-6)
            hb_g.set_clim(vmin=p_mean - 5 * p_std, vmax=p_mean + 5 * p_std)
        plt.colorbar(hb_g, ax=ax, shrink=0.5, pad=0.02)
    else:
        ax.text(0.5, 0.5, "GAM failed", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
    return gam, None


def _plot_gam_projected(ax, gam, x_all, y_all, gridsize, mincnt):
    """Predict subtype GAM on the all-cells hexbin grid."""
    if gam is None or len(x_all) < 2:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_aspect("equal")
        ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
        return
    hb = ax.hexbin(
        x_all, y_all, gridsize=gridsize,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP,
    )
    offsets = hb.get_offsets()
    if len(offsets) > 0:
        preds = gam.predict(offsets)
        hb.set_array(preds)
        p_mean = float(np.mean(preds))
        p_std = max(float(np.std(preds)), 1e-6)
        hb.set_clim(vmin=p_mean - 5 * p_std, vmax=p_mean + 5 * p_std)
    plt.colorbar(hb, ax=ax, shrink=0.5, pad=0.02)
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)


def _plot_density_hexbin(ax, x, y, gridsize, mincnt):
    """Cell-count hexbin (no C argument)."""
    if len(x) < 2:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_aspect("equal")
        ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
        return
    hb = ax.hexbin(
        x, y, gridsize=gridsize,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap="viridis",
    )
    plt.colorbar(hb, ax=ax, shrink=0.5, pad=0.02)
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)


def _plot_step_trace(ax, trace_info, cond_label):
    """Plot mean step trace +/- std shading."""
    if trace_info is None:
        ax.text(0.5, 0.5, "No trace", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_title(cond_label, fontsize=9)
        return
    mean_t, std_t, n_cells = trace_info
    t = np.arange(len(mean_t)) / SAMPLING_RATE
    ax.fill_between(t, mean_t - std_t, mean_t + std_t, color="gray", alpha=0.2)
    ax.plot(t, mean_t, linewidth=1.0)
    ax.set_title(f"{cond_label} (n={n_cells})", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Hz", fontsize=8)
    ax.tick_params(labelsize=7)


# =====================================================================
# Phase 1 -- All-cells compare figures
# =====================================================================

def _generate_all_cells_figures(df, available_features, trace_dict,
                                allcell_coords, metrics_rows):
    """One figure per RF feature: 5 rows x 4 cols gridspec."""
    logger.info("\n=== Phase 1: All-cells compare figures ===")

    for fi, feat in enumerate(available_features):
        fig = plt.figure(figsize=(20, 26))
        gs = fig.add_gridspec(
            5, 4, hspace=0.35, wspace=0.30,
            height_ratios=[1, 1, 1, 1, 0.6],
        )

        for ri, cond in enumerate(CONDITIONS):
            cond_df = df[df["condition"] == cond]
            data = cond_df[[X_COL, Y_COL, feat]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            x = data[X_COL].to_numpy() * COORD_SCALE
            y = data[Y_COL].to_numpy() * COORD_SCALE
            c = data[feat].to_numpy()

            # Raw hexbin (cols 0-1)
            ax_raw = fig.add_subplot(gs[ri, 0:2])
            hb = _plot_raw_hexbin(ax_raw, x, y, c, GRIDSIZE_ALL, MINCNT_ALL)
            ax_raw.set_title(f"Raw Hexbin (n={len(c)})", fontsize=9)
            ax_raw.set_ylabel(cond, fontsize=11, fontweight="bold")
            if hb is not None:
                plt.colorbar(hb, ax=ax_raw, shrink=0.5, pad=0.02)

            # GAM (cols 2-3)
            ax_gam = fig.add_subplot(gs[ri, 2:4])
            _plot_gam_hexbin(ax_gam, x, y, c, GRIDSIZE_ALL, MINCNT_ALL, N_SPLINES_ALL)
            ax_gam.set_title("GAM Smoothed", fontsize=9)

            # Metrics
            if len(c) >= 10:
                centers, raw_means, counts = extract_hexbin_data(
                    x, y, c, GRIDSIZE_ALL, MINCNT_ALL,
                )
                m = compute_metrics(x, y, c, centers, raw_means)
                m["scope"] = "all_cells"
                m["condition"] = cond
                m["feature"] = feat
                metrics_rows.append(m)

        # Row 4: step traces
        if ri == 0:
            pass
        for ci, cond in enumerate(CONDITIONS):
            ax_step = fig.add_subplot(gs[4, ci])
            t_info = trace_dict.get(cond, {}).get("all_cells")
            _plot_step_trace(ax_step, t_info, cond)

        fig.suptitle(f"{feat}", fontsize=13, y=0.99)
        save_path = FIG_ALL_DIR / f"Hexbin_{feat}.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            "  [%d/%d] %s  -> %s",
            fi + 1, len(available_features), feat, save_path.name,
        )


# =====================================================================
# Phase 2 -- Per-subtype compare figures
# =====================================================================

def _generate_per_subtype_figures(df, available_features, trace_dict,
                                  allcell_coords, metrics_rows):
    """One figure per (subtype x feature): 5 rows x 4 cols gridspec."""
    logger.info("\n=== Phase 2: Per-subtype compare figures ===")

    subtypes = sorted(
        s for s in df["subtype"].dropna().unique() if s != ""
    )
    logger.info("  Subtypes to process: %d", len(subtypes))

    for si, subtype in enumerate(subtypes):
        sub_df = df[df["subtype"] == subtype]
        total_cells = len(sub_df)
        if total_cells < MIN_CELLS_PLOT:
            logger.info(
                "  [%d/%d] %s -- skipped (n=%d)",
                si + 1, len(subtypes), subtype, total_cells,
            )
            continue

        n_feats_done = 0
        for fi, feat in enumerate(available_features):
            fig = plt.figure(figsize=(24, 26))
            gs = fig.add_gridspec(
                5, 4, hspace=0.35, wspace=0.35,
                height_ratios=[1, 1, 1, 1, 0.6],
            )

            # Column headers on the first row
            col_titles = ["Raw Hexbin", "GAM (subtype)", "GAM (projected)", "Cell Density"]

            for ri, cond in enumerate(CONDITIONS):
                cond_sub = sub_df[sub_df["condition"] == cond]

                # Feature-valid data
                data = cond_sub[[X_COL, Y_COL, feat]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                x = data[X_COL].to_numpy() * COORD_SCALE
                y = data[Y_COL].to_numpy() * COORD_SCALE
                c = data[feat].to_numpy()

                # All coords for density (regardless of feature validity)
                xy_full = cond_sub[[X_COL, Y_COL]].dropna()
                x_dens = xy_full[X_COL].to_numpy() * COORD_SCALE
                y_dens = xy_full[Y_COL].to_numpy() * COORD_SCALE

                # All-cells coords for projection grid
                x_proj, y_proj = allcell_coords.get(cond, (np.array([]), np.array([])))

                # Col 0: raw hexbin
                ax0 = fig.add_subplot(gs[ri, 0])
                hb = _plot_raw_hexbin(ax0, x, y, c, GRIDSIZE_GRP, MINCNT_GRP)
                ax0.set_ylabel(cond, fontsize=11, fontweight="bold")
                if hb is not None:
                    plt.colorbar(hb, ax=ax0, shrink=0.5, pad=0.02)

                # Col 1: GAM (subtype)
                ax1 = fig.add_subplot(gs[ri, 1])
                gam, _ = _plot_gam_hexbin(ax1, x, y, c, GRIDSIZE_GRP, MINCNT_GRP, N_SPLINES_GRP)

                # Col 2: GAM projected onto all-cells grid
                ax2 = fig.add_subplot(gs[ri, 2])
                _plot_gam_projected(ax2, gam, x_proj, y_proj, GRIDSIZE_ALL, MINCNT_ALL)

                # Col 3: cell density
                ax3 = fig.add_subplot(gs[ri, 3])
                _plot_density_hexbin(ax3, x_dens, y_dens, GRIDSIZE_GRP, MINCNT_GRP)

                if ri == 0:
                    ax0.set_title(f"{col_titles[0]} (n={len(c)})", fontsize=9)
                    ax1.set_title(col_titles[1], fontsize=9)
                    ax2.set_title(col_titles[2], fontsize=9)
                    ax3.set_title(f"{col_titles[3]} (n={len(x_dens)})", fontsize=9)
                else:
                    ax0.set_title(f"n={len(c)}", fontsize=8)
                    ax3.set_title(f"n={len(x_dens)}", fontsize=8)

                # Metrics
                if len(c) >= 10:
                    centers, raw_means, counts = extract_hexbin_data(
                        x, y, c, GRIDSIZE_GRP, MINCNT_GRP,
                    )
                    m = compute_metrics(x, y, c, centers, raw_means)
                    m["scope"] = subtype
                    m["condition"] = cond
                    m["feature"] = feat
                    metrics_rows.append(m)

            # Row 4: step traces
            for ci, cond in enumerate(CONDITIONS):
                ax_step = fig.add_subplot(gs[4, ci])
                t_info = trace_dict.get(cond, {}).get(subtype)
                _plot_step_trace(ax_step, t_info, cond)

            fig.suptitle(f"[{subtype}] {feat}", fontsize=13, y=0.99)
            save_path = FIG_SUB_DIR / f"Hexbin_{subtype}_{feat}.png"
            fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
            plt.close(fig)
            n_feats_done += 1

        logger.info(
            "  [%d/%d] %s: %d cells, %d feature figs",
            si + 1, len(subtypes), subtype, total_cells, n_feats_done,
        )


# =====================================================================
# Main
# =====================================================================

def main():
    for d in [OUTPUT_DIR, FIG_ALL_DIR, FIG_SUB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Load combined RF compare data ---
    input_path = OUTPUT_DIR / "combined_rf_compare.parquet"
    if not input_path.exists():
        logger.error("Input not found: %s\nRun prepare_rf_compare_data.py first.", input_path)
        return

    logger.info("Loading %s ...", input_path.name)
    df = pd.read_parquet(input_path)
    logger.info("  Shape: %s", df.shape)

    available_features = [f for f in RF_SCALAR_FEATURES if f in df.columns]
    logger.info("  Available RF features: %d", len(available_features))
    if not available_features:
        logger.error("No RF features found -- nothing to plot.")
        return

    for cond in CONDITIONS:
        n = (df["condition"] == cond).sum()
        logger.info("  %s: %d cells", cond, n)

    # --- Pre-compute all-cells coordinates per condition ---
    logger.info("\n=== Pre-computing all-cells hexbin grids ===")
    allcell_coords = {}
    for cond in CONDITIONS:
        cond_df = df[df["condition"] == cond]
        xy = cond_df[[X_COL, Y_COL]].dropna()
        x_all = xy[X_COL].to_numpy() * COORD_SCALE
        y_all = xy[Y_COL].to_numpy() * COORD_SCALE
        allcell_coords[cond] = (x_all, y_all)
        logger.info("  %s: %d cells with coords", cond, len(x_all))

    # --- Load step traces ---
    logger.info("\n=== Loading step-up traces ===")
    trace_df = _load_step_traces()
    if len(trace_df) > 0:
        valid_ids = set(df["before_dataset_id"].dropna().unique())
        trace_dict = _precompute_traces(trace_df, valid_ids)
        for cond in CONDITIONS:
            n_keys = len(trace_dict.get(cond, {}))
            logger.info("  %s: %d scope entries", cond, n_keys)
        del trace_df
    else:
        logger.warning("  No trace data loaded")
        trace_dict = {c: {} for c in CONDITIONS}

    # --- Generate figures ---
    metrics_rows = []

    _generate_all_cells_figures(
        df, available_features, trace_dict, allcell_coords, metrics_rows,
    )

    _generate_per_subtype_figures(
        df, available_features, trace_dict, allcell_coords, metrics_rows,
    )

    # --- Save metrics ---
    if metrics_rows:
        met_path = OUTPUT_DIR / "rf_compare_metrics.parquet"
        pd.DataFrame(metrics_rows).to_parquet(met_path, index=False)
        logger.info("\nSaved metrics: %s (%d rows)", met_path, len(metrics_rows))

    logger.info("Done.")


if __name__ == "__main__":
    main()
