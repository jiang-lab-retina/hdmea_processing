"""
Mean ipRGC Trace per Subtype
============================
Loads the after_iprgc_test firing-rate traces from all 3 source parquets,
filters to ipRGC cells, and plots the trial-averaged mean +/- std trace
for each ipRGC subtype in a grid layout.  An "All ipRGC" summary panel is
included.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import SOURCE_PARQUETS, FIG_DIR_BASE, SAMPLING_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IPRGC_TRACE_COL = "after_iprgc_test"
FIG_DIR = FIG_DIR_BASE / "iprgc"

SUBTYPE_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#264653",
    "#E76F51", "#6A4C93", "#1982C4", "#8AC926", "#FF595E",
    "#6D6875", "#B5838D", "#FFCDB2", "#4361EE", "#3A0CA3",
]

MIN_CELLS = 3


def _trials_to_mean_trace(cell_data):
    """Average across trials for a single cell -> 1-D array or None."""
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
    """Stack trial-averaged traces for a group of cells -> (n_cells, T) or None."""
    traces = []
    for val in trace_series:
        mt = _trials_to_mean_trace(val)
        if mt is not None and len(mt) > 0:
            traces.append(mt)
    if len(traces) == 0:
        return None
    min_len = min(len(t) for t in traces)
    return np.vstack([t[:min_len] for t in traces])


def _load_iprgc_data():
    """Load after_iprgc_test traces and subtype labels for ipRGC cells."""
    want_cols = [IPRGC_TRACE_COL, "group", "subtype"]
    frames = []
    for exp, path in SOURCE_PARQUETS.items():
        if not path.exists():
            logger.warning("Missing source parquet: %s", path)
            continue
        logger.info("Loading %s ...", path.name)
        schema_names = set(pq.read_schema(path).names)
        cols = [c for c in want_cols if c in schema_names]
        if IPRGC_TRACE_COL not in cols:
            logger.warning("  %s not found in %s -- skipping", IPRGC_TRACE_COL, path.name)
            continue
        df = pd.read_parquet(path, columns=cols)
        df["source_experiment"] = exp
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    iprgc = combined[combined["group"] == "ipRGC"].copy()
    logger.info("Total ipRGC cells: %d", len(iprgc))
    return iprgc


def _compute_subtype_traces(df):
    """Compute mean/std trace per ipRGC subtype.

    Returns dict:  {subtype_name: (mean_1d, std_1d, n_cells)}
    Also returns "All ipRGC" pooled entry.
    """
    result = {}

    all_stacked = _collect_traces(df[IPRGC_TRACE_COL])
    if all_stacked is not None and all_stacked.shape[0] >= 1:
        result["All ipRGC"] = (
            all_stacked.mean(axis=0),
            all_stacked.std(axis=0),
            all_stacked.shape[0],
        )

    subtypes = sorted(
        s for s in df["subtype"].dropna().unique() if s and str(s).startswith("ipRGC")
    )
    for st in subtypes:
        st_rows = df[df["subtype"] == st]
        stacked = _collect_traces(st_rows[IPRGC_TRACE_COL])
        if stacked is not None and stacked.shape[0] >= MIN_CELLS:
            result[st] = (
                stacked.mean(axis=0),
                stacked.std(axis=0),
                stacked.shape[0],
            )
    return result


def _plot_trace(ax, mean_t, std_t, n_cells, label, color):
    """Plot mean trace with std shading."""
    t = np.arange(len(mean_t)) / SAMPLING_RATE
    ax.fill_between(t, mean_t - std_t, mean_t + std_t, color=color, alpha=0.18)
    ax.plot(t, mean_t, linewidth=1.2, color=color)
    ax.set_title(f"{label}  (n={n_cells})", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Firing rate (Hz)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "src").mkdir(exist_ok=True)

    logger.info("=== Loading ipRGC traces ===")
    df = _load_iprgc_data()
    if len(df) == 0:
        logger.error("No ipRGC data loaded.")
        return

    logger.info("=== Computing per-subtype traces ===")
    traces = _compute_subtype_traces(df)
    if not traces:
        logger.error("No valid traces computed.")
        return

    for name, (_, _, n) in traces.items():
        logger.info("  %s: %d cells", name, n)

    keys = list(traces.keys())
    n_panels = len(keys)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharey=True,
        squeeze=False,
    )

    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mean_t, std_t, n_cells = traces[key]
        color = "#333333" if key == "All ipRGC" else SUBTYPE_COLORS[idx % len(SUBTYPE_COLORS)]
        _plot_trace(ax, mean_t, std_t, n_cells, key, color)

    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Mean ipRGC Trace per Subtype (after-blocker)", fontsize=13, y=1.01)
    fig.tight_layout()

    save_path = FIG_DIR / "mean_iprgc_trace_per_subtype.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)

    # --- Overlay figure: all subtypes on one axes ---
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    color_idx = 0
    for key in keys:
        if key == "All ipRGC":
            continue
        mean_t, std_t, n_cells = traces[key]
        t = np.arange(len(mean_t)) / SAMPLING_RATE
        c = SUBTYPE_COLORS[color_idx % len(SUBTYPE_COLORS)]
        ax2.fill_between(t, mean_t - std_t, mean_t + std_t, color=c, alpha=0.10)
        ax2.plot(t, mean_t, linewidth=1.2, color=c, label=f"{key} (n={n_cells})")
        color_idx += 1

    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Firing rate (Hz)", fontsize=10)
    ax2.set_title("ipRGC Subtypes -- Mean Trace Overlay", fontsize=12)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=8)
    fig2.tight_layout()

    save_path2 = FIG_DIR / "mean_iprgc_trace_overlay.png"
    fig2.savefig(str(save_path2), dpi=200, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved: %s", save_path2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
