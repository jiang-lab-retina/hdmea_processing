"""
DS Bar Plots -- Condition effects on DSI and DS p-value
=======================================================
Grouped bar charts showing mean +/- SEM for each blocker condition
(before, STR, PTX, STR_PTX) across three scopes:

  1. All cells
  2. DSGC group (pooled)
  3. Each DSGC cluster (DSGC_0 .. DSGC_8)

Produces one figure per feature (dsi, ds_p_value).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import OUTPUT_DIR, FIG_DIR_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "direction_selectivity" / "bar_plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["before", "STR", "PTX", "STR_PTX"]
COND_COLORS = {
    "before":  "#4A90D9",
    "STR":     "#E67E22",
    "PTX":     "#27AE60",
    "STR_PTX": "#C0392B",
}

FEATURES = ["dsi", "ds_p_value", "osi", "os_p_value"]
FEATURE_LABELS = {
    "dsi": "Direction Selectivity Index (DSI)",
    "ds_p_value": "DS p-value",
    "osi": "Orientation Selectivity Index (OSI)",
    "os_p_value": "OS p-value",
}


def _add_significance(ax, x1, x2, y, h, p_val):
    """Draw a bracket with significance stars between two bars."""
    if p_val < 0.001:
        txt = "***"
    elif p_val < 0.01:
        txt = "**"
    elif p_val < 0.05:
        txt = "*"
    else:
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="k")
    ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=8)


def _plot_bars_for_scope(ax, df, feat, scope_label):
    """Draw grouped bars for one subplot (one scope)."""
    bar_width = 0.7
    x_positions = np.arange(len(CONDITIONS))
    means = []
    sems = []
    ns = []

    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, feat].dropna()
        n = len(vals)
        ns.append(n)
        if n > 0:
            means.append(vals.mean())
            sems.append(vals.sem())
        else:
            means.append(0)
            sems.append(0)

    bars = ax.bar(
        x_positions, means, bar_width,
        yerr=sems, capsize=3, ecolor="gray",
        color=[COND_COLORS[c] for c in CONDITIONS],
        edgecolor="white", linewidth=0.5, alpha=0.85,
    )

    # significance tests: compare each after condition to before
    before_vals = df.loc[df["condition"] == "before", feat].dropna()
    if len(before_vals) >= 3:
        y_max = max(m + s for m, s in zip(means, sems)) if max(means) > 0 else 0.1
        step = y_max * 0.08
        offset = y_max * 0.04

        for i, cond in enumerate(CONDITIONS[1:], start=1):
            after_vals = df.loc[df["condition"] == cond, feat].dropna()
            if len(after_vals) >= 3:
                _, p = stats.mannwhitneyu(
                    before_vals, after_vals, alternative="two-sided"
                )
                bracket_y = y_max + offset + step * (i - 1)
                _add_significance(ax, 0, i, bracket_y, step * 0.4, p)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(CONDITIONS, ns)],
        fontsize=7,
    )
    ax.set_title(scope_label, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    input_path = OUTPUT_DIR / "combined_ds_compare.parquet"
    if not input_path.exists():
        logger.error(
            "Input not found: %s\nRun prepare_ds_compare_data.py first.",
            input_path,
        )
        return

    logger.info("Loading %s ...", input_path.name)
    df = pd.read_parquet(input_path)
    logger.info("  Shape: %s", df.shape)

    dsgc_clusters = sorted(
        s for s in df["subtype"].dropna().unique() if s.startswith("DSGC")
    )
    osgc_clusters = sorted(
        s for s in df["subtype"].dropna().unique() if s.startswith("OSGC")
    )
    logger.info("  DSGC clusters: %s", dsgc_clusters)
    logger.info("  OSGC clusters: %s", osgc_clusters)

    FEAT_GROUP = {
        "dsi":        ("DSGC", dsgc_clusters),
        "ds_p_value": ("DSGC", dsgc_clusters),
        "osi":        ("OSGC", osgc_clusters),
        "os_p_value": ("OSGC", osgc_clusters),
    }

    for feat in FEATURES:
        group_name, clusters = FEAT_GROUP[feat]

        scopes = [
            ("All cells", df),
            (f"{group_name} (group)", df[df["group"] == group_name]),
        ]
        for cl in clusters:
            scopes.append((cl, df[df["subtype"] == cl]))

        n_scopes = len(scopes)
        n_cols = 4
        n_rows = int(np.ceil(n_scopes / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 4.0 * n_rows),
            squeeze=False,
        )

        for idx, (scope_label, scope_df) in enumerate(scopes):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            _plot_bars_for_scope(ax, scope_df, feat, scope_label)
            if col == 0:
                ax.set_ylabel(FEATURE_LABELS[feat], fontsize=9)

        for idx in range(n_scopes, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"Blocker effects on {FEATURE_LABELS[feat]}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        save_path = FIG_DIR / f"bar_{feat}_conditions.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: %s", save_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
