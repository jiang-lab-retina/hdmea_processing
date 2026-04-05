"""
Step 5: Visualize Spatial Quantification Results
================================================
Creates comparison visualizations from quantification results.

Adapted from visualize_spatial_quant.py.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

from compare_config import OUTPUT_DIR, FIG_DIR_BASE

FIG_DIR = FIG_DIR_BASE / "spatial" / "figures_quant"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FDR = 0.05


def short(f):
    return (f.replace("green_blue_", "gb_")
             .replace("_extreme", "")
             .replace("_ratio", "_r")
             .replace("_high", "_H")
             .replace("_sustained", "_sus"))


def categorize(f):
    if "green_blue" in f or "gb_" in f:
        return "Chromatic"
    if "dsi" in f or "osi" in f:
        return "DS/OS"
    if "iprgc" in f:
        return "ipRGC"
    if "step_up" in f or "on_" in f or "off_" in f:
        return "Step response"
    return "Other"


def main():
    combined_path = OUTPUT_DIR / "spatial_quant_combined.parquet"
    if not combined_path.exists():
        logger.error(f"Missing: {combined_path}")
        return

    logger.info("Loading data ...")
    df = pd.read_parquet(combined_path)
    logger.info(f"  Shape: {df.shape}")

    # All-cells data
    ac = df[df["scope"] == "all_cells"].copy()
    before = ac[ac["condition"] == "before"].set_index("feature")
    after = ac[ac["condition"] == "after"].set_index("feature")
    delta = ac[ac["condition"] == "delta"].set_index("feature")

    common_feats = sorted(before.index.intersection(after.index))
    logger.info(f"  Common features: {len(common_feats)}")

    if len(common_feats) == 0:
        logger.warning("No common features between before and after")
        return

    # ---- Fig 1: Gradient R2 scatter (before vs after) ----
    logger.info("Fig 1: Gradient R2 scatter ...")
    fig, ax = plt.subplots(figsize=(8, 8))
    b_r2 = before.loc[common_feats, "plane_r2"].to_numpy()
    a_r2 = after.loc[common_feats, "plane_r2"].to_numpy()
    colors = [categorize(f) for f in common_feats]
    cat_colors = {"Chromatic": "green", "DS/OS": "blue", "ipRGC": "red",
                  "Step response": "orange", "Other": "gray"}
    for f, bv, av, cat in zip(common_feats, b_r2, a_r2, colors):
        ax.scatter(bv, av, c=cat_colors.get(cat, "gray"), s=40, alpha=0.7, zorder=3)
    max_r2 = max(np.nanmax(b_r2), np.nanmax(a_r2)) * 1.1
    ax.plot([0, max_r2], [0, max_r2], "k--", alpha=0.3, zorder=1)
    ax.set_xlabel("Before Blocker - Plane R2", fontsize=12)
    ax.set_ylabel("After Blocker - Plane R2", fontsize=12)
    ax.set_title("Spatial Gradient Strength: Before vs After", fontsize=13)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in cat_colors.items()]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "gradient_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: Moran's I comparison bar chart ----
    logger.info("Fig 2: Moran's I comparison ...")
    fig, ax = plt.subplots(figsize=(14, 6))
    n_feat = len(common_feats)
    x_pos = np.arange(n_feat)
    width = 0.35
    b_moran = before.loc[common_feats, "moran_i"].to_numpy()
    a_moran = after.loc[common_feats, "moran_i"].to_numpy()
    ax.bar(x_pos - width / 2, b_moran, width, label="Before", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width / 2, a_moran, width, label="After", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short(f) for f in common_feats], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title("Spatial Autocorrelation: Before vs After Blocker", fontsize=13)
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "moran_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: Radial r comparison ----
    logger.info("Fig 3: Radial r comparison ...")
    fig, ax = plt.subplots(figsize=(14, 6))
    b_rad = before.loc[common_feats, "radial_r"].to_numpy()
    a_rad = after.loc[common_feats, "radial_r"].to_numpy()
    ax.bar(x_pos - width / 2, b_rad, width, label="Before", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width / 2, a_rad, width, label="After", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short(f) for f in common_feats], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Radial r", fontsize=12)
    ax.set_title("Radial Correlation: Before vs After Blocker", fontsize=13)
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "radial_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: Gradient polar plot (before vs after) ----
    logger.info("Fig 4: Gradient polar ...")
    fig = plt.figure(figsize=(12, 6))
    for pi, (cond_df, cond_label) in enumerate([(before, "Before"), (after, "After")]):
        ax = fig.add_subplot(1, 2, pi + 1, projection="polar")
        cond_common = cond_df.loc[cond_df.index.isin(common_feats)]
        theta = np.radians(cond_common["grad_dir_deg"].to_numpy())
        r_vals = cond_common["plane_r2"].to_numpy()
        cats = [categorize(f) for f in cond_common.index]
        for t, rv, cat in zip(theta, r_vals, cats):
            ax.scatter(t, rv, c=cat_colors.get(cat, "gray"), s=40, alpha=0.7)
        ax.set_title(f"{cond_label} Blocker", fontsize=12, pad=15)
    fig.suptitle("Gradient Direction & Strength", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "gradient_polar_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: Delta strength summary ----
    logger.info("Fig 5: Delta strength summary ...")
    if len(delta) > 0:
        delta_common = delta.loc[delta.index.isin(common_feats)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Delta gradient mag
        ax = axes[0]
        d_grad = delta_common["grad_mag"].dropna().sort_values(ascending=True)
        ax.barh(range(len(d_grad)), d_grad.values, color="mediumpurple", alpha=0.8)
        ax.set_yticks(range(len(d_grad)))
        ax.set_yticklabels([short(f) for f in d_grad.index], fontsize=8)
        ax.set_xlabel("Gradient Magnitude (delta)")
        ax.set_title("Delta: Spatial Gradient Strength")

        # Delta Moran
        ax = axes[1]
        d_moran = delta_common["moran_i"].dropna().sort_values(ascending=True)
        ax.barh(range(len(d_moran)), d_moran.values, color="mediumpurple", alpha=0.8)
        ax.set_yticks(range(len(d_moran)))
        ax.set_yticklabels([short(f) for f in d_moran.index], fontsize=8)
        ax.set_xlabel("Moran's I (delta)")
        ax.set_title("Delta: Spatial Clustering")

        # Delta plane R2
        ax = axes[2]
        d_r2 = delta_common["plane_r2"].dropna().sort_values(ascending=True)
        ax.barh(range(len(d_r2)), d_r2.values, color="mediumpurple", alpha=0.8)
        ax.set_yticks(range(len(d_r2)))
        ax.set_yticklabels([short(f) for f in d_r2.index], fontsize=8)
        ax.set_xlabel("Plane R2 (delta)")
        ax.set_title("Delta: Gradient Explained Variance")

        fig.suptitle("Spatially Non-Uniform Blocker Effects", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / "delta_strength_summary.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- Fig 6: Summary dashboard ----
    logger.info("Fig 6: Summary dashboard ...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for pi, (metric, ylabel) in enumerate([
        ("plane_r2", "Plane R2"),
        ("moran_i", "Moran's I"),
        ("radial_r", "Radial r"),
        ("grad_mag", "Gradient Mag"),
    ]):
        ax_scatter = axes[0, pi]
        bv = before.reindex(common_feats)[metric].to_numpy()
        av = after.reindex(common_feats)[metric].to_numpy()
        for f, b, a, cat in zip(common_feats, bv, av, colors):
            ax_scatter.scatter(b, a, c=cat_colors.get(cat, "gray"), s=30, alpha=0.7)
        lim = [min(np.nanmin(bv), np.nanmin(av)), max(np.nanmax(bv), np.nanmax(av))]
        ax_scatter.plot(lim, lim, "k--", alpha=0.3)
        ax_scatter.set_xlabel(f"Before {ylabel}")
        ax_scatter.set_ylabel(f"After {ylabel}")
        ax_scatter.set_title(ylabel)

        ax_bar = axes[1, pi]
        if metric in delta.columns:
            dv = delta.reindex(common_feats)[metric].dropna()
            dv_sorted = dv.sort_values()
            clr = ["coral" if v > 0 else "steelblue" for v in dv_sorted.values]
            ax_bar.barh(range(len(dv_sorted)), dv_sorted.values, color=clr, alpha=0.8)
            ax_bar.set_yticks(range(len(dv_sorted)))
            ax_bar.set_yticklabels([short(f) for f in dv_sorted.index], fontsize=7)
            ax_bar.set_xlabel(f"Delta {ylabel}")
            ax_bar.axvline(0, color="k", linewidth=0.5)

    fig.suptitle("Spatial Analysis: Before vs After Blocker", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "summary_dashboard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Done.")


if __name__ == "__main__":
    main()
