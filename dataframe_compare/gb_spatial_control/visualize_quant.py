"""
Step 5: Visualize Spatial Quantification Results (Single-Condition)
==================================================================
Creates single-condition figures from quantification results for
GB control data.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import OUTPUT_DIR, FIG_DIR_BASE, categorize, CAT_COLORS, short

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "spatial" / "figures_quant"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    combined_path = OUTPUT_DIR / "spatial_quant_combined.parquet"
    if not combined_path.exists():
        logger.error(f"Missing: {combined_path}")
        return

    logger.info("Loading data ...")
    df = pd.read_parquet(combined_path)
    logger.info(f"  Shape: {df.shape}")

    ac = df[df["scope"] == "all_cells"].copy().set_index("feature")
    feats = sorted(ac.index)
    logger.info(f"  All-cells features: {len(feats)}")

    if len(feats) == 0:
        logger.warning("No features")
        return

    cats = [categorize(f) for f in feats]
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]

    # ---- Fig 1: Gradient polar plot ----
    logger.info("Fig 1: Gradient polar ...")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    theta = np.radians(ac.loc[feats, "grad_dir_deg"].to_numpy())
    r_vals = ac.loc[feats, "plane_r2"].to_numpy()
    for t, rv, cat in zip(theta, r_vals, cats):
        ax.scatter(t, rv, c=CAT_COLORS.get(cat, "#757575"), s=50, alpha=0.7, edgecolors="k", linewidth=0.3)
    ax.set_title("Gradient Direction & Strength (Plane R2)", fontsize=13, pad=20)
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.3, 1.0))
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "gradient_polar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: Plane R2 vs GAM R2 scatter ----
    logger.info("Fig 2: Plane R2 vs GAM R2 ...")
    fig, ax = plt.subplots(figsize=(8, 8))
    p_r2 = ac.loc[feats, "plane_r2"].to_numpy()
    g_r2 = ac.loc[feats, "gam_r2"].to_numpy()
    for f, pr, gr, cat in zip(feats, p_r2, g_r2, cats):
        ax.scatter(pr, gr, c=CAT_COLORS.get(cat, "#757575"), s=50, alpha=0.7, edgecolors="k", linewidth=0.3)
        ax.annotate(short(f), (pr, gr), fontsize=5, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")
    max_val = max(np.nanmax(p_r2), np.nanmax(g_r2)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, zorder=1)
    ax.set_xlabel("Plane R2 (linear gradient)", fontsize=12)
    ax.set_ylabel("GAM R2 (nonlinear)", fontsize=12)
    ax.set_title("Linear vs Nonlinear Spatial Structure", fontsize=13)
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "plane_vs_gam_r2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: Moran's I bar chart ----
    logger.info("Fig 3: Moran's I bar chart ...")
    fig, ax = plt.subplots(figsize=(14, 6))
    moran = ac.loc[feats, "moran_i"].to_numpy()
    x_pos = np.arange(len(feats))
    bar_colors = [CAT_COLORS.get(cat, "#757575") for cat in cats]
    ax.bar(x_pos, moran, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short(f) for f in feats], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title("Spatial Autocorrelation (GB Control)", fontsize=13)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "moran_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: Radial correlation bar chart ----
    logger.info("Fig 4: Radial correlation bar chart ...")
    fig, ax = plt.subplots(figsize=(14, 6))
    rad_r = ac.loc[feats, "radial_r"].to_numpy()
    ax.bar(x_pos, rad_r, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    if "radial_r_lo" in ac.columns and "radial_r_hi" in ac.columns:
        lo = ac.loc[feats, "radial_r_lo"].to_numpy()
        hi = ac.loc[feats, "radial_r_hi"].to_numpy()
        err_lo = rad_r - lo
        err_hi = hi - rad_r
        ax.errorbar(x_pos, rad_r, yerr=[err_lo, err_hi],
                     fmt="none", ecolor="gray", elinewidth=0.8, capsize=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short(f) for f in feats], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Radial r", fontsize=12)
    ax.set_title("Radial Correlation from ONH (GB Control)", fontsize=13)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "radial_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: Gradient magnitude ranked ----
    logger.info("Fig 5: Gradient magnitude ranked ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    grad_mag = ac.loc[feats, "grad_mag"].dropna().sort_values(ascending=True)
    grad_cats = [categorize(f) for f in grad_mag.index]
    grad_colors = [CAT_COLORS.get(cat, "#757575") for cat in grad_cats]
    ax.barh(range(len(grad_mag)), grad_mag.values, color=grad_colors, alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(grad_mag)))
    ax.set_yticklabels([short(f) for f in grad_mag.index], fontsize=7)
    ax.set_xlabel("Gradient Magnitude", fontsize=12)
    ax.set_title("Spatial Gradient Strength (GB Control)", fontsize=13)
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "gradient_magnitude_ranked.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 6: Summary dashboard ----
    logger.info("Fig 6: Summary dashboard ...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for pi, (metric, ylabel) in enumerate([
        ("plane_r2", "Plane R2"),
        ("moran_i", "Moran's I"),
        ("radial_r", "Radial r"),
    ]):
        ax_bar = axes[0, pi]
        vals = ac.loc[feats, metric].to_numpy()
        ax_bar.bar(range(len(feats)), vals, color=bar_colors, alpha=0.8,
                   edgecolor="k", linewidth=0.2)
        ax_bar.set_xticks(range(len(feats)))
        ax_bar.set_xticklabels([short(f) for f in feats], rotation=55, ha="right", fontsize=6)
        ax_bar.set_ylabel(ylabel)
        ax_bar.set_title(ylabel)
        ax_bar.axhline(0, color="k", linewidth=0.5)

    for pi, (metric, ylabel) in enumerate([
        ("grad_mag", "Gradient Mag"),
        ("gam_r2", "GAM R2"),
        ("gam_dynamic_range", "GAM Range"),
    ]):
        ax_bar = axes[1, pi]
        if metric in ac.columns:
            vals = ac.loc[feats, metric].to_numpy()
            ax_bar.bar(range(len(feats)), vals, color=bar_colors, alpha=0.8,
                       edgecolor="k", linewidth=0.2)
            ax_bar.set_xticks(range(len(feats)))
            ax_bar.set_xticklabels([short(f) for f in feats], rotation=55, ha="right", fontsize=6)
            ax_bar.set_ylabel(ylabel)
            ax_bar.set_title(ylabel)

    fig.suptitle("GB Control Spatial Analysis: Summary Dashboard", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "summary_dashboard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Done.")


if __name__ == "__main__":
    main()
