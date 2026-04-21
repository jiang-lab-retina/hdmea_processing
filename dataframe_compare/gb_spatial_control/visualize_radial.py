"""
Step 6: Visualize Radial Center Analysis Results (Single-Condition)
==================================================================
Creates radial center visualizations for GB control data.
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
from matplotlib.lines import Line2D

from config import OUTPUT_DIR, FIG_DIR_BASE, categorize, CAT_COLORS, short

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "spatial" / "figures_radial"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    combined_path = OUTPUT_DIR / "radial_center_combined.parquet"
    if not combined_path.exists():
        logger.error(f"Missing: {combined_path}")
        return

    logger.info("Loading data ...")
    df = pd.read_parquet(combined_path)
    logger.info(f"  Shape: {df.shape}")

    raw_ac = df[(df["data_type"] == "raw_mean") & (df["scope"] == "all_cells")].copy()
    raw_ac = raw_ac.set_index("feature")
    feats = sorted(raw_ac.index)
    logger.info(f"  All-cells features (raw_mean): {len(feats)}")

    if len(feats) == 0:
        logger.warning("No features")
        return

    cats = [categorize(f) for f in feats]
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]

    # ---- Fig 1: Optimal center map ----
    logger.info("Fig 1: Optimal center map ...")
    fig, ax = plt.subplots(figsize=(10, 10))
    for feat, cat in zip(feats, cats):
        row = raw_ac.loc[feat]
        ax.scatter(
            row["best_center_x"], row["best_center_y"],
            c=CAT_COLORS.get(cat, "#757575"),
            s=abs(row["best_r"]) * 300 + 30,
            alpha=0.7, edgecolors="k", linewidth=0.5,
        )
        ax.annotate(short(feat), (row["best_center_x"], row["best_center_y"]),
                     fontsize=5, alpha=0.6, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.scatter(0, 0, marker="+", c="red", s=200, zorder=5, linewidth=2, label="ONH (origin)")
    ax.set_xlabel("Center X (um)", fontsize=12)
    ax.set_ylabel("Center Y (um)", fontsize=12)
    ax.set_title("Optimal Radial Centers (GB Control, marker size ~ |r|)", fontsize=13)
    ax.set_aspect("equal")
    handles = legend_elements + [Line2D([0], [0], marker="+", color="red", linestyle="None",
                                         markersize=10, label="ONH (origin)")]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "radial_center_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: Origin vs optimal improvement ----
    logger.info("Fig 2: Origin vs optimal improvement ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_feats = raw_ac.loc[feats].sort_values("abs_r_improvement", ascending=True).index
    origin_r = raw_ac.loc[sorted_feats, "origin_r"].abs().to_numpy()
    best_r = raw_ac.loc[sorted_feats, "best_r"].abs().to_numpy()
    y_pos = np.arange(len(sorted_feats))
    ax.barh(y_pos, best_r, height=0.6, color="#2196F3", alpha=0.8, label="Optimal center")
    ax.barh(y_pos, origin_r, height=0.3, color="#FF5722", alpha=0.8, label="ONH origin")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short(f) for f in sorted_feats], fontsize=7)
    ax.set_xlabel("|r|", fontsize=12)
    ax.set_title("Radial Correlation: ONH Origin vs Optimal Center", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "origin_vs_optimal.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: Radial profiles for top features ----
    logger.info("Fig 3: Radial profiles (top features) ...")
    hex_path = OUTPUT_DIR / "hexbin_all_cells.parquet"
    if hex_path.exists():
        df_hex = pd.read_parquet(hex_path)
        top_feats = raw_ac.loc[feats].reindex(
            raw_ac.loc[feats, "best_r"].abs().nlargest(6).index
        ).index.tolist()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for pi, feat in enumerate(top_feats):
            ax = axes[pi // 3, pi % 3]
            sub = df_hex[(df_hex["feature"] == feat) & (df_hex["scope"] == "all_cells")]
            if len(sub) < 5:
                continue
            bx = sub["bin_x"].to_numpy()
            by = sub["bin_y"].to_numpy()
            vals = sub["raw_mean"].to_numpy()

            row = raw_ac.loc[feat]
            cx, cy = row["best_center_x"], row["best_center_y"]
            radius = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

            ax.scatter(radius, vals, s=10, alpha=0.5, c=CAT_COLORS.get(categorize(feat), "#757575"))
            order = np.argsort(radius)
            n_smooth = min(20, len(radius) // 3)
            if n_smooth > 2:
                kernel = np.ones(n_smooth) / n_smooth
                smoothed = np.convolve(vals[order], kernel, mode="valid")
                r_smooth = np.convolve(radius[order], kernel, mode="valid")
                ax.plot(r_smooth, smoothed, "k-", linewidth=2, alpha=0.8)

            ax.set_xlabel("Distance from optimal center (um)")
            ax.set_ylabel("Mean value")
            ax.set_title(f"{short(feat)} (r={row['best_r']:.3f})", fontsize=10)

        fig.suptitle("Radial Profiles: Top 6 Features", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / "radial_profiles_top.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- Fig 4: Feature category clustering ----
    logger.info("Fig 4: Feature category clustering ...")
    fig, ax = plt.subplots(figsize=(10, 10))
    for cat_name, color in CAT_COLORS.items():
        cat_feats = [f for f in feats if categorize(f) == cat_name]
        if not cat_feats:
            continue
        cx_vals = raw_ac.loc[cat_feats, "best_center_x"].to_numpy()
        cy_vals = raw_ac.loc[cat_feats, "best_center_y"].to_numpy()
        ax.scatter(cx_vals, cy_vals, c=color, s=60, alpha=0.7,
                   edgecolors="k", linewidth=0.5, label=cat_name)
        if len(cat_feats) >= 3:
            from matplotlib.patches import Ellipse
            mean_x, mean_y = np.mean(cx_vals), np.mean(cy_vals)
            std_x, std_y = np.std(cx_vals), np.std(cy_vals)
            if std_x > 0 and std_y > 0:
                ell = Ellipse((mean_x, mean_y), width=2 * std_x, height=2 * std_y,
                              fill=False, edgecolor=color, linewidth=2, linestyle="--", alpha=0.7)
                ax.add_patch(ell)

    ax.axhline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.scatter(0, 0, marker="+", c="red", s=200, zorder=5, linewidth=2)
    ax.set_xlabel("Center X (um)", fontsize=12)
    ax.set_ylabel("Center Y (um)", fontsize=12)
    ax.set_title("Optimal Centers by Feature Category (1-SD ellipses)", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "category_clustering.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: Radial dashboard ----
    logger.info("Fig 5: Radial dashboard ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 5a: |best_r| bar chart
    ax = axes[0, 0]
    abs_r = raw_ac.loc[feats, "best_r"].abs().sort_values(ascending=True)
    ac_cats = [categorize(f) for f in abs_r.index]
    ac_colors = [CAT_COLORS.get(c, "#757575") for c in ac_cats]
    ax.barh(range(len(abs_r)), abs_r.values, color=ac_colors, alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(abs_r)))
    ax.set_yticklabels([short(f) for f in abs_r.index], fontsize=6)
    ax.set_xlabel("|best_r|")
    ax.set_title("Radial Trend Strength")

    # 5b: Improvement bar chart
    ax = axes[0, 1]
    imp = raw_ac.loc[feats, "abs_r_improvement"].sort_values(ascending=True)
    imp_cats = [categorize(f) for f in imp.index]
    imp_colors = [CAT_COLORS.get(c, "#757575") for c in imp_cats]
    ax.barh(range(len(imp)), imp.values, color=imp_colors, alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([short(f) for f in imp.index], fontsize=6)
    ax.set_xlabel("Improvement over origin")
    ax.set_title("Center Search Benefit")

    # 5c: Center distance from origin
    ax = axes[1, 0]
    dist = np.sqrt(raw_ac.loc[feats, "best_center_x"] ** 2 +
                   raw_ac.loc[feats, "best_center_y"] ** 2)
    dist_sorted = dist.sort_values(ascending=True)
    d_cats = [categorize(f) for f in dist_sorted.index]
    d_colors = [CAT_COLORS.get(c, "#757575") for c in d_cats]
    ax.barh(range(len(dist_sorted)), dist_sorted.values, color=d_colors, alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(dist_sorted)))
    ax.set_yticklabels([short(f) for f in dist_sorted.index], fontsize=6)
    ax.set_xlabel("Distance from origin (um)")
    ax.set_title("Optimal Center Distance from ONH")

    # 5d: Slope direction histogram
    ax = axes[1, 1]
    slopes = raw_ac.loc[feats, "best_slope"].to_numpy()
    positive = np.sum(slopes > 0)
    negative = np.sum(slopes < 0)
    ax.bar(["Center-high\n(slope < 0)", "Periphery-high\n(slope > 0)"],
           [negative, positive],
           color=["#2196F3", "#FF9800"], alpha=0.8, edgecolor="k")
    ax.set_ylabel("Number of features")
    ax.set_title("Radial Direction: Center vs Periphery")

    fig.suptitle("Radial Center Analysis: GB Control Dashboard", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "radial_dashboard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Done.")


if __name__ == "__main__":
    main()
