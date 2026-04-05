"""
Step 6: Visualize Radial Center Analysis Results
================================================
Creates radial center comparison visualizations.

Adapted from visualize_radial_centers.py.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

from compare_config import OUTPUT_DIR, FIG_DIR_BASE

FIG_DIR = FIG_DIR_BASE / "spatial" / "figures_radial"
FIG_DIR.mkdir(parents=True, exist_ok=True)


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


CAT_COLORS = {
    "Chromatic": "green", "DS/OS": "blue", "ipRGC": "red",
    "Step response": "orange", "Other": "gray",
}


def short(f):
    return (f.replace("green_blue_", "gb_")
             .replace("_extreme", "")
             .replace("_ratio", "_r")
             .replace("_high", "_H")
             .replace("_sustained", "_sus"))


def main():
    combined_path = OUTPUT_DIR / "radial_center_combined.parquet"
    if not combined_path.exists():
        logger.error(f"Missing: {combined_path}")
        return

    logger.info("Loading data ...")
    df = pd.read_parquet(combined_path)
    logger.info(f"  Shape: {df.shape}")

    raw = df[df["data_type"] == "raw_mean"].copy()
    before = raw[raw["condition"] == "before"].set_index("feature")
    after = raw[raw["condition"] == "after"].set_index("feature")
    delta = raw[raw["condition"] == "delta"].set_index("feature")

    common = sorted(before.index.intersection(after.index))
    logger.info(f"  Common features: {len(common)}")

    if len(common) == 0:
        logger.warning("No common features")
        return

    # ---- Fig 1: Radial center map (before vs after) ----
    logger.info("Fig 1: Radial center map ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for pi, (cond_df, label) in enumerate([(before, "Before"), (after, "After")]):
        ax = axes[pi]
        cond_common = cond_df.loc[cond_df.index.isin(common)]
        for feat in cond_common.index:
            row = cond_common.loc[feat]
            cat = categorize(feat)
            ax.scatter(
                row["best_center_x"], row["best_center_y"],
                c=CAT_COLORS.get(cat, "gray"),
                s=abs(row["best_r"]) * 200 + 20,
                alpha=0.7, edgecolors="k", linewidth=0.5,
            )
        ax.axhline(0, color="k", linewidth=0.3, alpha=0.3)
        ax.axvline(0, color="k", linewidth=0.3, alpha=0.3)
        ax.set_xlabel("Center X (um)", fontsize=11)
        ax.set_ylabel("Center Y (um)", fontsize=11)
        ax.set_title(f"{label} Blocker - Optimal Radial Centers", fontsize=12)
        ax.set_aspect("equal")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]
    axes[1].legend(handles=legend_elements, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "radial_center_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: Center shift arrows ----
    logger.info("Fig 2: Center shift arrows ...")
    fig, ax = plt.subplots(figsize=(10, 10))
    for feat in common:
        br = before.loc[feat]
        ar = after.loc[feat]
        cat = categorize(feat)
        ax.annotate(
            "", xy=(ar["best_center_x"], ar["best_center_y"]),
            xytext=(br["best_center_x"], br["best_center_y"]),
            arrowprops=dict(arrowstyle="->", color=CAT_COLORS.get(cat, "gray"),
                            lw=1.5, alpha=0.7),
        )
        ax.scatter(br["best_center_x"], br["best_center_y"],
                   c=CAT_COLORS.get(cat, "gray"), s=30, marker="o", zorder=3)
        ax.scatter(ar["best_center_x"], ar["best_center_y"],
                   c=CAT_COLORS.get(cat, "gray"), s=30, marker="s", zorder=3)
    ax.axhline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.set_xlabel("Center X (um)", fontsize=12)
    ax.set_ylabel("Center Y (um)", fontsize=12)
    ax.set_title("Radial Center Shifts: Before (circle) -> After (square)", fontsize=13)
    ax.set_aspect("equal")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]
    legend_elements.append(Line2D([0], [0], marker="o", color="k", linestyle="None",
                                  markersize=6, label="Before"))
    legend_elements.append(Line2D([0], [0], marker="s", color="k", linestyle="None",
                                  markersize=6, label="After"))
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "center_shift_arrows.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: Improvement comparison (before vs after) ----
    logger.info("Fig 3: Improvement comparison ...")
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(common))
    width = 0.35
    b_abs_r = before.loc[common, "best_r"].abs().to_numpy()
    a_abs_r = after.loc[common, "best_r"].abs().to_numpy()
    ax.bar(x_pos - width / 2, b_abs_r, width, label="Before", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width / 2, a_abs_r, width, label="After", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short(f) for f in common], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|best_r|", fontsize=12)
    ax.set_title("Optimal Radial Correlation: Before vs After", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "improvement_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: Origin vs optimal comparison ----
    logger.info("Fig 4: Origin vs optimal ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for pi, (cond_df, label) in enumerate([(before, "Before"), (after, "After")]):
        ax = axes[pi]
        cond_common = cond_df.loc[cond_df.index.isin(common)]
        feats_sorted = cond_common.sort_values("abs_r_improvement", ascending=True).index
        origin_r = cond_common.loc[feats_sorted, "origin_r"].abs().to_numpy()
        best_r = cond_common.loc[feats_sorted, "best_r"].abs().to_numpy()
        y_pos = np.arange(len(feats_sorted))
        ax.barh(y_pos, best_r, height=0.6, color="steelblue", alpha=0.8, label="Optimal")
        ax.barh(y_pos, origin_r, height=0.3, color="coral", alpha=0.8, label="Origin")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([short(f) for f in feats_sorted], fontsize=7)
        ax.set_xlabel("|r|")
        ax.set_title(f"{label}: Origin vs Optimal |r|")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "origin_vs_optimal.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: Radial dashboard ----
    logger.info("Fig 5: Radial dashboard ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 5a: |best_r| scatter
    ax = axes[0, 0]
    for feat in common:
        cat = categorize(feat)
        ax.scatter(
            abs(before.loc[feat, "best_r"]),
            abs(after.loc[feat, "best_r"]),
            c=CAT_COLORS.get(cat, "gray"), s=40, alpha=0.7,
        )
    lim = max(abs(before.loc[common, "best_r"]).max(),
              abs(after.loc[common, "best_r"]).max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("Before |best_r|")
    ax.set_ylabel("After |best_r|")
    ax.set_title("Radial Trend Strength")
    ax.set_aspect("equal")

    # 5b: improvement scatter
    ax = axes[0, 1]
    for feat in common:
        cat = categorize(feat)
        ax.scatter(
            before.loc[feat, "abs_r_improvement"],
            after.loc[feat, "abs_r_improvement"],
            c=CAT_COLORS.get(cat, "gray"), s=40, alpha=0.7,
        )
    ax.set_xlabel("Before: improvement over origin")
    ax.set_ylabel("After: improvement over origin")
    ax.set_title("Center Search Improvement")

    # 5c: Center distance from origin
    ax = axes[1, 0]
    b_dist = np.sqrt(before.loc[common, "best_center_x"] ** 2 +
                     before.loc[common, "best_center_y"] ** 2)
    a_dist = np.sqrt(after.loc[common, "best_center_x"] ** 2 +
                     after.loc[common, "best_center_y"] ** 2)
    for i, feat in enumerate(common):
        cat = categorize(feat)
        ax.scatter(b_dist.loc[feat], a_dist.loc[feat],
                   c=CAT_COLORS.get(cat, "gray"), s=40, alpha=0.7)
    ax.set_xlabel("Before: center distance from origin (um)")
    ax.set_ylabel("After: center distance from origin (um)")
    ax.set_title("Optimal Center Distance")

    # 5d: shift magnitude histogram
    ax = axes[1, 1]
    shifts = []
    for feat in common:
        dx = before.loc[feat, "best_center_x"] - after.loc[feat, "best_center_x"]
        dy = before.loc[feat, "best_center_y"] - after.loc[feat, "best_center_y"]
        shifts.append(np.sqrt(dx ** 2 + dy ** 2))
    ax.hist(shifts, bins=15, color="mediumpurple", alpha=0.8, edgecolor="k")
    ax.set_xlabel("Center shift (um)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Radial Center Shifts")
    ax.axvline(np.median(shifts), color="red", linestyle="--",
               label=f"Median={np.median(shifts):.0f} um")
    ax.legend()

    fig.suptitle("Radial Center Analysis: Before vs After Blocker", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "radial_dashboard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Done.")


if __name__ == "__main__":
    main()
