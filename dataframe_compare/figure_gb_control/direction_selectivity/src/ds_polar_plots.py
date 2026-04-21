"""
DS Polar Bar Plots -- Preferred direction distributions
=======================================================
Polar histograms of preferred_direction for:

  1. All DSGC cells (pooled)
  2. Each DSGC cluster (DSGC_0 .. DSGC_8)

Uses before-blocker (control) data from all 3 source experiments.
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

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import SOURCE_PARQUETS, FIG_DIR_BASE, X_COL, Y_COL, COORD_LIMIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "direction_selectivity" / "polar_plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 36
BIN_WIDTH = 2 * np.pi / N_BINS

CLUSTER_COLORS = {
    "DSGC_0": "#1f77b4",
    "DSGC_1": "#ff7f0e",
    "DSGC_2": "#2ca02c",
    "DSGC_3": "#d62728",
    "DSGC_4": "#9467bd",
    "DSGC_5": "#8c564b",
    "DSGC_6": "#e377c2",
    "DSGC_7": "#7f7f7f",
    "DSGC_8": "#bcbd22",
    "OSGC_0": "#17becf",
    "OSGC_1": "#aec7e8",
    "OSGC_2": "#98df8a",
    "OSGC_3": "#ff9896",
    "OSGC_4": "#c5b0d5",
    "OSGC_5": "#c49c94",
    "OSGC_6": "#f7b6d2",
}


def _load_before_preferred_direction():
    """Load before_preferred_direction/orientation and metadata from source parquets."""
    want_cols = [
        "before_preferred_direction", "before_preferred_orientation",
        "before_dsi", "group", "subtype", "before_dataset_id",
        X_COL, Y_COL,
    ]
    frames = []
    for exp, path in SOURCE_PARQUETS.items():
        if not path.exists():
            logger.warning("  Missing: %s", path)
            continue
        logger.info("Loading %s ...", path.name)
        schema_names = set(pq.read_schema(path).names)
        cols = [c for c in want_cols if c in schema_names]
        df = pd.read_parquet(path, columns=cols)
        df["source_experiment"] = exp
        frames.append(df)
        logger.info("  %d rows", len(df))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={
        "before_preferred_direction": "preferred_direction",
        "before_preferred_orientation": "preferred_orientation",
        "before_dsi": "dsi",
    })
    combined = combined.dropna(subset=[X_COL, Y_COL])
    mask = (combined[X_COL].abs() < COORD_LIMIT) & (combined[Y_COL].abs() < COORD_LIMIT)
    return combined[mask].copy()


def _polar_bar(ax, angles_deg, color, title):
    """Draw a polar bar chart (angular histogram) on a polar Axes."""
    valid = angles_deg.dropna()
    n_total = len(valid)
    if n_total == 0:
        ax.set_title(f"{title}\n(n=0)", fontsize=10, pad=12)
        return

    angles_rad = np.deg2rad(valid.values)
    bin_edges = np.linspace(-BIN_WIDTH / 2, 2 * np.pi - BIN_WIDTH / 2, N_BINS + 1)
    counts, _ = np.histogram(angles_rad % (2 * np.pi), bins=bin_edges)
    bin_centers = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)

    bars = ax.bar(
        bin_centers, counts, width=BIN_WIDTH * 0.85,
        color=color, edgecolor="white", linewidth=0.8, alpha=0.8,
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    direction_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(direction_labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # mean vector
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    mean_angle = np.arctan2(mean_y, mean_x) % (2 * np.pi)
    mean_r = np.sqrt(mean_x**2 + mean_y**2)
    r_max = ax.get_ylim()[1]
    ax.annotate(
        "", xy=(mean_angle, r_max * 0.95),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
    )

    ax.set_title(
        f"{title}\n(n={n_total}, R={mean_r:.2f})",
        fontsize=10, fontweight="bold", pad=14,
    )


N_BINS_ORI = 18
BIN_WIDTH_ORI = np.pi / N_BINS_ORI


def _polar_bar_orientation(ax, angles_deg, color, title):
    """Polar bar chart for orientation (0-180), mirrored to full circle."""
    valid = angles_deg.dropna()
    n_total = len(valid)
    if n_total == 0:
        ax.set_title(f"{title}\n(n=0)", fontsize=10, pad=12)
        return

    angles_rad = np.deg2rad(valid.values)
    bin_edges = np.linspace(-BIN_WIDTH_ORI / 2, np.pi - BIN_WIDTH_ORI / 2, N_BINS_ORI + 1)
    counts, _ = np.histogram(angles_rad % np.pi, bins=bin_edges)
    bin_centers = np.linspace(0, np.pi, N_BINS_ORI, endpoint=False)

    # mirror to opposite side (orientation has 180-degree symmetry)
    all_centers = np.concatenate([bin_centers, bin_centers + np.pi])
    all_counts = np.concatenate([counts, counts])

    ax.bar(
        all_centers, all_counts, width=BIN_WIDTH_ORI * 0.85,
        color=color, edgecolor="white", linewidth=0.8, alpha=0.8,
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ori_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(ori_labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # mean orientation vector (doubled-angle space for circular mean)
    doubled = 2 * angles_rad
    mean_x = np.mean(np.cos(doubled))
    mean_y = np.mean(np.sin(doubled))
    mean_r = np.sqrt(mean_x**2 + mean_y**2)
    mean_ori = (np.arctan2(mean_y, mean_x) / 2) % np.pi
    r_max = ax.get_ylim()[1]
    # draw axis line through both directions
    ax.annotate(
        "", xy=(mean_ori, r_max * 0.95), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
    )
    ax.annotate(
        "", xy=(mean_ori + np.pi, r_max * 0.95), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
    )

    ax.set_title(
        f"{title}\n(n={n_total}, R={mean_r:.2f})",
        fontsize=10, fontweight="bold", pad=14,
    )


def _generate_group_figure(df, group_name, prefix, save_dir, use_orientation=False):
    """Generate a polar figure for a cell group and its clusters."""
    group_df = df[df["group"] == group_name].copy()
    logger.info("%s cells: %d", group_name, len(group_df))

    clusters = sorted(
        s for s in group_df["subtype"].dropna().unique() if s.startswith(prefix)
    )
    logger.info("  %s clusters: %s", group_name, clusters)

    scopes = [(f"All {group_name}", group_df, "#4A90D9")]
    for cl in clusters:
        scopes.append((cl, group_df[group_df["subtype"] == cl], CLUSTER_COLORS.get(cl, "#999")))

    if use_orientation:
        angle_col = "preferred_orientation"
        plot_fn = _polar_bar_orientation
        angle_label = "Preferred Orientation"
        file_tag = "orientation"
    else:
        angle_col = "preferred_direction"
        plot_fn = _polar_bar
        angle_label = "Preferred Direction"
        file_tag = "direction"

    n_scopes = len(scopes)
    n_cols = 5
    n_rows = int(np.ceil(n_scopes / n_cols))

    fig = plt.figure(figsize=(4.5 * n_cols, 4.5 * n_rows))

    for idx, (label, scope_df, color) in enumerate(scopes):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="polar")
        plot_fn(ax, scope_df[angle_col], color, label)

    for idx in range(n_scopes, n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.set_visible(False)

    fig.suptitle(
        f"{angle_label} Distribution -- {group_name} (before blocker)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_path = save_dir / f"polar_preferred_{file_tag}_{group_name.lower()}.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", save_path)


def main():
    logger.info("=== Loading preferred direction data ===")
    df = _load_before_preferred_direction()
    logger.info("Total rows after coord filter: %d", len(df))

    _generate_group_figure(df, "DSGC", "DSGC", FIG_DIR, use_orientation=False)
    _generate_group_figure(df, "OSGC", "OSGC", FIG_DIR, use_orientation=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()
