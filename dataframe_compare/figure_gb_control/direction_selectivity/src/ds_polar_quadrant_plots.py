"""
DS Polar Bar Plots -- Preferred direction by spatial quadrant
=============================================================
Polar histograms of preferred_direction split by retinal quadrant
(defined by cell spatial coordinates improved_tx, improved_ty).

Quadrants (origin at 0, 0):
  - Dorsal-Nasal     (D-N):  x >= 0, y >= 0
  - Dorsal-Temporal   (D-T):  x <  0, y >= 0
  - Ventral-Temporal  (V-T):  x <  0, y <  0
  - Ventral-Nasal    (V-N):  x >= 0, y <  0

One figure per scope (All DSGC, each DSGC cluster), each with 4 polar
subplots (one per quadrant).

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

FIG_DIR = FIG_DIR_BASE / "direction_selectivity" / "polar_plots" / "by_quadrant"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 24
BIN_WIDTH = 2 * np.pi / N_BINS

QUADRANTS = [
    ("D-N (x>=0, y>=0)", lambda df: (df[X_COL] >= 0) & (df[Y_COL] >= 0), "#1f77b4"),
    ("D-T (x<0, y>=0)",  lambda df: (df[X_COL] < 0)  & (df[Y_COL] >= 0), "#ff7f0e"),
    ("V-T (x<0, y<0)",   lambda df: (df[X_COL] < 0)  & (df[Y_COL] < 0),  "#2ca02c"),
    ("V-N (x>=0, y<0)",  lambda df: (df[X_COL] >= 0) & (df[Y_COL] < 0),  "#d62728"),
]


N_BINS_ORI = 12
BIN_WIDTH_ORI = np.pi / N_BINS_ORI


def _load_before_preferred_direction():
    """Load before_preferred_direction/orientation and metadata."""
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

    ax.bar(
        bin_centers, counts, width=BIN_WIDTH * 0.85,
        color=color, edgecolor="white", linewidth=0.8, alpha=0.8,
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    direction_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(direction_labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

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

    all_centers = np.concatenate([bin_centers, bin_centers + np.pi])
    all_counts = np.concatenate([counts, counts])

    ax.bar(
        all_centers, all_counts, width=BIN_WIDTH_ORI * 0.85,
        color=color, edgecolor="white", linewidth=0.8, alpha=0.8,
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0", "45", "90", "135", "180", "225", "270", "315"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

    doubled = 2 * angles_rad
    mean_x = np.mean(np.cos(doubled))
    mean_y = np.mean(np.sin(doubled))
    mean_r = np.sqrt(mean_x**2 + mean_y**2)
    mean_ori = (np.arctan2(mean_y, mean_x) / 2) % np.pi
    r_max = ax.get_ylim()[1]
    ax.annotate("", xy=(mean_ori, r_max * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0))
    ax.annotate("", xy=(mean_ori + np.pi, r_max * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0))

    ax.set_title(
        f"{title}\n(n={n_total}, R={mean_r:.2f})",
        fontsize=10, fontweight="bold", pad=14,
    )


def _generate_quadrant_figure(scope_df, scope_label, save_path,
                              angle_col="preferred_direction",
                              plot_fn=None, angle_label="Preferred Direction"):
    """Create a 1x4 polar figure with one panel per quadrant."""
    if plot_fn is None:
        plot_fn = _polar_bar

    fig, axes = plt.subplots(
        1, 4, figsize=(20, 5.5),
        subplot_kw={"projection": "polar"},
    )

    for qi, (q_label, q_mask_fn, q_color) in enumerate(QUADRANTS):
        mask = q_mask_fn(scope_df)
        q_df = scope_df[mask]
        ax = axes[qi]
        plot_fn(ax, q_df[angle_col], q_color, q_label)

    fig.suptitle(
        f"{scope_label} -- {angle_label} by Quadrant",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    logger.info("=== Loading preferred direction data ===")
    df = _load_before_preferred_direction()
    logger.info("Total rows after coord filter: %d", len(df))

    dsgc = df[df["group"] == "DSGC"].copy()
    logger.info("DSGC cells: %d", len(dsgc))

    clusters = sorted(
        s for s in dsgc["subtype"].dropna().unique() if s.startswith("DSGC")
    )
    logger.info("DSGC clusters: %s", clusters)

    # All DSGC
    save_path = FIG_DIR / "polar_quadrant_All_DSGC.png"
    _generate_quadrant_figure(dsgc, "All DSGC", save_path)
    logger.info("  Saved: %s", save_path.name)

    # Each DSGC cluster
    for cl in clusters:
        cl_df = dsgc[dsgc["subtype"] == cl]
        if len(cl_df) < 5:
            logger.info("  %s -- skipped (n=%d)", cl, len(cl_df))
            continue
        save_path = FIG_DIR / f"polar_quadrant_{cl}.png"
        _generate_quadrant_figure(cl_df, cl, save_path)
        logger.info("  Saved: %s", save_path.name)

    # --- OSGC (preferred orientation, 0-180 mirrored) ---
    osgc = df[df["group"] == "OSGC"].copy()
    logger.info("OSGC cells: %d", len(osgc))

    osgc_clusters = sorted(
        s for s in osgc["subtype"].dropna().unique() if s.startswith("OSGC")
    )
    logger.info("OSGC clusters: %s", osgc_clusters)

    save_path = FIG_DIR / "polar_quadrant_All_OSGC.png"
    _generate_quadrant_figure(
        osgc, "All OSGC", save_path,
        angle_col="preferred_orientation",
        plot_fn=_polar_bar_orientation,
        angle_label="Preferred Orientation",
    )
    logger.info("  Saved: %s", save_path.name)

    for cl in osgc_clusters:
        cl_df = osgc[osgc["subtype"] == cl]
        if len(cl_df) < 5:
            logger.info("  %s -- skipped (n=%d)", cl, len(cl_df))
            continue
        save_path = FIG_DIR / f"polar_quadrant_{cl}.png"
        _generate_quadrant_figure(
            cl_df, cl, save_path,
            angle_col="preferred_orientation",
            plot_fn=_polar_bar_orientation,
            angle_label="Preferred Orientation",
        )
        logger.info("  Saved: %s", save_path.name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
