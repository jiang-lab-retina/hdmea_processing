"""fig9_threshold_on_ratio.py

Hexbin sweep of green_blue_on_ratio_low filtered by ON-response threshold.

For each threshold value, only cells where EITHER green_on_peak_extreme_low OR
blue_on_peak_extreme_low exceeds the threshold are included. Five thresholds
are swept (0, 20, 50, 100, 150 Hz) to show how the spatial pattern evolves as
weak-responding cells are progressively removed.

Layout: 5 columns (one per threshold) x 2 rows (raw hexbin / GAM smoothed).
All panels share the same colorbar scale (derived from the 0 Hz baseline data).
"""

from __future__ import annotations

import sys
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
from pygam import LinearGAM, te

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_combined, savefig, savetable, FIG_DIR, TAB_DIR

# ------------------------------------------------------------------ #
# Constants (match gb_spatial_control/config.py)
# ------------------------------------------------------------------ #
COORD_SCALE = 16
XY_RANGE = (-1600, 1600)
GRIDSIZE = 40
MINCNT = 2
CMAP = "coolwarm"
N_SPLINES = 20

FEAT = "green_blue_on_ratio_low"
THRESH_COL_G = "green_on_peak_extreme_low"
THRESH_COL_B = "blue_on_peak_extreme_low"

THRESHOLDS = [0, 20, 50, 100, 150]   # Hz


# ------------------------------------------------------------------ #
# GAM helpers (adapted from gb_spatial_control/spatial_plots.py)
# ------------------------------------------------------------------ #

def _fit_gam(x: np.ndarray, y: np.ndarray, c: np.ndarray,
             n_splines: int = N_SPLINES):
    """Fit a 2D tensor-product LinearGAM; return None on failure."""
    X_train = np.column_stack([x, y])
    gam = LinearGAM(te(0, 1, n_splines=[n_splines, n_splines]))
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


def _draw_raw_hexbin(ax, x, y, c, vmin, vmax, title):
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean,
        gridsize=GRIDSIZE, extent=(*XY_RANGE, *XY_RANGE),
        mincnt=MINCNT, cmap=CMAP, vmin=vmin, vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE)
    ax.set_ylim(XY_RANGE)
    ax.set_title(title, fontsize=8)
    return hb


def _draw_gam_hexbin(ax, x, y, c, vmin, vmax, gam):
    """Draw GAM-predicted values onto hexbin cells."""
    hb = ax.hexbin(
        x, y, gridsize=GRIDSIZE,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=MINCNT, cmap=CMAP,
    )
    offsets = hb.get_offsets()
    if gam is not None and len(offsets) > 0:
        preds = gam.predict(offsets)
        hb.set_array(preds)
    hb.set_clim(vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE)
    ax.set_ylim(XY_RANGE)
    return hb


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    # Load data (load_combined applies coord filter and adds SC etc.)
    df_raw = load_combined()

    # Keep only rows with valid feature and threshold columns
    needed = [FEAT, THRESH_COL_G, THRESH_COL_B, "improved_tx", "improved_ty"]
    df_raw = df_raw[needed].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Cells after NaN drop: {len(df_raw)}")

    x_all = df_raw["improved_tx"].to_numpy() * COORD_SCALE
    y_all = df_raw["improved_ty"].to_numpy() * COORD_SCALE
    c_all = df_raw[FEAT].to_numpy()
    g_raw = df_raw[THRESH_COL_G].to_numpy()
    b_raw = df_raw[THRESH_COL_B].to_numpy()

    # Stage 1: keep only cells with positive ON-peak in BOTH channels
    both_pos_mask = (g_raw > 0) & (b_raw > 0)
    x_all   = x_all[both_pos_mask]
    y_all   = y_all[both_pos_mask]
    c_all   = c_all[both_pos_mask]
    g_all   = g_raw[both_pos_mask]
    b_all   = b_raw[both_pos_mask]
    n_both_pos = int(both_pos_mask.sum())
    print(f"Both-positive pre-filter: {n_both_pos} cells ({n_both_pos/len(df_raw)*100:.1f}%)")

    # ---------------------------------------------------------------- #
    # Build figure: 2 rows x 5 cols
    # Each column gets its own mean +/- std colour scale so within-group
    # spatial patterns are clearly visible regardless of the overall shift
    # in mean ratio across thresholds.
    # ---------------------------------------------------------------- #
    n_thr = len(THRESHOLDS)
    fig, axes = plt.subplots(
        2, n_thr,
        figsize=(4.5 * n_thr, 9),
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    summary_rows = []

    for col, thr in enumerate(THRESHOLDS):
        # Stage 2: threshold sweep on the both-positive subset
        either_mask = (g_all >= thr) | (b_all >= thr)
        x = x_all[either_mask]
        y = y_all[either_mask]
        c = c_all[either_mask]
        n = int(either_mask.sum())
        pct = 100.0 * n / n_both_pos
        print(f"Threshold >= {thr} Hz: {n} cells ({pct:.1f}%)")

        # Per-subplot colour scale: mean +/- 1 std of this threshold's data
        c_mean = float(np.nanmean(c)) if n > 0 else 0.5
        c_std  = float(np.nanstd(c))  if n > 0 else 0.5
        vmin = c_mean - c_std
        vmax = c_mean + c_std
        print(f"  Color range: [{vmin:.3f}, {vmax:.3f}]  (mean={c_mean:.3f}, std={c_std:.3f})")

        # Pearson r with Y_um
        if n >= 10 and np.std(y) > 0 and np.std(c) > 0:
            r_dv, p_dv = pearsonr(y, c)
        else:
            r_dv, p_dv = np.nan, np.nan

        summary_rows.append({
            "threshold_hz": thr,
            "both_pos_baseline_n": n_both_pos,
            "n_cells": n,
            "pct_of_both_pos": round(pct, 1),
            "mean_ratio": float(np.mean(c)) if n > 0 else np.nan,
            "std_ratio":  float(np.std(c))  if n > 0 else np.nan,
            "vmin": round(vmin, 4),
            "vmax": round(vmax, 4),
            "pearson_r_dv": round(r_dv, 4) if not np.isnan(r_dv) else np.nan,
            "pearson_p_dv": round(p_dv, 4) if not np.isnan(p_dv) else np.nan,
        })

        # Fit GAM
        print(f"  Fitting GAM...")
        gam = _fit_gam(x, y, c) if n >= 50 else None

        col_title = f">= {thr} Hz  (n={n:,}, {pct:.0f}%)\n[{vmin:.2f}, {vmax:.2f}]"

        # Row 0: raw hexbin
        ax_raw = axes[0, col]
        hb_r = _draw_raw_hexbin(ax_raw, x, y, c, vmin, vmax, col_title)
        if col == 0:
            ax_raw.set_ylabel("V <-- Y (um) --> D", fontsize=8)
        ax_raw.set_xlabel("T <-- X (um) --> N", fontsize=7)
        fig.colorbar(hb_r, ax=ax_raw, shrink=0.55, pad=0.03,
                     label="ratio" if col == n_thr - 1 else "")

        # Row 1: GAM smoothed
        ax_gam = axes[1, col]
        hb_g = _draw_gam_hexbin(ax_gam, x, y, c, vmin, vmax, gam)
        if col == 0:
            ax_gam.set_ylabel("V <-- Y (um) --> D", fontsize=8)
        ax_gam.set_xlabel("T <-- X (um) --> N", fontsize=7)
        fig.colorbar(hb_g, ax=ax_gam, shrink=0.55, pad=0.03,
                     label="ratio" if col == n_thr - 1 else "")

    # Row labels
    axes[0, 0].annotate(
        "Raw hexbin", xy=(-0.30, 0.5), xycoords="axes fraction",
        fontsize=9, rotation=90, va="center", ha="center",
        fontweight="bold",
    )
    axes[1, 0].annotate(
        "GAM smoothed", xy=(-0.30, 0.5), xycoords="axes fraction",
        fontsize=9, rotation=90, va="center", ha="center",
        fontweight="bold",
    )

    fig.suptitle(
        f"Fig 9 -- {FEAT}: ON-response threshold sweep\n"
        f"Pre-filter: both green AND blue on-peak-low > 0 (n={n_both_pos:,});"
        f"  then either >= threshold",
        fontsize=10, y=1.01,
    )

    out_path = savefig(fig, "fig9_threshold_on_ratio.png", dpi=180)
    print(f"Saved: {out_path}")

    # ---------------------------------------------------------------- #
    # Summary table
    # ---------------------------------------------------------------- #
    summary_df = pd.DataFrame(summary_rows)
    tab_path = savetable(summary_df, "fig9_threshold_summary.csv")
    print(f"Saved: {tab_path}")
    print(summary_df.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
