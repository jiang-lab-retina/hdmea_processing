"""fig10_fullfield_opponency.py

Implement Szatko et al. 2020 Fig. 6c "full-field opponency" from MEA data,
using LOW-INTENSITY (64-level) traces only (trials 0-2 of the 9-trial
before_green_blue_3s_3i_3x_64_128_255 column).

For each cell, correlate green and blue event kernels at light onset and offset:
  rho_onset  = pearsonr(green_onset_kernel,  blue_onset_kernel)
  rho_offset = pearsonr(green_offset_kernel, blue_offset_kernel)
  ff_opponency = min(rho_onset, rho_offset)
  is_ff_opp    = ff_opponency < -0.3

Event kernels are 120-frame (2.0 s at 60 Hz) windows immediately after each
stimulus transition.  The blue-OFF kernel is only 119 frames long (trace ends
at 719); the green-OFF kernel is truncated to match.

Stimulus layout (60 Hz, 719 frames per trial):
  [0, 60)    baseline
  [60, 240)  green ON   -> green_onset  = [60, 180)
  [240, 420) green OFF  -> green_offset = [240, 359)  (truncated to 119)
  [420, 600) blue  ON   -> blue_onset   = [420, 540)
  [600, 719) blue  OFF  -> blue_offset  = [600, 719)  (119 frames)
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
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr as _pearsonr
from pygam import LinearGAM, te

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    load_combined,
    savefig,
    savetable,
    dv_bin_fraction,
    wilson_ci,
    style_xy_axes,
    COORD_SCALE,
    XY_RANGE,
)

# ------------------------------------------------------------------ #
# Paths and constants
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[4]
COORD_LIMIT = 100.0

SRC_PARQUETS = {
    "_ptx_str": PROJECT_ROOT / "dataframe_compare" / "output_ptx_str"
                / "compared_dataframe_v2_labeled_spatial.parquet",
    "_ptx":     PROJECT_ROOT / "dataframe_compare" / "output_ptx"
                / "compared_dataframe_v2_labeled_spatial.parquet",
    "_str":     PROJECT_ROOT / "dataframe_compare" / "output_str"
                / "compared_dataframe_v2_labeled_spatial.parquet",
}

# 9-trial column: trials 0-2 = low (64), 3-5 = mid (128), 6-8 = high (255)
TRACE_COL_9 = "before_green_blue_3s_3i_3x_64_128_255"
LOW_TRIAL_SLICE = slice(0, 3)   # low intensity trials

# Event-kernel windows (120 frames = 2.0 s)
KERNEL_LEN   = 120
GREEN_ON_S   = 60
GREEN_ON_E   = GREEN_ON_S + KERNEL_LEN         # 180
BLUE_ON_S    = 420
BLUE_ON_E    = BLUE_ON_S + KERNEL_LEN           # 540
GREEN_OFF_S  = 240
BLUE_OFF_S   = 600
TRACE_END    = 719
BLUE_OFF_LEN = TRACE_END - BLUE_OFF_S           # 119
OFF_KERNEL_LEN = BLUE_OFF_LEN                   # truncate green_off to match
GREEN_OFF_E  = GREEN_OFF_S + OFF_KERNEL_LEN     # 359

# Plot constants
GRIDSIZE  = 40
MINCNT    = 2
N_SPLINES = 20
CMAP_RHO  = "RdBu_r"
CMAP_OPP  = "coolwarm"
FF_OPP_THRESHOLD = -0.3


# ------------------------------------------------------------------ #
# Trace helpers (adapted from fig8)
# ------------------------------------------------------------------ #

def _mean_trace_low(trace_val_9) -> np.ndarray | None:
    """Average the LOW-intensity trials (0-2) from the 9-trial object-array."""
    try:
        arr = np.array(trace_val_9)
        low_trials = arr[LOW_TRIAL_SLICE]
        trials = np.stack([np.asarray(t, dtype=float) for t in low_trials])
        mean = trials.mean(axis=0)
        return mean if len(mean) >= BLUE_OFF_S else None
    except Exception:
        return None


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns NaN when either vector has zero variance."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = _pearsonr(a, b)
    return float(r)


def extract_ff_opponency(trace_series: pd.Series) -> pd.DataFrame:
    """Compute rho_onset, rho_offset, ff_opponency for every row.

    The pre-stimulus baseline mean (frames [0, GREEN_ON_S)) is subtracted
    from each mean trace before kernel extraction.
    """
    n = len(trace_series)
    rho_onset  = np.full(n, np.nan)
    rho_offset = np.full(n, np.nan)

    for i, val in enumerate(trace_series):
        mt = _mean_trace_low(val)
        if mt is None:
            continue
        # Subtract pre-stimulus baseline mean (frames [0, GREEN_ON_S))
        baseline_mean = float(np.mean(mt[:GREEN_ON_S]))
        mt = mt - baseline_mean
        g_on  = mt[GREEN_ON_S  : GREEN_ON_E]
        b_on  = mt[BLUE_ON_S   : BLUE_ON_E]
        g_off = mt[GREEN_OFF_S : GREEN_OFF_E]
        actual_end = min(TRACE_END, len(mt))
        b_off = mt[BLUE_OFF_S  : actual_end]
        # Truncate to equal length
        off_len = min(len(g_off), len(b_off))
        g_off = g_off[:off_len]
        b_off = b_off[:off_len]

        rho_onset[i]  = _safe_pearsonr(g_on, b_on)
        rho_offset[i] = _safe_pearsonr(g_off, b_off)

    ff_opp = np.fmin(rho_onset, rho_offset)
    return pd.DataFrame({
        "rho_onset":     rho_onset,
        "rho_offset":    rho_offset,
        "ff_opponency":  ff_opp,
        "is_ff_opp":     ff_opp < FF_OPP_THRESHOLD,
    })


# ------------------------------------------------------------------ #
# Data loading (mirrors fig8 pattern)
# ------------------------------------------------------------------ #

def load_with_ff_opponency() -> pd.DataFrame:
    """Load combined parquet + append full-field opponency features.

    Loads WITHOUT response_filter first (so row counts align with source
    parquets), extracts per-cell rho values, then applies the filter.
    Same pattern as fig8_sustained_opponency.py.
    """
    df = load_combined()   # no filter yet -- keeps alignment with src_filt

    rho_on_all  = np.full(len(df), np.nan)
    rho_off_all = np.full(len(df), np.nan)

    for exp, src_path in SRC_PARQUETS.items():
        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping {exp}")
            continue

        exp_mask = df["source_experiment"] == exp
        exp_rows = np.where(exp_mask.values)[0]
        print(f"  Loading {exp} traces ({len(exp_rows)} rows)...")

        src = pd.read_parquet(
            src_path,
            columns=["improved_tx", "improved_ty", TRACE_COL_9],
        )
        src_mask = (src["improved_tx"].abs() < COORD_LIMIT) & \
                   (src["improved_ty"].abs() < COORD_LIMIT)
        src_filt = src[src_mask].reset_index(drop=True)

        assert len(src_filt) == len(exp_rows), (
            f"Row count mismatch for {exp}: "
            f"source={len(src_filt)} vs combined={len(exp_rows)}"
        )

        feats = extract_ff_opponency(src_filt[TRACE_COL_9])

        rho_on_all[exp_rows]  = feats["rho_onset"].values
        rho_off_all[exp_rows] = feats["rho_offset"].values

    df["rho_onset"]    = rho_on_all
    df["rho_offset"]   = rho_off_all
    df["ff_opponency"] = np.fmin(rho_on_all, rho_off_all)
    df["is_ff_opp"]    = df["ff_opponency"] < FF_OPP_THRESHOLD

    # Now apply response filter (same as load_combined(response_filter=True))
    _pos = (df["green_on_peak_extreme"] > 0) & (df["blue_on_peak_extreme"] > 0)
    _thr = (df["green_on_peak_extreme"] >= 50) | (df["blue_on_peak_extreme"] >= 50)
    df = df[_pos & _thr].reset_index(drop=True)
    print(f"After response filter: {len(df)} cells")

    return df


# ------------------------------------------------------------------ #
# GAM helpers (adapted from fig9)
# ------------------------------------------------------------------ #

def _fit_gam(x, y, c, n_splines=N_SPLINES):
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


def _draw_raw_hexbin(ax, x, y, c, vmin, vmax, cmap, title):
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean,
        gridsize=GRIDSIZE, extent=(*XY_RANGE, *XY_RANGE),
        mincnt=MINCNT, cmap=cmap, vmin=vmin, vmax=vmax,
    )
    style_xy_axes(ax, title=title)
    return hb


def _draw_gam_hexbin(ax, x, y, c, vmin, vmax, cmap, gam, title,
                     use_pred_scale: bool = False):
    hb = ax.hexbin(
        x, y, gridsize=GRIDSIZE,
        extent=(*XY_RANGE, *XY_RANGE), mincnt=MINCNT, cmap=cmap,
    )
    offsets = hb.get_offsets()
    if gam is not None and len(offsets) > 0:
        preds = gam.predict(offsets)
        hb.set_array(preds)
        if use_pred_scale:
            p_mean = float(np.mean(preds))
            p_std  = max(float(np.std(preds)), 1e-6)
            vmin, vmax = p_mean - 2 * p_std, p_mean + 2 * p_std
    hb.set_clim(vmin=vmin, vmax=vmax)
    style_xy_axes(ax, title=title)
    return hb


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    print("Loading data with full-field opponency features...")
    df = load_with_ff_opponency()

    valid = df["ff_opponency"].notna()
    dv = df[valid].copy()
    print(f"Valid cells: {valid.sum()} / {len(df)}")
    n_opp = int(dv["is_ff_opp"].sum())
    n_total = len(dv)
    frac_opp = n_opp / n_total if n_total > 0 else 0

    x = dv["X_um"].to_numpy()
    y = dv["Y_um"].to_numpy()

    # ---------------------------------------------------------------- #
    # Figure: 5 rows x 3 cols
    # Row 0: rho_onset
    # Row 1: rho_offset
    # Row 2: ff_opponency  (clean hexbin/GAM, no dot overlay)
    # Row 3: negative ff_opponency cells as individual dots
    # Row 4: binary is_ff_opp (opponent fraction per hexbin)
    # ---------------------------------------------------------------- #
    fig = plt.figure(figsize=(17, 27))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.42, wspace=0.35,
                           width_ratios=[1, 1, 0.8])

    rows_cfg = [
        ("rho_onset",    CMAP_RHO, -1, 1),
        ("rho_offset",   CMAP_RHO, -1, 1),
        ("ff_opponency", CMAP_RHO, -1, 1),
    ]

    gam_models = {}

    for ri, (col, cmap, vmin, vmax) in enumerate(rows_cfg):
        c_vals = dv[col].to_numpy()
        label = {
            "rho_onset": "A. rho_onset (green-on vs blue-on correlation)",
            "rho_offset": "B. rho_offset (green-off vs blue-off correlation)",
            "ff_opponency": "C. ff_opponency = min(rho_on, rho_off)",
        }[col]

        # Fit GAM
        finite = np.isfinite(c_vals)
        gam = _fit_gam(x[finite], y[finite], c_vals[finite]) if finite.sum() >= 50 else None
        gam_models[col] = gam

        n_fin = int(finite.sum())
        mean_v = float(np.nanmean(c_vals))
        median_v = float(np.nanmedian(c_vals))

        # Raw hexbin
        ax_raw = fig.add_subplot(gs[ri, 0])
        hb_raw = _draw_raw_hexbin(
            ax_raw, x[finite], y[finite], c_vals[finite],
            vmin, vmax, cmap,
            f"{label}\nn={n_fin:,}  mean={mean_v:.3f}",
        )
        fig.colorbar(hb_raw, ax=ax_raw, shrink=0.6, pad=0.02)

        # GAM smoothed (no dot overlay)
        ax_gam = fig.add_subplot(gs[ri, 1])
        hb_gam = _draw_gam_hexbin(
            ax_gam, x[finite], y[finite], c_vals[finite],
            vmin, vmax, cmap, gam,
            f"{label} (GAM)",
            use_pred_scale=True,
        )
        fig.colorbar(hb_gam, ax=ax_gam, shrink=0.6, pad=0.02)

        # Distribution histogram
        ax_hist = fig.add_subplot(gs[ri, 2])
        ax_hist.hist(c_vals[finite], bins=60, color="steelblue",
                     edgecolor="white", linewidth=0.3, alpha=0.85)
        if col == "ff_opponency":
            ax_hist.axvline(FF_OPP_THRESHOLD, color="red", ls="--", lw=1.5,
                            label=f"threshold = {FF_OPP_THRESHOLD}")
            ax_hist.legend(fontsize=7)
        ax_hist.axvline(mean_v, color="black", ls="-", lw=1, alpha=0.7)
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Cell count")
        ax_hist.set_title(
            f"mean={mean_v:.3f}  median={median_v:.3f}\nn={n_fin:,}",
            fontsize=8,
        )
        ax_hist.grid(True, alpha=0.3)

    # ---------------------------------------------------------------- #
    # Row 3: dot map of negative ff_opponency cells
    # ---------------------------------------------------------------- #
    ff_vals = dv["ff_opponency"].to_numpy()
    finite_ff = np.isfinite(ff_vals)
    neg_mask  = finite_ff & (ff_vals < 0)
    pos_mask  = finite_ff & (ff_vals >= 0)
    n_neg = int(neg_mask.sum())

    # Col 0: all cells as grey dots, negative cells colored
    ax_dot0 = fig.add_subplot(gs[3, 0])
    ax_dot0.scatter(x[pos_mask], y[pos_mask], c="lightgrey", s=6,
                    alpha=0.4, linewidths=0, zorder=2, label="ff >= 0")
    sc0 = ax_dot0.scatter(
        x[neg_mask], y[neg_mask], c=ff_vals[neg_mask],
        s=25, marker="o", cmap=CMAP_RHO, vmin=-1, vmax=0,
        edgecolors="black", linewidths=0.6, zorder=4,
        label=f"ff < 0 (n={n_neg})",
    )
    fig.colorbar(sc0, ax=ax_dot0, shrink=0.6, pad=0.02, label="ff_opponency")
    style_xy_axes(ax_dot0,
                  title=f"D. Negative ff_opponency cells (all cells background)\n"
                        f"n_neg={n_neg}  n_total={int(finite_ff.sum()):,}")
    ax_dot0.legend(fontsize=7, loc="lower right")

    # Col 1: negative cells only, colored by ff_opponency value
    ax_dot1 = fig.add_subplot(gs[3, 1])
    sc1 = ax_dot1.scatter(
        x[neg_mask], y[neg_mask], c=ff_vals[neg_mask],
        s=30, marker="o", cmap=CMAP_RHO, vmin=-1, vmax=0,
        edgecolors="black", linewidths=0.6, zorder=4,
    )
    fig.colorbar(sc1, ax=ax_dot1, shrink=0.6, pad=0.02, label="ff_opponency")
    style_xy_axes(ax_dot1,
                  title=f"D. Negative ff_opponency cells only\n"
                        f"n={n_neg}")

    # Col 2: histogram of negative ff_opponency values + summary text
    ax_dot2 = fig.add_subplot(gs[3, 2])
    if n_neg > 0:
        ax_dot2.hist(ff_vals[neg_mask], bins=20, color="steelblue",
                     edgecolor="white", linewidth=0.3, alpha=0.85)
        ax_dot2.axvline(float(np.mean(ff_vals[neg_mask])), color="black",
                        ls="-", lw=1, alpha=0.7, label="mean")
        ax_dot2.legend(fontsize=7)
    ax_dot2.set_xlabel("ff_opponency (negative only)")
    ax_dot2.set_ylabel("Cell count")
    ax_dot2.set_title(
        f"Negative ff_opponency\n"
        f"n={n_neg}  mean={float(np.nanmean(ff_vals[neg_mask])):.3f}" if n_neg > 0
        else "No negative ff_opponency cells",
        fontsize=8,
    )
    ax_dot2.grid(True, alpha=0.3)

    # ---------------------------------------------------------------- #
    # Row 4: binary is_ff_opp (opponent fraction per hexbin)
    # ---------------------------------------------------------------- #
    opp_flag = dv["is_ff_opp"].astype(float).to_numpy()

    ax_raw4 = fig.add_subplot(gs[4, 0])
    hb_raw4 = _draw_raw_hexbin(
        ax_raw4, x, y, opp_flag,
        0, 0.5, CMAP_OPP,
        f"E. Full-field opponent fraction (rho < {FF_OPP_THRESHOLD})\n"
        f"n={n_total:,}  opponent={n_opp:,} ({frac_opp:.1%})",
    )
    fig.colorbar(hb_raw4, ax=ax_raw4, shrink=0.6, pad=0.02,
                 label="opponent fraction")

    # GAM on binary
    gam_opp = _fit_gam(x, y, opp_flag) if len(x) >= 50 else None
    ax_gam4 = fig.add_subplot(gs[4, 1])
    hb_gam4 = _draw_gam_hexbin(
        ax_gam4, x, y, opp_flag,
        0, 0.5, CMAP_OPP, gam_opp,
        "E. Opponent fraction (GAM)",
        use_pred_scale=True,
    )
    fig.colorbar(hb_gam4, ax=ax_gam4, shrink=0.6, pad=0.02,
                 label="opponent fraction")

    # Summary text panel
    ax_txt = fig.add_subplot(gs[4, 2])
    ax_txt.axis("off")

    dorsal  = dv[dv["Y_um"] > 0]
    ventral = dv[dv["Y_um"] < 0]
    n_d = len(dorsal)
    n_v = len(ventral)
    k_d = int(dorsal["is_ff_opp"].sum())
    k_v = int(ventral["is_ff_opp"].sum())
    ci_d = wilson_ci(k_d, n_d)
    ci_v = wilson_ci(k_v, n_v)
    frac_d = k_d / n_d if n_d > 0 else 0
    frac_v = k_v / n_v if n_v > 0 else 0

    ff_vals = dv["ff_opponency"].dropna()
    lines = [
        "Summary Statistics",
        "",
        f"Total valid cells: {n_total:,}",
        f"  Opponent (ff < {FF_OPP_THRESHOLD}): {n_opp:,}  ({frac_opp:.1%})",
        f"  Non-opponent: {n_total - n_opp:,}",
        "",
        f"ff_opponency distribution:",
        f"  mean   = {ff_vals.mean():.4f}",
        f"  median = {ff_vals.median():.4f}",
        f"  std    = {ff_vals.std():.4f}",
        "",
        f"Dorsal  (Y > 0):  {k_d}/{n_d} = {frac_d:.3f}",
        f"  95% Wilson CI: [{ci_d[0]:.3f}, {ci_d[1]:.3f}]",
        f"Ventral (Y < 0):  {k_v}/{n_v} = {frac_v:.3f}",
        f"  95% Wilson CI: [{ci_v[0]:.3f}, {ci_v[1]:.3f}]",
        "",
        f"Paper reference (Fig. 6):",
        f"  ventral opp frac: 0.309",
        f"  dorsal  opp frac: 0.114",
        "",
        f"Intensity: LOW (64-level, trials 0-2)",
        f"Kernel window: 120 fr (2.0 s)",
        f"  onset:  green[60:180] vs blue[420:540]",
        f"  offset: green[240:359] vs blue[600:719]",
        f"  (blue-off truncated to 119 frames)",
    ]
    ax_txt.text(
        0.05, 0.97, "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=7.5, va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    fig.suptitle(
        "Fig 10 -- Full-field opponency (Szatko 2020 Fig. 6c analog)  "
        "[LOW intensity, 64-level]\n"
        "ff_opponency = min(corr(green_onset, blue_onset), "
        "corr(green_offset, blue_offset))",
        fontsize=12, y=1.00,
    )

    out_path = savefig(fig, "fig10_fullfield_opponency.png", dpi=180)
    print(f"Saved: {out_path}")

    # ---------------------------------------------------------------- #
    # Tables
    # ---------------------------------------------------------------- #
    # Per-cell table
    cell_out = dv[[
        "before_dataset_id", "source_experiment",
        "X_um", "Y_um", "group", "subtype",
        "rho_onset", "rho_offset", "ff_opponency", "is_ff_opp",
    ]].copy()
    savetable(cell_out, "fig10_per_cell_ffopp.csv")
    print("Saved: fig10_per_cell_ffopp.csv")

    # D-V bin summary
    dv_summary = dv_bin_fraction(dv, "is_ff_opp")
    savetable(dv_summary, "fig10_dv_summary.csv")
    print("Saved: fig10_dv_summary.csv")

    print("Done.")


if __name__ == "__main__":
    main()
