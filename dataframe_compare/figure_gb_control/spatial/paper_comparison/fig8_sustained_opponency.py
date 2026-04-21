"""fig8_sustained_opponency.py

Evaluate the paper's claim (Szatko et al. 2020) that more-sustained RGCs show
more color opponency, using GB-specific transient/sustained ratios extracted
from raw green-blue traces.

Stimulus timing (60 Hz, 719 samples total):
  Baseline:          [0,  60)  = 0.0-1.0 s
  Green ON:         [60, 240)  = 1.0-4.0 s
  Blue  ON:        [420, 600)  = 7.0-10.0 s
  Green sustained: [180, 240)  = 3.0-4.0 s (1 s before green off)
  Blue  sustained: [540, 600)  = 9.0-10.0 s (1 s before blue off)

Transient/sustained ratio per channel:
  R = tanh(peak_extreme / (sustained_mean - baseline_mean))
  Values near +/-1 = transient; near 0 = sustained-dominated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    load_combined,
    GROUP_COLORS,
    GROUP_ORDER,
    savefig,
    savetable,
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

TRACE_COL = "before_green_blue_3s_3i_3x"

# Sample windows (indices into 719-sample mean trace)
BASELINE_WIN = (0,   60)   # 0.0 - 1.0 s
SUS_WIN_G    = (180, 240)  # 3.0 - 4.0 s  (1 s before green light off)
SUS_WIN_B    = (540, 600)  # 9.0 - 10.0 s (1 s before blue light off)


# ------------------------------------------------------------------ #
# Feature extraction helpers
# ------------------------------------------------------------------ #

def _mean_trace(trace_val) -> np.ndarray | None:
    """Average 3 trials stored as object-array of 719-sample arrays."""
    try:
        arr = np.array(trace_val)
        # arr is shape (3,) of float32 arrays each length 719
        trials = np.stack([np.asarray(t, dtype=float) for t in arr])
        mean = trials.mean(axis=0)
        return mean if len(mean) >= 600 else None
    except Exception:
        return None


def extract_sustained_features(trace_series: pd.Series) -> pd.DataFrame:
    """Return per-row baseline, green_sus, blue_sus means."""
    n = len(trace_series)
    baseline   = np.full(n, np.nan)
    green_sus  = np.full(n, np.nan)
    blue_sus   = np.full(n, np.nan)

    for i, val in enumerate(trace_series):
        mt = _mean_trace(val)
        if mt is None:
            continue
        baseline[i]  = mt[BASELINE_WIN[0]: BASELINE_WIN[1]].mean()
        green_sus[i] = mt[SUS_WIN_G[0]:    SUS_WIN_G[1]].mean()
        blue_sus[i]  = mt[SUS_WIN_B[0]:    SUS_WIN_B[1]].mean()

    return pd.DataFrame({
        "baseline_mean": baseline,
        "green_sus_mean": green_sus,
        "blue_sus_mean":  blue_sus,
    })


def ts_ratio(peak_extreme: np.ndarray, sus_net: np.ndarray) -> np.ndarray:
    """tanh(peak / sustained_net); NaN where sus_net == 0."""
    sus_safe = np.where(sus_net == 0, np.nan, sus_net)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.tanh(peak_extreme / sus_safe)


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_with_sustained() -> pd.DataFrame:
    """Combined parquet with appended GB sustained features."""
    df = load_combined()

    green_sus_all = np.full(len(df), np.nan)
    blue_sus_all  = np.full(len(df), np.nan)
    baseline_all  = np.full(len(df), np.nan)

    for exp, src_path in SRC_PARQUETS.items():
        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping {exp}")
            continue

        exp_mask = df["source_experiment"] == exp
        exp_rows = np.where(exp_mask.values)[0]
        print(f"  Loading {exp} traces ({len(exp_rows)} rows)...")

        src = pd.read_parquet(
            src_path,
            columns=["improved_tx", "improved_ty", TRACE_COL],
        )
        src_mask = (src["improved_tx"].abs() < COORD_LIMIT) & \
                   (src["improved_ty"].abs() < COORD_LIMIT)
        src_filt = src[src_mask].reset_index(drop=True)

        assert len(src_filt) == len(exp_rows), (
            f"Row count mismatch for {exp}: "
            f"source={len(src_filt)} vs combined={len(exp_rows)}"
        )

        feats = extract_sustained_features(src_filt[TRACE_COL])

        baseline_all[exp_rows]  = feats["baseline_mean"].values
        green_sus_all[exp_rows] = feats["green_sus_mean"].values
        blue_sus_all[exp_rows]  = feats["blue_sus_mean"].values

    df["baseline_mean"]  = baseline_all
    df["green_sus_mean"] = green_sus_all
    df["blue_sus_mean"]  = blue_sus_all

    df["green_sus_net"] = df["green_sus_mean"] - df["baseline_mean"]
    df["blue_sus_net"]  = df["blue_sus_mean"]  - df["baseline_mean"]

    df["green_ts_ratio"] = ts_ratio(
        df["green_on_peak_extreme"].values, df["green_sus_net"].values
    )
    df["blue_ts_ratio"] = ts_ratio(
        df["blue_on_peak_extreme"].values, df["blue_sus_net"].values
    )

    # Combined T/S: mean of absolute ratios across both channels
    df["ts_ratio_mean"] = (
        df["green_ts_ratio"].abs() + df["blue_ts_ratio"].abs()
    ) / 2.0

    return df


# ------------------------------------------------------------------ #
# Plotting helpers
# ------------------------------------------------------------------ #

def _decile_opponency(ts_vals: np.ndarray, opp_vals: np.ndarray):
    """Bin ts_vals into deciles; return bin mids and opponent fractions."""
    edges = np.unique(np.nanpercentile(ts_vals, np.linspace(0, 100, 11)))
    labels = pd.cut(ts_vals, bins=edges, include_lowest=True)
    tmp = pd.DataFrame({"bin": labels, "opp": opp_vals.astype(float)})
    grp = tmp.groupby("bin", observed=True)["opp"].agg(["sum", "count"])
    grp["frac"] = grp["sum"] / grp["count"]
    grp["mid"] = grp.index.map(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
    grp = grp.dropna(subset=["mid"])
    return grp["mid"].values, grp["frac"].values, int(grp["count"].sum())


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    print("Loading data with sustained features...")
    df = load_with_sustained()

    # Response filter: positive ON peaks in both channels, either >= 50 Hz
    _pos = (df["green_on_peak_extreme"] > 0) & (df["blue_on_peak_extreme"] > 0)
    _thr = (df["green_on_peak_extreme"] >= 50) | (df["blue_on_peak_extreme"] >= 50)
    df = df[_pos & _thr].reset_index(drop=True)
    print(f"After response filter: {len(df)} cells")

    valid = (
        df["ts_ratio_mean"].notna()
        & df["green_ts_ratio"].notna()
        & df["blue_ts_ratio"].notna()
    )
    dv = df[valid].copy()
    print(f"Valid rows: {valid.sum()} / {len(df)}")
    print(f"Opponent: {dv['is_opponent'].sum()}, Non-opp: {(~dv['is_opponent']).sum()}")

    # ---------------------------------------------------------------- #
    # Figure
    # ---------------------------------------------------------------- #
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])

    # -- Panel A: Green T/S decile vs opponency fraction --
    g_vals = dv["green_ts_ratio"].dropna()
    g_opp  = dv.loc[g_vals.index, "is_opponent"].values
    mid_A, frac_A, n_A = _decile_opponency(g_vals.values, g_opp)
    ax_A.scatter(mid_A, frac_A, s=55, color="#1f77b4", zorder=3)
    if len(mid_A) >= 3:
        r_A, p_A = stats.pearsonr(mid_A, frac_A)
        xf = np.linspace(mid_A.min(), mid_A.max(), 50)
        slope_A, inter_A, *_ = stats.linregress(mid_A, frac_A)
        ax_A.plot(xf, slope_A * xf + inter_A, "--", color="gray", lw=1.2)
    else:
        r_A, p_A = np.nan, np.nan
    ax_A.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax_A.set_xlabel("Green T/S ratio (bin mid)")
    ax_A.set_ylabel("Opponent fraction")
    ax_A.set_title(
        f"A  Green T/S ratio vs Opponency\n"
        f"r={r_A:.2f}  p={p_A:.3f}  n={n_A:,}",
        fontsize=9,
    )
    ax_A.grid(True, alpha=0.3)

    # -- Panel B: Blue T/S decile vs opponency fraction --
    b_vals = dv["blue_ts_ratio"].dropna()
    b_opp  = dv.loc[b_vals.index, "is_opponent"].values
    mid_B, frac_B, n_B = _decile_opponency(b_vals.values, b_opp)
    ax_B.scatter(mid_B, frac_B, s=55, color="#ff7f0e", zorder=3)
    if len(mid_B) >= 3:
        r_B, p_B = stats.pearsonr(mid_B, frac_B)
        xf = np.linspace(mid_B.min(), mid_B.max(), 50)
        slope_B, inter_B, *_ = stats.linregress(mid_B, frac_B)
        ax_B.plot(xf, slope_B * xf + inter_B, "--", color="gray", lw=1.2)
    else:
        r_B, p_B = np.nan, np.nan
    ax_B.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax_B.set_xlabel("Blue T/S ratio (bin mid)")
    ax_B.set_ylabel("Opponent fraction")
    ax_B.set_title(
        f"B  Blue T/S ratio vs Opponency\n"
        f"r={r_B:.2f}  p={p_B:.3f}  n={n_B:,}",
        fontsize=9,
    )
    ax_B.grid(True, alpha=0.3)

    # -- Panel C: hexbin green T/S vs blue T/S, colored by density,
    #    opponent cells overlaid --
    c_valid = dv.dropna(subset=["green_ts_ratio", "blue_ts_ratio"])
    hb = ax_C.hexbin(
        c_valid["green_ts_ratio"], c_valid["blue_ts_ratio"],
        gridsize=30, cmap="Blues", mincnt=1, alpha=0.8,
    )
    plt.colorbar(hb, ax=ax_C, label="Cell count")
    opp_c = c_valid[c_valid["is_opponent"]]
    ax_C.scatter(
        opp_c["green_ts_ratio"], opp_c["blue_ts_ratio"],
        s=4, color="#FFB000", alpha=0.4, label="Opponent",
    )
    ax_C.axhline(0, color="gray", lw=0.7, ls="--")
    ax_C.axvline(0, color="gray", lw=0.7, ls="--")
    ax_C.set_xlabel("Green T/S ratio")
    ax_C.set_ylabel("Blue T/S ratio")
    ax_C.set_title(
        "C  Green vs Blue T/S ratio\n(orange = opponent cells)",
        fontsize=9,
    )
    ax_C.legend(fontsize=7)

    # -- Panel D: violin of mean T/S by opponent vs non-opponent --
    opp_ts  = dv.loc[dv["is_opponent"],  "ts_ratio_mean"].dropna().values
    nopp_ts = dv.loc[~dv["is_opponent"], "ts_ratio_mean"].dropna().values
    vp = ax_D.violinplot(
        [nopp_ts, opp_ts], positions=[0, 1],
        showmedians=True, showextrema=False,
    )
    vp["cmedians"].set_color("black")
    colors_D = ["#aec7e8", "#ffbb78"]
    for body, c in zip(vp["bodies"], colors_D):
        body.set_facecolor(c)
        body.set_alpha(0.75)
    # Overlay box stats
    for xi, arr in zip([0, 1], [nopp_ts, opp_ts]):
        q25, q75 = np.percentile(arr, [25, 75])
        ax_D.vlines(xi, q25, q75, lw=4, color="k", alpha=0.4)
    t_stat, t_p = stats.ttest_ind(opp_ts, nopp_ts, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(opp_ts, nopp_ts, alternative="two-sided")
    ax_D.set_xticks([0, 1])
    ax_D.set_xticklabels(["Non-opp.", "Opponent"])
    ax_D.set_ylabel("Mean |T/S| ratio")
    ax_D.set_title(
        f"D  Mean T/S by Opponency\n"
        f"t-test p={t_p:.2e}  MWU p={u_p:.2e}",
        fontsize=9,
    )
    ax_D.grid(True, alpha=0.3, axis="y")

    # -- Panel E: per-group mean T/S vs opponency fraction --
    grp_rows = []
    for gname in GROUP_ORDER:
        sub = dv[dv["group"] == gname].dropna(subset=["ts_ratio_mean", "is_opponent"])
        if len(sub) < 10:
            continue
        grp_rows.append({
            "group":    gname,
            "n":        len(sub),
            "mean_ts":  float(sub["ts_ratio_mean"].mean()),
            "opp_frac": float(sub["is_opponent"].mean()),
            "color":    GROUP_COLORS.get(gname, "#aaaaaa"),
        })
    grp_df = pd.DataFrame(grp_rows)
    for _, row in grp_df.iterrows():
        ax_E.scatter(
            row["mean_ts"], row["opp_frac"],
            s=120, color=row["color"], zorder=3,
        )
        ax_E.annotate(
            f"{row['group']} (n={row['n']:,})",
            (row["mean_ts"], row["opp_frac"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    ax_E.set_xlabel("Mean |T/S| ratio")
    ax_E.set_ylabel("Opponent fraction")
    ax_E.set_title("E  Per-group T/S vs Opponency", fontsize=9)
    ax_E.grid(True, alpha=0.3)

    # -- Panel F: summary stats text --
    ax_F.axis("off")
    lines = [
        "Summary Statistics",
        "",
        f"N cells (valid T/S): {valid.sum():,}",
        f"  Opponent:     {int(dv['is_opponent'].sum()):,}",
        f"  Non-opponent: {int((~dv['is_opponent']).sum()):,}",
        "",
        "Mean |T/S| ratio:",
        f"  Opponent:     {opp_ts.mean():.3f} +/- {opp_ts.std():.3f}",
        f"  Non-opp.:     {nopp_ts.mean():.3f} +/- {nopp_ts.std():.3f}",
        "",
        "Welch t-test:",
        f"  t={t_stat:.2f}  p={t_p:.2e}",
        "Mann-Whitney U:",
        f"  U={u_stat:.0f}  p={u_p:.2e}",
        "",
        "Decile Pearson r:",
        f"  Green: r={r_A:.3f}  p={p_A:.3f}",
        f"  Blue:  r={r_B:.3f}  p={p_B:.3f}",
    ]
    ax_F.text(
        0.05, 0.97, "\n".join(lines),
        transform=ax_F.transAxes,
        fontsize=8, va="top", family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    fig.suptitle(
        "Fig 8 -- GB Transient/Sustained Ratio vs Color Opponency\n"
        "(Paper claim: more-sustained RGCs show more color opponency)",
        fontsize=11, y=1.01,
    )

    out_path = savefig(fig, "fig8_sustained_opponency.png")
    print(f"Saved: {out_path}")

    # ---------------------------------------------------------------- #
    # Save tables
    # ---------------------------------------------------------------- #
    # Decile stats table
    dec_rows = []
    for mid, frac in zip(mid_A, frac_A):
        dec_rows.append({"channel": "green", "ts_bin_mid": mid, "opp_frac": frac})
    for mid, frac in zip(mid_B, frac_B):
        dec_rows.append({"channel": "blue",  "ts_bin_mid": mid, "opp_frac": frac})
    dec_df = pd.DataFrame(dec_rows)
    savetable(dec_df, "fig8_sustained_vs_opponency.csv")
    print("Saved: fig8_sustained_vs_opponency.csv")

    # Group stats table
    grp_out = grp_df.drop(columns=["color"]) if "color" in grp_df.columns else grp_df
    savetable(grp_out, "fig8_sustained_per_group.csv")
    print("Saved: fig8_sustained_per_group.csv")

    # Per-cell sustained features (for external use)
    cell_out = dv[
        ["before_dataset_id", "source_experiment", "improved_tx", "improved_ty",
         "group", "subtype", "is_opponent", "SC_on", "SC_off", "SC_diff",
         "green_ts_ratio", "blue_ts_ratio", "ts_ratio_mean",
         "green_sus_net", "blue_sus_net", "baseline_mean"]
    ].copy()
    savetable(cell_out, "fig8_per_cell_sustained.csv")
    print("Saved: fig8_per_cell_sustained.csv")

    print("Done.")


if __name__ == "__main__":
    main()
