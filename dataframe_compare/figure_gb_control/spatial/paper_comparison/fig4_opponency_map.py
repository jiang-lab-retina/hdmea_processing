"""
Figure 4 analog: ON-OFF color-opponency spatial distribution.

Mirrors Szatko et al. 2020 Fig. 6c-d: spatial layout of opponent cells,
fraction of opponent cells per D-V bin with Wilson confidence intervals,
and distribution of SC_diff (ON vs OFF spectral mismatch) across
ventral/dorsal retina.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C


def _spatial_opponent_scatter(ax, df) -> None:
    ax.scatter(
        df.loc[~df["is_opponent"], "X_um"],
        df.loc[~df["is_opponent"], "Y_um"],
        s=2, c="lightgray", alpha=0.35, edgecolor="none",
        label=f"non-opponent  (n={(~df['is_opponent']).sum():,})",
        rasterized=True,
    )
    ax.scatter(
        df.loc[df["is_opponent"], "X_um"],
        df.loc[df["is_opponent"], "Y_um"],
        s=4, c=C.COLOR_OPP, alpha=0.75, edgecolor="none",
        label=f"opponent  (n={(df['is_opponent']).sum():,})",
        rasterized=True,
    )
    C.style_xy_axes(
        ax,
        title=f"A. Spatial distribution of ON-OFF color-opponent cells\n"
              f"(threshold |SC_on - SC_off| > {C.OPP_THRESHOLD})",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)


def _dv_fraction(ax, df) -> pd.DataFrame:
    frac = C.dv_bin_fraction(df, "is_opponent").dropna(subset=["fraction"])

    err_lo = frac["fraction"] - frac["ci_lo"]
    err_hi = frac["ci_hi"] - frac["fraction"]
    ax.errorbar(
        frac["center_um"], frac["fraction"],
        yerr=[err_lo, err_hi],
        fmt="o-", color="#333333", lw=2, ms=7, capsize=4,
        label="User: fraction opponent (95% Wilson CI)",
    )

    ax.axhline(
        C.PAPER_SC["rgc_frac_opp_ventral"], color=C.COLOR_VENTRAL,
        ls=":", lw=1.3,
        label=f"Paper ventral frac = {C.PAPER_SC['rgc_frac_opp_ventral']:.3f}",
    )
    ax.axhline(
        C.PAPER_SC["rgc_frac_opp_dorsal"], color=C.COLOR_DORSAL,
        ls=":", lw=1.3,
        label=f"Paper dorsal frac = {C.PAPER_SC['rgc_frac_opp_dorsal']:.3f}",
    )

    v_sel = df.loc[df["retina_half"] == "ventral", "is_opponent"]
    d_sel = df.loc[df["retina_half"] == "dorsal", "is_opponent"]
    frac_v = v_sel.mean() if len(v_sel) else np.nan
    frac_d = d_sel.mean() if len(d_sel) else np.nan
    ax.text(
        0.02, 0.98,
        f"User ventral  frac = {frac_v:.3f}  (n={len(v_sel):,})\n"
        f"User dorsal   frac = {frac_d:.3f}  (n={len(d_sel):,})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"),
    )

    C.style_dv_axes(ax, x_label="Y (um)   V (ventral) <-- --> D (dorsal)")
    ax.set_ylabel("fraction of cells ON-OFF color-opponent")
    ax.set_ylim(0, max(0.6, frac["ci_hi"].max() * 1.1 + 0.05))
    ax.set_title(
        "B. Opponent fraction vs Y position  (paper Fig. 6c-d analog)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    return frac


def _sc_diff_histograms(ax, df) -> None:
    bins = np.linspace(-2.0, 2.0, 81)
    for half, color in [("ventral", C.COLOR_VENTRAL), ("dorsal", C.COLOR_DORSAL)]:
        vals = df.loc[df["retina_half"] == half, "SC_diff"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(
            vals, bins=bins, density=True, alpha=0.5,
            color=color,
            label=f"{half}  (n={len(vals):,}, mean={vals.mean():+.3f})",
        )
    ax.axvline(-C.OPP_THRESHOLD, color="black", ls="--", lw=1.0)
    ax.axvline(+C.OPP_THRESHOLD, color="black", ls="--", lw=1.0,
               label=f"|SC_diff| = {C.OPP_THRESHOLD} (opponency threshold)")
    ax.set_xlabel("SC_on - SC_off")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "C. Distribution of SC_diff by retinal half",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(response_filter=True)
    print(f"[fig4] loaded {len(df):,} rows "
          f"(opponent frac = {df['is_opponent'].mean():.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    _spatial_opponent_scatter(axes[0], df)
    frac_df = _dv_fraction(axes[1], df)
    _sc_diff_histograms(axes[2], df)

    fig.suptitle(
        "Fig. 4: ON-OFF color opponency spatial distribution "
        "(Szatko 2020 Fig. 6c-d analog)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig4_opponency_map.png")
    print(f"[fig4] saved {out}")

    out_tab = C.savetable(frac_df, "fig4_opponency_dv_fractions.csv")
    print(f"[fig4] saved {out_tab}")


if __name__ == "__main__":
    main()
