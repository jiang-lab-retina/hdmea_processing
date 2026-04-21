"""
Figure 1 analog: Opsin gradient + user's chromatic preference along D-V.

Mirrors Szatko et al. 2020 Fig. 1b (schematic opsin distribution) alongside
the observed mean paper-style spectral contrast (SC_on) as a function of
the retinal Y (dorsal-ventral) axis in the user's data.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

import _common as C


def _draw_schematic(ax) -> None:
    """Redraw Szatko et al. Fig. 1b schematic: a disc retina with a
    ventral-to-dorsal gradient in S/M-opsin co-expression, plus D/V/N/T
    axis annotations.
    """
    cmap = LinearSegmentedColormap.from_list(
        "opsin", ["#C71585", "#E0E0E0", "#2CA02C"], N=256,
    )
    n = 256
    y = np.linspace(-1.0, 1.0, n)
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    mask = (X ** 2 + Y ** 2) <= 1.0
    img = np.where(mask, Y, np.nan)
    ax.imshow(
        img, extent=(-1, 1, -1, 1), origin="lower",
        cmap=cmap, vmin=-1, vmax=1, aspect="equal",
    )
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="black", lw=1.5))

    rng = np.random.default_rng(7)
    n_dots = 70
    r = np.sqrt(rng.random(n_dots)) * 0.95
    th = rng.random(n_dots) * 2 * np.pi
    xd, yd = r * np.cos(th), r * np.sin(th)
    ax.scatter(xd, yd, s=10, c="magenta", edgecolor="white",
               lw=0.4, alpha=0.85, zorder=3, label="True S-cone")

    ax.annotate("D (dorsal)", xy=(0, 1.02), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.annotate("V (ventral)", xy=(0, -1.02), ha="center", va="top",
                fontsize=10, fontweight="bold")
    ax.annotate("N", xy=(1.02, 0), ha="left", va="center",
                fontsize=10, fontweight="bold")
    ax.annotate("T", xy=(-1.02, 0), ha="right", va="center",
                fontsize=10, fontweight="bold")

    ax.text(0.85, 0.75, "M-opsin\n(green)", ha="center", va="center",
            fontsize=9, color="#1B5E20")
    ax.text(0.85, -0.75, "S-opsin\n(UV)", ha="center", va="center",
            fontsize=9, color="#880E4F")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Paper Fig. 1b (schematic)\nS/M-opsin co-expression increases ventrally",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=8, frameon=False)


def _draw_user_sc_vs_y(ax, df) -> None:
    """SC_on (and SC_off) means per D-V bin with SEM shading, overlaid on
    the per-cell scatter."""
    y = df["Y_um"].to_numpy()
    sc_on = df["SC_on"].to_numpy()

    rng = np.random.default_rng(0)
    subset = rng.choice(len(df), size=min(6000, len(df)), replace=False)
    ax.scatter(
        y[subset], sc_on[subset], s=2, c="lightsteelblue", alpha=0.35,
        zorder=1, rasterized=True,
    )

    stats_on = C.dv_bin_stats(df, "SC_on").dropna(subset=["mean"])
    stats_off = C.dv_bin_stats(df, "SC_off").dropna(subset=["mean"])

    ax.errorbar(
        stats_on["center_um"], stats_on["mean"], yerr=stats_on["sem"],
        fmt="o-", color="#1f77b4", lw=2, ms=6, capsize=3,
        label="User SC_on (mean +/- SEM)", zorder=3,
    )
    ax.errorbar(
        stats_off["center_um"], stats_off["mean"], yerr=stats_off["sem"],
        fmt="s--", color="#d62728", lw=1.5, ms=5, capsize=3,
        label="User SC_off (mean +/- SEM)", zorder=3,
    )

    ax.axhline(
        C.PAPER_SC["rgc_ventral_center_mean"], color=C.COLOR_VENTRAL,
        ls=":", lw=1.2,
        label=f"Paper: ventral SC_center = {C.PAPER_SC['rgc_ventral_center_mean']}",
    )
    ax.axhline(
        C.PAPER_SC["rgc_dorsal_center_mean"], color=C.COLOR_DORSAL,
        ls=":", lw=1.2,
        label=f"Paper: dorsal SC_center = {C.PAPER_SC['rgc_dorsal_center_mean']}",
    )

    r, p = pearsonr(y[~np.isnan(sc_on)], sc_on[~np.isnan(sc_on)])
    ax.text(
        0.02, 0.98,
        f"Pearson r(Y, SC_on) = {r:+.3f}  (p={p:.1e}, n={np.sum(~np.isnan(sc_on)):,})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"),
    )

    C.style_dv_axes(ax, x_label="Y (um)   V (ventral) <-- --> D (dorsal)")
    ax.set_ylabel("Spectral contrast  SC = (G - B) / (|G| + |B|)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        "User data: spectral contrast vs Y position",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True, ncol=1)


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(response_filter=True)
    print(f"[fig1] loaded {len(df):,} rows")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _draw_schematic(axes[0])
    _draw_user_sc_vs_y(axes[1], df)

    fig.suptitle(
        "Fig. 1: Opsin gradient (Szatko 2020 Fig. 1b) vs user RGC spectral "
        "contrast (paper analog)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig1_opsin_gradient.png")
    print(f"[fig1] saved {out}")

    on_stats = C.dv_bin_stats(df, "SC_on")
    on_stats["feature"] = "SC_on"
    off_stats = C.dv_bin_stats(df, "SC_off")
    off_stats["feature"] = "SC_off"
    import pandas as pd
    out_tab = C.savetable(
        pd.concat([on_stats, off_stats], ignore_index=True),
        "fig1_dv_bin_sc_stats.csv",
    )
    print(f"[fig1] saved {out_tab}")


if __name__ == "__main__":
    main()
