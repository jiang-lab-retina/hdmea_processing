"""
Figure 2 analog: Spatial maps of chromatic preference.

Mirrors Szatko et al. 2020 Fig. 6a: per-cell spatial scatter color-coded
by spectral contrast (center / surround / full-field opponency). The user
data has no separate center/surround stimulus, so we substitute SC_on and
SC_off as the two chromatic channels and use the ON-OFF SC_diff as the
color-opponency analog.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import _common as C


def _scatter_panel(ax, df, col, vmin=-1, vmax=1, cmap=C.CMAP_SC,
                   title="", cbar_label="", s=3, subsample=None):
    x = df["X_um"].to_numpy()
    y = df["Y_um"].to_numpy()
    c = df[col].to_numpy()

    if subsample is not None and len(df) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=subsample, replace=False)
        x, y, c = x[idx], y[idx], c[idx]

    order = np.argsort(np.abs(c))
    sc = ax.scatter(
        x[order], y[order], c=c[order], s=s, cmap=cmap,
        vmin=vmin, vmax=vmax, alpha=0.75, edgecolor="none",
        rasterized=True,
    )
    C.style_xy_axes(ax, title=title)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _hexbin_panel(ax, df, col, vmin=-1, vmax=1, gridsize=30,
                  cmap=C.CMAP_SC, title="", cbar_label=""):
    x = df["X_um"].to_numpy()
    y = df["Y_um"].to_numpy()
    c = df[col].to_numpy()
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
        extent=(*C.XY_RANGE, *C.XY_RANGE), mincnt=3, cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    C.style_xy_axes(ax, title=title)
    cbar = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(response_filter=True)
    print(f"[fig2] loaded {len(df):,} rows")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))

    _scatter_panel(
        axes[0, 0], df, "SC_on",
        title=f"A. SC_on (ON-phase spectral contrast)\nn = {df['SC_on'].notna().sum():,}",
        cbar_label="SC_on   (blue < 0 < green)",
        subsample=15000,
    )
    _scatter_panel(
        axes[0, 1], df, "SC_off",
        title=f"B. SC_off (OFF-phase spectral contrast)\nn = {df['SC_off'].notna().sum():,}",
        cbar_label="SC_off   (blue < 0 < green)",
        subsample=15000,
    )

    opp_df = df.copy()
    opp_df["opp_flag"] = opp_df["is_opponent"].astype(float)
    opp_mean = opp_df["opp_flag"].mean()
    _scatter_panel(
        axes[0, 2], opp_df, "opp_flag",
        vmin=0, vmax=1, cmap=C.CMAP_OPP,
        title=(
            f"C. ON-OFF color opponency (|SC_on - SC_off| > {C.OPP_THRESHOLD})\n"
            f"opponent fraction = {opp_mean:.3f}"
        ),
        cbar_label="opponent (1) vs non-opponent (0)",
        s=3, subsample=15000,
    )

    _hexbin_panel(
        axes[1, 0], df, "SC_on",
        title="D. Hexbin mean SC_on",
        cbar_label="mean SC_on",
    )
    _hexbin_panel(
        axes[1, 1], df, "SC_off",
        title="E. Hexbin mean SC_off",
        cbar_label="mean SC_off",
    )
    _hexbin_panel(
        axes[1, 2], opp_df, "opp_flag",
        vmin=0, vmax=0.5, cmap=C.CMAP_OPP,
        title=f"F. Hexbin fraction ON-OFF color-opponent (threshold {C.OPP_THRESHOLD})",
        cbar_label="opponent fraction",
    )

    fig.suptitle(
        "Fig. 2: Spatial maps of chromatic preference "
        "(Szatko 2020 Fig. 6a analog)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig2_spatial_maps.png")
    print(f"[fig2] saved {out}")


if __name__ == "__main__":
    main()
