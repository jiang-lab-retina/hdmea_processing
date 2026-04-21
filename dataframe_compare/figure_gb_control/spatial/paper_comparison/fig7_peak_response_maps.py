"""
Figure 7 (extra): Raw green/blue ON/OFF peak hexbin maps.

Complements the SC-based figures by showing the absolute response
amplitudes in each channel, so that a weak SC gradient can be traced to
either channel individually. Mirrors the style used in
`gb_spatial_control/spatial_plots.py` (hexbin, coolwarm, gridsize=25).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C


PEAK_FEATURES = [
    ("green_on_peak_extreme",  "green ON peak",  "#2ca02c"),
    ("blue_on_peak_extreme",   "blue  ON peak",  "#1f77b4"),
    ("green_off_peak_extreme", "green OFF peak", "#006d2c"),
    ("blue_off_peak_extreme",  "blue  OFF peak", "#08306b"),
]


def _hexbin(ax, x, y, c, feature, gridsize=25, mincnt=3) -> None:
    vmin, vmax = np.nanpercentile(c, [5, 95])
    if vmin == vmax:
        vmin, vmax = float(np.nanmin(c)), float(np.nanmax(c))
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
        extent=(*C.XY_RANGE, *C.XY_RANGE), mincnt=mincnt,
        cmap="coolwarm", vmin=vmin, vmax=vmax,
    )
    C.style_xy_axes(ax, title=f"{feature}\nn = {len(c):,}")
    cbar = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("mean peak amplitude", fontsize=9)


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(response_filter=True)
    print(f"[fig7] loaded {len(df):,} rows")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    rows = []
    for ax, (feat, label, _c) in zip(axes.flat, PEAK_FEATURES):
        sub = df.dropna(subset=[feat, "X_um", "Y_um"])
        x = sub["X_um"].to_numpy()
        y = sub["Y_um"].to_numpy()
        c = sub[feat].to_numpy()
        _hexbin(ax, x, y, c, label)

        stats = C.dv_bin_stats(sub, feat).dropna(subset=["mean"])
        stats["feature"] = feat
        rows.append(stats)

    fig.suptitle(
        "Fig. 7 (extra): Raw peak amplitude hexbins for each chromatic channel",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig7_peak_response_maps.png")
    print(f"[fig7] saved {out}")

    ventral_dorsal = []
    for feat, label, _ in PEAK_FEATURES:
        v = df.loc[df["retina_half"] == "ventral", feat].dropna()
        d = df.loc[df["retina_half"] == "dorsal", feat].dropna()
        ventral_dorsal.append({
            "feature": feat,
            "ventral_mean": float(v.mean()) if len(v) else np.nan,
            "ventral_n": int(len(v)),
            "dorsal_mean": float(d.mean()) if len(d) else np.nan,
            "dorsal_n": int(len(d)),
            "dorsal_minus_ventral": float(d.mean() - v.mean()) if len(d) and len(v) else np.nan,
        })
    out_tab = C.savetable(pd.DataFrame(ventral_dorsal), "fig7_peak_vd_means.csv")
    print(f"[fig7] saved {out_tab}")

    out_tab2 = C.savetable(pd.concat(rows, ignore_index=True), "fig7_peak_dv_bin_stats.csv")
    print(f"[fig7] saved {out_tab2}")


if __name__ == "__main__":
    main()
