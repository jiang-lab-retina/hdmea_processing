"""
Figure 6 (extra): Contrast-dependent D-V gradients of spectral contrast.

The user's dataset contains low / mid / high contrast variants of each
peak response. This figure asks whether any D-V spectral gradient becomes
stronger at a particular contrast, since Szatko et al. did not report a
contrast-dependence (all their analyses used full-field-flash amplitudes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import _common as C


CONTRASTS = ["low", "mid", "high"]


def _enrich_contrast_sc(df: pd.DataFrame) -> pd.DataFrame:
    for suffix in CONTRASTS:
        g_on = df[f"green_on_peak_extreme_{suffix}"]
        b_on = df[f"blue_on_peak_extreme_{suffix}"]
        g_off = df[f"green_off_peak_extreme_{suffix}"]
        b_off = df[f"blue_off_peak_extreme_{suffix}"]
        df[f"SC_on_{suffix}"] = C.spectral_contrast(g_on, b_on)
        df[f"SC_off_{suffix}"] = C.spectral_contrast(g_off, b_off)
    return df


def _panel(ax, df, feature, color, title) -> dict:
    stats = C.dv_bin_stats(df, feature).dropna(subset=["mean"])
    ax.fill_between(
        stats["center_um"], stats["mean"] - stats["sem"],
        stats["mean"] + stats["sem"],
        color=color, alpha=0.25,
    )
    ax.plot(
        stats["center_um"], stats["mean"], "o-",
        color=color, lw=2, ms=6,
    )
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)

    v = df[feature].to_numpy()
    y = df["Y_um"].to_numpy()
    m = ~np.isnan(v)
    if m.sum() > 2:
        r, p = pearsonr(y[m], v[m])
    else:
        r, p = np.nan, np.nan

    ax.text(
        0.02, 0.98, f"r(Y) = {r:+.3f}\np = {p:.2e}\nn = {m.sum():,}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"),
    )
    ax.set_ylim(-0.5, 0.5)
    C.style_dv_axes(ax, x_label="Y (um)  V <-- --> D")
    ax.set_ylabel(feature)
    ax.set_title(title, fontsize=10)
    return {"feature": feature, "pearson_r_y": r, "pearson_p_y": p,
            "n": int(m.sum())}


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(response_filter=True)
    df = _enrich_contrast_sc(df)
    print(f"[fig6] loaded {len(df):,} rows")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    rows = []
    colors_on = ["#4dabf7", "#1c7ed6", "#0b4884"]
    colors_off = ["#ff8787", "#e03131", "#8c1e1e"]
    for i, suffix in enumerate(CONTRASTS):
        rows.append(_panel(
            axes[0, i], df, f"SC_on_{suffix}", colors_on[i],
            f"A{i+1}. SC_on  {suffix} contrast",
        ))
        rows.append(_panel(
            axes[1, i], df, f"SC_off_{suffix}", colors_off[i],
            f"B{i+1}. SC_off  {suffix} contrast",
        ))

    # Paper reference lines on one panel for context
    for ax in axes.flat:
        ax.axhline(C.PAPER_SC["rgc_ventral_center_mean"], color=C.COLOR_VENTRAL,
                   ls=":", lw=0.8, alpha=0.6)
        ax.axhline(C.PAPER_SC["rgc_dorsal_center_mean"], color=C.COLOR_DORSAL,
                   ls=":", lw=0.8, alpha=0.6)

    fig.suptitle(
        "Fig. 6 (extra): D-V spectral contrast gradient per stimulus contrast",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig6_contrast_breakdown.png")
    print(f"[fig6] saved {out}")

    out_tab = C.savetable(
        pd.DataFrame(rows), "fig6_contrast_sc_correlations.csv",
    )
    print(f"[fig6] saved {out_tab}")


if __name__ == "__main__":
    main()
