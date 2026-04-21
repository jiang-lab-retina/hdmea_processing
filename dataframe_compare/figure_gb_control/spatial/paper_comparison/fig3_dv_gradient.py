"""
Figure 3 analog: D-V binned distributions and mean SC per bin.

Mirrors Szatko et al. 2020 Fig. 6b: for each 0.5 mm D-V bin, overlay the
distribution of center (SC_on) and surround (SC_off) spectral contrast,
plus a mean SC vs Y-position line with SEM shading and paper reference
values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C


def _bin_histograms(ax, df) -> None:
    bins = np.arange(-1.05, 1.06, 0.1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    valid_bins = sorted(b for b in df["dv_bin"].unique() if b >= 0)
    n_bins = len(valid_bins)
    if n_bins == 0:
        return

    yticks = []
    ytick_labels = []
    for row, bi in enumerate(valid_bins):
        sel = df[df["dv_bin"] == bi]
        if len(sel) == 0:
            continue
        h_on, _ = np.histogram(
            sel["SC_on"].dropna(), bins=bins, density=False,
        )
        h_off, _ = np.histogram(
            sel["SC_off"].dropna(), bins=bins, density=False,
        )
        if h_on.max() > 0:
            h_on = h_on / h_on.max() * 0.45
        if h_off.max() > 0:
            h_off = h_off / h_off.max() * 0.45

        y0 = row
        ax.fill_between(
            centers, y0, y0 + h_on, color="#1f77b4", alpha=0.7,
            label="SC_on (ON, 'center')" if row == 0 else None,
        )
        ax.fill_between(
            centers, y0, y0 - h_off, color="#d62728", alpha=0.55,
            label="SC_off (OFF, 'surround')" if row == 0 else None,
        )
        ax.axvline(np.mean(sel["SC_on"].dropna()), ymin=(row) / n_bins,
                   ymax=(row + 0.5) / n_bins, color="#1f77b4", lw=1.2)
        ax.axvline(np.mean(sel["SC_off"].dropna()), ymin=(row) / n_bins,
                   ymax=(row + 0.5) / n_bins, color="#d62728", lw=1.2)

        yticks.append(row + 0.25)
        ytick_labels.append(
            f"bin {bi}: Y~{int(C.DV_BIN_CENTERS[bi])}  n={len(sel)}"
        )

    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, n_bins + 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("Spectral contrast  SC")
    ax.set_ylabel("Ventral  <---  D-V bin  --->  Dorsal")
    ax.set_title(
        "A. Per-bin distributions of SC_on vs SC_off  "
        "(paper Fig. 6b analog)", fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def _mean_sc_vs_y(ax, df) -> pd.DataFrame:
    s_on = C.dv_bin_stats(df, "SC_on").dropna(subset=["mean"])
    s_off = C.dv_bin_stats(df, "SC_off").dropna(subset=["mean"])

    ax.fill_between(
        s_on["center_um"], s_on["mean"] - s_on["sem"],
        s_on["mean"] + s_on["sem"],
        color="#1f77b4", alpha=0.25,
    )
    ax.plot(
        s_on["center_um"], s_on["mean"], "o-", color="#1f77b4",
        lw=2, ms=7, label="User SC_on (mean +/- SEM)",
    )
    ax.fill_between(
        s_off["center_um"], s_off["mean"] - s_off["sem"],
        s_off["mean"] + s_off["sem"],
        color="#d62728", alpha=0.25,
    )
    ax.plot(
        s_off["center_um"], s_off["mean"], "s--", color="#d62728",
        lw=2, ms=7, label="User SC_off (mean +/- SEM)",
    )

    ax.axhline(C.PAPER_SC["rgc_ventral_center_mean"], color=C.COLOR_VENTRAL,
               ls=":", lw=1.2,
               label=f"Paper ventral SC_center = {C.PAPER_SC['rgc_ventral_center_mean']}")
    ax.axhline(C.PAPER_SC["rgc_dorsal_center_mean"], color=C.COLOR_DORSAL,
               ls=":", lw=1.2,
               label=f"Paper dorsal SC_center = {C.PAPER_SC['rgc_dorsal_center_mean']}")
    ax.axhline(C.PAPER_SC["rgc_ventral_surround_mean"], color=C.COLOR_VENTRAL,
               ls="-.", lw=1.0, alpha=0.8,
               label=f"Paper ventral SC_surround = {C.PAPER_SC['rgc_ventral_surround_mean']}")

    ax.set_ylim(-0.5, 0.5)
    C.style_dv_axes(ax, x_label="Y (um)   V (ventral) <-- --> D (dorsal)")
    ax.set_ylabel("mean SC across cells in each D-V bin")
    ax.set_title(
        "B. Mean SC vs Y position  with paper reference values",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    stats_out = pd.concat(
        [s_on.assign(feature="SC_on"), s_off.assign(feature="SC_off")],
        ignore_index=True,
    )
    return stats_out


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(response_filter=True)
    print(f"[fig3] loaded {len(df):,} rows")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _bin_histograms(axes[0], df)
    stats = _mean_sc_vs_y(axes[1], df)

    fig.suptitle(
        "Fig. 3: D-V gradient of spectral contrast (Szatko 2020 Fig. 6b analog)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig3_dv_gradient.png")
    print(f"[fig3] saved {out}")

    out_tab = C.savetable(stats, "fig3_dv_sc_stats.csv")
    print(f"[fig3] saved {out_tab}")


if __name__ == "__main__":
    main()
