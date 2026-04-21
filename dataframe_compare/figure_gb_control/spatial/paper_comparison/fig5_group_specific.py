"""
Figure 5 analog: Cell-type-specific color-opponency.

Mirrors Szatko et al. 2020 Fig. 7a: per functional group, quantify the
fraction of color-opponent cells and compare ventral vs dorsal. Because
ipRGCs are sustained-On, they are the closest counterpart in this dataset
to the paper's sustained-On G24 (alpha) group that was enriched for
color-opponency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

import _common as C


def _group_opponency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in C.GROUP_ORDER:
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            continue
        v = sub[sub["retina_half"] == "ventral"]
        d = sub[sub["retina_half"] == "dorsal"]
        row = {
            "group": grp,
            "n_total": int(len(sub)),
            "frac_opp_total": float(sub["is_opponent"].mean()),
            "n_ventral": int(len(v)),
            "k_opp_ventral": int(v["is_opponent"].sum()),
            "frac_opp_ventral": float(v["is_opponent"].mean()) if len(v) else np.nan,
            "n_dorsal": int(len(d)),
            "k_opp_dorsal": int(d["is_opponent"].sum()),
            "frac_opp_dorsal": float(d["is_opponent"].mean()) if len(d) else np.nan,
        }
        lo_v, hi_v = C.wilson_ci(row["k_opp_ventral"], row["n_ventral"])
        lo_d, hi_d = C.wilson_ci(row["k_opp_dorsal"], row["n_dorsal"])
        row["ci_lo_ventral"] = lo_v
        row["ci_hi_ventral"] = hi_v
        row["ci_lo_dorsal"] = lo_d
        row["ci_hi_dorsal"] = hi_d
        if row["n_ventral"] > 0 and row["n_dorsal"] > 0:
            table = [
                [row["k_opp_ventral"], row["n_ventral"] - row["k_opp_ventral"]],
                [row["k_opp_dorsal"],  row["n_dorsal"]  - row["k_opp_dorsal"]],
            ]
            try:
                _, p = fisher_exact(table)
                row["fisher_p_vd"] = float(p)
            except Exception:
                row["fisher_p_vd"] = np.nan
        else:
            row["fisher_p_vd"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _bar_panel(ax, stats: pd.DataFrame) -> None:
    groups = stats["group"].tolist()
    x = np.arange(len(groups))
    w = 0.35

    v_err = np.vstack([
        stats["frac_opp_ventral"] - stats["ci_lo_ventral"],
        stats["ci_hi_ventral"] - stats["frac_opp_ventral"],
    ])
    d_err = np.vstack([
        stats["frac_opp_dorsal"] - stats["ci_lo_dorsal"],
        stats["ci_hi_dorsal"] - stats["frac_opp_dorsal"],
    ])
    ax.bar(
        x - w / 2, stats["frac_opp_ventral"], w,
        yerr=v_err, color=C.COLOR_VENTRAL, alpha=0.85,
        capsize=3, label="ventral",
    )
    ax.bar(
        x + w / 2, stats["frac_opp_dorsal"], w,
        yerr=d_err, color=C.COLOR_DORSAL, alpha=0.85,
        capsize=3, label="dorsal",
    )

    ax.axhline(
        C.PAPER_SC["rgc_frac_opp_ventral"], color=C.COLOR_VENTRAL,
        ls=":", lw=1.3, alpha=0.8,
        label=f"Paper ventral pooled = {C.PAPER_SC['rgc_frac_opp_ventral']:.3f}",
    )
    ax.axhline(
        C.PAPER_SC["rgc_frac_opp_dorsal"], color=C.COLOR_DORSAL,
        ls=":", lw=1.3, alpha=0.8,
        label=f"Paper dorsal  pooled = {C.PAPER_SC['rgc_frac_opp_dorsal']:.3f}",
    )

    for i, p in enumerate(stats["fisher_p_vd"].values):
        if np.isnan(p):
            continue
        star = ""
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        if star:
            top = max(stats.loc[i, "ci_hi_ventral"] or 0, stats.loc[i, "ci_hi_dorsal"] or 0)
            ax.text(
                i, top + 0.02, star, ha="center", va="bottom", fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(x)
    xticklabels = [
        f"{g}\nn_v={int(v)}, n_d={int(d)}"
        for g, v, d in zip(groups, stats["n_ventral"], stats["n_dorsal"])
    ]
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("fraction ON-OFF color-opponent")
    ax.set_ylim(0, max(0.6, (stats[["ci_hi_ventral", "ci_hi_dorsal"]].max().max() or 0) + 0.1))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        "A. Per-group opponency fraction (ventral vs dorsal)  "
        "Fisher exact *: p<0.05, **: p<0.01, ***: p<0.001",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)


def _heatmap_panel(ax, df) -> pd.DataFrame:
    bins_valid = sorted(b for b in df["dv_bin"].unique() if b >= 0)
    rows = []
    matrix = []
    for grp in C.GROUP_ORDER:
        sub = df[df["group"] == grp]
        row_vals = []
        for bi in bins_valid:
            cells = sub[sub["dv_bin"] == bi]
            frac = cells["is_opponent"].mean() if len(cells) else np.nan
            row_vals.append(frac)
            rows.append({
                "group": grp,
                "bin": int(bi),
                "center_um": float(C.DV_BIN_CENTERS[bi]),
                "n": int(len(cells)),
                "k_opp": int(cells["is_opponent"].sum()) if len(cells) else 0,
                "frac_opp": float(frac) if not np.isnan(frac) else np.nan,
            })
        matrix.append(row_vals)
    matrix = np.array(matrix, dtype=float)

    im = ax.imshow(matrix, cmap=C.CMAP_OPP, vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(bins_valid)))
    ax.set_xticklabels(
        [f"{int(C.DV_BIN_CENTERS[b])}" for b in bins_valid],
        rotation=0, fontsize=9,
    )
    ax.set_yticks(range(len(C.GROUP_ORDER)))
    ax.set_yticklabels(C.GROUP_ORDER, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if v < 0.25 else "white",
                )
    ax.set_xlabel("D-V bin center (um)")
    ax.set_ylabel("group")
    ax.set_title(
        "B. Heatmap: fraction opponent per (group x D-V bin)",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="fraction opponent")
    return pd.DataFrame(rows)


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(drop_empty_group=True, response_filter=True)
    print(f"[fig5] loaded {len(df):,} rows with labeled groups")

    stats = _group_opponency_table(df)
    print(stats.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    _bar_panel(axes[0], stats)
    heatmap_df = _heatmap_panel(axes[1], df)

    fig.suptitle(
        "Fig. 5: Cell-type-specific color opponency "
        "(Szatko 2020 Fig. 7a analog)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = C.savefig(fig, "fig5_group_specific.png")
    print(f"[fig5] saved {out}")

    out1 = C.savetable(stats, "fig5_group_opponency_summary.csv")
    out2 = C.savetable(heatmap_df, "fig5_group_x_bin_opponency.csv")
    print(f"[fig5] saved {out1} and {out2}")


if __name__ == "__main__":
    main()
