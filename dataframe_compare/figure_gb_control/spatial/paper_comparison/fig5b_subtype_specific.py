"""
Figure 5b: Subtype-level color-opponency analysis.

Extends the group-level analysis of fig5_group_specific.py by splitting
cells into their 33 labeled subtypes (e.g. ipRGC_4, DSGC_8, Other_0).
This mirrors Szatko et al. 2020 Fig. 7a at a finer granularity: the paper
shows that opponency varies strongly across functional RGC groups, so we
ask whether the same is true across subtypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, pearsonr

import _common as C


def _subtype_opponency_table(df: pd.DataFrame, subtypes: list[str]) -> pd.DataFrame:
    rows = []
    for st in subtypes:
        sub = df[df["subtype"] == st]
        if len(sub) < C.MIN_SUBTYPE_N:
            continue
        v = sub[sub["retina_half"] == "ventral"]
        d = sub[sub["retina_half"] == "dorsal"]
        row = {
            "subtype": st,
            "parent_group": C.parent_group(st),
            "n_total": int(len(sub)),
            "frac_opp_total": float(sub["is_opponent"].mean()),
            "SC_on_mean": float(sub["SC_on"].mean()),
            "SC_off_mean": float(sub["SC_off"].mean()),
            "n_ventral": int(len(v)),
            "k_opp_ventral": int(v["is_opponent"].sum()),
            "frac_opp_ventral": float(v["is_opponent"].mean()) if len(v) else np.nan,
            "n_dorsal": int(len(d)),
            "k_opp_dorsal": int(d["is_opponent"].sum()),
            "frac_opp_dorsal": float(d["is_opponent"].mean()) if len(d) else np.nan,
        }
        lo_v, hi_v = C.wilson_ci(row["k_opp_ventral"], row["n_ventral"])
        lo_d, hi_d = C.wilson_ci(row["k_opp_dorsal"], row["n_dorsal"])
        row["ci_lo_total"] = C.wilson_ci(
            int(sub["is_opponent"].sum()), row["n_total"]
        )[0]
        row["ci_hi_total"] = C.wilson_ci(
            int(sub["is_opponent"].sum()), row["n_total"]
        )[1]
        row["ci_lo_ventral"] = lo_v
        row["ci_hi_ventral"] = hi_v
        row["ci_lo_dorsal"] = lo_d
        row["ci_hi_dorsal"] = hi_d
        if row["n_ventral"] > 0 and row["n_dorsal"] > 0:
            table = [
                [row["k_opp_ventral"], row["n_ventral"] - row["k_opp_ventral"]],
                [row["k_opp_dorsal"], row["n_dorsal"] - row["k_opp_dorsal"]],
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
    """Horizontal bar chart sorted by opponency fraction."""
    stats_sorted = stats.sort_values("frac_opp_total").reset_index(drop=True)
    y_pos = np.arange(len(stats_sorted))

    colors = [C.GROUP_COLORS.get(g, "#808080") for g in stats_sorted["parent_group"]]
    err_lo = np.maximum(0, stats_sorted["frac_opp_total"] - stats_sorted["ci_lo_total"])
    err_hi = np.maximum(0, stats_sorted["ci_hi_total"] - stats_sorted["frac_opp_total"])

    ax.barh(
        y_pos, stats_sorted["frac_opp_total"],
        xerr=[err_lo, err_hi],
        color=colors, alpha=0.85, capsize=2, height=0.7,
    )

    for i, row in stats_sorted.iterrows():
        p = row["fisher_p_vd"]
        if not np.isnan(p) and p < 0.05:
            star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
            ax.text(
                row["ci_hi_total"] + 0.01, i, star,
                va="center", ha="left", fontsize=8, fontweight="bold",
            )

    ax.set_yticks(y_pos)
    labels = [
        f"{row['subtype']}  (n={row['n_total']})"
        for _, row in stats_sorted.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("fraction ON-OFF color-opponent")
    ax.set_xlim(0, min(1.0, stats_sorted["ci_hi_total"].max() + 0.1))
    ax.grid(True, axis="x", alpha=0.3)

    ax.axvline(C.PAPER_SC["rgc_frac_opp_ventral"], color=C.COLOR_VENTRAL,
               ls=":", lw=1.0, alpha=0.8)
    ax.axvline(C.PAPER_SC["rgc_frac_opp_dorsal"], color=C.COLOR_DORSAL,
               ls=":", lw=1.0, alpha=0.8)

    for g in C.GROUP_ORDER:
        ax.barh([], [], color=C.GROUP_COLORS[g], label=g)
    ax.legend(loc="lower right", fontsize=8, title="Parent group")
    ax.set_title(
        "A. Opponency fraction per subtype (sorted)\n"
        "Fisher *: p<0.05 (ventral vs dorsal)",
        fontsize=10,
    )


def _heatmap_panel(ax, df: pd.DataFrame, subtypes: list[str]) -> pd.DataFrame:
    """Heatmap: subtype x D-V bin, colored by opponent fraction."""
    bins_valid = sorted(b for b in df["dv_bin"].unique() if b >= 0)
    filtered_subs = [
        s for s in subtypes
        if len(df[df["subtype"] == s]) >= C.MIN_SUBTYPE_N
    ]
    sub_fracs = {}
    for st in filtered_subs:
        sub = df[df["subtype"] == st]
        sub_fracs[st] = sub["is_opponent"].mean() if len(sub) else 0.0

    sorted_subs = sorted(
        filtered_subs,
        key=lambda s: (
            C.GROUP_ORDER.index(C.parent_group(s))
            if C.parent_group(s) in C.GROUP_ORDER else 99,
            -sub_fracs.get(s, 0),
        ),
    )

    rows = []
    matrix = []
    for st in sorted_subs:
        sub = df[df["subtype"] == st]
        row_vals = []
        for bi in bins_valid:
            cells = sub[sub["dv_bin"] == bi]
            frac = cells["is_opponent"].mean() if len(cells) else np.nan
            row_vals.append(frac)
            rows.append({
                "subtype": st,
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
        rotation=0, fontsize=8,
    )
    ax.set_yticks(range(len(sorted_subs)))

    ytick_colors = [C.GROUP_COLORS.get(C.parent_group(s), "#333") for s in sorted_subs]
    ax.set_yticklabels(sorted_subs, fontsize=6.5)
    for ticklabel, color in zip(ax.get_yticklabels(), ytick_colors):
        ticklabel.set_color(color)

    ax.set_xlabel("D-V bin center (um)")
    ax.set_ylabel("subtype")
    ax.set_title(
        "B. Fraction opponent per (subtype x D-V bin)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="fraction opponent")
    return pd.DataFrame(rows)


def _scatter_panel(ax, stats: pd.DataFrame) -> None:
    """SC_on mean vs opponency fraction, one point per subtype."""
    for g in C.GROUP_ORDER:
        mask = stats["parent_group"] == g
        if not mask.any():
            continue
        sub = stats[mask]
        ax.scatter(
            sub["SC_on_mean"], sub["frac_opp_total"],
            c=C.GROUP_COLORS[g], s=sub["n_total"].clip(upper=300) * 0.5,
            alpha=0.8, edgecolor="white", lw=0.5,
            label=g, zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["subtype"].split("_")[1],
                (row["SC_on_mean"], row["frac_opp_total"]),
                fontsize=6, ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 4),
            )

    x = stats["SC_on_mean"].to_numpy()
    y = stats["frac_opp_total"].to_numpy()
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() > 2:
        r, p = pearsonr(x[m], y[m])
        ax.text(
            0.02, 0.98,
            f"r = {r:+.3f}, p = {p:.2e}\nn = {m.sum()} subtypes",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"),
        )

    ax.set_xlabel("mean SC_on per subtype")
    ax.set_ylabel("fraction opponent per subtype")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, title="Parent group")
    ax.set_title(
        "C. SC_on vs opponency fraction\n(size ~ n cells)",
        fontsize=10,
    )


def main() -> None:
    C.ensure_dirs()
    df = C.load_combined(drop_empty_group=True, response_filter=True)
    subtypes = C.get_subtype_order(df)
    print(f"[fig5b] loaded {len(df):,} labeled cells, "
          f"{len(subtypes)} subtypes")

    stats = _subtype_opponency_table(df, subtypes)
    print(f"[fig5b] {len(stats)} subtypes with n >= {C.MIN_SUBTYPE_N}")
    print(stats.sort_values("frac_opp_total", ascending=False)[
        ["subtype", "parent_group", "n_total", "frac_opp_total",
         "frac_opp_ventral", "frac_opp_dorsal", "fisher_p_vd"]
    ].to_string(index=False))

    n_subtypes = len(stats)
    fig_h = max(8, n_subtypes * 0.32 + 2)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, fig_h),
        gridspec_kw={"width_ratios": [1.0, 0.8, 0.7], "wspace": 0.35},
    )

    _bar_panel(axes[0], stats)
    heatmap_df = _heatmap_panel(axes[1], df, subtypes)
    _scatter_panel(axes[2], stats)

    fig.suptitle(
        "Fig. 5b: Subtype-level color opponency "
        "(Szatko 2020 Fig. 7a analog, fine-grained)",
        fontsize=13, y=1.01,
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06)
    out = C.savefig(fig, "fig5b_subtype_specific.png")
    print(f"[fig5b] saved {out}")

    out1 = C.savetable(stats, "fig5b_subtype_opponency_summary.csv")
    out2 = C.savetable(heatmap_df, "fig5b_subtype_x_bin_opponency.csv")
    print(f"[fig5b] saved {out1} and {out2}")


if __name__ == "__main__":
    main()
