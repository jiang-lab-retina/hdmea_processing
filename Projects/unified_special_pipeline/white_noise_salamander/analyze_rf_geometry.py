"""
RF Geometry Analysis: Young vs Old Larval Salamander
=====================================================

Compares receptive-field geometry from white-noise STA between:
  - Young larval (2024.03.04 recordings)
  - Old   larval (2026.03.03 recordings)

Quality gating is based on the Gaussian-fit R^2 (default >= 0.5).
Within each stimulus resolution (15x15 / 32x32) the two age groups
are compared on RF area, equivalent diameter, Gaussian sigma, and
several other metrics.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "rf_geometry_analysis"

AGE_GROUPS: Dict[str, str] = {
    "2024.03.04": "Young larval",
    "2026.03.03": "Old larval",
}

R2_THRESHOLD_DEFAULT = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

@dataclass
class UnitGeometry:
    recording: str
    unit_id: str
    age_group: str
    stimulus: str          # e.g. "15x15_5hz" or "32x32_15hz"
    stimulus_full: str     # full feature key
    area: float
    equivalent_diameter: float
    center_row: float
    center_col: float
    size_x: float
    size_y: float
    peak_frame: int
    gauss_r2: float
    gauss_sigma_x: float
    gauss_sigma_y: float
    gauss_sigma_mean: float
    gauss_amplitude: float
    gauss_theta: float
    dog_r2: float
    dog_sigma_exc: float
    dog_sigma_inh: float
    dog_amp_exc: float
    dog_amp_inh: float
    surround_strength: float   # |amp_inh| / |amp_exc|
    on_r2: float
    off_r2: float
    on_amplitude: float
    off_amplitude: float
    on_off_ratio: float        # |on_amp| / (|on_amp| + |off_amp|)
    lnl_bits_per_spike: float
    lnl_r2: float
    lnl_rectification_index: float
    lnl_nonlinearity_index: float
    sta_time_course: Optional[np.ndarray] = field(default=None, repr=False)


def _parse_stimulus_tag(feature_key: str) -> str:
    """'sta_perfect_dense_noise_15x15_5hz_r42_10min' -> '15x15_5hz'."""
    m = re.search(r"(\d+x\d+)_(\d+hz)", feature_key)
    return f"{m.group(1)}_{m.group(2)}" if m else feature_key


def _safe_scalar(grp, key, default=np.nan):
    if key in grp:
        return float(grp[key][()])
    return default


def extract_units_from_h5(h5_path: Path, age_group: str) -> List[UnitGeometry]:
    """Read all units with sta_geometry from a single HDF5 file."""
    rec_name = h5_path.stem
    units: List[UnitGeometry] = []

    with h5py.File(str(h5_path), "r") as f:
        units_grp = f.get("units")
        if units_grp is None:
            return units

        for uid in units_grp:
            feats = units_grp[uid].get("features")
            if feats is None:
                continue
            for fk in feats:
                geom = feats[fk].get("sta_geometry")
                if geom is None:
                    continue

                gf = geom.get("gaussian_fit", {})
                dog = geom.get("DoG", {})
                oo = geom.get("ONOFF_model", {})
                lnl = geom.get("lnl", {})

                gauss_sx = _safe_scalar(gf, "sigma_x")
                gauss_sy = _safe_scalar(gf, "sigma_y")
                on_amp = _safe_scalar(oo, "on_amplitude")
                off_amp = _safe_scalar(oo, "off_amplitude")
                dog_aexc = _safe_scalar(dog, "amp_exc")
                dog_ainh = _safe_scalar(dog, "amp_inh")

                on_off_denom = abs(on_amp) + abs(off_amp)
                on_off_ratio = abs(on_amp) / on_off_denom if on_off_denom > 0 else np.nan
                surround = abs(dog_ainh) / abs(dog_aexc) if abs(dog_aexc) > 0 else np.nan

                stc = None
                if "sta_time_course" in geom:
                    stc = geom["sta_time_course"][()]

                units.append(UnitGeometry(
                    recording=rec_name,
                    unit_id=uid,
                    age_group=age_group,
                    stimulus=_parse_stimulus_tag(fk),
                    stimulus_full=fk,
                    area=_safe_scalar(geom, "area"),
                    equivalent_diameter=_safe_scalar(geom, "equivalent_diameter"),
                    center_row=_safe_scalar(geom, "center_row"),
                    center_col=_safe_scalar(geom, "center_col"),
                    size_x=_safe_scalar(geom, "size_x"),
                    size_y=_safe_scalar(geom, "size_y"),
                    peak_frame=int(_safe_scalar(geom, "peak_frame", -1)),
                    gauss_r2=_safe_scalar(gf, "r_squared"),
                    gauss_sigma_x=gauss_sx,
                    gauss_sigma_y=gauss_sy,
                    gauss_sigma_mean=np.sqrt(gauss_sx * gauss_sy),
                    gauss_amplitude=_safe_scalar(gf, "amplitude"),
                    gauss_theta=_safe_scalar(gf, "theta"),
                    dog_r2=_safe_scalar(dog, "r_squared"),
                    dog_sigma_exc=_safe_scalar(dog, "sigma_exc"),
                    dog_sigma_inh=_safe_scalar(dog, "sigma_inh"),
                    dog_amp_exc=dog_aexc,
                    dog_amp_inh=dog_ainh,
                    surround_strength=surround,
                    on_r2=_safe_scalar(oo, "on_r_squared"),
                    off_r2=_safe_scalar(oo, "off_r_squared"),
                    on_amplitude=on_amp,
                    off_amplitude=off_amp,
                    on_off_ratio=on_off_ratio,
                    lnl_bits_per_spike=_safe_scalar(lnl, "bits_per_spike"),
                    lnl_r2=_safe_scalar(lnl, "r_squared"),
                    lnl_rectification_index=_safe_scalar(lnl, "rectification_index"),
                    lnl_nonlinearity_index=_safe_scalar(lnl, "nonlinearity_index"),
                    sta_time_course=stc,
                ))
    return units


def load_all_units(output_dir: Path) -> pd.DataFrame:
    """Load geometry data from all HDF5 files and return as DataFrame."""
    all_units: List[UnitGeometry] = []

    for h5_path in sorted(output_dir.glob("*.h5")):
        date_prefix = h5_path.stem[:10]
        age = AGE_GROUPS.get(date_prefix)
        if age is None:
            logger.warning("Skipping %s -- date prefix not in AGE_GROUPS", h5_path.name)
            continue
        units = extract_units_from_h5(h5_path, age)
        logger.info("  %s: %d units  (%s)", h5_path.name, len(units), age)
        all_units.extend(units)

    cols_no_stc = [f.name for f in UnitGeometry.__dataclass_fields__.values()
                   if f.name != "sta_time_course"]
    rows = [{c: getattr(u, c) for c in cols_no_stc} for u in all_units]
    df = pd.DataFrame(rows)
    logger.info("Total: %d units from %d files", len(df), df["recording"].nunique())
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compare_groups(
    df: pd.DataFrame,
    column: str,
    group_col: str = "age_group",
) -> Dict:
    """Mann-Whitney U test between the two age groups for *column*."""
    groups = sorted(df[group_col].unique())
    if len(groups) != 2:
        return {}
    a = df.loc[df[group_col] == groups[0], column].dropna()
    b = df.loc[df[group_col] == groups[1], column].dropna()
    if len(a) < 3 or len(b) < 3:
        return {}
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "column": column,
        "group_a": groups[0],
        "n_a": len(a),
        "median_a": np.median(a),
        "mean_a": np.mean(a),
        "group_b": groups[1],
        "n_b": len(b),
        "median_b": np.median(b),
        "mean_b": np.mean(b),
        "U_stat": stat,
        "p_value": p,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {"Young larval": "#3498db", "Old larval": "#e74c3c"}


def _set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })


def plot_r2_distributions(df_raw: pd.DataFrame, fig_dir: Path):
    """Histogram of Gaussian R2 by age group, before thresholding."""
    _set_style()
    stims = sorted(df_raw["stimulus"].unique())

    fig, axes = plt.subplots(1, len(stims), figsize=(5 * len(stims), 4), squeeze=False)
    for ax, stim in zip(axes[0], stims):
        sub = df_raw[df_raw["stimulus"] == stim]
        for ag in sorted(sub["age_group"].unique()):
            vals = sub.loc[sub["age_group"] == ag, "gauss_r2"]
            ax.hist(vals, bins=25, alpha=0.55, label=ag, color=PALETTE.get(ag))
        ax.set_xlabel("Gaussian fit $R^2$")
        ax.set_ylabel("Count")
        ax.set_title(stim)
        ax.legend(frameon=False)
    fig.suptitle("Gaussian $R^2$ distribution (all units, before filtering)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "r2_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_violin_comparisons(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str]],
    stim_tag: str,
    fig_dir: Path,
    title_tag: Optional[str] = None,
):
    """Side-by-side violin/box plots for multiple metrics."""
    _set_style()
    display_tag = title_tag or stim_tag
    n = len(metrics)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    groups = sorted(df["age_group"].unique())
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]
        data = [df.loc[df["age_group"] == g, col].dropna().values for g in groups]
        valid_pos = [i for i, d in enumerate(data) if len(d) >= 2]
        valid_data = [data[i] for i in valid_pos]

        if len(valid_data) >= 1:
            parts = ax.violinplot(valid_data, positions=valid_pos,
                                  showmedians=True, showextrema=False)
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(PALETTE[groups[valid_pos[j]]])
                pc.set_alpha(0.45)
            parts["cmedians"].set_color("black")

        for i, d in enumerate(data):
            if len(d) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
            ax.scatter(np.full_like(d, i) + jitter, d, s=8, alpha=0.35,
                       color=PALETTE[groups[i]], edgecolors="none")

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel(label)

        result = compare_groups(df, col)
        if result:
            p = result["p_value"]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_title(f"{label}\np={p:.2e} {stars}", fontsize=10)
        else:
            ax.set_title(label)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"RF Geometry Comparison  [{display_tag}]", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = f"comparison_{stim_tag.replace(' ', '_')}.png"
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def plot_rf_center_scatter(df: pd.DataFrame, stim_tag: str, fig_dir: Path,
                           title_tag: Optional[str] = None):
    """Scatter plot of RF center positions."""
    _set_style()
    display_tag = title_tag or stim_tag
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for ag in sorted(df["age_group"].unique()):
        sub = df[df["age_group"] == ag]
        ax.scatter(sub["center_col"], sub["center_row"], s=sub["area"] * 2,
                   alpha=0.45, label=ag, color=PALETTE[ag], edgecolors="white",
                   linewidth=0.3)
    ax.set_xlabel("Center column (px)")
    ax.set_ylabel("Center row (px)")
    ax.set_title(f"RF center positions  [{display_tag}]")
    ax.legend(frameon=False, markerscale=0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / f"rf_centers_{stim_tag.replace(' ', '_')}.png", bbox_inches="tight")
    plt.close(fig)


def plot_peak_frame_hist(df: pd.DataFrame, stim_tag: str, fig_dir: Path,
                         title_tag: Optional[str] = None):
    """Histogram of peak STA frame by age group."""
    _set_style()
    display_tag = title_tag or stim_tag
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for ag in sorted(df["age_group"].unique()):
        vals = df.loc[df["age_group"] == ag, "peak_frame"]
        ax.hist(vals, bins=20, alpha=0.55, label=ag, color=PALETTE[ag])
    ax.set_xlabel("Peak STA frame index")
    ax.set_ylabel("Count")
    ax.set_title(f"Peak frame distribution  [{display_tag}]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"peak_frame_{stim_tag.replace(' ', '_')}.png", bbox_inches="tight")
    plt.close(fig)


def plot_surround_index(df: pd.DataFrame, stim_tag: str, fig_dir: Path,
                        title_tag: Optional[str] = None):
    """Scatter of surround_strength vs gauss_r2, colored by age."""
    _set_style()
    display_tag = title_tag or stim_tag
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for ag in sorted(df["age_group"].unique()):
        sub = df[df["age_group"] == ag]
        ax.scatter(sub["gauss_r2"], sub["surround_strength"],
                   s=20, alpha=0.5, label=ag, color=PALETTE[ag])
    ax.set_xlabel("Gaussian fit $R^2$")
    ax.set_ylabel("Surround strength ($|A_{inh}|/|A_{exc}|$)")
    ax.set_title(f"Center-surround index  [{display_tag}]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"surround_index_{stim_tag.replace(' ', '_')}.png", bbox_inches="tight")
    plt.close(fig)


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_bar_comparison(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str]],
    stim_tag: str,
    fig_dir: Path,
    title_tag: Optional[str] = None,
):
    """
    Grouped bar chart showing mean +/- SEM for each metric,
    with significance stars annotated above each pair.
    """
    _set_style()
    display_tag = title_tag or stim_tag
    groups = sorted(df["age_group"].unique())
    n = len(metrics)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_2d(axes)

    bar_width = 0.35
    x_positions = np.arange(len(groups))

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]
        means = []
        sems = []
        for g in groups:
            vals = df.loc[df["age_group"] == g, col].dropna().values
            if len(vals) > 0:
                means.append(np.mean(vals))
                sems.append(stats.sem(vals))
            else:
                means.append(0)
                sems.append(0)

        bars = ax.bar(
            x_positions, means, bar_width * 2,
            yerr=sems, capsize=5,
            color=[PALETTE[g] for g in groups],
            edgecolor="white", linewidth=0.8,
            error_kw={"lw": 1.2, "capthick": 1.2},
        )

        result = compare_groups(df, col)
        if result:
            stars = _p_to_stars(result["p_value"])
        else:
            stars = ""

        if stars:
            y_top = max(m + e for m, e in zip(means, sems))
            bracket_y = y_top * 1.08
            star_y = y_top * 1.14
            ax.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
                    lw=1.0, color="0.3")
            ax.text(0.5, star_y, stars, ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="0.2")
            ax.set_ylim(top=y_top * 1.28)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"RF Geometry: Mean +/- SEM  [{display_tag}]", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = f"bar_comparison_{stim_tag.replace(' ', '_')}.png"
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table with stats per age group per stimulus."""
    metrics_of_interest = [
        "area", "equivalent_diameter",
        "gauss_sigma_x", "gauss_sigma_y", "gauss_sigma_mean", "gauss_r2",
        "dog_r2", "dog_sigma_exc", "dog_sigma_inh", "surround_strength",
        "on_off_ratio", "peak_frame",
        "lnl_bits_per_spike", "lnl_rectification_index", "lnl_nonlinearity_index",
    ]
    rows = []
    for stim in sorted(df["stimulus"].unique()):
        for ag in sorted(df["age_group"].unique()):
            sub = df[(df["stimulus"] == stim) & (df["age_group"] == ag)]
            row = {"stimulus": stim, "age_group": ag, "n_units": len(sub)}
            for m in metrics_of_interest:
                vals = sub[m].dropna()
                row[f"{m}_median"] = np.median(vals) if len(vals) else np.nan
                row[f"{m}_mean"] = np.mean(vals) if len(vals) else np.nan
                row[f"{m}_std"] = np.std(vals) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sample STA plots
# ---------------------------------------------------------------------------

STA_FIGURES_ROOT = Path(__file__).resolve().parent / "figures"
N_SAMPLE_PER_GROUP = 10


def copy_sample_sta_plots(
    df: pd.DataFrame,
    thr_dir: Path,
    *,
    n_per_group: int = N_SAMPLE_PER_GROUP,
    seed: int = 42,
):
    """Copy STA spatial plots for a random sample of units into *thr_dir*/sample_sta_plots/."""
    out = thr_dir / "sample_sta_plots"
    out.mkdir(parents=True, exist_ok=True)

    for old_file in out.glob("*.png"):
        old_file.unlink()

    copied = 0
    for group in sorted(df["age_group"].unique()):
        sub = df[df["age_group"] == group]
        n = min(n_per_group, len(sub))
        if n == 0:
            continue
        sample = sub.sample(n=n, random_state=seed)
        group_tag = "old" if "Old" in group else "young"

        for i, (_, row) in enumerate(sample.iterrows()):
            src = STA_FIGURES_ROOT / row.recording / f"sta_spatial_{row.unit_id}.png"
            if src.exists():
                dst = out / f"{group_tag}_{i+1:02d}_{row.recording}_{row.unit_id}.png"
                shutil.copy2(str(src), str(dst))
                copied += 1
            else:
                logger.warning("STA plot not found: %s", src)

    logger.info("  Copied %d sample STA plots to %s", copied, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RF geometry comparison: young vs old larval salamander")
    parser.add_argument("--r2-threshold", type=float, default=R2_THRESHOLD_DEFAULT,
                        help="Gaussian R2 threshold for quality filtering (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    r2_thr = args.r2_threshold
    output_dir = Path(args.output_dir)
    fig_dir = FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading geometry data from %s ...", output_dir)
    df_raw = load_all_units(output_dir)

    # ---- Filter to 15x15_5hz only --------------------------------------------
    STIM_FILTER = "15x15_5hz"
    df_raw = df_raw[df_raw["stimulus"] == STIM_FILTER].copy()
    logger.info("Stimulus filter (%s): %d units retained", STIM_FILTER, len(df_raw))

    # ---- R2 distribution before thresholding ---------------------------------
    logger.info("Plotting R2 distributions (before filtering) ...")
    plot_r2_distributions(df_raw, fig_dir)

    # ---- Metrics to compare --------------------------------------------------
    metrics = [
        ("area", "RF area (px)"),
        ("equivalent_diameter", "Equiv. diameter (px)"),
        ("gauss_sigma_mean", "Gaussian $\\sigma_{geo}$ (px)"),
        ("gauss_sigma_x", "Gaussian $\\sigma_x$ (px)"),
        ("gauss_sigma_y", "Gaussian $\\sigma_y$ (px)"),
        ("gauss_r2", "Gaussian $R^2$"),
        ("dog_r2", "DoG $R^2$"),
        ("dog_sigma_exc", "DoG $\\sigma_{exc}$ (px)"),
        ("dog_sigma_inh", "DoG $\\sigma_{inh}$ (px)"),
        ("surround_strength", "Surround strength"),
        ("on_off_ratio", "ON / (ON + OFF)"),
        ("peak_frame", "Peak STA frame"),
        ("lnl_bits_per_spike", "LNL bits/spike"),
        ("lnl_rectification_index", "Rectification index"),
        ("lnl_nonlinearity_index", "Nonlinearity index"),
    ]

    # ---- Loop over R2 thresholds ---------------------------------------------
    r2_thresholds = [0.5, 0.7, 0.8, 0.9]
    stim_tag = STIM_FILTER

    for r2_thr in r2_thresholds:
        thr_label = f"R2_{r2_thr:.1f}".replace(".", "p")
        thr_dir = fig_dir / thr_label
        thr_dir.mkdir(parents=True, exist_ok=True)

        n_before = len(df_raw)
        df = df_raw[df_raw["gauss_r2"] >= r2_thr].copy()
        n_after = len(df)
        logger.info("=== R2 >= %.2f : %d -> %d units (removed %d) ===",
                     r2_thr, n_before, n_after, n_before - n_after)

        title_tag = f"{stim_tag}  $R^2 \\geq {r2_thr}$"
        file_tag = stim_tag
        plot_violin_comparisons(df, metrics, file_tag, thr_dir, title_tag=title_tag)
        plot_bar_comparison(df, metrics, file_tag, thr_dir, title_tag=title_tag)
        plot_rf_center_scatter(df, file_tag, thr_dir, title_tag=title_tag)
        plot_peak_frame_hist(df, file_tag, thr_dir, title_tag=title_tag)
        plot_surround_index(df, file_tag, thr_dir, title_tag=title_tag)
        copy_sample_sta_plots(df, thr_dir)

        all_stats_rows = []
        for col, label in metrics:
            result = compare_groups(df, col)
            if result:
                result["stimulus"] = stim_tag
                result["r2_threshold"] = r2_thr
                all_stats_rows.append(result)

        df_stats = pd.DataFrame(all_stats_rows)
        if len(df_stats):
            df_stats = df_stats.sort_values("p_value")

        print("\n" + "=" * 90)
        print(f"Statistical Comparisons  (Gaussian R2 >= {r2_thr})")
        print("=" * 90)
        for _, r in df_stats.iterrows():
            stars = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "   "
            print(f"  {r.column:28s}  {r.group_a}: {r.median_a:8.3f} (n={r.n_a:3d})  "
                  f"{r.group_b}: {r.median_b:8.3f} (n={r.n_b:3d})  "
                  f"U={r.U_stat:8.0f}  p={r.p_value:.4e}  {stars}")

        stats_csv = thr_dir / "statistical_tests.csv"
        df_stats.to_csv(stats_csv, index=False)

        summary = build_summary_table(df)
        summary.to_csv(thr_dir / "summary_table.csv", index=False)

        df.to_csv(thr_dir / "filtered_units.csv", index=False)
        logger.info("  Saved to %s  (%d units)", thr_dir, len(df))

    print(f"\nAll figures saved under: {fig_dir}")


if __name__ == "__main__":
    main()
