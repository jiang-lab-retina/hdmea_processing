"""
Shared utilities for paper comparison figures.

Reproduces quantities analogous to those in Szatko et al. 2020
(Nat. Commun. 11:3481) from the user's before-blocker GB dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures"
TAB_DIR = THIS_DIR / "tables"

PROJECT_ROOT = THIS_DIR.parents[3]
DATA_PARQUET = (
    PROJECT_ROOT / "dataframe_compare" / "output_gb_control"
    / "combined_gb_control.parquet"
)

# Coordinate conventions (match gb_spatial_control/config.py)
X_COL = "improved_tx"
Y_COL = "improved_ty"
COORD_SCALE = 16
COORD_LIMIT_UM = 1600
XY_RANGE = (-COORD_LIMIT_UM, COORD_LIMIT_UM)

# D-V binning: paper uses 0.5 mm bins spanning the retina
DV_BIN_EDGES = np.arange(-1750.0, 1751.0, 500.0)
DV_BIN_CENTERS = 0.5 * (DV_BIN_EDGES[:-1] + DV_BIN_EDGES[1:])
DV_BIN_LABELS = [f"{int(c)}" for c in DV_BIN_CENTERS]

# Paper-reported values (Szatko et al. 2020, Figs. 2, 6 and text)
PAPER_SC = {
    # RGC (GCL) level, which best matches our RGC MEA recordings
    "rgc_ventral_center_mean": -0.35,
    "rgc_ventral_center_std": 0.27,
    "rgc_dorsal_center_mean": 0.06,
    "rgc_dorsal_center_std": 0.25,
    "rgc_ventral_surround_mean": 0.21,
    "rgc_ventral_surround_std": 0.82,
    "rgc_dorsal_surround_mean": 0.17,
    "rgc_dorsal_surround_std": 0.62,
    # GCL color-opponent fraction (Fig. 6)
    "rgc_frac_opp_ventral": 1312.0 / 4247.0,  # 0.309
    "rgc_frac_opp_dorsal": 191.0 / 1675.0,    # 0.114
    # Cone (Fig. 2) and BC (Fig. 4) levels for reference
    "cone_ventral_center_mean": -0.70,
    "cone_dorsal_center_mean": 0.38,
    "bc_ventral_center_mean": -0.44,
    "bc_dorsal_center_mean": 0.10,
}

# Color-opponent proxy threshold on |SC_on - SC_off|
OPP_THRESHOLD = 0.6

# Response filter: both ON peaks positive and at least one >= this value (Hz)
RESPONSE_FILTER_THRESHOLD_HZ = 50

# Plot styling
CMAP_SC = "RdBu_r"       # diverging, +green / -blue
CMAP_DENSITY = "magma"
CMAP_OPP = "coolwarm"

COLOR_VENTRAL = "#C71585"  # magenta-ish (UV analog)
COLOR_DORSAL = "#2CA02C"   # green
COLOR_OPP = "#FFB000"

GROUP_ORDER = ["Other", "OSGC", "DSGC", "ipRGC"]
GROUP_COLORS = {
    "Other": "#808080",
    "OSGC":  "#1f77b4",
    "DSGC":  "#d62728",
    "ipRGC": "#9467bd",
}


MIN_SUBTYPE_N = 20


def parent_group(subtype: str) -> str:
    """Extract the parent group name from a subtype label like 'ipRGC_4'."""
    for g in GROUP_ORDER:
        if subtype.startswith(g):
            return g
    return "Unknown"


def get_subtype_order(df: pd.DataFrame) -> list[str]:
    """Return subtypes sorted by parent group order then numeric suffix."""
    subs = sorted(
        s for s in df["subtype"].unique()
        if isinstance(s, str) and s != ""
    )
    def _sort_key(s: str) -> tuple[int, int]:
        g = parent_group(s)
        gi = GROUP_ORDER.index(g) if g in GROUP_ORDER else 99
        parts = s.rsplit("_", 1)
        num = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 99
        return (gi, num)
    return sorted(subs, key=_sort_key)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def load_combined(drop_empty_group: bool = False,
                  response_filter: bool = False) -> pd.DataFrame:
    """Load combined GB control parquet and add derived columns.

    Parameters
    ----------
    drop_empty_group : bool
        If True, drop rows whose ``group`` column is an empty string.
    response_filter : bool
        If True, keep only cells where both green and blue ON peak extremes are
        positive (> 0) AND at least one is >= RESPONSE_FILTER_THRESHOLD_HZ.

    Adds
    ----
    X_um, Y_um : float
        Spatial coordinates in micrometers, clipped to +/- COORD_LIMIT_UM.
    SC_on, SC_off : float
        Paper-style spectral contrast computed from raw peak amplitudes.
    SC_diff : float
        SC_on - SC_off.
    is_opponent : bool
        |SC_on - SC_off| > OPP_THRESHOLD.
    dv_bin : int
        Index into DV_BIN_CENTERS (NaN if outside binning range).
    retina_half : str
        "ventral" if Y_um < 0, "dorsal" if Y_um > 0, else "equator".
    """
    df = pd.read_parquet(DATA_PARQUET).copy()

    df["X_um"] = df[X_COL] * COORD_SCALE
    df["Y_um"] = df[Y_COL] * COORD_SCALE
    mask = (df["X_um"].abs() < COORD_LIMIT_UM) & (df["Y_um"].abs() < COORD_LIMIT_UM)
    df = df.loc[mask].reset_index(drop=True)

    df["SC_on"] = _spectral_contrast(
        df["green_on_peak_extreme"], df["blue_on_peak_extreme"]
    )
    df["SC_off"] = _spectral_contrast(
        df["green_off_peak_extreme"], df["blue_off_peak_extreme"]
    )
    df["SC_diff"] = df["SC_on"] - df["SC_off"]
    df["is_opponent"] = df["SC_diff"].abs() > OPP_THRESHOLD

    y = df["Y_um"].to_numpy()
    dv_bin = np.digitize(y, DV_BIN_EDGES) - 1
    dv_bin = np.where(
        (dv_bin >= 0) & (dv_bin < len(DV_BIN_CENTERS)),
        dv_bin,
        -1,
    )
    df["dv_bin"] = dv_bin

    df["retina_half"] = np.where(
        df["Y_um"] > 0, "dorsal",
        np.where(df["Y_um"] < 0, "ventral", "equator"),
    )

    if response_filter:
        pos_mask = (df["green_on_peak_extreme"] > 0) & (df["blue_on_peak_extreme"] > 0)
        thr_mask = (
            (df["green_on_peak_extreme"] >= RESPONSE_FILTER_THRESHOLD_HZ)
            | (df["blue_on_peak_extreme"] >= RESPONSE_FILTER_THRESHOLD_HZ)
        )
        df = df[pos_mask & thr_mask].reset_index(drop=True)

    if drop_empty_group and "group" in df.columns:
        df = df[df["group"].astype(str) != ""].reset_index(drop=True)

    return df


def spectral_contrast(green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Public alias for the paper-style spectral contrast."""
    return _spectral_contrast(green, blue)


def _spectral_contrast(green, blue, eps: float = 1e-6):
    """Paper-style spectral contrast.

    SC = (G - B) / (|G| + |B| + eps)

    The absolute values in the denominator ensure that SC continues to report
    spectral preference (not response polarity) when one or both peaks are
    negative, as in ~10 percent of the user's data where the cell's extreme
    within the ON or OFF window is a decrement.
    """
    g = np.asarray(green, dtype=float)
    b = np.asarray(blue, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = (g - b) / (np.abs(g) + np.abs(b) + eps)
    sc = np.clip(sc, -1.0, 1.0)
    return sc


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 percent confidence interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def dv_bin_stats(
    df: pd.DataFrame, col: str, bin_col: str = "dv_bin"
) -> pd.DataFrame:
    """Return mean, SEM, std, count per D-V bin for a numeric column."""
    rows = []
    for bi, center_um in enumerate(DV_BIN_CENTERS):
        sel = df[(df[bin_col] == bi)]
        v = sel[col].dropna().to_numpy()
        if v.size == 0:
            rows.append({
                "bin": bi, "center_um": center_um, "n": 0,
                "mean": np.nan, "sem": np.nan, "std": np.nan,
            })
            continue
        rows.append({
            "bin": bi,
            "center_um": center_um,
            "n": v.size,
            "mean": float(np.mean(v)),
            "sem": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
            "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def dv_bin_fraction(
    df: pd.DataFrame, bool_col: str, bin_col: str = "dv_bin"
) -> pd.DataFrame:
    """Return fraction (plus Wilson CI and count) of True per D-V bin."""
    rows = []
    for bi, center_um in enumerate(DV_BIN_CENTERS):
        sel = df[(df[bin_col] == bi)]
        n = int(len(sel))
        k = int(sel[bool_col].sum()) if n > 0 else 0
        lo, hi = wilson_ci(k, n) if n > 0 else (np.nan, np.nan)
        rows.append({
            "bin": bi,
            "center_um": center_um,
            "n": n,
            "k_true": k,
            "fraction": (k / n) if n > 0 else np.nan,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def style_dv_axes(ax, x_label: str = "Y (um)  V <-- --> D",
                  xlim: tuple[float, float] | None = None) -> None:
    """Apply consistent D-V styling to a line/bar chart along Y_um."""
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)


def style_xy_axes(ax, title: str | None = None) -> None:
    """Apply D/V/N/T axis labels consistent with the project's convention."""
    ax.set_aspect("equal")
    ax.set_xlim(XY_RANGE)
    ax.set_ylim(XY_RANGE)
    ax.set_xlabel("T <-- X (um) --> N")
    ax.set_ylabel("V <-- Y (um) --> D")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.axvline(0, color="black", lw=0.5, alpha=0.4)
    if title is not None:
        ax.set_title(title, fontsize=10)


def savefig(fig: plt.Figure, name: str, dpi: int = 200) -> Path:
    """Save figure to FIG_DIR/name and close it; return the path."""
    ensure_dirs()
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def savetable(df: pd.DataFrame, name: str) -> Path:
    ensure_dirs()
    path = TAB_DIR / name
    df.to_csv(path, index=False)
    return path
