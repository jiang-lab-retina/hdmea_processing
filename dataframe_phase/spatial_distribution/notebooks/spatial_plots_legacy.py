# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Spatial Distribution Plots (Legacy Coordinates)
# Same plots as spatial_plots.py but using legacy_transformed_x / y.
# Input: labeled_dataframe_with_legacy_coords.parquet

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import math
import warnings
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout

from pygam import LinearGAM, LogisticGAM, PoissonGAM, te

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %%
# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPATIAL_DIR = os.path.dirname(SCRIPT_DIR)  # spatial_distribution/
PROJECT_ROOT = os.path.abspath(os.path.join(SPATIAL_DIR, "..", ".."))

INPUT_PARQUET = os.path.join(
    SPATIAL_DIR, "results",
    "labeled_dataframe_with_legacy_coords.parquet",
)

FIG_DIR = os.path.join(SPATIAL_DIR, "figures_legacy")
FIG_ALL_DIR = os.path.join(FIG_DIR, "all_cells")
FIG_SUB_DIR = os.path.join(FIG_DIR, "per_subtype")

for d in [FIG_DIR, FIG_ALL_DIR, FIG_SUB_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- parameters ----------
X_COL = "legacy_transformed_x"
Y_COL = "legacy_transformed_y"
COORD_SCALE = 16  # electrode-units -> microns
COORD_LIMIT = 100  # filter range in electrode units
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)  # microns

GRIDSIZE_ALL = 40
GRIDSIZE_SUB = 15
MINCNT_ALL = 2
MINCNT_SUB = 1
CMAP = "coolwarm"
N_SPLINES_ALL = 30
N_SPLINES_SUB = 15

# %%
# ---------- load & filter data ----------
print("Loading data ...")
df_raw = pd.read_parquet(INPUT_PARQUET)
print(f"  Raw shape: {df_raw.shape}")

# Drop rows with NaN legacy coordinates
df_raw = df_raw.dropna(subset=[X_COL, Y_COL])
print(f"  After dropping NaN legacy coords: {df_raw.shape}")

# Filter to valid spatial range
mask = (
    (df_raw[X_COL] > -COORD_LIMIT) & (df_raw[X_COL] < COORD_LIMIT) &
    (df_raw[Y_COL] > -COORD_LIMIT) & (df_raw[Y_COL] < COORD_LIMIT)
)
df = df_raw[mask].copy()
print(f"  After spatial filter: {df.shape}")

# %%
# ---------- feature column selection ----------
EXCLUDE_PATTERNS = [
    "transformed_", "legacy_transformed_", "cartesian_", "polar_",
    "freq_step_", "ap_slope", "ap_intercept", "ap_r_value",
    "soma_row", "soma_col", "axon_centroids", "center_xy",
]

float_cols = sorted([
    c for c in df.columns
    if df[c].dtype == "float64"
    and not any(pat in c for pat in EXCLUDE_PATTERNS)
])
print(f"  Feature columns ({len(float_cols)}): {float_cols[:10]} ...")

# %%
# ---------- valid subtypes ----------
valid_subtypes = sorted(
    df.loc[df["valid_mosaic"] == True, "subtype"].unique().tolist()
)
print(f"  Valid subtypes ({len(valid_subtypes)}): {valid_subtypes}")

# ==========================================================================
# Plot helpers
# ==========================================================================

# %%
def plot_dot_plot(df, save_path):
    """Scatter plot of all cell positions (black dots, square aspect)."""
    x = df[X_COL].to_numpy() * COORD_SCALE
    y = df[Y_COL].to_numpy() * COORD_SCALE

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=1, c="black", alpha=0.5, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XY_RANGE)
    ax.set_ylim(XY_RANGE)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dot plot -> {save_path}")


# %%
def plot_hexbin_single(ax, x, y, c, gridsize, mincnt, cmap, vmin, vmax):
    """Draw a hexbin on the given axes. Returns the hexbin collection."""
    hb = ax.hexbin(
        x, y,
        C=c,
        reduce_C_function=np.mean,
        gridsize=gridsize,
        extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
        mincnt=mincnt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XY_RANGE)
    ax.set_ylim(XY_RANGE)
    return hb


# %%
def _choose_gam_family(y):
    """Pick GAM family from the response vector."""
    uniq = np.unique(y[np.isfinite(y)])
    if set(uniq).issubset({0, 1}):
        return LogisticGAM
    if np.all(y >= 0) and np.allclose(y, np.round(y)) and y.max() > 5:
        return PoissonGAM
    return LinearGAM


def fit_gam_predict_hexbin(ax, x, y, c, gridsize, mincnt, cmap, vmin, vmax,
                           n_splines=30):
    """
    Fit GAM on (x, y) -> c, draw hexbin coloured by GAM predictions at bin
    centres.  Returns the hexbin collection (or None on failure).
    """
    GamClass = _choose_gam_family(c)
    X_train = np.column_stack([x, y])
    gam = GamClass(te(0, 1, n_splines=[n_splines, n_splines]))

    # Fit (suppress verbose output from pyGAM gridsearch)
    try:
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            gam = gam.gridsearch(X_train, c)
    except Exception:
        try:
            gam.fit(X_train, c)
        except Exception:
            return None

    # Create a count-only hexbin to get bin centres that contain data
    hb = ax.hexbin(
        x, y,
        gridsize=gridsize,
        extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
        mincnt=mincnt,
        cmap=cmap,
    )
    offsets = hb.get_offsets()
    if len(offsets) == 0:
        return None

    z_pred = gam.predict(offsets)
    hb.set_array(z_pred)
    hb.set_clim(vmin=vmin, vmax=vmax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XY_RANGE)
    ax.set_ylim(XY_RANGE)
    return hb


# %%
def plot_hexbin_all_cells(df, feature, save_dir):
    """Two-panel figure: Raw hexbin (left) + GAM hexbin (right)."""
    cols = [X_COL, Y_COL, feature]
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        print(f"  [skip] {feature}: only {len(data)} valid rows")
        return

    x = data[X_COL].to_numpy() * COORD_SCALE
    y = data[Y_COL].to_numpy() * COORD_SCALE
    c = data[feature].to_numpy()

    c_mean = float(np.mean(c))
    if c_mean == 0:
        print(f"  [skip] {feature}: zero mean")
        return
    vmin = c_mean - 0.5 * abs(c_mean)
    vmax = c_mean + 0.5 * abs(c_mean)

    fig, (ax_raw, ax_gam) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(right=0.90, wspace=0.15)

    # Left: raw hexbin
    hb = plot_hexbin_single(ax_raw, x, y, c, GRIDSIZE_ALL, MINCNT_ALL,
                            CMAP, vmin, vmax)
    ax_raw.set_title("Raw mean", fontsize=11)

    # Right: GAM hexbin
    hb_gam = fit_gam_predict_hexbin(ax_gam, x, y, c, GRIDSIZE_ALL,
                                    MINCNT_ALL, CMAP, vmin, vmax,
                                    n_splines=N_SPLINES_ALL)
    ax_gam.set_title("GAM smoothed", fontsize=11)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_raw, ax_gam], shrink=0.75, pad=0.02)
    cbar.set_label(feature, fontsize=11)

    fig.suptitle(feature, fontsize=13)
    out = os.path.join(save_dir, f"Hexbin_{feature}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
def plot_hexbin_per_subtype(df, feature, valid_subtypes, save_dir):
    """
    One figure per feature.  Each valid subtype gets a paired column:
    left = raw hexbin, right = GAM hexbin.  Shared colour scale.
    """
    sub_df = df[df["valid_mosaic"] == True]
    cols = [X_COL, Y_COL, feature, "subtype"]
    data = sub_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        print(f"  [skip subtype] {feature}: only {len(data)} valid rows")
        return

    c_all = data[feature].to_numpy()
    c_mean = float(np.mean(c_all))
    if c_mean == 0:
        print(f"  [skip subtype] {feature}: zero mean")
        return
    vmin = c_mean - 0.5 * abs(c_mean)
    vmax = c_mean + 0.5 * abs(c_mean)

    n = len(valid_subtypes)
    pairs_per_row = 3          # 3 subtypes per row (6 columns: raw|gam x3)
    grid_cols = pairs_per_row * 2
    nrows = math.ceil(n / pairs_per_row)

    fig, axes = plt.subplots(
        nrows, grid_cols,
        figsize=(4 * grid_cols + 1.5, 3.8 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(right=0.90, wspace=0.20, hspace=0.40)

    for idx, stype in enumerate(valid_subtypes):
        row = idx // pairs_per_row
        pair_col = idx % pairs_per_row
        ax_raw = axes[row][pair_col * 2]
        ax_gam = axes[row][pair_col * 2 + 1]

        sdata = data[data["subtype"] == stype]
        if len(sdata) < 3:
            for ax in (ax_raw, ax_gam):
                ax.set_title(stype, fontsize=8)
                ax.text(0.5, 0.5, "n < 3", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
                ax.set_aspect("equal", adjustable="box")
            continue

        sx = sdata[X_COL].to_numpy() * COORD_SCALE
        sy = sdata[Y_COL].to_numpy() * COORD_SCALE
        sc = sdata[feature].to_numpy()

        plot_hexbin_single(ax_raw, sx, sy, sc, GRIDSIZE_SUB, MINCNT_SUB,
                           CMAP, vmin, vmax)
        ax_raw.set_title(f"{stype} raw (n={len(sdata)})", fontsize=8)

        fit_gam_predict_hexbin(ax_gam, sx, sy, sc, GRIDSIZE_SUB, MINCNT_SUB,
                               CMAP, vmin, vmax, n_splines=N_SPLINES_SUB)
        ax_gam.set_title(f"{stype} GAM", fontsize=8)

    # Hide unused subplots
    total_slots = nrows * grid_cols
    used_slots = n * 2
    for idx in range(used_slots, total_slots):
        row, col = divmod(idx, grid_cols)
        axes[row][col].set_visible(False)
    for idx in range(n, nrows * pairs_per_row):
        row = idx // pairs_per_row
        pair_col = idx % pairs_per_row
        axes[row][pair_col * 2].set_visible(False)
        axes[row][pair_col * 2 + 1].set_visible(False)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(feature, fontsize=11)

    fig.suptitle(feature, fontsize=13, y=1.01)
    out = os.path.join(save_dir, f"Hexbin_{feature}_subtypes.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Main execution
# ==========================================================================

# %%
# 1. Dot plot
print("\n=== Dot plot (legacy coords) ===")
plot_dot_plot(df, os.path.join(FIG_DIR, "dot_plot_all_cells.png"))

# %%
# 2. All-cells hexbin heatmaps
print("\n=== All-cells hexbin heatmaps (legacy coords) ===")
for i, feat in enumerate(float_cols):
    print(f"  [{i+1}/{len(float_cols)}] {feat}")
    plot_hexbin_all_cells(df, feat, FIG_ALL_DIR)

# %%
# 3. Per-subtype hexbin heatmaps
print("\n=== Per-subtype hexbin heatmaps (legacy coords) ===")
for i, feat in enumerate(float_cols):
    print(f"  [{i+1}/{len(float_cols)}] {feat}")
    plot_hexbin_per_subtype(df, feat, valid_subtypes, FIG_SUB_DIR)

print("\nDone. All figures saved under:", FIG_DIR)
