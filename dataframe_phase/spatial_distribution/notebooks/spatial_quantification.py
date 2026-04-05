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
# # Spatial Distribution Quantification (Legacy Coordinates)
#
# Uses `labeled_dataframe_with_legacy_coords_freq.parquet`.
#
# **Outputs:**
#   - hexbin_data_all_cells.parquet   (bin centres, raw means, GAM predictions)
#   - hexbin_data_per_subtype.parquet  (bin centres, raw means)
#   - spatial_metrics.parquet          (unevenness, gradient, radial, Moran's I, ...)
#   - spatial_analysis_summary.md      (human-readable summary)
#   - figures_legacy/ plots            (raw + GAM hexbin for all-cells and per-subtype)

# %%
import sys, os, math, json, warnings
from pathlib import Path
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from pygam import LinearGAM, LogisticGAM, PoissonGAM, te

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Imports done", flush=True)

# %%
# ==========================================================================
# Paths & parameters
# ==========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SPATIAL_DIR.parent.parent

INPUT_PARQUET = SPATIAL_DIR / "results" / "labeled_dataframe_with_legacy_coords_freq.parquet"
RESULTS_DIR = SPATIAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = SPATIAL_DIR / "figures_legacy"
FIG_ALL_DIR = FIG_DIR / "all_cells"
FIG_SUB_DIR = FIG_DIR / "per_subtype"
for d in [FIG_DIR, FIG_ALL_DIR, FIG_SUB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

X_COL = "legacy_transformed_x"
Y_COL = "legacy_transformed_y"
COORD_SCALE = 16
COORD_LIMIT = 100
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)

GRIDSIZE_ALL = 40
GRIDSIZE_SUB = 15
MINCNT_ALL = 2
MINCNT_SUB = 1
CMAP = "coolwarm"
N_SPLINES_ALL = 30
N_SPLINES_SUB = 15

EXCLUDE_PATTERNS = [
    "transformed_", "legacy_transformed_", "cartesian_", "polar_",
    "freq_step_", "ap_slope", "ap_intercept", "ap_r_value",
    "soma_row", "soma_col", "axon_centroids", "center_xy", "freq_section_",
]

# %%
# ==========================================================================
# Load & filter data
# ==========================================================================
print("Loading data ...", flush=True)
df_raw = pd.read_parquet(INPUT_PARQUET)
print(f"  Raw shape: {df_raw.shape}", flush=True)
df_raw = df_raw.dropna(subset=[X_COL, Y_COL])
mask = (
    (df_raw[X_COL] > -COORD_LIMIT) & (df_raw[X_COL] < COORD_LIMIT) &
    (df_raw[Y_COL] > -COORD_LIMIT) & (df_raw[Y_COL] < COORD_LIMIT)
)
df = df_raw[mask].copy()
print(f"  After spatial filter: {df.shape}", flush=True)

float_cols = sorted([
    c for c in df.columns
    if df[c].dtype == "float64"
    and not any(pat in c for pat in EXCLUDE_PATTERNS)
])
print(f"  Feature columns: {len(float_cols)}", flush=True)

valid_subtypes = sorted(
    df.loc[df["valid_mosaic"] == True, "subtype"].unique().tolist()
)
print(f"  Valid subtypes: {len(valid_subtypes)}", flush=True)


# ==========================================================================
# Helper functions
# ==========================================================================

# %%
def extract_hexbin_data(x, y, c, gridsize, mincnt):
    """Compute hexbin bins. Returns (bin_centers, raw_means, counts)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.mean,
                   gridsize=gridsize,
                   extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
                   mincnt=mincnt, cmap=CMAP)
    centers = hb.get_offsets().copy()
    means = hb.get_array().copy()

    # counts: same bins, count aggregation
    ax.cla()
    hb_cnt = ax.hexbin(x, y, gridsize=gridsize,
                       extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
                       mincnt=mincnt, cmap=CMAP)
    counts = hb_cnt.get_array().copy()
    plt.close(fig)

    # lengths should match; if not, truncate to min
    n = min(len(centers), len(means), len(counts))
    return centers[:n], means[:n], counts[:n]


# %%
def _choose_gam_family(y):
    uniq = np.unique(y[np.isfinite(y)])
    if set(uniq).issubset({0, 1}):
        return LogisticGAM
    if np.all(y >= 0) and np.allclose(y, np.round(y)) and y.max() > 5:
        return PoissonGAM
    return LinearGAM


def fit_gam_at_centers(x, y, c, bin_centers, n_splines=30):
    """Fit GAM and predict at bin centres. Returns predictions array or None."""
    if len(c) < 30 or len(bin_centers) == 0:
        return None
    GamClass = _choose_gam_family(c)
    X_train = np.column_stack([x, y])
    gam = GamClass(te(0, 1, n_splines=[n_splines, n_splines]))
    try:
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            gam = gam.gridsearch(X_train, c)
    except Exception:
        try:
            gam.fit(X_train, c)
        except Exception:
            return None
    try:
        return gam.predict(bin_centers)
    except Exception:
        return None


# %%
def compute_moran_i(bin_centers, bin_values, k=6):
    """Spatial autocorrelation (Moran's I) with k-nearest-neighbor weights."""
    n = len(bin_values)
    if n < k + 1:
        return np.nan
    z = bin_values - np.mean(bin_values)
    denom = np.sum(z ** 2)
    if denom == 0:
        return np.nan

    tree = cKDTree(bin_centers)
    _, idx = tree.query(bin_centers, k=min(k + 1, n))
    # idx[:, 0] is self
    numer = 0.0
    W = 0.0
    for i in range(n):
        for j_pos in range(1, idx.shape[1]):
            j = idx[i, j_pos]
            numer += z[i] * z[j]
            W += 1.0
    if W == 0:
        return np.nan
    return float((n / W) * (numer / denom))


# %%
def compute_metrics(x_um, y_um, c, bin_centers, bin_means, bin_counts):
    """Full suite of spatial distribution metrics."""
    m = {}
    m["n_valid"] = len(c)
    m["n_bins"] = len(bin_means)
    m["overall_mean"] = float(np.mean(c))
    m["overall_std"] = float(np.std(c))

    # --- hexbin unevenness ---
    if len(bin_means) > 2:
        bm = bin_means
        bm_mean = np.mean(bm)
        m["hexbin_cv"] = float(np.std(bm) / abs(bm_mean)) if bm_mean != 0 else np.nan
        m["hexbin_iqr"] = float(np.percentile(bm, 75) - np.percentile(bm, 25))
        m["hexbin_range"] = float(np.ptp(bm))
        # Gini coefficient
        sorted_bm = np.sort(bm)
        n_b = len(sorted_bm)
        cumulative = np.cumsum(sorted_bm)
        gini = (2 * np.sum((np.arange(1, n_b + 1) * sorted_bm)) /
                (n_b * np.sum(sorted_bm)) - (n_b + 1) / n_b)
        m["hexbin_gini"] = float(gini) if np.isfinite(gini) else np.nan
    else:
        for k in ["hexbin_cv", "hexbin_iqr", "hexbin_range", "hexbin_gini"]:
            m[k] = np.nan

    # --- linear gradient ---
    if len(c) >= 10:
        A = np.column_stack([x_um, y_um, np.ones(len(x_um))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, c, rcond=None)
            gx, gy = coeffs[0], coeffs[1]
            pred = A @ coeffs
            ss_res = np.sum((c - pred) ** 2)
            ss_tot = np.sum((c - np.mean(c)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            m["gradient_x"] = float(gx)
            m["gradient_y"] = float(gy)
            m["gradient_mag"] = float(np.sqrt(gx ** 2 + gy ** 2))
            m["gradient_dir_deg"] = float(np.degrees(np.arctan2(gy, gx)))
            m["gradient_r2"] = float(r2)
        except Exception:
            for k in ["gradient_x", "gradient_y", "gradient_mag",
                       "gradient_dir_deg", "gradient_r2"]:
                m[k] = np.nan
    else:
        for k in ["gradient_x", "gradient_y", "gradient_mag",
                   "gradient_dir_deg", "gradient_r2"]:
            m[k] = np.nan

    # --- radial analysis ---
    radius = np.sqrt(x_um ** 2 + y_um ** 2)
    if len(c) >= 10 and np.std(radius) > 0 and np.std(c) > 0:
        try:
            r_val, p_val = pearsonr(radius, c)
            m["radial_r"] = float(r_val)
            m["radial_p"] = float(p_val)
        except Exception:
            m["radial_r"] = np.nan
            m["radial_p"] = np.nan
        try:
            m["radial_slope"] = float(np.polyfit(radius, c, 1)[0])
        except Exception:
            m["radial_slope"] = np.nan
    else:
        m["radial_r"] = np.nan
        m["radial_p"] = np.nan
        m["radial_slope"] = np.nan

    # --- centre vs periphery ---
    if len(c) >= 20:
        med_r = np.median(radius)
        ctr = c[radius < med_r]
        per = c[radius >= med_r]
        m["center_mean"] = float(np.mean(ctr))
        m["periphery_mean"] = float(np.mean(per))
        m["center_periphery_ratio"] = (
            float(np.mean(ctr) / np.mean(per))
            if np.mean(per) != 0 else np.nan
        )
    else:
        m["center_mean"] = np.nan
        m["periphery_mean"] = np.nan
        m["center_periphery_ratio"] = np.nan

    # --- quadrant analysis ---
    quads = {
        "NE": (x_um >= 0) & (y_um >= 0),
        "NW": (x_um < 0) & (y_um >= 0),
        "SE": (x_um >= 0) & (y_um < 0),
        "SW": (x_um < 0) & (y_um < 0),
    }
    q_means = {}
    for q, mask_q in quads.items():
        if mask_q.sum() > 0:
            q_means[q] = float(np.mean(c[mask_q]))
    for q in ["NE", "NW", "SE", "SW"]:
        m[f"quadrant_{q}_mean"] = q_means.get(q, np.nan)
    m["dominant_quadrant"] = max(q_means, key=q_means.get) if q_means else ""

    # --- Moran's I ---
    if len(bin_means) >= 7:
        m["moran_i"] = compute_moran_i(bin_centers, bin_means, k=6)
    else:
        m["moran_i"] = np.nan

    return m


# %%
# ==========================================================================
# Plot helpers (inline, saves figure)
# ==========================================================================

def _plot_hexbin_single(ax, x, y, c, gridsize, mincnt, vmin, vmax):
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.mean,
                   gridsize=gridsize,
                   extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
                   mincnt=mincnt, cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
    return hb


def _plot_gam_hexbin(ax, x, y, c, gridsize, mincnt, vmin, vmax, n_splines):
    GamClass = _choose_gam_family(c)
    X_train = np.column_stack([x, y])
    gam = GamClass(te(0, 1, n_splines=[n_splines, n_splines]))
    try:
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            gam = gam.gridsearch(X_train, c)
    except Exception:
        try:
            gam.fit(X_train, c)
        except Exception:
            return None
    hb = ax.hexbin(x, y, gridsize=gridsize,
                   extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
                   mincnt=mincnt, cmap=CMAP)
    offsets = hb.get_offsets()
    if len(offsets) == 0:
        return None
    z_pred = gam.predict(offsets)
    hb.set_array(z_pred)
    hb.set_clim(vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
    return hb


def save_allcells_plot(x, y, c, feature, save_dir, vmin, vmax):
    fig, (ax_raw, ax_gam) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(right=0.90, wspace=0.15)
    _plot_hexbin_single(ax_raw, x, y, c, GRIDSIZE_ALL, MINCNT_ALL, vmin, vmax)
    ax_raw.set_title("Raw mean", fontsize=11)
    _plot_gam_hexbin(ax_gam, x, y, c, GRIDSIZE_ALL, MINCNT_ALL, vmin, vmax,
                     n_splines=N_SPLINES_ALL)
    ax_gam.set_title("GAM smoothed", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_raw, ax_gam], shrink=0.75, pad=0.02).set_label(feature, fontsize=11)
    fig.suptitle(feature, fontsize=13)
    fig.savefig(str(save_dir / f"Hexbin_{feature}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_persubtype_plot(df_sub, feature, valid_subtypes, save_dir, vmin, vmax):
    n = len(valid_subtypes)
    pairs_per_row = 3
    grid_cols = pairs_per_row * 2
    nrows = math.ceil(n / pairs_per_row)
    fig, axes = plt.subplots(nrows, grid_cols,
                             figsize=(4 * grid_cols + 1.5, 3.8 * nrows), squeeze=False)
    fig.subplots_adjust(right=0.90, wspace=0.20, hspace=0.40)

    for idx, stype in enumerate(valid_subtypes):
        row = idx // pairs_per_row
        pair_col = idx % pairs_per_row
        ax_raw = axes[row][pair_col * 2]
        ax_gam = axes[row][pair_col * 2 + 1]
        sdata = df_sub[df_sub["subtype"] == stype]
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
        _plot_hexbin_single(ax_raw, sx, sy, sc, GRIDSIZE_SUB, MINCNT_SUB, vmin, vmax)
        ax_raw.set_title(f"{stype} raw (n={len(sdata)})", fontsize=8)
        _plot_gam_hexbin(ax_gam, sx, sy, sc, GRIDSIZE_SUB, MINCNT_SUB,
                         vmin, vmax, n_splines=N_SPLINES_SUB)
        ax_gam.set_title(f"{stype} GAM", fontsize=8)

    for idx_hide in range(n, nrows * pairs_per_row):
        r = idx_hide // pairs_per_row
        pc = idx_hide % pairs_per_row
        axes[r][pc * 2].set_visible(False)
        axes[r][pc * 2 + 1].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02).set_label(feature, fontsize=11)
    fig.suptitle(feature, fontsize=13, y=1.01)
    fig.savefig(str(save_dir / f"Hexbin_{feature}_subtypes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Phase 1: All-cells analysis  (hexbin + GAM + metrics + plot)
# ==========================================================================

# %%
print("\n=== Phase 1: All-cells analysis ===", flush=True)

hexbin_rows_all = []
metrics_rows_all = []

n_feat = len(float_cols)
for fi, feature in enumerate(float_cols):
    cols_needed = [X_COL, Y_COL, feature]
    data = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        print(f"  [{fi+1}/{n_feat}] {feature} -- SKIP (n={len(data)})", flush=True)
        continue

    x = data[X_COL].to_numpy() * COORD_SCALE
    y = data[Y_COL].to_numpy() * COORD_SCALE
    c = data[feature].to_numpy()

    c_mean = float(np.mean(c))
    if c_mean == 0:
        vmin_plt, vmax_plt = -1, 1
    else:
        vmin_plt = c_mean - 0.5 * abs(c_mean)
        vmax_plt = c_mean + 0.5 * abs(c_mean)

    # 1) extract hexbin data
    centers, raw_means, counts = extract_hexbin_data(
        x, y, c, GRIDSIZE_ALL, MINCNT_ALL)

    # 2) GAM predictions at bin centres
    gam_preds = fit_gam_at_centers(x, y, c, centers, n_splines=N_SPLINES_ALL)

    # 3) store hexbin rows
    for bi in range(len(centers)):
        row = {
            "scope": "all_cells",
            "feature": feature,
            "bin_x": centers[bi, 0],
            "bin_y": centers[bi, 1],
            "count": int(counts[bi]),
            "raw_mean": float(raw_means[bi]),
            "gam_pred": float(gam_preds[bi]) if gam_preds is not None else np.nan,
        }
        hexbin_rows_all.append(row)

    # 4) metrics
    m = compute_metrics(x, y, c, centers, raw_means, counts)
    m["scope"] = "all_cells"
    m["feature"] = feature
    metrics_rows_all.append(m)

    # 5) plot
    save_allcells_plot(x, y, c, feature, FIG_ALL_DIR, vmin_plt, vmax_plt)

    if (fi + 1) % 10 == 0 or (fi + 1) == n_feat:
        print(f"  [{fi+1}/{n_feat}] {feature}", flush=True)

print(f"  All-cells: {len(hexbin_rows_all)} hexbin rows, "
      f"{len(metrics_rows_all)} metric rows", flush=True)


# ==========================================================================
# Phase 2: Per-subtype analysis  (hexbin + metrics + plot)
# ==========================================================================

# %%
print("\n=== Phase 2: Per-subtype analysis ===", flush=True)

hexbin_rows_sub = []
metrics_rows_sub = []
sub_df = df[df["valid_mosaic"] == True].copy()

for fi, feature in enumerate(float_cols):
    cols_needed = [X_COL, Y_COL, feature, "subtype"]
    data = sub_df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        continue

    c_all = data[feature].to_numpy()
    c_mean = float(np.mean(c_all))
    if c_mean == 0:
        vmin_plt, vmax_plt = -1, 1
    else:
        vmin_plt = c_mean - 0.5 * abs(c_mean)
        vmax_plt = c_mean + 0.5 * abs(c_mean)

    for stype in valid_subtypes:
        sdata = data[data["subtype"] == stype]
        if len(sdata) < 3:
            continue
        sx = sdata[X_COL].to_numpy() * COORD_SCALE
        sy = sdata[Y_COL].to_numpy() * COORD_SCALE
        sc = sdata[feature].to_numpy()

        centers, raw_means, counts = extract_hexbin_data(
            sx, sy, sc, GRIDSIZE_SUB, MINCNT_SUB)

        for bi in range(len(centers)):
            hexbin_rows_sub.append({
                "scope": stype,
                "feature": feature,
                "bin_x": centers[bi, 0],
                "bin_y": centers[bi, 1],
                "count": int(counts[bi]),
                "raw_mean": float(raw_means[bi]),
                "gam_pred": np.nan,  # GAM omitted for per-subtype speed
            })

        m = compute_metrics(sx, sy, sc, centers, raw_means, counts)
        m["scope"] = stype
        m["feature"] = feature
        metrics_rows_sub.append(m)

    # save per-subtype plot
    save_persubtype_plot(data, feature, valid_subtypes, FIG_SUB_DIR, vmin_plt, vmax_plt)

    if (fi + 1) % 10 == 0 or (fi + 1) == n_feat:
        print(f"  [{fi+1}/{n_feat}] {feature}", flush=True)

print(f"  Per-subtype: {len(hexbin_rows_sub)} hexbin rows, "
      f"{len(metrics_rows_sub)} metric rows", flush=True)


# ==========================================================================
# Phase 3: Save parquet files
# ==========================================================================

# %%
print("\n=== Phase 3: Saving results ===", flush=True)

# Hexbin data
df_hex_all = pd.DataFrame(hexbin_rows_all)
df_hex_sub = pd.DataFrame(hexbin_rows_sub)
hex_all_path = RESULTS_DIR / "hexbin_data_all_cells.parquet"
hex_sub_path = RESULTS_DIR / "hexbin_data_per_subtype.parquet"
df_hex_all.to_parquet(hex_all_path, index=False)
df_hex_sub.to_parquet(hex_sub_path, index=False)
print(f"  Saved {hex_all_path.name}: {df_hex_all.shape}", flush=True)
print(f"  Saved {hex_sub_path.name}: {df_hex_sub.shape}", flush=True)

# Metrics
df_met = pd.DataFrame(metrics_rows_all + metrics_rows_sub)
met_path = RESULTS_DIR / "spatial_metrics.parquet"
df_met.to_parquet(met_path, index=False)
print(f"  Saved {met_path.name}: {df_met.shape}", flush=True)


# ==========================================================================
# Phase 4: Markdown summary
# ==========================================================================

# %%
print("\n=== Phase 4: Writing summary ===", flush=True)

all_met = df_met[df_met["scope"] == "all_cells"].copy()
all_met = all_met.set_index("feature")

lines = []
lines.append("# Spatial Distribution Analysis Summary\n")
lines.append(f"**Input**: `{INPUT_PARQUET.name}`  ")
lines.append(f"**Coordinate system**: legacy_transformed_x/y (electrode units x{COORD_SCALE} = microns)  ")
lines.append(f"**Spatial filter**: |coord| < {COORD_LIMIT} electrode units  ")
lines.append(f"**Cells after filter**: {len(df)}  ")
lines.append(f"**Features analysed**: {len(float_cols)}  ")
lines.append(f"**Valid subtypes**: {len(valid_subtypes)}  ")
lines.append("")

# --- Top features by gradient strength ---
lines.append("## Strongest Spatial Gradients (all cells)\n")
lines.append("Features with the largest linear gradient magnitude (feature ~ x + y).\n")
top_grad = all_met.nlargest(15, "gradient_mag")
lines.append("| Feature | Gradient Mag | Direction (deg) | R^2 |")
lines.append("|---------|-------------|-----------------|-----|")
for feat, row in top_grad.iterrows():
    lines.append(
        f"| {feat} | {row['gradient_mag']:.6f} | {row['gradient_dir_deg']:.1f} | {row['gradient_r2']:.4f} |"
    )
lines.append("")

# --- Top features by unevenness ---
lines.append("## Most Spatially Uneven Features (all cells)\n")
lines.append("Ranked by coefficient of variation (CV) of hexbin means.\n")
top_cv = all_met.nlargest(15, "hexbin_cv")
lines.append("| Feature | Hexbin CV | Gini | Moran's I |")
lines.append("|---------|----------|------|-----------|")
for feat, row in top_cv.iterrows():
    lines.append(
        f"| {feat} | {row['hexbin_cv']:.4f} | {row['hexbin_gini']:.4f} | {row['moran_i']:.4f} |"
    )
lines.append("")

# --- Top features by radial correlation ---
lines.append("## Strongest Radial Trends (all cells)\n")
lines.append("Features most correlated with distance from retinal centre.\n")
top_radial = all_met.reindex(
    all_met["radial_r"].abs().nlargest(15).index
)
lines.append("| Feature | Radial r | p-value | Slope (per um) | Centre mean | Periphery mean | C/P ratio |")
lines.append("|---------|---------|---------|----------------|-------------|----------------|-----------|")
for feat, row in top_radial.iterrows():
    lines.append(
        f"| {feat} | {row['radial_r']:.4f} | {row['radial_p']:.2e} | "
        f"{row['radial_slope']:.6f} | {row['center_mean']:.2f} | "
        f"{row['periphery_mean']:.2f} | {row['center_periphery_ratio']:.4f} |"
    )
lines.append("")

# --- Spatial autocorrelation ---
lines.append("## Spatial Autocorrelation (Moran's I, all cells)\n")
lines.append("Positive Moran's I indicates clustering; values near 0 indicate random distribution.\n")
top_moran = all_met.nlargest(15, "moran_i")
lines.append("| Feature | Moran's I | Hexbin CV | Gradient R^2 |")
lines.append("|---------|----------|----------|-------------|")
for feat, row in top_moran.iterrows():
    lines.append(
        f"| {feat} | {row['moran_i']:.4f} | {row['hexbin_cv']:.4f} | {row['gradient_r2']:.4f} |"
    )
lines.append("")

# --- Dominant quadrant summary ---
lines.append("## Dominant Quadrant Summary (all cells)\n")
lines.append("Quadrant with the highest mean value for each feature.\n")
quad_counts = all_met["dominant_quadrant"].value_counts()
lines.append("| Quadrant | # Features |")
lines.append("|----------|-----------|")
for q, cnt in quad_counts.items():
    lines.append(f"| {q} | {cnt} |")
lines.append("")

# --- Per-subtype highlights ---
lines.append("## Per-Subtype Highlights\n")
sub_met = df_met[df_met["scope"] != "all_cells"].copy()
if len(sub_met) > 0:
    # Subtype with highest gradient on average
    avg_grad = sub_met.groupby("scope")["gradient_mag"].mean().nlargest(5)
    lines.append("### Subtypes with strongest average gradient\n")
    lines.append("| Subtype | Avg Gradient Mag |")
    lines.append("|---------|-----------------|")
    for st, val in avg_grad.items():
        lines.append(f"| {st} | {val:.6f} |")
    lines.append("")

    # Subtype with highest Moran's I on average
    avg_moran = sub_met.groupby("scope")["moran_i"].mean().nlargest(5)
    lines.append("### Subtypes with strongest spatial clustering (avg Moran's I)\n")
    lines.append("| Subtype | Avg Moran's I |")
    lines.append("|---------|--------------|")
    for st, val in avg_moran.items():
        lines.append(f"| {st} | {val:.4f} |")
    lines.append("")

# --- Output file list ---
lines.append("## Output Files\n")
lines.append(f"- `{hex_all_path.name}` -- hexbin bin centres, raw means, GAM predictions (all cells)")
lines.append(f"- `{hex_sub_path.name}` -- hexbin bin centres, raw means (per subtype)")
lines.append(f"- `{met_path.name}` -- quantification metrics (all cells + per subtype)")
lines.append(f"- `figures_legacy/all_cells/` -- hexbin heatmaps (Raw + GAM) per feature")
lines.append(f"- `figures_legacy/per_subtype/` -- hexbin heatmaps per feature per subtype")
lines.append("")

md_path = RESULTS_DIR / "spatial_analysis_summary.md"
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  Saved {md_path.name}", flush=True)

print("\nDone.", flush=True)
