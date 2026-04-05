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
# # Spatial Distribution Plots v2 (Improved Coordinates - v6 robust ONH)
#
# Changes from v1:
#   - Per-subtype heatmaps: each subplot has its OWN color scale
#   - Saves hexbin data (raw + GAM) as parquet
#   - Writes detailed MD summary

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, math, warnings
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial import cKDTree

from pygam import LinearGAM, LogisticGAM, PoissonGAM, te

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %%
SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = SCRIPT_DIR.parent

INPUT_PARQUET = SCRIPT_DIR / "labeled_dataframe_improved_coords.parquet"
FIG_DIR = SCRIPT_DIR / "figures_v2"
FIG_ALL_DIR = FIG_DIR / "all_cells"
FIG_SUB_DIR = FIG_DIR / "per_subtype"
RESULTS_DIR = SCRIPT_DIR / "results"

for d in [FIG_DIR, FIG_ALL_DIR, FIG_SUB_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

X_COL = "improved_tx"
Y_COL = "improved_ty"
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
    "transformed_", "legacy_transformed_", "improved_t", "cartesian_",
    "polar_", "freq_step_", "ap_slope", "ap_intercept", "ap_r_value",
    "soma_row", "soma_col", "axon_centroids", "center_xy", "freq_section_",
]

# %%
print("Loading data ...", flush=True)
df_raw = pd.read_parquet(INPUT_PARQUET)
df_raw = df_raw.dropna(subset=[X_COL, Y_COL])
mask = (df_raw[X_COL].abs() < COORD_LIMIT) & (df_raw[Y_COL].abs() < COORD_LIMIT)
df = df_raw[mask].copy()
print(f"  Shape after filter: {df.shape}", flush=True)

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
# Helpers
# ==========================================================================

# %%
def _choose_gam_family(y):
    uniq = np.unique(y[np.isfinite(y)])
    if set(uniq).issubset({0, 1}):
        return LogisticGAM
    if np.all(y >= 0) and np.allclose(y, np.round(y)) and y.max() > 5:
        return PoissonGAM
    return LinearGAM


def _fit_gam(x, y, c, n_splines):
    """Fit GAM, return model or None."""
    GamClass = _choose_gam_family(c)
    X_train = np.column_stack([x, y])
    gam = GamClass(te(0, 1, n_splines=[n_splines, n_splines]))
    try:
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            gam = gam.gridsearch(X_train, c)
        return gam
    except Exception:
        try:
            gam.fit(X_train, c)
            return gam
        except Exception:
            return None


def extract_hexbin_data(x, y, c, gridsize, mincnt):
    """Returns (centers, raw_means, counts)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.mean, gridsize=gridsize,
                   extent=(*XY_RANGE, *XY_RANGE), mincnt=mincnt, cmap=CMAP)
    centers = hb.get_offsets().copy()
    means = hb.get_array().copy()
    ax.cla()
    hb2 = ax.hexbin(x, y, gridsize=gridsize, extent=(*XY_RANGE, *XY_RANGE),
                    mincnt=mincnt, cmap=CMAP)
    counts = hb2.get_array().copy()
    plt.close(fig)
    n = min(len(centers), len(means), len(counts))
    return centers[:n], means[:n], counts[:n]


def compute_moran_i(bin_centers, bin_values, k=6):
    n = len(bin_values)
    if n < k + 1:
        return np.nan
    z = bin_values - np.mean(bin_values)
    denom = np.sum(z ** 2)
    if denom == 0:
        return np.nan
    tree = cKDTree(bin_centers)
    _, idx = tree.query(bin_centers, k=min(k + 1, n))
    numer, W = 0.0, 0.0
    for i in range(n):
        for j_pos in range(1, idx.shape[1]):
            j = idx[i, j_pos]
            numer += z[i] * z[j]
            W += 1.0
    return float((n / W) * (numer / denom)) if W > 0 else np.nan


def compute_metrics(x_um, y_um, c, bin_centers, bin_means):
    m = {}
    m["n_valid"] = len(c)
    m["n_bins"] = len(bin_means)
    m["overall_mean"] = float(np.mean(c))
    m["overall_std"] = float(np.std(c))
    if len(bin_means) > 2:
        bm_mean = np.mean(bin_means)
        m["hexbin_cv"] = float(np.std(bin_means) / abs(bm_mean)) if bm_mean != 0 else np.nan
    else:
        m["hexbin_cv"] = np.nan
    if len(c) >= 10:
        A = np.column_stack([x_um, y_um, np.ones(len(x_um))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, c, rcond=None)
            gx, gy = coeffs[0], coeffs[1]
            pred = A @ coeffs
            ss_res = np.sum((c - pred) ** 2)
            ss_tot = np.sum((c - np.mean(c)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            m["gradient_mag"] = float(np.sqrt(gx**2 + gy**2))
            m["gradient_dir_deg"] = float(np.degrees(np.arctan2(gy, gx)))
            m["gradient_r2"] = float(r2)
        except Exception:
            m["gradient_mag"] = m["gradient_dir_deg"] = m["gradient_r2"] = np.nan
    else:
        m["gradient_mag"] = m["gradient_dir_deg"] = m["gradient_r2"] = np.nan
    radius = np.sqrt(x_um**2 + y_um**2)
    if len(c) >= 10 and np.std(radius) > 0 and np.std(c) > 0:
        try:
            r_val, p_val = pearsonr(radius, c)
            m["radial_r"] = float(r_val)
            m["radial_p"] = float(p_val)
        except Exception:
            m["radial_r"] = m["radial_p"] = np.nan
    else:
        m["radial_r"] = m["radial_p"] = np.nan
    if len(bin_means) >= 7:
        m["moran_i"] = compute_moran_i(bin_centers, bin_means)
    else:
        m["moran_i"] = np.nan
    return m


# ==========================================================================
# Plot functions
# ==========================================================================

# %%
def plot_allcells(x, y, c, feature, gam_model, save_dir):
    c_mean = float(np.mean(c))
    if c_mean == 0:
        return
    vmin = c_mean - 0.5 * abs(c_mean)
    vmax = c_mean + 0.5 * abs(c_mean)

    fig, (ax_raw, ax_gam) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(right=0.90, wspace=0.15)

    ax_raw.hexbin(x, y, C=c, reduce_C_function=np.mean, gridsize=GRIDSIZE_ALL,
                  extent=(*XY_RANGE, *XY_RANGE), mincnt=MINCNT_ALL,
                  cmap=CMAP, vmin=vmin, vmax=vmax)
    ax_raw.set_aspect("equal"); ax_raw.set_xlim(XY_RANGE); ax_raw.set_ylim(XY_RANGE)
    ax_raw.set_title("Raw mean", fontsize=11)

    if gam_model is not None:
        hb = ax_gam.hexbin(x, y, gridsize=GRIDSIZE_ALL,
                           extent=(*XY_RANGE, *XY_RANGE), mincnt=MINCNT_ALL, cmap=CMAP)
        offsets = hb.get_offsets()
        if len(offsets) > 0:
            hb.set_array(gam_model.predict(offsets))
            hb.set_clim(vmin=vmin, vmax=vmax)
    ax_gam.set_aspect("equal"); ax_gam.set_xlim(XY_RANGE); ax_gam.set_ylim(XY_RANGE)
    ax_gam.set_title("GAM smoothed", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_raw, ax_gam], shrink=0.75, pad=0.02).set_label(feature)
    fig.suptitle(feature, fontsize=13)
    fig.savefig(str(save_dir / f"Hexbin_{feature}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_persubtype(df_valid, feature, valid_subtypes, save_dir):
    """Per-subtype plot with INDIVIDUAL color scales per subplot."""
    n = len(valid_subtypes)
    pairs_per_row = 3
    grid_cols = pairs_per_row * 2
    nrows = math.ceil(n / pairs_per_row)
    fig, axes = plt.subplots(nrows, grid_cols,
                             figsize=(4 * grid_cols + 1.5, 3.8 * nrows), squeeze=False)
    fig.subplots_adjust(right=0.95, wspace=0.30, hspace=0.50)

    for idx, stype in enumerate(valid_subtypes):
        row = idx // pairs_per_row
        pc = idx % pairs_per_row
        ax_raw = axes[row][pc * 2]
        ax_gam = axes[row][pc * 2 + 1]

        sdata = df_valid[df_valid["subtype"] == stype]
        if len(sdata) < 3:
            for ax in (ax_raw, ax_gam):
                ax.set_title(stype, fontsize=8)
                ax.text(0.5, 0.5, "n < 3", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
                ax.set_aspect("equal")
            continue

        sx = sdata[X_COL].to_numpy() * COORD_SCALE
        sy = sdata[Y_COL].to_numpy() * COORD_SCALE
        sc = sdata[feature].to_numpy()

        # Per-subplot color scale
        sc_mean = float(np.mean(sc))
        if sc_mean == 0:
            sv_min, sv_max = np.min(sc), np.max(sc)
        else:
            sv_min = sc_mean - 0.5 * abs(sc_mean)
            sv_max = sc_mean + 0.5 * abs(sc_mean)

        hb_raw = ax_raw.hexbin(sx, sy, C=sc, reduce_C_function=np.mean,
                               gridsize=GRIDSIZE_SUB, extent=(*XY_RANGE, *XY_RANGE),
                               mincnt=MINCNT_SUB, cmap=CMAP, vmin=sv_min, vmax=sv_max)
        ax_raw.set_aspect("equal"); ax_raw.set_xlim(XY_RANGE); ax_raw.set_ylim(XY_RANGE)
        ax_raw.set_title(f"{stype} raw (n={len(sdata)})", fontsize=8)
        fig.colorbar(hb_raw, ax=ax_raw, shrink=0.6, pad=0.02)

        # GAM
        gam = _fit_gam(sx, sy, sc, N_SPLINES_SUB)
        if gam is not None:
            hb_g = ax_gam.hexbin(sx, sy, gridsize=GRIDSIZE_SUB,
                                 extent=(*XY_RANGE, *XY_RANGE),
                                 mincnt=MINCNT_SUB, cmap=CMAP)
            offsets = hb_g.get_offsets()
            if len(offsets) > 0:
                preds = gam.predict(offsets)
                hb_g.set_array(preds)
                hb_g.set_clim(vmin=sv_min, vmax=sv_max)
            fig.colorbar(hb_g, ax=ax_gam, shrink=0.6, pad=0.02)
        ax_gam.set_aspect("equal"); ax_gam.set_xlim(XY_RANGE); ax_gam.set_ylim(XY_RANGE)
        ax_gam.set_title(f"{stype} GAM", fontsize=8)

    for idx_h in range(n, nrows * pairs_per_row):
        r = idx_h // pairs_per_row
        pc = idx_h % pairs_per_row
        axes[r][pc * 2].set_visible(False)
        axes[r][pc * 2 + 1].set_visible(False)

    fig.suptitle(feature, fontsize=13, y=1.01)
    fig.savefig(str(save_dir / f"Hexbin_{feature}_subtypes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Main loop
# ==========================================================================

# %%
print("\n=== Dot plot ===", flush=True)
x_dot = df[X_COL].to_numpy() * COORD_SCALE
y_dot = df[Y_COL].to_numpy() * COORD_SCALE
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x_dot, y_dot, s=1, c="black", alpha=0.5, linewidths=0)
ax.set_aspect("equal"); ax.set_xlim(XY_RANGE); ax.set_ylim(XY_RANGE)
ax.set_xlabel("T <-- X (um) --> N", fontsize=12)
ax.set_ylabel("V <-- Y (um) --> D", fontsize=12)
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "dot_plot_all_cells.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved dot plot", flush=True)

# %%
print("\n=== Phase 1: All-cells (hexbin + GAM + metrics + plots) ===", flush=True)
hexbin_rows_all = []
metrics_rows_all = []
n_feat = len(float_cols)

for fi, feature in enumerate(float_cols):
    cols_needed = [X_COL, Y_COL, feature]
    data = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        continue
    x = data[X_COL].to_numpy() * COORD_SCALE
    y = data[Y_COL].to_numpy() * COORD_SCALE
    c = data[feature].to_numpy()

    # Hexbin data
    centers, raw_means, counts = extract_hexbin_data(x, y, c, GRIDSIZE_ALL, MINCNT_ALL)

    # GAM
    gam = _fit_gam(x, y, c, N_SPLINES_ALL)
    gam_preds = gam.predict(centers) if gam is not None and len(centers) > 0 else None

    for bi in range(len(centers)):
        hexbin_rows_all.append({
            "scope": "all_cells", "feature": feature,
            "bin_x": centers[bi, 0], "bin_y": centers[bi, 1],
            "count": int(counts[bi]), "raw_mean": float(raw_means[bi]),
            "gam_pred": float(gam_preds[bi]) if gam_preds is not None else np.nan,
        })

    m = compute_metrics(x, y, c, centers, raw_means)
    m["scope"] = "all_cells"
    m["feature"] = feature
    metrics_rows_all.append(m)

    # Plot
    plot_allcells(x, y, c, feature, gam, FIG_ALL_DIR)

    if (fi + 1) % 10 == 0 or (fi + 1) == n_feat:
        print(f"  [{fi+1}/{n_feat}] {feature}", flush=True)

print(f"  All-cells: {len(hexbin_rows_all)} hexbin rows, {len(metrics_rows_all)} metrics", flush=True)

# %%
print("\n=== Phase 2: Per-subtype (hexbin + metrics + plots) ===", flush=True)
hexbin_rows_sub = []
metrics_rows_sub = []
sub_df = df[df["valid_mosaic"] == True].copy()

for fi, feature in enumerate(float_cols):
    cols_needed = [X_COL, Y_COL, feature, "subtype"]
    data = sub_df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        continue

    for stype in valid_subtypes:
        sdata = data[data["subtype"] == stype]
        if len(sdata) < 3:
            continue
        sx = sdata[X_COL].to_numpy() * COORD_SCALE
        sy = sdata[Y_COL].to_numpy() * COORD_SCALE
        sc = sdata[feature].to_numpy()

        centers, raw_means, counts = extract_hexbin_data(sx, sy, sc, GRIDSIZE_SUB, MINCNT_SUB)
        for bi in range(len(centers)):
            hexbin_rows_sub.append({
                "scope": stype, "feature": feature,
                "bin_x": centers[bi, 0], "bin_y": centers[bi, 1],
                "count": int(counts[bi]), "raw_mean": float(raw_means[bi]),
                "gam_pred": np.nan,
            })
        m = compute_metrics(sx, sy, sc, centers, raw_means)
        m["scope"] = stype
        m["feature"] = feature
        metrics_rows_sub.append(m)

    # Per-subtype plot with individual color scales
    plot_persubtype(data, feature, valid_subtypes, FIG_SUB_DIR)

    if (fi + 1) % 10 == 0 or (fi + 1) == n_feat:
        print(f"  [{fi+1}/{n_feat}] {feature}", flush=True)

print(f"  Per-subtype: {len(hexbin_rows_sub)} hexbin rows, {len(metrics_rows_sub)} metrics", flush=True)


# ==========================================================================
# Phase 3: Save parquets
# ==========================================================================

# %%
print("\n=== Phase 3: Saving results ===", flush=True)

df_hex_all = pd.DataFrame(hexbin_rows_all)
df_hex_sub = pd.DataFrame(hexbin_rows_sub)
hex_all_path = RESULTS_DIR / "hexbin_data_all_cells.parquet"
hex_sub_path = RESULTS_DIR / "hexbin_data_per_subtype.parquet"
df_hex_all.to_parquet(hex_all_path, index=False)
df_hex_sub.to_parquet(hex_sub_path, index=False)
print(f"  {hex_all_path.name}: {df_hex_all.shape}", flush=True)
print(f"  {hex_sub_path.name}: {df_hex_sub.shape}", flush=True)

df_met = pd.DataFrame(metrics_rows_all + metrics_rows_sub)
met_path = RESULTS_DIR / "spatial_metrics.parquet"
df_met.to_parquet(met_path, index=False)
print(f"  {met_path.name}: {df_met.shape}", flush=True)


# ==========================================================================
# Phase 4: MD summary
# ==========================================================================

# %%
print("\n=== Phase 4: Writing summary ===", flush=True)

all_met = df_met[df_met["scope"] == "all_cells"].set_index("feature")
sub_met = df_met[df_met["scope"] != "all_cells"]

lines = []
lines.append("# Spatial Distribution Analysis (Improved Coordinates v6)\n")
lines.append(f"**Input**: `{INPUT_PARQUET.name}`  ")
lines.append(f"**ONH method**: Robust (R^2>0.7, median+MAD), legacy transform  ")
lines.append(f"**Coordinates**: improved_tx / improved_ty (electrode units x{COORD_SCALE} = microns)  ")
lines.append(f"**Spatial filter**: |coord| < {COORD_LIMIT} electrode units  ")
lines.append(f"**Cells after filter**: {len(df)}  ")
lines.append(f"**Features analysed**: {len(float_cols)}  ")
lines.append(f"**Valid subtypes**: {len(valid_subtypes)}  ")
lines.append(f"**Color scale**: per-subplot (mean +/- 50% of |mean|)  ")
lines.append("")

# Strongest gradients
lines.append("## Strongest Spatial Gradients (all cells)\n")
lines.append("| Feature | Gradient Mag | Direction (deg) | R^2 |")
lines.append("|---------|-------------|-----------------|-----|")
for feat, row in all_met.nlargest(15, "gradient_mag").iterrows():
    lines.append(f"| {feat} | {row['gradient_mag']:.6f} | {row['gradient_dir_deg']:.1f} | {row['gradient_r2']:.4f} |")
lines.append("")

# Most uneven
lines.append("## Most Spatially Uneven (all cells)\n")
lines.append("| Feature | Hexbin CV | Moran's I |")
lines.append("|---------|----------|-----------|")
for feat, row in all_met.nlargest(15, "hexbin_cv").iterrows():
    lines.append(f"| {feat} | {row['hexbin_cv']:.4f} | {row['moran_i']:.4f} |")
lines.append("")

# Strongest radial
lines.append("## Strongest Radial Trends (all cells)\n")
lines.append("| Feature | Radial r | p-value |")
lines.append("|---------|---------|---------|")
top_radial = all_met.reindex(all_met["radial_r"].abs().nlargest(15).index)
for feat, row in top_radial.iterrows():
    lines.append(f"| {feat} | {row['radial_r']:.4f} | {row['radial_p']:.2e} |")
lines.append("")

# Highest Moran's I
lines.append("## Strongest Spatial Clustering (Moran's I, all cells)\n")
lines.append("| Feature | Moran's I | Hexbin CV | Gradient R^2 |")
lines.append("|---------|----------|----------|-------------|")
for feat, row in all_met.nlargest(15, "moran_i").iterrows():
    lines.append(f"| {feat} | {row['moran_i']:.4f} | {row['hexbin_cv']:.4f} | {row['gradient_r2']:.4f} |")
lines.append("")

# Per-subtype summary
if len(sub_met) > 0:
    lines.append("## Per-Subtype Highlights\n")
    avg_grad = sub_met.groupby("scope")["gradient_mag"].mean().nlargest(10)
    lines.append("### Subtypes with strongest average gradient\n")
    lines.append("| Subtype | Avg Gradient Mag |")
    lines.append("|---------|-----------------|")
    for st, val in avg_grad.items():
        lines.append(f"| {st} | {val:.6f} |")
    lines.append("")

    avg_moran = sub_met.groupby("scope")["moran_i"].mean().nlargest(10)
    lines.append("### Subtypes with strongest spatial clustering\n")
    lines.append("| Subtype | Avg Moran's I |")
    lines.append("|---------|--------------|")
    for st, val in avg_moran.items():
        lines.append(f"| {st} | {val:.4f} |")
    lines.append("")

# Full feature table
lines.append("## All Features Summary (all cells)\n")
lines.append("| Feature | Mean | Std | n_bins | CV | Grad Mag | Grad Dir | Grad R^2 | Radial r | Moran I |")
lines.append("|---------|------|-----|--------|-----|----------|----------|----------|----------|---------|")
for feat in sorted(all_met.index):
    r = all_met.loc[feat]
    lines.append(
        f"| {feat} | {r['overall_mean']:.3f} | {r['overall_std']:.3f} | {r['n_bins']} | "
        f"{r['hexbin_cv']:.3f} | {r['gradient_mag']:.6f} | {r['gradient_dir_deg']:.1f} | "
        f"{r['gradient_r2']:.4f} | {r['radial_r']:.4f} | {r['moran_i']:.4f} |"
    )
lines.append("")

# Output files
lines.append("## Output Files\n")
lines.append(f"- `{hex_all_path.name}` ({df_hex_all.shape[0]} rows) -- hexbin data, all cells")
lines.append(f"- `{hex_sub_path.name}` ({df_hex_sub.shape[0]} rows) -- hexbin data, per subtype")
lines.append(f"- `{met_path.name}` ({df_met.shape[0]} rows) -- spatial metrics")
lines.append(f"- `figures_v2/all_cells/` -- {n_feat} all-cells heatmaps (Raw + GAM)")
lines.append(f"- `figures_v2/per_subtype/` -- {n_feat} per-subtype heatmaps (individual color scales)")

md_path = RESULTS_DIR / "spatial_analysis_summary.md"
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  {md_path.name}", flush=True)
print("\nDone.", flush=True)
