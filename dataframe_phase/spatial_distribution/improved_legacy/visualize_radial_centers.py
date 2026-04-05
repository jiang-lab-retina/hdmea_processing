"""
Visualize Radial Center Analysis Results
=========================================
Creates multiple figures:
  1. Radial center map (all cells, raw) - where each feature's optimal center is
  2. Radial center map (all cells, GAM)
  3. Improvement bar chart - origin vs optimal |r| for top features
  4. Radial profiles - for top features, feature value vs radius from optimal center
  5. Feature-group center clustering - centers colored by feature category
  6. Per-subtype center scatter for selected features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "figures_radial"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COORD_SCALE = 16
XY_RANGE = (-1600, 1600)

# ------------------------------------------------------------------
print("Loading data ...", flush=True)
df_rc_all = pd.read_parquet(RESULTS_DIR / "radial_center_all_cells.parquet")
df_rc_sub = pd.read_parquet(RESULTS_DIR / "radial_center_per_subtype.parquet")
df_hex = pd.read_parquet(RESULTS_DIR / "hexbin_data_all_cells.parquet")

rc_raw = df_rc_all[df_rc_all["data_type"] == "raw_mean"].copy()
rc_gam = df_rc_all[df_rc_all["data_type"] == "gam_pred"].copy()

# Feature categories for colour coding
def categorize_feature(f):
    if "freq_sinefit" in f:
        return "Freq sine-fit"
    if any(k in f for k in ["gaussian_", "dog_"]):
        return "RF spatial"
    if any(k in f for k in ["green_", "blue_", "gb_", "on_off", "on_peak", "off_peak",
                             "on_sustained", "off_sustained", "on_trans", "off_trans"]):
        return "Light response"
    if any(k in f for k in ["lnl_"]):
        return "LNL model"
    if any(k in f for k in ["dsi", "ds_p", "osi", "os_p", "preferred_direction"]):
        return "DS / OS"
    if any(k in f for k in ["time_to_"]):
        return "Temporal"
    if any(k in f for k in ["iprgc_"]):
        return "ipRGC"
    if any(k in f for k in ["base_mean", "base_std", "step_up", "chip_effective"]):
        return "Basic"
    return "Other"

CAT_COLORS = {
    "Freq sine-fit": "#e41a1c",
    "RF spatial": "#377eb8",
    "Light response": "#4daf4a",
    "LNL model": "#984ea3",
    "DS / OS": "#ff7f00",
    "Temporal": "#a65628",
    "ipRGC": "#f781bf",
    "Basic": "#999999",
    "Other": "#666666",
}

rc_raw["category"] = rc_raw["feature"].apply(categorize_feature)
rc_gam["category"] = rc_gam["feature"].apply(categorize_feature)


# ==================================================================
# Figure 1: Radial center map (raw mean)
# ==================================================================
print("Fig 1: Center map (raw) ...", flush=True)
fig, ax = plt.subplots(figsize=(10, 10))

# Background: light grey circle showing data extent
circle = plt.Circle((0, 0), 1600, fill=False, edgecolor="lightgrey", linewidth=1.5, linestyle="--")
ax.add_patch(circle)
ax.axhline(0, color="lightgrey", linewidth=0.5)
ax.axvline(0, color="lightgrey", linewidth=0.5)
ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2, zorder=5)

sig = rc_raw[rc_raw["best_p"] < 0.05]
nonsig = rc_raw[rc_raw["best_p"] >= 0.05]

# Non-significant: small grey dots
if len(nonsig) > 0:
    ax.scatter(nonsig["best_center_x"], nonsig["best_center_y"],
               s=30, c="lightgrey", edgecolors="grey", linewidths=0.5,
               alpha=0.5, zorder=3)

# Significant: colored by category, sized by |r|
for cat, color in CAT_COLORS.items():
    cat_data = sig[sig["category"] == cat]
    if len(cat_data) == 0:
        continue
    sizes = cat_data["best_r"].abs() * 500
    ax.scatter(cat_data["best_center_x"], cat_data["best_center_y"],
               s=sizes, c=color, edgecolors="black", linewidths=0.5,
               alpha=0.8, zorder=4, label=cat)

# Annotate top 10
top10 = rc_raw.reindex(rc_raw["best_r"].abs().nlargest(10).index)
for _, row in top10.iterrows():
    short = row["feature"].replace("freq_sinefit_", "fs_").replace("_r_squared", "_R2")
    short = short.replace("gaussian_", "g_").replace("_extreme", "")
    ax.annotate(short, (row["best_center_x"], row["best_center_y"]),
                fontsize=6.5, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect("equal")
ax.set_xlabel("X (um)  T <-- --> N", fontsize=12)
ax.set_ylabel("Y (um)  V <-- --> D", fontsize=12)
ax.set_title("Optimal Radial Centers (Raw Mean, All Cells)\n"
             "Marker size ~ |r|, color = feature category, + = origin", fontsize=12)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=0.5)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "radial_center_map_raw.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 2: Radial center map (GAM)
# ==================================================================
print("Fig 2: Center map (GAM) ...", flush=True)
fig, ax = plt.subplots(figsize=(10, 10))
circle = plt.Circle((0, 0), 1600, fill=False, edgecolor="lightgrey", linewidth=1.5, linestyle="--")
ax.add_patch(circle)
ax.axhline(0, color="lightgrey", linewidth=0.5)
ax.axvline(0, color="lightgrey", linewidth=0.5)
ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2, zorder=5)

sig_g = rc_gam[rc_gam["best_p"] < 0.05]
for cat, color in CAT_COLORS.items():
    cat_data = sig_g[sig_g["category"] == cat]
    if len(cat_data) == 0:
        continue
    sizes = cat_data["best_r"].abs() * 400
    ax.scatter(cat_data["best_center_x"], cat_data["best_center_y"],
               s=sizes, c=color, edgecolors="black", linewidths=0.5,
               alpha=0.8, zorder=4, label=cat)

top10_g = rc_gam.reindex(rc_gam["best_r"].abs().nlargest(10).index)
for _, row in top10_g.iterrows():
    short = row["feature"].replace("freq_sinefit_", "fs_").replace("_r_squared", "_R2")
    short = short.replace("gaussian_", "g_").replace("_extreme", "")
    ax.annotate(short, (row["best_center_x"], row["best_center_y"]),
                fontsize=6.5, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect("equal")
ax.set_xlabel("X (um)  T <-- --> N", fontsize=12)
ax.set_ylabel("Y (um)  V <-- --> D", fontsize=12)
ax.set_title("Optimal Radial Centers (GAM Smoothed, All Cells)\n"
             "Marker size ~ |r|, color = feature category", fontsize=12)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=0.5)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "radial_center_map_gam.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 3: Improvement bar chart - origin vs optimal
# ==================================================================
print("Fig 3: Improvement bar chart ...", flush=True)
top20 = rc_raw.reindex(rc_raw["abs_r_improvement"].nlargest(20).index).sort_values("abs_r_improvement")

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(top20))
bars_origin = ax.barh(y_pos - 0.15, top20["origin_r"].abs(), height=0.3,
                       color="#4292c6", label="Origin (0,0)", alpha=0.85)
bars_best = ax.barh(y_pos + 0.15, top20["best_r"].abs(), height=0.3,
                     color="#ef6548", label="Optimal center", alpha=0.85)

labels = [f.replace("freq_sinefit_", "fs_").replace("_r_squared", "_R2")
          for f in top20["feature"]]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("|Pearson r|", fontsize=12)
ax.set_title("Radial Trend: Origin vs Optimal Center (Raw Mean, Top 20 Improvements)", fontsize=12)
ax.legend(fontsize=10, loc="lower right")

# Add improvement text
for i, (_, row) in enumerate(top20.iterrows()):
    ax.text(row["best_r"].__abs__() + 0.005, y_pos[i] + 0.15,
            f"+{row['abs_r_improvement']:.3f}", fontsize=7, va="center")

ax.set_xlim(0, max(top20["best_r"].abs()) * 1.15)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "improvement_bar_chart.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 4: Radial profiles for top features
# ==================================================================
print("Fig 4: Radial profiles ...", flush=True)
top8 = rc_raw.reindex(rc_raw["best_r"].abs().nlargest(8).index)
# Exclude angle_correction_applied (not a biological feature)
top8 = top8[top8["feature"] != "angle_correction_applied"]
if len(top8) > 6:
    top8 = top8.head(6)

nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
axes_flat = axes.flatten()

for idx, (_, row) in enumerate(top8.iterrows()):
    if idx >= len(axes_flat):
        break
    ax = axes_flat[idx]
    feat = row["feature"]
    cx, cy = row["best_center_x"], row["best_center_y"]

    hex_feat = df_hex[df_hex["feature"] == feat]
    bx = hex_feat["bin_x"].to_numpy()
    by = hex_feat["bin_y"].to_numpy()
    vals_raw = hex_feat["raw_mean"].to_numpy()
    vals_gam = hex_feat["gam_pred"].to_numpy()

    # Radius from optimal center
    r_opt = np.sqrt((bx - cx)**2 + (by - cy)**2)
    # Radius from origin
    r_orig = np.sqrt(bx**2 + by**2)

    # Sort for plotting
    order_opt = np.argsort(r_opt)
    order_orig = np.argsort(r_orig)

    # Scatter: raw vs radius from optimal center
    ax.scatter(r_opt, vals_raw, s=8, c="#4292c6", alpha=0.3, label="Raw bins", zorder=2)

    # GAM vs radius from optimal center (if available)
    gam_mask = np.isfinite(vals_gam)
    if gam_mask.sum() > 10:
        ax.scatter(r_opt[gam_mask], vals_gam[gam_mask], s=8, c="#ef6548", alpha=0.3,
                   label="GAM pred", zorder=3)

    # Running median for raw (binned by radius)
    n_rbins = 15
    r_edges = np.linspace(0, r_opt.max(), n_rbins + 1)
    r_mids, med_vals = [], []
    for bi in range(n_rbins):
        mask_b = (r_opt >= r_edges[bi]) & (r_opt < r_edges[bi + 1])
        if mask_b.sum() >= 3:
            r_mids.append((r_edges[bi] + r_edges[bi + 1]) / 2)
            med_vals.append(np.median(vals_raw[mask_b]))
    if len(r_mids) > 2:
        ax.plot(r_mids, med_vals, "k-o", markersize=4, linewidth=2, label="Binned median", zorder=5)

    short = feat.replace("freq_sinefit_", "fs_").replace("_r_squared", "_R2")
    short = short.replace("gaussian_", "g_").replace("_extreme", "")
    direction = "periphery-high" if row["best_r"] > 0 else "center-high"
    ax.set_title(f"{short}\nr={row['best_r']:.3f}, center=({cx:.0f},{cy:.0f})\n{direction}",
                 fontsize=9)
    ax.set_xlabel("Radius from optimal center (um)", fontsize=8)
    ax.set_ylabel(feat, fontsize=7)
    ax.legend(fontsize=6, loc="best", framealpha=0.7)
    ax.tick_params(labelsize=7)

# Hide unused axes
for idx in range(len(top8), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Radial Profiles from Optimal Centers (Raw Mean, All Cells)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "radial_profiles_top.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 5: Feature-group center clustering
# ==================================================================
print("Fig 5: Feature-group clustering ...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

for ax, df_rc, title_suffix in [(ax1, rc_raw, "Raw Mean"), (ax2, rc_gam, "GAM")]:
    sig_data = df_rc[df_rc["best_p"] < 0.05].copy()
    circle = plt.Circle((0, 0), 1600, fill=False, edgecolor="lightgrey",
                         linewidth=1.5, linestyle="--")
    ax.add_patch(circle)
    ax.axhline(0, color="lightgrey", linewidth=0.5)
    ax.axvline(0, color="lightgrey", linewidth=0.5)
    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2, zorder=5)

    for cat, color in CAT_COLORS.items():
        cat_data = sig_data[sig_data["category"] == cat]
        if len(cat_data) == 0:
            continue
        ax.scatter(cat_data["best_center_x"], cat_data["best_center_y"],
                   s=100, c=color, edgecolors="black", linewidths=0.5,
                   alpha=0.75, zorder=4, label=f"{cat} (n={len(cat_data)})")

        # Draw ellipse at 1-std for each category with >= 3 features
        if len(cat_data) >= 3:
            mx = cat_data["best_center_x"].mean()
            my = cat_data["best_center_y"].mean()
            sx = cat_data["best_center_x"].std()
            sy = cat_data["best_center_y"].std()
            ellipse = matplotlib.patches.Ellipse(
                (mx, my), width=2*sx, height=2*sy,
                fill=False, edgecolor=color, linewidth=2, linestyle="--", alpha=0.6)
            ax.add_patch(ellipse)
            ax.plot(mx, my, "x", color=color, markersize=10, markeredgewidth=2, zorder=6)

    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-2200, 2200)
    ax.set_aspect("equal")
    ax.set_xlabel("X (um)", fontsize=11)
    ax.set_ylabel("Y (um)", fontsize=11)
    ax.set_title(f"Radial Centers by Feature Category ({title_suffix})", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

fig.tight_layout()
fig.savefig(str(FIG_DIR / "feature_group_clustering.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 6: Center direction + strength (vector plot)
# ==================================================================
print("Fig 6: Vector plot ...", flush=True)
fig, ax = plt.subplots(figsize=(10, 10))
circle = plt.Circle((0, 0), 1600, fill=False, edgecolor="lightgrey",
                     linewidth=1.5, linestyle="--")
ax.add_patch(circle)
ax.axhline(0, color="lightgrey", linewidth=0.5)
ax.axvline(0, color="lightgrey", linewidth=0.5)
ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2, zorder=5)

sig_raw = rc_raw[rc_raw["best_p"] < 0.05].copy()
# Arrow from center to origin (showing direction of "center"), length scaled by |r|
for _, row in sig_raw.iterrows():
    cx, cy = row["best_center_x"], row["best_center_y"]
    r_val = row["best_r"]
    # Color: red for center-high (negative r), blue for periphery-high (positive r)
    color = "#ef6548" if r_val < 0 else "#4292c6"
    alpha = min(1.0, abs(r_val) * 2.5)
    # Draw a dot at the center
    ax.plot(cx, cy, "o", color=color, markersize=max(3, abs(r_val) * 25),
            alpha=alpha, zorder=4, markeredgecolor="black", markeredgewidth=0.3)

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#4292c6", edgecolor="black", label="Periphery-high (r > 0)"),
    mpatches.Patch(facecolor="#ef6548", edgecolor="black", label="Center-high (r < 0)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

# Annotate top features
top_sig = sig_raw.reindex(sig_raw["best_r"].abs().nlargest(12).index)
for _, row in top_sig.iterrows():
    short = row["feature"].replace("freq_sinefit_", "fs_").replace("_r_squared", "_R2")
    short = short.replace("gaussian_", "g_").replace("_extreme", "")
    ax.annotate(short, (row["best_center_x"], row["best_center_y"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(6, 6), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, ec="none"))

ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect("equal")
ax.set_xlabel("X (um)  T <-- --> N", fontsize=12)
ax.set_ylabel("Y (um)  V <-- --> D", fontsize=12)
ax.set_title("Radial Centers: Direction of Trend (Raw Mean)\n"
             "Blue = increasing with distance, Red = decreasing with distance\n"
             "Marker size ~ |r|", fontsize=11)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "radial_direction_map.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 7: Per-subtype center scatter for key features
# ==================================================================
print("Fig 7: Per-subtype centers ...", flush=True)
key_features = ["green_blue_on_ratio", "dog_sigma_exc", "gaussian_sigma_y",
                "base_std", "off_peak_extreme", "dsi"]
# Filter to features that exist
available = set(df_rc_sub["feature"].unique())
key_features = [f for f in key_features if f in available]

if len(key_features) >= 3:
    ncols = 3
    nrows = (len(key_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, feat in enumerate(key_features):
        ax = axes_flat[idx]
        feat_data = df_rc_sub[(df_rc_sub["feature"] == feat) & (df_rc_sub["best_p"] < 0.05)]

        circle = plt.Circle((0, 0), 1600, fill=False, edgecolor="lightgrey",
                             linewidth=1, linestyle="--")
        ax.add_patch(circle)
        ax.axhline(0, color="lightgrey", linewidth=0.5)
        ax.axvline(0, color="lightgrey", linewidth=0.5)
        ax.plot(0, 0, "k+", markersize=10, markeredgewidth=1.5, zorder=5)

        if len(feat_data) > 0:
            sc = ax.scatter(feat_data["best_center_x"], feat_data["best_center_y"],
                           s=feat_data["best_r"].abs() * 400,
                           c=feat_data["best_r"], cmap="RdBu",
                           vmin=-0.6, vmax=0.6,
                           edgecolors="black", linewidths=0.5, alpha=0.8, zorder=4)
            fig.colorbar(sc, ax=ax, shrink=0.7, label="best_r")

            # Annotate top subtypes
            top3 = feat_data.reindex(feat_data["best_r"].abs().nlargest(3).index)
            for _, row in top3.iterrows():
                ax.annotate(row["scope"], (row["best_center_x"], row["best_center_y"]),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

            # Median center
            mx = feat_data["best_center_x"].median()
            my = feat_data["best_center_y"].median()
            ax.plot(mx, my, "D", color="gold", markersize=10, markeredgecolor="black",
                    markeredgewidth=1.5, zorder=6)

        ax.set_xlim(-2200, 2200)
        ax.set_ylim(-2200, 2200)
        ax.set_aspect("equal")
        short = feat.replace("freq_sinefit_", "fs_").replace("_r_squared", "_R2")
        n_sig = len(feat_data)
        ax.set_title(f"{short} ({n_sig} subtypes sig.)", fontsize=10)
        ax.set_xlabel("X (um)", fontsize=9)
        ax.set_ylabel("Y (um)", fontsize=9)

    for idx in range(len(key_features), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Per-Subtype Radial Centers for Key Features\n"
                 "Color = best_r (blue=periphery-high, red=center-high), "
                 "diamond = median center",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "per_subtype_centers.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Figure 8: Summary dashboard
# ==================================================================
print("Fig 8: Summary dashboard ...", flush=True)
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

# Panel A: Histogram of |best_r| raw
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(rc_raw["best_r"].abs(), bins=25, color="#4292c6", edgecolor="black", alpha=0.8)
ax_a.axvline(0.2, color="red", linestyle="--", linewidth=1, label="|r|=0.2")
ax_a.axvline(rc_raw["best_r"].abs().median(), color="orange", linestyle="--",
             linewidth=1.5, label=f"median={rc_raw['best_r'].abs().median():.3f}")
ax_a.set_xlabel("|best_r| (raw mean)", fontsize=10)
ax_a.set_ylabel("# features", fontsize=10)
ax_a.set_title("A. Distribution of |best_r| (raw)", fontsize=11)
ax_a.legend(fontsize=8)

# Panel B: Histogram of improvement
ax_b = fig.add_subplot(gs[0, 1])
ax_b.hist(rc_raw["abs_r_improvement"], bins=25, color="#ef6548", edgecolor="black", alpha=0.8)
ax_b.axvline(rc_raw["abs_r_improvement"].median(), color="orange", linestyle="--",
             linewidth=1.5, label=f"median={rc_raw['abs_r_improvement'].median():.3f}")
ax_b.set_xlabel("|r| improvement over origin", fontsize=10)
ax_b.set_ylabel("# features", fontsize=10)
ax_b.set_title("B. Improvement by Center Optimization", fontsize=11)
ax_b.legend(fontsize=8)

# Panel C: Scatter of origin_r vs best_r
ax_c = fig.add_subplot(gs[0, 2])
ax_c.scatter(rc_raw["origin_r"].abs(), rc_raw["best_r"].abs(),
             s=50, c="#4292c6", edgecolors="black", linewidths=0.5, alpha=0.7)
lim = max(rc_raw["origin_r"].abs().max(), rc_raw["best_r"].abs().max()) * 1.05
ax_c.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="y=x")
ax_c.set_xlabel("|origin_r|", fontsize=10)
ax_c.set_ylabel("|best_r|", fontsize=10)
ax_c.set_title("C. Origin |r| vs Optimal |r|", fontsize=11)
ax_c.legend(fontsize=8)
ax_c.set_xlim(0, lim)
ax_c.set_ylim(0, lim)
ax_c.set_aspect("equal")

# Panel D: Histogram of center distances from origin
dist_from_origin = np.sqrt(rc_raw["best_center_x"]**2 + rc_raw["best_center_y"]**2)
ax_d = fig.add_subplot(gs[1, 0])
ax_d.hist(dist_from_origin, bins=25, color="#41b6c4", edgecolor="black", alpha=0.8)
ax_d.axvline(dist_from_origin.median(), color="orange", linestyle="--",
             linewidth=1.5, label=f"median={dist_from_origin.median():.0f} um")
ax_d.set_xlabel("Distance of optimal center from origin (um)", fontsize=10)
ax_d.set_ylabel("# features", fontsize=10)
ax_d.set_title("D. How Far Centers Shift from Origin", fontsize=11)
ax_d.legend(fontsize=8)

# Panel E: Center angle distribution
angle_from_origin = np.degrees(np.arctan2(rc_raw["best_center_y"], rc_raw["best_center_x"]))
sig_angles = angle_from_origin[rc_raw["best_p"] < 0.05]
ax_e = fig.add_subplot(gs[1, 1], projection="polar")
# Histogram on polar
bins_theta = np.linspace(-180, 180, 25)
counts, _ = np.histogram(sig_angles, bins=bins_theta)
theta_centers = np.radians((bins_theta[:-1] + bins_theta[1:]) / 2)
width = np.radians(360 / 24)
ax_e.bar(theta_centers, counts, width=width, color="#4292c6", edgecolor="black",
         alpha=0.7)
ax_e.set_title("E. Angular Distribution\nof Optimal Centers", fontsize=11, pad=15)

# Panel F: Per-subtype number of significant features
sig_sub = df_rc_sub[df_rc_sub["best_p"] < 0.05]
if len(sig_sub) > 0:
    ax_f = fig.add_subplot(gs[1, 2])
    counts_sub = sig_sub.groupby("scope").size().sort_values(ascending=True)
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts_sub)))
    ax_f.barh(range(len(counts_sub)), counts_sub.values, color=colors_bar,
              edgecolor="black", linewidth=0.3)
    ax_f.set_yticks(range(len(counts_sub)))
    ax_f.set_yticklabels(counts_sub.index, fontsize=7)
    ax_f.set_xlabel("# features with sig. radial trend", fontsize=10)
    ax_f.set_title("F. Significant Radial Features per Subtype", fontsize=11)

fig.suptitle("Radial Center Analysis Dashboard", fontsize=15, y=1.01)
fig.savefig(str(FIG_DIR / "radial_dashboard.png"), dpi=250, bbox_inches="tight")
plt.close(fig)


print(f"\nDone. All figures saved in: {FIG_DIR}", flush=True)
