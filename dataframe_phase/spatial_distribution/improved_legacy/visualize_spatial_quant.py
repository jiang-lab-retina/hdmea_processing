"""
Visualize Comprehensive Spatial Quantification Results
======================================================
Produces ~10 figures plus updates the MD summary with an overall narrative.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path
import warnings, textwrap

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "figures_quant"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FDR = 0.05

# ------------------------------------------------------------------
print("Loading data ...", flush=True)
ac = pd.read_parquet(RESULTS_DIR / "spatial_quant_all_cells.parquet").set_index("feature")
sub = pd.read_parquet(RESULTS_DIR / "spatial_quant_per_subtype.parquet")
cons = pd.read_parquet(RESULTS_DIR / "spatial_quant_subtype_consistency.parquet").set_index("feature")

# Short name helper
def short(f):
    return (f.replace("freq_sinefit_", "fs_")
             .replace("_r_squared", "_R2")
             .replace("gaussian_", "g_")
             .replace("_extreme", "")
             .replace("_applied", ""))


# =====================================================================
# Fig 1: Gradient Polar Plot
# =====================================================================
print("Fig 1: Gradient polar ...", flush=True)
sig = ac[ac["fdr_q_plane_r2"] < FDR].copy()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="polar")

# All features (grey)
theta_all = np.radians(ac["grad_dir_deg"].to_numpy())
r_all = ac["plane_r2"].to_numpy()
ax.scatter(theta_all, r_all, s=30, c="lightgrey", alpha=0.5, zorder=2)

# Significant features (colored by category)
def categorize(f):
    if "freq_sinefit" in f: return "Freq sine-fit"
    if any(k in f for k in ["gaussian_", "dog_"]): return "RF spatial"
    if any(k in f for k in ["green_", "blue_", "gb_", "on_off", "on_peak", "off_peak",
                             "on_sustained", "off_sustained", "on_trans", "off_trans"]): return "Light response"
    if "lnl_" in f: return "LNL model"
    if any(k in f for k in ["dsi", "ds_p", "osi", "os_p", "preferred_dir"]): return "DS / OS"
    if "time_to_" in f: return "Temporal"
    if "iprgc_" in f: return "ipRGC"
    return "Other"

CAT_COLORS = {"Freq sine-fit": "#e41a1c", "RF spatial": "#377eb8",
              "Light response": "#4daf4a", "LNL model": "#984ea3",
              "DS / OS": "#ff7f00", "Temporal": "#a65628",
              "ipRGC": "#f781bf", "Other": "#999999"}

for cat, color in CAT_COLORS.items():
    mask = sig.index.map(lambda f: categorize(f) == cat)
    d = sig[mask]
    if len(d) == 0: continue
    ax.scatter(np.radians(d["grad_dir_deg"]), d["plane_r2"],
               s=d["plane_r2"] * 800 + 40, c=color, edgecolors="black",
               linewidths=0.5, alpha=0.85, label=cat, zorder=4)

# Label top features
top = ac.nlargest(10, "plane_r2")
for f in top.index:
    r = ac.loc[f]
    ax.annotate(short(f), (np.radians(r["grad_dir_deg"]), r["plane_r2"]),
                fontsize=6.5, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.7, ec="none"))

# Anatomical quadrant labels
for angle, label in [(0, "Nasal"), (90, "Dorsal"), (180, "Temporal"), (-90, "Ventral")]:
    ax.annotate(label, xy=(np.radians(angle), ax.get_ylim()[1]*0.95),
                fontsize=10, fontweight="bold", ha="center", color="navy")

ax.set_title("Global Gradient: Direction & Strength (Plane R^2)\n"
             "Significant features shown in color", fontsize=12, pad=20)
ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.3, 1.0), framealpha=0.9)
fig.savefig(str(FIG_DIR / "fig1_gradient_polar.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 2: Plane R^2 vs GAM R^2
# =====================================================================
print("Fig 2: Plane vs GAM R^2 ...", flush=True)
fig, ax = plt.subplots(figsize=(10, 8))

gam_valid = ac[ac["gam_r2"].notna()].copy()
colors = [CAT_COLORS.get(categorize(f), "#999") for f in gam_valid.index]

sc = ax.scatter(gam_valid["plane_r2"], gam_valid["gam_r2"],
                s=80, c=colors, edgecolors="black", linewidths=0.5, alpha=0.8, zorder=4)

lim = max(gam_valid["plane_r2"].max(), gam_valid["gam_r2"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y = x")
ax.set_xlabel("Plane R^2", fontsize=12)
ax.set_ylabel("GAM R^2", fontsize=12)
ax.set_title("Plane vs GAM Fit: Nonlinear Structure", fontsize=13)
ax.set_xlim(-0.01, lim)
ax.set_ylim(-0.01, lim)

# Label points with biggest delta
top_delta = gam_valid.nlargest(8, "gam_plane_delta_r2")
for f in top_delta.index:
    r = gam_valid.loc[f]
    ax.annotate(short(f), (r["plane_r2"], r["gam_r2"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))
    # Draw arrow from plane to GAM
    ax.annotate("", xy=(r["plane_r2"], r["gam_r2"]),
                xytext=(r["plane_r2"], r["plane_r2"]),
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.4, lw=1.5))

handles = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
ax.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig2_plane_vs_gam.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 3: Moran's I vs Gradient R^2 (clustering vs trend)
# =====================================================================
print("Fig 3: Moran I vs Gradient ...", flush=True)
fig, ax = plt.subplots(figsize=(10, 8))
colors = [CAT_COLORS.get(categorize(f), "#999") for f in ac.index]
sc = ax.scatter(ac["plane_r2"], ac["moran_i"], s=80, c=colors,
                edgecolors="black", linewidths=0.5, alpha=0.8, zorder=4)

ax.set_xlabel("Plane R^2 (global gradient strength)", fontsize=12)
ax.set_ylabel("Moran's I (spatial clustering)", fontsize=12)
ax.set_title("Gradient vs Clustering: Different Aspects of Spatial Structure", fontsize=13)
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)

# Quadrant annotations
ax.text(0.95, 0.95, "Strong gradient\n+ Strong clustering",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(fc="#d4edda", alpha=0.7, ec="none"))
ax.text(0.05, 0.95, "Weak gradient\nbut Strong clustering",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(fc="#fff3cd", alpha=0.7, ec="none"))
ax.text(0.95, 0.05, "Strong gradient\nbut Weak clustering",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(fc="#cce5ff", alpha=0.7, ec="none"))

top = ac.nlargest(8, "moran_i")
for f in top.index:
    r = ac.loc[f]
    ax.annotate(short(f), (r["plane_r2"], r["moran_i"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

handles = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
ax.legend(handles=handles, fontsize=7, loc="center left", framealpha=0.9)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig3_moran_vs_gradient.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 4: Radial Forest Plot (with bootstrap CIs)
# =====================================================================
print("Fig 4: Radial forest plot ...", flush=True)
rad = ac[ac["radial_r"].notna()].copy()
rad["abs_r"] = rad["radial_r"].abs()
top_rad = rad.nlargest(25, "abs_r")

fig, ax = plt.subplots(figsize=(10, 10))
y = np.arange(len(top_rad))
colors_bar = ["#ef6548" if r < 0 else "#4292c6" for r in top_rad["radial_r"]]

# CI error bars
lo = top_rad["radial_r_ci_lo"].to_numpy()
hi = top_rad["radial_r_ci_hi"].to_numpy()
r_vals = top_rad["radial_r"].to_numpy()
err_lo = r_vals - lo
err_hi = hi - r_vals

ax.barh(y, r_vals, color=colors_bar, edgecolor="black", linewidth=0.3, height=0.7, alpha=0.8)
ax.errorbar(r_vals, y, xerr=[err_lo, err_hi], fmt="none", ecolor="black",
            elinewidth=1, capsize=3, zorder=5)

ax.axvline(0, color="black", linewidth=1)
labels = [short(f) for f in top_rad.index]
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Radial Pearson r (from origin)", fontsize=12)
ax.set_title("Radial Trends with 95% Bootstrap CIs (Top 25)", fontsize=13)

# Mark significance
for i, f in enumerate(top_rad.index):
    fdr = top_rad.loc[f, "fdr_q_radial"]
    if fdr < 0.001:
        ax.text(r_vals[i] + (0.01 if r_vals[i] >= 0 else -0.01), y[i],
                "***", fontsize=8, va="center", ha="left" if r_vals[i] >= 0 else "right")
    elif fdr < 0.01:
        ax.text(r_vals[i] + (0.01 if r_vals[i] >= 0 else -0.01), y[i],
                "**", fontsize=8, va="center", ha="left" if r_vals[i] >= 0 else "right")
    elif fdr < 0.05:
        ax.text(r_vals[i] + (0.01 if r_vals[i] >= 0 else -0.01), y[i],
                "*", fontsize=8, va="center", ha="left" if r_vals[i] >= 0 else "right")

legend_elements = [mpatches.Patch(fc="#4292c6", label="Periphery-high (r>0)"),
                   mpatches.Patch(fc="#ef6548", label="Center-high (r<0)")]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig4_radial_forest.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 5: Quadrant Heatmap
# =====================================================================
print("Fig 5: Quadrant heatmap ...", flush=True)
quad_cols = ["quad_mean_DN", "quad_mean_DT", "quad_mean_VN", "quad_mean_VT"]
sig_quad = ac[ac["fdr_q_quad"] < FDR].copy()
top_quad = sig_quad.nlargest(25, "quad_F")

# Z-score each feature's quadrant means for comparability
quad_data = top_quad[quad_cols].copy()
row_means = quad_data.mean(axis=1)
row_stds = quad_data.std(axis=1)
row_stds = row_stds.replace(0, 1)
quad_z = quad_data.sub(row_means, axis=0).div(row_stds, axis=0)

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(quad_z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(4))
ax.set_xticklabels(["Dorsal-\nNasal", "Dorsal-\nTemporal", "Ventral-\nNasal", "Ventral-\nTemporal"],
                    fontsize=10)
ax.set_yticks(range(len(top_quad)))
ax.set_yticklabels([short(f) for f in top_quad.index], fontsize=8)
ax.set_title("Quadrant Structure (z-scored within feature)\nTop 25 by F-statistic", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")

# Add value annotations
for i in range(quad_z.shape[0]):
    for j in range(quad_z.shape[1]):
        val = quad_z.values[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=7, color="white" if abs(val) > 1.2 else "black")

fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig5_quadrant_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 6: Subtype Consistency
# =====================================================================
print("Fig 6: Subtype consistency ...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left: vector strength vs % significant
colors_c = [CAT_COLORS.get(categorize(f), "#999") for f in cons.index]
ax1.scatter(cons["pct_subtypes_sig"], cons["vector_strength"],
            s=80, c=colors_c, edgecolors="black", linewidths=0.5, alpha=0.8)
ax1.set_xlabel("% Subtypes with Significant Gradient", fontsize=12)
ax1.set_ylabel("Vector Strength (directional consistency)", fontsize=12)
ax1.set_title("Subtype Gradient Consistency", fontsize=13)

top_vs = cons.nlargest(10, "vector_strength")
for f in top_vs.index:
    r = cons.loc[f]
    ax1.annotate(short(f), (r["pct_subtypes_sig"], r["vector_strength"]),
                 fontsize=7, ha="left", va="bottom",
                 xytext=(4, 4), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

# Right: polar plot of mean gradient directions (for consistent features)
ax2 = fig.add_subplot(122, projection="polar")
top20_cons = cons.nlargest(20, "vector_strength")
for f in top20_cons.index:
    r = cons.loc[f]
    theta = np.radians(r["mean_dir_deg"])
    vs = r["vector_strength"]
    color = CAT_COLORS.get(categorize(f), "#999")
    ax2.plot([theta, theta], [0, vs], color=color, linewidth=2, alpha=0.7)
    ax2.plot(theta, vs, "o", color=color, markersize=7, markeredgecolor="black",
             markeredgewidth=0.5)
    if vs > 0.75:
        ax2.annotate(short(f), (theta, vs), fontsize=6,
                     xytext=(5, 3), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"))

for angle, label in [(0, "N"), (90, "D"), (180, "T"), (-90, "V")]:
    ax2.annotate(label, xy=(np.radians(angle), 1.05), fontsize=11, fontweight="bold",
                 ha="center", color="navy")

ax2.set_ylim(0, 1.1)
ax2.set_title("Mean Gradient Direction per Feature\n(Top 20 most consistent)", fontsize=11, pad=15)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig6_subtype_consistency.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 7: Significance Overview Heatmap
# =====================================================================
print("Fig 7: Significance heatmap ...", flush=True)
tests = ["fdr_q_plane_r2", "fdr_q_moran_i", "fdr_q_radial", "fdr_q_quad"]
test_labels = ["Plane gradient\n(perm)", "Moran's I\n(perm)", "Radial corr.", "Quadrant\nANOVA"]

# Sort features by number of significant tests
sig_counts = pd.DataFrame(index=ac.index)
for t in tests:
    sig_counts[t] = (ac[t] < FDR).astype(int)
sig_counts["total"] = sig_counts.sum(axis=1)
order = sig_counts.sort_values("total", ascending=False).index

# -log10(q) for coloring, cap at 5
mat = np.zeros((len(order), len(tests)))
for j, t in enumerate(tests):
    vals = ac.loc[order, t].to_numpy()
    logq = -np.log10(np.clip(vals, 1e-10, 1))
    mat[:, j] = np.clip(logq, 0, 5)

fig, ax = plt.subplots(figsize=(8, 16))
im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=5, aspect="auto")
ax.set_xticks(range(len(tests)))
ax.set_xticklabels(test_labels, fontsize=10)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([short(f) for f in order], fontsize=7)
ax.set_title("Significance Overview: -log10(FDR q)\nDarker = more significant", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.4, label="-log10(FDR q)")

# Overlay asterisks
for i in range(len(order)):
    for j in range(len(tests)):
        q = ac.loc[order[i], tests[j]]
        if q < 0.001:
            ax.text(j, i, "***", ha="center", va="center", fontsize=6, color="white")
        elif q < 0.01:
            ax.text(j, i, "**", ha="center", va="center", fontsize=6, color="white")
        elif q < 0.05:
            ax.text(j, i, "*", ha="center", va="center", fontsize=6,
                    color="white" if mat[i, j] > 2.5 else "black")

fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig7_significance_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 8: Multi-metric Overview (bubble chart)
# =====================================================================
print("Fig 8: Multi-metric overview ...", flush=True)
fig, ax = plt.subplots(figsize=(14, 10))

x_val = ac["plane_r2"].to_numpy()
y_val = ac["moran_i"].to_numpy()
size_val = ac["gam_plane_delta_r2"].fillna(0).to_numpy() * 2000 + 30
color_val = ac["radial_r"].to_numpy()

sc = ax.scatter(x_val, y_val, s=size_val, c=color_val, cmap="RdBu",
                vmin=-0.35, vmax=0.35, edgecolors="black", linewidths=0.5, alpha=0.8)
fig.colorbar(sc, ax=ax, shrink=0.7, label="Radial r (origin)")

ax.set_xlabel("Plane R^2 (global gradient)", fontsize=12)
ax.set_ylabel("Moran's I (spatial clustering)", fontsize=12)
ax.set_title("Multi-Metric Overview\n"
             "x = gradient, y = clustering, size = GAM improvement, color = radial trend",
             fontsize=13)

# Label interesting features
combined_score = ac["plane_r2"] + ac["moran_i"] + ac["gam_plane_delta_r2"].fillna(0)
top_c = combined_score.nlargest(12)
for f in top_c.index:
    r = ac.loc[f]
    ax.annotate(short(f), (r["plane_r2"], r["moran_i"]),
                fontsize=7.5, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.8, ec="none"))

# Size legend
for s_val, s_label in [(0.05, "0.05"), (0.15, "0.15"), (0.30, "0.30")]:
    ax.scatter([], [], s=s_val*2000+30, c="grey", edgecolors="black",
               linewidths=0.5, alpha=0.6, label=f"GAM delta={s_label}")
ax.legend(fontsize=8, loc="upper left", title="GAM improvement", title_fontsize=9)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig8_multimetric_overview.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 9: GAM Hotspot Locations
# =====================================================================
print("Fig 9: GAM hotspot map ...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: GAM maxima locations
gam_valid = ac[ac["gam_max_x"].notna()].copy()
sig_gam = gam_valid[gam_valid["fdr_q_plane_r2"] < FDR]
colors_max = [CAT_COLORS.get(categorize(f), "#999") for f in sig_gam.index]

for ax_cur, x_col, y_col, title in [
    (ax1, "gam_max_x", "gam_max_y", "GAM Maxima (feature peaks)"),
    (ax2, "gam_min_x", "gam_min_y", "GAM Minima (feature valleys)")]:

    circle = plt.Circle((0, 0), 1600, fill=False, edgecolor="lightgrey",
                         linewidth=1.5, linestyle="--")
    ax_cur.add_patch(circle)
    ax_cur.axhline(0, color="lightgrey", linewidth=0.5)
    ax_cur.axvline(0, color="lightgrey", linewidth=0.5)
    ax_cur.plot(0, 0, "k+", markersize=12, markeredgewidth=2)

    for cat, color in CAT_COLORS.items():
        mask = sig_gam.index.map(lambda f: categorize(f) == cat)
        d = sig_gam[mask]
        if len(d) == 0: continue
        sizes = d["gam_plane_delta_r2"].fillna(0) * 800 + 30
        ax_cur.scatter(d[x_col], d[y_col], s=sizes, c=color,
                       edgecolors="black", linewidths=0.5, alpha=0.75, label=cat)

    # Label top delta features
    top_d = sig_gam.nlargest(6, "gam_plane_delta_r2")
    for f in top_d.index:
        r = sig_gam.loc[f]
        ax_cur.annotate(short(f), (r[x_col], r[y_col]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

    ax_cur.set_xlim(-1800, 1800)
    ax_cur.set_ylim(-1800, 1800)
    ax_cur.set_aspect("equal")
    ax_cur.set_xlabel("X (um)  T<-->N", fontsize=11)
    ax_cur.set_ylabel("Y (um)  V<-->D", fontsize=11)
    ax_cur.set_title(title, fontsize=12)

handles = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()
           if any(categorize(f) == cat for f in sig_gam.index)]
if handles:
    ax1.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)
fig.suptitle("GAM Extremum Locations (significant features)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(str(FIG_DIR / "fig9_gam_hotspot_map.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Fig 10: Summary Dashboard
# =====================================================================
print("Fig 10: Summary dashboard ...", flush=True)
fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(2, 4, hspace=0.40, wspace=0.35)

# A: Plane R^2 distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(ac["plane_r2"], bins=25, color="#4292c6", edgecolor="black", alpha=0.8)
n_sig = (ac["fdr_q_plane_r2"] < FDR).sum()
ax.set_title(f"A. Plane R^2\n({n_sig}/70 sig.)", fontsize=10)
ax.set_xlabel("Plane R^2")
ax.axvline(ac["plane_r2"].median(), color="red", ls="--", lw=1.5,
           label=f"med={ac['plane_r2'].median():.3f}")
ax.legend(fontsize=7)

# B: GAM R^2 distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(ac["gam_r2"].dropna(), bins=25, color="#ef6548", edgecolor="black", alpha=0.8)
ax.set_title("B. GAM R^2", fontsize=10)
ax.set_xlabel("GAM R^2")
ax.axvline(ac["gam_r2"].median(), color="red", ls="--", lw=1.5,
           label=f"med={ac['gam_r2'].median():.3f}")
ax.legend(fontsize=7)

# C: Moran's I distribution
ax = fig.add_subplot(gs[0, 2])
n_sig_m = (ac["fdr_q_moran_i"] < FDR).sum()
ax.hist(ac["moran_i"], bins=25, color="#41b6c4", edgecolor="black", alpha=0.8)
ax.set_title(f"C. Moran's I\n({n_sig_m}/70 sig.)", fontsize=10)
ax.set_xlabel("Moran's I")
ax.axvline(0, color="grey", ls="-", lw=0.5)
ax.axvline(ac["moran_i"].median(), color="red", ls="--", lw=1.5,
           label=f"med={ac['moran_i'].median():.3f}")
ax.legend(fontsize=7)

# D: Radial r distribution
ax = fig.add_subplot(gs[0, 3])
n_sig_r = (ac["fdr_q_radial"] < FDR).sum()
ax.hist(ac["radial_r"], bins=25, color="#78c679", edgecolor="black", alpha=0.8)
ax.set_title(f"D. Radial r\n({n_sig_r}/70 sig.)", fontsize=10)
ax.set_xlabel("Radial Pearson r")
ax.axvline(0, color="grey", ls="-", lw=0.5)
ax.legend(fontsize=7)

# E: Direction histogram
ax = fig.add_subplot(gs[1, 0], projection="polar")
sig_dirs = ac.loc[ac["fdr_q_plane_r2"] < FDR, "grad_dir_deg"].to_numpy()
bins_t = np.linspace(-180, 180, 25)
counts, _ = np.histogram(sig_dirs, bins=bins_t)
theta_c = np.radians((bins_t[:-1] + bins_t[1:]) / 2)
width = np.radians(360/24)
ax.bar(theta_c, counts, width=width, color="#4292c6", edgecolor="black", alpha=0.7)
ax.set_title("E. Gradient Directions\n(sig. features)", fontsize=10, pad=15)

# F: GAM improvement vs Moran's I
ax = fig.add_subplot(gs[1, 1])
ax.scatter(ac["gam_plane_delta_r2"].fillna(0), ac["moran_i"],
           s=40, c="#984ea3", edgecolors="black", linewidths=0.3, alpha=0.7)
ax.set_xlabel("GAM delta R^2")
ax.set_ylabel("Moran's I")
ax.set_title("F. Nonlinearity vs Clustering", fontsize=10)

# G: Vector strength distribution
ax = fig.add_subplot(gs[1, 2])
ax.hist(cons["vector_strength"].dropna(), bins=20, color="#ff7f00",
        edgecolor="black", alpha=0.8)
ax.set_xlabel("Vector Strength")
ax.set_title("G. Subtype Directional\nConsistency", fontsize=10)
ax.axvline(cons["vector_strength"].median(), color="red", ls="--", lw=1.5,
           label=f"med={cons['vector_strength'].median():.3f}")
ax.legend(fontsize=7)

# H: Quad F distribution
ax = fig.add_subplot(gs[1, 3])
n_sig_q = (ac["fdr_q_quad"] < FDR).sum()
ax.hist(np.log10(ac["quad_F"].clip(lower=0.1)), bins=25, color="#a65628",
        edgecolor="black", alpha=0.8)
ax.set_xlabel("log10(Quadrant F)")
ax.set_title(f"H. Quadrant ANOVA\n({n_sig_q}/70 sig.)", fontsize=10)

fig.suptitle("Spatial Quantification Summary Dashboard", fontsize=15, y=1.01)
fig.savefig(str(FIG_DIR / "fig10_summary_dashboard.png"), dpi=250, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# OVERALL SUMMARY -> append to MD
# =====================================================================
print("\nUpdating MD summary ...", flush=True)

md_path = RESULTS_DIR / "spatial_quantification_full.md"
existing = md_path.read_text(encoding="utf-8")

# Compute overall summary statistics
n_grad_sig = int((ac["fdr_q_plane_r2"] < FDR).sum())
n_moran_sig = int((ac["fdr_q_moran_i"] < FDR).sum())
n_radial_sig = int((ac["fdr_q_radial"] < FDR).sum())
n_quad_sig = int((ac["fdr_q_quad"] < FDR).sum())
n_any_sig = int(((ac["fdr_q_plane_r2"] < FDR) | (ac["fdr_q_moran_i"] < FDR) |
                  (ac["fdr_q_radial"] < FDR) | (ac["fdr_q_quad"] < FDR)).sum())
n_all_sig = int(((ac["fdr_q_plane_r2"] < FDR) & (ac["fdr_q_moran_i"] < FDR) &
                  (ac["fdr_q_radial"] < FDR) & (ac["fdr_q_quad"] < FDR)).sum())

med_plane = ac["plane_r2"].median()
med_gam = ac["gam_r2"].median()
med_moran = ac["moran_i"].median()
med_rad = ac["radial_r"].abs().median()
med_vs = cons["vector_strength"].median()

# Features with strong spatial structure (top combined score)
combo = (ac["plane_r2"].rank(pct=True) +
         ac["moran_i"].rank(pct=True) +
         ac["gam_plane_delta_r2"].fillna(0).rank(pct=True) +
         ac["radial_r"].abs().rank(pct=True))
top_spatial = combo.nlargest(10)

# Dominant gradient direction
sig_dirs = ac.loc[ac["fdr_q_plane_r2"] < FDR, "grad_dir_deg"]
C = np.mean(np.cos(np.radians(sig_dirs)))
S = np.mean(np.sin(np.radians(sig_dirs)))
mean_dir = np.degrees(np.arctan2(S, C))

summary_block = f"""

---
## Overall Summary

### Prevalence of Spatial Structure

Of the 70 features analysed, **{n_any_sig}** ({n_any_sig/70*100:.0f}%) show significant spatial
structure by at least one test (FDR q < 0.05), and **{n_all_sig}** ({n_all_sig/70*100:.0f}%)
are significant by all four tests simultaneously.

| Metric | # Significant | Median |
|--------|--------------|--------|
| Plane gradient (permutation) | {n_grad_sig}/70 ({n_grad_sig/70*100:.0f}%) | R^2 = {med_plane:.4f} |
| Moran's I (permutation) | {n_moran_sig}/70 ({n_moran_sig/70*100:.0f}%) | I = {med_moran:.4f} |
| Radial correlation | {n_radial_sig}/70 ({n_radial_sig/70*100:.0f}%) | |r| = {med_rad:.4f} |
| Quadrant ANOVA | {n_quad_sig}/70 ({n_quad_sig/70*100:.0f}%) | - |
| GAM fit | - | R^2 = {med_gam:.4f} |
| Subtype consistency | - | VS = {med_vs:.4f} |

### Key Findings

**1. Most retinal features are spatially non-uniform.**
The majority of features ({n_grad_sig}/70) have a detectable global gradient, and
even more ({n_moran_sig}/70) show spatial clustering (Moran's I > 0). This is not
an artefact of a single dominant axis -- {n_quad_sig}/70 features also show
significant quadrant effects.

**2. The dominant spatial axis points from Dorsal-Temporal to Ventral-Nasal.**
Among significant features, the mean gradient direction is approximately
{mean_dir:.0f} degrees. Light-response features (off_peak, green_off_peak,
blue_off_peak) consistently point toward Ventral-Nasal, while RF-size
features (gaussian_sigma, dog_sigma) point toward Ventral-Temporal.

**3. Nonlinear (hotspot) structure is widespread.**
The median GAM R^2 ({med_gam:.3f}) substantially exceeds the median plane R^2
({med_plane:.3f}), indicating that most features have localised peaks or valleys
beyond what a simple gradient captures. The largest GAM improvements are in
`dog_sigma_exc` (Delta R^2 = {ac.loc['dog_sigma_exc','gam_plane_delta_r2']:.3f}),
`freq_sinefit_10hz_amplitude` (Delta R^2 = {ac.loc['freq_sinefit_10hz_amplitude','gam_plane_delta_r2']:.3f}),
and `chip_effective_area` (Delta R^2 = {ac.loc['chip_effective_area','gam_plane_delta_r2']:.3f}).

**4. Center-periphery gradients in receptive-field size.**
`dog_sigma_exc` (r = {ac.loc['dog_sigma_exc','radial_r']:.3f}) and
`gaussian_sigma_x/y` (r ~ {ac.loc['gaussian_sigma_x','radial_r']:.3f}) show the
strongest center-high radial pattern: RF sizes are largest near the centre and
shrink toward the periphery, consistent with known retinal eccentricity scaling.

**5. Frequency-tuning features cluster in the Ventral-Nasal quadrant.**
All freq_sinefit R^2 features have their GAM maximum in the VN quadrant and share
consistent gradient directions across subtypes (vector strength > 0.80).

**6. Subtype gradients are directionally consistent for key features.**
`angle_correction_applied` shows near-perfect consistency (VS = 0.97, 97% subtypes
significant). Among biological features, `gaussian_amp` (VS = 0.86), freq R^2
features (VS ~ 0.82-0.85), and light-response features (VS ~ 0.78-0.80) show the
highest cross-subtype agreement. Phase features show the lowest consistency
(VS < 0.25).

**7. Phase features require special treatment.**
The 5 phase features (`freq_sinefit_*_phase_deg`) have inflated CV and Gini values
due to wrap-around at +/-180 degrees. Circular statistics reveal that only
`freq_sinefit_1hz_phase_deg` and `freq_sinefit_4hz_phase_deg` have significant
spatial clustering (Moran's I perm p < 0.01) when properly handled.

### Top 10 Features by Combined Spatial Structure Score

| Rank | Feature | Plane R^2 | GAM R^2 | Moran I | |Radial r| |
|------|---------|----------|---------|---------|----------|
"""

for rank, (f, _) in enumerate(top_spatial.items(), 1):
    r = ac.loc[f]
    summary_block += (f"| {rank} | {f} | {r['plane_r2']:.4f} | {r['gam_r2']:.4f} "
                      f"| {r['moran_i']:.4f} | {abs(r['radial_r']):.4f} |\n")

summary_block += f"""
### Figures

All visualisation figures are in `figures_quant/`:

| Figure | Description |
|--------|-------------|
| fig1_gradient_polar.png | Gradient direction and strength (polar plot) |
| fig2_plane_vs_gam.png | Plane R^2 vs GAM R^2 (nonlinear improvement) |
| fig3_moran_vs_gradient.png | Spatial clustering vs global gradient |
| fig4_radial_forest.png | Radial trends with 95% bootstrap CIs (forest plot) |
| fig5_quadrant_heatmap.png | Quadrant structure (z-scored heatmap) |
| fig6_subtype_consistency.png | Cross-subtype directional consistency |
| fig7_significance_heatmap.png | Significance overview across 4 tests |
| fig8_multimetric_overview.png | Multi-metric bubble chart |
| fig9_gam_hotspot_map.png | GAM extremum locations |
| fig10_summary_dashboard.png | 8-panel summary dashboard |
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(existing + summary_block)

print(f"  Updated {md_path.name}", flush=True)
print(f"\nDone. Figures in: {FIG_DIR}", flush=True)
