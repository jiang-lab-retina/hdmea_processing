"""
Radial Center Search & Radial Trend Analysis
=============================================
Uses saved hexbin data (raw_mean and gam_pred) to:
  1. For each feature, search for the radial center that maximises |Pearson r|
     between radius-from-center and the hexbin value.
  2. Report the optimal center, correlation strength, slope, and direction.
  3. Compare with origin-centred (0, 0) radial trend as baseline.
  4. Save all results as parquet, write detailed MD summary.

Input
-----
  results/hexbin_data_all_cells.parquet
  results/hexbin_data_per_subtype.parquet

Output
------
  results/radial_center_all_cells.parquet
  results/radial_center_per_subtype.parquet
  results/radial_center_summary.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import minimize
import warnings, time

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEX_ALL_PATH = RESULTS_DIR / "hexbin_data_all_cells.parquet"
HEX_SUB_PATH = RESULTS_DIR / "hexbin_data_per_subtype.parquet"

# Grid search parameters
COARSE_STEP = 200        # um -- coarse grid spacing
FINE_RADIUS = 300        # um -- refine within this radius of coarse best
FINE_STEP = 50           # um -- fine grid spacing
SEARCH_LIMIT = 1200      # um -- search centers within +/- this range
MIN_BINS = 15            # need at least this many bins

# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def radial_corr(cx, cy, bx, by, vals):
    """Pearson r between radius-from-(cx,cy) and vals."""
    r = np.sqrt((bx - cx)**2 + (by - cy)**2)
    if np.std(r) < 1e-12 or np.std(vals) < 1e-12:
        return 0.0, 1.0, 0.0
    try:
        rho, pval = pearsonr(r, vals)
    except Exception:
        return 0.0, 1.0, 0.0
    # slope via simple regression
    A = np.column_stack([r, np.ones_like(r)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
        slope = coeffs[0]
    except Exception:
        slope = 0.0
    return float(rho), float(pval), float(slope)


def search_radial_center(bx, by, vals):
    """
    Two-pass grid search + scipy refinement for center that maximises |r|.
    Returns dict with best center, r, p, slope, and origin baseline.
    """
    # 1) Origin baseline
    r0, p0, s0 = radial_corr(0, 0, bx, by, vals)

    # 2) Coarse grid
    xs = np.arange(-SEARCH_LIMIT, SEARCH_LIMIT + 1, COARSE_STEP)
    ys = np.arange(-SEARCH_LIMIT, SEARCH_LIMIT + 1, COARSE_STEP)
    best_abs_r, best_cx, best_cy = abs(r0), 0.0, 0.0
    for cx in xs:
        for cy in ys:
            rho, _, _ = radial_corr(cx, cy, bx, by, vals)
            if abs(rho) > best_abs_r:
                best_abs_r = abs(rho)
                best_cx, best_cy = cx, cy

    # 3) Fine grid around coarse best
    xs2 = np.arange(best_cx - FINE_RADIUS, best_cx + FINE_RADIUS + 1, FINE_STEP)
    ys2 = np.arange(best_cy - FINE_RADIUS, best_cy + FINE_RADIUS + 1, FINE_STEP)
    for cx in xs2:
        for cy in ys2:
            rho, _, _ = radial_corr(cx, cy, bx, by, vals)
            if abs(rho) > best_abs_r:
                best_abs_r = abs(rho)
                best_cx, best_cy = cx, cy

    # 4) Scipy refinement (minimize -|r|) with bounds
    MAX_CENTER = SEARCH_LIMIT * 1.5  # hard limit on center location

    def neg_abs_r(params):
        # penalise out-of-bounds
        if abs(params[0]) > MAX_CENTER or abs(params[1]) > MAX_CENTER:
            return 0.0
        rho, _, _ = radial_corr(params[0], params[1], bx, by, vals)
        return -abs(rho)

    try:
        res = minimize(neg_abs_r, [best_cx, best_cy], method="Nelder-Mead",
                       options={"xatol": 10, "fatol": 1e-6, "maxiter": 500})
        cx_cand, cy_cand = float(res.x[0]), float(res.x[1])
        if (abs(cx_cand) <= MAX_CENTER and abs(cy_cand) <= MAX_CENTER
                and (res.success or -res.fun > best_abs_r)):
            best_cx, best_cy = cx_cand, cy_cand
    except Exception:
        pass

    # Final stats at best center
    r_best, p_best, s_best = radial_corr(best_cx, best_cy, bx, by, vals)

    return {
        "origin_r": r0, "origin_p": p0, "origin_slope": s0,
        "best_center_x": best_cx, "best_center_y": best_cy,
        "best_r": r_best, "best_p": p_best, "best_slope": s_best,
        "abs_r_improvement": abs(r_best) - abs(r0),
    }


# ------------------------------------------------------------------
# Process all-cells
# ------------------------------------------------------------------
print("Loading hexbin data ...", flush=True)
df_all = pd.read_parquet(HEX_ALL_PATH)
features = sorted(df_all["feature"].unique())
print(f"  All-cells features: {len(features)}", flush=True)

print("\n=== All-cells radial center search ===", flush=True)
rows_all = []
t0 = time.time()
for fi, feat in enumerate(features):
    sub = df_all[df_all["feature"] == feat]
    if len(sub) < MIN_BINS:
        continue
    bx = sub["bin_x"].to_numpy()
    by = sub["bin_y"].to_numpy()

    for data_type in ["raw_mean", "gam_pred"]:
        vals = sub[data_type].to_numpy()
        mask = np.isfinite(vals)
        if mask.sum() < MIN_BINS:
            continue
        res = search_radial_center(bx[mask], by[mask], vals[mask])
        res["scope"] = "all_cells"
        res["feature"] = feat
        res["data_type"] = data_type
        res["n_bins"] = int(mask.sum())
        rows_all.append(res)

    if (fi + 1) % 10 == 0 or (fi + 1) == len(features):
        elapsed = time.time() - t0
        print(f"  [{fi+1}/{len(features)}] {feat}  ({elapsed:.0f}s)", flush=True)

df_results_all = pd.DataFrame(rows_all)
print(f"  All-cells results: {len(df_results_all)} rows", flush=True)


# ------------------------------------------------------------------
# Process per-subtype
# ------------------------------------------------------------------
print("\n=== Per-subtype radial center search ===", flush=True)
df_sub = pd.read_parquet(HEX_SUB_PATH)
subtypes = sorted(df_sub["scope"].unique())
sub_features = sorted(df_sub["feature"].unique())
print(f"  Subtypes: {len(subtypes)}, Features: {len(sub_features)}", flush=True)

rows_sub = []
t0 = time.time()
total = len(sub_features)
for fi, feat in enumerate(sub_features):
    feat_data = df_sub[df_sub["feature"] == feat]
    for stype in subtypes:
        sub = feat_data[feat_data["scope"] == stype]
        if len(sub) < MIN_BINS:
            continue
        bx = sub["bin_x"].to_numpy()
        by = sub["bin_y"].to_numpy()
        # Per-subtype only has raw_mean (gam_pred is NaN)
        vals = sub["raw_mean"].to_numpy()
        mask = np.isfinite(vals)
        if mask.sum() < MIN_BINS:
            continue
        res = search_radial_center(bx[mask], by[mask], vals[mask])
        res["scope"] = stype
        res["feature"] = feat
        res["data_type"] = "raw_mean"
        res["n_bins"] = int(mask.sum())
        rows_sub.append(res)

    if (fi + 1) % 10 == 0 or (fi + 1) == total:
        elapsed = time.time() - t0
        print(f"  [{fi+1}/{total}] {feat}  ({elapsed:.0f}s)", flush=True)

df_results_sub = pd.DataFrame(rows_sub)
print(f"  Per-subtype results: {len(df_results_sub)} rows", flush=True)


# ------------------------------------------------------------------
# Save parquets
# ------------------------------------------------------------------
print("\n=== Saving results ===", flush=True)
out_all = RESULTS_DIR / "radial_center_all_cells.parquet"
out_sub = RESULTS_DIR / "radial_center_per_subtype.parquet"
df_results_all.to_parquet(out_all, index=False)
df_results_sub.to_parquet(out_sub, index=False)
print(f"  {out_all.name}: {df_results_all.shape}", flush=True)
print(f"  {out_sub.name}: {df_results_sub.shape}", flush=True)


# ------------------------------------------------------------------
# MD summary
# ------------------------------------------------------------------
print("\n=== Writing summary ===", flush=True)

lines = []
lines.append("# Radial Center Analysis\n")
lines.append("## Method\n")
lines.append("For each feature (and each data type: raw hexbin mean, GAM prediction),")
lines.append("we search for the 2-D center point that maximises the absolute Pearson")
lines.append("correlation between the radial distance from that center and the bin value.\n")
lines.append("**Search procedure:**")
lines.append(f"1. Coarse grid: +/-{SEARCH_LIMIT} um, step {COARSE_STEP} um")
lines.append(f"2. Fine grid: +/-{FINE_RADIUS} um around coarse best, step {FINE_STEP} um")
lines.append("3. Nelder-Mead optimisation from the fine-grid best\n")
lines.append("A positive `best_r` means the feature increases with distance from the center")
lines.append("(periphery-high). A negative `best_r` means the feature decreases with")
lines.append("distance (center-high).\n")

# --- All cells: raw_mean ---
ac_raw = df_results_all[df_results_all["data_type"] == "raw_mean"].copy()
ac_gam = df_results_all[df_results_all["data_type"] == "gam_pred"].copy()

if len(ac_raw) > 0:
    lines.append("---\n## All Cells -- Raw Mean\n")
    lines.append("### Strongest radial trends (by |best_r|)\n")
    lines.append("| Feature | Best Cx (um) | Best Cy (um) | best_r | best_p | best_slope | origin_r | origin_p | Improvement |")
    lines.append("|---------|-------------|-------------|--------|--------|-----------|----------|----------|------------|")
    top = ac_raw.reindex(ac_raw["best_r"].abs().nlargest(20).index)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature']} | {row['best_center_x']:.0f} | {row['best_center_y']:.0f} "
            f"| {row['best_r']:.4f} | {row['best_p']:.2e} | {row['best_slope']:.6f} "
            f"| {row['origin_r']:.4f} | {row['origin_p']:.2e} | {row['abs_r_improvement']:.4f} |"
        )
    lines.append("")

    # Features where center search improved |r| most
    lines.append("### Largest improvement over origin center\n")
    lines.append("| Feature | Origin |r| | Best |r| | Improvement | Best Cx | Best Cy |")
    lines.append("|---------|----------|---------|------------|---------|---------|")
    top_imp = ac_raw.nlargest(15, "abs_r_improvement")
    for _, row in top_imp.iterrows():
        lines.append(
            f"| {row['feature']} | {abs(row['origin_r']):.4f} | {abs(row['best_r']):.4f} "
            f"| {row['abs_r_improvement']:.4f} | {row['best_center_x']:.0f} | {row['best_center_y']:.0f} |"
        )
    lines.append("")

    # Center clustering
    lines.append("### Radial center distribution (raw mean)\n")
    cx_vals = ac_raw["best_center_x"].to_numpy()
    cy_vals = ac_raw["best_center_y"].to_numpy()
    lines.append(f"- Center X: mean={np.mean(cx_vals):.0f}, median={np.median(cx_vals):.0f}, std={np.std(cx_vals):.0f} um")
    lines.append(f"- Center Y: mean={np.mean(cy_vals):.0f}, median={np.median(cy_vals):.0f}, std={np.std(cy_vals):.0f} um")
    # significant features only
    sig = ac_raw[ac_raw["best_p"] < 0.05]
    if len(sig) > 0:
        lines.append(f"- Among {len(sig)} features with p<0.05:")
        lines.append(f"  - Center X: mean={sig['best_center_x'].mean():.0f}, median={sig['best_center_x'].median():.0f} um")
        lines.append(f"  - Center Y: mean={sig['best_center_y'].mean():.0f}, median={sig['best_center_y'].median():.0f} um")
    lines.append("")

# --- All cells: GAM ---
if len(ac_gam) > 0:
    lines.append("---\n## All Cells -- GAM Smoothed\n")
    lines.append("### Strongest radial trends (by |best_r|)\n")
    lines.append("| Feature | Best Cx (um) | Best Cy (um) | best_r | best_p | best_slope | origin_r | origin_p | Improvement |")
    lines.append("|---------|-------------|-------------|--------|--------|-----------|----------|----------|------------|")
    top = ac_gam.reindex(ac_gam["best_r"].abs().nlargest(20).index)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature']} | {row['best_center_x']:.0f} | {row['best_center_y']:.0f} "
            f"| {row['best_r']:.4f} | {row['best_p']:.2e} | {row['best_slope']:.6f} "
            f"| {row['origin_r']:.4f} | {row['origin_p']:.2e} | {row['abs_r_improvement']:.4f} |"
        )
    lines.append("")

    lines.append("### Largest improvement over origin center (GAM)\n")
    lines.append("| Feature | Origin |r| | Best |r| | Improvement | Best Cx | Best Cy |")
    lines.append("|---------|----------|---------|------------|---------|---------|")
    top_imp = ac_gam.nlargest(15, "abs_r_improvement")
    for _, row in top_imp.iterrows():
        lines.append(
            f"| {row['feature']} | {abs(row['origin_r']):.4f} | {abs(row['best_r']):.4f} "
            f"| {row['abs_r_improvement']:.4f} | {row['best_center_x']:.0f} | {row['best_center_y']:.0f} |"
        )
    lines.append("")

    sig_gam = ac_gam[ac_gam["best_p"] < 0.05]
    if len(sig_gam) > 0:
        lines.append("### Radial center distribution (GAM, p<0.05)\n")
        lines.append(f"- {len(sig_gam)} features significant")
        lines.append(f"- Center X: mean={sig_gam['best_center_x'].mean():.0f}, median={sig_gam['best_center_x'].median():.0f} um")
        lines.append(f"- Center Y: mean={sig_gam['best_center_y'].mean():.0f}, median={sig_gam['best_center_y'].median():.0f} um")
    lines.append("")

# --- Per-subtype ---
if len(df_results_sub) > 0:
    lines.append("---\n## Per-Subtype Highlights\n")

    # Which subtypes have the most features with significant radial trends?
    sig_sub = df_results_sub[df_results_sub["best_p"] < 0.05]
    if len(sig_sub) > 0:
        counts = sig_sub.groupby("scope").size().sort_values(ascending=False)
        lines.append("### Subtypes with most significant radial features (p<0.05)\n")
        lines.append("| Subtype | # Significant Features | Avg |best_r| |")
        lines.append("|---------|----------------------|---------------|")
        for st, cnt in counts.head(15).items():
            avg_r = sig_sub[sig_sub["scope"] == st]["best_r"].abs().mean()
            lines.append(f"| {st} | {cnt} | {avg_r:.4f} |")
        lines.append("")

    # Strongest per-subtype radial trends overall
    lines.append("### Strongest per-subtype radial trends\n")
    lines.append("| Subtype | Feature | Best Cx | Best Cy | best_r | best_p | n_bins |")
    lines.append("|---------|---------|---------|---------|--------|--------|--------|")
    top_sub = df_results_sub.reindex(df_results_sub["best_r"].abs().nlargest(25).index)
    for _, row in top_sub.iterrows():
        lines.append(
            f"| {row['scope']} | {row['feature']} | {row['best_center_x']:.0f} "
            f"| {row['best_center_y']:.0f} | {row['best_r']:.4f} | {row['best_p']:.2e} | {row['n_bins']} |"
        )
    lines.append("")

    # Per-subtype center clustering for significant features
    if len(sig_sub) > 0:
        lines.append("### Radial center distribution per subtype (significant features)\n")
        lines.append("| Subtype | n_sig | Median Cx | Median Cy | Std Cx | Std Cy |")
        lines.append("|---------|-------|-----------|-----------|--------|--------|")
        for st in counts.head(15).index:
            st_data = sig_sub[sig_sub["scope"] == st]
            lines.append(
                f"| {st} | {len(st_data)} | {st_data['best_center_x'].median():.0f} "
                f"| {st_data['best_center_y'].median():.0f} | {st_data['best_center_x'].std():.0f} "
                f"| {st_data['best_center_y'].std():.0f} |"
            )
        lines.append("")

# --- Full table: all-cells raw ---
if len(ac_raw) > 0:
    lines.append("---\n## Full Table: All Cells (Raw Mean)\n")
    lines.append("| Feature | n_bins | origin_r | origin_p | best_cx | best_cy | best_r | best_p | best_slope | improvement |")
    lines.append("|---------|--------|----------|----------|---------|---------|--------|--------|-----------|------------|")
    for _, row in ac_raw.sort_values("feature").iterrows():
        lines.append(
            f"| {row['feature']} | {row['n_bins']} "
            f"| {row['origin_r']:.4f} | {row['origin_p']:.2e} "
            f"| {row['best_center_x']:.0f} | {row['best_center_y']:.0f} "
            f"| {row['best_r']:.4f} | {row['best_p']:.2e} "
            f"| {row['best_slope']:.6f} | {row['abs_r_improvement']:.4f} |"
        )
    lines.append("")

# --- Full table: all-cells GAM ---
if len(ac_gam) > 0:
    lines.append("---\n## Full Table: All Cells (GAM Smoothed)\n")
    lines.append("| Feature | n_bins | origin_r | origin_p | best_cx | best_cy | best_r | best_p | best_slope | improvement |")
    lines.append("|---------|--------|----------|----------|---------|---------|--------|--------|-----------|------------|")
    for _, row in ac_gam.sort_values("feature").iterrows():
        lines.append(
            f"| {row['feature']} | {row['n_bins']} "
            f"| {row['origin_r']:.4f} | {row['origin_p']:.2e} "
            f"| {row['best_center_x']:.0f} | {row['best_center_y']:.0f} "
            f"| {row['best_r']:.4f} | {row['best_p']:.2e} "
            f"| {row['best_slope']:.6f} | {row['abs_r_improvement']:.4f} |"
        )
    lines.append("")

lines.append("---\n## Output Files\n")
lines.append(f"- `{out_all.name}` ({df_results_all.shape[0]} rows)")
lines.append(f"- `{out_sub.name}` ({df_results_sub.shape[0]} rows)")
lines.append(f"- `radial_center_summary.md` (this file)")

md_path = RESULTS_DIR / "radial_center_summary.md"
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  {md_path.name}", flush=True)
print("\nDone.", flush=True)
