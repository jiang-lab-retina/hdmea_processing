import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COORD_SCALE = 16
COORD_LIMIT = 100
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)

# Load legacy
df_leg = pd.read_parquet(SCRIPT_DIR.parent / "results" / "labeled_dataframe_with_legacy_coords_freq.parquet")
df_leg = df_leg.dropna(subset=["legacy_transformed_x", "legacy_transformed_y"])
mask_l = (df_leg["legacy_transformed_x"].abs() < COORD_LIMIT) & (df_leg["legacy_transformed_y"].abs() < COORD_LIMIT)
df_leg = df_leg[mask_l]

# Load improved
df_imp = pd.read_parquet(SCRIPT_DIR / "labeled_dataframe_improved_coords.parquet")
df_imp = df_imp.dropna(subset=["improved_tx", "improved_ty"])
mask_i = (df_imp["improved_tx"].abs() < COORD_LIMIT) & (df_imp["improved_ty"].abs() < COORD_LIMIT)
df_imp = df_imp[mask_i]

print(f"Legacy: {len(df_leg)} cells", flush=True)
print(f"  tx range: [{df_leg['legacy_transformed_x'].min():.1f}, {df_leg['legacy_transformed_x'].max():.1f}]")
print(f"  ty range: [{df_leg['legacy_transformed_y'].min():.1f}, {df_leg['legacy_transformed_y'].max():.1f}]")
print(f"  tx std: {df_leg['legacy_transformed_x'].std():.1f}")
print(f"  ty std: {df_leg['legacy_transformed_y'].std():.1f}")
print(f"Improved: {len(df_imp)} cells")
print(f"  tx range: [{df_imp['improved_tx'].min():.1f}, {df_imp['improved_tx'].max():.1f}]")
print(f"  ty range: [{df_imp['improved_ty'].min():.1f}, {df_imp['improved_ty'].max():.1f}]")
print(f"  tx std: {df_imp['improved_tx'].std():.1f}")
print(f"  ty std: {df_imp['improved_ty'].std():.1f}")

# Radius distribution
r_leg = np.sqrt(df_leg["legacy_transformed_x"]**2 + df_leg["legacy_transformed_y"]**2)
r_imp = np.sqrt(df_imp["improved_tx"]**2 + df_imp["improved_ty"]**2)
print(f"\nRadius distribution:")
print(f"  Legacy: median={r_leg.median():.1f}, mean={r_leg.mean():.1f}, max={r_leg.max():.1f}")
print(f"  Improved: median={r_imp.median():.1f}, mean={r_imp.mean():.1f}, max={r_imp.max():.1f}")

# How many cells are close to center (< 10 electrode units)?
print(f"\n  Legacy cells with r < 10: {(r_leg < 10).sum()}")
print(f"  Improved cells with r < 10: {(r_imp < 10).sum()}")
print(f"  Legacy cells with r < 20: {(r_leg < 20).sum()}")
print(f"  Improved cells with r < 20: {(r_imp < 20).sum()}")

# Side-by-side comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.scatter(df_leg["legacy_transformed_x"].values * COORD_SCALE,
            df_leg["legacy_transformed_y"].values * COORD_SCALE,
            s=1, c="black", alpha=0.5, linewidths=0)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlim(XY_RANGE); ax1.set_ylim(XY_RANGE)
ax1.set_title(f"Legacy coords (n={len(df_leg)})", fontsize=13)
ax1.set_xlabel("X (um)"); ax1.set_ylabel("Y (um)")

ax2.scatter(df_imp["improved_tx"].values * COORD_SCALE,
            df_imp["improved_ty"].values * COORD_SCALE,
            s=1, c="black", alpha=0.5, linewidths=0)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlim(XY_RANGE); ax2.set_ylim(XY_RANGE)
ax2.set_title(f"Improved coords (n={len(df_imp)})", fontsize=13)
ax2.set_xlabel("X (um)"); ax2.set_ylabel("Y (um)")

fig.tight_layout()
fig.savefig(str(FIG_DIR / "dot_plot_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved comparison plot", flush=True)
