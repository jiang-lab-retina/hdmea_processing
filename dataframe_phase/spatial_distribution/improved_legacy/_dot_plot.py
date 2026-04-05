import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "labeled_dataframe_improved_coords.parquet"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COORD_SCALE = 16
COORD_LIMIT = 100
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)

df = pd.read_parquet(INPUT)
df = df.dropna(subset=["improved_tx", "improved_ty"])
mask = (df["improved_tx"].abs() < COORD_LIMIT) & (df["improved_ty"].abs() < COORD_LIMIT)
df = df[mask]

x = df["improved_tx"].values * COORD_SCALE
y = df["improved_ty"].values * COORD_SCALE

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x, y, s=1, c="black", alpha=0.5, linewidths=0)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(XY_RANGE)
ax.set_ylim(XY_RANGE)
ax.set_xlabel("Temporal  <--  X (um)  -->  Nasal", fontsize=12)
ax.set_ylabel("Ventral  <--  Y (um)  -->  Dorsal", fontsize=12)
ax.set_title(f"All cells (improved coords, n={len(df)})", fontsize=13)
ax.tick_params(labelsize=11)
fig.tight_layout()

out = FIG_DIR / "dot_plot_improved.png"
fig.savefig(str(out), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
