import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pygam import LinearGAM, te

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COORD_SCALE = 16
COORD_LIMIT = 100
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)
GRIDSIZE = 40
MINCNT = 2
CMAP = "coolwarm"
N_SPLINES = 30

# Load data
df = pd.read_parquet(SCRIPT_DIR / "labeled_dataframe_improved_coords.parquet")
df = df.dropna(subset=["improved_tx", "improved_ty", "green_blue_on_ratio", "green_on_peak_extreme"])
df = df[(df["improved_tx"].abs() < COORD_LIMIT) & (df["improved_ty"].abs() < COORD_LIMIT)]

# ON cells only
on = df[df["green_on_peak_extreme"] > 0].copy()
print(f"ON cells: {len(on)}")

x = on["improved_tx"].values * COORD_SCALE
y = on["improved_ty"].values * COORD_SCALE
c = on["green_blue_on_ratio"].values

c_mean = float(np.mean(c))
vmin = c_mean - 0.5 * abs(c_mean)
vmax = c_mean + 0.5 * abs(c_mean)

fig, (ax_raw, ax_gam) = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(right=0.90, wspace=0.15)

# Raw hexbin
hb = ax_raw.hexbin(x, y, C=c, reduce_C_function=np.mean, gridsize=GRIDSIZE,
                   extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
                   mincnt=MINCNT, cmap=CMAP, vmin=vmin, vmax=vmax)
ax_raw.set_aspect("equal", adjustable="box")
ax_raw.set_xlim(XY_RANGE); ax_raw.set_ylim(XY_RANGE)
ax_raw.set_title("Raw mean", fontsize=11)
ax_raw.set_xlabel("T <-- X (um) --> N", fontsize=10)
ax_raw.set_ylabel("V <-- Y (um) --> D", fontsize=10)

# GAM hexbin
X_train = np.column_stack([x, y])
gam = LinearGAM(te(0, 1, n_splines=[N_SPLINES, N_SPLINES]))
with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    gam = gam.gridsearch(X_train, c)

hb_gam = ax_gam.hexbin(x, y, gridsize=GRIDSIZE,
                        extent=(XY_RANGE[0], XY_RANGE[1], XY_RANGE[0], XY_RANGE[1]),
                        mincnt=MINCNT, cmap=CMAP)
offsets = hb_gam.get_offsets()
z_pred = gam.predict(offsets)
hb_gam.set_array(z_pred)
hb_gam.set_clim(vmin=vmin, vmax=vmax)
ax_gam.set_aspect("equal", adjustable="box")
ax_gam.set_xlim(XY_RANGE); ax_gam.set_ylim(XY_RANGE)
ax_gam.set_title("GAM smoothed", fontsize=11)
ax_gam.set_xlabel("T <-- X (um) --> N", fontsize=10)
ax_gam.set_ylabel("V <-- Y (um) --> D", fontsize=10)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_raw, ax_gam], shrink=0.75, pad=0.02)
cbar.set_label("green_blue_on_ratio", fontsize=11)

fig.suptitle(f"green_blue_on_ratio  (ON cells, n={len(on)})", fontsize=13)
out = FIG_DIR / "Hexbin_green_blue_on_ratio_ON_cells.png"
fig.savefig(str(out), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
