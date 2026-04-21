"""
RF Spatial Hexbin + GAM Heatmaps
================================
Creates hexbin heatmaps (raw + GAM smoothed) for each scalar receptive-field
feature, at two scopes:

  1. **All cells** -- every unit pooled across experiments.
  2. **Per subtype** -- each ``subtype`` (cluster) separately.

Reuses the hexbin / GAM helpers from ``gb_spatial_control/spatial_plots.py``
so that visual style matches the existing GB spatial analysis figures.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import (
    OUTPUT_DIR, FIG_DIR_BASE,
    X_COL, Y_COL, COORD_SCALE, XY_RANGE,
    GRIDSIZE_ALL, GRIDSIZE_GRP, MINCNT_ALL, MINCNT_GRP,
    N_SPLINES_ALL, N_SPLINES_GRP,
)
from spatial_plots import (
    plot_heatmap, extract_hexbin_data, compute_metrics, _fit_gam,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIG_DIR = FIG_DIR_BASE / "receptive_field"
FIG_ALL_DIR = FIG_DIR / "all_cells"
FIG_SUB_DIR = FIG_DIR / "per_subtype"

for d in [OUTPUT_DIR, FIG_ALL_DIR, FIG_SUB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RF_SCALAR_FEATURES = [
    "gaussian_sigma_x", "gaussian_sigma_y", "gaussian_amp", "gaussian_r2",
    "dog_sigma_exc", "dog_sigma_inh", "dog_amp_exc", "dog_amp_inh", "dog_r2",
    "lnl_a_norm", "lnl_bits_per_spike", "lnl_r_squared",
    "lnl_rectification_index", "lnl_nonlinearity_index", "lnl_threshold_g",
]


def _process_scope(df, features, gridsize, mincnt, n_splines,
                   fig_dir, scope_label, title_prefix=""):
    """Run hexbin + GAM for one scope (all_cells or a single subtype)."""
    hexbin_rows = []
    metrics_rows = []

    for fi, feat in enumerate(features):
        cols = [X_COL, Y_COL, feat]
        data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 10:
            continue

        x = data[X_COL].to_numpy() * COORD_SCALE
        y = data[Y_COL].to_numpy() * COORD_SCALE
        c = data[feat].to_numpy()

        centers, raw_means, counts = extract_hexbin_data(
            x, y, c, gridsize, mincnt,
        )
        gam = _fit_gam(x, y, c, n_splines)
        gam_preds = (
            gam.predict(centers)
            if gam is not None and len(centers) > 0
            else None
        )

        for bi in range(len(centers)):
            hexbin_rows.append({
                "scope": scope_label,
                "feature": feat,
                "bin_x": centers[bi, 0],
                "bin_y": centers[bi, 1],
                "count": int(counts[bi]),
                "raw_mean": float(raw_means[bi]),
                "gam_pred": (
                    float(gam_preds[bi]) if gam_preds is not None else np.nan
                ),
            })

        m = compute_metrics(x, y, c, centers, raw_means)
        m["scope"] = scope_label
        m["feature"] = feat
        metrics_rows.append(m)

        if scope_label == "all_cells":
            fname = f"Hexbin_{feat}.png"
        else:
            fname = f"Hexbin_{scope_label}_{feat}.png"

        save_path = fig_dir / fname
        plot_heatmap(
            x, y, c, feat, gridsize, mincnt, n_splines,
            save_path, title_prefix=title_prefix,
        )

    return hexbin_rows, metrics_rows


def main():
    input_path = OUTPUT_DIR / "combined_rf_control.parquet"
    if not input_path.exists():
        logger.error(
            f"Input not found: {input_path}\n"
            "Run prepare_rf_data.py first."
        )
        return

    logger.info(f"Loading {input_path.name} ...")
    df = pd.read_parquet(input_path)
    logger.info(f"  Shape: {df.shape}")

    available_features = [f for f in RF_SCALAR_FEATURES if f in df.columns]
    logger.info(f"  Available RF features: {len(available_features)}")
    if not available_features:
        logger.error("No RF features found in the parquet -- nothing to plot.")
        return

    hexbin_rows = []
    metrics_rows = []

    # =================================================================
    # Phase 1: All-cells heatmaps
    # =================================================================
    logger.info("\n=== Phase 1: All-cells heatmaps ===")
    h, m = _process_scope(
        df, available_features,
        GRIDSIZE_ALL, MINCNT_ALL, N_SPLINES_ALL,
        FIG_ALL_DIR, "all_cells",
    )
    hexbin_rows.extend(h)
    metrics_rows.extend(m)
    logger.info(f"  Generated {len(m)} feature maps")

    # =================================================================
    # Phase 2: Per-subtype (cluster) heatmaps
    # =================================================================
    logger.info("\n=== Phase 2: Per-subtype heatmaps ===")
    subtypes = sorted(
        s for s in df["subtype"].dropna().unique() if s != ""
    )
    logger.info(f"  Subtypes: {len(subtypes)}")

    for si, subtype in enumerate(subtypes):
        sub_df = df[df["subtype"] == subtype]
        n_cells = len(sub_df)
        if n_cells < 5:
            logger.info(f"  [{si+1}/{len(subtypes)}] {subtype} -- skipped (n={n_cells})")
            continue

        h, m = _process_scope(
            sub_df, available_features,
            GRIDSIZE_GRP, MINCNT_GRP, N_SPLINES_GRP,
            FIG_SUB_DIR, subtype, title_prefix=f"[{subtype}] ",
        )
        hexbin_rows.extend(h)
        metrics_rows.extend(m)

        logger.info(f"  [{si+1}/{len(subtypes)}] {subtype}: {n_cells} cells, {len(m)} features")

    # =================================================================
    # Phase 3: Save result parquets
    # =================================================================
    logger.info("\n=== Phase 3: Saving results ===")

    all_cells_rows = [r for r in hexbin_rows if r["scope"] == "all_cells"]
    per_sub_rows = [r for r in hexbin_rows if r["scope"] != "all_cells"]

    if all_cells_rows:
        out = OUTPUT_DIR / "rf_hexbin_all_cells.parquet"
        pd.DataFrame(all_cells_rows).to_parquet(out, index=False)
        logger.info(f"  {out.name}: {len(all_cells_rows)} rows")

    if per_sub_rows:
        out = OUTPUT_DIR / "rf_hexbin_per_subtype.parquet"
        pd.DataFrame(per_sub_rows).to_parquet(out, index=False)
        logger.info(f"  {out.name}: {len(per_sub_rows)} rows")

    if metrics_rows:
        out = OUTPUT_DIR / "rf_spatial_metrics.parquet"
        pd.DataFrame(metrics_rows).to_parquet(out, index=False)
        logger.info(f"  {out.name}: {len(metrics_rows)} rows")

    logger.info("Done.")


if __name__ == "__main__":
    main()
