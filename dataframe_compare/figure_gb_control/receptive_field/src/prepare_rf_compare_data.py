"""
Prepare Combined RF Compare Dataset
====================================
Loads before- and after-blocker receptive-field feature columns from all 3
experiments (_ptx_str, _ptx, _str), reshapes into long form with a
``condition`` column in {before, STR, PTX, STR_PTX}, drops rows without
valid spatial coordinates, and saves a single combined parquet for RF
comparison analysis.

before rows are pooled from all 3 experiments.  Each after condition
(STR / PTX / STR_PTX) comes from its single matching experiment only.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import (
    SOURCE_PARQUETS, OUTPUT_DIR,
    X_COL, Y_COL, COORD_LIMIT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RF_SCALAR_FEATURES = [
    "gaussian_sigma_x", "gaussian_sigma_y", "gaussian_amp", "gaussian_r2",
    "dog_sigma_exc", "dog_sigma_inh", "dog_amp_exc", "dog_amp_inh", "dog_r2",
    "lnl_a_norm", "lnl_bits_per_spike", "lnl_r_squared",
    "lnl_rectification_index", "lnl_nonlinearity_index", "lnl_threshold_g",
]

EXP_TO_CONDITION = {
    "_str": "STR",
    "_ptx": "PTX",
    "_ptx_str": "STR_PTX",
}


def _extract_side(df, side, rf_features, condition, exp):
    """Extract before_ or after_ columns, strip prefix, add condition label."""
    prefix = f"{side}_"
    rf_cols = [
        f"{prefix}{feat}" for feat in rf_features
        if f"{prefix}{feat}" in df.columns
    ]
    keep = rf_cols + [X_COL, Y_COL, "group", "subtype", "before_dataset_id"]
    keep = [c for c in keep if c in df.columns]

    sub = df[keep].copy()
    sub["condition"] = condition
    sub["source_experiment"] = exp

    rename_map = {c: c.replace(prefix, "", 1) for c in rf_cols}
    sub = sub.rename(columns=rename_map)
    return sub


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    before_frames = []
    after_frames = []

    for exp, parquet_path in SOURCE_PARQUETS.items():
        if not parquet_path.exists():
            logger.warning("  Missing: %s", parquet_path)
            continue

        logger.info("Loading %s from %s ...", exp, parquet_path.name)
        df = pd.read_parquet(parquet_path)
        logger.info("  Shape: %s", df.shape)

        before_sub = _extract_side(df, "before", RF_SCALAR_FEATURES, "before", exp)
        before_frames.append(before_sub)
        logger.info("  before: %d rows", len(before_sub))

        after_cond = EXP_TO_CONDITION[exp]
        after_sub = _extract_side(df, "after", RF_SCALAR_FEATURES, after_cond, exp)
        after_frames.append(after_sub)
        logger.info("  %s: %d rows", after_cond, len(after_sub))

    if not before_frames:
        logger.error("No data loaded")
        return

    combined = pd.concat(before_frames + after_frames, ignore_index=True)
    logger.info("Combined (all conditions): %s", combined.shape)

    n_before = len(combined)
    combined = combined.dropna(subset=[X_COL, Y_COL])
    mask = (
        (combined[X_COL].abs() < COORD_LIMIT)
        & (combined[Y_COL].abs() < COORD_LIMIT)
    )
    combined = combined[mask].copy()
    logger.info(
        "After coord filter: %d (dropped %d)", len(combined), n_before - len(combined)
    )

    for cond in ["before", "STR", "PTX", "STR_PTX"]:
        n = (combined["condition"] == cond).sum()
        logger.info("  %s: %d cells", cond, n)

    available = [f for f in RF_SCALAR_FEATURES if f in combined.columns]
    logger.info("Available RF features: %d / %d", len(available), len(RF_SCALAR_FEATURES))
    for f in available:
        nv = combined[f].notna().sum()
        logger.info("  %s: %d valid (%.1f%%)", f, nv, nv / len(combined) * 100)

    subtypes = sorted(
        s for s in combined["subtype"].dropna().unique() if s != ""
    )
    logger.info("Subtypes: %d", len(subtypes))

    out_path = OUTPUT_DIR / "combined_rf_compare.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved: %s  (%d rows)", out_path, len(combined))
    logger.info("Done.")


if __name__ == "__main__":
    main()
