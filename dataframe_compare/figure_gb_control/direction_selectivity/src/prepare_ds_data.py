"""
Prepare Combined DS Control Dataset
====================================
Loads before-blocker direction-selectivity feature columns from all 3
experiments (_ptx_str, _ptx, _str), strips the ``before_`` prefix,
computes amplitude ratio features from freq-step data, drops rows
without valid spatial coordinates, and saves a single combined parquet
for DS spatial analysis.
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

DS_SCALAR_FEATURES = [
    "dsi", "osi", "ds_p_value", "os_p_value",
]

AMP_SOURCE_FEATURES = [
    "freq_step_10hz_amp",
    "freq_step_4hz_amp",
    "freq_step_05hz_amp",
]

AMP_RATIO_DEFS = [
    ("freq_step_10hz_amp_ratio_05hz", "freq_step_10hz_amp", "freq_step_05hz_amp"),
    ("freq_step_4hz_amp_ratio_05hz",  "freq_step_4hz_amp",  "freq_step_05hz_amp"),
]


def _compute_ratios(df):
    """Compute amplitude ratio features (numerator / denominator)."""
    for ratio_name, num_col, den_col in AMP_RATIO_DEFS:
        if num_col in df.columns and den_col in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = df[num_col] / df[den_col]
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            df[ratio_name] = ratio
        else:
            df[ratio_name] = np.nan
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_load_features = DS_SCALAR_FEATURES + AMP_SOURCE_FEATURES

    frames = []
    for exp, parquet_path in SOURCE_PARQUETS.items():
        if not parquet_path.exists():
            logger.warning("  Missing: %s", parquet_path)
            continue

        logger.info("Loading %s from %s ...", exp, parquet_path.name)
        df = pd.read_parquet(parquet_path)
        logger.info("  Shape: %s", df.shape)

        before_cols = [
            f"before_{feat}" for feat in all_load_features
            if f"before_{feat}" in df.columns
        ]
        keep_cols = before_cols + [X_COL, Y_COL, "group", "subtype", "before_dataset_id"]
        keep_cols = [c for c in keep_cols if c in df.columns]

        sub = df[keep_cols].copy()
        sub["source_experiment"] = exp

        rename_map = {c: c.replace("before_", "", 1) for c in before_cols}
        sub = sub.rename(columns=rename_map)

        frames.append(sub)
        logger.info("  Kept %d feature columns, %d rows", len(before_cols), len(sub))

    if not frames:
        logger.error("No data loaded")
        return

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined: %s", combined.shape)

    combined = _compute_ratios(combined)
    logger.info("Added %d ratio features", len(AMP_RATIO_DEFS))

    n_before = len(combined)
    combined = combined.dropna(subset=[X_COL, Y_COL])
    mask = (combined[X_COL].abs() < COORD_LIMIT) & (combined[Y_COL].abs() < COORD_LIMIT)
    combined = combined[mask].copy()
    logger.info("After coord filter: %d (dropped %d)", len(combined), n_before - len(combined))

    all_features = DS_SCALAR_FEATURES + [r[0] for r in AMP_RATIO_DEFS]
    available = [f for f in all_features if f in combined.columns]
    logger.info("Available DS features: %d / %d", len(available), len(all_features))
    for f in available:
        nv = combined[f].notna().sum()
        logger.info("  %s: %d valid (%.1f%%)", f, nv, nv / len(combined) * 100)

    subtypes = sorted(
        s for s in combined["subtype"].dropna().unique() if s != ""
    )
    logger.info("Subtypes: %d", len(subtypes))
    for s in subtypes:
        logger.info("  %s: %d cells", s, (combined["subtype"] == s).sum())

    out_path = OUTPUT_DIR / "combined_ds_control.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved: %s  (%d rows)", out_path, len(combined))
    logger.info("Done.")


if __name__ == "__main__":
    main()
