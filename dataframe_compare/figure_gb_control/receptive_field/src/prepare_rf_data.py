"""
Prepare Combined RF Control Dataset
====================================
Loads before-blocker receptive-field feature columns from all 3 experiments
(_ptx_str, _ptx, _str), strips the ``before_`` prefix, drops rows without
valid spatial coordinates, and saves a single combined parquet for RF
spatial analysis.
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for exp, parquet_path in SOURCE_PARQUETS.items():
        if not parquet_path.exists():
            logger.warning(f"  Missing: {parquet_path}")
            continue

        logger.info(f"Loading {exp} from {parquet_path.name} ...")
        df = pd.read_parquet(parquet_path)
        logger.info(f"  Shape: {df.shape}")

        before_cols = [
            f"before_{feat}" for feat in RF_SCALAR_FEATURES
            if f"before_{feat}" in df.columns
        ]
        keep_cols = before_cols + [X_COL, Y_COL, "group", "subtype", "before_dataset_id"]
        keep_cols = [c for c in keep_cols if c in df.columns]

        sub = df[keep_cols].copy()
        sub["source_experiment"] = exp

        rename_map = {c: c.replace("before_", "", 1) for c in before_cols}
        sub = sub.rename(columns=rename_map)

        frames.append(sub)
        logger.info(f"  Kept {len(before_cols)} RF columns, {len(sub)} rows")

    if not frames:
        logger.error("No data loaded")
        return

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined: {combined.shape}")

    n_before = len(combined)
    combined = combined.dropna(subset=[X_COL, Y_COL])
    mask = (combined[X_COL].abs() < COORD_LIMIT) & (combined[Y_COL].abs() < COORD_LIMIT)
    combined = combined[mask].copy()
    logger.info(f"After coord filter: {len(combined)} (dropped {n_before - len(combined)})")

    available = [f for f in RF_SCALAR_FEATURES if f in combined.columns]
    logger.info(f"Available RF features: {len(available)} / {len(RF_SCALAR_FEATURES)}")
    for f in available:
        nv = combined[f].notna().sum()
        logger.info(f"  {f}: {nv} valid ({nv / len(combined) * 100:.1f}%)")

    subtypes = sorted(
        s for s in combined["subtype"].dropna().unique() if s != ""
    )
    logger.info(f"Subtypes: {len(subtypes)}")
    for s in subtypes:
        logger.info(f"  {s}: {(combined['subtype'] == s).sum()} cells")

    out_path = OUTPUT_DIR / "combined_rf_control.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({len(combined)} rows)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
