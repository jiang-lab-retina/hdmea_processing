"""
Step 1: Prepare Combined GB Control Dataset
============================================
Loads before-blocker green-blue feature columns from all 3 experiments,
strips the ``before_`` prefix, drops rows without valid spatial coords,
and saves a single combined parquet.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SOURCE_PARQUETS, OUTPUT_DIR, ALL_GB_FEATURES,
    X_COL, Y_COL, COORD_LIMIT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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

        before_cols = [f"before_{feat}" for feat in ALL_GB_FEATURES if f"before_{feat}" in df.columns]
        keep_cols = before_cols + [X_COL, Y_COL, "group", "subtype", "before_dataset_id"]
        keep_cols = [c for c in keep_cols if c in df.columns]

        sub = df[keep_cols].copy()
        sub["source_experiment"] = exp

        rename_map = {c: c.replace("before_", "", 1) for c in before_cols}
        sub = sub.rename(columns=rename_map)

        frames.append(sub)
        logger.info(f"  Kept {len(before_cols)} GB columns, {len(sub)} rows")

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

    available = [f for f in ALL_GB_FEATURES if f in combined.columns]
    logger.info(f"Available GB features: {len(available)} / {len(ALL_GB_FEATURES)}")
    for f in available:
        nv = combined[f].notna().sum()
        logger.info(f"  {f}: {nv} valid ({nv/len(combined)*100:.1f}%)")

    groups = sorted(g for g in combined["group"].unique() if g != "")
    logger.info(f"Groups: {groups}")
    for g in groups:
        logger.info(f"  {g}: {(combined['group'] == g).sum()} cells")

    out_path = OUTPUT_DIR / "combined_gb_control.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({len(combined)} rows)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
