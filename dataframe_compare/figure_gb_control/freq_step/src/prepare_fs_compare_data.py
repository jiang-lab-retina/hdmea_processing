"""
Prepare Combined Freq-Step Compare Dataset
============================================
Loads before- and after-blocker frequency-step feature columns from all 3
experiments (_ptx_str, _ptx, _str), reshapes into long form with a
``condition`` column in {before, STR, PTX, STR_PTX}, computes amplitude
ratio features, drops rows without valid spatial coordinates, and saves a
single combined parquet for freq-step comparison analysis.

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

FREQ_STEP_FREQUENCIES = [0.5, 1, 2, 4, 10]

def _freq_to_col_str(freq):
    return str(freq).replace(".", "")

FS_SCALAR_FEATURES = []
for _f in FREQ_STEP_FREQUENCIES:
    _fs = _freq_to_col_str(_f)
    FS_SCALAR_FEATURES.extend([
        f"freq_step_{_fs}hz_amp",
        f"freq_step_{_fs}hz_phase",
        f"freq_step_{_fs}hz_r_squared",
        f"freq_step_{_fs}hz_offset",
        f"freq_step_{_fs}hz_std",
    ])

AMP_RATIO_FEATURES = [
    ("freq_step_1hz_amp_ratio_05hz",  "freq_step_1hz_amp",  "freq_step_05hz_amp"),
    ("freq_step_2hz_amp_ratio_05hz",  "freq_step_2hz_amp",  "freq_step_05hz_amp"),
    ("freq_step_4hz_amp_ratio_05hz",  "freq_step_4hz_amp",  "freq_step_05hz_amp"),
    ("freq_step_10hz_amp_ratio_05hz", "freq_step_10hz_amp", "freq_step_05hz_amp"),
]

EXP_TO_CONDITION = {
    "_str": "STR",
    "_ptx": "PTX",
    "_ptx_str": "STR_PTX",
}


def _compute_ratios(df):
    """Compute amplitude ratio features (numerator / denominator)."""
    for ratio_name, num_col, den_col in AMP_RATIO_FEATURES:
        if num_col in df.columns and den_col in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = df[num_col] / df[den_col]
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            df[ratio_name] = ratio
        else:
            df[ratio_name] = np.nan
    return df


def _extract_side(df, side, fs_features, condition, exp):
    """Extract before_ or after_ columns, strip prefix, add condition label."""
    prefix = f"{side}_"
    fs_cols = [
        f"{prefix}{feat}" for feat in fs_features
        if f"{prefix}{feat}" in df.columns
    ]
    keep = fs_cols + [X_COL, Y_COL, "group", "subtype", "before_dataset_id"]
    keep = [c for c in keep if c in df.columns]

    sub = df[keep].copy()
    sub["condition"] = condition
    sub["source_experiment"] = exp

    rename_map = {c: c.replace(prefix, "", 1) for c in fs_cols}
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

        before_sub = _extract_side(df, "before", FS_SCALAR_FEATURES, "before", exp)
        before_frames.append(before_sub)
        logger.info("  before: %d rows", len(before_sub))

        after_cond = EXP_TO_CONDITION[exp]
        after_sub = _extract_side(df, "after", FS_SCALAR_FEATURES, after_cond, exp)
        after_frames.append(after_sub)
        logger.info("  %s: %d rows", after_cond, len(after_sub))

    if not before_frames:
        logger.error("No data loaded")
        return

    combined = pd.concat(before_frames + after_frames, ignore_index=True)
    logger.info("Combined (all conditions): %s", combined.shape)

    combined = _compute_ratios(combined)
    logger.info("Added %d ratio features", len(AMP_RATIO_FEATURES))

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

    all_features = FS_SCALAR_FEATURES + [r[0] for r in AMP_RATIO_FEATURES]
    available = [f for f in all_features if f in combined.columns]
    logger.info("Available FS features: %d / %d", len(available), len(all_features))
    for f in available:
        nv = combined[f].notna().sum()
        logger.info("  %s: %d valid (%.1f%%)", f, nv, nv / len(combined) * 100)

    subtypes = sorted(
        s for s in combined["subtype"].dropna().unique() if s != ""
    )
    logger.info("Subtypes: %d", len(subtypes))

    out_path = OUTPUT_DIR / "combined_fs_compare.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved: %s  (%d rows)", out_path, len(combined))
    logger.info("Done.")


if __name__ == "__main__":
    main()
