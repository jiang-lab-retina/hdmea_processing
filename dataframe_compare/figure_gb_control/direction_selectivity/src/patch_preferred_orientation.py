"""
Patch: Add preferred_orientation to existing source parquets
============================================================
Computes preferred_orientation from the corrected direction response
columns already present in each compared_dataframe parquet. Only the
OSI vector-sum angle is needed (no permutation testing), so this runs
in seconds.

Adds before_preferred_orientation and after_preferred_orientation
columns, then overwrites each parquet in place.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GB_SPATIAL_DIR = SCRIPT_DIR.parents[2] / "gb_spatial_control"
sys.path.insert(0, str(GB_SPATIAL_DIR))

from config import SOURCE_PARQUETS

COMPARE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMPARE_DIR.parent / "dataframe_phase" / "extract_feature"))
from extract_feature_dsgc import (
    DIRECTION_ANGLES,
    CORRECTED_DIRECTION_COLUMNS,
    calculate_orientation_index,
    get_total_firing_rate_per_trial,
)

DIRECTIONS = np.array(DIRECTION_ANGLES)


def compute_preferred_orientation_for_side(df, prefix):
    """Compute preferred_orientation for before_ or after_ columns."""
    dir_cols = [f"{prefix}{c}" for c in CORRECTED_DIRECTION_COLUMNS]
    corr_col = f"{prefix}angle_correction_applied"
    out_col = f"{prefix}preferred_orientation"

    has_all = all(c in df.columns for c in dir_cols)
    has_corr = corr_col in df.columns
    if not has_all or not has_corr:
        print(f"  {prefix}: missing columns, skipping")
        df[out_col] = np.nan
        return df

    orientations = []
    for idx in df.index:
        row = df.loc[idx]

        if pd.isna(row.get(corr_col)):
            orientations.append(np.nan)
            continue

        all_totals = []
        valid = True
        for dcol in dir_cols:
            trial_traces = row.get(dcol)
            if trial_traces is None:
                valid = False
                break
            if isinstance(trial_traces, list):
                trial_traces = np.array(trial_traces, dtype=object)
            if len(trial_traces) == 0:
                valid = False
                break
            totals = get_total_firing_rate_per_trial(trial_traces)
            all_totals.append(totals)

        if not valid or len(all_totals) != len(DIRECTIONS):
            orientations.append(np.nan)
            continue

        all_totals = np.array(all_totals)
        mean_per_dir = np.mean(all_totals, axis=1)
        _, _, ori_angle = calculate_orientation_index(DIRECTIONS, mean_per_dir)

        if ori_angle is not None:
            orientations.append((ori_angle / 2) % 180)
        else:
            orientations.append(np.nan)

    df[out_col] = orientations
    n_valid = sum(1 for v in orientations if not (isinstance(v, float) and np.isnan(v)))
    print(f"  {prefix}preferred_orientation: {n_valid} valid / {len(df)} total")
    return df


def main():
    for exp, path in SOURCE_PARQUETS.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue

        print(f"\nProcessing {exp}: {path.name}")
        df = pd.read_parquet(path)
        print(f"  Shape: {df.shape}")

        already_has = (
            f"before_preferred_orientation" in df.columns
            and f"after_preferred_orientation" in df.columns
        )
        if already_has:
            bv = df["before_preferred_orientation"].notna().sum()
            av = df["after_preferred_orientation"].notna().sum()
            print(f"  Already has preferred_orientation (before={bv}, after={av}), re-computing...")

        df = compute_preferred_orientation_for_side(df, "before_")
        df = compute_preferred_orientation_for_side(df, "after_")

        df.to_parquet(path, index=False)
        print(f"  Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
