"""
Pipeline Compare Update: Intensity-specific green-blue features.

Loads the existing compared_dataframe.parquet, computes:
  1. High-intensity trace column (green_blue_3s_3i_3x) from trials 6-8
  2. Intensity-specific GB features (_low, _mid, _high) from trial groups
  3. Saves as compared_dataframe_v2.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dataframe_phase" / "extract_feature"))

from extract_feature_gb import compute_mean_trace, extract_gb_features_from_trace

# =============================================================================
# Configuration
# =============================================================================

from compare_config import OUTPUT_DIR as _OUTPUT_DIR

INPUT_PATH = _OUTPUT_DIR / "compared_dataframe.parquet"
OUTPUT_PATH = _OUTPUT_DIR / "compared_dataframe_v2.parquet"

# Source columns (9 trials = 3 intensities x 3 repeats)
SOURCE_COLUMN = "green_blue_3s_3i_3x_64_128_255"

# Intensity groups: trial index ranges
INTENSITY_LEVELS = {
    "low":  (0, 3),   # trials 0-2, intensity 64
    "mid":  (3, 6),   # trials 3-5, intensity 128
    "high": (6, 9),   # trials 6-8, intensity 255
}

# Features to keep (drop time-to-peak columns)
KEEP_FEATURES = [
    "green_on_peak_extreme",
    "blue_on_peak_extreme",
    "green_off_peak_extreme",
    "blue_off_peak_extreme",
    "gb_base_mean",
    "gb_base_std",
    "green_blue_on_ratio",
    "green_blue_off_ratio",
]


# =============================================================================
# Processing
# =============================================================================

def process_prefix(df, prefix):
    """
    Process one prefix (before_ or after_).

    Adds:
      - {prefix}green_blue_3s_3i_3x (high-intensity traces, trials 6-8)
      - {prefix}{feature}_{level} for each intensity level
    """
    src_col = f"{prefix}{SOURCE_COLUMN}"
    trace_col = f"{prefix}green_blue_3s_3i_3x"

    if src_col not in df.columns:
        print(f"  SKIPPED: column {src_col} not found")
        return df

    # Initialize new columns
    df[trace_col] = None
    for level in INTENSITY_LEVELS:
        for feat in KEEP_FEATURES:
            df[f"{prefix}{feat}_{level}"] = np.nan

    valid_count = 0
    for idx in tqdm(df.index, desc=f"  {prefix}GB intensity features"):
        trials_data = df.at[idx, src_col]

        # Skip invalid data
        if trials_data is None:
            continue
        if isinstance(trials_data, float) and np.isnan(trials_data):
            continue

        try:
            trials_list = list(trials_data)
        except (TypeError, ValueError):
            continue

        if len(trials_list) < 9:
            continue

        # 1. Store high-intensity traces (trials 6-8)
        high_trials = [np.array(t) for t in trials_list[6:9]]
        df.at[idx, trace_col] = high_trials

        # 2. Compute intensity-specific features
        for level, (start, end) in INTENSITY_LEVELS.items():
            subset = trials_list[start:end]
            mean_trace = compute_mean_trace(subset)
            if mean_trace is None:
                continue

            try:
                features = extract_gb_features_from_trace(mean_trace)
                for feat in KEEP_FEATURES:
                    df.at[idx, f"{prefix}{feat}_{level}"] = features[feat]
            except Exception:
                continue

        valid_count += 1

    print(f"  Processed {valid_count} / {len(df)} units")
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Pipeline Compare Update: Intensity-Specific GB Features")
    print("=" * 70)

    print(f"\nLoading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Shape: {df.shape}")

    original_cols = set(df.columns)

    print("\nProcessing BEFORE columns...")
    df = process_prefix(df, "before_")

    print("\nProcessing AFTER columns...")
    df = process_prefix(df, "after_")

    new_cols = sorted(set(df.columns) - original_cols)
    print(f"\nNew columns added: {len(new_cols)}")
    for c in new_cols:
        print(f"  {c}")

    print(f"\nSaving: {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}")
    df.to_parquet(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
