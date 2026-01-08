"""
Combined Feature Extraction Pipeline.

Runs all feature extraction steps in sequence:
1. Step/ipRGC QI: step_up_QI, iprgc_2hz_QI, iprgc_20hz_QI, good_count, good_rgc_count
2. DSGC: DSI, OSI, preferred_direction, ds_p_value, os_p_value, corrected direction columns
3. Step Response: ON/OFF peak features and ratios
4. Green-Blue: Chromatic response features
5. Frequency: Sine wave fit features for each frequency step
6. Frequency Sectioned Traces: 5 columns with sectioned freq traces (0.5, 1, 2, 4, 10 Hz)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Step/ipRGC QI imports
from extract_feature_step_iprgc import (
    compute_step_up_qi,
    compute_iprgc_qi,
    add_good_cell_counts,
)

# DSGC imports
from extract_feature_dsgc import (
    remap_direction_columns,
    process_unit,
    DIRECTION_ANGLES,
    CORRECTED_DIRECTION_COLUMNS,
    N_PERMUTATIONS,
    N_TRIALS,
)

# Step response imports
from extract_feature_step import extract_step_features

# Green-blue imports
from extract_feature_gb import extract_gb_features

# Frequency step imports
from extract_feature_freq import extract_freq_step_features, extract_freq_sectioned_traces


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PARQUET = Path(__file__).parent.parent / "load_feature" / "firing_rate_by_movie_loaded20260104.parquet"
OUTPUT_PARQUET = Path(__file__).parent / "firing_rate_with_all_features_loaded_extracted20260104.parquet"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the combined feature extraction pipeline."""
    print("=" * 80)
    print("COMBINED FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    
    # Load input data
    print(f"\nLoading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} units with {len(df.columns)} columns")
    
    # -------------------------------------------------------------------------
    # Step 1: Step/ipRGC Quality Index
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 1: Step/ipRGC Quality Index")
    print("-" * 80)
    
    print("\nComputing step_up quality index...")
    step_up_qi = compute_step_up_qi(
        df,
        movie_col="step_up_5s_5i_b0_3x",
        apply_filter=True,
        cutoff_freq=10.0,
        filter_order=5,
        sampling_rate=60.0,
    )
    df["step_up_QI"] = step_up_qi
    print(f"  step_up_QI: {step_up_qi.notna().sum()} valid values")
    
    print("\nComputing ipRGC quality indices...")
    iprgc_2hz_qi, iprgc_20hz_qi = compute_iprgc_qi(
        df,
        movie_col="iprgc_test",
        filter_order=5,
        sampling_rate=60.0,
    )
    df["iprgc_2hz_QI"] = iprgc_2hz_qi
    df["iprgc_20hz_QI"] = iprgc_20hz_qi
    print(f"  iprgc_2hz_QI: {iprgc_2hz_qi.notna().sum()} valid values")
    print(f"  iprgc_20hz_QI: {iprgc_20hz_qi.notna().sum()} valid values")
    
    print("\nComputing good cell counts per recording...")
    df = add_good_cell_counts(df)
    print(f"  Added good_count and good_rgc_count columns")
    
    # -------------------------------------------------------------------------
    # Step 2: DSGC Features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 2: DSGC Features (Direction/Orientation Selectivity)")
    print("-" * 80)
    
    # Create corrected direction columns
    print("\nCreating corrected direction columns...")
    corrected_results = []
    for idx in tqdm(df.index, desc="Remapping directions"):
        row = df.loc[idx]
        corrected_row = remap_direction_columns(row)
        corrected_results.append(corrected_row)
    
    # Add corrected columns to DataFrame
    corrected_df = pd.DataFrame(corrected_results, index=df.index)
    for col in CORRECTED_DIRECTION_COLUMNS:
        df[col] = corrected_df[col]
    
    valid_corrections = df["angle_correction_applied"].notna().sum()
    print(f"Valid corrections applied: {valid_corrections} / {len(df)}")
    
    # Compute DSI/OSI for each unit
    print(f"\nComputing DSI/OSI with {N_PERMUTATIONS} permutations per unit...")
    directions = np.array(DIRECTION_ANGLES)
    
    results = []
    for idx in tqdm(df.index, desc="Computing DSI/OSI"):
        row = df.loc[idx]
        
        # Skip if angle_correction_applied is NaN
        if pd.isna(row.get("angle_correction_applied")):
            results.append({
                "dsi": np.nan,
                "osi": np.nan,
                "preferred_direction": np.nan,
                "ds_p_value": np.nan,
                "os_p_value": np.nan,
            })
            continue
        
        result = process_unit(
            row,
            directions,
            CORRECTED_DIRECTION_COLUMNS,
            N_PERMUTATIONS,
            N_TRIALS,
        )
        results.append(result)
    
    # Add DSGC results to DataFrame
    results_df = pd.DataFrame(results, index=df.index)
    for col in ["dsi", "osi", "preferred_direction", "ds_p_value", "os_p_value"]:
        df[col] = results_df[col]
    
    valid_dsi = df["dsi"].notna().sum()
    print(f"  Valid DSI/OSI values: {valid_dsi} / {len(df)}")
    
    # -------------------------------------------------------------------------
    # Step 3: Step Response Features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 3: Step Response Features")
    print("-" * 80)
    
    df = extract_step_features(df, skip_filtering=True)
    
    # -------------------------------------------------------------------------
    # Step 4: Green-Blue Features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 4: Green-Blue Features")
    print("-" * 80)
    
    df = extract_gb_features(df, skip_filtering=True)
    
    # -------------------------------------------------------------------------
    # Step 5: Frequency Step Features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 5: Frequency Step Features")
    print("-" * 80)
    
    df = extract_freq_step_features(df)
    
    # -------------------------------------------------------------------------
    # Step 6: Frequency Sectioned Traces
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 6: Frequency Sectioned Traces")
    print("-" * 80)
    
    df = extract_freq_sectioned_traces(df)
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    print(f"\nSaving to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET)
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    main()

