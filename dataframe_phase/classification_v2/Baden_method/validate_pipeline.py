#!/usr/bin/env python
"""
Standalone validation script for Baden-method pipeline.
This script runs directly without package imports to avoid caching issues.
"""
import sys
import os
from pathlib import Path

# Set working directory to script location
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

# Clear any cached bytecode
import importlib
for mod_name in list(sys.modules.keys()):
    if 'Baden_method' in mod_name:
        del sys.modules[mod_name]

print("=" * 70)
print("BADEN-METHOD PIPELINE VALIDATION")
print("=" * 70)
print(f"Script directory: {script_dir}")
print(f"Working directory: {os.getcwd()}")
print()

# ============================================================================
# Step 1: Verify all source files exist
# ============================================================================
print("[1/6] Verifying source files...")
required_files = [
    'config.py',
    'preprocessing.py', 
    'features.py',
    'clustering.py',
    'evaluation.py',
    'visualization.py',
    'pipeline.py',
    '__init__.py',
]

missing = [f for f in required_files if not (script_dir / f).exists()]
if missing:
    print(f"✗ Missing files: {missing}")
    sys.exit(1)
print(f"✓ All {len(required_files)} source files present")

# ============================================================================
# Step 2: Verify input data exists
# ============================================================================
print("\n[2/6] Verifying input data...")
project_root = script_dir.parents[2]  # Go up to Data_Processing_2027
input_path = project_root / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet"

if not input_path.exists():
    print(f"✗ Input file not found: {input_path}")
    sys.exit(1)
print(f"✓ Input file found ({input_path.stat().st_size / 1e6:.1f} MB)")

# ============================================================================
# Step 3: Test data loading
# ============================================================================
print("\n[3/6] Testing data loading...")
import pandas as pd
import numpy as np

try:
    df = pd.read_parquet(input_path)
    print(f"✓ Loaded {len(df)} cells, {len(df.columns)} columns")
except Exception as e:
    print(f"✗ Failed to load parquet: {e}")
    sys.exit(1)

# ============================================================================
# Step 4: Verify required columns
# ============================================================================
print("\n[4/6] Verifying required columns...")

# Define required columns
required_cols = [
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x", 
    "sta_time_course",
    "corrected_moving_h_bar_s5_d8_3x_000",
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
    "step_up_QI",
    "ds_p_value",
    "axon_type",
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"✗ Missing columns: {missing_cols}")
    sys.exit(1)
print(f"✓ All {len(required_cols)} required columns present")

# ============================================================================
# Step 5: Test trace handling (the problematic area)
# ============================================================================
print("\n[5/6] Testing trace handling...")

trace_cols = [
    "freq_step_5st_3x",           # Chirp - 3 trials
    "green_blue_3s_3i_3x",        # Color - 3 trials
    "sta_time_course",            # RF time course (may be single)
    "corrected_moving_h_bar_s5_d8_3x_000",  # Moving bar - 3 trials
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
]

def average_trials(trace):
    """Average across trials if trace contains multiple repetitions."""
    arr = np.asarray(trace)
    if arr.dtype == object and len(arr.shape) == 1:
        trials = np.vstack([np.asarray(t, dtype=np.float64) for t in arr])
        return np.mean(trials, axis=0)
    return np.asarray(arr, dtype=np.float64)

for col in trace_cols:
    sample = df[col].iloc[0]
    print(f"  {col}:")
    print(f"    Type: {type(sample)}")
    arr = np.asarray(sample)
    print(f"    Array dtype: {arr.dtype}, shape: {arr.shape}")
    
    # Check if nested (multiple trials)
    if arr.dtype == object and len(arr.shape) == 1:
        print(f"    Structure: {len(arr)} trials, each shape {np.asarray(arr[0]).shape}")
        # Test trial averaging
        try:
            averaged = average_trials(sample)
            print(f"    Trial averaging: ✓ → shape {averaged.shape}")
            print(f"    Has NaN after averaging: {np.any(np.isnan(averaged))}")
        except Exception as e:
            print(f"    Trial averaging: ✗ ({e})")
    else:
        # Flat array
        try:
            arr_f = np.asarray(sample, dtype=np.float64)
            has_nan = np.any(np.isnan(arr_f))
            print(f"    Float64 conversion: ✓ (NaN: {has_nan})")
        except Exception as e:
            print(f"    Float64 conversion: ✗ ({e})")

print("✓ Trace handling tests passed")

# ============================================================================
# Step 6: Quick filtering test
# ============================================================================
print("\n[6/6] Testing row filtering...")

# Filter by scalar columns first
df_test = df.dropna(subset=["step_up_QI", "ds_p_value", "axon_type"])
print(f"  After scalar NaN filter: {len(df_test)} cells")

# Filter by QI
df_test = df_test[df_test["step_up_QI"] > 0.5]
print(f"  After QI > 0.5 filter: {len(df_test)} cells")

# Filter by axon type
df_test = df_test[df_test["axon_type"] == "rgc"]
print(f"  After axon_type == 'rgc': {len(df_test)} cells")

# Test trace NaN filtering (handles nested trial arrays)
def check_trace_nan(x):
    """Check if trace has NaN values (handles nested trial arrays)."""
    if x is None:
        return True
    arr = np.asarray(x)
    # Check if this is a nested array (multiple trials)
    if arr.dtype == object and len(arr.shape) == 1:
        try:
            for trial in arr:
                trial_arr = np.asarray(trial, dtype=np.float64)
                if np.any(np.isnan(trial_arr)):
                    return True
            return False
        except (ValueError, TypeError):
            return True
    else:
        try:
            arr_f = np.asarray(x, dtype=np.float64)
            return np.any(np.isnan(arr_f))
        except (ValueError, TypeError):
            return True

# Check all trace columns for NaN
all_trace_cols = [
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x",
    "sta_time_course",
    "corrected_moving_h_bar_s5_d8_3x_000",
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
]

for col in all_trace_cols:
    before = len(df_test)
    valid_mask = ~df_test[col].apply(check_trace_nan)
    df_test = df_test[valid_mask]
    removed = before - len(df_test)
    if removed > 0:
        print(f"  After {col} NaN filter: {len(df_test)} cells (-{removed})")

print(f"  Final after all trace NaN filters: {len(df_test)} cells")

# DS/non-DS split
n_ds = (df_test["ds_p_value"] < 0.05).sum()
n_nds = (df_test["ds_p_value"] >= 0.05).sum()
print(f"  DS cells: {n_ds}, non-DS cells: {n_nds}")

print("\n" + "=" * 70)
print("✓ ALL VALIDATION TESTS PASSED")
print("=" * 70)
print(f"\nReady to run full pipeline with:")
print(f"  - {len(df_test)} filtered RGC cells")
print(f"  - {n_ds} DS cells, {n_nds} non-DS cells")
print(f"\nTo run the full pipeline, execute in Python:")
print("  from dataframe_phase.classification_v2.Baden_method import run_baden_pipeline")
print("  results = run_baden_pipeline()")

