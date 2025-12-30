"""
Reshape Interframe Firing Rate DataFrame

Transforms the trial-based DataFrame into a movie-based DataFrame where:
- Index: {dataset_id}_{unit_id} (unchanged)
- Columns: movie names (with direction for moving_h_bar)
- Cell values: 2D numpy array with shape (n_trials, n_bins)

Includes validation to reject units with inconsistent trial lengths.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
import pandas as pd

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
INPUT_FILE = OUTPUT_DIR / "interframe_firing_rate.parquet"
OUTPUT_FILE = OUTPUT_DIR / "firing_rate_by_movie.parquet"

# Moving bar movie name pattern
MOVING_BAR_PREFIX = "moving_h_bar_s5_d8_3x"

# Columns to exclude (inconsistent trial lengths)
EXCLUDED_COLUMNS = ["iprgc_test_2"]  # Trial 2 has incomplete data in some recordings


def parse_column_groups(columns: List[str]) -> Dict[str, List[str]]:
    """
    Group columns by movie name.
    
    Regular movies: {movie}_{trial} -> group by movie
    Moving bar: {movie}_{direction}_{trial} -> group by {movie}_{direction}
    
    Returns:
        Dict mapping movie_group_name -> list of column names
    """
    groups: Dict[str, List[str]] = {}
    
    for col in columns:
        # Skip excluded columns
        if col in EXCLUDED_COLUMNS:
            continue
        if col.startswith(MOVING_BAR_PREFIX):
            # Pattern: moving_h_bar_s5_d8_3x_{direction}_{trial}
            # Split from the end to get trial number
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                movie_dir = parts[0]  # e.g., moving_h_bar_s5_d8_3x_45
                trial = parts[1]
                
                if movie_dir not in groups:
                    groups[movie_dir] = []
                groups[movie_dir].append(col)
        else:
            # Pattern: {movie}_{trial}
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                movie = parts[0]
                trial = parts[1]
                
                if movie not in groups:
                    groups[movie] = []
                groups[movie].append(col)
    
    # Sort columns within each group by trial number
    for movie in groups:
        groups[movie] = sorted(groups[movie], key=lambda x: int(x.rsplit("_", 1)[1]))
    
    return groups


def get_trial_lengths(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Get the length of each trial for each unit.
    
    Returns:
        DataFrame with index=unit, columns=trial_columns, values=lengths
    """
    lengths = pd.DataFrame(index=df.index)
    
    for col in columns:
        lengths[col] = df[col].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
        )
    
    return lengths


def validate_units(
    df: pd.DataFrame, 
    movie_groups: Dict[str, List[str]]
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Validate units: reject any unit where ANY trial differs by >1 frame from mode.
    
    Returns:
        rejected_units: Set of unit indices to reject
        mode_lengths: Dict mapping movie_group -> mode length
    """
    rejected_units: Set[str] = set()
    mode_lengths: Dict[str, int] = {}
    rejection_reasons: Dict[str, List[str]] = {}
    
    for movie, columns in movie_groups.items():
        # Get trial lengths for this movie
        lengths = get_trial_lengths(df, columns)
        
        # Flatten all lengths to find mode
        all_lengths = lengths.values.flatten()
        all_lengths = all_lengths[all_lengths > 0]  # Exclude missing (0 length)
        
        if len(all_lengths) == 0:
            continue
        
        # Find mode (most frequent length)
        length_counts = Counter(all_lengths)
        mode_length = length_counts.most_common(1)[0][0]
        mode_lengths[movie] = mode_length
        
        # Check each unit
        for unit_idx in df.index:
            unit_lengths = lengths.loc[unit_idx].values
            
            for i, length in enumerate(unit_lengths):
                if length == 0:
                    continue  # Skip missing trials
                
                diff = abs(length - mode_length)
                if diff > 1:
                    rejected_units.add(unit_idx)
                    if unit_idx not in rejection_reasons:
                        rejection_reasons[unit_idx] = []
                    rejection_reasons[unit_idx].append(
                        f"{movie} trial {columns[i]}: {length} (mode={mode_length}, diff={diff})"
                    )
    
    return rejected_units, mode_lengths, rejection_reasons


def compute_target_lengths(
    df: pd.DataFrame, 
    movie_groups: Dict[str, List[str]],
    valid_units: pd.Index
) -> Dict[str, int]:
    """
    Compute the shortest trial length for each movie across all valid units.
    
    Returns:
        Dict mapping movie_group -> target length (minimum)
    """
    target_lengths: Dict[str, int] = {}
    
    valid_df = df.loc[valid_units]
    
    for movie, columns in movie_groups.items():
        min_length = float('inf')
        
        for col in columns:
            col_lengths = valid_df[col].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
            )
            col_lengths = col_lengths[col_lengths > 0]  # Exclude missing
            
            if len(col_lengths) > 0:
                min_length = min(min_length, col_lengths.min())
        
        if min_length < float('inf'):
            target_lengths[movie] = int(min_length)
    
    return target_lengths


def stack_and_trim_trials(
    df: pd.DataFrame,
    movie_groups: Dict[str, List[str]],
    target_lengths: Dict[str, int],
    valid_units: pd.Index
) -> pd.DataFrame:
    """
    Stack trials into 2D arrays and trim to target length.
    
    Returns:
        New DataFrame with movie columns containing 2D numpy arrays
    """
    valid_df = df.loc[valid_units]
    
    # Create new DataFrame
    result_data = {movie: [] for movie in movie_groups.keys()}
    result_index = []
    
    for unit_idx in valid_df.index:
        result_index.append(unit_idx)
        
        for movie, columns in movie_groups.items():
            target_len = target_lengths.get(movie, 0)
            
            if target_len == 0:
                result_data[movie].append(None)
                continue
            
            # Collect and trim trials
            trials = []
            for col in columns:
                trial_data = valid_df.loc[unit_idx, col]
                
                if trial_data is not None and isinstance(trial_data, (list, np.ndarray)) and len(trial_data) > 0:
                    arr = np.array(trial_data, dtype=np.float32)
                    # Trim to target length
                    arr = arr[:target_len]
                    trials.append(arr)
            
            if len(trials) > 0:
                # Stack into 2D array: (n_trials, n_bins)
                stacked = np.stack(trials, axis=0)
                result_data[movie].append(stacked)
            else:
                result_data[movie].append(None)
    
    result_df = pd.DataFrame(result_data, index=result_index)
    result_df.index.name = df.index.name
    
    return result_df


def main():
    print("=" * 80)
    print("Reshape Interframe Firing Rate DataFrame")
    print("=" * 80)
    
    # ==========================================================================
    # Step 1: Load and Parse Column Structure
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Load and Parse Column Structure")
    print("=" * 80)
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"\nLoaded DataFrame: {df.shape}")
    print(f"  Rows (units): {len(df)}")
    print(f"  Columns (trials): {len(df.columns)}")
    
    # Parse column groups
    movie_groups = parse_column_groups(df.columns.tolist())
    
    print(f"\nMovie groups identified: {len(movie_groups)}")
    for movie, columns in sorted(movie_groups.items()):
        print(f"  {movie}: {len(columns)} trials")
    
    # ==========================================================================
    # Step 2: Validation (Reject Outlier Units)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Validation (Reject Outlier Units)")
    print("=" * 80)
    
    rejected_units, mode_lengths, rejection_reasons = validate_units(df, movie_groups)
    
    print(f"\nMode lengths per movie:")
    for movie, mode in sorted(mode_lengths.items()):
        print(f"  {movie}: {mode} bins")
    
    if len(rejected_units) > 0:
        # Print error in red
        print("\n" + "\033[91m" + "=" * 80)
        print("ERROR: Units with inconsistent trial lengths detected!")
        print("=" * 80 + "\033[0m")
        print(f"\n\033[91mRejected units: {len(rejected_units)}\033[0m")
        
        for unit in sorted(rejected_units)[:20]:  # Show first 20
            print(f"\n\033[91m  {unit}:\033[0m")
            for reason in rejection_reasons.get(unit, [])[:5]:  # Show first 5 reasons
                print(f"\033[91m    - {reason}\033[0m")
        
        if len(rejected_units) > 20:
            print(f"\n\033[91m  ... and {len(rejected_units) - 20} more units\033[0m")
        
        # Raise error
        raise ValueError(
            f"Validation failed: {len(rejected_units)} units have trials with >1 frame "
            f"difference from mode length. See details above."
        )
    else:
        print("\nAll units passed validation (all trials within 1 frame of mode).")
    
    valid_units = df.index
    
    # ==========================================================================
    # Step 3: Compute Target Lengths
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Compute Target Lengths (Minimum per Movie)")
    print("=" * 80)
    
    target_lengths = compute_target_lengths(df, movie_groups, valid_units)
    
    print(f"\nTarget lengths (shortest trial per movie):")
    for movie, target in sorted(target_lengths.items()):
        mode = mode_lengths.get(movie, target)
        diff = mode - target
        note = f" (trimming {diff} bins)" if diff > 0 else ""
        print(f"  {movie}: {target} bins{note}")
    
    # ==========================================================================
    # Step 4: Stack and Trim Trials
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Stack and Trim Trials into 2D Arrays")
    print("=" * 80)
    
    result_df = stack_and_trim_trials(df, movie_groups, target_lengths, valid_units)
    
    print(f"\nResult DataFrame: {result_df.shape}")
    print(f"  Rows (units): {len(result_df)}")
    print(f"  Columns (movies): {len(result_df.columns)}")
    
    # Verify shapes
    print("\nArray shapes per movie:")
    for movie in sorted(result_df.columns):
        sample = result_df[movie].dropna().iloc[0]
        if sample is not None:
            print(f"  {movie}: {sample.shape} (n_trials x n_bins)")
    
    # ==========================================================================
    # Step 5: Save Output
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Save Output")
    print("=" * 80)
    
    # Convert numpy arrays to lists for parquet storage
    result_df_save = result_df.copy()
    for col in result_df_save.columns:
        result_df_save[col] = result_df_save[col].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
    
    result_df_save.to_parquet(OUTPUT_FILE)
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"""
Reshaped DataFrame:
-------------------
- Index: {{dataset_id}}_{{unit_id}}
- Total units: {len(result_df)}
- Total movie columns: {len(result_df.columns)}

Movies processed:
""")
    
    for movie in sorted(result_df.columns):
        sample = result_df[movie].dropna().iloc[0]
        if sample is not None:
            n_trials, n_bins = sample.shape
            print(f"  - {movie}: {n_trials} trials x {n_bins} bins")
    
    print(f"""
Output file: {OUTPUT_FILE.name}
""")
    
    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

