"""
DSGC Feature Extraction.

Computes Direction Selectivity Index (DSI), Orientation Selectivity Index (OSI),
and their p-values using permutation testing following the legacy code methodology.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Tuple, Optional, List
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
# INPUT_PARQUET = Path(__file__).parent.parent /"load_feature" / "firing_rate_with_dsgc_features_typed20251230.parquet"
# OUTPUT_PARQUET = Path(__file__).parent / "firing_rate_with_dsgc_features_typed20251230_dsgc_corrected.parquet"

INPUT_PARQUET = Path(__file__).parent.parent /"extract_feature" / "firing_rate_with_all_features_loaded_extracted20260104.parquet"
OUTPUT_PARQUET = Path(__file__).parent / "firing_rate_with_all_features_loaded_extracted20260104.parquet"



# Direction columns in order (degrees)
DIRECTION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DIRECTION_COLUMNS = [f"moving_h_bar_s5_d8_3x_{angle}" for angle in DIRECTION_ANGLES]

# Corrected direction columns (with 3-digit angle format, derived from DIRECTION_ANGLES)
CORRECTED_DIRECTION_COLUMNS = [f"corrected_moving_h_bar_s5_d8_3x_{angle:03d}" for angle in DIRECTION_ANGLES]

# Permutation test settings
N_PERMUTATIONS = 2000
N_TRIALS = 3

# =============================================================================
# CORE FUNCTIONS (from Legacy Code)
# =============================================================================


def compute_corrected_angle(raw_angle: float, correction: float) -> str:
    """
    Compute corrected angle and return as 3-digit string.
    
    Parameters
    ----------
    raw_angle : float
        Original angle in degrees (0, 45, 90, etc.)
    correction : float
        Angle correction to apply (angle_correction_applied)
        
    Returns
    -------
    str
        Corrected angle as 3-character string ("000", "045", "090", etc.)
    """
    # Add correction
    corrected_angle = raw_angle + correction
    
    # Round to nearest 45 degrees
    corrected_angle = round(corrected_angle / 45) * 45
    
    # Normalize to 0-360 range (handle negative and >= 360)
    corrected_angle = corrected_angle % 360
    
    # Convert to 3-digit string
    return f"{int(corrected_angle):03d}"


def remap_direction_columns(row: pd.Series) -> dict:
    """
    Remap original direction columns to corrected direction columns.
    
    For each original direction column, computes the corrected angle based on
    angle_correction_applied and assigns the data to the appropriate corrected column.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing direction columns and angle_correction_applied
        
    Returns
    -------
    dict
        Dictionary mapping corrected column names to their values.
        All values are None if angle_correction_applied is NaN.
    """
    # Initialize result with None for all corrected columns
    result = {col: None for col in CORRECTED_DIRECTION_COLUMNS}
    
    # Get angle correction value
    correction = row.get("angle_correction_applied")
    
    # If correction is NaN, return all None
    if pd.isna(correction):
        return result
    
    # For each original direction, compute corrected angle and remap data
    for raw_angle in DIRECTION_ANGLES:
        # Get original column name and data
        original_col = f"moving_h_bar_s5_d8_3x_{raw_angle}"
        original_data = row.get(original_col)
        
        # Compute corrected angle string
        corrected_angle_str = compute_corrected_angle(raw_angle, correction)
        
        # Build corrected column name
        corrected_col = f"corrected_moving_h_bar_s5_d8_3x_{corrected_angle_str}"
        
        # Assign data to corrected column
        if corrected_col in result:
            result[corrected_col] = original_data
    
    return result


def calculate_direction_index(
    directions: np.ndarray,
    responses: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Direction Selectivity Index using vector sum method.
    
    DSI = |Σ(R_i * e^(j*θ_i))| / max(|R_i|)
    
    Parameters
    ----------
    directions : np.ndarray
        Direction angles in degrees (e.g., [0, 45, 90, ...])
    responses : np.ndarray
        Response magnitude for each direction
        
    Returns
    -------
    Tuple of (direction_index, vector_sum_length, vector_sum_angle)
        Returns (None, None, None) if inputs are invalid
    """
    if directions is None or responses is None:
        return None, None, None
    if len(responses) == 0 or np.all(responses == 0):
        return 0.0, 0.0, 0.0
    
    directions_rad = np.deg2rad(directions)
    
    # Calculate the complex exponential for each direction
    complex_exponential = responses * np.exp(1j * directions_rad)
    
    # Calculate the vector sum
    vector_sum = np.sum(complex_exponential)
    
    vector_sum_length = np.abs(vector_sum)
    all_vector_length = np.abs(complex_exponential)
    max_length = np.max(all_vector_length)
    
    if max_length == 0:
        return 0.0, 0.0, 0.0
    
    direction_index = vector_sum_length / max_length
    
    # Calculate the angle of the vector sum
    vector_sum_angle = np.angle(vector_sum, deg=True)
    
    return direction_index, vector_sum_length, vector_sum_angle


def calculate_orientation_index(
    directions: np.ndarray,
    responses: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Orientation Selectivity Index using vector sum method.
    
    OSI = |Σ(R_i * e^(2j*θ_i))| / Σ(R_i)
    
    Note: Uses 2j (doubling angle) to treat opposite directions as same orientation.
    
    Parameters
    ----------
    directions : np.ndarray
        Direction angles in degrees (e.g., [0, 45, 90, ...])
    responses : np.ndarray
        Response magnitude for each direction
        
    Returns
    -------
    Tuple of (orientation_index, vector_sum_length, vector_sum_angle)
        Returns (None, None, None) if inputs are invalid
    """
    if directions is None or responses is None:
        return None, None, None
    if len(responses) == 0 or np.all(responses == 0):
        return 0.0, 0.0, 0.0
    
    directions_rad = np.deg2rad(directions)
    
    # Calculate the complex exponential for each direction (2j for orientation)
    complex_exponential = responses * np.exp(2j * directions_rad)
    
    # Calculate the vector sum
    vector_sum = np.sum(complex_exponential)
    
    # Calculate the orientation index
    vector_sum_length = np.abs(vector_sum)
    response_sum = np.sum(responses)
    
    if response_sum == 0:
        return 0.0, 0.0, 0.0
    
    orientation_index = vector_sum_length / response_sum
    
    vector_sum_angle = np.angle(vector_sum, deg=True)
    
    return orientation_index, vector_sum_length, vector_sum_angle


def get_total_firing_rate_per_trial(trial_traces: np.ndarray) -> np.ndarray:
    """
    Compute total firing rate for each trial by summing the trace.
    
    Parameters
    ----------
    trial_traces : np.ndarray
        Array of shape (n_trials,) where each element is a 1D trace array
        
    Returns
    -------
    np.ndarray
        Array of shape (n_trials,) with total firing rate per trial
    """
    totals = []
    for trace in trial_traces:
        if trace is not None and len(trace) > 0:
            totals.append(np.sum(trace))
        else:
            totals.append(0.0)
    return np.array(totals)


def compute_permutation_p_value(
    all_trial_values: np.ndarray,
    real_index: float,
    directions: np.ndarray,
    n_permutations: int,
    n_trials: int,
    mode: str = "direction",
) -> Tuple[float, List[float]]:
    """
    Compute p-value via permutation/shuffle test.
    
    Parameters
    ----------
    all_trial_values : np.ndarray
        Flattened array of all trial values (n_trials * n_directions)
    real_index : float
        The actual DSI or OSI value
    directions : np.ndarray
        Direction angles in degrees
    n_permutations : int
        Number of shuffle iterations
    n_trials : int
        Number of trials per direction
    mode : str
        "direction" for DSI or "orientation" for OSI
        
    Returns
    -------
    Tuple of (p_value, shuffled_distribution)
    """
    n_directions = len(directions)
    shuffled_indices = []
    
    for _ in range(n_permutations):
        # Shuffle the trial values
        shuffled_values = all_trial_values.copy()
        np.random.shuffle(shuffled_values)
        
        # Reshape back to (n_trials, n_directions)
        reshaped = shuffled_values.reshape(n_trials, n_directions)
        
        # Compute mean per direction
        mean_per_direction = np.mean(reshaped, axis=0)
        
        # Calculate index
        if mode == "direction":
            idx, _, _ = calculate_direction_index(directions, mean_per_direction)
        else:
            idx, _, _ = calculate_orientation_index(directions, mean_per_direction)
        
        if idx is not None:
            shuffled_indices.append(idx)
    
    shuffled_indices = np.array(shuffled_indices)
    
    if len(shuffled_indices) == 0 or np.std(shuffled_indices) == 0:
        return 1.0, []
    
    # Compute z-score and p-value
    shuffled_mean = np.mean(shuffled_indices)
    shuffled_std = np.std(shuffled_indices)
    
    z_value = (real_index - shuffled_mean) / shuffled_std
    p_value = 1 - stats.norm.cdf(z_value)
    
    return p_value, shuffled_indices.tolist()


def process_unit(
    row: pd.Series,
    directions: np.ndarray,
    direction_columns: List[str],
    n_permutations: int,
    n_trials: int,
) -> dict:
    """
    Process a single unit to compute DSI, OSI, and p-values.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing direction columns
    directions : np.ndarray
        Direction angles in degrees
    direction_columns : List[str]
        Column names for each direction
    n_permutations : int
        Number of permutation iterations
    n_trials : int
        Number of trials per direction
        
    Returns
    -------
    dict
        Dictionary with dsi, osi, preferred_direction, ds_p_value, os_p_value
    """
    result = {
        "dsi": np.nan,
        "osi": np.nan,
        "preferred_direction": np.nan,
        "ds_p_value": np.nan,
        "os_p_value": np.nan,
    }
    
    # Step 1: Collect total firing rates for all trials and directions
    # Shape will be (n_directions, n_trials)
    all_totals = []
    
    for col in direction_columns:
        trial_traces = row.get(col)
        
        if trial_traces is None:
            return result
        
        # Convert to numpy array if needed
        if isinstance(trial_traces, list):
            trial_traces = np.array(trial_traces, dtype=object)
        
        if len(trial_traces) == 0:
            return result
        
        # Get total firing rate per trial
        totals = get_total_firing_rate_per_trial(trial_traces)
        all_totals.append(totals)
    
    # Convert to 2D array: (n_directions, n_trials)
    all_totals = np.array(all_totals)
    
    if all_totals.shape != (len(directions), n_trials):
        return result
    
    # Step 2: Compute mean response per direction
    mean_per_direction = np.mean(all_totals, axis=1)  # Shape: (n_directions,)
    
    # Step 3: Calculate DSI and OSI
    dsi, _, preferred_angle = calculate_direction_index(directions, mean_per_direction)
    osi, _, _ = calculate_orientation_index(directions, mean_per_direction)
    
    if dsi is None or osi is None:
        return result
    
    result["dsi"] = dsi
    result["osi"] = osi
    # Convert preferred_direction to 0-360 range
    result["preferred_direction"] = preferred_angle % 360 if preferred_angle is not None else np.nan
    
    # Step 4: Permutation test for p-values
    # Flatten all trial values: (n_trials, n_directions) for shuffling
    # Note: all_totals is (n_directions, n_trials), need to transpose
    all_trial_values = all_totals.T.flatten()  # Shape: (n_trials * n_directions,)
    
    # DSI p-value
    ds_p_value, _ = compute_permutation_p_value(
        all_trial_values,
        dsi,
        directions,
        n_permutations,
        n_trials,
        mode="direction",
    )
    result["ds_p_value"] = ds_p_value
    
    # OSI p-value
    os_p_value, _ = compute_permutation_p_value(
        all_trial_values,
        osi,
        directions,
        n_permutations,
        n_trials,
        mode="orientation",
    )
    result["os_p_value"] = os_p_value
    
    return result


def main():
    """Main pipeline: load parquet, process all units, save results."""
    print("=" * 80)
    print("DSGC Feature Extraction")
    print("=" * 80)
    
    # Load input data
    print(f"\nLoading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} units with {len(df.columns)} columns")
    
    # Check for required columns
    missing_cols = [col for col in DIRECTION_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"\nERROR: Missing direction columns: {missing_cols}")
        return
    
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
    
    # Count valid corrected rows
    valid_corrections = df["angle_correction_applied"].notna().sum()
    print(f"Valid corrections applied: {valid_corrections} / {len(df)}")
    
    print(f"\nProcessing with {N_PERMUTATIONS} permutations per unit...")
    print("(Using corrected direction columns, skipping rows with NaN angle_correction_applied)")
    
    # Convert directions to numpy array
    directions = np.array(DIRECTION_ANGLES)
    
    # Process each unit (only those with valid angle correction)
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
            CORRECTED_DIRECTION_COLUMNS,  # Use corrected columns
            N_PERMUTATIONS,
            N_TRIALS,
        )
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results, index=df.index)
    
    # Add new columns to original DataFrame
    for col in ["dsi", "osi", "preferred_direction", "ds_p_value", "os_p_value"]:
        df[col] = results_df[col]
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    valid_dsi = df["dsi"].dropna()
    valid_osi = df["osi"].dropna()
    valid_ds_p = df["ds_p_value"].dropna()
    valid_os_p = df["os_p_value"].dropna()
    
    print(f"\nValid DSI values: {len(valid_dsi)} / {len(df)}")
    print(f"  Mean DSI: {valid_dsi.mean():.4f}")
    print(f"  Median DSI: {valid_dsi.median():.4f}")
    print(f"  Range: [{valid_dsi.min():.4f}, {valid_dsi.max():.4f}]")
    
    print(f"\nValid OSI values: {len(valid_osi)} / {len(df)}")
    print(f"  Mean OSI: {valid_osi.mean():.4f}")
    print(f"  Median OSI: {valid_osi.median():.4f}")
    print(f"  Range: [{valid_osi.min():.4f}, {valid_osi.max():.4f}]")
    
    # Count significant cells
    ds_significant = (df["ds_p_value"] < 0.05).sum()
    os_significant = (df["os_p_value"] < 0.05).sum()
    print(f"\nSignificant direction-selective cells (p < 0.05): {ds_significant}")
    print(f"Significant orientation-selective cells (p < 0.05): {os_significant}")
    
    # Save output
    print(f"\nSaving: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET)
    
    print(f"\nNew columns added:")
    print(f"  - dsi, osi, preferred_direction, ds_p_value, os_p_value")
    print(f"  - {', '.join(CORRECTED_DIRECTION_COLUMNS)}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

