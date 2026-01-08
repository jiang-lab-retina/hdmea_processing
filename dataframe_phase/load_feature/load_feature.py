"""
Load Cell Type Features from HDF5 to DataFrame.

This script loads multiple features from HDF5 files and adds them to the 
parquet DataFrame, processing one file at a time to avoid redundant HDF5 loading.

Features loaded:
- axon_type (cell classification)
- Soma polar coordinates (transformed_x/y, polar_radius, theta, cartesian_x/y)
- STA geometry (gaussian fit parameters, DoG parameters)
- STA time course (array stored as list)
- LNL model fit parameters (a, b, a_norm, bits_per_spike, r_squared, etc.)
- LNL nonlinearity metrics (rectification_index, nonlinearity_index, threshold_g)
- LNL arrays (g_bin_centers, rate_vs_g - stored as lists)
- total_unit_count (total units in the source HDF5 file)

Usage:
    python load_feature.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Input/Output paths
#INPUT_PARQUET = PROJECT_ROOT / "dataframe_phase/extract_feature/firing_rate_with_dsgc_features20251230.parquet"
INPUT_PARQUET = PROJECT_ROOT / "dataframe_phase/load_traces/output/firing_rate_by_movie20260104.parquet"

OUTPUT_PARQUET = Path(__file__).parent / "firing_rate_by_movie_loaded20260104.parquet"

# HDF5 source directory
HDF5_DIR = PROJECT_ROOT / "Projects/unified_pipeline/export_dsgc_sta_updated"

# Feature paths mapping: DataFrame column name -> HDF5 path (relative to units/{unit_id}/)
FEATURE_PATHS = {
    # Existing
    "axon_type": "auto_label/axon_type",
    
    # AP tracking / soma polar coordinates
    "angle_correction_applied": "features/ap_tracking/soma_polar_coordinates/angle_correction_applied",
    "transformed_x": "features/ap_tracking/soma_polar_coordinates/transformed_x",
    "transformed_y": "features/ap_tracking/soma_polar_coordinates/transformed_y",
    "polar_radius": "features/ap_tracking/soma_polar_coordinates/radius",
    "polar_theta_deg": "features/ap_tracking/soma_polar_coordinates/theta_deg",
    "polar_theta_deg_raw": "features/ap_tracking/soma_polar_coordinates/theta_deg_raw",
    "cartesian_x": "features/ap_tracking/soma_polar_coordinates/cartesian_x",
    "cartesian_y": "features/ap_tracking/soma_polar_coordinates/cartesian_y",
    
    # STA geometry - Gaussian fit
    "gaussian_sigma_x": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/sigma_x",
    "gaussian_sigma_y": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/sigma_y",
    "gaussian_amp": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/amplitude",
    "gaussian_r2": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/r_squared",
    
    # STA geometry - DoG
    "dog_sigma_exc": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/sigma_exc",
    "dog_sigma_inh": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/sigma_inh",
    "dog_amp_exc": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/amp_exc",
    "dog_amp_inh": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/amp_inh",
    "dog_r2": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/r_squared",
    
    # STA geometry - time course (array)
    "sta_time_course": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/sta_time_course",
    
    # STA geometry - LNL model fit
    "lnl_a": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/a",
    "lnl_b": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/b",
    "lnl_a_norm": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/a_norm",
    "lnl_bits_per_spike": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/bits_per_spike",
    "lnl_r_squared": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/r_squared",
    "lnl_rectification_index": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/rectification_index",
    "lnl_nonlinearity_index": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/nonlinearity_index",
    "lnl_threshold_g": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/threshold_g",
    "lnl_log_likelihood": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/log_likelihood",
    "lnl_null_log_likelihood": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/null_log_likelihood",
    "lnl_n_frames": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/n_frames",
    "lnl_n_spikes": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/n_spikes",
    "lnl_g_bin_centers": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/g_bin_centers",  # array
    "lnl_rate_vs_g": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/rate_vs_g",  # array
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def parse_index(index_value: str) -> tuple:
    """
    Parse DataFrame index to extract dataset_id and unit_id.
    
    Index format: {dataset_id}_unit_{unit_id}
    Example: "2024.02.26-10.53.19-Rec_unit_001" -> ("2024.02.26-10.53.19-Rec", "unit_001")
    
    Parameters
    ----------
    index_value : str
        The DataFrame index value
        
    Returns
    -------
    tuple
        (dataset_id, unit_id)
    """
    # Split from the right on "_unit_" to handle dataset_ids that might contain underscores
    parts = index_value.rsplit("_unit_", 1)
    if len(parts) == 2:
        dataset_id = parts[0]
        unit_id = f"unit_{parts[1]}"
        return dataset_id, unit_id
    else:
        raise ValueError(f"Cannot parse index: {index_value}")


def group_indices_by_file(df: pd.DataFrame) -> dict:
    """
    Group DataFrame indices by their source HDF5 file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with indices in format {dataset_id}_unit_{unit_id}
        
    Returns
    -------
    dict
        Dictionary mapping dataset_id -> list of (index, unit_id) tuples
    """
    grouped = defaultdict(list)
    
    for idx in df.index:
        dataset_id, unit_id = parse_index(idx)
        grouped[dataset_id].append((idx, unit_id))
    
    return dict(grouped)


def load_features_from_hdf5(h5_path: Path, unit_ids: list, feature_paths: dict) -> tuple:
    """
    Load multiple features for multiple units from a single HDF5 file.
    
    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file
    unit_ids : list
        List of unit IDs to extract (e.g., ["unit_001", "unit_002"])
    feature_paths : dict
        Dictionary mapping column names to HDF5 paths (relative to units/{unit_id}/)
        
    Returns
    -------
    tuple
        (unit_features_dict, total_unit_count)
        - unit_features_dict: {unit_id: {column_name: value, ...}, ...}
        - total_unit_count: int, total number of units in the HDF5 file
    """
    results = {}
    total_unit_count = 0
    
    with h5py.File(h5_path, "r") as f:
        # Get total unit count from the file (count all units in the units/ group)
        if "units" in f:
            total_unit_count = len(f["units"].keys())
        
        for unit_id in unit_ids:
            unit_features = {}
            
            for col_name, h5_subpath in feature_paths.items():
                full_path = f"units/{unit_id}/{h5_subpath}"
                
                if full_path in f:
                    value = f[full_path][()]
                    # Decode bytes to string if necessary
                    if isinstance(value, bytes):
                        unit_features[col_name] = value.decode("utf-8")
                    elif isinstance(value, np.ndarray):
                        # Convert arrays to lists for parquet compatibility
                        unit_features[col_name] = value.tolist()
                    else:
                        # Keep numeric values as-is
                        unit_features[col_name] = value
                else:
                    unit_features[col_name] = None
            
            results[unit_id] = unit_features
    
    return results, total_unit_count


def main():
    """Main pipeline: load parquet, add features from HDF5, save results."""
    print("=" * 80)
    print("Load Cell Type Features from HDF5")
    print("=" * 80)
    
    # Load input parquet
    print(f"\nLoading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} units with {len(df.columns)} columns")
    
    # Group indices by source file
    print("\nGrouping indices by source file...")
    grouped = group_indices_by_file(df)
    print(f"Found {len(grouped)} unique HDF5 files")
    
    # Initialize Series for each feature column
    feature_series = {col: pd.Series(index=df.index, dtype=object) for col in FEATURE_PATHS.keys()}
    # Initialize Series for file-level metadata
    total_unit_count_series = pd.Series(index=df.index, dtype=object)
    
    # Track statistics
    missing_files = []
    feature_counts = {col: 0 for col in FEATURE_PATHS.keys()}  # Count of non-None values
    
    # Process each file
    for dataset_id, index_unit_pairs in tqdm(grouped.items(), desc="Processing HDF5 files"):
        h5_path = HDF5_DIR / f"{dataset_id}.h5"
        
        if not h5_path.exists():
            missing_files.append(dataset_id)
            continue
        
        # Get all unit_ids for this file
        unit_ids = [unit_id for _, unit_id in index_unit_pairs]
        
        # Load all features from this file
        unit_features, file_total_unit_count = load_features_from_hdf5(h5_path, unit_ids, FEATURE_PATHS)
        
        # Map to DataFrame indices
        for idx, unit_id in index_unit_pairs:
            features = unit_features.get(unit_id, {})
            
            for col_name in FEATURE_PATHS.keys():
                value = features.get(col_name)
                feature_series[col_name][idx] = value
                
                if value is not None:
                    feature_counts[col_name] += 1
            
            # Set file-level metadata (same for all units in this file)
            total_unit_count_series[idx] = file_total_unit_count
    
    # Add all columns to DataFrame
    for col_name, series in feature_series.items():
        df[col_name] = series
    df["total_unit_count"] = total_unit_count_series
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_units = len(df)
    print(f"\nTotal units: {total_units}")
    
    if missing_files:
        print(f"\nMissing HDF5 files: {len(missing_files)}")
        for f in missing_files[:5]:
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    # Show feature extraction summary
    print("\nFeature extraction summary:")
    for col_name, count in feature_counts.items():
        missing = total_units - count
        print(f"  {col_name}: {count} found, {missing} missing")
    
    # Show axon_type distribution (special case for categorical)
    print("\nAxon type distribution:")
    value_counts = df["axon_type"].value_counts(dropna=False)
    for axon_type, count in value_counts.items():
        label = axon_type if axon_type is not None else "None/Missing"
        print(f"  {label}: {count}")
    
    # Save output
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET)
    
    print(f"\nNew columns added: {len(FEATURE_PATHS) + 1}")  # +1 for total_unit_count
    print(f"Total columns: {len(df.columns)}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

