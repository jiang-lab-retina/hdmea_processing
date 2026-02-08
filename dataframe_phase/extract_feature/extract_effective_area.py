"""
Extract Chip Effective Area from eimage_sta data.

This script computes the chip_effective_area for each recording by:
1. Loading eimage_sta data for each unit from HDF5 files
2. Finding responsive electrodes (electrodes with std > 3 * overall_std)
3. Aggregating responsive electrode coordinates from good units (step_up_QI > 0.5)
4. Computing the union area of circles centered at those electrodes

Usage:
    python extract_effective_area.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
from pathlib import Path
from collections import defaultdict
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Shapely imports for geometric calculations
from shapely.geometry import Point
from shapely.ops import unary_union


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Input/Output paths
INPUT_PARQUET = Path(__file__).parent / "firing_rate_with_all_features_loaded_extracted20260104.parquet"
OUTPUT_PARQUET = Path(__file__).parent / "firing_rate_with_all_features_loaded_extracted_area20260205.parquet"

# HDF5 source directory
HDF5_DIR = PROJECT_ROOT / "Projects/unified_pipeline/export_dsgc_sta_updated"

# eimage_sta path relative to units/{unit_id}/
EIMAGE_STA_PATH = "features/eimage_sta/data"

# Responsive electrode detection parameters
STD_THRESHOLD = 3.0  # Electrodes with std > threshold * overall_std are considered responsive
SOMA_TEMPORAL_RANGE = (5, 9)  # Frames 5-8 (soma response window)

# Circles union area parameters
CIRCLE_RADIUS = 3.0
CIRCLE_RESOLUTION = 64

# Good unit threshold
GOOD_UNIT_QI_THRESHOLD = 0.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_responsive_electrodes(
    eimage_sta: np.ndarray,
    std_threshold: float = STD_THRESHOLD,
    temporal_range: tuple = SOMA_TEMPORAL_RANGE,
) -> np.ndarray:
    """
    Extract responsive electrode coordinates from eimage_sta data.
    
    Parameters
    ----------
    eimage_sta : np.ndarray
        3D array of shape (window_length, rows, cols), typically (50, 64, 64)
    std_threshold : float
        Multiplier for overall_std to determine responsive threshold
    temporal_range : tuple
        (start, end) frame indices to analyze for soma response
        
    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing (row, col) coordinates of responsive electrodes.
        Returns empty array with shape (0, 2) if no responsive electrodes found.
    """
    # Extract temporal window for soma detection
    t_start, t_end = temporal_range
    soma_sta = eimage_sta[t_start:t_end]
    
    # Calculate overall standard deviation
    overall_std = soma_sta.std()
    
    if overall_std == 0:
        return np.empty((0, 2), dtype=int)
    
    # Get the standard deviation along the time axis for each electrode
    stds = soma_sta.std(axis=0)
    
    # Create boolean mask for electrodes where std exceeds threshold
    mask = stds > (std_threshold * overall_std)
    
    # Find indices where condition is true
    responsive_electrodes = np.argwhere(mask)
    
    return responsive_electrodes


def circles_union_area(
    points: np.ndarray,
    radius: float = CIRCLE_RADIUS,
    resolution: int = CIRCLE_RESOLUTION,
) -> dict:
    """
    Compute union area of circles centered at given points.
    
    Expand each input point to a circle of given radius, compute union area
    (overlap handled via unary_union), return dict with area metrics.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing (row, col) or (x, y) coordinates
    radius : float
        Radius of each circle
    resolution : int
        Number of segments used to approximate each circle
        
    Returns
    -------
    dict
        Dictionary containing:
        - n: number of points
        - sum_individual_area: sum of individual circle areas (pi * r^2 * n)
        - sum_buffered_area: sum of actual buffered polygon areas
        - union_area: area of the union of all circles
        - overlap_area: total overlap area (sum_individual - union)
    """
    pts = np.asarray(points)
    
    # Handle edge cases
    if pts.size == 0:
        return {
            "n": 0,
            "sum_individual_area": 0.0,
            "sum_buffered_area": 0.0,
            "union_area": 0.0,
            "overlap_area": 0.0,
        }
    
    # Ensure shape is (N, 2)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
        pts = pts.T
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (N, 2), got {pts.shape}")
    
    n = len(pts)
    
    # Build shapely circle buffers (resolution controls polygon approximation)
    circles = [Point(tuple(p)).buffer(radius, resolution=resolution) for p in pts]
    
    # Sum of individual circle areas (theoretical)
    sum_individual_area = n * math.pi * (radius ** 2)
    
    # Sum of actual buffered polygon areas (should be ~equal to theoretical)
    sum_buffered_area = sum(c.area for c in circles)
    
    # Union (handles overlaps automatically)
    union = unary_union(circles)
    union_area = union.area if union is not None else 0.0
    
    # Overlap area = sum individual - union
    overlap_area = sum_individual_area - union_area
    
    return {
        "n": n,
        "sum_individual_area": sum_individual_area,
        "sum_buffered_area": sum_buffered_area,
        "union_area": union_area,
        "overlap_area": overlap_area,
    }



def parse_index(index_value: str) -> tuple:
    """
    Parse DataFrame index to extract dataset_id and unit_id.
    
    Index format: {dataset_id}_unit_{unit_id}
    Example: "2024.02.26-10.53.19-Rec_unit_001" -> ("2024.02.26-10.53.19-Rec", "unit_001")
    
    Parameters
    ----------
    index_value : str
        DataFrame index value
        
    Returns
    -------
    tuple
        (dataset_id, unit_id)
    """
    parts = index_value.rsplit("_unit_", 1)
    if len(parts) == 2:
        dataset_id = parts[0]
        unit_id = f"unit_{parts[1]}"
        return dataset_id, unit_id
    else:
        raise ValueError(f"Cannot parse index: {index_value}")


def group_indices_by_file(df: pd.DataFrame) -> dict:
    """
    Group DataFrame indices by source HDF5 file (dataset_id).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index format "dataset_id_unit_XXX"
        
    Returns
    -------
    dict
        Dictionary mapping dataset_id -> list of (index, unit_id) tuples
    """
    grouped = defaultdict(list)
    
    for idx in df.index:
        try:
            dataset_id, unit_id = parse_index(idx)
            grouped[dataset_id].append((idx, unit_id))
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    return dict(grouped)


def load_eimage_sta(h5_file: h5py.File, unit_id: str) -> Optional[np.ndarray]:
    """
    Load eimage_sta data for a unit from an open HDF5 file.
    
    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file
    unit_id : str
        Unit identifier (e.g., "unit_001")
        
    Returns
    -------
    Optional[np.ndarray]
        eimage_sta array of shape (window_length, rows, cols), or None if not found
    """
    sta_path = f"units/{unit_id}/{EIMAGE_STA_PATH}"
    
    if sta_path in h5_file:
        return h5_file[sta_path][()]
    return None



def compute_effective_area_for_file(
    h5_path: Path,
    unit_indices: list,
    df: pd.DataFrame,
    qi_threshold: float = GOOD_UNIT_QI_THRESHOLD,
) -> float:
    """
    Compute effective area for a single recording file.
    
    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file
    unit_indices : list
        List of (index, unit_id) tuples for units in this file
    df : pd.DataFrame
        DataFrame containing step_up_QI values
    qi_threshold : float
        Quality index threshold for "good" units
        
    Returns
    -------
    float
        Effective area (union area of circles centered at responsive electrodes)
    """
    all_responsive_coords = []
    
    with h5py.File(h5_path, "r") as f:
        for idx, unit_id in unit_indices:
            # Check if this is a good unit
            qi_value = df.loc[idx, "step_up_QI"]
            if pd.isna(qi_value) or qi_value <= qi_threshold:
                continue
            
            # Load eimage_sta
            eimage_sta = load_eimage_sta(f, unit_id)
            if eimage_sta is None:
                continue
            
            # Get responsive electrodes
            responsive = get_responsive_electrodes(eimage_sta)
            if len(responsive) > 0:
                all_responsive_coords.append(responsive)
    
    # Aggregate all coordinates
    if len(all_responsive_coords) == 0:
        return 0.0
    
    all_coords = np.vstack(all_responsive_coords)
    
    # Deduplicate coordinates (same electrode may be responsive in multiple units)
    all_coords = np.unique(all_coords, axis=0)
    
    # Compute union area
    result = circles_union_area(all_coords)
    
    return result["union_area"]



# =============================================================================
# MAIN PIPELINE
# =============================================================================


def extract_chip_effective_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract chip_effective_area for all recordings and add to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with index format "dataset_id_unit_XXX" and step_up_QI column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with chip_effective_area column added
    """
    print("\nGrouping indices by source file...")
    grouped = group_indices_by_file(df)
    print(f"Found {len(grouped)} unique recording files")
    
    # Initialize the new column
    df["chip_effective_area"] = np.nan
    
    # Track statistics
    processed_files = 0
    missing_files = 0
    files_with_area = 0
    
    print("\nProcessing files...")
    for dataset_id, unit_indices in tqdm(grouped.items(), desc="Computing effective areas"):
        h5_path = HDF5_DIR / f"{dataset_id}.h5"
        
        if not h5_path.exists():
            missing_files += 1
            continue
        
        # Compute effective area for this file
        effective_area = compute_effective_area_for_file(h5_path, unit_indices, df)
        
        # Assign to all rows from this file
        indices = [idx for idx, _ in unit_indices]
        df.loc[indices, "chip_effective_area"] = effective_area
        
        processed_files += 1
        if effective_area > 0:
            files_with_area += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total files: {len(grouped)}")
    print(f"  Processed: {processed_files}")
    print(f"  Missing HDF5: {missing_files}")
    print(f"  Files with area > 0: {files_with_area}")
    
    # Print statistics on the new column
    valid_values = df["chip_effective_area"].notna().sum()
    print(f"\nchip_effective_area statistics:")
    print(f"  Valid values: {valid_values}")
    if valid_values > 0:
        print(f"  Min: {df['chip_effective_area'].min():.2f}")
        print(f"  Max: {df['chip_effective_area'].max():.2f}")
        print(f"  Mean: {df['chip_effective_area'].mean():.2f}")
    
    return df



def main():
    """Main function to extract chip_effective_area and save to parquet."""
    print("=" * 80)
    print("CHIP EFFECTIVE AREA EXTRACTION")
    print("=" * 80)
    
    # Load input data
    print(f"\nLoading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} units with {len(df.columns)} columns")
    
    # Check required column
    if "step_up_QI" not in df.columns:
        raise ValueError("Input DataFrame must have 'step_up_QI' column")
    
    # Extract chip_effective_area
    df = extract_chip_effective_area(df)
    
    # Save output
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    print(f"\nSaving to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET)
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    main()
