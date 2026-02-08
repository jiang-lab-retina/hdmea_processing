"""
Plot Chip Effective Area visualizations for randomly selected recordings.

This script generates visualization plots showing:
- Black dots for responsive electrode coordinates
- Blue shaded circles representing electrode coverage
- Red outline showing the union boundary
- Statistics in title (n, union_area, overlap_area)

Usage:
    python plot_effective_area.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import random
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from shapely.geometry import Point
from shapely.ops import unary_union


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Input paths
INPUT_PARQUET = Path(__file__).parent / "firing_rate_with_all_features_loaded_extracted_area20260205.parquet"
HDF5_DIR = PROJECT_ROOT / "Projects/unified_pipeline/export_dsgc_sta_updated"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "figures" / "chip_effective_area"

# eimage_sta path relative to units/{unit_id}/
EIMAGE_STA_PATH = "features/eimage_sta/data"

# Responsive electrode detection parameters
STD_THRESHOLD = 3.0
SOMA_TEMPORAL_RANGE = (5, 9)

# Circle parameters
CIRCLE_RADIUS = 3.0
CIRCLE_RESOLUTION = 64

# Good unit threshold
GOOD_UNIT_QI_THRESHOLD = 0.5

# Number of random samples
N_SAMPLES = 10

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_responsive_electrodes(
    eimage_sta: np.ndarray,
    std_threshold: float = STD_THRESHOLD,
    temporal_range: tuple = SOMA_TEMPORAL_RANGE,
) -> np.ndarray:
    """Extract responsive electrode coordinates from eimage_sta data."""
    t_start, t_end = temporal_range
    soma_sta = eimage_sta[t_start:t_end]
    
    overall_std = soma_sta.std()
    
    if overall_std == 0:
        return np.empty((0, 2), dtype=int)
    
    stds = soma_sta.std(axis=0)
    mask = stds > (std_threshold * overall_std)
    responsive_electrodes = np.argwhere(mask)
    
    return responsive_electrodes


def circles_union_area(points: np.ndarray, radius: float = CIRCLE_RADIUS, resolution: int = CIRCLE_RESOLUTION) -> dict:
    """Compute union area of circles centered at given points."""
    pts = np.asarray(points)
    
    if pts.size == 0:
        return {
            "n": 0,
            "sum_individual_area": 0.0,
            "sum_buffered_area": 0.0,
            "union_area": 0.0,
            "overlap_area": 0.0,
            "union_geom": None,
        }
    
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
        pts = pts.T
    
    n = len(pts)
    
    # Use (col, row) = (x, y) to match scatter plot coordinates
    circles = [Point(p[1], p[0]).buffer(radius, resolution=resolution) for p in pts]
    
    sum_individual_area = n * math.pi * (radius ** 2)
    sum_buffered_area = sum(c.area for c in circles)
    
    union = unary_union(circles)
    union_area = union.area if union is not None else 0.0
    
    overlap_area = sum_individual_area - union_area
    
    return {
        "n": n,
        "sum_individual_area": sum_individual_area,
        "sum_buffered_area": sum_buffered_area,
        "union_area": union_area,
        "overlap_area": overlap_area,
        "union_geom": union,
    }


def parse_index(index_value: str) -> tuple:
    """Parse DataFrame index to extract dataset_id and unit_id."""
    parts = index_value.rsplit("_unit_", 1)
    if len(parts) == 2:
        dataset_id = parts[0]
        unit_id = f"unit_{parts[1]}"
        return dataset_id, unit_id
    else:
        raise ValueError(f"Cannot parse index: {index_value}")


def group_indices_by_file(df: pd.DataFrame) -> dict:
    """Group DataFrame indices by source HDF5 file (dataset_id)."""
    grouped = defaultdict(list)
    
    for idx in df.index:
        try:
            dataset_id, unit_id = parse_index(idx)
            grouped[dataset_id].append((idx, unit_id))
        except ValueError:
            continue
    
    return dict(grouped)


def load_eimage_sta(h5_file: h5py.File, unit_id: str):
    """Load eimage_sta data for a unit from an open HDF5 file."""
    sta_path = f"units/{unit_id}/{EIMAGE_STA_PATH}"
    
    if sta_path in h5_file:
        return h5_file[sta_path][()]
    return None


def get_responsive_coords_for_file(
    h5_path: Path,
    unit_indices: list,
    df: pd.DataFrame,
    qi_threshold: float = GOOD_UNIT_QI_THRESHOLD,
) -> np.ndarray:
    """Get all responsive electrode coordinates for a recording file."""
    all_responsive_coords = []
    
    with h5py.File(h5_path, "r") as f:
        for idx, unit_id in unit_indices:
            qi_value = df.loc[idx, "step_up_QI"]
            if pd.isna(qi_value) or qi_value <= qi_threshold:
                continue
            
            eimage_sta = load_eimage_sta(f, unit_id)
            if eimage_sta is None:
                continue
            
            responsive = get_responsive_electrodes(eimage_sta)
            if len(responsive) > 0:
                all_responsive_coords.append(responsive)
    
    if len(all_responsive_coords) == 0:
        return np.empty((0, 2), dtype=int)
    
    all_coords = np.vstack(all_responsive_coords)
    all_coords = np.unique(all_coords, axis=0)
    
    return all_coords


def plot_effective_area(
    coords: np.ndarray,
    dataset_id: str,
    output_path: Path,
    radius: float = CIRCLE_RADIUS,
):
    """
    Create visualization of effective area.
    
    Style matches the reference image:
    - Black dots for electrode points
    - Blue shaded circles with transparency gradient
    - Red outline for union boundary
    - Statistics in title
    """
    result = circles_union_area(coords, radius=radius)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if result["union_geom"] is not None and result["n"] > 0:
        # Plot the union geometry with blue fill and red boundary
        union_geom = result["union_geom"]
        
        # Handle both Polygon and MultiPolygon
        if union_geom.geom_type == 'Polygon':
            polygons = [union_geom]
        else:  # MultiPolygon
            polygons = list(union_geom.geoms)
        
        # Create gradient effect by plotting multiple layers
        for poly in polygons:
            # Get exterior coordinates
            x, y = poly.exterior.xy
            
            # Fill with blue gradient effect (multiple layers with decreasing alpha)
            for i, alpha in enumerate([0.15, 0.25, 0.35]):
                scale = 1.0 - i * 0.05
                ax.fill(x, y, color='steelblue', alpha=alpha, zorder=1)
            
            # Red boundary
            ax.plot(x, y, color='red', linewidth=1.5, zorder=3)
            
            # Handle any interior rings (holes)
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, color='white', zorder=2)
                ax.plot(ix, iy, color='red', linewidth=1.5, zorder=3)
    
    # Plot the points as black dots
    if len(coords) > 0:
        ax.scatter(coords[:, 1], coords[:, 0], c='black', s=15, zorder=4)
    
    # Set title with statistics
    n = result["n"]
    union_area = result["union_area"]
    overlap_area = result["overlap_area"]
    
    ax.set_title(f"n-{n}  union_area={union_area:.2f}  overlap_area={overlap_area:.2f}", fontsize=12)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add some padding to the limits
    if len(coords) > 0:
        margin = radius * 2
        ax.set_xlim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
        ax.set_ylim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return result


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Generate effective area plots for randomly selected recordings."""
    print("=" * 80)
    print("CHIP EFFECTIVE AREA VISUALIZATION")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} units")
    
    # Group by file
    print("\nGrouping by recording file...")
    grouped = group_indices_by_file(df)
    print(f"Found {len(grouped)} unique recordings")
    
    # Filter to recordings with chip_effective_area > 0
    valid_recordings = []
    for dataset_id, unit_indices in grouped.items():
        h5_path = HDF5_DIR / f"{dataset_id}.h5"
        if not h5_path.exists():
            continue
        
        # Check if any unit has area > 0
        indices = [idx for idx, _ in unit_indices]
        areas = df.loc[indices, "chip_effective_area"]
        if areas.max() > 0:
            valid_recordings.append(dataset_id)
    
    print(f"Recordings with area > 0: {len(valid_recordings)}")
    
    # Randomly select N recordings
    random.seed(RANDOM_SEED)
    selected = random.sample(valid_recordings, min(N_SAMPLES, len(valid_recordings)))
    print(f"\nRandomly selected {len(selected)} recordings for visualization")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    for i, dataset_id in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {dataset_id}")
        
        h5_path = HDF5_DIR / f"{dataset_id}.h5"
        unit_indices = grouped[dataset_id]
        
        # Get responsive coordinates
        coords = get_responsive_coords_for_file(h5_path, unit_indices, df)
        
        if len(coords) == 0:
            print(f"    -> No responsive electrodes found, skipping")
            continue
        
        # Create plot
        output_path = OUTPUT_DIR / f"{dataset_id}_effective_area.png"
        result = plot_effective_area(coords, dataset_id, output_path)
        
        print(f"    -> n={result['n']}, area={result['union_area']:.2f}")
    
    print("\n" + "=" * 80)
    print(f"DONE - Plots saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
