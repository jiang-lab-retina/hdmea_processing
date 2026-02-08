"""
Mosaic Validation Pipeline for RGC Subtypes.

This module validates RGC subtype clusters by computing receptive field (RF) coverage.
A valid subtype should have RF coverage > threshold, indicating cells tile the retina.

Key formula:
    conversion_factor = total_good_cell_count / (total_area_mm2 * NORMAL_RGC_PER_MM2)
    rf_coverage = (total_subtype_rf_area / conversion_factor) / total_area_mm2
    
Usage:
    python mosaic_validation.py
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
_PACKAGE_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(_PROJECT_ROOT))

# Import from the divide_conquer_method package
from dataframe_phase.classification_v2.divide_conquer_method.data_loader import load_and_filter_data
from dataframe_phase.classification_v2.divide_conquer_method.grouping import assign_groups
from dataframe_phase.classification_v2.divide_conquer_method import config

# ===============================================================================
# Configuration
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
NORMAL_RGC_PER_MM2 = 3000       # Expected RGC density in mouse retina (cells/mm²)
ELECTRODE_PIXELS_PER_MM = 65   # Electrode array resolution: 65 pixels per mm (for chip_effective_area)
NOISE_PIXELS_PER_MM = 15       # Noise video resolution: 15 pixels per mm (for gaussian_sigma)
RF_COVERAGE_THRESHOLD = 0.8    # Minimum coverage for valid subtype
FWHM_FACTOR = 2.355            # 2 * sqrt(2 * ln(2)) for Gaussian FWHM

# Paths
PACKAGE_DIR = Path(__file__).parent.parent
RESULTS_DIR = PACKAGE_DIR / "results"
VALIDATION_DIR = Path(__file__).parent
OUTPUT_RESULTS_DIR = VALIDATION_DIR / "result"
OUTPUT_FIGURES_DIR = VALIDATION_DIR / "figure"

# Input file (same as used in pipeline)
INPUT_PATH = config.INPUT_PATH


# ===============================================================================
# NaN Validation
# ===============================================================================

# Critical columns for mosaic validation
CRITICAL_COLUMNS = [
    "chip_effective_area",
    "gaussian_sigma_x",
    "gaussian_sigma_y",
    "step_up_QI",
]


def validate_dataframe(
    df: pd.DataFrame,
    critical_columns: list = CRITICAL_COLUMNS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate DataFrame for NaN values in critical columns.
    
    Removes rows with NaN in any critical column and logs statistics.
    
    Args:
        df: Input DataFrame.
        critical_columns: List of columns that must not have NaN values.
        
    Returns:
        Tuple of:
        - valid_df: DataFrame with only valid rows (no NaN in critical columns)
        - nan_report: DataFrame with NaN statistics per column
    """
    logger.info("Validating DataFrame for NaN values...")
    logger.info(f"  Input rows: {len(df)}")
    
    # Check which columns exist
    missing_cols = [col for col in critical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing critical columns in DataFrame: {missing_cols}")
    
    # Count NaN per column
    nan_counts = {}
    for col in critical_columns:
        nan_count = df[col].isna().sum()
        nan_counts[col] = nan_count
        if nan_count > 0:
            logger.warning(f"  Column '{col}': {nan_count} NaN values ({100*nan_count/len(df):.2f}%)")
    
    # Create combined mask: True if ALL critical columns are valid (not NaN)
    valid_mask = pd.Series(True, index=df.index)
    for col in critical_columns:
        valid_mask &= df[col].notna()
    
    # Filter DataFrame
    valid_df = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()
    
    n_removed = len(df) - len(valid_df)
    logger.info(f"  Removed {n_removed} rows with NaN values ({100*n_removed/len(df):.2f}%)")
    logger.info(f"  Valid rows: {len(valid_df)}")
    
    # Create NaN report
    nan_report = pd.DataFrame({
        'column': critical_columns,
        'nan_count': [nan_counts[col] for col in critical_columns],
        'nan_percent': [100 * nan_counts[col] / len(df) for col in critical_columns],
    })
    
    # Log batch-level NaN statistics
    if n_removed > 0:
        invalid_df['batch_id'] = invalid_df.index.map(extract_batch_from_index)
        nan_by_batch = invalid_df.groupby('batch_id').size().sort_values(ascending=False)
        
        logger.info(f"  NaN distribution by batch (top 10):")
        for batch_id, count in nan_by_batch.head(10).items():
            logger.info(f"    {batch_id}: {count} cells with NaN")
        
        if len(nan_by_batch) > 10:
            logger.info(f"    ... and {len(nan_by_batch) - 10} more batches")
    
    return valid_df, nan_report


def extract_batch_from_index(index_value: str) -> str:
    """
    Extract batch (recording) ID from DataFrame index.
    
    Index format: "{batch}_unit_{unit_num}"
    Example: "2024.02.26-10.53.19-Rec_unit_001" -> "2024.02.26-10.53.19-Rec"
    """
    parts = index_value.rsplit('_unit_', 1)
    if len(parts) == 2:
        return parts[0]
    raise ValueError(f"Cannot parse index: {index_value}")


# ===============================================================================
# Index Mapping Reconstruction
# ===============================================================================

def reconstruct_index_mapping(
    input_path: Path = INPUT_PATH,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reconstruct the mapping from positional cell_id to original DataFrame index.
    
    This replicates the filtering and grouping logic from the pipeline to create
    a mapping between positional indices (0, 1, 2...) used in cluster_assignments
    and the original DataFrame index (e.g., "2024.02.26-10.53.19-Rec_unit_001").
    
    Args:
        input_path: Path to input parquet file.
        
    Returns:
        Tuple of:
        - df: Full filtered DataFrame with group assignments
        - group_mappings: Dict mapping group name to DataFrame with columns:
            - 'cell_id': positional index (0, 1, 2...)
            - 'original_index': original DataFrame index string
    """
    logger.info("Reconstructing index mapping...")
    
    # Step 1: Load and filter data (same as pipeline)
    df, reject_reasons = load_and_filter_data(input_path)
    logger.info(f"After filtering: {len(df)} cells")
    
    # Step 2: Assign groups (same as pipeline)
    df = assign_groups(df)
    
    # Step 3: Create mapping for each group
    group_mappings = {}
    
    for group_name in config.GROUP_NAMES:
        # Filter to group (preserving original index)
        group_df = df[df['group'] == group_name].copy()
        
        # Create mapping DataFrame
        mapping = pd.DataFrame({
            'cell_id': range(len(group_df)),
            'original_index': group_df.index.values,
        })
        
        group_mappings[group_name] = mapping
        logger.info(f"  {group_name}: {len(mapping)} cells")
    
    return df, group_mappings


# ===============================================================================
# Total Area Calculation
# ===============================================================================

def extract_batch_from_index(index_value: str) -> str:
    """
    Extract batch (recording) ID from DataFrame index.
    
    Index format: "{batch}_unit_{unit_num}"
    Example: "2024.02.26-10.53.19-Rec_unit_001" -> "2024.02.26-10.53.19-Rec"
    """
    parts = index_value.rsplit('_unit_', 1)
    if len(parts) == 2:
        return parts[0]
    raise ValueError(f"Cannot parse index: {index_value}")


def calculate_total_area(
    df: pd.DataFrame,
    area_col: str = "chip_effective_area",
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculate total effective area from all batches.
    
    Validates that all units within a batch have the same chip_effective_area.
    
    Args:
        df: DataFrame with chip_effective_area column.
        area_col: Column name for chip effective area.
        
    Returns:
        Tuple of:
        - total_area_px2: Total area in pixel^2
        - total_area_mm2: Total area in mm^2
        - batch_areas: Dict mapping batch_id to area in mm^2
        
    Raises:
        ValueError: If units in same batch have different chip_effective_area.
    """
    logger.info("Calculating total effective area...")
    
    # Extract batch for each row
    df = df.copy()
    df['batch_id'] = df.index.map(extract_batch_from_index)
    
    # Group by batch and validate consistency
    batch_areas = {}
    
    for batch_id, batch_df in df.groupby('batch_id'):
        unique_areas = batch_df[area_col].unique()
        
        if len(unique_areas) > 1:
            raise ValueError(
                f"Batch {batch_id} has inconsistent chip_effective_area values: {unique_areas}"
            )
        
        area_px2 = unique_areas[0]
        area_mm2 = area_px2 / (ELECTRODE_PIXELS_PER_MM ** 2)
        batch_areas[batch_id] = area_mm2
    
    # Sum total area (one per batch)
    total_area_px2 = sum(
        df.groupby('batch_id')[area_col].first()
    )
    total_area_mm2 = total_area_px2 / (ELECTRODE_PIXELS_PER_MM ** 2)
    
    logger.info(f"  Total batches: {len(batch_areas)}")
    logger.info(f"  Total area: {total_area_px2:.2f} px^2 = {total_area_mm2:.4f} mm^2")
    
    return total_area_px2, total_area_mm2, batch_areas


# ===============================================================================
# Conversion Factor Calculation
# ===============================================================================

def calculate_conversion_factor(
    df: pd.DataFrame,
    total_area_mm2: float,
    qi_threshold: float = 0.5,
    qi_col: str = "step_up_QI",
) -> Tuple[float, int]:
    """
    Calculate the conversion factor for RF coverage normalization.
    
    conversion_factor = total_good_cell_count / (total_area_mm2 * NORMAL_RGC_PER_MM2)
    
    Args:
        df: DataFrame with QI column.
        total_area_mm2: Total effective area in mm^2.
        qi_threshold: Threshold for "good" cells (step_up_QI > threshold).
        qi_col: Column name for quality index.
        
    Returns:
        Tuple of (conversion_factor, total_good_cell_count)
    """
    logger.info("Calculating conversion factor...")
    
    # Count good cells (step_up_QI > threshold)
    good_cell_mask = df[qi_col] > qi_threshold
    total_good_cell_count = good_cell_mask.sum()
    
    # Check: df should already be filtered to good cells
    if total_good_cell_count != len(df):
        logger.warning(
            f"total_good_cell_count ({total_good_cell_count}) != df length ({len(df)}). "
            f"The DataFrame may not be pre-filtered by QI threshold."
        )
    
    # Expected cell count based on area and density
    expected_cells = total_area_mm2 * NORMAL_RGC_PER_MM2
    
    # Conversion factor
    conversion_factor = total_good_cell_count / expected_cells
    
    logger.info(f"  Good cells (QI > {qi_threshold}): {total_good_cell_count}")
    logger.info(f"  Expected cells (density-based): {expected_cells:.1f}")
    logger.info(f"  Conversion factor: {conversion_factor:.4f}")
    
    return conversion_factor, total_good_cell_count


# ===============================================================================
# RF Area Calculation
# ===============================================================================

def calculate_rf_area(
    sigma_x_px: float,
    sigma_y_px: float,
) -> float:
    """
    Calculate receptive field area from Gaussian sigma parameters.
    
    Uses FWHM (Full Width at Half Maximum) as the effective RF diameter:
        FWHM = 2.355 * sigma
        Area = pi * (FWHM_x / 2) * (FWHM_y / 2)
    
    Args:
        sigma_x_px: Gaussian sigma X in pixels.
        sigma_y_px: Gaussian sigma Y in pixels.
        
    Returns:
        RF area in mm^2.
    """
    # Convert sigma from pixels to mm
    sigma_x_mm = sigma_x_px / NOISE_PIXELS_PER_MM
    sigma_y_mm = sigma_y_px / NOISE_PIXELS_PER_MM
    
    # Ellipse area using sigma as semi-axes (radii)
    # For a 2D Gaussian, sigma represents the standard deviation
    # The RF area is the ellipse with semi-axes = sigma_x and sigma_y
    rf_area = math.pi * sigma_x_mm * sigma_y_mm
    
    return rf_area


def calculate_rf_areas_for_cells(
    df: pd.DataFrame,
    sigma_x_col: str = "gaussian_sigma_x",
    sigma_y_col: str = "gaussian_sigma_y",
) -> pd.Series:
    """
    Calculate RF area for all cells in DataFrame.
    
    Args:
        df: DataFrame with sigma columns.
        sigma_x_col: Column name for sigma X.
        sigma_y_col: Column name for sigma Y.
        
    Returns:
        Series of RF areas in mm^2, indexed by original DataFrame index.
    """
    rf_areas = df.apply(
        lambda row: calculate_rf_area(row[sigma_x_col], row[sigma_y_col]),
        axis=1
    )
    return rf_areas


# ===============================================================================
# Coverage Calculation
# ===============================================================================

def calculate_subtype_coverage(
    df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    index_mapping: pd.DataFrame,
    group_name: str,
    conversion_factor: float,
    total_area_mm2: float,
) -> pd.DataFrame:
    results = []
    clusters = cluster_assignments['dec_cluster'].unique()
    
    for cluster_id in sorted(clusters):
        cluster_mask = cluster_assignments['dec_cluster'] == cluster_id
        cell_ids = cluster_assignments.loc[cluster_mask, 'cell_id'].values
        
        original_indices = index_mapping[
            index_mapping['cell_id'].isin(cell_ids)
        ]['original_index'].values
        
        cells_df = df.loc[original_indices]
        rf_areas = calculate_rf_areas_for_cells(cells_df)
        
        n_cells = len(cells_df)
        median_rf_area_mm2 = rf_areas.median()
        # Calculate total RF area as median x cell count (robust to outliers)
        total_rf_area_mm2 = median_rf_area_mm2 * n_cells
        
        rf_coverage = (total_rf_area_mm2 / conversion_factor) / total_area_mm2
        mosaic_valid = rf_coverage > RF_COVERAGE_THRESHOLD
        subtype_name = f"{group_name}_{cluster_id}"
        
        results.append({
            'group': group_name,
            'dec_cluster': cluster_id,
            'subtype_name': subtype_name,
            'n_cells': n_cells,
            'median_rf_area_mm2': median_rf_area_mm2,
            'total_rf_area_mm2': total_rf_area_mm2,
            'rf_coverage': rf_coverage,
            'mosaic_validation': mosaic_valid,
        })
    
    return pd.DataFrame(results)


# ===============================================================================
# Visualization
# ===============================================================================

def plot_rf_coverage_bar_chart(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df = results_df.sort_values(['group', 'rf_coverage'], ascending=[True, False])
    
    colors = ['#4CAF50' if valid else '#F44336' 
              for valid in df['mosaic_validation']]
    
    x = range(len(df))
    bars = ax.bar(x, df['rf_coverage'], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=RF_COVERAGE_THRESHOLD, color='navy', linestyle='--', 
               linewidth=2, label=f'Threshold = {RF_COVERAGE_THRESHOLD}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['subtype_name'], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Subtype', fontsize=12)
    ax.set_ylabel('RF Coverage', fontsize=12)
    ax.set_title('Receptive Field Coverage by RGC Subtype\n(Green = Valid, Red = Invalid)', fontsize=14)
    ax.legend(loc='upper right')
    
    for i, (bar, coverage) in enumerate(zip(bars, df['rf_coverage'])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{coverage:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    groups = df['group'].unique()
    group_boundaries = []
    for group in groups:
        group_df = df[df['group'] == group]
        if len(group_df) > 0:
            last_idx = df[df['group'] == group].index[-1]
            pos = list(df.index).index(last_idx)
            group_boundaries.append(pos + 0.5)
    
    for boundary in group_boundaries[:-1]:
        ax.axvline(x=boundary, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved RF coverage bar chart to {output_path}")


def plot_validation_heatmap(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    groups = config.GROUP_NAMES
    max_clusters = results_df.groupby('group')['dec_cluster'].max().max() + 1
    
    matrix = np.full((len(groups), int(max_clusters)), np.nan)
    
    for _, row in results_df.iterrows():
        group_idx = groups.index(row['group'])
        cluster_idx = int(row['dec_cluster'])
        matrix[group_idx, cluster_idx] = 1 if row['mosaic_validation'] else 0
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#F44336', '#4CAF50'])
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    for i in range(len(groups)):
        for j in range(int(max_clusters)):
            if not np.isnan(matrix[i, j]):
                text = 'Pass' if matrix[i, j] == 1 else 'Fail'
                color = 'white'
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color=color)
    
    ax.set_xticks(range(int(max_clusters)))
    ax.set_xticklabels([str(i) for i in range(int(max_clusters))])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Group', fontsize=12)
    ax.set_title(f'Mosaic Validation Results (Threshold = {RF_COVERAGE_THRESHOLD})', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Fail', 'Pass'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved validation heatmap to {output_path}")


# ===============================================================================
# Main Pipeline
# ===============================================================================

def run_mosaic_validation(
    input_path: Path = INPUT_PATH,
    results_dir: Path = RESULTS_DIR,
    output_results_dir: Path = OUTPUT_RESULTS_DIR,
    output_figures_dir: Path = OUTPUT_FIGURES_DIR,
) -> pd.DataFrame:
    logger.info("=" * 80)
    logger.info("MOSAIC VALIDATION PIPELINE")
    logger.info("=" * 80)
    
    output_results_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("-" * 40)
    logger.info("Step 1: Reconstructing index mapping")
    logger.info("-" * 40)
    df, group_mappings = reconstruct_index_mapping(input_path)
    
    logger.info("-" * 40)
    logger.info("Step 2: Validating DataFrame for NaN values")
    logger.info("-" * 40)
    df, nan_report = validate_dataframe(df, CRITICAL_COLUMNS)
    
    # Save NaN report
    nan_report_path = output_results_dir / "nan_validation_report.csv"
    nan_report.to_csv(nan_report_path, index=False)
    logger.info(f"  Saved NaN report to: {nan_report_path}")
    
    # Update group mappings to only include valid cells
    logger.info("  Updating group mappings with validated indices...")
    for group_name in config.GROUP_NAMES:
        if group_name in group_mappings:
            # Keep only mappings where original_index is still in validated df
            valid_indices = set(df.index)
            old_mapping = group_mappings[group_name]
            new_mapping = old_mapping[old_mapping['original_index'].isin(valid_indices)].copy()
            # Reset cell_id to be sequential again
            new_mapping = new_mapping.reset_index(drop=True)
            new_mapping['cell_id'] = range(len(new_mapping))
            group_mappings[group_name] = new_mapping
            logger.info(f"    {group_name}: {len(old_mapping)} -> {len(new_mapping)} cells")
    
    logger.info("-" * 40)
    logger.info("Step 3: Calculating total area")
    logger.info("-" * 40)
    total_area_px2, total_area_mm2, batch_areas = calculate_total_area(df)

    
    logger.info("-" * 40)
    logger.info("Step 4: Calculating conversion factor")
    logger.info("-" * 40)
    conversion_factor, total_good_cell_count = calculate_conversion_factor(
        df, total_area_mm2
    )
    
    logger.info("-" * 40)
    logger.info("Step 5: Calculating RF coverage per subtype")
    logger.info("-" * 40)
    
    all_results = []
    
    for group_name in config.GROUP_NAMES:
        logger.info(f"Processing group: {group_name}")
        
        cluster_path = results_dir / group_name / "cluster_assignments.parquet"
        
        if not cluster_path.exists():
            logger.warning(f"  Cluster assignments not found: {cluster_path}")
            continue
        
        cluster_assignments = pd.read_parquet(cluster_path)
        index_mapping = group_mappings[group_name]
        
        group_results = calculate_subtype_coverage(
            df=df,
            cluster_assignments=cluster_assignments,
            index_mapping=index_mapping,
            group_name=group_name,
            conversion_factor=conversion_factor,
            total_area_mm2=total_area_mm2,
        )
        
        all_results.append(group_results)
        
        n_valid = group_results['mosaic_validation'].sum()
        n_total = len(group_results)
        logger.info(f"  {group_name}: {n_valid}/{n_total} subtypes pass validation")
    
    results_df = pd.concat(all_results, ignore_index=True)

    
    logger.info("-" * 40)
    logger.info("Step 6: Saving results")
    logger.info("-" * 40)
    
    parquet_path = output_results_dir / "mosaic_validation_results.parquet"
    results_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved: {parquet_path}")
    
    csv_path = output_results_dir / "mosaic_summary.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")
    
    logger.info("-" * 40)
    logger.info("Step 7: Generating visualizations")
    logger.info("-" * 40)
    
    plot_rf_coverage_bar_chart(
        results_df,
        output_figures_dir / "rf_coverage_by_subtype.png"
    )
    
    plot_validation_heatmap(
        results_df,
        output_figures_dir / "validation_heatmap.png"
    )

    
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total batches: {len(batch_areas)}")
    logger.info(f"Total area: {total_area_mm2:.4f} mm^2")
    logger.info(f"Total good cells: {total_good_cell_count}")
    logger.info(f"Conversion factor: {conversion_factor:.4f}")
    logger.info(f"RF coverage threshold: {RF_COVERAGE_THRESHOLD}")
    logger.info("")
    
    for group_name in config.GROUP_NAMES:
        group_df = results_df[results_df['group'] == group_name]
        if len(group_df) > 0:
            n_valid = group_df['mosaic_validation'].sum()
            n_total = len(group_df)
            logger.info(f"{group_name}: {n_valid}/{n_total} subtypes valid")
    
    total_valid = results_df['mosaic_validation'].sum()
    total_subtypes = len(results_df)
    logger.info("")
    logger.info(f"TOTAL: {total_valid}/{total_subtypes} subtypes pass mosaic validation")
    logger.info("=" * 80)
    
    return results_df


if __name__ == "__main__":
    results = run_mosaic_validation()
    print("\nResults DataFrame:")
    print(results.to_string())
