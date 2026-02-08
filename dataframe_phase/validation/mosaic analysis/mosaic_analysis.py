"""
Mosaic Analysis Visualization.

Generates plots from mosaic validation results to assess RF coverage
for RGC subtypes.

Usage:
    python mosaic_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# ===============================================================================
# Configuration
# ===============================================================================

# Paths
VALIDATION_RESULTS_DIR = Path(__file__).parent.parent.parent / \
    "classification_v2/divide_conquer_method/validation/result"
OUTPUT_DIR = Path(__file__).parent / "output"

# Constants (matching mosaic_validation.py)
RF_COVERAGE_THRESHOLD = 0.8
GROUP_NAMES = ["DSGC", "OSGC", "Other"]


# ===============================================================================
# Data Loading
# ===============================================================================

def load_results(results_path: Optional[Path] = None) -> pd.DataFrame:
    """Load mosaic validation results."""
    if results_path is None:
        results_path = VALIDATION_RESULTS_DIR / "mosaic_validation_results.parquet"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    df = pd.read_parquet(results_path)
    print(f"Loaded {len(df)} subtype results from {results_path}")
    return df


# ===============================================================================
# Visualization Functions
# ===============================================================================

def plot_rf_coverage_bar_chart(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Bar chart of RF coverage by subtype.
    
    Green = passes validation, Red = fails validation.
    """
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
    
    # Add value labels on bars
    for i, (bar, coverage) in enumerate(zip(bars, df['rf_coverage'])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{coverage:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Add group separation lines
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
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_validation_heatmap(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Heatmap showing pass/fail status for each group and cluster.
    """
    from matplotlib.colors import ListedColormap
    
    groups = GROUP_NAMES
    max_clusters = int(results_df.groupby('group')['dec_cluster'].max().max()) + 1
    
    matrix = np.full((len(groups), max_clusters), np.nan)
    
    for _, row in results_df.iterrows():
        if row['group'] in groups:
            group_idx = groups.index(row['group'])
            cluster_idx = int(row['dec_cluster'])
            matrix[group_idx, cluster_idx] = 1 if row['mosaic_validation'] else 0
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    cmap = ListedColormap(['#F44336', '#4CAF50'])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(groups)):
        for j in range(max_clusters):
            if not np.isnan(matrix[i, j]):
                text = 'Pass' if matrix[i, j] == 1 else 'Fail'
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
    
    ax.set_xticks(range(max_clusters))
    ax.set_xticklabels([str(i) for i in range(max_clusters)])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Group', fontsize=12)
    ax.set_title(f'Mosaic Validation Results (Threshold = {RF_COVERAGE_THRESHOLD})', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Fail', 'Pass'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_coverage_distribution(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Histogram of RF coverage values across all subtypes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coverages = results_df['rf_coverage']
    
    # Create histogram
    bins = np.arange(0, coverages.max() + 0.5, 0.5)
    ax.hist(coverages, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=RF_COVERAGE_THRESHOLD, color='red', linestyle='--', 
               linewidth=2, label=f'Threshold = {RF_COVERAGE_THRESHOLD}')
    
    # Add statistics
    ax.axvline(x=coverages.mean(), color='green', linestyle='-', 
               linewidth=2, alpha=0.7, label=f'Mean = {coverages.mean():.2f}')
    ax.axvline(x=coverages.median(), color='orange', linestyle='-', 
               linewidth=2, alpha=0.7, label=f'Median = {coverages.median():.2f}')
    
    ax.set_xlabel('RF Coverage', fontsize=12)
    ax.set_ylabel('Number of Subtypes', fontsize=12)
    ax.set_title('Distribution of RF Coverage Across Subtypes', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_coverage_by_group(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Box plot of RF coverage by group.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    groups = [g for g in GROUP_NAMES if g in results_df['group'].values]
    data = [results_df[results_df['group'] == g]['rf_coverage'].values for g in groups]
    
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    
    colors = ['#2196F3', '#FF9800', '#9C27B0']
    for patch, color in zip(bp['boxes'], colors[:len(groups)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=RF_COVERAGE_THRESHOLD, color='red', linestyle='--', 
               linewidth=2, label=f'Threshold = {RF_COVERAGE_THRESHOLD}')
    
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('RF Coverage', fontsize=12)
    ax.set_title('RF Coverage Distribution by Group', fontsize=14)
    ax.legend()
    
    # Add count annotations
    for i, (g, d) in enumerate(zip(groups, data)):
        n_pass = (d > RF_COVERAGE_THRESHOLD).sum()
        n_total = len(d)
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'{n_pass}/{n_total}',
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_cell_count_vs_coverage(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Scatter plot of cell count vs RF coverage for each subtype.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    group_colors = {'DSGC': '#2196F3', 'OSGC': '#FF9800', 'Other': '#9C27B0'}
    
    for group in GROUP_NAMES:
        group_df = results_df[results_df['group'] == group]
        if len(group_df) == 0:
            continue
        
        colors = ['green' if v else 'red' for v in group_df['mosaic_validation']]
        ax.scatter(group_df['n_cells'], group_df['rf_coverage'], 
                  c=colors, label=group, alpha=0.7, s=60,
                  edgecolors=group_colors.get(group, 'black'), linewidths=2)
    
    ax.axhline(y=RF_COVERAGE_THRESHOLD, color='navy', linestyle='--', 
               linewidth=2, label=f'Threshold = {RF_COVERAGE_THRESHOLD}')
    
    ax.set_xlabel('Number of Cells', fontsize=12)
    ax.set_ylabel('RF Coverage', fontsize=12)
    ax.set_title('Cell Count vs RF Coverage\n(Green = Valid, Red = Invalid, Edge = Group)', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rf_area_summary(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Bar chart of mean RF area per subtype.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate mean RF area per cell
    results_df = results_df.copy()
    results_df['mean_rf_area'] = results_df['total_rf_area_mm2'] / results_df['n_cells']
    
    df = results_df.sort_values(['group', 'mean_rf_area'], ascending=[True, False])
    
    group_colors = {'DSGC': '#2196F3', 'OSGC': '#FF9800', 'Other': '#9C27B0'}
    colors = [group_colors.get(g, 'gray') for g in df['group']]
    
    x = range(len(df))
    bars = ax.bar(x, df['mean_rf_area'] * 1000, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['subtype_name'], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Subtype', fontsize=12)
    ax.set_ylabel('Mean RF Area (x10^-3 mm^2)', fontsize=12)
    ax.set_title('Mean Receptive Field Area by Subtype', fontsize=14)
    
    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ===============================================================================
# Main
# ===============================================================================

def generate_all_plots(
    results_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Generate all mosaic analysis plots."""
    if results_df is None:
        results_df = load_results()
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    print("=" * 60)
    
    plot_rf_coverage_bar_chart(
        results_df, 
        output_dir / "rf_coverage_bar_chart.png",
        show=show
    )
    
    plot_validation_heatmap(
        results_df,
        output_dir / "validation_heatmap.png",
        show=show
    )
    
    plot_coverage_distribution(
        results_df,
        output_dir / "coverage_distribution.png",
        show=show
    )
    
    plot_coverage_by_group(
        results_df,
        output_dir / "coverage_by_group.png",
        show=show
    )
    
    plot_cell_count_vs_coverage(
        results_df,
        output_dir / "cell_count_vs_coverage.png",
        show=show
    )
    
    plot_rf_area_summary(
        results_df,
        output_dir / "rf_area_summary.png",
        show=show
    )
    
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")


def print_summary(results_df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("MOSAIC VALIDATION SUMMARY")
    print("=" * 60)
    
    total_valid = results_df['mosaic_validation'].sum()
    total_subtypes = len(results_df)
    
    print(f"\nTotal subtypes: {total_subtypes}")
    print(f"Passing validation: {total_valid} ({100*total_valid/total_subtypes:.1f}%)")
    print(f"Failing validation: {total_subtypes - total_valid}")
    
    print("\nBy group:")
    for group in GROUP_NAMES:
        group_df = results_df[results_df['group'] == group]
        if len(group_df) == 0:
            continue
        n_valid = group_df['mosaic_validation'].sum()
        n_total = len(group_df)
        print(f"  {group}: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    print("\nRF Coverage statistics:")
    coverages = results_df['rf_coverage']
    print(f"  Min: {coverages.min():.3f}")
    print(f"  Max: {coverages.max():.3f}")
    print(f"  Mean: {coverages.mean():.3f}")
    print(f"  Median: {coverages.median():.3f}")
    print(f"  Std: {coverages.std():.3f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Load results
    results = load_results()
    
    # Print summary
    print_summary(results)
    
    # Generate all plots
    generate_all_plots(results, show=False)
