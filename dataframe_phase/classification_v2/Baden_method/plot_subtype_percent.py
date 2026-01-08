"""
Plot the number of cells in each cluster for DS and non-DS populations.

Creates a figure with six subplots showing cluster size distributions,
with stacked bars showing RGC vs AC composition.
- Row 1: sorted by total cell count
- Row 2: sorted by RGC count
- Row 3: sorted by RGC percentage
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataframe_phase.classification_v2.Baden_method import config


def plot_cluster_counts(
    results_path: Path = None,
    input_data_path: Path = None,
    output_path: Path = None,
    figsize: tuple = (18, 18),
):
    """
    Create a figure showing the number of cells in each cluster.
    
    Creates 6 subplots (3 rows × 2 columns):
    - Row 1: sorted by total count (high to low)
    - Row 2: sorted by RGC count (high to low)
    - Row 3: sorted by RGC percentage (high to low)
    
    Each bar shows RGC vs AC composition with different colors.
    
    Args:
        results_path: Path to clustering_results.parquet. Defaults to config path.
        input_data_path: Path to original data with axon_type. Defaults to config path.
        output_path: Path to save the figure. Defaults to plots/ directory.
        figsize: Figure size (width, height).
    """
    # Default paths
    if results_path is None:
        results_path = config.RESULTS_DIR / "clustering_results.parquet"
    if input_data_path is None:
        input_data_path = config.INPUT_PATH
    if output_path is None:
        output_path = config.PLOTS_DIR / "cluster_cell_counts.png"
    
    # Load clustering results
    print(f"Loading clustering results from: {results_path}")
    df_results = pd.read_parquet(results_path)
    
    # Load original data to get axon_type
    print(f"Loading original data from: {input_data_path}")
    df_original = pd.read_parquet(input_data_path)
    
    # Create a mapping from cell_id to axon_type
    # The cell_id in results matches the index in original data
    df_original['cell_id'] = df_original.index
    axon_map = df_original.set_index('cell_id')['axon_type'].to_dict()
    
    # Add axon_type to results
    df_results['axon_type'] = df_results['cell_id'].map(axon_map)
    
    # Separate populations
    df_ds = df_results[df_results['population'] == 'DS'].copy()
    df_nds = df_results[df_results['population'] == 'non-DS'].copy()
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Color palette
    rgc_color = '#2A9D8F'  # Teal for RGC
    ac_color = '#E9C46A'   # Gold for AC
    
    def get_cluster_counts(df_pop):
        """Get cluster counts by axon type."""
        cluster_axon_counts = df_pop.groupby(['cluster_label', 'axon_type']).size().unstack(fill_value=0)
        
        # Ensure both columns exist
        if 'rgc' not in cluster_axon_counts.columns:
            cluster_axon_counts['rgc'] = 0
        if 'ac' not in cluster_axon_counts.columns:
            cluster_axon_counts['ac'] = 0
        
        # Calculate total per cluster and RGC percentage
        cluster_axon_counts['total'] = cluster_axon_counts['rgc'] + cluster_axon_counts['ac']
        cluster_axon_counts['rgc_pct'] = cluster_axon_counts['rgc'] / cluster_axon_counts['total'] * 100
        
        return cluster_axon_counts
    
    def plot_population(ax, cluster_counts, title, sort_by='total'):
        """Plot stacked bar chart for one population."""
        # Sort by specified column (descending)
        sorted_counts = cluster_counts.sort_values(sort_by, ascending=False)
        
        # Get sorted cluster labels for x-axis
        sorted_clusters = sorted_counts.index.tolist()
        x = np.arange(len(sorted_clusters))
        
        # Plot stacked bars
        rgc_counts = sorted_counts['rgc'].values
        ac_counts = sorted_counts['ac'].values
        
        bars_rgc = ax.bar(x, rgc_counts, color=rgc_color, edgecolor='white', linewidth=0.5, label='RGC')
        bars_ac = ax.bar(x, ac_counts, bottom=rgc_counts, color=ac_color, edgecolor='white', linewidth=0.5, label='AC')
        
        # Labels and title
        sort_labels = {'total': 'total count', 'rgc': 'RGC count', 'rgc_pct': 'RGC %'}
        sort_label = sort_labels.get(sort_by, sort_by)
        ax.set_xlabel(f'Cluster (sorted by {sort_label})', fontsize=11)
        ax.set_ylabel('Number of Cells', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # X-axis labels (cluster IDs)
        if len(sorted_clusters) <= 50:
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_clusters, rotation=90, fontsize=6)
        else:
            # Show every other tick for readability
            ax.set_xticks(x[::2])
            ax.set_xticklabels([sorted_clusters[i] for i in range(0, len(sorted_clusters), 2)], rotation=90, fontsize=6)
        
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Summary statistics
        total_rgc = rgc_counts.sum()
        total_ac = ac_counts.sum()
        total_cells = total_rgc + total_ac
        stats_text = (f"Total: {total_cells:,} | RGC: {total_rgc:,} ({100*total_rgc/total_cells:.1f}%) | "
                     f"AC: {total_ac:,} ({100*total_ac/total_cells:.1f}%)")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return sorted_counts
    
    # Get cluster counts for both populations
    ds_counts = get_cluster_counts(df_ds)
    nds_counts = get_cluster_counts(df_nds)
    
    # Row 1: sorted by total
    plot_population(axes[0, 0], ds_counts, 'DS Cells - Sorted by Total Count', sort_by='total')
    plot_population(axes[0, 1], nds_counts, 'Non-DS Cells - Sorted by Total Count', sort_by='total')
    
    # Row 2: sorted by RGC count
    plot_population(axes[1, 0], ds_counts, 'DS Cells - Sorted by RGC Count', sort_by='rgc')
    plot_population(axes[1, 1], nds_counts, 'Non-DS Cells - Sorted by RGC Count', sort_by='rgc')
    
    # Row 3: sorted by RGC percentage
    plot_population(axes[2, 0], ds_counts, 'DS Cells - Sorted by RGC Percentage', sort_by='rgc_pct')
    plot_population(axes[2, 1], nds_counts, 'Non-DS Cells - Sorted by RGC Percentage', sort_by='rgc_pct')
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTER COMPOSITION SUMMARY")
    print("="*60)
    
    print(f"\nDS Population ({len(df_ds):,} cells, {len(ds_counts)} clusters):")
    print(f"  RGC: {ds_counts['rgc'].sum():,} ({100*ds_counts['rgc'].sum()/len(df_ds):.1f}%)")
    print(f"  AC:  {ds_counts['ac'].sum():,} ({100*ds_counts['ac'].sum()/len(df_ds):.1f}%)")
    print(f"  RGC% range: {ds_counts['rgc_pct'].min():.1f}% - {ds_counts['rgc_pct'].max():.1f}%")
    
    print(f"\nNon-DS Population ({len(df_nds):,} cells, {len(nds_counts)} clusters):")
    print(f"  RGC: {nds_counts['rgc'].sum():,} ({100*nds_counts['rgc'].sum()/len(df_nds):.1f}%)")
    print(f"  AC:  {nds_counts['ac'].sum():,} ({100*nds_counts['ac'].sum()/len(df_nds):.1f}%)")
    print(f"  RGC% range: {nds_counts['rgc_pct'].min():.1f}% - {nds_counts['rgc_pct'].max():.1f}%")
    
    return fig


if __name__ == "__main__":
    plot_cluster_counts()
