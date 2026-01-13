"""
Plot the number of cells in each cluster for DS and non-DS populations.

Creates a figure with six subplots showing cluster size distributions,
with stacked bars showing RGC vs AC composition.
- Row 1: sorted by total cell count
- Row 2: sorted by RGC count
- Row 3: sorted by RGC percentage

Also performs statistical classification of clusters as RGC-enriched, AC-enriched,
or mixed using binomial test with FDR correction.
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
from scipy import stats

from dataframe_phase.classification_v2.Baden_method import config


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float = 0.05):
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        p_values: Array of raw p-values.
        alpha: False discovery rate threshold.
        
    Returns:
        Tuple of (reject, p_adjusted) where:
        - reject: Boolean array indicating which hypotheses to reject
        - p_adjusted: FDR-adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    
    # Sort p-values and get original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Calculate adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    
    # Ensure adjusted p-values don't exceed 1
    adjusted = np.minimum(adjusted, 1.0)
    
    # Map back to original order
    p_adjusted = np.zeros(n)
    p_adjusted[sorted_indices] = adjusted
    
    # Determine rejections
    reject = p_adjusted <= alpha
    
    return reject, p_adjusted


def classify_clusters_statistically(
    cluster_counts: pd.DataFrame,
    population_rgc_proportion: float,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Statistically classify clusters as RGC-enriched, AC-enriched, or mixed.
    
    Uses two-sided binomial test to determine if a cluster's RGC proportion
    significantly differs from the overall population proportion.
    
    Args:
        cluster_counts: DataFrame with 'rgc', 'ac', 'total' columns indexed by cluster.
        population_rgc_proportion: Overall RGC proportion in the population (null hypothesis).
        alpha: Significance level after multiple testing correction.
        
    Returns:
        DataFrame with added columns:
        - p_value: Raw binomial test p-value
        - p_adjusted: Benjamini-Hochberg FDR-corrected p-value
        - classification: 'RGC-enriched', 'AC-enriched', or 'Mixed'
        - significant: Boolean indicating statistical significance
    """
    results = cluster_counts.copy()
    
    # Perform binomial test for each cluster
    p_values = []
    for idx in results.index:
        n_total = int(results.loc[idx, 'total'])
        n_rgc = int(results.loc[idx, 'rgc'])
        
        if n_total == 0:
            p_values.append(1.0)
            continue
        
        # Two-sided binomial test: is observed RGC count different from expected?
        # H0: proportion = population_rgc_proportion
        # H1: proportion != population_rgc_proportion
        # Use scipy.stats.binomtest (newer API) or binom_test
        try:
            # Try newer API first (scipy >= 1.7)
            result = stats.binomtest(n_rgc, n_total, population_rgc_proportion, alternative='two-sided')
            p_val = result.pvalue
        except AttributeError:
            # Fall back to older API
            p_val = stats.binom_test(n_rgc, n_total, population_rgc_proportion, alternative='two-sided')
        p_values.append(p_val)
    
    results['p_value'] = p_values
    
    # Multiple testing correction (Benjamini-Hochberg FDR)
    p_array = np.array(p_values)
    reject, p_adjusted = benjamini_hochberg_correction(p_array, alpha=alpha)
    results['p_adjusted'] = p_adjusted
    results['significant'] = reject
    
    # Classify clusters
    def classify(row):
        if not row['significant']:
            return 'Mixed'
        observed_prop = row['rgc'] / row['total'] if row['total'] > 0 else 0
        if observed_prop > population_rgc_proportion:
            return 'RGC-enriched'
        else:
            return 'AC-enriched'
    
    results['classification'] = results.apply(classify, axis=1)
    
    return results


def plot_cluster_counts(
    results_path: Path = None,
    input_data_path: Path = None,
    output_path: Path = None,
    figsize: tuple = (18, 24),
    iprgc_qi_threshold: float = 0.8,
):
    """
    Create a figure showing the number of cells in each cluster.
    
    Creates 8 subplots (4 rows × 2 columns):
    - Row 1: RGC vs AC sorted by total count (high to low)
    - Row 2: RGC vs AC sorted by RGC count (high to low)
    - Row 3: RGC vs AC sorted by RGC percentage (high to low)
    - Row 4: ipRGC vs non-ipRGC sorted by ipRGC percentage (high to low)
    
    Each bar shows composition with different colors.
    
    Args:
        results_path: Path to clustering_results.parquet. Defaults to config path.
        input_data_path: Path to original data with axon_type. Defaults to config path.
        output_path: Path to save the figure. Defaults to plots/ directory.
        figsize: Figure size (width, height).
        iprgc_qi_threshold: Quality index threshold for ipRGC classification (default 0.8).
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
    
    # Load original data to get axon_type and iprgc_qi
    print(f"Loading original data from: {input_data_path}")
    df_original = pd.read_parquet(input_data_path)
    
    # Create mappings from cell_id
    df_original['cell_id'] = df_original.index
    axon_map = df_original.set_index('cell_id')['axon_type'].to_dict()
    
    # ipRGC classification based on iprgc_2hz_QI threshold
    iprgc_qi_col = 'iprgc_2hz_QI'
    if iprgc_qi_col in df_original.columns:
        iprgc_map = (df_original.set_index('cell_id')[iprgc_qi_col] > iprgc_qi_threshold).to_dict()
        print(f"Using {iprgc_qi_col} > {iprgc_qi_threshold} for ipRGC classification")
    else:
        print(f"Warning: {iprgc_qi_col} column not found, skipping ipRGC analysis")
        iprgc_map = None
    
    # Add axon_type and is_iprgc to results
    df_results['axon_type'] = df_results['cell_id'].map(axon_map)
    if iprgc_map is not None:
        df_results['is_iprgc'] = df_results['cell_id'].map(iprgc_map).fillna(False)
    
    # Separate populations
    df_ds = df_results[df_results['population'] == 'DS'].copy()
    df_nds = df_results[df_results['population'] == 'non-DS'].copy()
    
    # Create figure with 4x2 subplots
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    
    # Color palette
    rgc_color = '#2A9D8F'  # Teal for RGC
    ac_color = '#E9C46A'   # Gold for AC
    iprgc_color = '#E76F51'  # Coral for ipRGC
    non_iprgc_color = '#457B9D'  # Steel blue for non-ipRGC
    
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
    
    def get_iprgc_cluster_counts(df_pop):
        """Get cluster counts by ipRGC status."""
        if 'is_iprgc' not in df_pop.columns:
            return None
        
        cluster_iprgc_counts = df_pop.groupby(['cluster_label', 'is_iprgc']).size().unstack(fill_value=0)
        
        # Rename columns for clarity
        cluster_iprgc_counts.columns = ['non_iprgc' if not c else 'iprgc' for c in cluster_iprgc_counts.columns]
        
        # Ensure both columns exist
        if 'iprgc' not in cluster_iprgc_counts.columns:
            cluster_iprgc_counts['iprgc'] = 0
        if 'non_iprgc' not in cluster_iprgc_counts.columns:
            cluster_iprgc_counts['non_iprgc'] = 0
        
        # Calculate total per cluster and ipRGC percentage
        cluster_iprgc_counts['total'] = cluster_iprgc_counts['iprgc'] + cluster_iprgc_counts['non_iprgc']
        cluster_iprgc_counts['iprgc_pct'] = cluster_iprgc_counts['iprgc'] / cluster_iprgc_counts['total'] * 100
        
        return cluster_iprgc_counts
    
    def plot_population(ax, cluster_counts, title, sort_by='total', show_significance=True):
        """Plot stacked bar chart for one population with significance markers."""
        # Sort by specified column (descending)
        sorted_counts = cluster_counts.sort_values(sort_by, ascending=False)
        
        # Get sorted cluster labels for x-axis
        sorted_clusters = sorted_counts.index.tolist()
        x = np.arange(len(sorted_clusters))
        
        # Plot stacked bars
        rgc_counts = sorted_counts['rgc'].values
        ac_counts = sorted_counts['ac'].values
        
        # Color bars by classification if available
        if 'classification' in sorted_counts.columns and show_significance:
            bar_colors_rgc = []
            bar_colors_ac = []
            for idx in sorted_counts.index:
                classification = sorted_counts.loc[idx, 'classification']
                if classification == 'RGC-enriched':
                    bar_colors_rgc.append('#1B7F79')  # Darker teal
                    bar_colors_ac.append('#E9C46A')
                elif classification == 'AC-enriched':
                    bar_colors_rgc.append('#2A9D8F')
                    bar_colors_ac.append('#D4A844')  # Darker gold
                else:
                    bar_colors_rgc.append('#2A9D8F')
                    bar_colors_ac.append('#E9C46A')
            
            # Plot with individual colors
            for i, (r, a) in enumerate(zip(rgc_counts, ac_counts)):
                ax.bar(x[i], r, color=bar_colors_rgc[i], edgecolor='white', linewidth=0.5)
                ax.bar(x[i], a, bottom=r, color=bar_colors_ac[i], edgecolor='white', linewidth=0.5)
            
            # Add significance markers
            for i, idx in enumerate(sorted_counts.index):
                if sorted_counts.loc[idx, 'significant']:
                    total_height = sorted_counts.loc[idx, 'total']
                    marker = '*' if sorted_counts.loc[idx, 'p_adjusted'] < 0.01 else '†'
                    ax.text(x[i], total_height + 1, marker, ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
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
        
        # Legend with classification info
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=rgc_color, edgecolor='white', label='RGC'),
            Patch(facecolor=ac_color, edgecolor='white', label='AC'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Summary statistics
        total_rgc = rgc_counts.sum()
        total_ac = ac_counts.sum()
        total_cells = total_rgc + total_ac
        
        # Add classification counts if available
        if 'classification' in sorted_counts.columns:
            n_rgc_enriched = (sorted_counts['classification'] == 'RGC-enriched').sum()
            n_ac_enriched = (sorted_counts['classification'] == 'AC-enriched').sum()
            n_mixed = (sorted_counts['classification'] == 'Mixed').sum()
            stats_text = (f"Total: {total_cells:,} | RGC: {total_rgc:,} ({100*total_rgc/total_cells:.1f}%)\n"
                         f"Clusters: {n_rgc_enriched} RGC-enriched, {n_ac_enriched} AC-enriched, {n_mixed} Mixed")
        else:
            stats_text = (f"Total: {total_cells:,} | RGC: {total_rgc:,} ({100*total_rgc/total_cells:.1f}%) | "
                         f"AC: {total_ac:,} ({100*total_ac/total_cells:.1f}%)")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return sorted_counts
    
    def plot_iprgc_population(ax, cluster_counts, title, show_significance=True):
        """Plot stacked bar chart for ipRGC vs non-ipRGC composition."""
        if cluster_counts is None:
            ax.text(0.5, 0.5, 'ipRGC data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            return None
        
        # Sort by ipRGC percentage (descending)
        sorted_counts = cluster_counts.sort_values('iprgc_pct', ascending=False)
        
        # Get sorted cluster labels for x-axis
        sorted_clusters = sorted_counts.index.tolist()
        x = np.arange(len(sorted_clusters))
        
        # Plot stacked bars
        iprgc_counts = sorted_counts['iprgc'].values
        non_iprgc_counts = sorted_counts['non_iprgc'].values
        
        # Color bars by classification if available
        if 'classification' in sorted_counts.columns and show_significance:
            for i in range(len(x)):
                classification = sorted_counts.iloc[i]['classification']
                if classification == 'ipRGC-enriched':
                    color_ip = '#C44536'  # Darker coral
                    color_non = '#457B9D'
                elif classification == 'non-ipRGC-enriched':
                    color_ip = '#E76F51'
                    color_non = '#2B5F82'  # Darker blue
                else:
                    color_ip = '#E76F51'
                    color_non = '#457B9D'
                
                ax.bar(x[i], iprgc_counts[i], color=color_ip, edgecolor='white', linewidth=0.5)
                ax.bar(x[i], non_iprgc_counts[i], bottom=iprgc_counts[i], color=color_non, edgecolor='white', linewidth=0.5)
            
            # Add significance markers
            for i, idx in enumerate(sorted_counts.index):
                if sorted_counts.loc[idx, 'significant']:
                    total_height = sorted_counts.loc[idx, 'total']
                    marker = '*' if sorted_counts.loc[idx, 'p_adjusted'] < 0.01 else '+'
                    ax.text(x[i], total_height + 1, marker, ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.bar(x, iprgc_counts, color=iprgc_color, edgecolor='white', linewidth=0.5, label='ipRGC')
            ax.bar(x, non_iprgc_counts, bottom=iprgc_counts, color=non_iprgc_color, edgecolor='white', linewidth=0.5, label='non-ipRGC')
        
        # Labels and title
        ax.set_xlabel('Cluster (sorted by ipRGC %)', fontsize=11)
        ax.set_ylabel('Number of Cells', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # X-axis labels
        if len(sorted_clusters) <= 50:
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_clusters, rotation=90, fontsize=6)
        else:
            ax.set_xticks(x[::2])
            ax.set_xticklabels([sorted_clusters[i] for i in range(0, len(sorted_clusters), 2)], rotation=90, fontsize=6)
        
        ax.grid(axis='y', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=iprgc_color, edgecolor='white', label='ipRGC'),
            Patch(facecolor=non_iprgc_color, edgecolor='white', label='non-ipRGC'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Summary statistics
        total_iprgc = iprgc_counts.sum()
        total_non = non_iprgc_counts.sum()
        total_cells = total_iprgc + total_non
        
        if 'classification' in sorted_counts.columns:
            n_iprgc_enriched = (sorted_counts['classification'] == 'ipRGC-enriched').sum()
            n_non_enriched = (sorted_counts['classification'] == 'non-ipRGC-enriched').sum()
            n_mixed = (sorted_counts['classification'] == 'Mixed').sum()
            stats_text = (f"Total: {total_cells:,} | ipRGC: {total_iprgc:,} ({100*total_iprgc/total_cells:.1f}%)\n"
                         f"Clusters: {n_iprgc_enriched} ipRGC-enriched, {n_non_enriched} non-ipRGC-enriched, {n_mixed} Mixed")
        else:
            stats_text = (f"Total: {total_cells:,} | ipRGC: {total_iprgc:,} ({100*total_iprgc/total_cells:.1f}%)")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return sorted_counts
    
    # Get cluster counts for both populations
    ds_counts = get_cluster_counts(df_ds)
    nds_counts = get_cluster_counts(df_nds)
    
    # Calculate overall RGC proportion for each population
    ds_rgc_prop = ds_counts['rgc'].sum() / ds_counts['total'].sum()
    nds_rgc_prop = nds_counts['rgc'].sum() / nds_counts['total'].sum()
    
    print(f"\nDS population RGC proportion: {ds_rgc_prop:.3f}")
    print(f"Non-DS population RGC proportion: {nds_rgc_prop:.3f}")
    
    # Perform statistical classification
    print("\nPerforming statistical classification (binomial test, FDR correction)...")
    ds_counts = classify_clusters_statistically(ds_counts, ds_rgc_prop)
    nds_counts = classify_clusters_statistically(nds_counts, nds_rgc_prop)
    
    # Row 1: sorted by total
    plot_population(axes[0, 0], ds_counts, 'DS Cells - Sorted by Total Count', sort_by='total')
    plot_population(axes[0, 1], nds_counts, 'Non-DS Cells - Sorted by Total Count', sort_by='total')
    
    # Row 2: sorted by RGC count
    plot_population(axes[1, 0], ds_counts, 'DS Cells - Sorted by RGC Count', sort_by='rgc')
    plot_population(axes[1, 1], nds_counts, 'Non-DS Cells - Sorted by RGC Count', sort_by='rgc')
    
    # Row 3: sorted by RGC percentage
    plot_population(axes[2, 0], ds_counts, 'DS Cells - Sorted by RGC Percentage', sort_by='rgc_pct')
    plot_population(axes[2, 1], nds_counts, 'Non-DS Cells - Sorted by RGC Percentage', sort_by='rgc_pct')
    
    # Row 4: ipRGC vs non-ipRGC analysis
    ds_iprgc_counts = get_iprgc_cluster_counts(df_ds)
    nds_iprgc_counts = get_iprgc_cluster_counts(df_nds)
    
    if ds_iprgc_counts is not None:
        # Calculate overall ipRGC proportion for each population
        ds_iprgc_prop = ds_iprgc_counts['iprgc'].sum() / ds_iprgc_counts['total'].sum()
        nds_iprgc_prop = nds_iprgc_counts['iprgc'].sum() / nds_iprgc_counts['total'].sum()
        
        print(f"\nDS population ipRGC proportion (QI>{iprgc_qi_threshold}): {ds_iprgc_prop:.3f}")
        print(f"Non-DS population ipRGC proportion (QI>{iprgc_qi_threshold}): {nds_iprgc_prop:.3f}")
        
        # Perform statistical classification for ipRGC
        print("\nPerforming ipRGC statistical classification...")
        ds_iprgc_counts = classify_clusters_statistically(
            ds_iprgc_counts.rename(columns={'iprgc': 'rgc', 'non_iprgc': 'ac'}), 
            ds_iprgc_prop
        )
        # Rename back and update classification labels
        ds_iprgc_counts = ds_iprgc_counts.rename(columns={'rgc': 'iprgc', 'ac': 'non_iprgc'})
        ds_iprgc_counts['classification'] = ds_iprgc_counts['classification'].replace({
            'RGC-enriched': 'ipRGC-enriched', 
            'AC-enriched': 'non-ipRGC-enriched'
        })
        
        nds_iprgc_counts = classify_clusters_statistically(
            nds_iprgc_counts.rename(columns={'iprgc': 'rgc', 'non_iprgc': 'ac'}), 
            nds_iprgc_prop
        )
        nds_iprgc_counts = nds_iprgc_counts.rename(columns={'rgc': 'iprgc', 'ac': 'non_iprgc'})
        nds_iprgc_counts['classification'] = nds_iprgc_counts['classification'].replace({
            'RGC-enriched': 'ipRGC-enriched', 
            'AC-enriched': 'non-ipRGC-enriched'
        })
    
    plot_iprgc_population(axes[3, 0], ds_iprgc_counts, f'DS Cells - ipRGC (QI>{iprgc_qi_threshold}) vs non-ipRGC')
    plot_iprgc_population(axes[3, 1], nds_iprgc_counts, f'Non-DS Cells - ipRGC (QI>{iprgc_qi_threshold}) vs non-ipRGC')
    
    # Add figure-level annotation about significance markers
    fig.text(0.5, 0.01, '* p < 0.01, + p < 0.05 (binomial test, FDR corrected)', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("CLUSTER COMPOSITION AND STATISTICAL CLASSIFICATION SUMMARY")
    print("="*70)
    
    def print_population_summary(name, df_pop, counts):
        n_rgc_enriched = (counts['classification'] == 'RGC-enriched').sum()
        n_ac_enriched = (counts['classification'] == 'AC-enriched').sum()
        n_mixed = (counts['classification'] == 'Mixed').sum()
        
        print(f"\n{name} Population ({len(df_pop):,} cells, {len(counts)} clusters):")
        print(f"  Cell composition:")
        print(f"    RGC: {counts['rgc'].sum():,} ({100*counts['rgc'].sum()/len(df_pop):.1f}%)")
        print(f"    AC:  {counts['ac'].sum():,} ({100*counts['ac'].sum()/len(df_pop):.1f}%)")
        print(f"  RGC% range across clusters: {counts['rgc_pct'].min():.1f}% - {counts['rgc_pct'].max():.1f}%")
        print(f"\n  Statistical classification (binomial test, FDR alpha=0.05):")
        print(f"    RGC-enriched clusters: {n_rgc_enriched} ({100*n_rgc_enriched/len(counts):.1f}%)")
        print(f"    AC-enriched clusters:  {n_ac_enriched} ({100*n_ac_enriched/len(counts):.1f}%)")
        print(f"    Mixed clusters:        {n_mixed} ({100*n_mixed/len(counts):.1f}%)")
        
        # Show details for enriched clusters
        if n_rgc_enriched > 0:
            rgc_enriched = counts[counts['classification'] == 'RGC-enriched'].sort_values('rgc_pct', ascending=False)
            print(f"\n  RGC-enriched clusters:")
            for idx in rgc_enriched.index[:10]:  # Show top 10
                row = rgc_enriched.loc[idx]
                print(f"    Cluster {idx}: {row['rgc']:.0f} RGC / {row['total']:.0f} total "
                      f"({row['rgc_pct']:.1f}%), p={row['p_adjusted']:.4f}")
            if len(rgc_enriched) > 10:
                print(f"    ... and {len(rgc_enriched) - 10} more")
        
        if n_ac_enriched > 0:
            ac_enriched = counts[counts['classification'] == 'AC-enriched'].sort_values('rgc_pct', ascending=True)
            print(f"\n  AC-enriched clusters:")
            for idx in ac_enriched.index[:10]:  # Show top 10
                row = ac_enriched.loc[idx]
                ac_pct = 100 - row['rgc_pct']
                print(f"    Cluster {idx}: {row['ac']:.0f} AC / {row['total']:.0f} total "
                      f"({ac_pct:.1f}% AC), p={row['p_adjusted']:.4f}")
            if len(ac_enriched) > 10:
                print(f"    ... and {len(ac_enriched) - 10} more")
    
    print_population_summary("DS", df_ds, ds_counts)
    print_population_summary("Non-DS", df_nds, nds_counts)
    
    # Save classification results to CSV
    classification_output = config.RESULTS_DIR / "cluster_classification.csv"
    all_results = pd.concat([
        ds_counts.assign(population='DS'),
        nds_counts.assign(population='non-DS')
    ])
    all_results.to_csv(classification_output)
    print(f"\n\nSaved classification results to: {classification_output}")
    
    return fig, ds_counts, nds_counts


if __name__ == "__main__":
    plot_cluster_counts()
