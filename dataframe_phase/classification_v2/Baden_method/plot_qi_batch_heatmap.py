"""
Plot heatmap showing distribution of batch sizes across different QI thresholds.

Creates a heatmap where:
- Rows: different step_up_QI thresholds
- Columns: ranges of good cells per batch
- Values: number of batches in each cell count range
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
import seaborn as sns
import pandas as pd
import numpy as np

from dataframe_phase.classification_v2.Baden_method import config


def get_batch(idx: str) -> str:
    """Extract batch name from index (filename without unit ID)."""
    s = str(idx)
    parts = s.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return s


def plot_qi_batch_heatmap(
    input_path: Path = None,
    output_path: Path = None,
    qi_thresholds: list = None,
    bin_edges: list = None,
    figsize: tuple = (14, 8),
):
    """
    Create heatmap showing batch size distribution across QI thresholds.
    
    Args:
        input_path: Path to input parquet file. Defaults to config path.
        output_path: Path to save the figure. Defaults to plots/ directory.
        qi_thresholds: List of QI threshold values. Defaults to 0.1 to 0.9.
        bin_edges: Bin edges for cell count ranges. Defaults to standard bins.
        figsize: Figure size (width, height).
    """
    # Default paths
    if input_path is None:
        input_path = config.INPUT_PATH
    if output_path is None:
        output_path = config.PLOTS_DIR / "qi_threshold_batch_heatmap.png"
    
    # Default QI thresholds
    if qi_thresholds is None:
        qi_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Default bin edges
    if bin_edges is None:
        bin_edges = [0, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400]
    
    # Create bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            bin_labels.append(f'{bin_edges[i]}+')
        else:
            bin_labels.append(f'{bin_edges[i]}-{bin_edges[i+1]}')
    
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Extract batch from index
    df['batch'] = df.index.map(get_batch)
    
    # Build heatmap data
    heatmap_data = []  # Number of batches
    cell_count_data = []  # Total cells in those batches
    summary_data = []
    
    for qi_thresh in qi_thresholds:
        # Filter by QI threshold
        df_good = df[df['step_up_QI'] > qi_thresh]
        
        # Count cells per batch
        batch_counts = df_good.groupby('batch').size()
        
        # Bin the counts
        binned = pd.cut(batch_counts, bins=bin_edges, labels=bin_labels, right=True)
        bin_counts = binned.value_counts().reindex(bin_labels, fill_value=0)
        
        # Calculate total cells in each bin
        cells_per_bin = []
        for label in bin_labels:
            batches_in_bin = batch_counts[binned == label]
            cells_per_bin.append(batches_in_bin.sum())
        
        heatmap_data.append(bin_counts.values)
        cell_count_data.append(cells_per_bin)
        summary_data.append({
            'threshold': qi_thresh,
            'total_cells': len(df_good),
            'batches_with_cells': len(batch_counts),
            'mean_cells_per_batch': batch_counts.mean() if len(batch_counts) > 0 else 0,
            'median_cells_per_batch': batch_counts.median() if len(batch_counts) > 0 else 0,
        })
        
        print(f"QI > {qi_thresh}: {len(df_good):,} cells, {len(batch_counts)} batches")
    
    # Create DataFrames for heatmap
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[f'QI > {t}' for t in qi_thresholds],
        columns=bin_labels
    )
    
    cell_count_df = pd.DataFrame(
        cell_count_data,
        index=[f'QI > {t}' for t in qi_thresholds],
        columns=bin_labels
    )
    
    # Create custom annotations showing "batches\n(cells)"
    annot_matrix = []
    for i in range(len(heatmap_data)):
        row = []
        for j in range(len(bin_labels)):
            batches = heatmap_data[i][j]
            cells = cell_count_data[i][j]
            if batches > 0:
                row.append(f'{batches}\n({cells:,})')
            else:
                row.append('0\n(0)')
        annot_matrix.append(row)
    annot_df = pd.DataFrame(annot_matrix, index=heatmap_df.index, columns=heatmap_df.columns)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        heatmap_df,
        annot=annot_df,
        fmt='',
        cmap='YlOrRd',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Number of Batches'},
        annot_kws={'fontsize': 8}
    )
    
    ax.set_xlabel('Good Cells per Batch (range)', fontsize=12)
    ax.set_ylabel('QI Threshold', fontsize=12)
    ax.set_title(
        'Distribution of Batch Sizes by QI Threshold\n'
        'Each cell shows: number of batches (total cells)',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}")
    
    plt.close()
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY BY QI THRESHOLD")
    print("="*70)
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    return heatmap_df, summary_df


if __name__ == "__main__":
    plot_qi_batch_heatmap()

