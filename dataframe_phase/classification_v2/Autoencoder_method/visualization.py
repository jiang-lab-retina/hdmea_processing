"""
Visualization functions for the clustering pipeline.

Generates publication-ready plots:
- UMAP embeddings colored by group/cluster
- BIC curves with selected k marked
- Response prototypes per cluster
- Cluster size distributions
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (10, 8),
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    title: str | None = None,
    cmap: str = "tab20",
    alpha: float = 0.7,
    point_size: float = 10,
) -> plt.Figure:
    """
    Create UMAP visualization of embeddings.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        labels: (n_cells,) labels for coloring.
        label_name: Name of label for legend/title.
        output_path: Path to save figure.
        figsize: Figure size.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        title: Plot title.
        cmap: Colormap name.
        alpha: Point transparency.
        point_size: Point size.
    
    Returns:
        Figure object.
    """
    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed. Install with: pip install umap-learn")
        return None
    
    n_neighbors = n_neighbors if n_neighbors is not None else config.UMAP_N_NEIGHBORS
    min_dist = min_dist if min_dist is not None else config.UMAP_MIN_DIST
    
    logger.info(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    # Compute UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=config.UMAP_RANDOM_STATE,
    )
    umap_coords = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different label types
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    if n_labels <= 20:
        # Categorical coloring with legend
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[colors[i]],
                label=str(label),
                alpha=alpha,
                s=point_size,
            )
        ax.legend(title=label_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Many labels: convert to numeric indices for coloring
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_idx[l] for l in labels])
        
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=numeric_labels,
            cmap=cmap,
            alpha=alpha,
            s=point_size,
        )
        # No colorbar for categorical labels with many categories
        # (colorbar would be meaningless for cluster IDs)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title or f'UMAP colored by {label_name}')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved UMAP plot to {output_path}")
    
    return fig


def plot_bic_curves_combined(
    group_results: dict,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot BIC curves for all groups in a combined figure.
    
    Args:
        group_results: Dict with per-group clustering info (bic_values, k_range, k_selected).
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    n_groups = len(group_results)
    ncols = 2
    nrows = (n_groups + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_groups > 1 else [axes]
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_groups))
    
    for i, (group_name, results) in enumerate(group_results.items()):
        ax = axes[i]
        
        if 'bic_values' not in results or len(results['bic_values']) <= 1:
            ax.text(0.5, 0.5, f'{group_name}\n(single cluster)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(group_name)
            continue
        
        bic_values = np.array(results['bic_values'])
        k_values = results['k_range']
        k_selected = results['k_selected']
        
        # Normalize BIC for better comparison (shift to start at 0)
        bic_normalized = bic_values - bic_values.min()
        
        # Plot BIC curve
        ax.plot(k_values, bic_values, '-o', color=colors[i], linewidth=2, markersize=5)
        
        # Mark selected k
        selected_idx = k_values.index(k_selected) if k_selected in k_values else -1
        if selected_idx >= 0:
            ax.axvline(x=k_selected, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.scatter([k_selected], [bic_values[selected_idx]], 
                      color='red', s=150, zorder=5, marker='*',
                      label=f'Selected k={k_selected}')
        
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('BIC')
        ax.set_title(f'{group_name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_groups, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Cluster Number Selection (BIC Curves)', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved combined BIC plot to {output_path}")
    
    return fig


def plot_bic_curve(
    bic_values: np.ndarray,
    k_values: list[int],
    k_selected: int,
    group_name: str,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot BIC curve with selected k marked.
    
    Args:
        bic_values: BIC values for each k.
        k_values: k values tested.
        k_selected: Selected k (will be highlighted).
        group_name: Group name for title.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot BIC curve
    ax.plot(k_values, bic_values, 'b-o', linewidth=2, markersize=6, label='BIC')
    
    # Mark selected k
    selected_idx = k_values.index(k_selected) if k_selected in k_values else -1
    if selected_idx >= 0:
        ax.axvline(x=k_selected, color='r', linestyle='--', linewidth=2, label=f'Selected k={k_selected}')
        ax.scatter([k_selected], [bic_values[selected_idx]], color='r', s=150, zorder=5, marker='*')
    
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('BIC')
    ax.set_title(f'BIC Curve: {group_name}')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved BIC plot to {output_path}")
    
    return fig


def plot_response_prototypes(
    segments: dict[str, np.ndarray],
    cluster_labels: np.ndarray,
    groups: np.ndarray,
    group_name: str,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (16, 12),
    max_clusters: int = 10,
) -> plt.Figure:
    """
    Plot mean±SEM response traces for each cluster.
    
    Args:
        segments: Preprocessed segment arrays.
        cluster_labels: Cluster assignments.
        groups: Coarse group labels.
        group_name: Group to plot (filters to this group).
        output_path: Path to save figure.
        figsize: Figure size.
        max_clusters: Maximum clusters to show.
    
    Returns:
        Figure object.
    """
    # Filter to group
    group_mask = groups == group_name
    group_clusters = cluster_labels[group_mask]
    unique_clusters = np.unique(group_clusters)
    
    if len(unique_clusters) > max_clusters:
        logger.warning(f"Too many clusters ({len(unique_clusters)}), showing first {max_clusters}")
        unique_clusters = unique_clusters[:max_clusters]
    
    n_clusters = len(unique_clusters)
    n_segments = len(segments)
    
    fig, axes = plt.subplots(n_clusters, n_segments, figsize=figsize)
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    if n_segments == 1:
        axes = axes.reshape(-1, 1)
    
    segment_names = list(segments.keys())
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = group_clusters == cluster_id
        n_cells = cluster_mask.sum()
        
        for j, seg_name in enumerate(segment_names):
            ax = axes[i, j]
            
            # Get segment data for this cluster
            seg_data = segments[seg_name][group_mask][cluster_mask]
            
            # Compute mean and SEM
            mean_trace = np.mean(seg_data, axis=0)
            sem_trace = np.std(seg_data, axis=0) / np.sqrt(n_cells)
            
            # Plot
            time = np.arange(len(mean_trace))
            ax.plot(time, mean_trace, 'b-', linewidth=1.5)
            ax.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, 
                          alpha=0.3, color='b')
            
            # Labels
            if i == 0:
                ax.set_title(seg_name, fontsize=8)
            if j == 0:
                ax.set_ylabel(f'Cluster {cluster_id}\n(n={n_cells})', fontsize=8)
            
            ax.tick_params(labelsize=6)
    
    fig.suptitle(f'Response Prototypes: {group_name}', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved response prototypes to {output_path}")
    
    return fig


def plot_cluster_sizes(
    cluster_labels: np.ndarray,
    groups: np.ndarray,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot histogram of cluster sizes per group.
    
    Args:
        cluster_labels: Cluster assignments.
        groups: Coarse group labels.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)
    if n_groups == 1:
        axes = [axes]
    
    for i, group in enumerate(unique_groups):
        ax = axes[i]
        group_mask = groups == group
        group_clusters = cluster_labels[group_mask]
        
        # Count cluster sizes
        unique, counts = np.unique(group_clusters, return_counts=True)
        
        ax.bar(range(len(counts)), sorted(counts, reverse=True), color=f'C{i}')
        ax.set_xlabel('Cluster Rank')
        ax.set_ylabel('Cell Count')
        ax.set_title(f'{group}\n({len(unique)} clusters)')
        
        # Add min size threshold line
        ax.axhline(y=config.MIN_CLUSTER_SIZE, color='r', linestyle='--', 
                  label=f'Min size ({config.MIN_CLUSTER_SIZE})')
        ax.legend(fontsize=8)
    
    fig.suptitle('Cluster Size Distribution', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster sizes plot to {output_path}")
    
    return fig


def plot_cv_purity(
    cv_results: dict,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot cross-validation purity results as a bar chart.
    
    Args:
        cv_results: Dict from compute_cv_purity_posthoc.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Purity by omitted label
    purity_data = cv_results.get('purity_by_label', {})
    labels = []
    purities = []
    descriptions = []
    
    for label_name, data in purity_data.items():
        labels.append(label_name.replace('_', '\n'))
        purities.append(data['purity'] * 100)
        descriptions.append(data['description'])
    
    colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(labels)))
    bars = ax1.bar(labels, purities, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, purity in zip(bars, purities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{purity:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Purity (%)', fontsize=12)
    ax1.set_title('Cluster Purity by Omitted Label', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect purity')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right panel: Clusters per group
    group_data = cv_results.get('clusters_per_group', {})
    group_names = list(group_data.keys())
    n_clusters = [data['n_clusters'] for data in group_data.values()]
    n_cells = [data['n_cells'] for data in group_data.values()]
    
    x = np.arange(len(group_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, n_clusters, width, label='Clusters', color='steelblue', edgecolor='black')
    ax2.set_ylabel('Number of Clusters', color='steelblue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, n_cells, width, label='Cells', color='coral', edgecolor='black')
    ax2_twin.set_ylabel('Number of Cells', color='coral', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_names, rotation=15, ha='right')
    ax2.set_title('Clusters & Cells per Group', fontsize=12)
    
    # Add combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add overall CV score as text
    cv_score = cv_results.get('cv_score', 0)
    fig.text(0.5, 0.02, f'Overall CVScore: {cv_score:.4f} ({cv_score*100:.1f}%)', 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved CV purity plot to {output_path}")
    
    return fig


def generate_all_plots(
    embeddings: np.ndarray,
    segments: dict[str, np.ndarray],
    groups: np.ndarray,
    cluster_labels: np.ndarray,
    group_results: dict,
    cv_results: dict | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Generate all visualization outputs.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        segments: Preprocessed segment arrays.
        groups: Coarse group labels.
        cluster_labels: Cluster assignments.
        group_results: Per-group clustering results (for BIC curves).
        output_dir: Output directory. Defaults to config.PLOTS_DIR.
    
    Returns:
        Dict mapping plot name to file path.
    """
    output_dir = output_dir if output_dir is not None else config.PLOTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # UMAP by group
    logger.info("Generating UMAP by group...")
    fig = plot_umap_embeddings(
        embeddings, groups, 'Coarse Group',
        output_path=output_dir / 'umap_by_group.png'
    )
    if fig:
        saved_plots['umap_by_group'] = output_dir / 'umap_by_group.png'
        plt.close(fig)
    
    # UMAP by cluster
    logger.info("Generating UMAP by cluster...")
    # Create composite labels: group::cluster
    composite_labels = np.array([
        f"{g}::{c}" for g, c in zip(groups, cluster_labels)
    ])
    fig = plot_umap_embeddings(
        embeddings, composite_labels, 'Subtype',
        output_path=output_dir / 'umap_by_cluster.png'
    )
    if fig:
        saved_plots['umap_by_cluster'] = output_dir / 'umap_by_cluster.png'
        plt.close(fig)
    
    # Combined BIC curves (all groups in one figure)
    logger.info("Generating combined BIC curves...")
    fig = plot_bic_curves_combined(
        group_results,
        output_path=output_dir / 'bic_all_groups.png'
    )
    if fig:
        saved_plots['bic_all_groups'] = output_dir / 'bic_all_groups.png'
        plt.close(fig)
    
    # BIC curves per group (individual files)
    logger.info("Generating individual BIC curves per group...")
    for group_name, results in group_results.items():
        if 'bic_values' in results and len(results['bic_values']) > 1:
            fig = plot_bic_curve(
                np.array(results['bic_values']),
                results['k_range'],
                results['k_selected'],
                group_name,
                output_path=output_dir / f'bic_{group_name.replace("-", "_")}.png'
            )
            if fig:
                saved_plots[f'bic_{group_name}'] = output_dir / f'bic_{group_name.replace("-", "_")}.png'
                plt.close(fig)
    
    # Cluster sizes
    logger.info("Generating cluster size distribution...")
    fig = plot_cluster_sizes(
        cluster_labels, groups,
        output_path=output_dir / 'cluster_sizes.png'
    )
    if fig:
        saved_plots['cluster_sizes'] = output_dir / 'cluster_sizes.png'
        plt.close(fig)
    
    # Response prototypes per group
    logger.info("Generating response prototypes...")
    for group_name in np.unique(groups):
        fig = plot_response_prototypes(
            segments, cluster_labels, groups, group_name,
            output_path=output_dir / f'prototypes_{group_name.replace("-", "_")}.png'
        )
        if fig:
            saved_plots[f'prototypes_{group_name}'] = output_dir / f'prototypes_{group_name.replace("-", "_")}.png'
            plt.close(fig)
    
    # CV purity plot (if results provided)
    if cv_results is not None:
        logger.info("Generating CV purity plot...")
        fig = plot_cv_purity(
            cv_results,
            output_path=output_dir / 'cv_purity.png'
        )
        if fig:
            saved_plots['cv_purity'] = output_dir / 'cv_purity.png'
            plt.close(fig)
    
    logger.info(f"Generated {len(saved_plots)} plots in {output_dir}")
    return saved_plots
