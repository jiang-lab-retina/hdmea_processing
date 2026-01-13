"""
Visualization for the DEC-refined RGC clustering pipeline.

Includes:
- BIC curve plots
- UMAP visualizations (GMM vs DEC)
- ipRGC enrichment bar charts
- Cluster prototype traces
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import umap

# Support direct execution
if __name__ == "__main__" and __package__ is None:
    _this_dir = Path(__file__).resolve().parent
    _parent_dir = _this_dir.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    __package__ = "divide_conquer_method"

from divide_conquer_method import config

logger = logging.getLogger(__name__)


def plot_bic_curve(
    k_range: list,
    bic_values: np.ndarray,
    k_selected: int,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot BIC curve with selected k* marked.
    
    Args:
        k_range: List of k values.
        bic_values: BIC values for each k.
        k_selected: Selected k*.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot BIC curve
    ax.plot(k_range, bic_values, 'b-o', markersize=4, linewidth=1.5)
    
    # Mark selected k
    if k_selected in k_range:
        idx = k_range.index(k_selected)
        ax.axvline(k_selected, color='r', linestyle='--', alpha=0.7, label=f'k* = {k_selected}')
        ax.plot(k_selected, bic_values[idx], 'r*', markersize=15)
    
    ax.set_xlabel('Number of clusters (k)', fontsize=12)
    ax.set_ylabel('BIC', fontsize=12)
    ax.set_title('Model Selection: BIC vs k', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved BIC curve to {output_path}")
    
    return fig


def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Projection",
    iprgc_labels: np.ndarray | None = None,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot UMAP visualization of embeddings colored by cluster.
    
    Args:
        embeddings: (n_cells, 49) embeddings.
        labels: (n_cells,) cluster labels.
        title: Plot title.
        iprgc_labels: Optional binary labels to highlight ipRGCs.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    # Compute UMAP
    reducer = umap.UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels for color mapping
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Create colormap
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)],
            label=f'Cluster {label}',
            s=15,
            alpha=0.7,
        )
    
    # Highlight ipRGCs if provided
    if iprgc_labels is not None:
        iprgc_mask = iprgc_labels.astype(bool)
        ax.scatter(
            coords[iprgc_mask, 0], coords[iprgc_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=50,
            linewidths=1.5,
            label='ipRGC',
            alpha=0.8,
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Legend (limit to 10 clusters + ipRGC)
    if n_clusters <= 10:
        ax.legend(loc='best', fontsize=8, markerscale=1.5)
    else:
        # Show colorbar instead
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_clusters))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Cluster')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved UMAP plot to {output_path}")
    
    return fig


def plot_umap_comparison(
    embeddings_gmm: np.ndarray,
    embeddings_dec: np.ndarray,
    labels_gmm: np.ndarray,
    labels_dec: np.ndarray,
    iprgc_labels: np.ndarray | None = None,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Side-by-side UMAP comparison of GMM vs DEC clustering.
    
    Args:
        embeddings_gmm: Initial embeddings.
        embeddings_dec: DEC-refined embeddings.
        labels_gmm: GMM cluster labels.
        labels_dec: DEC cluster labels.
        iprgc_labels: Optional ipRGC labels to highlight.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute UMAP for both
    reducer = umap.UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.UMAP_RANDOM_STATE,
    )
    
    coords_gmm = reducer.fit_transform(embeddings_gmm)
    coords_dec = reducer.fit_transform(embeddings_dec)
    
    # Plot GMM
    ax = axes[0]
    unique_labels = np.unique(labels_gmm)
    n_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    for i, label in enumerate(unique_labels):
        mask = labels_gmm == label
        ax.scatter(coords_gmm[mask, 0], coords_gmm[mask, 1], c=[cmap(i)], s=10, alpha=0.6)
    
    if iprgc_labels is not None:
        iprgc_mask = iprgc_labels.astype(bool)
        ax.scatter(coords_gmm[iprgc_mask, 0], coords_gmm[iprgc_mask, 1],
                  facecolors='none', edgecolors='red', s=40, linewidths=1)
    
    ax.set_title('Initial GMM Clustering', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Plot DEC
    ax = axes[1]
    unique_labels = np.unique(labels_dec)
    n_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    for i, label in enumerate(unique_labels):
        mask = labels_dec == label
        ax.scatter(coords_dec[mask, 0], coords_dec[mask, 1], c=[cmap(i)], s=10, alpha=0.6)
    
    if iprgc_labels is not None:
        iprgc_mask = iprgc_labels.astype(bool)
        ax.scatter(coords_dec[iprgc_mask, 0], coords_dec[iprgc_mask, 1],
                  facecolors='none', edgecolors='red', s=40, linewidths=1)
    
    ax.set_title('DEC-Refined Clustering', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved UMAP comparison to {output_path}")
    
    return fig


def plot_iprgc_enrichment(
    metrics: dict,
    title: str = "ipRGC Enrichment by Cluster",
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Bar chart showing ipRGC enrichment per cluster.
    
    Args:
        metrics: ipRGC metrics dict from compute_iprgc_metrics.
        title: Plot title.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    per_cluster = metrics.get('per_cluster', {})
    if not per_cluster:
        ax.text(0.5, 0.5, "No cluster data", ha='center', va='center')
        return fig
    
    # Sort by enrichment
    sorted_clusters = sorted(per_cluster.items(), key=lambda x: x[1]['enrichment'], reverse=True)
    
    clusters = [c for c, _ in sorted_clusters]
    enrichments = [s['enrichment'] for _, s in sorted_clusters]
    fractions = [s['fraction'] for _, s in sorted_clusters]
    sizes = [s['n_cells'] for _, s in sorted_clusters]
    
    # Create bars
    x = np.arange(len(clusters))
    bars = ax.bar(x, enrichments, color='steelblue', alpha=0.8)
    
    # Color bars by enrichment level
    for bar, enr in zip(bars, enrichments):
        if enr > 2:
            bar.set_color('darkred')
        elif enr > 1:
            bar.set_color('orange')
    
    # Add baseline
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (no enrichment)')
    
    # Add labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}\n(n={s})' for c, s in zip(clusters, sizes)], fontsize=8)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Enrichment (fold over baseline)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved enrichment plot to {output_path}")
    
    return fig


def plot_cluster_prototypes(
    segments: dict,
    labels: np.ndarray,
    cluster_id: int,
    output_path: Path | None = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot mean±SEM traces for a single cluster.
    
    Args:
        segments: Dict of segment arrays.
        labels: Cluster labels.
        cluster_id: Cluster to plot.
        output_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Figure object.
    """
    mask = labels == cluster_id
    n_cells = mask.sum()
    
    n_segments = len(segments)
    n_cols = 3
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(segments.items()):
        ax = axes[i]
        
        cluster_data = data[mask]
        mean_trace = np.mean(cluster_data, axis=0)
        sem_trace = np.std(cluster_data, axis=0) / np.sqrt(n_cells)
        
        x = np.arange(len(mean_trace))
        
        ax.plot(x, mean_trace, 'b-', linewidth=1.5)
        ax.fill_between(x, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)
        
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Response')
    
    # Hide empty subplots
    for i in range(len(segments), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f'Cluster {cluster_id} Prototypes (n={n_cells})', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved prototype plot to {output_path}")
    
    return fig


def generate_all_plots(
    artifacts: dict,
    group: str,
    output_dir: Path,
) -> None:
    """
    Generate all visualization plots from artifacts.
    
    Args:
        artifacts: Dict with embeddings, labels, metrics, etc.
        group: Group name for titles.
        output_dir: Output directory for plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating plots for group {group}")
    
    # BIC curve
    if 'bic_values' in artifacts and 'k_range' in artifacts:
        plot_bic_curve(
            k_range=artifacts['k_range'],
            bic_values=artifacts['bic_values'],
            k_selected=artifacts.get('k_selected', artifacts['k_range'][np.argmin(artifacts['bic_values'])]),
            output_path=output_dir / "bic_curve.png",
        )
        plt.close()
    
    # UMAP plots
    if 'embeddings_initial' in artifacts:
        emb_init = artifacts['embeddings_initial']
        if isinstance(emb_init, np.ndarray):
            emb = emb_init
        else:
            # DataFrame case
            z_cols = [c for c in emb_init.columns if c.startswith('z_')]
            emb = emb_init[z_cols].values
        
        plot_umap_embeddings(
            embeddings=emb,
            labels=artifacts.get('gmm_labels', np.zeros(len(emb))),
            title=f'{group}: GMM Clustering',
            iprgc_labels=artifacts.get('iprgc_labels'),
            output_path=output_dir / "umap_gmm.png",
        )
        plt.close()
    
    if 'embeddings_dec' in artifacts:
        emb_dec = artifacts['embeddings_dec']
        if isinstance(emb_dec, np.ndarray):
            emb = emb_dec
        else:
            z_cols = [c for c in emb_dec.columns if c.startswith('z_')]
            emb = emb_dec[z_cols].values
        
        plot_umap_embeddings(
            embeddings=emb,
            labels=artifacts.get('dec_labels', np.zeros(len(emb))),
            title=f'{group}: DEC-Refined Clustering',
            iprgc_labels=artifacts.get('iprgc_labels'),
            output_path=output_dir / "umap_dec.png",
        )
        plt.close()
    
    # UMAP comparison
    if 'embeddings_initial' in artifacts and 'embeddings_dec' in artifacts:
        emb_init = artifacts['embeddings_initial']
        emb_dec = artifacts['embeddings_dec']
        
        if not isinstance(emb_init, np.ndarray):
            z_cols = [c for c in emb_init.columns if c.startswith('z_')]
            emb_init = emb_init[z_cols].values
        if not isinstance(emb_dec, np.ndarray):
            z_cols = [c for c in emb_dec.columns if c.startswith('z_')]
            emb_dec = emb_dec[z_cols].values
        
        plot_umap_comparison(
            embeddings_gmm=emb_init,
            embeddings_dec=emb_dec,
            labels_gmm=artifacts.get('gmm_labels', np.zeros(len(emb_init))),
            labels_dec=artifacts.get('dec_labels', np.zeros(len(emb_dec))),
            iprgc_labels=artifacts.get('iprgc_labels'),
            output_path=output_dir / "umap_comparison.png",
        )
        plt.close()
    
    # ipRGC enrichment
    if 'dec_metrics' in artifacts:
        plot_iprgc_enrichment(
            metrics=artifacts['dec_metrics'],
            title=f'{group}: ipRGC Enrichment (DEC)',
            output_path=output_dir / "iprgc_enrichment.png",
        )
        plt.close()
    
    # Cluster prototypes (top 3 enriched)
    if 'segments' in artifacts and 'dec_labels' in artifacts and 'dec_metrics' in artifacts:
        prototypes_dir = output_dir / "prototypes"
        prototypes_dir.mkdir(exist_ok=True)
        
        top_enriched = artifacts['dec_metrics'].get('top_enriched', [])
        for cluster_info in top_enriched[:3]:
            cluster_id = cluster_info['cluster']
            plot_cluster_prototypes(
                segments=artifacts['segments'],
                labels=artifacts['dec_labels'],
                cluster_id=cluster_id,
                output_path=prototypes_dir / f"cluster_{cluster_id}.png",
            )
            plt.close()
    
    logger.info(f"Generated all plots in {output_dir}")


if __name__ == "__main__":
    # Demo mode: generate sample plots with synthetic data
    print("Visualization module for divide_conquer_method pipeline")
    print("Run via run_pipeline.py to generate actual plots.")
    print("\nAvailable functions:")
    print("  - plot_bic_curve(bic_data, output_path)")
    print("  - plot_umap_embeddings(embeddings, labels, output_path)")
    print("  - plot_umap_comparison(emb_gmm, labels_gmm, emb_dec, labels_dec, output_path)")
    print("  - plot_iprgc_enrichment(metrics, output_path)")
    print("  - plot_cluster_prototypes(segments, labels, cluster_id, output_path)")
    print("  - generate_all_plots(artifacts, output_dir, group)")
