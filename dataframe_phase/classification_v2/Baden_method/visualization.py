"""
Visualization module for the Baden-method RGC clustering pipeline.

This module implements:
- BIC curve plotting
- Posterior probability separability curves
- Bootstrap stability plotting
- UMAP cluster projections
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Use non-interactive backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from . import config

logger = logging.getLogger(__name__)

# Try to import UMAP, but make it optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed. UMAP projections will use PCA fallback.")


# =============================================================================
# Matplotlib Style Configuration
# =============================================================================

def configure_style():
    """Configure matplotlib style for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# BIC Curve Plotting
# =============================================================================

def plot_bic_curve(
    bic_table: pd.DataFrame,
    population_name: str,
    save_path: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot BIC values vs cluster count k.
    
    Shows both absolute BIC and normalized BIC (BIC/n) on secondary y-axis.
    
    Args:
        bic_table: DataFrame with 'k', 'bic', and optionally 'bic_normalized' columns.
        population_name: Name for the population (e.g., "DS", "non-DS").
        save_path: Path to save the figure. If None, figure is not saved.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        
    Returns:
        Matplotlib Figure object.
    """
    configure_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Filter valid BIC values
    valid = bic_table[bic_table['bic'] < np.inf].copy()
    
    # Plot BIC curve (primary axis)
    line1, = ax.plot(valid['k'], valid['bic'], 'o-', color='#2E86AB', linewidth=2, markersize=6, label='BIC')
    
    # Mark optimal k
    optimal_idx = valid['bic'].idxmin()
    optimal_k = valid.loc[optimal_idx, 'k']
    optimal_bic = valid.loc[optimal_idx, 'bic']
    
    ax.axvline(optimal_k, color='#E94F37', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.scatter([optimal_k], [optimal_bic], color='#E94F37', s=100, zorder=5)
    
    # Labels for primary axis
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('BIC', color='#2E86AB')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    
    # Plot normalized BIC on secondary y-axis if available
    if 'bic_normalized' in valid.columns:
        ax2 = ax.twinx()
        line2, = ax2.plot(valid['k'], valid['bic_normalized'], 's--', color='#8B5CF6', 
                         linewidth=1.5, markersize=4, alpha=0.8, label='BIC/n')
        ax2.set_ylabel('BIC/n (normalized)', color='#8B5CF6')
        ax2.tick_params(axis='y', labelcolor='#8B5CF6')
        
        # Get normalized BIC at optimal k
        optimal_bic_norm = valid.loc[optimal_idx, 'bic_normalized']
        n_samples = valid.loc[optimal_idx, 'n_samples'] if 'n_samples' in valid.columns else 'N/A'
        
        # Combined legend
        ax.legend([line1, line2], 
                 [f'BIC (optimal k={optimal_k}, BIC={optimal_bic:.0f})', 
                  f'BIC/n (n={n_samples}, BIC/n={optimal_bic_norm:.4f})'],
                 loc='upper right')
    else:
        ax.legend([f'Optimal k = {optimal_k}'], loc='upper right')
    
    # Title
    ax.set_title(f'BIC Model Selection - {population_name}')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved BIC curve to {save_path}")
    
    return fig


# =============================================================================
# Posterior Probability Curves
# =============================================================================

def plot_posterior_curves(
    curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
    population_name: str,
    save_path: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot posterior probability separability curves.
    
    Args:
        curves: Dictionary mapping cluster ID to (x, y) curves.
        population_name: Name for the population.
        save_path: Path to save the figure.
        ax: Matplotlib axes to plot on.
        
    Returns:
        Matplotlib Figure object.
    """
    configure_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    n_clusters = len(curves)
    colors = cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    
    # Plot each cluster's curve
    for i, (k, (x, y)) in enumerate(sorted(curves.items())):
        color = colors[i % len(colors)]
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.7, label=f'Cluster {k}')
    
    # Compute and plot average curve
    x_common = np.linspace(0, 1, 100)
    y_values = []
    for k, (x, y) in curves.items():
        y_interp = np.interp(x_common, x, y)
        y_values.append(y_interp)
    y_mean = np.mean(y_values, axis=0)
    
    ax.plot(x_common, y_mean, 'k-', linewidth=3, label='Average')
    
    # Reference line for ideal separation
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Normalized Rank (fraction of cluster)')
    ax.set_ylabel('Posterior Probability')
    ax.set_title(f'Posterior Separability Curves - {population_name}')
    
    # Legend (only show if reasonable number of clusters)
    if n_clusters <= 10:
        ax.legend(loc='lower left', ncol=2)
    else:
        ax.legend(['Average'], loc='lower left')
    
    # Limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved posterior curves to {save_path}")
    
    return fig


# =============================================================================
# Bootstrap Stability Plot
# =============================================================================

def plot_bootstrap_stability(
    all_correlations: list,
    median_correlation: float,
    population_name: str,
    save_path: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot bootstrap stability results as a box plot with individual points.
    
    Args:
        all_correlations: List of correlation values from each bootstrap iteration.
        median_correlation: Median correlation across iterations.
        population_name: Name for the population (e.g., "DS", "non-DS").
        save_path: Path to save the figure. If None, figure is not saved.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        
    Returns:
        Matplotlib Figure object.
    """
    configure_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Filter out NaN values
    correlations = [c for c in all_correlations if not np.isnan(c)]
    
    if len(correlations) == 0:
        ax.text(0.5, 0.5, 'No bootstrap data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Bootstrap Stability - {population_name}')
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
        return fig
    
    correlations = np.array(correlations)
    
    # Create box plot
    bp = ax.boxplot([correlations], positions=[1], widths=0.5, patch_artist=True)
    
    # Style the box plot
    bp['boxes'][0].set_facecolor('#A8D5BA')
    bp['boxes'][0].set_edgecolor('#2E86AB')
    bp['boxes'][0].set_linewidth(1.5)
    bp['medians'][0].set_color('#E94F37')
    bp['medians'][0].set_linewidth(2)
    
    # Add individual points with jitter
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(correlations))
    ax.scatter(1 + jitter, correlations, c='#2E86AB', alpha=0.6, s=40, zorder=3)
    
    # Add stability threshold line
    threshold = config.STABILITY_THRESHOLD
    ax.axhline(threshold, color='#E94F37', linestyle='--', linewidth=1.5, 
               label=f'Stability threshold ({threshold:.2f})')
    
    # Add median annotation
    is_stable = median_correlation >= threshold if median_correlation is not None else False
    stability_text = "STABLE" if is_stable else "UNSTABLE"
    color = '#2E7D32' if is_stable else '#C62828'
    
    ax.text(1.4, median_correlation, f'Median: {median_correlation:.3f}\n({stability_text})',
            va='center', ha='left', fontsize=10, color=color, fontweight='bold')
    
    # Statistics annotation
    stats_text = (f'n iterations: {len(correlations)}\n'
                  f'Mean: {np.mean(correlations):.3f}\n'
                  f'Std: {np.std(correlations):.3f}\n'
                  f'Min: {np.min(correlations):.3f}\n'
                  f'Max: {np.max(correlations):.3f}')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and title
    ax.set_ylabel('Cluster Matching Correlation')
    ax.set_title(f'Bootstrap Stability Analysis - {population_name}')
    ax.set_xlim(0.5, 2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1])
    ax.set_xticklabels([population_name])
    ax.legend(loc='lower right')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved bootstrap stability plot to {save_path}")
    
    return fig


# =============================================================================
# UMAP Projection
# =============================================================================

def compute_umap_projection(
    X: np.ndarray,
    n_neighbors: int = None,
    min_dist: float = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute 2D UMAP projection of feature matrix.
    
    Falls back to PCA if UMAP is not available.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.
        random_state: Random seed.
        
    Returns:
        2D projection of shape (n_samples, 2).
    """
    if n_neighbors is None:
        n_neighbors = config.UMAP_N_NEIGHBORS
    if min_dist is None:
        min_dist = config.UMAP_MIN_DIST
    
    if UMAP_AVAILABLE:
        try:
            logger.info(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                n_jobs=1,  # Avoid multiprocessing issues
            )
            projection = reducer.fit_transform(X)
            return projection
        except Exception as e:
            logger.warning(f"UMAP failed ({e}), falling back to PCA...")
    
    # Fallback to PCA
    if not UMAP_AVAILABLE:
        logger.info("UMAP not available, using PCA for projection...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state)
    projection = pca.fit_transform(X)
    
    return projection


def plot_umap_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    population_name: str,
    save_path: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
    projection: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot 2D UMAP projection colored by cluster assignment.
    
    Args:
        X: Feature matrix (used for UMAP if projection not provided).
        labels: Cluster assignments.
        population_name: Name for the population.
        save_path: Path to save the figure.
        ax: Matplotlib axes to plot on.
        projection: Pre-computed 2D projection. If None, computes UMAP.
        
    Returns:
        Matplotlib Figure object.
    """
    configure_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Compute projection if not provided
    if projection is None:
        projection = compute_umap_projection(X)
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Use a colormap that works well for many clusters
    if n_clusters <= 20:
        colors = cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = cm.nipy_spectral(np.linspace(0.1, 0.9, n_clusters))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label}',
            s=10,
            alpha=0.6,
        )
    
    # Labels and title
    projection_type = "UMAP" if UMAP_AVAILABLE else "PCA"
    ax.set_xlabel(f'{projection_type} 1')
    ax.set_ylabel(f'{projection_type} 2')
    ax.set_title(f'{projection_type} Cluster Projection - {population_name} ({n_clusters} clusters)')
    
    # Legend (only show if reasonable number)
    if n_clusters <= 20:
        ax.legend(loc='upper right', ncol=2, markerscale=2, fontsize=8)
    
    # Remove grid for scatter plots
    ax.grid(False)
    
    fig.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved cluster projection to {save_path}")
    
    return fig


# =============================================================================
# Combined Visualization
# =============================================================================

def create_all_plots(
    bic_table: pd.DataFrame,
    labels: np.ndarray,
    posteriors: np.ndarray,
    X: np.ndarray,
    population_name: str,
    output_dir: str | Path,
    evaluation_metrics: Optional[Dict] = None,
) -> Dict[str, Path]:
    """
    Generate all visualization plots for a population.
    
    Args:
        bic_table: BIC table from model selection.
        labels: Cluster assignments.
        posteriors: Posterior probabilities.
        X: Standardized feature matrix.
        population_name: "DS" or "non-DS".
        output_dir: Directory to save plots.
        evaluation_metrics: Optional dict with bootstrap results.
        
    Returns:
        Dictionary mapping plot type to saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # BIC curve
    bic_path = output_dir / f"bic_curves_{population_name.lower().replace('-', '')}.png"
    plot_bic_curve(bic_table, population_name, save_path=bic_path)
    saved_paths['bic'] = bic_path
    plt.close()
    
    # Posterior curves
    from . import evaluation
    curves = evaluation.compute_posterior_curves(labels, posteriors)
    posterior_path = output_dir / f"posterior_curves_{population_name.lower().replace('-', '')}.png"
    plot_posterior_curves(curves, population_name, save_path=posterior_path)
    saved_paths['posterior'] = posterior_path
    plt.close()
    
    # Bootstrap stability plot (if bootstrap was run)
    if evaluation_metrics is not None:
        all_corrs = evaluation_metrics.get('bootstrap_all_correlations', [])
        median_corr = evaluation_metrics.get('bootstrap_median_correlation')
        
        if all_corrs and len(all_corrs) > 0:
            bootstrap_path = output_dir / f"bootstrap_stability_{population_name.lower().replace('-', '')}.png"
            plot_bootstrap_stability(
                all_corrs, 
                median_corr, 
                population_name, 
                save_path=bootstrap_path
            )
            saved_paths['bootstrap'] = bootstrap_path
            plt.close()
    
    # UMAP projection
    umap_path = output_dir / f"umap_clusters_{population_name.lower().replace('-', '')}.png"
    plot_umap_clusters(X, labels, population_name, save_path=umap_path)
    saved_paths['umap'] = umap_path
    plt.close()
    
    logger.info(f"Created {len(saved_paths)} plots for {population_name}")
    return saved_paths

