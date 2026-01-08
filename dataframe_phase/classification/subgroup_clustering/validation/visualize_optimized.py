"""
Visualize Optimized Clustering Results.

Generate UMAP plots for each subgroup using the optimized hyperparameters.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataframe_phase.classification.subgroup_clustering.config import (
    OUTPUT_DIR,
    PLOTS_DIR,
    SUBGROUPS,
    SUBGROUP_COLORS,
    CLUSTER_PALETTE,
    OPTIMIZED_PARAMS,
    APPROACH_NAMES,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_SPREAD,
    UMAP_METRIC,
    UMAP_RANDOM_STATE,
    SUBGROUP_UMAP_PARAMS,
)


def load_optimized_results():
    """Load the optimized results."""
    results_file = OUTPUT_DIR / "optimized_results.pkl"
    metrics_file = OUTPUT_DIR / "optimized_metrics_summary.json"
    
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    return results, metrics


def compute_embedding(X, method="pca", n_components=2, random_state=42, subgroup=None, labels=None):
    """Compute 2D embedding using PCA, TSNE, or UMAP with subgroup-specific params.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    method : str
        "umap", "supervised_umap", "tsne", or "pca"
    n_components : int
        Number of output dimensions
    random_state : int
        Random seed
    subgroup : str
        Subgroup name for subgroup-specific UMAP params
    labels : np.ndarray
        Cluster labels for supervised UMAP
    """
    # First reduce to 50 dims if high-dimensional
    if X.shape[1] > 50:
        pca_pre = PCA(n_components=min(50, X.shape[0] - 1), random_state=random_state)
        X = pca_pre.fit_transform(X)
    
    if method in ["umap", "supervised_umap"] and HAS_UMAP:
        try:
            # Get subgroup-specific UMAP parameters
            if subgroup and subgroup in SUBGROUP_UMAP_PARAMS:
                params = SUBGROUP_UMAP_PARAMS[subgroup]
                n_neighbors = params.get("n_neighbors", UMAP_N_NEIGHBORS)
                min_dist = params.get("min_dist", UMAP_MIN_DIST)
                spread = params.get("spread", UMAP_SPREAD)
            else:
                n_neighbors = UMAP_N_NEIGHBORS
                min_dist = UMAP_MIN_DIST
                spread = UMAP_SPREAD
            
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                metric=UMAP_METRIC,
                n_components=n_components,
                random_state=random_state,
                n_jobs=1,
            )
            
            # Supervised UMAP uses labels to guide embedding
            if method == "supervised_umap" and labels is not None:
                return reducer.fit_transform(X, y=labels), "Supervised UMAP"
            else:
                return reducer.fit_transform(X), "UMAP"
        except Exception as e:
            print(f"  UMAP failed: {e}, falling back to t-SNE")
            method = "tsne"
    
    if method == "tsne":
        try:
            perplexity = min(30, len(X) - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                n_jobs=1,
            )
            return reducer.fit_transform(X), "t-SNE"
        except Exception as e:
            print(f"  t-SNE failed: {e}, falling back to PCA")
    
    # Fallback to PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X), "PCA"


def get_axis_limits(embedding, percentile=99, padding=0.1):
    """
    Get axis limits that exclude outliers using percentile clipping.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding array [N, 2]
    percentile : float
        Percentile for clipping (default 99 = use 1st to 99th percentile)
    padding : float
        Fraction of range to add as padding (default 0.1 = 10%)
    
    Returns
    -------
    tuple
        ((x_low, x_high), (y_low, y_high))
    """
    x_low = np.percentile(embedding[:, 0], 100 - percentile)
    x_high = np.percentile(embedding[:, 0], percentile)
    y_low = np.percentile(embedding[:, 1], 100 - percentile)
    y_high = np.percentile(embedding[:, 1], percentile)
    
    # Add padding
    x_range = x_high - x_low
    y_range = y_high - y_low
    x_low -= padding * x_range
    x_high += padding * x_range
    y_low -= padding * y_range
    y_high += padding * y_range
    
    return (x_low, x_high), (y_low, y_high)


def plot_subgroup_clusters(
    latents: np.ndarray,
    labels: np.ndarray,
    subgroup: str,
    metrics: dict,
    save_path: Path,
):
    """Plot UMAP visualization for a single subgroup with optimized clustering."""
    print(f"  Plotting {subgroup}...")
    
    # Compute embedding with subgroup-specific UMAP parameters
    embedding, method_name = compute_embedding(latents, method="umap", subgroup=subgroup)
    
    n_clusters = len(np.unique(labels))
    colors = CLUSTER_PALETTE[:n_clusters]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, cluster_id in enumerate(sorted(np.unique(labels))):
        mask = labels == cluster_id
        count = np.sum(mask)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=f"Cluster {cluster_id} (n={count})",
            s=30,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.3,
        )
    
    # Title with metrics
    silhouette = metrics.get("silhouette", 0)
    ch_index = metrics.get("calinski_harabasz", 0)
    params = OPTIMIZED_PARAMS.get(subgroup, {})
    
    ax.set_title(
        f"{subgroup} Optimized Clustering\n"
        f"k={n_clusters}, Silhouette={silhouette:.3f}, CH={ch_index:.1f}\n"
        f"(latent={params.get('latent_dim', 'N/A')}, layers={params.get('n_conv_layers', 'N/A')}, "
        f"channels={params.get('base_channels', 'N/A')})",
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel(f"{method_name} 1", fontsize=12)
    ax.set_ylabel(f"{method_name} 2", fontsize=12)
    
    # Apply 99th percentile axis limits with 10% padding
    xlim, ylim = get_axis_limits(embedding, percentile=99, padding=0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
    )
    
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_supervised_umap(
    latents: np.ndarray,
    labels: np.ndarray,
    subgroup: str,
    metrics: dict,
    save_path: Path,
):
    """Plot Supervised UMAP visualization using cluster labels to guide embedding."""
    print(f"  Plotting Supervised UMAP for {subgroup}...")
    
    # Compute supervised embedding
    embedding, method_name = compute_embedding(
        latents, method="supervised_umap", subgroup=subgroup, labels=labels
    )
    
    n_clusters = len(np.unique(labels))
    colors = CLUSTER_PALETTE[:n_clusters]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, cluster_id in enumerate(sorted(np.unique(labels))):
        mask = labels == cluster_id
        count = np.sum(mask)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=f"Cluster {cluster_id} (n={count})",
            s=30,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.3,
        )
    
    # Title with metrics
    silhouette = metrics.get("silhouette", 0)
    ch_index = metrics.get("calinski_harabasz", 0)
    
    ax.set_title(
        f"{subgroup} - Supervised UMAP\n"
        f"k={n_clusters}, Silhouette={silhouette:.3f}, CH={ch_index:.1f}\n"
        f"(Labels used to guide UMAP embedding)",
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel(f"{method_name} 1", fontsize=12)
    ax.set_ylabel(f"{method_name} 2", fontsize=12)
    
    # Apply 99th percentile axis limits with 10% padding
    xlim, ylim = get_axis_limits(embedding, percentile=99, padding=0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
    )
    
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_umap_comparison(
    latents: np.ndarray,
    labels: np.ndarray,
    subgroup: str,
    metrics: dict,
    save_path: Path,
):
    """Plot side-by-side comparison of unsupervised vs supervised UMAP."""
    print(f"  Plotting UMAP comparison for {subgroup}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Unsupervised UMAP
    embedding_unsup, _ = compute_embedding(latents, method="umap", subgroup=subgroup)
    
    # Supervised UMAP
    embedding_sup, _ = compute_embedding(
        latents, method="supervised_umap", subgroup=subgroup, labels=labels
    )
    
    n_clusters = len(np.unique(labels))
    colors = CLUSTER_PALETTE[:n_clusters]
    
    for ax_idx, (embedding, title) in enumerate([
        (embedding_unsup, "Unsupervised UMAP"),
        (embedding_sup, "Supervised UMAP"),
    ]):
        ax = axes[ax_idx]
        
        for i, cluster_id in enumerate(sorted(np.unique(labels))):
            mask = labels == cluster_id
            count = np.sum(mask)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=f"C{cluster_id} (n={count})",
                s=25,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.2,
            )
        
        # Apply axis limits
        xlim, ylim = get_axis_limits(embedding, percentile=99, padding=0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_title(f"{title}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dim 1", fontsize=11)
        ax.set_ylabel("Dim 2", fontsize=11)
        ax.set_facecolor('#fafafa')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    silhouette = metrics.get("silhouette", 0)
    plt.suptitle(
        f"{subgroup} - UMAP Comparison (k={n_clusters}, Silhouette={silhouette:.3f})",
        fontsize=16,
        fontweight='bold',
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_approach_comparison(
    results: dict,
    subgroup: str,
    save_path: Path,
):
    """
    Plot side-by-side comparison of Optimized AE vs Contrastive AE for a subgroup.
    
    Parameters
    ----------
    results : dict
        Results dict containing 'optimized_ae' and optionally 'contrastive_ae'
    subgroup : str
        Subgroup name
    save_path : Path
        Output path
    """
    # Check if contrastive results exist
    has_contrastive = "contrastive_ae" in results
    
    if not has_contrastive:
        print(f"  Skipping approach comparison for {subgroup} (no contrastive results)")
        return
    
    print(f"  Plotting approach comparison for {subgroup}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    approaches = ["optimized_ae", "contrastive_ae"]
    titles = ["Optimized AE + K-Means", "Contrastive AE + K-Means"]
    
    for ax_idx, (approach, title) in enumerate(zip(approaches, titles)):
        ax = axes[ax_idx]
        
        result = results[approach]
        latents = result["latents"]
        labels = result["labels"]
        metrics = result.get("metrics", {})
        
        # Compute embedding
        embedding, method_name = compute_embedding(latents, method="umap", subgroup=subgroup)
        
        n_clusters = len(np.unique(labels))
        colors = CLUSTER_PALETTE[:n_clusters]
        
        for i, cluster_id in enumerate(sorted(np.unique(labels))):
            mask = labels == cluster_id
            count = np.sum(mask)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=f"C{cluster_id} (n={count})",
                s=25,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.2,
            )
        
        # Apply axis limits
        xlim, ylim = get_axis_limits(embedding, percentile=99, padding=0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        silhouette = metrics.get("silhouette", 0)
        ax.set_title(f"{title}\nk={n_clusters}, Silhouette={silhouette:.3f}", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"{method_name} 1", fontsize=11)
        ax.set_ylabel(f"{method_name} 2", fontsize=11)
        ax.set_facecolor('#fafafa')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Compute improvement
    sil_opt = results["optimized_ae"]["metrics"].get("silhouette", 0)
    sil_con = results["contrastive_ae"]["metrics"].get("silhouette", 0)
    improvement = ((sil_con - sil_opt) / max(sil_opt, 0.001)) * 100
    improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
    
    plt.suptitle(
        f"{subgroup} - Approach Comparison\n"
        f"Contrastive improvement: {improvement_str}",
        fontsize=16,
        fontweight='bold',
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_comparison_grid(all_results: dict, all_metrics: dict, save_path: Path):
    """Create a 2x2 grid comparing all subgroups."""
    print("  Creating comparison grid...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, subgroup in enumerate(SUBGROUPS):
        ax = axes[idx]
        
        if subgroup not in all_results:
            ax.text(0.5, 0.5, f"No data for {subgroup}", ha='center', va='center')
            ax.set_title(subgroup)
            continue
        
        result = all_results[subgroup]
        metrics = all_metrics.get(subgroup, {})
        
        latents = result["latents"]
        labels = result["labels"]
        
        # Compute embedding with subgroup-specific UMAP parameters
        embedding, method_name = compute_embedding(latents, method="umap", subgroup=subgroup)
        
        n_clusters = len(np.unique(labels))
        colors = CLUSTER_PALETTE[:n_clusters]
        
        for i, cluster_id in enumerate(sorted(np.unique(labels))):
            mask = labels == cluster_id
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                s=15,
                alpha=0.6,
                edgecolors='none',
            )
        
        # Apply 99th percentile axis limits with 10% padding
        xlim, ylim = get_axis_limits(embedding, percentile=99, padding=0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Title
        silhouette = metrics.get("silhouette", 0)
        params = OPTIMIZED_PARAMS.get(subgroup, {})
        ax.set_title(
            f"{subgroup}\n"
            f"k={n_clusters}, Silhouette={silhouette:.3f}\n"
            f"(n={len(labels)} units)",
            fontsize=12,
            fontweight='bold',
            color=SUBGROUP_COLORS.get(subgroup, 'black'),
        )
        ax.set_xlabel(f"{method_name} 1", fontsize=10)
        ax.set_ylabel(f"{method_name} 2", fontsize=10)
        ax.set_facecolor('#fafafa')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(
        "Optimized Subgroup Clustering Comparison",
        fontsize=16,
        fontweight='bold',
        y=1.02,
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_all_approaches_grid(all_results: dict, save_path: Path):
    """
    Create a comprehensive grid showing all subgroups x all approaches.
    
    Shows Optimized AE and Contrastive AE side by side for each subgroup.
    """
    print("  Creating all-approaches comparison grid...")
    
    # Determine available approaches
    approaches = ["optimized_ae", "contrastive_ae"]
    approach_labels = ["Optimized AE", "Contrastive AE"]
    
    # Check which approaches are available
    available_approaches = []
    for approach in approaches:
        for subgroup in SUBGROUPS:
            if subgroup in all_results and approach in all_results[subgroup]:
                if approach not in available_approaches:
                    available_approaches.append(approach)
                break
    
    n_approaches = len(available_approaches)
    n_subgroups = len([s for s in SUBGROUPS if s in all_results])
    
    if n_approaches == 0 or n_subgroups == 0:
        print("  Skipping: no data available")
        return
    
    fig, axes = plt.subplots(n_subgroups, n_approaches, figsize=(8 * n_approaches, 6 * n_subgroups))
    
    # Handle case of single row/column
    if n_subgroups == 1:
        axes = axes.reshape(1, -1)
    if n_approaches == 1:
        axes = axes.reshape(-1, 1)
    
    row_idx = 0
    for subgroup in SUBGROUPS:
        if subgroup not in all_results:
            continue
        
        for col_idx, approach in enumerate(available_approaches):
            ax = axes[row_idx, col_idx]
            
            if approach not in all_results[subgroup]:
                ax.text(0.5, 0.5, "Not available", ha='center', va='center', fontsize=12)
                ax.set_title(f"{subgroup} - {APPROACH_NAMES.get(approach, approach)}")
                continue
            
            result = all_results[subgroup][approach]
            latents = result["latents"]
            labels = result["labels"]
            metrics = result.get("metrics", {})
            
            # Compute embedding
            embedding, method_name = compute_embedding(latents, method="umap", subgroup=subgroup)
            
            n_clusters = len(np.unique(labels))
            colors = CLUSTER_PALETTE[:n_clusters]
            
            for i, cluster_id in enumerate(sorted(np.unique(labels))):
                mask = labels == cluster_id
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[i]],
                    s=15,
                    alpha=0.6,
                    edgecolors='none',
                )
            
            # Apply axis limits
            xlim, ylim = get_axis_limits(embedding, percentile=99, padding=0.1)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            silhouette = metrics.get("silhouette", 0)
            approach_label = APPROACH_NAMES.get(approach, approach)
            ax.set_title(
                f"{subgroup} - {approach_label}\n"
                f"k={n_clusters}, Sil={silhouette:.3f}",
                fontsize=11,
                fontweight='bold',
                color=SUBGROUP_COLORS.get(subgroup, 'black'),
            )
            ax.set_xlabel(f"{method_name} 1", fontsize=9)
            ax.set_ylabel(f"{method_name} 2", fontsize=9)
            ax.set_facecolor('#fafafa')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        row_idx += 1
    
    plt.suptitle(
        "All Subgroups x All Approaches Comparison",
        fontsize=16,
        fontweight='bold',
        y=1.01,
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_metrics_summary(all_metrics: dict, save_path: Path):
    """Create a bar chart comparing metrics across subgroups."""
    print("  Creating metrics summary...")
    
    subgroups = list(all_metrics.keys())
    silhouettes = [all_metrics[s].get("silhouette", 0) for s in subgroups]
    n_clusters = [all_metrics[s].get("n_clusters", 0) for s in subgroups]
    n_samples = [all_metrics[s].get("n_samples", 0) for s in subgroups]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = [SUBGROUP_COLORS.get(s, '#888888') for s in subgroups]
    
    # Silhouette scores
    bars1 = axes[0].bar(subgroups, silhouettes, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel("Silhouette Score", fontsize=12)
    axes[0].set_title("Cluster Quality", fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, max(silhouettes) * 1.2)
    for bar, val in zip(bars1, silhouettes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # Number of clusters
    bars2 = axes[1].bar(subgroups, n_clusters, color=colors, edgecolor='white', linewidth=2)
    axes[1].set_ylabel("Number of Clusters", fontsize=12)
    axes[1].set_title("Cluster Count (k)", fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, n_clusters):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val}', ha='center', fontsize=11, fontweight='bold')
    
    # Sample counts
    bars3 = axes[2].bar(subgroups, n_samples, color=colors, edgecolor='white', linewidth=2)
    axes[2].set_ylabel("Number of Units", fontsize=12)
    axes[2].set_title("Sample Size", fontsize=14, fontweight='bold')
    for bar, val in zip(bars3, n_samples):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{val}', ha='center', fontsize=10, fontweight='bold')
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle("Optimized Clustering Summary", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def plot_hyperparameter_summary(save_path: Path):
    """Create a visual summary of optimized hyperparameters."""
    print("  Creating hyperparameter summary...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Table data
    columns = ['Subgroup', 'Latent Dim', 'LR', 'Layers', 'Channels', 'Dropout', 'k', 'Silhouette']
    rows = []
    for sg in SUBGROUPS:
        p = OPTIMIZED_PARAMS.get(sg, {})
        rows.append([
            sg,
            p.get('latent_dim', 'N/A'),
            f"{p.get('learning_rate', 0):.2e}",
            p.get('n_conv_layers', 'N/A'),
            p.get('base_channels', 'N/A'),
            f"{p.get('dropout', 0):.2f}",
            p.get('optimal_k', 'N/A'),
            f"{p.get('silhouette', 0):.3f}",
        ])
    
    ax.axis('off')
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#E8E8E8'] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color the subgroup column
    for i, sg in enumerate(SUBGROUPS):
        table[(i+1, 0)].set_facecolor(SUBGROUP_COLORS.get(sg, '#FFFFFF'))
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Optimized Hyperparameters Summary", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [OK] Saved: {save_path.name}")


def main():
    """Main visualization function."""
    print("=" * 70)
    print("Visualizing Optimized Clustering Results")
    print("=" * 70)
    
    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n[1] Loading results...")
    results, metrics = load_optimized_results()
    print(f"    Loaded {len(results)} subgroups")
    
    # Check what approaches are available
    available_approaches = set()
    for subgroup in results:
        available_approaches.update(results[subgroup].keys())
    print(f"    Available approaches: {', '.join(available_approaches)}")
    
    # Extract optimized_ae results (nested structure)
    optimized_results = {}
    optimized_metrics = {}
    for subgroup in SUBGROUPS:
        if subgroup in results and 'optimized_ae' in results[subgroup]:
            optimized_results[subgroup] = results[subgroup]['optimized_ae']
            optimized_metrics[subgroup] = results[subgroup]['optimized_ae'].get('metrics', {})
    
    # Generate plots for optimized_ae
    print("\n[2] Generating individual subgroup plots (Optimized AE - Unsupervised UMAP)...")
    for subgroup in SUBGROUPS:
        if subgroup in optimized_results:
            result = optimized_results[subgroup]
            subgroup_metrics = optimized_metrics.get(subgroup, {})
            plot_subgroup_clusters(
                latents=result["latents"],
                labels=result["labels"],
                subgroup=subgroup,
                metrics=subgroup_metrics,
                save_path=PLOTS_DIR / f"{subgroup}_optimized_clusters.png",
            )
    
    print("\n[3] Generating Supervised UMAP plots...")
    for subgroup in SUBGROUPS:
        if subgroup in optimized_results:
            result = optimized_results[subgroup]
            subgroup_metrics = optimized_metrics.get(subgroup, {})
            # Supervised UMAP
            plot_supervised_umap(
                latents=result["latents"],
                labels=result["labels"],
                subgroup=subgroup,
                metrics=subgroup_metrics,
                save_path=PLOTS_DIR / f"{subgroup}_supervised_umap.png",
            )
            # Side-by-side comparison
            plot_umap_comparison(
                latents=result["latents"],
                labels=result["labels"],
                subgroup=subgroup,
                metrics=subgroup_metrics,
                save_path=PLOTS_DIR / f"{subgroup}_umap_comparison.png",
            )
    
    # Generate contrastive-specific plots if available
    if "contrastive_ae" in available_approaches:
        print("\n[4] Generating Contrastive AE plots...")
        for subgroup in SUBGROUPS:
            if subgroup in results and "contrastive_ae" in results[subgroup]:
                result = results[subgroup]["contrastive_ae"]
                contrastive_metrics = result.get("metrics", {})
                
                # Contrastive clusters
                plot_subgroup_clusters(
                    latents=result["latents"],
                    labels=result["labels"],
                    subgroup=subgroup,
                    metrics=contrastive_metrics,
                    save_path=PLOTS_DIR / f"{subgroup}_contrastive_clusters.png",
                )
                
                # Approach comparison (Optimized AE vs Contrastive AE)
                plot_approach_comparison(
                    results=results[subgroup],
                    subgroup=subgroup,
                    save_path=PLOTS_DIR / f"{subgroup}_approach_comparison.png",
                )
    
    print("\n[5] Generating comparison grids...")
    plot_comparison_grid(optimized_results, optimized_metrics, PLOTS_DIR / "optimized_comparison_grid.png")
    
    # All approaches grid (includes contrastive if available)
    if "contrastive_ae" in available_approaches:
        plot_all_approaches_grid(results, PLOTS_DIR / "all_approaches_comparison.png")
    
    plot_metrics_summary(optimized_metrics, PLOTS_DIR / "optimized_metrics_summary.png")
    plot_hyperparameter_summary(PLOTS_DIR / "optimized_hyperparameters.png")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Clustering Results by Approach")
    print("=" * 70)
    print(f"\n{'Subgroup':<10} {'Approach':<20} {'k':<5} {'Silhouette':<12}")
    print("-" * 50)
    
    for subgroup in SUBGROUPS:
        if subgroup not in results:
            continue
        for approach in ["optimized_ae", "contrastive_ae"]:
            if approach not in results[subgroup]:
                continue
            result = results[subgroup][approach]
            k = result.get("optimal_k", len(np.unique(result["labels"])))
            sil = result["metrics"].get("silhouette", 0)
            approach_name = "OptimizedAE" if approach == "optimized_ae" else "ContrastiveAE"
            print(f"{subgroup:<10} {approach_name:<20} {k:<5} {sil:.3f}")
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

