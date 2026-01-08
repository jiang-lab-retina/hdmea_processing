"""
Optimize UMAP parameters to maximize cluster separation in the embedding.

Uses grid search or Optuna to find the best UMAP parameters
that maximize silhouette score in the 2D embedding space.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
from pathlib import Path
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataframe_phase.classification.subgroup_clustering.config import (
    OUTPUT_DIR,
    PLOTS_DIR,
    CLUSTER_PALETTE,
)


def load_results():
    """Load results."""
    results_file = OUTPUT_DIR / "optimized_results.pkl"
    with open(results_file, "rb") as f:
        return pickle.load(f)


def compute_umap_with_score(X, labels, n_neighbors, min_dist, spread, random_state=42):
    """
    Compute UMAP and return embedding with silhouette score.
    """
    # PCA pre-reduction
    if X.shape[1] > 50:
        pca = PCA(n_components=min(50, X.shape[0] - 1), random_state=random_state)
        X = pca.fit_transform(X)
    
    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            n_components=2,
            random_state=random_state,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(X)
        
        # Compute silhouette score on the 2D embedding
        score = silhouette_score(embedding, labels)
        return embedding, score
    except Exception as e:
        return None, -1.0


def grid_search_umap(latents, labels, verbose=True):
    """
    Grid search over UMAP parameters to find optimal settings.
    """
    # Parameter grid
    n_neighbors_range = [10, 15, 20, 25, 30]
    min_dist_range = [0.0, 0.01, 0.05, 0.1, 0.2]
    spread_range = [0.5, 1.0, 1.5, 2.0]
    
    best_score = -1
    best_params = None
    best_embedding = None
    all_results = []
    
    total = len(n_neighbors_range) * len(min_dist_range) * len(spread_range)
    
    if verbose:
        print(f"\nGrid search over {total} combinations...")
    
    count = 0
    for n_neighbors in n_neighbors_range:
        for min_dist in min_dist_range:
            for spread in spread_range:
                count += 1
                
                embedding, score = compute_umap_with_score(
                    latents, labels, n_neighbors, min_dist, spread
                )
                
                if embedding is not None:
                    all_results.append({
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'spread': spread,
                        'silhouette': score,
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = (n_neighbors, min_dist, spread)
                        best_embedding = embedding
                        
                        if verbose:
                            print(f"  [{count}/{total}] NEW BEST: n={n_neighbors}, d={min_dist}, s={spread} -> silhouette={score:.4f}")
    
    return best_params, best_score, best_embedding, all_results


def get_axis_limits(embedding, percentile=99):
    """Get axis limits excluding outliers."""
    x_low = np.percentile(embedding[:, 0], 100 - percentile)
    x_high = np.percentile(embedding[:, 0], percentile)
    y_low = np.percentile(embedding[:, 1], 100 - percentile)
    y_high = np.percentile(embedding[:, 1], percentile)
    
    x_range = x_high - x_low
    y_range = y_high - y_low
    x_low -= 0.1 * x_range
    x_high += 0.1 * x_range
    y_low -= 0.1 * y_range
    y_high += 0.1 * y_range
    
    return (x_low, x_high), (y_low, y_high)


def plot_optimized(embedding, labels, params, score, subgroup, save_path):
    """Plot the optimized UMAP."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    n_clusters = len(np.unique(labels))
    colors = CLUSTER_PALETTE[:n_clusters]
    
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
    
    # Apply axis limits
    xlim, ylim = get_axis_limits(embedding, percentile=99)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    n_neighbors, min_dist, spread = params
    ax.set_title(
        f"{subgroup} - Optimized UMAP\n"
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}\n"
        f"2D Silhouette Score: {score:.4f}",
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_top_results(latents, labels, all_results, subgroup, save_dir):
    """Plot top 6 parameter combinations."""
    # Sort by silhouette score
    sorted_results = sorted(all_results, key=lambda x: x['silhouette'], reverse=True)[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(sorted_results):
        ax = axes[idx]
        
        n_neighbors = result['n_neighbors']
        min_dist = result['min_dist']
        spread = result['spread']
        score = result['silhouette']
        
        embedding, _ = compute_umap_with_score(latents, labels, n_neighbors, min_dist, spread)
        
        if embedding is not None:
            xlim, ylim = get_axis_limits(embedding, percentile=99)
            
            n_clusters = len(np.unique(labels))
            colors = CLUSTER_PALETTE[:n_clusters]
            
            for i, cluster_id in enumerate(sorted(np.unique(labels))):
                mask = labels == cluster_id
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[i]],
                    s=10,
                    alpha=0.6,
                )
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        
        rank = idx + 1
        ax.set_title(f"#{rank}: n={n_neighbors}, d={min_dist}, s={spread}\nSilhouette: {score:.4f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#fafafa')
    
    plt.suptitle(f"{subgroup} - Top 6 UMAP Configurations (by 2D Silhouette)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"{subgroup}_umap_top6.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Main optimization function."""
    if not HAS_UMAP:
        print("UMAP not available, cannot run optimization")
        return
    
    print("=" * 70)
    print("Optimizing UMAP Parameters for Cluster Visualization")
    print("=" * 70)
    
    # Create output directory
    test_dir = PLOTS_DIR / "umap_tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Process each subgroup
    for subgroup, result in results.items():
        print(f"\n{'='*60}")
        print(f"Optimizing: {subgroup}")
        print(f"{'='*60}")
        
        if "optimized_ae" in result:
            result = result["optimized_ae"]
        
        latents = result["latents"]
        labels = result["labels"]
        
        print(f"Samples: {len(latents)}, Clusters: {len(np.unique(labels))}")
        
        # Grid search
        best_params, best_score, best_embedding, all_results = grid_search_umap(
            latents, labels, verbose=True
        )
        
        if best_params is not None:
            n_neighbors, min_dist, spread = best_params
            
            print(f"\n*** OPTIMAL PARAMETERS ***")
            print(f"  n_neighbors: {n_neighbors}")
            print(f"  min_dist: {min_dist}")
            print(f"  spread: {spread}")
            print(f"  2D Silhouette: {best_score:.4f}")
            
            # Save optimized plot
            plot_optimized(
                best_embedding, labels, best_params, best_score,
                subgroup, test_dir / f"{subgroup}_umap_optimized.png"
            )
            print(f"\n  Saved: {subgroup}_umap_optimized.png")
            
            # Save top 6 comparison
            plot_top_results(latents, labels, all_results, subgroup, test_dir)
            print(f"  Saved: {subgroup}_umap_top6.png")
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {test_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

