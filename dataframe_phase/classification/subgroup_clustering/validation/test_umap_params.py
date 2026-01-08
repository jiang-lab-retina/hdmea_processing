"""
Test different UMAP parameter combinations for visualization.
Generates multiple plots for comparison.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Different parameter combinations to test (n_neighbors=20 fixed)
PARAM_COMBINATIONS = [
    # (n_neighbors, min_dist, spread, name)
    (20, 0.0, 1.0, "n20_d0.0_s1.0"),       # min_dist=0, tight packing
    (20, 0.01, 1.0, "n20_d0.01_s1.0"),     # Very low min_dist
    (20, 0.05, 1.0, "n20_d0.05_s1.0"),     # Low min_dist
    (20, 0.1, 1.0, "n20_d0.1_s1.0"),       # Default min_dist
    (20, 0.01, 1.5, "n20_d0.01_s1.5"),     # Tight + more spread
    (20, 0.01, 2.0, "n20_d0.01_s2.0"),     # Tight + max spread
    (20, 0.05, 1.5, "n20_d0.05_s1.5"),     # Low dist + spread
    (20, 0.05, 2.0, "n20_d0.05_s2.0"),     # Low dist + max spread
]


def load_results():
    """Load results."""
    results_file = OUTPUT_DIR / "optimized_results.pkl"
    with open(results_file, "rb") as f:
        return pickle.load(f)


def compute_umap(X, n_neighbors, min_dist, spread, random_state=42):
    """Compute UMAP with specific parameters."""
    # PCA pre-reduction
    if X.shape[1] > 50:
        pca = PCA(n_components=min(50, X.shape[0] - 1), random_state=random_state)
        X = pca.fit_transform(X)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=2,
        random_state=random_state,
        n_jobs=1,
    )
    return reducer.fit_transform(X)


def get_axis_limits(embedding, percentile=99):
    """
    Get axis limits that exclude outliers.
    Uses percentile to determine bounds.
    """
    x_low = np.percentile(embedding[:, 0], 100 - percentile)
    x_high = np.percentile(embedding[:, 0], percentile)
    y_low = np.percentile(embedding[:, 1], 100 - percentile)
    y_high = np.percentile(embedding[:, 1], percentile)
    
    # Add 10% padding
    x_range = x_high - x_low
    y_range = y_high - y_low
    x_low -= 0.1 * x_range
    x_high += 0.1 * x_range
    y_low -= 0.1 * y_range
    y_high += 0.1 * y_range
    
    return (x_low, x_high), (y_low, y_high)


def plot_single(embedding, labels, title, save_path, clip_outliers=True):
    """Plot a single UMAP with optional outlier clipping."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    # Clip axes to exclude outliers
    if clip_outliers:
        xlim, ylim = get_axis_limits(embedding, percentile=99)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Generate comparison plots."""
    if not HAS_UMAP:
        print("UMAP not available, cannot run this script")
        return
    
    print("=" * 70)
    print("Testing UMAP Parameter Combinations")
    print("=" * 70)
    
    # Create output directory
    test_dir = PLOTS_DIR / "umap_tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Find first available subgroup
    subgroup = list(results.keys())[0]
    result = results[subgroup]
    
    if "optimized_ae" in result:
        result = result["optimized_ae"]
    
    latents = result["latents"]
    labels = result["labels"]
    
    print(f"\nUsing subgroup: {subgroup}")
    print(f"Samples: {len(latents)}, Clusters: {len(np.unique(labels))}")
    print(f"\nGenerating {len(PARAM_COMBINATIONS)} plots...")
    
    # Generate plots for each combination
    for n_neighbors, min_dist, spread, name in PARAM_COMBINATIONS:
        print(f"\n  [{name}] n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}")
        
        try:
            embedding = compute_umap(latents, n_neighbors, min_dist, spread)
            
            title = (f"{subgroup} - {name}\n"
                    f"n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}")
            
            save_path = test_dir / f"{subgroup}_umap_{name}.png"
            plot_single(embedding, labels, title, save_path)
            print(f"    Saved: {save_path.name}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Create comparison grid
    print("\n  Creating comparison grid...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (n_neighbors, min_dist, spread, name) in enumerate(PARAM_COMBINATIONS):
        ax = axes[idx]
        
        try:
            embedding = compute_umap(latents, n_neighbors, min_dist, spread)
            
            # Get axis limits excluding outliers
            xlim, ylim = get_axis_limits(embedding, percentile=99)
            
            n_clusters = len(np.unique(labels))
            colors = CLUSTER_PALETTE[:n_clusters]
            
            for i, cluster_id in enumerate(sorted(np.unique(labels))):
                mask = labels == cluster_id
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[i]],
                    s=8,
                    alpha=0.6,
                )
            
            # Apply axis limits to exclude outliers
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            ax.set_title(f"{name}\nn={n_neighbors}, d={min_dist}, s={spread}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#fafafa')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', fontsize=8)
            ax.set_title(name, fontsize=10)
    
    plt.suptitle(f"{subgroup} - UMAP Parameter Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(test_dir / f"{subgroup}_umap_comparison_grid.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n" + "=" * 70)
    print(f"All plots saved to: {test_dir}")
    print("=" * 70)
    print("\nPlots generated:")
    for n_neighbors, min_dist, spread, name in PARAM_COMBINATIONS:
        print(f"  - {subgroup}_umap_{name}.png")
    print(f"  - {subgroup}_umap_comparison_grid.png (all in one)")


if __name__ == "__main__":
    main()

