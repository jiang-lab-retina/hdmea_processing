"""
Compare Results and Generate Visualizations.

Creates UMAP plots for each subgroup/approach and comparison tables.

Usage:
    python compare_results.py
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Try UMAP, fall back to t-SNE
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

from ..config import (
    SUBGROUPS,
    SUBGROUP_COLORS,
    CLUSTER_PALETTE,
    APPROACH_NAMES,
    OUTPUT_DIR,
    PLOTS_DIR,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_RANDOM_STATE,
)


def compute_2d_embedding(
    X: np.ndarray,
    method: str = "auto",
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Compute 2D embedding using UMAP or t-SNE.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix [N, D]
    method : str
        'umap', 'tsne', or 'auto'
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        (embedding, method_used)
    """
    # PCA reduction if high dimensional
    if X.shape[1] > 30:
        pca = PCA(n_components=30, random_state=random_state)
        X = pca.fit_transform(X)
    
    # Try UMAP first
    if method in ("auto", "umap") and UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(
                n_neighbors=min(UMAP_N_NEIGHBORS, len(X) - 1),
                min_dist=UMAP_MIN_DIST,
                random_state=random_state,
                n_components=2,
                n_jobs=1,
            )
            embedding = reducer.fit_transform(X)
            return embedding, "UMAP"
        except Exception:
            pass
    
    # Fall back to t-SNE
    perplexity = min(30, len(X) // 4)
    perplexity = max(5, perplexity)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    embedding = tsne.fit_transform(X)
    return embedding, "t-SNE"


def plot_subgroup_clusters(
    latents: np.ndarray,
    labels: np.ndarray,
    subgroup: str,
    approach: str,
    output_path: Path,
    figsize: Tuple[int, int] = (8, 7),
):
    """
    Plot UMAP/t-SNE for a single subgroup with cluster coloring.
    
    Parameters
    ----------
    latents : np.ndarray
        Latent codes [N, D]
    labels : np.ndarray
        Cluster labels [N]
    subgroup : str
        Subgroup name
    approach : str
        Approach name
    output_path : Path
        Save path
    figsize : tuple
        Figure size
    """
    # Compute embedding
    embedding, method = compute_2d_embedding(latents)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    for i, label in enumerate(sorted(unique_labels)):
        mask = labels == label
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        count = mask.sum()
        
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=15,
            alpha=0.7,
            label=f"Cluster {label} (n={count})",
            edgecolors="none",
        )
    
    ax.set_xlabel(f"{method} 1", fontsize=12)
    ax.set_ylabel(f"{method} 2", fontsize=12)
    ax.set_title(f"{subgroup} - {APPROACH_NAMES[approach]}\n{n_clusters} clusters", 
                 fontsize=13, fontweight="bold")
    
    ax.legend(loc="best", fontsize=9, framealpha=0.9, markerscale=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_approach_comparison(
    all_results: Dict[str, Dict[str, Any]],
    subgroup: str,
    output_path: Path,
    figsize: Tuple[int, int] = (18, 6),
):
    """
    Compare all 3 approaches side-by-side for a subgroup.
    
    Parameters
    ----------
    all_results : dict
        Results for all approaches
    subgroup : str
        Subgroup name
    output_path : Path
        Save path
    figsize : tuple
        Figure size
    """
    approaches = list(all_results.keys())
    n_approaches = len(approaches)
    
    fig, axes = plt.subplots(1, n_approaches, figsize=figsize)
    if n_approaches == 1:
        axes = [axes]
    
    for ax, approach in zip(axes, approaches):
        results = all_results[approach]
        latents = results["latents"]
        labels = results["labels"]
        metrics = results["metrics"]
        
        # Compute embedding
        embedding, method = compute_2d_embedding(latents)
        
        # Plot clusters
        unique_labels = np.unique(labels)
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
            
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color,
                s=10,
                alpha=0.6,
                edgecolors="none",
            )
        
        # Title with metrics
        sil = metrics["silhouette"]
        sil_str = f"{sil:.3f}" if not np.isnan(sil) else "N/A"
        
        ax.set_xlabel(f"{method} 1", fontsize=11)
        ax.set_ylabel(f"{method} 2", fontsize=11)
        ax.set_title(f"{APPROACH_NAMES[approach]}\nk={results['optimal_k']}, Sil={sil_str}", 
                     fontsize=12, fontweight="bold")
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"{subgroup} - Approach Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_all_subgroups_comparison(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
    figsize: Tuple[int, int] = (20, 16),
):
    """
    Create master comparison plot: 4 subgroups x 3 approaches.
    
    Parameters
    ----------
    all_results : dict
        Nested dict: subgroup -> approach -> results
    output_path : Path
        Save path
    figsize : tuple
        Figure size
    """
    subgroups = [s for s in SUBGROUPS if s in all_results]
    approaches = ["ae_gmm", "vae_gmm", "dec"]
    
    n_rows = len(subgroups)
    n_cols = len(approaches)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    for i, subgroup in enumerate(subgroups):
        for j, approach in enumerate(approaches):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            
            if approach not in all_results[subgroup]:
                ax.set_visible(False)
                continue
            
            results = all_results[subgroup][approach]
            latents = results["latents"]
            labels = results["labels"]
            metrics = results["metrics"]
            
            # Compute embedding
            embedding, method = compute_2d_embedding(latents)
            
            # Plot clusters
            unique_labels = np.unique(labels)
            for k, label in enumerate(sorted(unique_labels)):
                mask = labels == label
                color = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
                
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=color,
                    s=8,
                    alpha=0.6,
                    edgecolors="none",
                )
            
            # Labels
            if i == n_rows - 1:
                ax.set_xlabel(f"{method} 1", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{method} 2", fontsize=10)
            
            # Title
            sil = metrics["silhouette"]
            sil_str = f"{sil:.2f}" if not np.isnan(sil) else "N/A"
            
            if i == 0:
                ax.set_title(f"{APPROACH_NAMES[approach]}", fontsize=11, fontweight="bold")
            
            # Add subgroup label on left
            if j == 0:
                ax.text(-0.2, 0.5, subgroup, transform=ax.transAxes, fontsize=12, 
                        fontweight="bold", va="center", ha="right",
                        color=SUBGROUP_COLORS[subgroup])
            
            # Add metrics annotation
            ax.text(0.02, 0.98, f"k={results['optimal_k']}\nSil={sil_str}", 
                    transform=ax.transAxes, fontsize=9, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
    
    fig.suptitle("Subgroup Clustering: All Approaches Comparison", 
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def create_metrics_table(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
):
    """
    Create and save metrics comparison table.
    
    Parameters
    ----------
    all_results : dict
        Nested dict: subgroup -> approach -> results
    output_path : Path
        Save path for CSV
    """
    rows = []
    
    for subgroup, approaches in all_results.items():
        for approach, results in approaches.items():
            m = results["metrics"]
            rows.append({
                "Subgroup": subgroup,
                "Approach": APPROACH_NAMES[approach],
                "n_samples": len(results["labels"]),
                "n_clusters": int(results["optimal_k"]),
                "Silhouette": m["silhouette"],
                "Calinski-Harabasz": m["calinski_harabasz"],
                "Davies-Bouldin": m["davies_bouldin"],
            })
    
    df = pd.DataFrame(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot bar charts comparing metrics across approaches.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    output_path : Path
        Save path
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Colors for approaches
    approach_colors = {
        "Standard AE + GMM": "#3498DB",
        "VAE + GMM": "#E74C3C",
        "Deep Embedded Clustering": "#2ECC71",
    }
    
    subgroups = metrics_df["Subgroup"].unique()
    approaches = metrics_df["Approach"].unique()
    x = np.arange(len(subgroups))
    width = 0.25
    
    # Plot 1: Number of clusters
    ax = axes[0, 0]
    for i, approach in enumerate(approaches):
        data = metrics_df[metrics_df["Approach"] == approach]["n_clusters"].values
        ax.bar(x + i * width, data, width, label=approach, color=approach_colors.get(approach, "gray"))
    ax.set_xticks(x + width)
    ax.set_xticklabels(subgroups)
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Optimal k by Approach")
    ax.legend(fontsize=8)
    
    # Plot 2: Silhouette Score
    ax = axes[0, 1]
    for i, approach in enumerate(approaches):
        data = metrics_df[metrics_df["Approach"] == approach]["Silhouette"].values
        ax.bar(x + i * width, data, width, label=approach, color=approach_colors.get(approach, "gray"))
    ax.set_xticks(x + width)
    ax.set_xticklabels(subgroups)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score (higher is better)")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    
    # Plot 3: Calinski-Harabasz Index
    ax = axes[1, 0]
    for i, approach in enumerate(approaches):
        data = metrics_df[metrics_df["Approach"] == approach]["Calinski-Harabasz"].values
        ax.bar(x + i * width, data, width, label=approach, color=approach_colors.get(approach, "gray"))
    ax.set_xticks(x + width)
    ax.set_xticklabels(subgroups)
    ax.set_ylabel("Calinski-Harabasz Index")
    ax.set_title("Calinski-Harabasz Index (higher is better)")
    
    # Plot 4: Davies-Bouldin Index
    ax = axes[1, 1]
    for i, approach in enumerate(approaches):
        data = metrics_df[metrics_df["Approach"] == approach]["Davies-Bouldin"].values
        ax.bar(x + i * width, data, width, label=approach, color=approach_colors.get(approach, "gray"))
    ax.set_xticks(x + width)
    ax.set_xticklabels(subgroups)
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_title("Davies-Bouldin Index (lower is better)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    """Generate all comparison visualizations."""
    print("=" * 80)
    print("Subgroup Clustering - Results Comparison")
    print("=" * 80)
    
    # Load results
    results_path = OUTPUT_DIR / "all_results.pkl"
    
    if not results_path.exists():
        print(f"\nERROR: Results file not found: {results_path}")
        print("Run 'python -m dataframe_phase.classification.subgroup_clustering.run_all_approaches' first.")
        return
    
    print(f"\nLoading results from: {results_path}")
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Individual subgroup/approach plots
    print("\n[1/4] Generating individual cluster plots...")
    for subgroup, approaches in all_results.items():
        for approach, results in approaches.items():
            output_path = PLOTS_DIR / f"{subgroup}_{approach}_clusters.png"
            plot_subgroup_clusters(
                results["latents"],
                results["labels"],
                subgroup,
                approach,
                output_path,
            )
    print(f"  Saved {len(all_results) * 3} plots")
    
    # 2. Per-subgroup comparison plots
    print("[2/4] Generating per-subgroup comparison plots...")
    for subgroup, approaches in all_results.items():
        output_path = PLOTS_DIR / f"{subgroup}_approach_comparison.png"
        plot_approach_comparison(approaches, subgroup, output_path)
    print(f"  Saved {len(all_results)} plots")
    
    # 3. Master comparison plot
    print("[3/4] Generating master comparison plot...")
    plot_all_subgroups_comparison(all_results, PLOTS_DIR / "all_subgroups_comparison.png")
    print("  Saved all_subgroups_comparison.png")
    
    # 4. Metrics table and charts
    print("[4/4] Generating metrics comparison...")
    metrics_df = create_metrics_table(all_results, PLOTS_DIR / "metrics_comparison.csv")
    plot_metrics_comparison(metrics_df, PLOTS_DIR / "metrics_comparison.png")
    print("  Saved metrics_comparison.csv and metrics_comparison.png")
    
    # Print summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    
    # Identify best approach per subgroup
    print("\n" + "=" * 80)
    print("BEST APPROACH PER SUBGROUP (by Silhouette Score)")
    print("=" * 80)
    
    for subgroup in metrics_df["Subgroup"].unique():
        sub_df = metrics_df[metrics_df["Subgroup"] == subgroup]
        best_idx = sub_df["Silhouette"].idxmax()
        best_row = metrics_df.loc[best_idx]
        print(f"  {subgroup}: {best_row['Approach']} (Sil={best_row['Silhouette']:.3f}, k={int(best_row['n_clusters'])})")
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

