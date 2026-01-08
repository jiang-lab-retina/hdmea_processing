"""
RGC Classification Validation Plots.

Visualizes the autoencoder latent space using UMAP/t-SNE to show
clustering of RGC subgroups (DSGC, OSGC, ipRGC, Other).

Usage:
    python validation.py
"""

import os
# Disable joblib parallelization to avoid psutil issues on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Tuple

# Try to import UMAP, fall back to t-SNE if unavailable
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use t-SNE instead")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
INPUT_PARQUET = Path(__file__).parent.parent / "rgc_classified_with_ae20251230.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots"

# Latent dimension
LATENT_DIM = 100

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = 42

# Subtype colors - distinctive and colorblind-friendly
SUBTYPE_COLORS = {
    "ipRGC": "#E63946",     # Vibrant red
    "DSGC": "#457B9D",      # Steel blue
    "OSGC": "#2A9D8F",      # Teal
    "Other": "#8D99AE",     # Slate gray
}

# Subtype order for legend
SUBTYPE_ORDER = ["ipRGC", "DSGC", "OSGC", "Other"]

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(parquet_path: Path) -> pd.DataFrame:
    """
    Load classified RGC data with latent codes.
    
    Parameters
    ----------
    parquet_path : Path
        Path to input parquet file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with latent codes and subtype labels
    """
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} units")
    
    # Verify latent columns exist
    latent_cols = [f"AE_latent_{i}" for i in range(LATENT_DIM)]
    missing = [col for col in latent_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing latent columns: {missing[:5]}...")
    
    print(f"  Found {LATENT_DIM} latent dimensions")
    
    # Print subtype distribution
    print("\n  Subtype distribution:")
    for subtype in SUBTYPE_ORDER:
        count = (df["rgc_subtype"] == subtype).sum()
        pct = 100 * count / len(df)
        print(f"    {subtype}: {count} ({pct:.1f}%)")
    
    return df


def extract_latent_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract latent codes as numpy matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latent columns
        
    Returns
    -------
    np.ndarray
        Latent codes matrix of shape [N, LATENT_DIM]
    """
    latent_cols = [f"AE_latent_{i}" for i in range(LATENT_DIM)]
    X = df[latent_cols].values
    return X


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

def compute_embedding(
    X: np.ndarray,
    method: str = "auto",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Compute 2D embedding using UMAP or t-SNE.
    
    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape [N, D]
    method : str
        "umap", "tsne", or "auto" (tries UMAP first, falls back to t-SNE)
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance parameter for UMAP
    perplexity : float
        Perplexity for t-SNE
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        (embedding array of shape [N, 2], method used)
    """
    # First reduce with PCA if high dimensional
    if X.shape[1] > 50:
        print(f"\nApplying PCA: {X.shape[1]} -> 50 dimensions...")
        pca = PCA(n_components=50, random_state=random_state)
        X_reduced = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained_var:.2%}")
    else:
        X_reduced = X
    
    # Try UMAP first if available and requested
    if method in ("auto", "umap") and UMAP_AVAILABLE:
        try:
            print(f"\nComputing UMAP embedding...")
            print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=random_state,
                n_components=2,
                n_jobs=1,
                verbose=False,
            )
            
            embedding = reducer.fit_transform(X_reduced)
            print(f"  Embedding shape: {embedding.shape}")
            return embedding, "UMAP"
        except Exception as e:
            print(f"  UMAP failed: {e}")
            if method == "umap":
                raise
            print("  Falling back to t-SNE...")
    
    # Use t-SNE as fallback or if requested
    print(f"\nComputing t-SNE embedding...")
    print(f"  perplexity={perplexity}")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    
    embedding = tsne.fit_transform(X_reduced)
    print(f"  Embedding shape: {embedding.shape}")
    
    return embedding, "t-SNE"


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_embedding_by_subtype(
    embedding: np.ndarray,
    subtypes: np.ndarray,
    method_name: str = "Embedding",
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 10),
    point_size: int = 8,
    alpha: float = 0.6,
):
    """
    Plot 2D embedding colored by RGC subtype.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding of shape [N, 2]
    subtypes : np.ndarray
        Array of subtype labels
    method_name : str
        Name of embedding method (for labels)
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    point_size : int
        Scatter point size
    alpha : float
        Point transparency
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each subtype separately for legend control
    for subtype in SUBTYPE_ORDER:
        mask = subtypes == subtype
        count = mask.sum()
        color = SUBTYPE_COLORS[subtype]
        
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=point_size,
            alpha=alpha,
            label=f"{subtype} (n={count:,})",
            edgecolors="none",
        )
    
    ax.set_xlabel(f"{method_name} 1", fontsize=13)
    ax.set_ylabel(f"{method_name} 2", fontsize=13)
    ax.set_title(f"RGC Subtype Clustering in Autoencoder Latent Space ({method_name})", fontsize=15, fontweight="bold")
    
    # Legend with custom styling
    legend = ax.legend(
        loc="upper right",
        framealpha=0.95,
        edgecolor="#333333",
        fancybox=False,
        markerscale=2,
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n  Saved: {output_path}")
    
    plt.close()


def plot_embedding_subtype_grid(
    embedding: np.ndarray,
    subtypes: np.ndarray,
    method_name: str = "Embedding",
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 12),
    point_size: int = 6,
):
    """
    Plot 2D embedding with each subtype highlighted in its own subplot.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding of shape [N, 2]
    subtypes : np.ndarray
        Array of subtype labels
    method_name : str
        Name of embedding method (for labels)
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    point_size : int
        Scatter point size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Get axis limits from all data
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    
    for idx, subtype in enumerate(SUBTYPE_ORDER):
        ax = axes[idx]
        mask = subtypes == subtype
        count = mask.sum()
        color = SUBTYPE_COLORS[subtype]
        
        # Plot background points (other subtypes) in light gray
        ax.scatter(
            embedding[~mask, 0],
            embedding[~mask, 1],
            c="#E0E0E0",
            s=point_size * 0.5,
            alpha=0.3,
            edgecolors="none",
        )
        
        # Plot highlighted subtype
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=point_size,
            alpha=0.7,
            edgecolors="none",
        )
        
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel(f"{method_name} 1", fontsize=11)
        ax.set_ylabel(f"{method_name} 2", fontsize=11)
        ax.set_title(f"{subtype} (n={count:,})", fontsize=13, fontweight="bold", color=color)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"RGC Subtypes in Latent Space - {method_name} (Individual Highlights)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {output_path}")
    
    plt.close()


def plot_embedding_density(
    embedding: np.ndarray,
    subtypes: np.ndarray,
    method_name: str = "Embedding",
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 12),
):
    """
    Plot 2D embedding with density contours for each subtype.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding of shape [N, 2]
    subtypes : np.ndarray
        Array of subtype labels
    method_name : str
        Name of embedding method (for labels)
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    from scipy.stats import gaussian_kde
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Get axis limits
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    # Create grid for density estimation
    xx, yy = np.mgrid[
        x_min - x_pad : x_max + x_pad : 100j,
        y_min - y_pad : y_max + y_pad : 100j,
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    for idx, subtype in enumerate(SUBTYPE_ORDER):
        ax = axes[idx]
        mask = subtypes == subtype
        count = mask.sum()
        color = SUBTYPE_COLORS[subtype]
        
        # Plot all points in light gray
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c="#E8E8E8",
            s=3,
            alpha=0.3,
            edgecolors="none",
        )
        
        # Compute and plot density contours
        if count > 10:
            try:
                kernel = gaussian_kde(embedding[mask].T)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Contour levels
                levels = np.linspace(f.max() * 0.1, f.max() * 0.9, 5)
                ax.contour(xx, yy, f, levels=levels, colors=[color], linewidths=1.5, alpha=0.8)
                ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.2)
            except Exception:
                pass
        
        # Plot subtype points
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=5,
            alpha=0.5,
            edgecolors="none",
        )
        
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel(f"{method_name} 1", fontsize=11)
        ax.set_ylabel(f"{method_name} 2", fontsize=11)
        ax.set_title(f"{subtype} Density (n={count:,})", fontsize=13, fontweight="bold", color=color)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"RGC Subtype Density in Latent Space ({method_name})", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {output_path}")
    
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Generate embedding validation plots for RGC classification."""
    print("=" * 80)
    print("RGC Classification Validation - Embedding Plots")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(INPUT_PARQUET)
    
    # Extract latent matrix
    X = extract_latent_matrix(df)
    subtypes = df["rgc_subtype"].values
    
    # Compute 2D embedding (UMAP or t-SNE)
    embedding, method_name = compute_embedding(
        X,
        method="auto",
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        perplexity=30.0,
        random_state=UMAP_RANDOM_STATE,
    )
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Determine filename prefix based on method
    prefix = method_name.lower().replace("-", "")
    
    print(f"\n[1/3] {method_name} by subtype (combined)...")
    plot_embedding_by_subtype(
        embedding,
        subtypes,
        method_name=method_name,
        output_path=OUTPUT_DIR / f"{prefix}_rgc_subtypes.png",
    )
    
    print(f"[2/3] {method_name} subtype grid (individual highlights)...")
    plot_embedding_subtype_grid(
        embedding,
        subtypes,
        method_name=method_name,
        output_path=OUTPUT_DIR / f"{prefix}_rgc_subtypes_grid.png",
    )
    
    print(f"[3/3] {method_name} density contours...")
    plot_embedding_density(
        embedding,
        subtypes,
        method_name=method_name,
        output_path=OUTPUT_DIR / f"{prefix}_rgc_subtypes_density.png",
    )
    
    print("\n" + "=" * 80)
    print("Done! Plots saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()

