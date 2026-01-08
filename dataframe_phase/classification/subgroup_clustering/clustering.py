"""
Clustering Methods for Subgroup Analysis.

Implements K-Means/GMM clustering with automatic k selection using silhouette score,
constrained to expected k ranges per subgroup.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Optional, Any
import warnings

from .config import (
    GMM_K_MIN, GMM_K_MAX, GMM_COVARIANCE_TYPE, GMM_REG_COVAR,
    EXPECTED_K_RANGES, RANDOM_SEED, USE_KMEANS,
)


def fit_clustering_with_expected_k(
    X: np.ndarray,
    subgroup: str,
    method: str = "kmeans",
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> Tuple[np.ndarray, int, Any, Dict]:
    """
    Fit clustering with k selection within expected range for subgroup.
    
    Uses silhouette score to find optimal k within biologically expected range.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix [N, D]
    subgroup : str
        Subgroup name (determines expected k range)
    method : str
        'kmeans' or 'gmm'
    random_state : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (cluster_labels, optimal_k, model, results_dict)
    """
    # Get expected k range for subgroup
    k_min, k_max = EXPECTED_K_RANGES.get(subgroup, (GMM_K_MIN, GMM_K_MAX))
    
    n_samples = X.shape[0]
    
    # Ensure we have enough samples
    min_samples_per_cluster = 5
    k_max = min(k_max, n_samples // min_samples_per_cluster)
    k_min = max(2, min(k_min, k_max))
    
    if verbose:
        print(f"    Fitting {method.upper()} with k in [{k_min}, {k_max}] (expected for {subgroup})")
    
    results = {
        "k_values": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "inertia": [],
    }
    
    best_sil = -1
    best_k = k_min
    best_labels = None
    best_model = None
    
    for k in range(k_min, k_max + 1):
        try:
            if method == "gmm":
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=GMM_COVARIANCE_TYPE,
                    reg_covar=GMM_REG_COVAR,
                    random_state=random_state,
                    n_init=10,
                    max_iter=300,
                )
                model.fit(X)
                labels = model.predict(X)
                inertia = -model.score(X) * n_samples
            else:  # kmeans
                model = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=20,
                    max_iter=500,
                )
                labels = model.fit_predict(X)
                inertia = model.inertia_
            
            # Compute metrics
            n_unique = len(np.unique(labels))
            if n_unique > 1 and n_unique < n_samples:
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
            else:
                sil = -1
                ch = 0
            
            results["k_values"].append(k)
            results["silhouette"].append(sil)
            results["calinski_harabasz"].append(ch)
            results["inertia"].append(inertia)
            
            if verbose and k % 2 == 0:
                print(f"      k={k}: silhouette={sil:.3f}")
            
            # Select best by silhouette
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels.copy()
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"      k={k} failed: {e}")
            continue
    
    # Fallback if all failed
    if best_labels is None:
        if verbose:
            print(f"    All k values failed, using k={k_min}")
        model = KMeans(n_clusters=k_min, random_state=random_state, n_init=10)
        best_labels = model.fit_predict(X)
        best_k = k_min
        best_model = model
        best_sil = silhouette_score(X, best_labels) if len(np.unique(best_labels)) > 1 else 0
    
    if verbose:
        print(f"    ✓ Optimal k={best_k} (silhouette={best_sil:.3f})")
    
    return best_labels, best_k, best_model, results


def fit_gmm_auto_k(
    X: np.ndarray,
    k_min: int = GMM_K_MIN,
    k_max: int = GMM_K_MAX,
    covariance_type: str = GMM_COVARIANCE_TYPE,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
    subgroup: str = None,
) -> Tuple[np.ndarray, int, GaussianMixture, Dict]:
    """
    Fit GMM with automatic k selection.
    
    If subgroup is provided, uses expected k range for that subgroup.
    """
    if subgroup and subgroup in EXPECTED_K_RANGES:
        k_min, k_max = EXPECTED_K_RANGES[subgroup]
    
    method = "kmeans" if USE_KMEANS else "gmm"
    
    return fit_clustering_with_expected_k(
        X, subgroup or "Other", method=method,
        random_state=random_state, verbose=verbose
    )


def fit_kmeans_auto_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
    subgroup: str = None,
) -> Tuple[np.ndarray, int, Dict]:
    """
    Fit K-Means with automatic k selection using silhouette score.
    
    If subgroup is provided, uses expected k range for that subgroup.
    """
    if subgroup and subgroup in EXPECTED_K_RANGES:
        k_min, k_max = EXPECTED_K_RANGES[subgroup]
    
    labels, optimal_k, model, results = fit_clustering_with_expected_k(
        X, subgroup or "Other", method="kmeans",
        random_state=random_state, verbose=verbose
    )
    
    return labels, optimal_k, results


def compute_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix [N, D]
    labels : np.ndarray
        Cluster labels [N]
        
    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    n_clusters = len(np.unique(labels))
    
    metrics = {
        "n_clusters": n_clusters,
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    
    if n_clusters < 2:
        return metrics
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return metrics
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            metrics["silhouette"] = silhouette_score(X, labels)
        except Exception:
            pass
        
        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except Exception:
            pass
        
        try:
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        except Exception:
            pass
    
    return metrics


def reduce_dims_for_clustering(
    X: np.ndarray,
    target_dim: int = 20,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Reduce dimensionality with PCA before clustering if needed.
    """
    if X.shape[1] <= target_dim:
        return X
    
    pca = PCA(n_components=target_dim, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    
    return X_reduced


def get_cluster_summary(labels: np.ndarray) -> Dict[int, int]:
    """Get summary of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))
