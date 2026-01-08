"""
Evaluation module for the Baden-method RGC clustering pipeline.

This module implements:
- Posterior probability separability curves
- Bootstrap stability analysis
- Cluster center matching via correlation
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from . import config
from . import clustering

logger = logging.getLogger(__name__)


# =============================================================================
# Posterior Probability Curves
# =============================================================================

def compute_posterior_curves(
    labels: np.ndarray,
    posteriors: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Compute rank-ordered posterior probability curves per cluster.
    
    For each cluster, cells are sorted by their posterior probability
    for that cluster (descending), then the x-axis is normalized by
    cluster size to enable comparison across clusters.
    
    Args:
        labels: Cluster assignments of shape (n_samples,).
        posteriors: Posterior probabilities of shape (n_samples, n_clusters).
        
    Returns:
        Dictionary mapping cluster ID to (x, y) curves.
        x: Normalized rank (0 to 1)
        y: Posterior probability at that rank
    """
    logger.info("Computing posterior separability curves...")
    
    n_clusters = posteriors.shape[1]
    curves = {}
    
    for k in range(n_clusters):
        # Get cells assigned to this cluster
        cluster_mask = labels == k
        n_in_cluster = cluster_mask.sum()
        
        if n_in_cluster == 0:
            logger.warning(f"Cluster {k} has no assigned cells")
            curves[k] = (np.array([0, 1]), np.array([0, 0]))
            continue
        
        # Get posterior probabilities for this cluster
        cluster_posteriors = posteriors[cluster_mask, k]
        
        # Sort descending
        sorted_posteriors = np.sort(cluster_posteriors)[::-1]
        
        # Normalize x-axis by cluster size
        x = np.linspace(0, 1, len(sorted_posteriors))
        y = sorted_posteriors
        
        curves[k] = (x, y)
    
    logger.info(f"Computed curves for {len(curves)} clusters")
    return curves


def compute_average_posterior_curve(
    curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average posterior curve across all clusters.
    
    Args:
        curves: Dictionary of per-cluster curves.
        n_points: Number of points for interpolation.
        
    Returns:
        Tuple of (x, y_mean) for the average curve.
    """
    x_common = np.linspace(0, 1, n_points)
    y_values = []
    
    for k, (x, y) in curves.items():
        # Interpolate to common x grid
        y_interp = np.interp(x_common, x, y)
        y_values.append(y_interp)
    
    y_mean = np.mean(y_values, axis=0)
    
    return x_common, y_mean


# =============================================================================
# Bootstrap Stability Analysis
# =============================================================================

def match_cluster_centers(
    original_means: np.ndarray,
    bootstrap_means: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """
    Match bootstrap cluster centers to original centers using maximum correlation.
    
    Args:
        original_means: Original cluster centers of shape (k, n_features).
        bootstrap_means: Bootstrap cluster centers of shape (k, n_features).
        
    Returns:
        Tuple of (matching_indices, correlations):
            - matching_indices: For each original cluster, the best-matching bootstrap cluster
            - correlations: Correlation values for each match
    """
    k_orig = len(original_means)
    k_boot = len(bootstrap_means)
    
    if k_orig != k_boot:
        logger.warning(f"Cluster count mismatch: original={k_orig}, bootstrap={k_boot}")
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(original_means, bootstrap_means)[:k_orig, k_orig:]
    
    # Greedy matching (could use Hungarian algorithm for optimal)
    matching = []
    correlations = []
    used = set()
    
    for i in range(k_orig):
        # Find best unused match
        corr_row = corr_matrix[i].copy()
        for j in used:
            corr_row[j] = -np.inf
        
        best_j = np.argmax(corr_row)
        matching.append(best_j)
        correlations.append(corr_row[best_j])
        used.add(best_j)
    
    return matching, np.array(correlations)


def _run_single_bootstrap(args):
    """Helper function for parallel bootstrap iteration."""
    X, k, idx, original_means, random_state, iteration = args
    try:
        X_boot = X[idx]
        boot_gmm = clustering.fit_gmm(X_boot, k, random_state=random_state + iteration + 1)
        boot_means = boot_gmm.means_
        
        # Match and compute correlations
        _, correlations = match_cluster_centers(original_means, boot_means)
        median_corr = np.median(correlations)
        return median_corr
    except Exception as e:
        return np.nan


def bootstrap_stability(
    X: np.ndarray,
    k: int,
    n_iter: int = None,
    frac: float = None,
    random_state: int = 42,
    show_progress: bool = True,
    n_jobs: int = None,
) -> Tuple[float, List[float]]:
    """
    Assess cluster stability via bootstrap resampling.
    
    For each iteration:
    1. Subsample frac of the data
    2. Refit GMM with k clusters
    3. Match bootstrap clusters to original via correlation
    4. Record median correlation across clusters
    
    Uses parallel processing for speedup when n_jobs != 1.
    
    Args:
        X: Standardized feature matrix.
        k: Number of clusters.
        n_iter: Number of bootstrap iterations. Defaults to config.BOOTSTRAP_N_ITERATIONS.
        frac: Fraction of data to sample. Defaults to config.BOOTSTRAP_SAMPLE_FRACTION.
        random_state: Random seed.
        show_progress: Whether to show progress bar.
        n_jobs: Number of parallel jobs. -1 uses all CPUs. Defaults to config.N_JOBS_BOOTSTRAP.
        
    Returns:
        Tuple of (median_correlation, list_of_iteration_correlations).
    """
    if n_iter is None:
        n_iter = config.BOOTSTRAP_N_ITERATIONS
    if frac is None:
        frac = config.BOOTSTRAP_SAMPLE_FRACTION
    if n_jobs is None:
        n_jobs = getattr(config, 'N_JOBS_BOOTSTRAP', 1)
    
    logger.info(f"Running bootstrap stability: {n_iter} iterations, {frac:.0%} sampling")
    
    # Fit original model
    original_gmm = clustering.fit_gmm(X, k, random_state=random_state)
    original_means = original_gmm.means_
    
    # Pre-generate all random indices for reproducibility
    rng = np.random.RandomState(random_state)
    n_samples = int(len(X) * frac)
    all_indices = [rng.choice(len(X), n_samples, replace=False) for _ in range(n_iter)]
    
    # Prepare arguments for each iteration
    args_list = [(X, k, idx, original_means, random_state, i) for i, idx in enumerate(all_indices)]
    
    # Use parallel processing if n_jobs != 1
    if n_jobs != 1:
        from joblib import Parallel, delayed
        import multiprocessing
        
        if n_jobs == -1:
            # Use configured fraction of CPU cores to avoid system overload
            cpu_frac = getattr(config, 'CPU_FRACTION', 0.8)
            n_jobs = max(1, int(multiprocessing.cpu_count() * cpu_frac))
            n_jobs = min(n_jobs, n_iter)
        
        logger.info(f"Using parallel bootstrap with {n_jobs} workers...")
        
        # Use loky backend (more robust than threads)
        all_correlations = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_single_bootstrap)(args)
            for args in tqdm(args_list, desc="Bootstrap (parallel)", disable=not show_progress)
        )
    else:
        # Sequential execution with progress bar
        all_correlations = []
        iterator = tqdm(range(n_iter), desc="Bootstrap", disable=not show_progress)
        for i in iterator:
            result = _run_single_bootstrap(args_list[i])
            all_correlations.append(result)
            if not np.isnan(result):
                iterator.set_postfix({'median_corr': f'{result:.3f}'})
            else:
                logger.warning(f"Bootstrap iteration {i} failed")
    
    # Compute overall median
    valid_corrs = [c for c in all_correlations if not np.isnan(c)]
    if len(valid_corrs) == 0:
        raise ValueError("All bootstrap iterations failed")
    
    median_stability = np.median(valid_corrs)
    
    # Check stability threshold
    if median_stability < config.STABILITY_THRESHOLD:
        logger.warning(f"Cluster stability ({median_stability:.3f}) below threshold ({config.STABILITY_THRESHOLD})")
    else:
        logger.info(f"Cluster stability: {median_stability:.3f} (threshold: {config.STABILITY_THRESHOLD})")
    
    return median_stability, all_correlations


# =============================================================================
# Evaluation Summary
# =============================================================================

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    posteriors: np.ndarray,
    k: int,
    random_state: int = 42,
    run_bootstrap: bool = True,
    show_progress: bool = True,
) -> Dict:
    """
    Run full clustering evaluation.
    
    Args:
        X: Standardized feature matrix.
        labels: Cluster assignments.
        posteriors: Posterior probabilities.
        k: Number of clusters.
        random_state: Random seed.
        run_bootstrap: Whether to run bootstrap stability analysis.
        show_progress: Whether to show progress bars.
        
    Returns:
        Dictionary with evaluation results.
    """
    logger.info("Running clustering evaluation...")
    
    results = {
        'k': k,
        'n_cells': len(labels),
    }
    
    # Posterior curves
    curves = compute_posterior_curves(labels, posteriors)
    results['posterior_curves'] = curves
    
    # Average curve
    x_avg, y_avg = compute_average_posterior_curve(curves)
    results['average_posterior_curve'] = (x_avg, y_avg)
    
    # Cluster sizes
    cluster_sizes = {}
    for i in range(k):
        cluster_sizes[i] = int((labels == i).sum())
    results['cluster_sizes'] = cluster_sizes
    
    # Max posterior statistics
    max_posteriors = posteriors.max(axis=1)
    results['max_posterior_mean'] = float(max_posteriors.mean())
    results['max_posterior_std'] = float(max_posteriors.std())
    results['max_posterior_min'] = float(max_posteriors.min())
    
    # Bootstrap stability
    if run_bootstrap:
        median_stability, all_corrs = bootstrap_stability(
            X, k, random_state=random_state, show_progress=show_progress
        )
        results['bootstrap_median_correlation'] = float(median_stability)
        results['bootstrap_all_correlations'] = [float(c) for c in all_corrs if not np.isnan(c)]
        results['is_stable'] = median_stability >= config.STABILITY_THRESHOLD
    else:
        results['bootstrap_median_correlation'] = None
        results['bootstrap_all_correlations'] = []
        results['is_stable'] = None
    
    logger.info("Evaluation complete")
    return results

