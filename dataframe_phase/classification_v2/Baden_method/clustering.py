"""
Clustering module for the Baden-method RGC clustering pipeline.

This module implements:
- Diagonal covariance GMM fitting with multiple restarts
- BIC-based model selection for optimal cluster count
- Log Bayes factor computation
- Cluster prediction with posterior probabilities
- GPU acceleration via PyTorch (optional)
"""

import logging
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)

# =============================================================================
# GPU Support Detection
# =============================================================================

# Try to import GPU GMM (PyTorch-based)
try:
    from .gpu_gmm import GaussianMixtureGPU, CUDA_AVAILABLE
    GPU_GMM_AVAILABLE = CUDA_AVAILABLE
    if GPU_GMM_AVAILABLE:
        logger.info("PyTorch CUDA detected - GPU acceleration available for GMM")
except ImportError:
    GPU_GMM_AVAILABLE = False
    logger.info("GPU GMM not available - using CPU for clustering")

# Check if we should use GPU (can be overridden in config)
USE_GPU = GPU_GMM_AVAILABLE and getattr(config, 'USE_GPU', True)


# =============================================================================
# GMM Fitting
# =============================================================================

def fit_gmm(
    X: np.ndarray,
    k: int,
    n_init: int = None,
    reg_covar: float = None,
    random_state: int = 42,
    use_gpu: bool = None,
):
    """
    Fit a diagonal covariance GMM with multiple restarts.
    
    Uses GPU acceleration if available and enabled.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        k: Number of clusters.
        n_init: Number of initializations. Defaults to config.GMM_N_INIT.
        reg_covar: Regularization added to diagonal. Defaults to config.GMM_REG_COVAR.
        random_state: Random seed.
        use_gpu: Whether to use GPU. Defaults to config.USE_GPU.
        
    Returns:
        Fitted GaussianMixture model (GPU or CPU).
    """
    if n_init is None:
        n_init = config.GMM_N_INIT
    if reg_covar is None:
        reg_covar = config.GMM_REG_COVAR
    if use_gpu is None:
        use_gpu = USE_GPU
    
    # Use GPU GMM if available and enabled
    if use_gpu and GPU_GMM_AVAILABLE:
        use_float64 = getattr(config, 'GPU_USE_FLOAT64', False)
        gmm = GaussianMixtureGPU(
            n_components=k,
            covariance_type='diag',
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=200,
            init_params='kmeans',
            use_float64=use_float64,
        )
    else:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=200,
            init_params='kmeans',
        )
    
    gmm.fit(X)
    
    return gmm


def _fit_single_k(args):
    """Helper function for parallel BIC fitting."""
    X, k, n_init, reg_covar, random_state = args
    n_samples = len(X)
    try:
        gmm = fit_gmm(X, k, n_init, reg_covar, random_state)
        
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        log_likelihood = gmm.score(X) * n_samples
        
        # Count parameters: means + diag covariances + weights
        n_features = X.shape[1]
        n_params = k * n_features + k * n_features + (k - 1)
        
        # Normalized BIC (per sample) for cross-sample-size comparison
        bic_normalized = bic / n_samples
        aic_normalized = aic / n_samples
        
        return {
            'k': k,
            'bic': bic,
            'bic_normalized': bic_normalized,
            'aic': aic,
            'aic_normalized': aic_normalized,
            'log_likelihood': log_likelihood,
            'n_parameters': n_params,
            'n_samples': n_samples,
            'converged': gmm.converged_,
        }
    except Exception as e:
        return {
            'k': k,
            'bic': np.inf,
            'bic_normalized': np.inf,
            'aic': np.inf,
            'aic_normalized': np.inf,
            'log_likelihood': -np.inf,
            'n_parameters': 0,
            'n_samples': n_samples,
            'converged': False,
            'error': str(e),
        }


# =============================================================================
# BIC-Based Model Selection
# =============================================================================

def fit_gmm_bic(
    X: np.ndarray,
    k_grid: range | list,
    n_init: int = None,
    reg_covar: float = None,
    random_state: int = 42,
    show_progress: bool = True,
    n_jobs: int = None,
) -> pd.DataFrame:
    """
    Fit GMMs for a range of k values and compute BIC.
    
    Uses parallel processing for speedup when n_jobs != 1.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        k_grid: Range or list of k values to evaluate.
        n_init: Number of initializations per k.
        reg_covar: Regularization added to diagonal.
        random_state: Random seed.
        show_progress: Whether to show progress bar.
        n_jobs: Number of parallel jobs. -1 uses all CPUs. Defaults to config.N_JOBS_BIC.
        
    Returns:
        DataFrame with columns: k, bic, log_likelihood, aic, n_parameters.
    """
    if n_jobs is None:
        n_jobs = getattr(config, 'N_JOBS_BIC', 1)
    
    k_list = list(k_grid)
    logger.info(f"Fitting GMM for k = {min(k_list)} to {max(k_list)}...")
    
    # Prepare arguments for each k (use same random_state for all k to match sequential behavior)
    args_list = [(X, k, n_init, reg_covar, random_state) for k in k_list]
    
    # Use parallel processing if n_jobs != 1
    if n_jobs != 1:
        from joblib import Parallel, delayed
        import multiprocessing
        
        if n_jobs == -1:
            # Use configured fraction of CPU cores to avoid system overload
            cpu_frac = getattr(config, 'CPU_FRACTION', 0.8)
            n_jobs = max(1, int(multiprocessing.cpu_count() * cpu_frac))
            n_jobs = min(n_jobs, len(k_list))
        
        logger.info(f"Using parallel BIC selection with {n_jobs} workers...")
        
        # Use joblib for parallel execution with loky backend (more robust)
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_single_k)(args) 
            for args in tqdm(args_list, desc="BIC selection (parallel)", disable=not show_progress)
        )
    else:
        # Sequential execution with progress bar
        results = []
        iterator = tqdm(args_list, desc="BIC selection", disable=not show_progress)
        for args in iterator:
            result = _fit_single_k(args)
            results.append(result)
            iterator.set_postfix({
                'k': result['k'], 
                'BIC': f"{result['bic']:.0f}",
                'BIC/n': f"{result['bic_normalized']:.4f}"
            })
    
    # Handle any errors
    for result in results:
        if 'error' in result:
            logger.warning(f"Failed to fit GMM with k={result['k']}: {result['error']}")
            del result['error']
    
    bic_table = pd.DataFrame(results)
    bic_table = bic_table.sort_values('k').reset_index(drop=True)
    logger.info(f"BIC range: {bic_table['bic'].min():.0f} to {bic_table['bic'].max():.0f}")
    logger.info(f"BIC/n range: {bic_table['bic_normalized'].min():.4f} to {bic_table['bic_normalized'].max():.4f}")
    
    return bic_table


def select_optimal_k(bic_table: pd.DataFrame) -> int:
    """
    Select optimal k using Log Bayes Factor stopping criterion.
    
    Stops when the evidence for adding more clusters becomes weak
    (log_bf < LOG_BF_THRESHOLD). Falls back to minimum BIC if
    log_bf never drops below threshold.
    
    Args:
        bic_table: DataFrame with 'k', 'bic', and 'log_bf' columns.
        
    Returns:
        Optimal number of clusters.
        
    Raises:
        ValueError: If no valid BIC values found.
    """
    # Filter out failed fits and sort by k
    valid = bic_table[bic_table['bic'] < np.inf].sort_values('k').reset_index(drop=True)
    
    if len(valid) == 0:
        raise ValueError("No valid GMM fits found")
    
    # Use Log Bayes Factor stopping criterion if available
    if 'log_bf' in valid.columns:
        for i in range(len(valid) - 1):
            log_bf = valid.iloc[i]['log_bf']
            k = int(valid.iloc[i]['k'])
            
            # Stop when evidence for k+1 over k becomes weak
            if pd.notna(log_bf) and log_bf < config.LOG_BF_THRESHOLD:
                optimal_k = k
                optimal_bic = valid.iloc[i]['bic']
                optimal_bic_norm = valid.iloc[i]['bic_normalized']
                
                logger.info(f"Optimal k = {optimal_k} (log_bf={log_bf:.2f} < {config.LOG_BF_THRESHOLD})")
                logger.info(f"  BIC = {optimal_bic:.0f}, BIC/n = {optimal_bic_norm:.4f}")
                return optimal_k
        
        # Log BF never dropped below threshold - use k_max
        logger.warning(f"Log BF never dropped below {config.LOG_BF_THRESHOLD}, using k_max")
    
    # Fallback: find minimum BIC
    optimal_idx = valid['bic'].idxmin()
    optimal_k = int(valid.loc[optimal_idx, 'k'])
    optimal_bic = valid.loc[optimal_idx, 'bic']
    optimal_bic_norm = valid.loc[optimal_idx, 'bic_normalized']
    
    logger.info(f"Optimal k = {optimal_k} (BIC = {optimal_bic:.0f}, BIC/n = {optimal_bic_norm:.4f})")
    
    # Check if BIC still decreasing at k_max
    k_max = valid['k'].max()
    if optimal_k == k_max:
        logger.warning(f"Optimal k = k_max ({k_max}). Consider increasing k_max or lowering LOG_BF_THRESHOLD.")
    
    return optimal_k


def compute_log_bayes_factors(bic_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log Bayes factors between adjacent k values.
    
    Log BF(k+1, k) ≈ -0.5 * (BIC_{k+1} - BIC_k)
    Values > 6 indicate strong evidence for k+1 over k.
    
    Args:
        bic_table: DataFrame with 'k' and 'bic' columns.
        
    Returns:
        DataFrame with added 'log_bf' column.
    """
    bic_table = bic_table.copy()
    bic_table = bic_table.sort_values('k').reset_index(drop=True)
    
    # Compute log Bayes factor for each k vs k+1
    log_bf = []
    for i in range(len(bic_table) - 1):
        bic_k = bic_table.loc[i, 'bic']
        bic_k1 = bic_table.loc[i + 1, 'bic']
        
        if np.isinf(bic_k) or np.isinf(bic_k1):
            log_bf.append(np.nan)
        else:
            # Positive log BF means k+1 is better than k
            lbf = -0.5 * (bic_k1 - bic_k)
            log_bf.append(lbf)
    
    log_bf.append(np.nan)  # No comparison for last k
    bic_table['log_bf'] = log_bf
    
    # Add interpretation
    def interpret_lbf(lbf):
        if pd.isna(lbf):
            return "N/A"
        elif lbf > config.LOG_BF_THRESHOLD:
            return "Strong evidence for k+1"
        elif lbf > 2:
            return "Moderate evidence for k+1"
        elif lbf > 0:
            return "Weak evidence for k+1"
        else:
            return "No evidence for k+1"
    
    bic_table['interpretation'] = bic_table['log_bf'].apply(interpret_lbf)
    
    return bic_table


# =============================================================================
# Final Model Fitting and Prediction
# =============================================================================

def fit_final_gmm(
    X: np.ndarray,
    k: int,
    n_init: int = None,
    reg_covar: float = None,
    random_state: int = 42,
) -> GaussianMixture:
    """
    Fit the final GMM model with the selected k.
    
    This is essentially the same as fit_gmm but with explicit logging
    for the final model.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        k: Selected number of clusters.
        n_init: Number of initializations.
        reg_covar: Regularization added to diagonal.
        random_state: Random seed.
        
    Returns:
        Fitted GaussianMixture model.
    """
    logger.info(f"Fitting final GMM with k = {k}...")
    
    gmm = fit_gmm(X, k, n_init, reg_covar, random_state)
    
    if not gmm.converged_:
        logger.warning("Final GMM did not converge")
    
    logger.info(f"Final model: {k} clusters, log-likelihood = {gmm.score(X) * len(X):.0f}")
    
    return gmm


def predict_clusters(
    gmm: GaussianMixture,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict cluster assignments and posterior probabilities.
    
    Args:
        gmm: Fitted GaussianMixture model.
        X: Feature matrix.
        
    Returns:
        Tuple of (labels, posteriors):
            - labels: 1D array of cluster assignments (0 to k-1)
            - posteriors: 2D array of shape (n_samples, k) with posterior probs
    """
    labels = gmm.predict(X)
    posteriors = gmm.predict_proba(X)
    
    # Validate
    assert len(labels) == len(X), "Label count mismatch"
    assert posteriors.shape == (len(X), gmm.n_components), "Posterior shape mismatch"
    assert np.allclose(posteriors.sum(axis=1), 1.0), "Posteriors don't sum to 1"
    
    return labels, posteriors


# =============================================================================
# Convenience Functions
# =============================================================================

def run_clustering(
    X: np.ndarray,
    k_max: int,
    random_state: int = 42,
    show_progress: bool = True,
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray, pd.DataFrame, int]:
    """
    Run complete clustering workflow: BIC selection + final model.
    
    Args:
        X: Standardized feature matrix.
        k_max: Maximum number of clusters to evaluate.
        random_state: Random seed.
        show_progress: Whether to show progress bar.
        
    Returns:
        Tuple of (gmm, labels, posteriors, bic_table, optimal_k).
    """
    # BIC model selection
    k_grid = range(1, k_max + 1)
    bic_table = fit_gmm_bic(X, k_grid, random_state=random_state, show_progress=show_progress)
    bic_table = compute_log_bayes_factors(bic_table)
    
    # Select optimal k
    optimal_k = select_optimal_k(bic_table)
    
    # Fit final model
    gmm = fit_final_gmm(X, optimal_k, random_state=random_state)
    
    # Get predictions
    labels, posteriors = predict_clusters(gmm, X)
    
    return gmm, labels, posteriors, bic_table, optimal_k

