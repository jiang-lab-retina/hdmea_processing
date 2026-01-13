"""
Baden-style GMM clustering with BIC selection.

Implements:
- Diagonal-covariance GMM fitting (CPU or GPU)
- BIC and log Bayes factor for model selection
- Per-group clustering respecting coarse group boundaries
"""

import logging
from typing import Tuple, Union

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config
from .models.gpu_gmm import GaussianMixtureGPU

logger = logging.getLogger(__name__)

# Type alias for GMM models
GMMType = Union[GaussianMixture, GaussianMixtureGPU]


def fit_gmm_bic(
    embeddings: np.ndarray,
    k_range: range | list[int] | None = None,
    k_max: int | None = None,
    n_init: int | None = None,
    reg_covar: float | None = None,
    random_state: int = 42,
    use_gpu: bool | None = None,
) -> Tuple[list[GMMType], np.ndarray]:
    """
    Fit GMMs for a range of k values and compute BIC.
    
    Args:
        embeddings: (n_cells, dim) embeddings to cluster.
        k_range: Range of k values to try. If None, uses range(1, k_max+1).
        k_max: Maximum k if k_range not provided.
        n_init: Number of GMM initializations. Defaults to config.GMM_N_INIT.
        reg_covar: Covariance regularization. Defaults to config.GMM_REG_COVAR.
        random_state: Random seed.
        use_gpu: Whether to use GPU-accelerated GMM. Defaults to config.GMM_USE_GPU.
    
    Returns:
        (list of fitted GMMs, BIC values array)
        Both indexed by k-1 (i.e., models[0] is k=1).
    """
    n_init = n_init if n_init is not None else config.GMM_N_INIT
    reg_covar = reg_covar if reg_covar is not None else config.GMM_REG_COVAR
    k_max = k_max if k_max is not None else config.K_MAX_DEFAULT
    use_gpu = use_gpu if use_gpu is not None else getattr(config, 'GMM_USE_GPU', False)
    
    # Check if GPU is actually available
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but CUDA not available, falling back to CPU GMM")
        use_gpu = False
    
    if k_range is None:
        k_range = range(1, k_max + 1)
    
    n_samples = len(embeddings)
    
    # Don't try k values larger than n_samples
    k_range = [k for k in k_range if k <= n_samples]
    
    models = []
    bic_values = []
    
    backend = "GPU" if use_gpu else "CPU"
    logger.info(f"Fitting GMMs ({backend}) for k in {list(k_range)[:5]}...{list(k_range)[-3:]}")
    
    for k in k_range:
        if use_gpu:
            gmm = GaussianMixtureGPU(
                n_components=k,
                n_init=n_init,
                reg_covar=reg_covar,
                random_state=random_state,
                max_iter=200,
                device='cuda',
            )
        else:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='diag',
                n_init=n_init,
                reg_covar=reg_covar,
                random_state=random_state,
                max_iter=200,
            )
        
        try:
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            
            models.append(gmm)
            bic_values.append(bic)
            
            logger.debug(f"  k={k}: BIC={bic:.2f}")
        except Exception as e:
            logger.warning(f"  k={k}: Failed to fit GMM: {e}")
            models.append(None)
            bic_values.append(np.inf)
    
    return models, np.array(bic_values)


def select_k_by_logbf(
    bic_values: np.ndarray,
    k_values: list[int],
    threshold: float | None = None,
) -> Tuple[int, int]:
    """
    Select optimal k using log Bayes factor threshold.
    
    The log Bayes factor between models k and k-1 is approximated as:
        logBF = (BIC_{k-1} - BIC_k) / 2
    
    Select smallest k where logBF < threshold (evidence for more clusters weakens).
    
    Args:
        bic_values: BIC values for each k.
        k_values: Corresponding k values.
        threshold: Log Bayes factor threshold. Defaults to config.LOG_BF_THRESHOLD.
    
    Returns:
        (selected_k, index in k_values)
    """
    threshold = threshold if threshold is not None else config.LOG_BF_THRESHOLD
    
    if len(bic_values) < 2:
        return k_values[0], 0
    
    # Find first k where additional clusters don't provide strong evidence
    for i in range(1, len(bic_values)):
        log_bf = (bic_values[i-1] - bic_values[i]) / 2
        
        if log_bf < threshold:
            # k-1 (previous) was the last with strong evidence
            logger.info(f"Selected k={k_values[i-1]} (logBF={log_bf:.2f} < {threshold})")
            return k_values[i-1], i-1
    
    # If all k values show strong evidence, select the last one with valid BIC
    best_idx = np.argmin(bic_values)
    logger.info(f"Selected k={k_values[best_idx]} (min BIC)")
    return k_values[best_idx], best_idx


def select_k_by_min_bic(
    bic_values: np.ndarray,
    k_values: list[int],
) -> Tuple[int, int]:
    """
    Select optimal k by minimum BIC.
    
    Args:
        bic_values: BIC values for each k.
        k_values: Corresponding k values.
    
    Returns:
        (selected_k, index in k_values)
    """
    best_idx = np.argmin(bic_values)
    logger.info(f"Selected k={k_values[best_idx]} (min BIC={bic_values[best_idx]:.2f})")
    return k_values[best_idx], best_idx


def cluster_per_group(
    embeddings: np.ndarray,
    groups: np.ndarray,
    k_max_per_group: dict[str, int] | None = None,
    selection_method: str = "logbf",
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Cluster embeddings separately for each group.
    
    This enforces the constraint that no cluster crosses group boundaries.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        groups: (n_cells,) coarse group labels.
        k_max_per_group: Maximum k per group. Defaults to config.K_MAX.
        selection_method: "logbf" or "min_bic".
        standardize: Whether to z-score standardize within each group.
    
    Returns:
        (cluster_ids, posterior_probs, group_results)
        cluster_ids: (n_cells,) cluster assignment within group
        posterior_probs: (n_cells,) max posterior probability
        group_results: Dict with per-group info (k, bic_values, models)
    """
    k_max_per_group = k_max_per_group if k_max_per_group is not None else config.K_MAX
    
    n_cells = len(embeddings)
    cluster_ids = np.zeros(n_cells, dtype=int)
    posterior_probs = np.zeros(n_cells)
    group_results = {}
    
    unique_groups = np.unique(groups)
    logger.info(f"Clustering {len(unique_groups)} groups...")
    
    for group in unique_groups:
        mask = groups == group
        group_embeddings = embeddings[mask]
        n_group = mask.sum()
        
        if n_group < config.MIN_CELLS_PER_GROUP:
            logger.warning(f"Group {group} has only {n_group} cells, assigning to single cluster")
            cluster_ids[mask] = 0
            posterior_probs[mask] = 1.0
            group_results[group] = {
                'k_selected': 1,
                'bic_values': [0.0],
                'k_range': [1],
            }
            continue
        
        # Standardize within group
        if standardize:
            scaler = StandardScaler()
            group_embeddings = scaler.fit_transform(group_embeddings)
        
        # Get k_max for this group
        k_max = k_max_per_group.get(group, config.K_MAX_DEFAULT)
        k_max = min(k_max, n_group - 1)  # Can't have more clusters than samples
        k_max = max(k_max, 1)
        
        k_range = list(range(1, k_max + 1))
        
        logger.info(f"  Group {group}: {n_group} cells, k_range=[1..{k_max}]")
        
        # Fit GMMs
        models, bic_values = fit_gmm_bic(
            group_embeddings,
            k_range=k_range,
        )
        
        # Select k
        if selection_method == "logbf":
            k_selected, best_idx = select_k_by_logbf(bic_values, k_range)
        else:
            k_selected, best_idx = select_k_by_min_bic(bic_values, k_range)
        
        # Get predictions from best model
        best_model = models[best_idx]
        if best_model is not None:
            labels = best_model.predict(group_embeddings)
            probs = best_model.predict_proba(group_embeddings).max(axis=1)
        else:
            labels = np.zeros(n_group, dtype=int)
            probs = np.ones(n_group)
        
        cluster_ids[mask] = labels
        posterior_probs[mask] = probs
        
        group_results[group] = {
            'k_selected': k_selected,
            'bic_values': bic_values.tolist(),
            'k_range': k_range,
            'model': best_model,
        }
        
        logger.info(f"    Selected k={k_selected}")
    
    logger.info(f"Clustering complete: {len(np.unique(cluster_ids))} total clusters")
    
    return cluster_ids, posterior_probs, group_results


def get_cluster_means(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute mean embedding for each cluster.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        cluster_labels: (n_cells,) cluster assignments.
    
    Returns:
        Dict mapping cluster_id to mean embedding.
    """
    means = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        means[cluster_id] = embeddings[mask].mean(axis=0)
    return means


def get_cluster_stds(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute std of embeddings for each cluster.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        cluster_labels: (n_cells,) cluster assignments.
    
    Returns:
        Dict mapping cluster_id to std embedding.
    """
    stds = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        stds[cluster_id] = embeddings[mask].std(axis=0)
    return stds
