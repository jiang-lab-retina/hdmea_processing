"""
GMM clustering with BIC-based k selection.

Implements Baden-style diagonal GMM + minimum BIC selection.
"""

import logging
import json
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from .. import config
from ..models.gpu_gmm import GaussianMixtureGPU

logger = logging.getLogger(__name__)

# Type alias for GMM
GMMType = Union[GaussianMixture, GaussianMixtureGPU]


def fit_gmm_bic(
    embeddings: np.ndarray,
    k_range: range | list[int] | None = None,
    n_init: int | None = None,
    reg_covar: float | None = None,
    max_iter: int | None = None,
    use_gpu: bool | None = None,
    show_progress: bool = True,
) -> Tuple[List[GMMType], np.ndarray]:
    """
    Fit diagonal GMM for each k in range and compute BIC.
    
    Args:
        embeddings: (n_cells, embedding_dim) standardized embeddings.
        k_range: Range of k values to try. Defaults to 1..K_MAX for group.
        n_init: Number of random restarts. Defaults to config.GMM_N_INIT.
        reg_covar: Covariance regularization. Defaults to config.GMM_REG_COVAR.
        max_iter: Max EM iterations. Defaults to config.GMM_MAX_ITER.
        use_gpu: Whether to use GPU GMM. Defaults to config.GMM_USE_GPU.
        show_progress: Whether to show progress bar.
    
    Returns:
        Tuple of (list of fitted GMMs, array of BIC values).
    """
    # Apply defaults
    if k_range is None:
        k_range = range(config.K_MIN, config.K_MAX.get("DSGC", 40) + 1)
    k_range = list(k_range)
    
    n_init = n_init if n_init is not None else config.GMM_N_INIT
    reg_covar = reg_covar if reg_covar is not None else config.GMM_REG_COVAR
    max_iter = max_iter if max_iter is not None else config.GMM_MAX_ITER
    use_gpu = use_gpu if use_gpu is not None else config.GMM_USE_GPU
    
    # Check GPU availability
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but CUDA not available, using CPU GMM")
        use_gpu = False
    
    n_samples = embeddings.shape[0]
    logger.info(f"Fitting GMM for k={k_range[0]}..{k_range[-1]} on {n_samples} samples")
    
    models = []
    bic_values = []
    
    iterator = tqdm(k_range, desc="GMM k-selection") if show_progress else k_range
    
    for k in iterator:
        try:
            if use_gpu:
                gmm = GaussianMixtureGPU(
                    n_components=k,
                    max_iter=max_iter,
                    n_init=n_init,
                    reg_covar=reg_covar,
                    device='cuda',
                )
            else:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='diag',
                    max_iter=max_iter,
                    n_init=n_init,
                    reg_covar=reg_covar,
                    random_state=42,
                )
            
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            
            models.append(gmm)
            bic_values.append(bic)
            
            if show_progress:
                iterator.set_postfix({'k': k, 'BIC': f'{bic:.1f}'})
                
        except Exception as e:
            logger.warning(f"GMM fitting failed for k={k}: {e}")
            models.append(None)
            bic_values.append(np.inf)
    
    return models, np.array(bic_values)


def select_k_min_bic(
    models: List[GMMType],
    bic_values: np.ndarray,
    k_range: list[int],
) -> Tuple[int, GMMType]:
    """
    Select k* as the k with minimum BIC.
    
    Args:
        models: List of fitted GMMs for each k.
        bic_values: Array of BIC values.
        k_range: List of k values corresponding to models.
    
    Returns:
        Tuple of (selected k, corresponding GMM model).
    
    Warns:
        If selected k is at the boundary of k_range.
    """
    # Find minimum BIC
    valid_mask = np.isfinite(bic_values)
    if not valid_mask.any():
        raise ValueError("No valid BIC values computed")
    
    best_idx = np.argmin(bic_values)
    k_selected = k_range[best_idx]
    best_model = models[best_idx]
    
    logger.info(f"Selected k*={k_selected} (BIC={bic_values[best_idx]:.1f})")
    
    # Warn if at boundary
    if best_idx == 0:
        logger.warning(f"k*={k_selected} is at lower boundary, consider decreasing K_MIN")
    elif best_idx == len(k_range) - 1:
        logger.warning(f"k*={k_selected} is at upper boundary, consider increasing K_MAX")
    
    return k_selected, best_model


def save_k_selection(
    k_range: list[int],
    bic_values: np.ndarray,
    k_selected: int,
    output_path: Path,
    group: str = "unknown",
) -> None:
    """
    Save k-selection results to JSON.
    
    Args:
        k_range: List of k values tried.
        bic_values: BIC values for each k.
        k_selected: Selected k*.
        output_path: Path to save JSON file.
        group: Group name for metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "group": group,
        "k_range": k_range,
        "bic_values": bic_values.tolist(),
        "k_selected": k_selected,
        "selection_method": "min_bic",
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved k-selection results to {output_path}")


def load_gmm_results(
    checkpoint_dir: Path,
    group: str,
    embeddings: np.ndarray,
) -> Tuple[int, GMMType, np.ndarray]:
    """
    Load cached GMM results if available.
    
    Args:
        checkpoint_dir: Directory containing k_selection.json.
        group: Group name.
        embeddings: Current embeddings to re-fit GMM.
    
    Returns:
        Tuple of (k_selected, fitted GMM, BIC values).
    
    Raises:
        FileNotFoundError: If cache doesn't exist.
    """
    k_selection_path = checkpoint_dir / f"{group}/k_selection.json"
    
    if not k_selection_path.exists():
        raise FileNotFoundError(f"No cached k-selection at {k_selection_path}")
    
    with open(k_selection_path, 'r') as f:
        data = json.load(f)
    
    k_selected = data['k_selected']
    bic_values = np.array(data['bic_values'])
    
    logger.info(f"Loaded cached k*={k_selected} for group {group}")
    
    # Re-fit GMM with selected k (faster than fitting all k)
    use_gpu = config.GMM_USE_GPU and torch.cuda.is_available()
    
    if use_gpu:
        gmm = GaussianMixtureGPU(
            n_components=k_selected,
            max_iter=config.GMM_MAX_ITER,
            n_init=config.GMM_N_INIT,
            reg_covar=config.GMM_REG_COVAR,
            device='cuda',
        )
    else:
        gmm = GaussianMixture(
            n_components=k_selected,
            covariance_type='diag',
            max_iter=config.GMM_MAX_ITER,
            n_init=config.GMM_N_INIT,
            reg_covar=config.GMM_REG_COVAR,
            random_state=42,
        )
    
    gmm.fit(embeddings)
    
    return k_selected, gmm, bic_values
