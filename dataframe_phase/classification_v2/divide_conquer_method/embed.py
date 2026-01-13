"""
Embedding extraction from trained autoencoder.

Handles:
- Extracting 49D embeddings from trained model
- Z-score standardization of embeddings
"""

import logging
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from . import config
from .models.autoencoder import MultiSegmentAutoencoder

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: MultiSegmentAutoencoder,
    segments: dict[str, np.ndarray],
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Extract embeddings for all cells.
    
    Args:
        model: Trained autoencoder.
        segments: Preprocessed segment arrays. Dict mapping segment_name
            to (n_cells, segment_length) arrays.
        batch_size: Inference batch size.
        device: "cuda" or "cpu".
    
    Returns:
        (n_cells, 49) embedding array.
    
    Notes:
        - Model set to eval mode
        - No gradients computed
    """
    device = device if device is not None else config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Get number of cells
    n_cells = next(iter(segments.values())).shape[0]
    
    # Convert segments to tensors
    segment_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in segments.items()
    }
    
    # Extract embeddings in batches
    all_embeddings = []
    
    with torch.no_grad():
        for start_idx in range(0, n_cells, batch_size):
            end_idx = min(start_idx + batch_size, n_cells)
            
            # Get batch
            batch_segments = {
                name: tensor[start_idx:end_idx].to(device)
                for name, tensor in segment_tensors.items()
            }
            
            # Forward pass (encode only)
            output = model(batch_segments)
            embeddings = output['full_embedding'].cpu().numpy()
            all_embeddings.append(embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Extracted embeddings: shape {embeddings.shape}")
    
    # Validate
    assert embeddings.shape == (n_cells, model.total_latent_dim), \
        f"Expected shape ({n_cells}, {model.total_latent_dim}), got {embeddings.shape}"
    assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN values"
    assert np.all(np.isfinite(embeddings)), "Embeddings contain non-finite values"
    
    return embeddings


def standardize_embeddings(
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Z-score standardize embeddings globally.
    
    Args:
        embeddings: (n_cells, 49) raw embeddings.
    
    Returns:
        (standardized_embeddings, scaler_params)
        scaler_params contains mean/std per feature.
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(embeddings)
    
    scaler_params = {
        'mean': scaler.mean_,
        'std': scaler.scale_,
    }
    
    logger.info(f"Standardized embeddings: shape {standardized.shape}")
    
    return standardized, scaler_params


def apply_standardization(
    embeddings: np.ndarray,
    scaler_params: dict,
) -> np.ndarray:
    """
    Apply previously computed standardization to new embeddings.
    
    Args:
        embeddings: (n_cells, dim) embeddings to standardize.
        scaler_params: Parameters from previous standardize_embeddings call.
    
    Returns:
        Standardized embeddings.
    """
    mean = scaler_params['mean']
    std = scaler_params['std']
    return (embeddings - mean) / (std + 1e-8)
