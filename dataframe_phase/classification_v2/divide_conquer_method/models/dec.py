"""
Deep Embedded Clustering (DEC) implementation.

Implements the DEC/IDEC approach for cluster refinement:
- Student-t kernel for soft cluster assignments
- Target distribution sharpening
- KL divergence loss with optional reconstruction term
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DECLayer(nn.Module):
    """
    DEC clustering layer using Student-t kernel for soft assignments.
    
    Computes soft cluster assignments:
    q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_j'(...)
    
    Args:
        n_clusters: Number of clusters.
        embedding_dim: Dimension of input embeddings.
        alpha: Student-t degrees of freedom (default 1.0).
        initial_centers: Optional initial cluster centers.
    """
    
    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
        initial_centers: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # Initialize cluster centers
        if initial_centers is not None:
            assert initial_centers.shape == (n_clusters, embedding_dim), \
                f"Expected centers shape ({n_clusters}, {embedding_dim}), got {initial_centers.shape}"
            centers = torch.tensor(initial_centers, dtype=torch.float32)
        else:
            centers = torch.randn(n_clusters, embedding_dim)
        
        # Cluster centers as learnable parameters
        self.centers = nn.Parameter(centers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments Q.
        
        Args:
            z: (batch, embedding_dim) embeddings.
        
        Returns:
            (batch, n_clusters) soft assignment matrix Q.
        """
        # Compute squared distances: ||z_i - mu_j||^2
        # z: (batch, dim), centers: (k, dim)
        dist_sq = torch.sum((z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        
        # Student-t kernel
        # q_ij = (1 + d^2 / alpha)^(-(alpha+1)/2)
        q = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        
        # Normalize
        q = q / q.sum(dim=1, keepdim=True)
        
        return q
    
    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution P that sharpens assignments.
        
        p_ij = (q_ij^2 / f_j) / sum_j'(q_ij'^2 / f_j')
        where f_j = sum_i(q_ij) is the soft cluster frequency.
        
        Args:
            q: (batch, n_clusters) soft assignments.
        
        Returns:
            (batch, n_clusters) target distribution P.
        """
        # Soft cluster frequencies
        f = q.sum(dim=0, keepdim=True)  # (1, k)
        
        # Squared assignments normalized by frequency
        p = q ** 2 / f
        
        # Normalize
        p = p / p.sum(dim=1, keepdim=True)
        
        return p


class IDEC(nn.Module):
    """
    Improved Deep Embedded Clustering (IDEC).
    
    Combines DEC with reconstruction loss to preserve local structure.
    
    Args:
        autoencoder: Pre-trained autoencoder model.
        n_clusters: Number of clusters.
        alpha: Student-t degrees of freedom.
        initial_centers: Initial cluster centers from GMM.
    """
    
    def __init__(
        self,
        autoencoder: nn.Module,
        n_clusters: int,
        alpha: float = 1.0,
        initial_centers: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        self.autoencoder = autoencoder
        self.embedding_dim = autoencoder.total_latent_dim
        
        # DEC clustering layer
        self.dec_layer = DECLayer(
            n_clusters=n_clusters,
            embedding_dim=self.embedding_dim,
            alpha=alpha,
            initial_centers=initial_centers,
        )
    
    def forward(self, segments: dict[str, torch.Tensor]) -> dict:
        """
        Forward pass through autoencoder and DEC layer.
        
        Args:
            segments: Dict of segment tensors.
        
        Returns:
            Dict with:
                - 'q': Soft cluster assignments
                - 'embedding': Full embedding
                - 'reconstructions': Per-segment reconstructions
        """
        # Get autoencoder output
        ae_output = self.autoencoder(segments)
        
        # Get soft assignments
        q = self.dec_layer(ae_output['full_embedding'])
        
        return {
            'q': q,
            'embedding': ae_output['full_embedding'],
            'reconstructions': ae_output['reconstructions'],
        }
    
    def get_cluster_labels(self, segments: dict[str, torch.Tensor]) -> np.ndarray:
        """
        Get hard cluster assignments.
        
        Args:
            segments: Dict of segment tensors.
        
        Returns:
            (batch,) cluster labels.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(segments)
            labels = output['q'].argmax(dim=1).cpu().numpy()
        return labels


def dec_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute DEC loss: KL(P || Q).
    
    Args:
        q: (batch, k) soft assignments.
        p: (batch, k) target distribution.
    
    Returns:
        Scalar KL divergence loss.
    """
    # KL(P || Q) = sum(p * log(p/q))
    # Use log for numerical stability
    loss = F.kl_div(q.log(), p, reduction='batchmean')
    return loss


def reconstruction_loss(
    inputs: dict[str, torch.Tensor],
    reconstructions: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute reconstruction loss (MSE).
    
    Args:
        inputs: Dict of input segment tensors.
        reconstructions: Dict of reconstructed segment tensors.
    
    Returns:
        Total reconstruction loss.
    """
    total_loss = 0.0
    
    for name in inputs:
        if name in reconstructions:
            x = inputs[name]
            x_hat = reconstructions[name]
            segment_loss = F.mse_loss(x_hat, x)
            total_loss = total_loss + segment_loss
    
    return total_loss
