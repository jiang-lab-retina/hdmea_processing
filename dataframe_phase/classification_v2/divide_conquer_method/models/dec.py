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
        initial_centers: Initial cluster centers from GMM (in standardized space).
        embedding_mean: Mean for standardization (from training data).
        embedding_std: Std for standardization (from training data).
    """
    
    def __init__(
        self,
        autoencoder: nn.Module,
        n_clusters: int,
        alpha: float = 1.0,
        initial_centers: Optional[np.ndarray] = None,
        embedding_mean: Optional[np.ndarray] = None,
        embedding_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        self.autoencoder = autoencoder
        self.embedding_dim = autoencoder.total_latent_dim
        
        # Store standardization parameters as buffers (not learnable)
        if embedding_mean is not None:
            self.register_buffer('embedding_mean', torch.tensor(embedding_mean, dtype=torch.float32))
            self.register_buffer('embedding_std', torch.tensor(embedding_std, dtype=torch.float32))
            self.use_standardization = True
        else:
            self.register_buffer('embedding_mean', torch.zeros(self.embedding_dim))
            self.register_buffer('embedding_std', torch.ones(self.embedding_dim))
            self.use_standardization = False
        
        # DEC clustering layer
        self.dec_layer = DECLayer(
            n_clusters=n_clusters,
            embedding_dim=self.embedding_dim,
            alpha=alpha,
            initial_centers=initial_centers,
        )
    
    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        """Standardize embeddings to match GMM center space."""
        return (z - self.embedding_mean) / (self.embedding_std + 1e-8)
    
    def forward(self, segments: dict[str, torch.Tensor]) -> dict:
        """
        Forward pass through autoencoder and DEC layer.
        
        Args:
            segments: Dict of segment tensors.
        
        Returns:
            Dict with:
                - 'q': Soft cluster assignments
                - 'embedding': Full embedding (standardized)
                - 'reconstructions': Per-segment reconstructions
        """
        # Get autoencoder output
        ae_output = self.autoencoder(segments)
        
        # Standardize embeddings to match GMM center space
        z_raw = ae_output['full_embedding']
        z_std = self._standardize(z_raw)
        
        # Get soft assignments using standardized embeddings
        q = self.dec_layer(z_std)
        
        return {
            'q': q,
            'embedding': z_std,  # Return standardized embedding
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


def cluster_balance_loss(q: torch.Tensor) -> torch.Tensor:
    """
    Compute cluster balance regularization to prevent collapse.
    
    Maximizes entropy of cluster proportions (encourages uniform cluster sizes).
    
    Args:
        q: (batch, k) soft assignments.
    
    Returns:
        (1 - normalized_entropy): 0 when balanced, 1 when collapsed.
    """
    # Cluster proportions: average soft assignment per cluster
    cluster_props = q.mean(dim=0)  # (k,)
    
    # Entropy of cluster proportions: H = -sum(p * log(p))
    # Higher entropy = more uniform distribution = balanced clusters
    entropy = -torch.sum(cluster_props * torch.log(cluster_props + 1e-10))
    
    # Normalize by log(k) so it's in [0, 1]
    max_entropy = torch.log(torch.tensor(q.size(1), dtype=torch.float32, device=q.device))
    normalized_entropy = entropy / max_entropy
    
    # Return (1 - entropy) so: balanced → 0, collapsed → 1
    # Minimizing this encourages balanced clusters
    return 1.0 - normalized_entropy


def iprgc_enrichment_loss(
    q: torch.Tensor,
    iprgc_mask: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage ipRGC cells to concentrate into precomputed target clusters
    while pushing non-ipRGC cells out of those clusters.

    Two components (weighted equally):
      1. **Concentration**: pull ipRGC cells toward target clusters.
      2. **Purity**: maximize the ipRGC fraction inside target clusters,
         which gives gradients to non-ipRGC cells to leave.

    Args:
        q: (batch, k) soft cluster assignments.
        iprgc_mask: (batch,) boolean mask for ipRGC cells.
        target_indices: (n_target,) precomputed global cluster indices
            (from ``compute_iprgc_target_clusters``).

    Returns:
        Loss in [0, 1]: 0 = perfect concentration and purity.
    """
    if iprgc_mask.sum() == 0:
        return torch.tensor(0.0, device=q.device)

    q_iprgc = q[iprgc_mask]  # (n_iprgc, k)

    # --- Part 1: Concentration ------------------------------------------------
    # Maximise the total soft assignment of ipRGC cells to target clusters.
    q_iprgc_top = q_iprgc[:, target_indices].sum(dim=1)  # (n_iprgc,)
    loss_conc = 1.0 - q_iprgc_top.mean()

    # --- Part 2: Purity -------------------------------------------------------
    # Maximise ipRGC fraction of the soft mass inside each target cluster.
    # This provides gradient for non-ipRGC cells to move *away* from targets.
    q_target = q[:, target_indices]                       # (batch, n_target)
    iprgc_mass = q_target[iprgc_mask].sum(dim=0)          # (n_target,)
    total_mass = q_target.sum(dim=0)                      # (n_target,)
    purity = iprgc_mass / (total_mass + 1e-10)            # (n_target,)
    loss_purity = 1.0 - purity.mean()

    return 0.5 * loss_conc + 0.5 * loss_purity


def compute_iprgc_target_clusters(
    q_all: torch.Tensor,
    iprgc_labels: np.ndarray,
    n_target_clusters: int,
) -> torch.Tensor:
    """
    Identify the top-N clusters by global ipRGC soft-assignment fraction.

    Called alongside ``_compute_target_distribution`` every ``update_interval``
    iterations so that the target clusters are stable within each phase.

    Args:
        q_all: (N, k) soft assignments for ALL samples (not a mini-batch).
        iprgc_labels: (N,) boolean array marking ipRGC cells.
        n_target_clusters: Number of target clusters to select.

    Returns:
        (n_target,) int64 tensor of cluster indices on CPU.
    """
    iprgc_mask = torch.tensor(iprgc_labels, dtype=torch.bool)
    q_iprgc = q_all[iprgc_mask]                           # (n_iprgc, k)
    iprgc_mass = q_iprgc.sum(dim=0)                       # (k,)
    total_mass = q_all.sum(dim=0)                          # (k,)
    iprgc_frac = iprgc_mass / (total_mass + 1e-10)        # (k,)

    n_top = min(n_target_clusters, q_all.size(1))
    _, top_indices = torch.topk(iprgc_frac, n_top)

    return top_indices.cpu()


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
