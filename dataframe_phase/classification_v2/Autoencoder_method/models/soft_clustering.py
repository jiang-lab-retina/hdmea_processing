"""
Differentiable soft clustering module for cluster-aware training.

Provides soft k-means clustering that can be used during training
to compute differentiable cluster assignments for purity loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftKMeans(nn.Module):
    """
    Differentiable soft k-means for computing cluster assignments.
    
    Uses learnable cluster centers and computes soft assignments
    based on distance to centers with temperature-scaled softmax.
    
    Args:
        n_clusters: Number of cluster centers.
        embedding_dim: Dimension of embeddings (49 for this pipeline).
        temperature: Softmax temperature. Lower = sharper assignments.
        init_std: Standard deviation for center initialization.
    
    Forward:
        embeddings: (batch, embedding_dim) embeddings
        → (batch, n_clusters) soft assignment probabilities
    """
    
    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        temperature: float = 1.0,
        init_std: float = 0.1,
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Learnable cluster centers
        self.centers = nn.Parameter(
            torch.randn(n_clusters, embedding_dim) * init_std
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments.
        
        Args:
            embeddings: (batch, embedding_dim) embeddings.
        
        Returns:
            (batch, n_clusters) soft assignment probabilities.
            Each row sums to 1.
        """
        # Compute squared distances: (batch, n_clusters)
        # ||z - mu||^2 = ||z||^2 + ||mu||^2 - 2 * z @ mu.T
        z_norm = (embeddings ** 2).sum(dim=1, keepdim=True)  # (batch, 1)
        mu_norm = (self.centers ** 2).sum(dim=1, keepdim=True).T  # (1, n_clusters)
        cross = embeddings @ self.centers.T  # (batch, n_clusters)
        
        distances_sq = z_norm + mu_norm - 2 * cross  # (batch, n_clusters)
        
        # Soft assignments via softmax over negative distances
        # Lower distance = higher probability
        logits = -distances_sq / self.temperature
        soft_assignments = F.softmax(logits, dim=1)
        
        return soft_assignments
    
    def get_hard_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get hard cluster assignments (argmax).
        
        Args:
            embeddings: (batch, embedding_dim) embeddings.
        
        Returns:
            (batch,) cluster indices.
        """
        soft = self.forward(embeddings)
        return soft.argmax(dim=1)
    
    def initialize_from_data(self, embeddings: torch.Tensor) -> None:
        """
        Initialize cluster centers using k-means++ style selection.
        
        Args:
            embeddings: (n_samples, embedding_dim) data for initialization.
        """
        device = self.centers.device
        embeddings = embeddings.to(device)
        n_samples = embeddings.shape[0]
        
        # First center: random sample
        indices = [torch.randint(n_samples, (1,)).item()]
        
        # Remaining centers: weighted by distance to nearest center
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest existing center
            centers_so_far = embeddings[indices]  # (k, dim)
            dists = torch.cdist(embeddings, centers_so_far)  # (n, k)
            min_dists = dists.min(dim=1).values  # (n,)
            
            # Sample proportional to squared distance
            probs = min_dists ** 2
            probs = probs / probs.sum()
            
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)
        
        # Set centers
        with torch.no_grad():
            self.centers.copy_(embeddings[indices])


class StudentTClustering(nn.Module):
    """
    Soft clustering using Student's t-distribution (as in DEC).
    
    Uses t-distribution with 1 degree of freedom for softer assignments
    that better handle outliers.
    
    q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_k(...)
    
    Args:
        n_clusters: Number of cluster centers.
        embedding_dim: Dimension of embeddings.
        alpha: Degrees of freedom parameter (default 1.0).
    """
    
    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # Learnable cluster centers
        self.centers = nn.Parameter(
            torch.randn(n_clusters, embedding_dim) * 0.1
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments using t-distribution.
        
        Args:
            embeddings: (batch, embedding_dim) embeddings.
        
        Returns:
            (batch, n_clusters) soft assignment probabilities.
        """
        # Compute squared distances
        z_norm = (embeddings ** 2).sum(dim=1, keepdim=True)
        mu_norm = (self.centers ** 2).sum(dim=1, keepdim=True).T
        cross = embeddings @ self.centers.T
        distances_sq = z_norm + mu_norm - 2 * cross
        
        # Student's t-distribution kernel
        # q_ij proportional to (1 + d^2/alpha)^(-(alpha+1)/2)
        exponent = -(self.alpha + 1) / 2
        numerator = (1 + distances_sq / self.alpha) ** exponent
        
        # Normalize
        soft_assignments = numerator / numerator.sum(dim=1, keepdim=True)
        
        return soft_assignments
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution for DEC-style training.
        
        The target distribution sharpens the soft assignments:
        p_ij = (q_ij^2 / f_j) / sum_k(q_ik^2 / f_k)
        where f_j = sum_i q_ij
        
        Args:
            q: (batch, n_clusters) soft assignments.
        
        Returns:
            (batch, n_clusters) target distribution.
        """
        # Cluster frequencies
        f = q.sum(dim=0, keepdim=True)  # (1, n_clusters)
        
        # Squared assignments normalized by frequency
        numerator = (q ** 2) / f
        
        # Normalize
        p = numerator / numerator.sum(dim=1, keepdim=True)
        
        return p
