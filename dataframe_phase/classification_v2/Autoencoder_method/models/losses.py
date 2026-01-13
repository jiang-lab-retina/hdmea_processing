"""
Loss functions for the autoencoder training.

Includes:
- WeightedReconstructionLoss: Inverse-length weighted MSE across segments
- SupervisedContrastiveLoss: SupCon loss for weak supervision
- ClusterPurityLoss: Conditional entropy loss for cluster purity
- CombinedAELoss: Combined reconstruction + supervision loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_clustering import SoftKMeans


class WeightedReconstructionLoss(nn.Module):
    """
    Weighted MSE loss across segments.
    
    Weights are inversely proportional to segment length to prevent
    long segments from dominating the loss.
    
    Args:
        segment_lengths: Dict mapping segment_name to input length.
        loss_type: "mse" or "huber".
    
    Forward:
        originals: Dict[str, Tensor]
        reconstructions: Dict[str, Tensor]
        → scalar loss (weighted by inverse segment length)
    """
    
    def __init__(
        self,
        segment_lengths: dict[str, int],
        loss_type: str = "mse",
    ):
        super().__init__()
        
        self.segment_lengths = segment_lengths
        self.loss_type = loss_type
        
        # Compute inverse-length weights (normalized)
        total_inv_len = sum(1.0 / length for length in segment_lengths.values())
        self.weights = {
            name: (1.0 / length) / total_inv_len
            for name, length in segment_lengths.items()
        }
        
        # Loss function
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        originals: dict[str, torch.Tensor],
        reconstructions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weighted reconstruction loss.
        
        Args:
            originals: Dict mapping segment_name to (batch, length) original traces.
            reconstructions: Dict mapping segment_name to (batch, length) reconstructions.
        
        Returns:
            Scalar weighted loss.
        """
        total_loss = 0.0
        
        for name in originals:
            if name in reconstructions:
                weight = self.weights.get(name, 1.0 / len(originals))
                segment_loss = self.loss_fn(reconstructions[name], originals[name])
                total_loss = total_loss + weight * segment_loss
        
        return total_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss using group labels.
    
    Pulls together embeddings with the same label and pushes apart
    embeddings with different labels.
    
    Args:
        temperature: Softmax temperature (default 0.1).
    
    Forward:
        embeddings: Tensor(batch, dim) - will be L2 normalized
        labels: Tensor(batch,) - integer group labels
        → scalar loss
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: (batch, dim) embeddings (will be normalized).
            labels: (batch,) integer labels.
        
        Returns:
            Scalar loss.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        mask_self = torch.eye(batch_size, device=device)
        mask_positive = mask_positive - mask_self
        
        # Number of positive pairs per sample
        num_positives = mask_positive.sum(dim=1)
        
        # Handle samples with no positives (skip them)
        has_positives = num_positives > 0
        
        if not has_positives.any():
            # No positive pairs in batch, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log-softmax
        # Mask out self-similarity with large negative value
        similarity = similarity - mask_self * 1e9
        
        # Log-sum-exp over all samples (denominator)
        log_sum_exp = torch.logsumexp(similarity, dim=1, keepdim=True)
        
        # Log-prob for each pair
        log_prob = similarity - log_sum_exp
        
        # Mean log-prob over positive pairs
        # Mask to only consider positive pairs
        mean_log_prob_positive = (mask_positive * log_prob).sum(dim=1) / (num_positives + 1e-8)
        
        # Only consider samples with positives
        loss = -mean_log_prob_positive[has_positives].mean()
        
        return loss


class CombinedAELoss(nn.Module):
    """
    Combined reconstruction + weak supervision loss.
    
    L_total = L_reconstruction + β × L_supervision
    
    Args:
        segment_lengths: Dict for reconstruction weighting.
        beta: Weight for supervision loss.
        temperature: SupCon temperature.
        loss_type: "mse" or "huber" for reconstruction.
    
    Forward:
        originals: Dict[str, Tensor]
        reconstructions: Dict[str, Tensor]
        embeddings: Tensor(batch, dim)
        labels: Tensor(batch,)
        → {
            'total': scalar,
            'reconstruction': scalar,
            'supervision': scalar,
        }
    """
    
    def __init__(
        self,
        segment_lengths: dict[str, int],
        beta: float = 0.1,
        temperature: float = 0.1,
        loss_type: str = "mse",
    ):
        super().__init__()
        
        self.beta = beta
        
        self.reconstruction_loss = WeightedReconstructionLoss(
            segment_lengths, loss_type
        )
        self.supervision_loss = SupervisedContrastiveLoss(temperature)
    
    def forward(
        self,
        originals: dict[str, torch.Tensor],
        reconstructions: dict[str, torch.Tensor],
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        excluded_labels: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            originals: Dict of original segment traces.
            reconstructions: Dict of reconstructed segment traces.
            embeddings: (batch, dim) full embeddings.
            labels: (batch,) integer group labels.
            excluded_labels: Labels to exclude from supervision (for CV turns).
        
        Returns:
            Dict with 'total', 'reconstruction', 'supervision' losses.
        """
        # Reconstruction loss
        rec_loss = self.reconstruction_loss(originals, reconstructions)
        
        # Supervision loss
        if excluded_labels is not None and len(excluded_labels) > 0:
            # Create mask for non-excluded labels
            mask = torch.ones(len(labels), dtype=torch.bool, device=labels.device)
            for excl in excluded_labels:
                mask = mask & (labels != excl)
            
            if mask.sum() > 1:  # Need at least 2 samples for contrastive loss
                sup_loss = self.supervision_loss(embeddings[mask], labels[mask])
            else:
                sup_loss = torch.tensor(0.0, device=embeddings.device)
        else:
            sup_loss = self.supervision_loss(embeddings, labels)
        
        # Combined loss
        total_loss = rec_loss + self.beta * sup_loss
        
        return {
            'total': total_loss,
            'reconstruction': rec_loss,
            'supervision': sup_loss,
        }


class ClusterPurityLoss(nn.Module):
    """
    Minimize conditional entropy H(Y|C) to encourage pure clusters.
    
    Lower conditional entropy means each cluster is dominated by
    one label value, which corresponds to high purity.
    
    H(Y|C) = -sum_c p(c) sum_y p(y|c) log p(y|c)
    
    Args:
        n_clusters: Number of soft clusters.
        embedding_dim: Dimension of embeddings.
        temperature: Softness of cluster assignments.
        eps: Small value for numerical stability.
    
    Forward:
        embeddings: (batch, embedding_dim) cell embeddings
        labels: (batch, n_labels) binary labels for each sample
        → scalar purity loss (lower = purer clusters)
    """
    
    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.eps = eps
        
        # Soft clustering module
        self.soft_kmeans = SoftKMeans(
            n_clusters=n_clusters,
            embedding_dim=embedding_dim,
            temperature=temperature,
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cluster purity loss (conditional entropy).
        
        Args:
            embeddings: (batch, embedding_dim) embeddings.
            labels: (batch, n_labels) binary labels (0 or 1).
                    Each column is a separate label to optimize.
        
        Returns:
            Scalar loss. Lower = purer clusters.
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Get soft cluster assignments: (batch, n_clusters)
        q = self.soft_kmeans(embeddings)
        
        # Ensure labels is 2D: (batch, n_labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        n_labels = labels.shape[1]
        
        # Compute p(c) - cluster proportions: (n_clusters,)
        p_c = q.sum(dim=0) / batch_size  # (n_clusters,)
        
        total_entropy = 0.0
        
        # For each label dimension
        for label_idx in range(n_labels):
            y = labels[:, label_idx].float()  # (batch,)
            
            # Compute p(y=1|c) for each cluster
            # Weighted sum of positive labels per cluster / cluster size
            # p(y=1|c) = sum_i(q_ic * y_i) / sum_i(q_ic)
            
            cluster_sums = q.sum(dim=0) + self.eps  # (n_clusters,)
            positive_sums = (q * y.unsqueeze(1)).sum(dim=0)  # (n_clusters,)
            
            p_y1_given_c = positive_sums / cluster_sums  # (n_clusters,)
            p_y0_given_c = 1 - p_y1_given_c  # (n_clusters,)
            
            # Conditional entropy for this label: H(Y|C)
            # H(Y|C) = -sum_c p(c) [p(y=0|c) log p(y=0|c) + p(y=1|c) log p(y=1|c)]
            
            # Binary entropy for each cluster
            h_y_given_c = -(
                p_y0_given_c * torch.log(p_y0_given_c + self.eps) +
                p_y1_given_c * torch.log(p_y1_given_c + self.eps)
            )  # (n_clusters,)
            
            # Weighted by cluster proportion
            label_entropy = (p_c * h_y_given_c).sum()
            
            total_entropy = total_entropy + label_entropy
        
        # Average across labels
        avg_entropy = total_entropy / n_labels
        
        return avg_entropy
    
    def initialize_centers(self, embeddings: torch.Tensor) -> None:
        """
        Initialize cluster centers from data using k-means++.
        
        Args:
            embeddings: (n_samples, embedding_dim) data for initialization.
        """
        self.soft_kmeans.initialize_from_data(embeddings)


class CombinedAELossWithPurity(nn.Module):
    """
    Combined reconstruction + contrastive + purity loss.
    
    L_total = L_rec + β × L_supcon + α × L_purity
    
    Args:
        segment_lengths: Dict for reconstruction weighting.
        beta: Weight for contrastive supervision loss.
        alpha: Weight for purity loss.
        temperature_supcon: SupCon temperature.
        n_clusters: Number of clusters for purity loss.
        embedding_dim: Embedding dimension.
        temperature_purity: Purity clustering temperature.
        loss_type: "mse" or "huber" for reconstruction.
    
    Forward:
        originals: Dict[str, Tensor]
        reconstructions: Dict[str, Tensor]
        embeddings: Tensor(batch, dim)
        group_labels: Tensor(batch,) - coarse group labels for SupCon
        purity_labels: Tensor(batch, n_labels) - binary labels for purity
        → {
            'total': scalar,
            'reconstruction': scalar,
            'supervision': scalar,
            'purity': scalar,
        }
    """
    
    def __init__(
        self,
        segment_lengths: dict[str, int],
        beta: float = 0.1,
        alpha: float = 1.0,
        temperature_supcon: float = 0.1,
        n_clusters: int = 100,
        embedding_dim: int = 49,
        temperature_purity: float = 1.0,
        loss_type: str = "mse",
    ):
        super().__init__()
        
        self.beta = beta
        self.alpha = alpha
        
        self.reconstruction_loss = WeightedReconstructionLoss(
            segment_lengths, loss_type
        )
        self.supervision_loss = SupervisedContrastiveLoss(temperature_supcon)
        self.purity_loss = ClusterPurityLoss(
            n_clusters=n_clusters,
            embedding_dim=embedding_dim,
            temperature=temperature_purity,
        )
    
    def forward(
        self,
        originals: dict[str, torch.Tensor],
        reconstructions: dict[str, torch.Tensor],
        embeddings: torch.Tensor,
        group_labels: torch.Tensor,
        purity_labels: torch.Tensor,
        excluded_labels: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss with purity.
        
        Args:
            originals: Dict of original segment traces.
            reconstructions: Dict of reconstructed segment traces.
            embeddings: (batch, dim) full embeddings.
            group_labels: (batch,) integer coarse group labels.
            purity_labels: (batch, n_labels) binary labels for purity.
            excluded_labels: Labels to exclude from SupCon (for CV turns).
        
        Returns:
            Dict with 'total', 'reconstruction', 'supervision', 'purity' losses.
        """
        # Reconstruction loss
        rec_loss = self.reconstruction_loss(originals, reconstructions)
        
        # Supervision loss (SupCon)
        if self.beta > 0:
            if excluded_labels is not None and len(excluded_labels) > 0:
                mask = torch.ones(len(group_labels), dtype=torch.bool, device=group_labels.device)
                for excl in excluded_labels:
                    mask = mask & (group_labels != excl)
                
                if mask.sum() > 1:
                    sup_loss = self.supervision_loss(embeddings[mask], group_labels[mask])
                else:
                    sup_loss = torch.tensor(0.0, device=embeddings.device)
            else:
                sup_loss = self.supervision_loss(embeddings, group_labels)
        else:
            sup_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Purity loss
        if self.alpha > 0:
            purity = self.purity_loss(embeddings, purity_labels)
        else:
            purity = torch.tensor(0.0, device=embeddings.device)
        
        # Combined loss
        total_loss = rec_loss + self.beta * sup_loss + self.alpha * purity
        
        return {
            'total': total_loss,
            'reconstruction': rec_loss,
            'supervision': sup_loss,
            'purity': purity,
        }
    
    def initialize_purity_centers(self, embeddings: torch.Tensor) -> None:
        """
        Initialize purity loss cluster centers from data.
        
        Args:
            embeddings: (n_samples, embedding_dim) data for initialization.
        """
        self.purity_loss.initialize_centers(embeddings)
