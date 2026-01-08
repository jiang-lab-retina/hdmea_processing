"""
Model Architectures for Subgroup Clustering.

Implements:
1. ConvAutoencoder - Standard 1D convolutional autoencoder
2. ConvVAE - Variational autoencoder with KL regularization
3. DECModel - Deep Embedded Clustering model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# BASE ENCODER/DECODER
# =============================================================================

class ConvEncoder(nn.Module):
    """Shared 1D convolutional encoder."""
    
    def __init__(self, input_length: int, latent_dim: int):
        super().__init__()
        self.input_length = input_length
        act = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), act,
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), act,
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), act,
        )
        
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            conv_out = self.conv(dummy)
            self._conv_channels = conv_out.shape[1]
            self._conv_length = conv_out.shape[2]
            self._flattened_size = self._conv_channels * self._conv_length
        
        self.fc = nn.Linear(self._flattened_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z
    
    @property
    def flattened_size(self) -> int:
        return self._flattened_size
    
    @property
    def conv_shape(self) -> Tuple[int, int]:
        return (self._conv_channels, self._conv_length)


class ConvDecoder(nn.Module):
    """Shared 1D convolutional decoder."""
    
    def __init__(
        self, 
        latent_dim: int, 
        flattened_size: int,
        conv_channels: int,
        conv_length: int,
        output_length: int,
    ):
        super().__init__()
        self.output_length = output_length
        self._conv_channels = conv_channels
        self._conv_length = conv_length
        act = nn.LeakyReLU(0.2, inplace=True)
        
        self.fc = nn.Linear(latent_dim, flattened_size)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), act,
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32), act,
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self._conv_channels, self._conv_length)
        x = self.deconv(x)
        x = x[..., :self.output_length]
        return x


# =============================================================================
# STANDARD AUTOENCODER
# =============================================================================

class ConvAutoencoder(nn.Module):
    """
    Standard 1D Convolutional Autoencoder.
    
    Used as baseline and for Approach 1 (AE + GMM).
    """
    
    def __init__(self, input_length: int, latent_dim: int = 64):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        self.encoder = ConvEncoder(input_length, latent_dim)
        self.decoder = ConvDecoder(
            latent_dim,
            self.encoder.flattened_size,
            self.encoder.conv_shape[0],
            self.encoder.conv_shape[1],
            input_length,
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# =============================================================================
# VARIATIONAL AUTOENCODER
# =============================================================================

class ConvVAE(nn.Module):
    """
    Variational Autoencoder with 1D convolutions.
    
    Adds KL-divergence regularization for smoother latent space.
    Used for Approach 2 (VAE + GMM).
    """
    
    def __init__(self, input_length: int, latent_dim: int = 64):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = ConvEncoder(input_length, latent_dim * 2)  # *2 for mu and logvar
        
        # Split into mu and logvar
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)
        
        # Decoder
        self.decoder = ConvDecoder(
            latent_dim,
            self.encoder.flattened_size,
            self.encoder.conv_shape[0],
            self.encoder.conv_shape[1],
            input_length,
        )
        # Need to adjust decoder fc since encoder outputs 2*latent_dim
        self.decoder.fc = nn.Linear(latent_dim, self.encoder.flattened_size)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent code (mu) for clustering."""
        mu, _ = self.encode(x)
        return mu
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from N(0,1)."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


# =============================================================================
# DEEP EMBEDDED CLUSTERING (DEC)
# =============================================================================

class ClusteringLayer(nn.Module):
    """
    Clustering layer for DEC.
    
    Computes soft cluster assignments using Student's t-distribution.
    """
    
    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Cluster centers (initialized with k-means)
        self.clusters = nn.Parameter(torch.zeros(n_clusters, latent_dim))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments q.
        
        q_ij = (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2) / Σ_j'(...)
        """
        # z: [batch, latent_dim]
        # clusters: [n_clusters, latent_dim]
        
        # Squared distances: [batch, n_clusters]
        q = 1.0 / (1.0 + torch.sum(
            (z.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2
        ) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution p from soft assignments q.
        
        p_ij = (q_ij^2 / f_j) / Σ_j'(q_ij'^2 / f_j')
        where f_j = Σ_i q_ij
        """
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p


class DECModel(nn.Module):
    """
    Deep Embedded Clustering Model.
    
    Combines autoencoder with clustering layer.
    Pre-train autoencoder, then fine-tune with KL divergence clustering loss.
    """
    
    def __init__(
        self, 
        input_length: int, 
        latent_dim: int = 64,
        n_clusters: int = 5,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # Autoencoder
        self.autoencoder = ConvAutoencoder(input_length, latent_dim)
        
        # Clustering layer
        self.clustering = ClusteringLayer(n_clusters, latent_dim, alpha)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (reconstruction, latent, soft_assignments).
        """
        z = self.encode(x)
        recon = self.decode(z)
        q = self.clustering(z)
        return recon, z, q
    
    def clustering_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """KL divergence between q and target p."""
        return F.kl_div(q.log(), p, reduction='batchmean')
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments."""
        z = self.encode(x)
        q = self.clustering(z)
        return q.argmax(dim=1)
    
    def initialize_clusters(self, z: torch.Tensor, method: str = "kmeans"):
        """
        Initialize cluster centers from encoded data.
        
        Parameters
        ----------
        z : torch.Tensor
            Encoded latent vectors [N, latent_dim]
        method : str
            Initialization method ('kmeans' or 'random')
        """
        from sklearn.cluster import KMeans
        
        z_np = z.detach().cpu().numpy()
        
        if method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
            kmeans.fit(z_np)
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
        else:
            # Random initialization
            indices = np.random.choice(len(z_np), self.n_clusters, replace=False)
            centers = z[indices].clone()
        
        self.clustering.clusters.data = centers.to(z.device)


def get_device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# FLEXIBLE ARCHITECTURE (for hyperparameter optimization)
# =============================================================================

class FlexibleConvAE(nn.Module):
    """
    Flexible Convolutional Autoencoder with configurable architecture.
    
    Used for hyperparameter optimization where architecture params vary.
    
    Parameters
    ----------
    input_length : int
        Length of input traces
    latent_dim : int
        Dimension of latent space
    n_conv_layers : int
        Number of convolutional layers (2-4)
    base_channels : int
        Base number of channels (doubles each layer)
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        input_length: int,
        latent_dim: int = 64,
        n_conv_layers: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.n_conv_layers = n_conv_layers
        self.base_channels = base_channels
        
        # Build encoder
        encoder_layers = []
        in_ch = 1
        current_length = input_length
        
        for i in range(n_conv_layers):
            out_ch = base_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if dropout > 0 and i < n_conv_layers - 1:
                encoder_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            current_length = (current_length + 1) // 2
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.flat_dim = in_ch * current_length
        self.final_channels = in_ch
        self.final_length = current_length
        
        # Bottleneck
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim * 2, self.flat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        channels = [base_channels * (2 ** i) for i in range(n_conv_layers)]
        channels = channels[::-1]  # Reverse
        
        for i in range(n_conv_layers):
            if i < n_conv_layers - 1:
                decoder_layers.extend([
                    nn.ConvTranspose1d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(channels[i], 1, kernel_size=5, stride=2, padding=2, output_padding=1)
                )
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.final_channels, self.final_length)
        x_recon = self.decoder_conv(h)
        # Adjust length if needed
        if x_recon.size(2) != self.input_length:
            x_recon = F.interpolate(
                x_recon, size=self.input_length, mode='linear', align_corners=False
            )
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# =============================================================================
# SUPERVISED CONTRASTIVE LEARNING
# =============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss from Khosla et al. (2020).
    
    For each anchor, pulls together samples with the same label
    and pushes apart samples with different labels.
    
    L = -sum_i (1/|P(i)|) * sum_{p in P(i)} log(
        exp(z_i · z_p / τ) / sum_{a in A(i)} exp(z_i · z_a / τ)
    )
    
    where:
        P(i) = set of positive samples (same class as anchor i)
        A(i) = set of all samples except anchor i
        τ = temperature parameter
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Parameters
        ----------
        features : torch.Tensor
            Normalized projection features [batch_size, projection_dim]
        labels : torch.Tensor
            Class labels [batch_size]
            
        Returns
        -------
        torch.Tensor
            Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # L2 normalize features for cosine similarity
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix (cosine similarity / temperature)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label, excluding self)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_positive = mask_positive - mask_self  # Exclude diagonal
        
        # Number of positives per sample (excluding self)
        positives_count = mask_positive.sum(dim=1)
        
        # For numerical stability, subtract max value
        max_sim = similarity_matrix.max(dim=1, keepdim=True)[0]
        similarity_matrix = similarity_matrix - max_sim.detach()
        
        # Compute log-softmax over all pairs except self
        exp_sim = torch.exp(similarity_matrix) * (1 - mask_self)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Only for samples that have at least one positive
        has_positives = positives_count > 0
        
        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1)
        mean_log_prob_pos = mean_log_prob_pos[has_positives] / positives_count[has_positives]
        
        # Contrastive loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ContrastiveAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder with Contrastive Projection Head.
    
    Combines reconstruction learning with supervised contrastive learning:
    - Encoder learns to compress the input to a latent space
    - Projection head maps latent codes to a space for contrastive learning
    - Decoder reconstructs the input from latent codes
    
    The combined loss:
        L_total = L_reconstruction + λ * L_contrastive
    
    This encourages the latent space to:
    1. Preserve information (reconstruction)
    2. Group similar samples and separate different samples (contrastive)
    """
    
    def __init__(
        self, 
        input_length: int, 
        latent_dim: int = 64,
        projection_dim: int = 128,
        n_conv_layers: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.n_conv_layers = n_conv_layers
        self.base_channels = base_channels
        
        act = nn.LeakyReLU(0.2, inplace=True)
        
        # Build encoder
        encoder_layers = []
        in_ch = 1
        current_length = input_length
        
        for i in range(n_conv_layers):
            out_ch = base_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_ch),
                act,
            ])
            if dropout > 0 and i < n_conv_layers - 1:
                encoder_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            current_length = (current_length + 1) // 2
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.flat_dim = in_ch * current_length
        self.final_channels = in_ch
        self.final_length = current_length
        
        # Bottleneck (encoder FC)
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # Projection head for contrastive learning
        # Maps latent codes to a space where contrastive loss is computed
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim, projection_dim),
        )
        
        # Decoder FC
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim * 2, self.flat_dim),
            act,
        )
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        channels = [base_channels * (2 ** i) for i in range(n_conv_layers)]
        channels = channels[::-1]  # Reverse
        
        for i in range(n_conv_layers):
            if i < n_conv_layers - 1:
                decoder_layers.extend([
                    nn.ConvTranspose1d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.BatchNorm1d(channels[i+1]),
                    act,
                ])
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(channels[i], 1, kernel_size=5, stride=2, padding=2, output_padding=1)
                )
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        return z
    
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent codes for contrastive learning."""
        return self.projection_head(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.final_channels, self.final_length)
        x_recon = self.decoder_conv(h)
        # Adjust length if needed
        if x_recon.size(2) != self.input_length:
            x_recon = F.interpolate(
                x_recon, size=self.input_length, mode='linear', align_corners=False
            )
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns
        -------
        tuple
            (reconstruction, latent_codes, projection)
        """
        z = self.encode(x)
        proj = self.project(z)
        x_recon = self.decode(z)
        return x_recon, z, proj


def build_model(
    model_type: str,
    input_length: int,
    latent_dim: int = 64,
    n_clusters: int = 5,
    projection_dim: int = 128,
    n_conv_layers: int = 3,
    base_channels: int = 32,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory function to build models.
    
    Parameters
    ----------
    model_type : str
        'ae', 'vae', 'dec', or 'contrastive'
    input_length : int
        Length of input traces
    latent_dim : int
        Latent space dimension
    n_clusters : int
        Number of clusters (only for DEC)
    projection_dim : int
        Projection dimension (only for contrastive)
    n_conv_layers : int
        Number of conv layers (only for contrastive)
    base_channels : int
        Base channel count (only for contrastive)
    dropout : float
        Dropout rate (only for contrastive)
        
    Returns
    -------
    nn.Module
        The constructed model
    """
    device = get_device()
    
    if model_type == "ae":
        model = ConvAutoencoder(input_length, latent_dim)
    elif model_type == "vae":
        model = ConvVAE(input_length, latent_dim)
    elif model_type == "dec":
        model = DECModel(input_length, latent_dim, n_clusters)
    elif model_type == "contrastive":
        model = ContrastiveAutoencoder(
            input_length, latent_dim, projection_dim,
            n_conv_layers, base_channels, dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

