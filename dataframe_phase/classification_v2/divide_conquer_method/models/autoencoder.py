"""
Multi-segment autoencoder combining all segment encoders and decoders.

This is the main model that produces 49D embeddings from 10 stimulus segments.
Simplified version for DEC pipeline (reconstruction-only, no SupCon/purity).

Supports multiple encoder architectures:
- TCN (default): Temporal Convolutional Networks with dilated convolutions
- CNN: Standard 1D convolutional networks
- MultiScale: Parallel branches at different temporal scales
"""

import torch
import torch.nn as nn

from .encoders import create_encoder
from .decoders import create_decoder


class MultiSegmentAutoencoder(nn.Module):
    """
    Multi-segment autoencoder with per-segment encoders/decoders.
    
    Supports optional semi-supervised training via a classification head
    that predicts binary labels (e.g. ipRGC status) from the full embedding.
    
    Args:
        segment_configs: Dict mapping segment_name to {
            'input_length': int,
            'latent_dim': int,
        }
        encoder_type: "tcn" (default), "cnn", or "multiscale".
        hidden_dims: List of hidden dimensions (for CNN encoder).
        dropout: Dropout probability.
        use_mlp_threshold: Use MLP for segments shorter than this.
        tcn_channels: Channel dimensions for TCN blocks.
        tcn_kernel_size: Kernel size for TCN.
        multiscale_kernel_sizes: Kernel sizes for multi-scale branches.
        multiscale_channels: Channels per branch for multi-scale.
        n_classes: Number of classification outputs (0 = no classifier).
        classifier_hidden: Hidden layer size for classifier head.
    
    Forward:
        segments: Dict[str, Tensor(batch, length)] -> {
            'embeddings': Dict[str, Tensor(batch, latent_dim)],
            'full_embedding': Tensor(batch, total_latent_dim),
            'reconstructions': Dict[str, Tensor(batch, length)],
            'class_logits': Tensor(batch, n_classes),  # only if n_classes > 0
        }
    
    Properties:
        total_latent_dim: Sum of all segment latent dims (49)
    """
    
    def __init__(
        self,
        segment_configs: dict[str, dict],
        encoder_type: str = "tcn",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_mlp_threshold: int = 30,
        tcn_channels: list[int] | None = None,
        tcn_kernel_size: int = 3,
        multiscale_kernel_sizes: list[int] | None = None,
        multiscale_channels: int = 32,
        n_classes: int = 0,
        classifier_hidden: int = 32,
    ):
        super().__init__()
        
        self.segment_configs = segment_configs
        self.segment_names = list(segment_configs.keys())
        self.encoder_type = encoder_type
        self.n_classes = n_classes
        
        # Calculate total latent dimension
        self.total_latent_dim = sum(
            cfg['latent_dim'] for cfg in segment_configs.values()
        )
        
        # Create encoders and decoders for each segment
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        for name, cfg in segment_configs.items():
            # Use safe key for ModuleDict (replace dots with underscores)
            safe_name = name.replace(".", "_")
            
            self.encoders[safe_name] = create_encoder(
                input_length=cfg['input_length'],
                latent_dim=cfg['latent_dim'],
                encoder_type=encoder_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_mlp_threshold=use_mlp_threshold,
                tcn_channels=tcn_channels,
                tcn_kernel_size=tcn_kernel_size,
                multiscale_kernel_sizes=multiscale_kernel_sizes,
                multiscale_channels=multiscale_channels,
            )
            
            self.decoders[safe_name] = create_decoder(
                latent_dim=cfg['latent_dim'],
                output_length=cfg['input_length'],
                encoder_type=encoder_type,
                hidden_dims=list(reversed(hidden_dims)) if hidden_dims else None,
                dropout=dropout,
                use_mlp_threshold=use_mlp_threshold,
                tcn_channels=tcn_channels,
                tcn_kernel_size=tcn_kernel_size,
                multiscale_kernel_sizes=multiscale_kernel_sizes,
                multiscale_channels=multiscale_channels,
            )
        
        # Optional classification head for semi-supervised training
        if n_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.total_latent_dim, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, n_classes),
            )
        else:
            self.classifier = None
        
        # Mapping from segment names to safe names
        self._name_map = {name: name.replace(".", "_") for name in segment_configs.keys()}
        self._reverse_name_map = {v: k for k, v in self._name_map.items()}
    
    def encode(self, segments: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Encode all segments to latent representations.
        
        Args:
            segments: Dict mapping segment_name to (batch, length) tensors.
        
        Returns:
            Dict mapping segment_name to (batch, latent_dim) embeddings.
        """
        embeddings = {}
        for name in self.segment_names:
            if name in segments:
                safe_name = self._name_map[name]
                embeddings[name] = self.encoders[safe_name](segments[name])
        return embeddings
    
    def decode(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Decode all segment embeddings to reconstructions.
        
        Args:
            embeddings: Dict mapping segment_name to (batch, latent_dim) embeddings.
        
        Returns:
            Dict mapping segment_name to (batch, length) reconstructions.
        """
        reconstructions = {}
        for name, z in embeddings.items():
            safe_name = self._name_map[name]
            reconstructions[name] = self.decoders[safe_name](z)
        return reconstructions
    
    def get_full_embedding(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate all segment embeddings into full 49D embedding.
        
        Args:
            embeddings: Dict mapping segment_name to (batch, latent_dim) embeddings.
        
        Returns:
            (batch, total_latent_dim) full embedding.
        """
        # Concatenate in fixed order
        ordered_embeddings = []
        for name in self.segment_names:
            if name in embeddings:
                ordered_embeddings.append(embeddings[name])
        
        return torch.cat(ordered_embeddings, dim=1)
    
    def forward(self, segments: dict[str, torch.Tensor]) -> dict:
        """
        Full forward pass: encode, concatenate, decode.
        
        Args:
            segments: Dict mapping segment_name to (batch, length) tensors.
        
        Returns:
            Dict with keys:
                'embeddings': Dict[str, Tensor] per-segment embeddings
                'full_embedding': Tensor (batch, 49) concatenated embedding
                'reconstructions': Dict[str, Tensor] per-segment reconstructions
        """
        # Encode all segments
        embeddings = self.encode(segments)
        
        # Get full concatenated embedding
        full_embedding = self.get_full_embedding(embeddings)
        
        # Decode all segments
        reconstructions = self.decode(embeddings)
        
        result = {
            'embeddings': embeddings,
            'full_embedding': full_embedding,
            'reconstructions': reconstructions,
        }
        
        # Classification head (semi-supervised)
        if self.classifier is not None:
            result['class_logits'] = self.classifier(full_embedding)
        
        return result
    
    @classmethod
    def from_segment_lengths(
        cls,
        segment_lengths: dict[str, int],
        segment_latent_dims: dict[str, int],
        encoder_type: str = "tcn",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_mlp_threshold: int = 30,
        tcn_channels: list[int] | None = None,
        tcn_kernel_size: int = 3,
        multiscale_kernel_sizes: list[int] | None = None,
        multiscale_channels: int = 32,
        n_classes: int = 0,
        classifier_hidden: int = 32,
    ) -> 'MultiSegmentAutoencoder':
        """
        Create autoencoder from segment lengths and latent dimensions.
        
        Args:
            segment_lengths: Dict mapping segment_name to input length.
            segment_latent_dims: Dict mapping segment_name to latent dimension.
            encoder_type: "tcn" (default), "cnn", or "multiscale".
            hidden_dims: Hidden layer dimensions (for CNN).
            dropout: Dropout probability.
            use_mlp_threshold: Use MLP for segments shorter than this.
            tcn_channels: Channel dimensions for TCN blocks.
            tcn_kernel_size: Kernel size for TCN.
            multiscale_kernel_sizes: Kernel sizes for multi-scale branches.
            multiscale_channels: Channels per branch for multi-scale.
            n_classes: Number of classification outputs (0 = no classifier).
            classifier_hidden: Hidden layer size for classifier head.
        
        Returns:
            MultiSegmentAutoencoder instance.
        """
        segment_configs = {}
        for name in segment_lengths:
            if name in segment_latent_dims:
                segment_configs[name] = {
                    'input_length': segment_lengths[name],
                    'latent_dim': segment_latent_dims[name],
                }
        
        return cls(
            segment_configs=segment_configs,
            encoder_type=encoder_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_mlp_threshold=use_mlp_threshold,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            multiscale_kernel_sizes=multiscale_kernel_sizes,
            multiscale_channels=multiscale_channels,
            n_classes=n_classes,
            classifier_hidden=classifier_hidden,
        )
