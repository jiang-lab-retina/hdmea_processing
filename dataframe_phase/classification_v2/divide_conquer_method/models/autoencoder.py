"""
Multi-segment autoencoder combining all segment encoders and decoders.

This is the main model that produces 49D embeddings from 10 stimulus segments.
Simplified version for DEC pipeline (reconstruction-only, no SupCon/purity).
"""

import torch
import torch.nn as nn

from .encoders import create_encoder
from .decoders import create_decoder


class MultiSegmentAutoencoder(nn.Module):
    """
    Multi-segment autoencoder with per-segment encoders/decoders.
    
    Simplified for reconstruction-only training (no weak supervision).
    
    Args:
        segment_configs: Dict mapping segment_name to {
            'input_length': int,
            'latent_dim': int,
        }
        hidden_dims: List of hidden dimensions for all encoders.
        dropout: Dropout probability.
    
    Forward:
        segments: Dict[str, Tensor(batch, length)] → {
            'embeddings': Dict[str, Tensor(batch, latent_dim)],
            'full_embedding': Tensor(batch, total_latent_dim),
            'reconstructions': Dict[str, Tensor(batch, length)],
        }
    
    Properties:
        total_latent_dim: Sum of all segment latent dims (49)
    """
    
    def __init__(
        self,
        segment_configs: dict[str, dict],
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.segment_configs = segment_configs
        self.segment_names = list(segment_configs.keys())
        
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
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            
            self.decoders[safe_name] = create_decoder(
                latent_dim=cfg['latent_dim'],
                output_length=cfg['input_length'],
                hidden_dims=list(reversed(hidden_dims)) if hidden_dims else None,
                dropout=dropout,
            )
        
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
        
        return {
            'embeddings': embeddings,
            'full_embedding': full_embedding,
            'reconstructions': reconstructions,
        }
    
    @classmethod
    def from_segment_lengths(
        cls,
        segment_lengths: dict[str, int],
        segment_latent_dims: dict[str, int],
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> 'MultiSegmentAutoencoder':
        """
        Create autoencoder from segment lengths and latent dimensions.
        
        Args:
            segment_lengths: Dict mapping segment_name to input length.
            segment_latent_dims: Dict mapping segment_name to latent dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout probability.
        
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
        
        return cls(segment_configs, hidden_dims, dropout)
