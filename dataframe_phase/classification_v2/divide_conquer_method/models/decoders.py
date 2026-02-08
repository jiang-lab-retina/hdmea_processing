"""
Segment decoders for the multi-segment autoencoder.

Each decoder reconstructs a trace segment from its latent representation.
Supports multiple decoder types: TCN (default), CNN, MultiScale.
"""

import torch
import torch.nn as nn

# Import TCN decoders - handle both package and direct import
try:
    from .tcn_encoder import TCNDecoder, MultiScaleDecoder
except ImportError:
    from tcn_encoder import TCNDecoder, MultiScaleDecoder


class SegmentDecoderCNN(nn.Module):
    """
    1D CNN decoder for a single trace segment.
    
    Args:
        latent_dim: Input embedding dimension.
        output_length: Target output trace length.
        hidden_dims: List of hidden channel dimensions (in reverse order from encoder).
        dropout: Dropout probability.
    
    Forward:
        z: (batch, latent_dim) → x_hat: (batch, output_length)
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_length = output_length
        
        # Default hidden dimensions (reverse of encoder)
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        # Initial projection from latent space
        self.initial_size = 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0] * self.initial_size),
            nn.ReLU(inplace=True),
        )
        
        # Build transposed convolutional layers
        layers = []
        in_channels = hidden_dims[0]
        
        for i, out_channels in enumerate(hidden_dims[1:]):
            if i == len(hidden_dims) - 2:
                kernel_size = 7
            elif i == len(hidden_dims) - 3:
                kernel_size = 5
            else:
                kernel_size = 3
            
            layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=2, padding=kernel_size // 2, output_padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        # Final layer to get single channel output
        layers.append(
            nn.ConvTranspose1d(in_channels, 1, kernel_size=3, stride=2, 
                              padding=1, output_padding=1)
        )
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final adjustment to exact output length
        self.adjust = nn.Linear(self._estimate_output_size(hidden_dims), output_length)
    
    def _estimate_output_size(self, hidden_dims: list[int]) -> int:
        """Estimate output size after transposed convolutions."""
        size = self.initial_size
        for i in range(len(hidden_dims)):
            size = size * 2
        return size
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: (batch, latent_dim) latent embeddings
        
        Returns:
            (batch, output_length) reconstructed traces
        """
        # Project from latent space
        x = self.fc(z)
        
        # Reshape for conv layers: (batch, features) -> (batch, channels, length)
        x = x.view(x.size(0), -1, self.initial_size)
        
        # Apply transposed conv layers
        x = self.conv_layers(x)
        
        # Flatten and adjust to exact output length
        x = x.squeeze(1)  # Remove channel dim if present
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        
        x = self.adjust(x)
        
        return x


class SegmentDecoderMLP(nn.Module):
    """
    MLP decoder for short segments.
    
    Simpler architecture matching the MLP encoder.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_length = output_length
        
        if hidden_dims is None:
            hidden_dims = [32, 64]
        
        layers = []
        in_features = latent_dim
        
        for out_features in hidden_dims:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = out_features
        
        layers.append(nn.Linear(in_features, output_length))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, latent_dim) -> (batch, output_length)"""
        return self.mlp(z)


def create_decoder(
    latent_dim: int,
    output_length: int,
    encoder_type: str = "tcn",
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
    use_mlp_threshold: int = 30,
    tcn_channels: list[int] | None = None,
    tcn_kernel_size: int = 3,
    multiscale_kernel_sizes: list[int] | None = None,
    multiscale_channels: int = 32,
) -> nn.Module:
    """
    Factory function to create appropriate decoder based on type and output length.
    
    Args:
        latent_dim: Input embedding dimension.
        output_length: Target output trace length.
        encoder_type: "tcn" (default), "cnn", or "multiscale" - matches encoder.
        hidden_dims: Hidden layer dimensions (for CNN).
        dropout: Dropout probability.
        use_mlp_threshold: Use MLP for segments shorter than this.
        tcn_channels: Channel dimensions for TCN blocks (reversed).
        tcn_kernel_size: Kernel size for TCN.
        multiscale_kernel_sizes: Kernel sizes for multi-scale branches.
        multiscale_channels: Channels per branch for multi-scale.
    
    Returns:
        Decoder module.
    """
    # Use MLP for very short segments regardless of encoder_type
    if output_length < use_mlp_threshold:
        return SegmentDecoderMLP(latent_dim, output_length, hidden_dims, dropout)
    
    # Select decoder based on type (matches encoder)
    if encoder_type == "tcn":
        # Reverse channels for decoder
        decoder_channels = list(reversed(tcn_channels)) if tcn_channels else None
        return TCNDecoder(
            latent_dim=latent_dim,
            output_length=output_length,
            num_channels=decoder_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
    elif encoder_type == "multiscale":
        return MultiScaleDecoder(
            latent_dim=latent_dim,
            output_length=output_length,
            branch_channels=multiscale_channels,
            kernel_sizes=multiscale_kernel_sizes,
            dropout=dropout,
        )
    elif encoder_type == "cnn":
        return SegmentDecoderCNN(latent_dim, output_length, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown decoder type: {encoder_type}. Use 'tcn', 'cnn', or 'multiscale'.")
