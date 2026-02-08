"""
Segment encoders for the multi-segment autoencoder.

Each encoder compresses a single trace segment into a fixed-size latent vector.
Supports multiple encoder types: TCN (default), CNN, MultiScale.
"""

import torch
import torch.nn as nn

# Import TCN encoders - handle both package and direct import
try:
    from .tcn_encoder import TCNEncoder, MultiScaleEncoder
except ImportError:
    from tcn_encoder import TCNEncoder, MultiScaleEncoder


class SegmentEncoderCNN(nn.Module):
    """
    1D CNN encoder for a single trace segment.
    
    Args:
        input_length: Length of input trace.
        latent_dim: Output embedding dimension.
        hidden_dims: List of hidden channel dimensions.
        dropout: Dropout probability.
    
    Forward:
        x: (batch, input_length) → z: (batch, latent_dim)
    """
    
    def __init__(
        self,
        input_length: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        # Build convolutional layers
        layers = []
        in_channels = 1  # Input is 1D signal
        
        for i, out_channels in enumerate(hidden_dims):
            # Determine kernel size based on position in network
            if i == 0:
                kernel_size = 7
            elif i == 1:
                kernel_size = 5
            else:
                kernel_size = 3
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after convolutions
        self._conv_output_size = self._get_conv_output_size(input_length, hidden_dims)
        
        # Final projection to latent space
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),  # Pool to fixed size
            nn.Flatten(),
            nn.Linear(hidden_dims[-1] * 4, latent_dim),
        )
    
    def _get_conv_output_size(self, input_length: int, hidden_dims: list[int]) -> int:
        """Calculate the output size after all conv layers."""
        size = input_length
        for i in range(len(hidden_dims)):
            kernel_size = 7 if i == 0 else (5 if i == 1 else 3)
            padding = kernel_size // 2
            stride = 2
            size = (size + 2 * padding - kernel_size) // stride + 1
        return size * hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_length) input traces
        
        Returns:
            (batch, latent_dim) latent embeddings
        """
        # Add channel dimension: (batch, length) -> (batch, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Project to latent space
        z = self.fc(x)
        
        return z


class SegmentEncoderMLP(nn.Module):
    """
    MLP encoder for short segments (e.g., STA time course).
    
    Simpler architecture for segments with few time points.
    """
    
    def __init__(
        self,
        input_length: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        in_features = input_length
        
        for out_features in hidden_dims:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = out_features
        
        layers.append(nn.Linear(in_features, latent_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, input_length) -> (batch, latent_dim)"""
        return self.mlp(x)


def create_encoder(
    input_length: int,
    latent_dim: int,
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
    Factory function to create appropriate encoder based on type and input length.
    
    Args:
        input_length: Length of input trace.
        latent_dim: Output embedding dimension.
        encoder_type: "tcn" (default), "cnn", or "multiscale".
        hidden_dims: Hidden layer dimensions (for CNN).
        dropout: Dropout probability.
        use_mlp_threshold: Use MLP for segments shorter than this.
        tcn_channels: Channel dimensions for TCN blocks.
        tcn_kernel_size: Kernel size for TCN.
        multiscale_kernel_sizes: Kernel sizes for multi-scale branches.
        multiscale_channels: Channels per branch for multi-scale.
    
    Returns:
        Encoder module.
    """
    # Use MLP for very short segments regardless of encoder_type
    if input_length < use_mlp_threshold:
        return SegmentEncoderMLP(input_length, latent_dim, hidden_dims, dropout)
    
    # Select encoder based on type
    if encoder_type == "tcn":
        return TCNEncoder(
            input_length=input_length,
            latent_dim=latent_dim,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
    elif encoder_type == "multiscale":
        return MultiScaleEncoder(
            input_length=input_length,
            latent_dim=latent_dim,
            branch_channels=multiscale_channels,
            kernel_sizes=multiscale_kernel_sizes,
            dropout=dropout,
        )
    elif encoder_type == "cnn":
        return SegmentEncoderCNN(input_length, latent_dim, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'tcn', 'cnn', or 'multiscale'.")
