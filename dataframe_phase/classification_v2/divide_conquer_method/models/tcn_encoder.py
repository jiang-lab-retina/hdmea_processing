"""
Temporal Convolutional Network (TCN) encoder for neural time series.

Better than standard CNN for capturing multi-scale temporal patterns
in neural firing rate data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CausalConv1d(nn.Module):
    """
    Causal convolution - only looks at past/current, not future.
    Important for temporal data where causality matters.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Single TCN block with residual connection.
    
    Structure:
        Input → CausalConv → BN → ReLU → Dropout → 
              → CausalConv → BN → ReLU → Dropout → + → Output
                                                   ↑
        Input ─────────────────────────────────────┘ (residual)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channels change)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return F.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.
    
    Uses dilated convolutions to capture patterns at multiple timescales
    without aggressive pooling that loses temporal information.
    
    Args:
        input_length: Length of input trace.
        latent_dim: Output embedding dimension.
        num_channels: List of channel dimensions for each TCN block.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
    
    Forward:
        x: (batch, input_length) → z: (batch, latent_dim)
    """
    
    def __init__(
        self,
        input_length: int,
        latent_dim: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Default channels with increasing dilation
        if num_channels is None:
            num_channels = [32, 32, 64, 64]
        
        # Build TCN blocks with exponentially increasing dilation
        blocks = []
        in_channels = 1  # Input is single-channel time series
        
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            blocks.append(TCNBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*blocks)
        
        # Global pooling + projection to latent space
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension: (batch, length) -> (batch, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply TCN blocks
        out = self.tcn(x)
        
        # Global average pooling
        out = self.pool(out).squeeze(-1)
        
        # Project to latent space
        z = self.fc(out)
        
        return z


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale CNN encoder with parallel branches.
    
    Captures features at different temporal scales simultaneously,
    which is useful for neural data with mixed fast/slow dynamics.
    
    Args:
        input_length: Length of input trace.
        latent_dim: Output embedding dimension.
        branch_channels: Channels per branch.
        kernel_sizes: Kernel sizes for each branch.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_length: int,
        latent_dim: int,
        branch_channels: int = 32,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]  # Fast, medium, slow scales
        
        # Create parallel branches
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(1, branch_channels, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(branch_channels, branch_channels, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(4),
            )
            self.branches.append(branch)
        
        # Combine branches
        combined_size = branch_channels * len(kernel_sizes) * 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply all branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        combined = torch.cat(branch_outputs, dim=1)
        
        # Project to latent space
        z = self.fc(combined)
        
        return z


class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network decoder.
    
    Mirrors the TCN encoder structure for reconstruction.
    
    Args:
        latent_dim: Input embedding dimension.
        output_length: Target output trace length.
        num_channels: List of channel dimensions (reversed from encoder).
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
    
    Forward:
        z: (batch, latent_dim) → x: (batch, output_length)
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_length = output_length
        
        # Default channels (reversed from encoder)
        if num_channels is None:
            num_channels = [64, 64, 32, 32]
        
        # Project latent to initial sequence
        self.initial_length = max(8, output_length // 16)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, num_channels[0] * self.initial_length),
            nn.ReLU(inplace=True),
        )
        
        # Build TCN blocks with decreasing dilation
        blocks = []
        in_channels = num_channels[0]
        num_blocks = len(num_channels)
        
        for i, out_channels in enumerate(num_channels):
            # Reverse dilation order
            dilation = 2 ** (num_blocks - 1 - i)
            blocks.append(TCNBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*blocks)
        
        # Final projection to output length
        self.final_conv = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        self.adjust = nn.Linear(self.initial_length, output_length)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project to initial sequence
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.initial_length)
        
        # Apply TCN blocks
        x = self.tcn(x)
        
        # Final projection
        x = self.final_conv(x)
        x = x.squeeze(1)
        x = self.adjust(x)
        
        return x


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale CNN decoder with parallel branches.
    
    Mirrors the MultiScaleEncoder structure.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_length: int,
        branch_channels: int = 32,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_length = output_length
        
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]
        
        # Initial projection
        self.initial_length = max(8, output_length // 8)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, branch_channels * len(kernel_sizes) * self.initial_length),
            nn.ReLU(inplace=True),
        )
        
        # Create parallel branches
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(branch_channels, branch_channels, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(branch_channels, 1, kernel_size=ks, padding=ks // 2),
            )
            self.branches.append(branch)
        
        self.num_branches = len(kernel_sizes)
        self.branch_channels = branch_channels
        
        # Combine and adjust
        self.adjust = nn.Linear(self.initial_length * len(kernel_sizes), output_length)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Project to initial sequences
        x = self.fc(z)
        x = x.view(batch_size, self.num_branches, self.branch_channels, self.initial_length)
        
        # Apply branches and collect outputs
        outputs = []
        for i, branch in enumerate(self.branches):
            branch_input = x[:, i, :, :]  # (batch, channels, length)
            out = branch(branch_input)    # (batch, 1, length)
            outputs.append(out.squeeze(1))
        
        # Concatenate and adjust
        combined = torch.cat(outputs, dim=1)
        out = self.adjust(combined)
        
        return out


def create_tcn_encoder(
    input_length: int,
    latent_dim: int,
    encoder_type: str = "tcn",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create TCN or multi-scale encoder.
    
    Args:
        input_length: Length of input trace.
        latent_dim: Output embedding dimension.
        encoder_type: "tcn" or "multiscale".
        **kwargs: Additional arguments for encoder.
    
    Returns:
        Encoder module.
    """
    if encoder_type == "tcn":
        return TCNEncoder(input_length, latent_dim, **kwargs)
    elif encoder_type == "multiscale":
        return MultiScaleEncoder(input_length, latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def create_tcn_decoder(
    latent_dim: int,
    output_length: int,
    encoder_type: str = "tcn",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create TCN or multi-scale decoder.
    
    Args:
        latent_dim: Input embedding dimension.
        output_length: Target output trace length.
        encoder_type: "tcn" or "multiscale" (matches encoder).
        **kwargs: Additional arguments for decoder.
    
    Returns:
        Decoder module.
    """
    if encoder_type == "tcn":
        return TCNDecoder(latent_dim, output_length, **kwargs)
    elif encoder_type == "multiscale":
        return MultiScaleDecoder(latent_dim, output_length, **kwargs)
    else:
        raise ValueError(f"Unknown decoder type: {encoder_type}")
