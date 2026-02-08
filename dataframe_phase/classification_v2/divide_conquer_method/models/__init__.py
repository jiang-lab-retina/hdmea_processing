"""
Neural network models for DEC-refined RGC clustering.

Includes:
- MultiSegmentAutoencoder: Multi-segment autoencoder (TCN/CNN/MultiScale)
- Segment encoders/decoders: 1D CNN, TCN, or MultiScale for individual stimulus segments
- DECLayer: Deep Embedded Clustering layer with Student-t kernel
"""

from .autoencoder import MultiSegmentAutoencoder
from .encoders import SegmentEncoderCNN, SegmentEncoderMLP, create_encoder
from .decoders import SegmentDecoderCNN, SegmentDecoderMLP, create_decoder
from .tcn_encoder import TCNEncoder, TCNDecoder, MultiScaleEncoder, MultiScaleDecoder

__all__ = [
    "MultiSegmentAutoencoder",
    "SegmentEncoderCNN",
    "SegmentEncoderMLP",
    "SegmentDecoderCNN",
    "SegmentDecoderMLP",
    "TCNEncoder",
    "TCNDecoder",
    "MultiScaleEncoder",
    "MultiScaleDecoder",
    "create_encoder",
    "create_decoder",
]
