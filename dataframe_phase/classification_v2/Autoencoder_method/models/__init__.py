"""
Neural network models for the autoencoder-based clustering pipeline.

This subpackage contains:
- encoders.py: SegmentEncoder class for per-segment encoding
- decoders.py: SegmentDecoder class for per-segment decoding
- autoencoder.py: MultiSegmentAutoencoder combining all encoders/decoders
- losses.py: Loss functions including reconstruction and supervised contrastive
"""

from .encoders import SegmentEncoder, SegmentEncoderMLP, create_encoder
from .decoders import SegmentDecoder, SegmentDecoderMLP, create_decoder
from .autoencoder import MultiSegmentAutoencoder
from .losses import WeightedReconstructionLoss, SupervisedContrastiveLoss, CombinedAELoss

__all__ = [
    "SegmentEncoder",
    "SegmentEncoderMLP",
    "create_encoder",
    "SegmentDecoder",
    "SegmentDecoderMLP",
    "create_decoder",
    "MultiSegmentAutoencoder",
    "WeightedReconstructionLoss",
    "SupervisedContrastiveLoss",
    "CombinedAELoss",
]
