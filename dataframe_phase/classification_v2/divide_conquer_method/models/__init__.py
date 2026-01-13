"""
Neural network models for DEC-refined RGC clustering.

Includes:
- MultiSegmentAutoencoder: CNN autoencoder for multi-segment traces
- Segment encoders/decoders: 1D CNN for individual stimulus segments
- DECLayer: Deep Embedded Clustering layer with Student-t kernel
"""

from .autoencoder import MultiSegmentAutoencoder
from .encoders import SegmentEncoder
from .decoders import SegmentDecoder

__all__ = [
    "MultiSegmentAutoencoder",
    "SegmentEncoder",
    "SegmentDecoder",
]
