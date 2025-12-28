"""
Feature extraction module for HD-MEA pipeline.

Provides:
    - FeatureRegistry for extensible feature registration
    - FeatureExtractor base class
    - Built-in extractors for core features

All extractors are automatically registered on import via the
@FeatureRegistry.register() decorator.
"""

from hdmea.features.registry import FeatureRegistry
from hdmea.features.base import FeatureExtractor

# Import all extractors to trigger registration
# Each extractor module uses @FeatureRegistry.register() decorator
from hdmea.features.on_off import StepUpFeatureExtractor
from hdmea.features.baseline import BaselineFeatureExtractor
from hdmea.features.direction import MovingBarFeatureExtractor
from hdmea.features.receptive_field import DenseNoiseFeatureExtractor
from hdmea.features.chromatic import ChromaticFeatureExtractor
from hdmea.features.frequency import FrequencyFeatureExtractor
from hdmea.features.cell_type import CellTypeFeatureExtractor
from hdmea.features.example import ExampleFeatureExtractor
from hdmea.features.frif import FRIFExtractor
from hdmea.features.sta import compute_sta, STAResult
from hdmea.features.eimage_sta import (
    EImageSTAExtractor,
    compute_eimage_sta,
    EImageSTAConfig,
    EImageSTAResult,
)
from hdmea.features.dsgc_direction import (
    section_by_direction,
    DirectionSectionResult,
    DIRECTION_LIST,
)

__all__ = [
    "FeatureRegistry",
    "FeatureExtractor",
    "StepUpFeatureExtractor",
    "BaselineFeatureExtractor",
    "MovingBarFeatureExtractor",
    "DenseNoiseFeatureExtractor",
    "ChromaticFeatureExtractor",
    "FrequencyFeatureExtractor",
    "CellTypeFeatureExtractor",
    "ExampleFeatureExtractor",
    "FRIFExtractor",
    "compute_sta",
    "STAResult",
    "EImageSTAExtractor",
    "compute_eimage_sta",
    "EImageSTAConfig",
    "EImageSTAResult",
    "section_by_direction",
    "DirectionSectionResult",
    "DIRECTION_LIST",
]

