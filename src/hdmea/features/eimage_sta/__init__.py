"""
Electrode Image Spike-Triggered Average (eimage_sta) feature extraction.

This module computes the spike-triggered average of sensor data across the
HD-MEA electrode array, revealing axonal footprints and network activity
patterns around each unit's spikes.

Key features:
    - Vectorized high-pass filtering for performance
    - Memory-mapped sensor data access for large files
    - Configurable filter and time window parameters
    - Optional filtered data caching

Example:
    >>> from hdmea.features.eimage_sta import compute_eimage_sta
    >>> result = compute_eimage_sta(
    ...     "artifacts/recording.h5",
    ...     "O:/data/recording.cmcr",
    ... )
    >>> print(f"Processed {result.units_processed} units")
"""

from hdmea.features.eimage_sta.compute import (
    compute_eimage_sta,
    compute_eimage_sta_chunked,
    compute_eimage_sta_legacy,
    compute_eimage_sta_legacy_improved,
    compute_sta_for_unit,
    compute_sta_for_unit_legacy,
    EImageSTAConfig,
    EImageSTAResult,
    STAAccumulator,
)
from hdmea.features.eimage_sta.extractor import EImageSTAExtractor

__all__ = [
    "compute_eimage_sta",
    "compute_eimage_sta_chunked",
    "compute_eimage_sta_legacy",
    "compute_eimage_sta_legacy_improved",
    "compute_sta_for_unit",
    "compute_sta_for_unit_legacy",
    "EImageSTAConfig",
    "EImageSTAResult",
    "STAAccumulator",
    "EImageSTAExtractor",
]

