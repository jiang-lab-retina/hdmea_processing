"""
AP Tracking (Axon Trace) Feature Module for HDF5 Pipeline.

This module provides axon tracking analysis for HD-MEA recordings stored in HDF5 format.
It detects soma location, predicts axon pathways using a trained CNN model, and calculates
soma polar coordinates relative to the optic disc.

Public API:
    compute_ap_tracking: Process a single HDF5 file with AP tracking analysis
    compute_ap_tracking_batch: Process multiple HDF5 files with progress tracking

Example:
    >>> from hdmea.features.ap_tracking import compute_ap_tracking
    >>> compute_ap_tracking(hdf5_path, model_path)
"""

from .core import compute_ap_tracking, compute_ap_tracking_batch

# Also export key data classes for type hints
from .ais_refiner import AxonInitialSegment, RefinedSoma
from .dvnt_parser import DVNTPosition
from .pathway_analysis import APIntersection, APPathway, SomaPolarCoordinates

__all__ = [
    # Main entry points
    "compute_ap_tracking",
    "compute_ap_tracking_batch",
    # Data classes
    "RefinedSoma",
    "AxonInitialSegment",
    "DVNTPosition",
    "APPathway",
    "APIntersection",
    "SomaPolarCoordinates",
]

