"""
HD-MEA Data Analysis Pipeline

A modular Python package for processing high-density multi-electrode array (HD-MEA)
recordings, extracting physiological features, and supporting extensible analyses.

Architecture:
    - io/         : Raw file I/O (CMCR/CMTR) and Zarr operations
    - preprocess/ : Data cleaning, alignment, filtering
    - features/   : Feature extractors with registry pattern
    - analysis/   : Downstream analyses
    - viz/        : Visualization utilities
    - pipeline/   : Orchestration and caching
    - utils/      : Shared utilities (logging, hashing, validation)
"""

__version__ = "0.1.0"
__author__ = "Jiang Lab"

# Lazy imports to avoid circular dependencies
# Users should import from subpackages directly:
#   from hdmea.io import load_recording
#   from hdmea.features import FeatureRegistry

