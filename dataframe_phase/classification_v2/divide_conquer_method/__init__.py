"""
DEC-Refined RGC Subtype Clustering Pipeline.

A two-phase clustering pipeline for RGC subtypes:
1. Train CNN autoencoder (reconstruction-only), select k* via GMM + BIC
2. Refine clusters using DEC/IDEC with GMM-initialized centers
3. Validate via ipRGC enrichment
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "Jiang Lab"

# Package root directory
PACKAGE_DIR = Path(__file__).parent

# Export main pipeline function
from .run_pipeline import main

__all__ = [
    "main",
    "PACKAGE_DIR",
    "__version__",
]
