"""
Clustering algorithms for RGC subtype discovery.

Includes:
- GMM/BIC: Gaussian Mixture Model with BIC-based k selection
- DEC refinement: Deep Embedded Clustering for cluster sharpening
"""

from .gmm_bic import fit_gmm_bic, select_k_min_bic
from .dec_refine import refine_with_dec

__all__ = [
    "fit_gmm_bic",
    "select_k_min_bic",
    "refine_with_dec",
]
