"""
Validation metrics for RGC clustering.

Includes:
- ipRGC enrichment: Validate cluster quality using ipRGC ground truth
- Purity metrics: Measure cluster-label correspondence
"""

from .iprgc_metrics import compute_iprgc_metrics, compute_enrichment, compute_purity

__all__ = [
    "compute_iprgc_metrics",
    "compute_enrichment",
    "compute_purity",
]
