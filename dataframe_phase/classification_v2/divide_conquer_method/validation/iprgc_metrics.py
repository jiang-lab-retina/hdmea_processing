"""
ipRGC validation metrics for cluster quality assessment.

Implements:
- ipRGC purity: How well clusters separate ipRGC vs non-ipRGC
- ipRGC enrichment: Which clusters are enriched for ipRGCs
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Support both direct and module execution
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from .. import config
except ImportError:
    from divide_conquer_method import config

logger = logging.getLogger(__name__)


def compute_iprgc_metrics(
    cluster_labels: np.ndarray,
    iprgc_labels: np.ndarray,
) -> dict:
    """
    Compute comprehensive ipRGC validation metrics.
    
    Args:
        cluster_labels: (n_cells,) cluster assignments.
        iprgc_labels: (n_cells,) binary ipRGC labels (True/False or 1/0).
    
    Returns:
        Dict with:
            - purity: Overall purity score
            - baseline_prevalence: P(ipRGC) in dataset
            - per_cluster: Dict of per-cluster stats
            - top_enriched: List of top 3 enriched clusters
    """
    iprgc_labels = np.asarray(iprgc_labels).astype(bool)
    cluster_labels = np.asarray(cluster_labels)
    
    n_cells = len(cluster_labels)
    n_iprgc = iprgc_labels.sum()
    baseline_prevalence = n_iprgc / n_cells
    
    logger.info(f"Computing ipRGC metrics: {n_cells} cells, {n_iprgc} ipRGCs "
               f"({100*baseline_prevalence:.1f}%)")
    
    # Compute per-cluster statistics
    unique_clusters = np.unique(cluster_labels)
    per_cluster = {}
    
    for c in unique_clusters:
        mask = cluster_labels == c
        n_in_cluster = mask.sum()
        n_iprgc_in_cluster = (mask & iprgc_labels).sum()
        
        fraction = n_iprgc_in_cluster / n_in_cluster if n_in_cluster > 0 else 0
        enrichment = fraction / baseline_prevalence if baseline_prevalence > 0 else 0
        
        per_cluster[int(c)] = {
            'n_cells': int(n_in_cluster),
            'n_iprgc': int(n_iprgc_in_cluster),
            'fraction': float(fraction),
            'enrichment': float(enrichment),
        }
    
    # Compute overall purity
    purity = compute_purity(cluster_labels, iprgc_labels)
    
    # Find top enriched clusters
    top_enriched = sorted(
        per_cluster.items(),
        key=lambda x: x[1]['enrichment'],
        reverse=True,
    )[:3]
    
    top_enriched_list = [
        {'cluster': c, **stats}
        for c, stats in top_enriched
    ]
    
    return {
        'purity': float(purity),
        'baseline_prevalence': float(baseline_prevalence),
        'per_cluster': per_cluster,
        'top_enriched': top_enriched_list,
    }


def compute_purity(
    cluster_labels: np.ndarray,
    binary_labels: np.ndarray,
) -> float:
    """
    Compute purity with respect to a binary label.
    
    Purity = (1/N) * sum_c max(n_c_positive, n_c_negative)
    
    This measures how well clusters separate the two classes.
    
    Args:
        cluster_labels: (n_cells,) cluster assignments.
        binary_labels: (n_cells,) binary labels (True/False or 1/0).
    
    Returns:
        Purity score in [0.5, 1.0] (0.5 = random, 1.0 = perfect separation).
    """
    binary_labels = np.asarray(binary_labels).astype(bool)
    cluster_labels = np.asarray(cluster_labels)
    
    n_cells = len(cluster_labels)
    if n_cells == 0:
        return 0.5
    
    unique_clusters = np.unique(cluster_labels)
    
    purity_sum = 0
    for c in unique_clusters:
        mask = cluster_labels == c
        n_positive = (mask & binary_labels).sum()
        n_negative = (mask & ~binary_labels).sum()
        purity_sum += max(n_positive, n_negative)
    
    purity = purity_sum / n_cells
    return purity


def compute_enrichment(
    cluster_labels: np.ndarray,
    binary_labels: np.ndarray,
) -> Dict[int, float]:
    """
    Compute enrichment for each cluster.
    
    Enrichment_c = P(positive | cluster=c) / P(positive)
    
    Args:
        cluster_labels: (n_cells,) cluster assignments.
        binary_labels: (n_cells,) binary labels.
    
    Returns:
        Dict mapping cluster ID to enrichment score.
    """
    binary_labels = np.asarray(binary_labels).astype(bool)
    cluster_labels = np.asarray(cluster_labels)
    
    n_cells = len(cluster_labels)
    baseline = binary_labels.sum() / n_cells if n_cells > 0 else 0
    
    if baseline == 0:
        logger.warning("No positive labels, enrichment undefined")
        return {}
    
    unique_clusters = np.unique(cluster_labels)
    enrichment = {}
    
    for c in unique_clusters:
        mask = cluster_labels == c
        n_in_cluster = mask.sum()
        n_positive = (mask & binary_labels).sum()
        
        if n_in_cluster > 0:
            fraction = n_positive / n_in_cluster
            enrichment[int(c)] = fraction / baseline
        else:
            enrichment[int(c)] = 0.0
    
    return enrichment


def get_iprgc_labels(
    df,
    qi_threshold: float | None = None,
) -> np.ndarray:
    """
    Extract ipRGC binary labels from DataFrame.
    
    Args:
        df: DataFrame with iprgc_2hz_QI column.
        qi_threshold: QI threshold for ipRGC classification.
    
    Returns:
        (n_cells,) boolean array.
    """
    qi_threshold = qi_threshold if qi_threshold is not None else config.IPRGC_QI_THRESHOLD
    
    if config.IPRGC_QI_COL not in df.columns:
        logger.warning(f"Column {config.IPRGC_QI_COL} not found, using all False")
        return np.zeros(len(df), dtype=bool)
    
    qi_values = df[config.IPRGC_QI_COL].fillna(0).values
    iprgc_labels = qi_values > qi_threshold
    
    n_iprgc = iprgc_labels.sum()
    logger.info(f"ipRGC labels: {n_iprgc}/{len(df)} cells "
               f"({100*n_iprgc/len(df):.1f}%) with QI > {qi_threshold}")
    
    return iprgc_labels


if __name__ == "__main__":
    # Demo with synthetic data
    logging.basicConfig(level=logging.INFO)
    
    print("ipRGC Metrics Demo")
    print("=" * 40)
    
    # Create synthetic data: 100 cells, 10 clusters, 10% ipRGC
    np.random.seed(42)
    n_cells = 100
    n_clusters = 10
    
    cluster_labels = np.random.randint(0, n_clusters, size=n_cells)
    
    # Make clusters 0 and 5 enriched for ipRGCs
    iprgc_labels = np.zeros(n_cells, dtype=bool)
    iprgc_labels[cluster_labels == 0] = np.random.rand((cluster_labels == 0).sum()) > 0.3  # 70% ipRGC
    iprgc_labels[cluster_labels == 5] = np.random.rand((cluster_labels == 5).sum()) > 0.5  # 50% ipRGC
    
    print(f"\nSynthetic data: {n_cells} cells, {n_clusters} clusters")
    print(f"Total ipRGCs: {iprgc_labels.sum()} ({100*iprgc_labels.mean():.1f}%)")
    
    # Compute metrics
    metrics = compute_iprgc_metrics(cluster_labels, iprgc_labels)
    
    print(f"\nResults:")
    print(f"  Purity: {metrics['purity']:.3f}")
    print(f"  Baseline prevalence: {metrics['baseline_prevalence']:.3f}")
    print(f"\nTop enriched clusters:")
    for i, cluster_info in enumerate(metrics['top_enriched']):
        print(f"  {i+1}. Cluster {cluster_info['cluster']}: "
              f"enrichment={cluster_info['enrichment']:.2f}x, "
              f"fraction={cluster_info['fraction']:.2f}, "
              f"n={cluster_info['n_cells']}")
