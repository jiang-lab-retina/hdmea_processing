"""
Evaluation metrics for the clustering pipeline.

Handles:
- Silhouette score for embedding quality
- Cluster purity metrics
- Cluster size validation
- Result saving
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from . import config

logger = logging.getLogger(__name__)


def compute_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 10000,
) -> float:
    """
    Compute silhouette score to verify group separation in embeddings.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        labels: (n_cells,) cluster or group labels.
        sample_size: Max samples for silhouette computation (for efficiency).
    
    Returns:
        Silhouette score in [-1, 1]. Higher is better separation.
    """
    n_samples = len(embeddings)
    n_unique = len(np.unique(labels))
    
    if n_unique < 2:
        logger.warning("Need at least 2 unique labels for silhouette score")
        return 0.0
    
    # Subsample if too large
    if sample_size is not None and n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    score = silhouette_score(embeddings, labels)
    logger.info(f"Silhouette score: {score:.4f}")
    
    return score


def compute_purity(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute cluster purity with respect to true labels.
    
    For each cluster, find the majority true label. Purity is the
    fraction of samples that match their cluster's majority label.
    
    Args:
        cluster_labels: Predicted cluster assignments.
        true_labels: Ground truth labels.
    
    Returns:
        Purity score in [0, 1]. Higher is better.
    
    Formula:
        purity = sum(max count per cluster) / total
    """
    total = len(cluster_labels)
    if total == 0:
        return 0.0
    
    purity_sum = 0
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[mask]
        
        if len(cluster_true_labels) > 0:
            # Find majority label count
            unique, counts = np.unique(cluster_true_labels, return_counts=True)
            purity_sum += counts.max()
    
    purity = purity_sum / total
    logger.debug(f"Purity: {purity:.4f}")
    
    return purity


def validate_group_purity(
    cluster_assignments: pd.DataFrame,
) -> bool:
    """
    Verify no cluster crosses group boundaries.
    
    Args:
        cluster_assignments: DataFrame with 'cluster_id' and 'coarse_group' columns.
    
    Returns:
        True if all clusters are group-pure, False otherwise.
    """
    for cluster_id in cluster_assignments['cluster_id'].unique():
        cluster_mask = cluster_assignments['cluster_id'] == cluster_id
        groups_in_cluster = cluster_assignments.loc[cluster_mask, 'coarse_group'].unique()
        
        if len(groups_in_cluster) > 1:
            logger.error(f"Cluster {cluster_id} contains multiple groups: {groups_in_cluster}")
            return False
    
    logger.info("All clusters are group-pure ✓")
    return True


def get_cluster_sizes(
    cluster_labels: np.ndarray,
) -> dict[int, int]:
    """
    Compute cluster size distribution.
    
    Args:
        cluster_labels: Cluster assignments.
    
    Returns:
        Dict mapping cluster_id to size.
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def validate_cluster_sizes(
    cluster_labels: np.ndarray,
    min_size: int | None = None,
) -> Tuple[bool, list[int]]:
    """
    Check if all clusters meet minimum size requirement.
    
    Args:
        cluster_labels: Cluster assignments.
        min_size: Minimum cluster size. Defaults to config.MIN_CLUSTER_SIZE.
    
    Returns:
        (all_valid, list of small cluster ids)
    """
    min_size = min_size if min_size is not None else config.MIN_CLUSTER_SIZE
    
    sizes = get_cluster_sizes(cluster_labels)
    small_clusters = [cid for cid, size in sizes.items() if size < min_size]
    
    if small_clusters:
        logger.warning(f"Clusters below min size {min_size}: {small_clusters}")
    
    return len(small_clusters) == 0, small_clusters


def save_embeddings(
    embeddings: np.ndarray,
    cell_ids: np.ndarray,
    groups: np.ndarray,
    model_version: str,
    output_path: Path | None = None,
) -> Path:
    """
    Save embeddings to parquet file.
    
    Args:
        embeddings: (n_cells, 49) embeddings.
        cell_ids: Cell identifiers.
        groups: Coarse group labels.
        model_version: Model version string.
        output_path: Output file path. Defaults to config.RESULTS_DIR/embeddings.parquet.
    
    Returns:
        Path to saved file.
    """
    output_path = output_path if output_path is not None else config.RESULTS_DIR / "embeddings.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'cell_id': cell_ids,
        'coarse_group': groups,
        'model_version': model_version,
    })
    
    # Add embedding columns
    for i in range(embeddings.shape[1]):
        df[f'z_{i}'] = embeddings[:, i]
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved embeddings to {output_path}")
    
    return output_path


def save_cluster_assignments(
    cell_ids: np.ndarray,
    groups: np.ndarray,
    cluster_ids: np.ndarray,
    posterior_probs: np.ndarray | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Save cluster assignments to parquet file.
    
    Args:
        cell_ids: Cell identifiers.
        groups: Coarse group labels.
        cluster_ids: Cluster assignments within group.
        posterior_probs: GMM posterior probabilities.
        output_path: Output file path.
    
    Returns:
        Path to saved file.
    """
    output_path = output_path if output_path is not None else config.RESULTS_DIR / "cluster_assignments.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create subtype labels
    subtype_labels = [
        f"{group}::cluster_{cid:02d}"
        for group, cid in zip(groups, cluster_ids)
    ]
    
    df = pd.DataFrame({
        'cell_id': cell_ids,
        'coarse_group': groups,
        'cluster_id': cluster_ids,
        'subtype_label': subtype_labels,
    })
    
    if posterior_probs is not None:
        df['posterior_prob'] = posterior_probs
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved cluster assignments to {output_path}")
    
    return output_path


def save_k_selection_data(
    group_results: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Save cluster number selection data (BIC curves, selected k) to JSON.
    
    Args:
        group_results: Dict from cluster_per_group with bic_values, k_range, k_selected.
        output_path: Output file path.
    
    Returns:
        Path to saved file.
    """
    output_path = output_path if output_path is not None else config.RESULTS_DIR / "k_selection.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON serialization
    selection_data = {
        'summary': {
            'total_clusters': sum(r.get('k_selected', 1) for r in group_results.values()),
            'groups': list(group_results.keys()),
        },
        'per_group': {}
    }
    
    for group_name, results in group_results.items():
        k_selected = results.get('k_selected', 1)
        k_range = results.get('k_range', [1])
        bic_values = results.get('bic_values', [0.0])
        
        # Compute log Bayes factors between consecutive k
        log_bf_values = []
        for i in range(1, len(bic_values)):
            log_bf = (bic_values[i-1] - bic_values[i]) / 2
            log_bf_values.append({
                'k': k_range[i],
                'k_prev': k_range[i-1],
                'log_bf': float(log_bf),
                'interpretation': 'strong' if log_bf >= 10 else ('moderate' if log_bf >= 3 else 'weak')
            })
        
        selection_data['per_group'][group_name] = {
            'k_selected': k_selected,
            'k_range': k_range,
            'bic_values': [float(b) for b in bic_values],
            'min_bic': float(min(bic_values)) if bic_values else None,
            'min_bic_k': k_range[np.argmin(bic_values)] if bic_values else None,
            'log_bayes_factors': log_bf_values,
            'selection_method': 'log_bayes_factor_threshold',
            'threshold': config.LOG_BF_THRESHOLD,
        }
    
    with open(output_path, 'w') as f:
        json.dump(selection_data, f, indent=2)
    
    logger.info(f"Saved k-selection data to {output_path}")
    
    # Also save a CSV summary
    csv_path = output_path.with_suffix('.csv')
    rows = []
    for group_name, data in selection_data['per_group'].items():
        rows.append({
            'group': group_name,
            'k_selected': data['k_selected'],
            'k_range_max': max(data['k_range']) if data['k_range'] else 0,
            'min_bic': data['min_bic'],
            'min_bic_k': data['min_bic_k'],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Saved k-selection summary to {csv_path}")
    
    return output_path


def compute_cv_purity_posthoc(
    cluster_labels: np.ndarray,
    groups: np.ndarray,
    df: pd.DataFrame | None = None,
) -> dict:
    """
    Compute post-hoc purity analysis for each coarse label.
    
    This measures how well clusters align with each label dimension,
    using ORIGINAL cell labels (not derived from groups).
    
    Args:
        cluster_labels: Subtype cluster assignments (e.g., "AC::cluster_01").
        groups: Coarse group labels (e.g., "AC", "DS-RGC", "ipRGC", "nonDS-RGC").
        df: Original DataFrame with axon_type, ds_p_value, iprgc_2hz_QI columns.
            If None, labels are derived from groups (which gives 100% by construction).
    
    Returns:
        Dict with purity results for each label.
    """
    if df is not None:
        # Use ORIGINAL cell-level labels from the data
        # axon_type: original column
        axon_type = df[config.AXON_COL].values
        
        # ds_cell: derived from p-value threshold (NOT from group assignment)
        ds_pval = df[config.DS_PVAL_COL].values
        ds_cell = np.array([1 if p < config.DS_P_THRESHOLD else 0 for p in ds_pval])
        
        # iprgc: derived from QI threshold (NOT from group assignment)
        iprgc_qi = df[config.IPRGC_QI_COL].fillna(0).values
        iprgc = np.array([1 if qi > config.IPRGC_QI_THRESHOLD else 0 for qi in iprgc_qi])
        
        logger.info("Using ORIGINAL cell labels for purity computation")
    else:
        # Fallback: derive binary labels from coarse groups (gives 100% by construction)
        logger.warning("No DataFrame provided - deriving labels from groups (will be 100%)")
        axon_type = np.array(['ac' if g == 'AC' else 'rgc' for g in groups])
        ds_cell = np.array([1 if g == 'DS-RGC' else 0 for g in groups])
        iprgc = np.array([1 if g == 'ipRGC' else 0 for g in groups])
    
    # Compute purity for each label (NOT including coarse_group - always 100% by construction)
    purity_axon = compute_purity(cluster_labels, axon_type)
    purity_ds = compute_purity(cluster_labels, ds_cell)
    purity_iprgc = compute_purity(cluster_labels, iprgc)
    
    # Cluster statistics
    n_clusters = len(np.unique(cluster_labels))
    n_cells = len(cluster_labels)
    
    # Clusters per group (for reporting only, not CV)
    clusters_per_group = {}
    for group in ['AC', 'DS-RGC', 'ipRGC', 'nonDS-RGC']:
        mask = groups == group
        clusters_per_group[group] = {
            'n_clusters': len(np.unique(cluster_labels[mask])),
            'n_cells': int(mask.sum()),
        }
    
    results = {
        'purity_by_label': {
            'axon_type': {
                'omitted_label': 'axon_type',
                'description': 'AC vs RGC',
                'purity': float(purity_axon),
            },
            'ds_cell': {
                'omitted_label': 'ds_cell', 
                'description': 'DS vs non-DS',
                'purity': float(purity_ds),
            },
            'iprgc': {
                'omitted_label': 'iprgc',
                'description': 'ipRGC vs non-ipRGC',
                'purity': float(purity_iprgc),
            },
            # NOTE: coarse_group excluded - always 100% since clustering is within groups
        },
        'cv_score': float(np.mean([purity_axon, purity_ds, purity_iprgc])),
        'n_clusters': n_clusters,
        'n_cells': n_cells,
        'clusters_per_group': clusters_per_group,
    }
    
    logger.info(f"CV Purity - axon_type: {purity_axon:.4f}, ds_cell: {purity_ds:.4f}, "
                f"iprgc: {purity_iprgc:.4f}")
    logger.info(f"CVScore (mean purity): {results['cv_score']:.4f}")
    
    return results


def save_cv_purity_results(
    cv_results: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Save cross-validation purity results to JSON.
    
    Args:
        cv_results: Dict from compute_cv_purity_posthoc.
        output_path: Output file path.
    
    Returns:
        Path to saved file.
    """
    output_path = output_path if output_path is not None else config.RESULTS_DIR / "cv_purity.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    logger.info(f"Saved CV purity results to {output_path}")
    
    return output_path
