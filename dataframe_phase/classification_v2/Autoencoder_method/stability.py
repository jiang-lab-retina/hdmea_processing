"""
Bootstrap stability testing for cluster robustness.

Implements Baden-style 90% subsampling bootstrap to assess
cluster stability via correlation of cluster means.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.stats import pearsonr

from . import config
from .clustering import cluster_per_group, get_cluster_means

logger = logging.getLogger(__name__)


def match_clusters_by_correlation(
    reference_means: dict[int, np.ndarray],
    bootstrap_means: dict[int, np.ndarray],
) -> dict[int, Tuple[int, float]]:
    """
    Match bootstrap clusters to reference clusters by correlation.
    
    For each bootstrap cluster, find the reference cluster with
    highest correlation of mean embeddings.
    
    Args:
        reference_means: Dict mapping reference cluster_id to mean embedding.
        bootstrap_means: Dict mapping bootstrap cluster_id to mean embedding.
    
    Returns:
        Dict mapping bootstrap_cluster_id to (matched_ref_id, correlation)
    """
    matches = {}
    
    ref_ids = list(reference_means.keys())
    ref_matrix = np.array([reference_means[k] for k in ref_ids])
    
    for boot_id, boot_mean in bootstrap_means.items():
        # Compute correlation with all reference clusters
        correlations = []
        for ref_mean in ref_matrix:
            if len(ref_mean) > 1:
                corr, _ = pearsonr(boot_mean, ref_mean)
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(1.0 if boot_mean[0] == ref_mean[0] else 0.0)
        
        # Find best match
        best_idx = np.argmax(correlations)
        matches[boot_id] = (ref_ids[best_idx], correlations[best_idx])
    
    return matches


def run_bootstrap_stability(
    embeddings: np.ndarray,
    groups: np.ndarray,
    n_iterations: int | None = None,
    sample_fraction: float | None = None,
    random_seed: int | None = None,
    k_max_per_group: dict | None = None,
) -> Tuple[dict, list[dict]]:
    """
    Run bootstrap stability testing.
    
    For each group, repeatedly subsample and recluster, measuring
    correlation between bootstrap and reference cluster means.
    
    Args:
        embeddings: (n_cells, dim) embeddings.
        groups: (n_cells,) coarse group labels.
        n_iterations: Number of bootstrap iterations.
        sample_fraction: Fraction of data to sample (e.g., 0.9).
        random_seed: Random seed.
        k_max_per_group: Maximum k per group for clustering.
    
    Returns:
        (stability_summary, bootstrap_details)
        stability_summary: Dict per group with mean±std of median correlations
        bootstrap_details: List of per-iteration results
    """
    n_iterations = n_iterations if n_iterations is not None else config.BOOTSTRAP_N_ITERATIONS
    sample_fraction = sample_fraction if sample_fraction is not None else config.BOOTSTRAP_SAMPLE_FRACTION
    random_seed = random_seed if random_seed is not None else config.BOOTSTRAP_RANDOM_SEED
    
    np.random.seed(random_seed)
    
    unique_groups = np.unique(groups)
    stability_summary = {}
    bootstrap_details = []
    
    logger.info(f"Running {n_iterations} bootstrap iterations (sample={sample_fraction:.0%})")
    
    for group in unique_groups:
        mask = groups == group
        group_embeddings = embeddings[mask]
        n_group = mask.sum()
        sample_size = int(n_group * sample_fraction)
        
        if n_group < config.MIN_CELLS_PER_GROUP:
            logger.warning(f"Group {group} too small for stability testing")
            stability_summary[group] = {
                'mean_median_corr': np.nan,
                'std_median_corr': np.nan,
                'stable': False,
            }
            continue
        
        logger.info(f"  Group {group}: {n_group} cells, sample_size={sample_size}")
        
        # Fit reference model on full data
        group_groups = np.array([group] * n_group)  # Dummy groups for single-group clustering
        ref_clusters, _, ref_results = cluster_per_group(
            group_embeddings, 
            group_groups,
            k_max_per_group=k_max_per_group,
        )
        ref_means = get_cluster_means(group_embeddings, ref_clusters)
        ref_k = len(np.unique(ref_clusters))
        
        # Bootstrap iterations
        median_correlations = []
        
        for b in range(n_iterations):
            # Subsample
            indices = np.random.choice(n_group, sample_size, replace=False)
            boot_embeddings = group_embeddings[indices]
            boot_groups = group_groups[indices]
            
            # Cluster subsample with same k
            try:
                boot_clusters, _, _ = cluster_per_group(
                    boot_embeddings,
                    boot_groups,
                    k_max_per_group={group: ref_k},
                )
            except Exception as e:
                logger.warning(f"    Bootstrap {b+1} failed: {e}")
                continue
            
            boot_means = get_cluster_means(boot_embeddings, boot_clusters)
            
            # Match clusters
            matches = match_clusters_by_correlation(ref_means, boot_means)
            
            # Compute median correlation
            correlations = [corr for _, (_, corr) in matches.items()]
            if correlations:
                median_corr = np.median(correlations)
                median_correlations.append(median_corr)
                
                bootstrap_details.append({
                    'group': group,
                    'iteration': b + 1,
                    'median_correlation': median_corr,
                    'n_clusters': len(boot_means),
                })
        
        # Summarize for this group
        if median_correlations:
            mean_corr = np.mean(median_correlations)
            std_corr = np.std(median_correlations)
            is_stable = mean_corr >= config.STABILITY_THRESHOLD
            
            stability_summary[group] = {
                'mean_median_corr': mean_corr,
                'std_median_corr': std_corr,
                'stable': is_stable,
                'n_iterations': len(median_correlations),
                'reference_k': ref_k,
            }
            
            logger.info(f"    Stability: {mean_corr:.3f}±{std_corr:.3f} "
                       f"({'STABLE' if is_stable else 'UNSTABLE'})")
        else:
            stability_summary[group] = {
                'mean_median_corr': np.nan,
                'std_median_corr': np.nan,
                'stable': False,
            }
    
    return stability_summary, bootstrap_details


def summarize_stability(
    stability_summary: dict,
    threshold: float | None = None,
) -> dict:
    """
    Compute overall stability summary.
    
    Args:
        stability_summary: Per-group stability results.
        threshold: Stability threshold.
    
    Returns:
        Overall summary dict.
    """
    threshold = threshold if threshold is not None else config.STABILITY_THRESHOLD
    
    mean_corrs = [
        s['mean_median_corr'] 
        for s in stability_summary.values() 
        if not np.isnan(s['mean_median_corr'])
    ]
    
    n_stable = sum(1 for s in stability_summary.values() if s.get('stable', False))
    n_groups = len(stability_summary)
    
    summary = {
        'overall_mean_correlation': np.mean(mean_corrs) if mean_corrs else np.nan,
        'overall_std_correlation': np.std(mean_corrs) if mean_corrs else np.nan,
        'n_stable_groups': n_stable,
        'n_total_groups': n_groups,
        'all_stable': n_stable == n_groups,
        'threshold': threshold,
    }
    
    logger.info(f"Overall stability: {summary['overall_mean_correlation']:.3f}±{summary['overall_std_correlation']:.3f}, "
               f"{n_stable}/{n_groups} groups stable")
    
    return summary


def save_stability_results(
    stability_summary: dict,
    overall_summary: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Save stability results to JSON.
    
    Args:
        stability_summary: Per-group results.
        overall_summary: Overall summary.
        output_path: Output file path.
    
    Returns:
        Path to saved file.
    """
    output_path = output_path if output_path is not None else config.RESULTS_DIR / "stability_metrics.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    results = {
        'per_group': convert_types(stability_summary),
        'overall': convert_types(overall_summary),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved stability results to {output_path}")
    return output_path
