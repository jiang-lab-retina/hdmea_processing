"""
Evaluation and output saving for the DEC-refined RGC clustering pipeline.

Handles saving:
- Embeddings (initial and DEC-refined)
- Cluster assignments (GMM and DEC)
- BIC curves
- ipRGC validation metrics
- Comparison tables
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def save_embeddings(
    embeddings: np.ndarray,
    cell_ids: np.ndarray,
    group: str,
    embedding_type: str,
    output_dir: Path,
) -> Path:
    """
    Save embeddings to parquet file.
    
    Args:
        embeddings: (n_cells, 49) embedding array.
        cell_ids: (n_cells,) cell identifiers.
        group: Group name (DSGC, OSGC, Other).
        embedding_type: "initial" or "dec_refined".
        output_dir: Output directory.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    n_cells, n_dims = embeddings.shape
    
    data = {
        'cell_id': cell_ids,
        'group': [group] * n_cells,
        'embedding_type': [embedding_type] * n_cells,
    }
    
    # Add embedding dimensions
    for i in range(n_dims):
        data[f'z_{i}'] = embeddings[:, i]
    
    df = pd.DataFrame(data)
    
    output_path = output_dir / f"embeddings_{embedding_type}.parquet"
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {embedding_type} embeddings to {output_path}")
    return output_path


def save_cluster_assignments(
    cell_ids: np.ndarray,
    group: str,
    gmm_labels: np.ndarray,
    gmm_posteriors: np.ndarray,
    dec_labels: np.ndarray,
    dec_soft_assignments: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    Save cluster assignments to parquet file.
    
    Args:
        cell_ids: (n_cells,) cell identifiers.
        group: Group name.
        gmm_labels: (n_cells,) GMM cluster labels.
        gmm_posteriors: (n_cells,) max GMM posterior per cell.
        dec_labels: (n_cells,) DEC cluster labels.
        dec_soft_assignments: (n_cells,) max DEC soft assignment per cell.
        output_dir: Output directory.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'cell_id': cell_ids,
        'group': group,
        'gmm_cluster': gmm_labels,
        'gmm_posterior': gmm_posteriors,
        'dec_cluster': dec_labels,
        'dec_soft_max': dec_soft_assignments,
    })
    
    output_path = output_dir / "cluster_assignments.parquet"
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved cluster assignments to {output_path}")
    return output_path


def save_k_selection(
    k_range: list,
    bic_values: np.ndarray,
    k_selected: int,
    group: str,
    output_dir: Path,
) -> Path:
    """
    Save k-selection results to JSON.
    
    Args:
        k_range: List of k values tried.
        bic_values: BIC values for each k.
        k_selected: Selected k*.
        group: Group name.
        output_dir: Output directory.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        'group': group,
        'k_range': list(k_range),
        'bic_values': bic_values.tolist(),
        'k_selected': int(k_selected),
        'selection_method': 'min_bic',
    }
    
    output_path = output_dir / "k_selection.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved k-selection to {output_path}")
    return output_path


def save_iprgc_validation(
    gmm_metrics: dict,
    dec_metrics: dict,
    group: str,
    output_dir: Path,
) -> Path:
    """
    Save ipRGC validation metrics to JSON.
    
    Args:
        gmm_metrics: Metrics for initial GMM clustering.
        dec_metrics: Metrics for DEC-refined clustering.
        group: Group name.
        output_dir: Output directory.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        'group': group,
        'baseline_prevalence': gmm_metrics.get('baseline_prevalence', 0),
        'initial_gmm': {
            'purity': gmm_metrics['purity'],
            'top_enriched': gmm_metrics['top_enriched'],
        },
        'dec_refined': {
            'purity': dec_metrics['purity'],
            'top_enriched': dec_metrics['top_enriched'],
        },
    }
    
    output_path = output_dir / "iprgc_validation.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved ipRGC validation to {output_path}")
    return output_path


def save_comparison_table(
    gmm_metrics: dict,
    dec_metrics: dict,
    group: str,
    k_selected: int,
    output_dir: Path,
) -> Path:
    """
    Save GMM vs DEC comparison table.
    
    Args:
        gmm_metrics: Metrics for initial GMM clustering.
        dec_metrics: Metrics for DEC-refined clustering.
        group: Group name.
        k_selected: Number of clusters.
        output_dir: Output directory.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        'group': group,
        'k': k_selected,
        'gmm_purity': gmm_metrics['purity'],
        'dec_purity': dec_metrics['purity'],
        'purity_improvement': dec_metrics['purity'] - gmm_metrics['purity'],
        'gmm_top_enrichment': gmm_metrics['top_enriched'][0]['enrichment'] if gmm_metrics['top_enriched'] else 0,
        'dec_top_enrichment': dec_metrics['top_enriched'][0]['enrichment'] if dec_metrics['top_enriched'] else 0,
    }
    
    output_path = output_dir / "comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Saved comparison table to {output_path}")
    return output_path


def generate_consolidated_report(
    results_by_group: Dict[str, dict],
    output_dir: Path,
) -> Path:
    """
    Generate consolidated report across all groups.
    
    Args:
        results_by_group: Dict mapping group name to results dict.
        output_dir: Output directory.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for group, results in results_by_group.items():
        rows.append({
            'group': group,
            'n_cells': results.get('n_cells', 0),
            'k_selected': results.get('k_selected', 0),
            'gmm_purity': results.get('gmm_metrics', {}).get('purity', 0),
            'dec_purity': results.get('dec_metrics', {}).get('purity', 0),
            'baseline_iprgc': results.get('gmm_metrics', {}).get('baseline_prevalence', 0),
        })
    
    df = pd.DataFrame(rows)
    
    output_path = output_dir / "consolidated_report.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved consolidated report to {output_path}")
    return output_path


def load_artifacts_for_visualization(
    results_dir: Path,
    group: str,
) -> dict:
    """
    Load saved artifacts for visualization.
    
    Args:
        results_dir: Directory containing saved results.
        group: Group name.
    
    Returns:
        Dict with loaded artifacts.
    """
    group_dir = results_dir / group
    
    artifacts = {}
    
    # Load embeddings
    for emb_type in ['initial', 'dec_refined']:
        emb_path = group_dir / f"embeddings_{emb_type}.parquet"
        if emb_path.exists():
            artifacts[f'embeddings_{emb_type}'] = pd.read_parquet(emb_path)
    
    # Load cluster assignments
    assign_path = group_dir / "cluster_assignments.parquet"
    if assign_path.exists():
        artifacts['assignments'] = pd.read_parquet(assign_path)
    
    # Load k-selection
    k_path = group_dir / "k_selection.json"
    if k_path.exists():
        with open(k_path, 'r') as f:
            artifacts['k_selection'] = json.load(f)
    
    # Load ipRGC validation
    val_path = group_dir / "iprgc_validation.json"
    if val_path.exists():
        with open(val_path, 'r') as f:
            artifacts['iprgc_validation'] = json.load(f)
    
    logger.info(f"Loaded {len(artifacts)} artifacts for group {group}")
    
    return artifacts
