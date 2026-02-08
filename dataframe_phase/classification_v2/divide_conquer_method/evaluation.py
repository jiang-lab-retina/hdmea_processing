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


def save_labeled_dataframe(
    df: pd.DataFrame,
    groups: list,
    results_dir: Path,
    mosaic_results: Optional[pd.DataFrame],
    output_path: Path,
) -> Path:
    """
    Merge cluster labels and mosaic validation back into the original DataFrame.

    Adds columns:
        - subtype: e.g. "DSGC_3" or "DSGC_3_invalid"
        - valid_mosaic: bool (True if subtype passes mosaic validation)

    Uses two strategies to match cell_ids back to the original DataFrame:
        1. Direct index match (when cell_id is the original string index).
        2. Positional alignment (when cell_id is an integer from a
           reset_index run): re-filters df by group and aligns rows
           by position.

    Args:
        df: Original filtered DataFrame with 'group' column.
        groups: List of group names that were processed.
        results_dir: Directory containing per-group cluster_assignments.parquet.
        mosaic_results: DataFrame from mosaic validation (group, dec_cluster,
            mosaic_validation columns), or None if mosaic was skipped.
        output_path: Where to save the labeled parquet file.

    Returns:
        Path to saved file.
    """
    # Build a lookup: (group, cluster_id) -> mosaic_valid (bool)
    mosaic_lookup = {}
    if mosaic_results is not None:
        for _, row in mosaic_results.iterrows():
            key = (row['group'], int(row['dec_cluster']))
            mosaic_lookup[key] = bool(row['mosaic_validation'])

    # Initialize new columns
    df = df.copy()
    df['subtype'] = ''
    df['valid_mosaic'] = False

    for group_name in groups:
        cluster_path = results_dir / group_name / "cluster_assignments.parquet"
        if not cluster_path.exists():
            logger.warning(f"Cluster file not found: {cluster_path}")
            continue

        cluster_df = pd.read_parquet(cluster_path)

        # Check whether cell_id matches df.index (string) or is positional (int)
        sample_id = cluster_df['cell_id'].iloc[0]
        use_positional = sample_id not in df.index

        if use_positional:
            # Positional fallback: re-filter df to this group in the same order
            group_mask = df['group'] == group_name
            group_indices = df.index[group_mask]

            if len(group_indices) != len(cluster_df):
                logger.warning(
                    f"  {group_name}: positional mismatch "
                    f"(df has {len(group_indices)} cells, "
                    f"cluster_assignments has {len(cluster_df)}). Skipping."
                )
                continue

            for pos, (_, row) in enumerate(cluster_df.iterrows()):
                orig_idx = group_indices[pos]
                cluster_id = int(row['dec_cluster'])

                is_valid = mosaic_lookup.get((group_name, cluster_id), False)
                suffix = '' if is_valid else '_invalid'
                df.at[orig_idx, 'subtype'] = f"{group_name}_{cluster_id}{suffix}"
                df.at[orig_idx, 'valid_mosaic'] = is_valid
        else:
            # Direct index match
            for _, row in cluster_df.iterrows():
                orig_idx = row['cell_id']
                cluster_id = int(row['dec_cluster'])

                is_valid = mosaic_lookup.get((group_name, cluster_id), False)
                suffix = '' if is_valid else '_invalid'

                if orig_idx in df.index:
                    df.at[orig_idx, 'subtype'] = f"{group_name}_{cluster_id}{suffix}"
                    df.at[orig_idx, 'valid_mosaic'] = is_valid

    # Summary
    n_labeled = (df['subtype'] != '').sum()
    n_valid = df['valid_mosaic'].sum()
    logger.info(f"Labeled {n_labeled}/{len(df)} cells, {n_valid} in valid subtypes")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=True)
    logger.info(f"Saved labeled DataFrame to {output_path}")

    return output_path
