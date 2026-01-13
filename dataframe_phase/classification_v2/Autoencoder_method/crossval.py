"""
Cross-validation using omitted-label purity.

Implements TRUE CV turns where one coarse label is omitted from:
1. Group assignment (for clustering)
2. Weak supervision (for AE training)

Then measures purity against the omitted label to test generalization.
"""

import json
import logging
from pathlib import Path
from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .evaluation import compute_purity

logger = logging.getLogger(__name__)


# CV Turn definitions
CV_TURNS = [
    {
        'name': 'omit_axon_type',
        'omit': 'axon_type',
        'active': ['ds_cell', 'iprgc'],
        'description': 'Omit AC/RGC split - test if clusters naturally separate axon types',
    },
    {
        'name': 'omit_ds_cell',
        'omit': 'ds_cell', 
        'active': ['axon_type', 'iprgc'],
        'description': 'Omit DS/non-DS split - test if clusters naturally separate DS cells',
    },
    {
        'name': 'omit_iprgc',
        'omit': 'iprgc',
        'active': ['axon_type', 'ds_cell'],
        'description': 'Omit ipRGC split - test if clusters naturally separate ipRGC cells',
    },
]


def create_cv_groups(
    df: pd.DataFrame,
    active_labels: list[str],
) -> np.ndarray:
    """
    Create group labels using only active labels (omitting one).
    
    Args:
        df: DataFrame with axon_type, ds_p_value, iprgc_2hz_QI columns.
        active_labels: Labels to use for grouping (e.g., ['ds_cell', 'iprgc']).
    
    Returns:
        Array of group labels (strings).
    """
    n = len(df)
    
    # Get individual label values
    axon = df[config.AXON_COL].values
    ds = (df[config.DS_PVAL_COL] < config.DS_P_THRESHOLD).values
    iprgc_qi = df[config.IPRGC_QI_COL].fillna(0).values
    iprgc = (iprgc_qi > config.IPRGC_QI_THRESHOLD)
    
    groups = []
    for i in range(n):
        parts = []
        
        if 'axon_type' in active_labels:
            parts.append(axon[i])  # 'ac' or 'rgc'
        
        if 'ds_cell' in active_labels:
            parts.append('DS' if ds[i] else 'nonDS')
        
        if 'iprgc' in active_labels:
            parts.append('ipRGC' if iprgc[i] else 'nonIP')
        
        groups.append('_'.join(parts))
    
    return np.array(groups)


def get_omitted_label_values(
    df: pd.DataFrame,
    omitted_label: str,
) -> np.ndarray:
    """
    Get binary values for the omitted label (for purity measurement).
    
    Args:
        df: DataFrame with label columns.
        omitted_label: Label that was omitted ('axon_type', 'ds_cell', or 'iprgc').
    
    Returns:
        Binary array (0/1) for the omitted label.
    """
    if omitted_label == 'axon_type':
        # 0=ac, 1=rgc
        return (df[config.AXON_COL] == 'rgc').astype(int).values
    
    elif omitted_label == 'ds_cell':
        # 0=non-DS, 1=DS
        return (df[config.DS_PVAL_COL] < config.DS_P_THRESHOLD).astype(int).values
    
    elif omitted_label == 'iprgc':
        # 0=non-ipRGC, 1=ipRGC
        iprgc_qi = df[config.IPRGC_QI_COL].fillna(0)
        return (iprgc_qi > config.IPRGC_QI_THRESHOLD).astype(int).values
    
    else:
        raise ValueError(f"Unknown omitted label: {omitted_label}")


def run_single_cv_turn(
    df: pd.DataFrame,
    segments: dict[str, np.ndarray],
    turn: dict,
    output_dir: Path,
    device: str = 'cuda',
) -> dict:
    """
    Run a single CV turn: train AE, cluster, measure purity.
    
    Args:
        df: Original DataFrame with all labels.
        segments: Preprocessed segments.
        turn: Turn definition with 'omit', 'active', 'name'.
        output_dir: Output directory for this turn.
        device: Torch device.
    
    Returns:
        Dict with turn results.
    """
    from .train import train_autoencoder
    from .embed import extract_embeddings, standardize_embeddings
    from .clustering import cluster_per_group
    from .grouping import encode_group_labels
    
    turn_name = turn['name']
    omitted = turn['omit']
    active = turn['active']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CV TURN: {turn_name}")
    logger.info(f"  Omitted: {omitted}")
    logger.info(f"  Active: {active}")
    logger.info(f"{'='*60}")
    
    start_time = datetime.now()
    
    # Create modified groups (without omitted label)
    cv_groups = create_cv_groups(df, active)
    unique_groups = np.unique(cv_groups)
    logger.info(f"  CV Groups: {list(unique_groups)}")
    
    # Encode groups for AE supervision
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    group_labels_encoded = np.array([group_to_id[g] for g in cv_groups])
    
    # Train AE with modified groups as supervision
    logger.info(f"  Training AE with {len(unique_groups)} groups...")
    model, train_history = train_autoencoder(
        segments=segments,
        group_labels=group_labels_encoded,
        device=device,
        checkpoint_dir=output_dir,
        epochs=config.AE_EPOCHS,
    )
    
    # Extract embeddings
    embeddings = extract_embeddings(model, segments, device=device)
    embeddings_std, _ = standardize_embeddings(embeddings)
    
    # Cluster within CV groups
    logger.info(f"  Clustering within CV groups...")
    cluster_ids, _, group_results = cluster_per_group(
        embeddings=embeddings_std,
        groups=cv_groups,
    )
    
    # Get omitted label values
    omitted_values = get_omitted_label_values(df, omitted)
    
    # Create cluster labels for purity
    cluster_labels = np.array([
        f"{g}::c{c}" for g, c in zip(cv_groups, cluster_ids)
    ])
    
    # Compute purity on OMITTED label
    purity = compute_purity(cluster_labels, omitted_values)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    result = {
        'turn_name': turn_name,
        'omitted_label': omitted,
        'active_labels': active,
        'purity': float(purity),
        'n_groups': len(unique_groups),
        'n_clusters': len(np.unique(cluster_labels)),
        'n_cells': len(df),
        'duration_seconds': duration,
        'description': turn['description'],
    }
    
    logger.info(f"  Purity({omitted}): {purity:.4f}")
    logger.info(f"  Duration: {duration:.1f}s")
    
    return result


def run_full_cv(
    df: pd.DataFrame,
    segments: dict[str, np.ndarray],
    output_dir: Path,
    device: str = 'cuda',
    turns: list[dict] | None = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Run full cross-validation with all turns.
    
    Args:
        df: Original DataFrame with all labels.
        segments: Preprocessed segments.
        output_dir: Base output directory.
        device: Torch device.
        turns: CV turn definitions. Defaults to CV_TURNS.
    
    Returns:
        (results_df, cv_score)
    """
    turns = turns if turns is not None else CV_TURNS
    
    cv_dir = output_dir / "cv_results"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("RUNNING FULL CROSS-VALIDATION")
    logger.info(f"  Turns: {len(turns)}")
    logger.info("="*60)
    
    start_time = datetime.now()
    results = []
    
    for turn in turns:
        turn_dir = cv_dir / turn['name']
        turn_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_single_cv_turn(
            df=df,
            segments=segments,
            turn=turn,
            output_dir=turn_dir,
            device=device,
        )
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute CV score (mean purity across turns)
    cv_score = results_df['purity'].mean()
    
    total_duration = (datetime.now() - start_time).total_seconds()
    
    # Save results
    results_df.to_csv(cv_dir / "cv_turns.csv", index=False)
    
    summary = {
        'cv_score': float(cv_score),
        'turns': results,
        'total_duration_seconds': total_duration,
    }
    with open(cv_dir / "cv_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION COMPLETE")
    logger.info("="*60)
    logger.info(f"  CVScore: {cv_score:.4f}")
    for r in results:
        logger.info(f"    {r['omitted_label']}: {r['purity']:.4f}")
    logger.info(f"  Total time: {total_duration/60:.1f} minutes")
    logger.info(f"  Results: {cv_dir}")
    
    return results_df, cv_score


def compute_cvscore(purity_table: pd.DataFrame) -> float:
    """
    Compute aggregate CV score from purity table.
    
    CVScore = mean(purity across all omitted labels)
    """
    return purity_table['purity'].mean()
