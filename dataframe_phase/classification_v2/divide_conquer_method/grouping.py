"""
Group assignment for RGC cells using ipRGC > DS > OS priority rule.

This module assigns RGC cells to functional groups:
- ipRGC: Intrinsically photosensitive (iprgc_2hz_QI > 0.8)
- DSGC: Direction-Selective (ds_p_value < 0.05, NOT ipRGC)
- OSGC: Orientation-Selective (os_p_value < 0.05, NOT ipRGC/DSGC)
- Other: Remaining RGCs
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def assign_groups(
    df: pd.DataFrame,
    ds_threshold: float | None = None,
    os_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Assign cells to groups using ipRGC > DS > OS priority rule.
    
    Priority order: ipRGC > DS > OS
    - If iprgc_2hz_QI > threshold: ipRGC
    - Else if ds_p_value < threshold: DSGC
    - Else if os_p_value < threshold: OSGC
    - Else: Other
    
    Args:
        df: DataFrame with ds_p_value, os_p_value, and iprgc_2hz_QI columns.
        ds_threshold: DS classification threshold. Defaults to config.DS_P_THRESHOLD.
        os_threshold: OS classification threshold. Defaults to config.OS_P_THRESHOLD.
    
    Returns:
        DataFrame with added columns:
            - 'group': str ("ipRGC", "DSGC", "OSGC", "Other")
            - 'is_ds': bool
            - 'is_os': bool
            - 'is_iprgc': bool
    
    Logs:
        - Overlap counts and group sizes
    """
    ds_threshold = ds_threshold if ds_threshold is not None else config.DS_P_THRESHOLD
    os_threshold = os_threshold if os_threshold is not None else config.OS_P_THRESHOLD
    iprgc_qi_threshold = config.IPRGC_QI_THRESHOLD
    
    df = df.copy()
    
    # Compute boolean masks
    is_ds = df[config.DS_PVAL_COL] < ds_threshold
    is_os = df[config.OS_PVAL_COL] < os_threshold
    is_iprgc = df[config.IPRGC_QI_COL].fillna(0) > iprgc_qi_threshold
    
    # Add boolean columns
    df['is_ds'] = is_ds
    df['is_os'] = is_os
    df['is_iprgc'] = is_iprgc
    
    # Track overlaps
    overlap_ds_os = (is_ds & is_os).sum()
    n_iprgc_total = is_iprgc.sum()
    n_iprgc_also_ds = (is_iprgc & is_ds).sum()
    n_iprgc_also_os = (is_iprgc & is_os).sum()
    
    # Apply priority rule: ipRGC > DS > OS > Other
    # Start with "Other" and assign in reverse priority order
    # (highest priority assigned last so it overwrites)
    group = pd.Series("Other", index=df.index)
    group[is_os] = "OSGC"     # OS cells (may be overwritten by DS or ipRGC)
    group[is_ds] = "DSGC"     # DS cells (may be overwritten by ipRGC)
    group[is_iprgc] = "ipRGC" # ipRGC cells (highest priority)
    
    df['group'] = group
    
    # Log diagnostics
    logger.info(f"Group assignment (ipRGC > DS > OS priority):")
    logger.info(f"  ipRGC QI threshold: {iprgc_qi_threshold}")
    logger.info(f"  DS threshold: {ds_threshold}")
    logger.info(f"  OS threshold: {os_threshold}")
    logger.info(f"  Overlap (both DS and OS): {overlap_ds_os} cells")
    logger.info(f"  ipRGC cells: {n_iprgc_total} total")
    logger.info(f"    Also DS-selective: {n_iprgc_also_ds}")
    logger.info(f"    Also OS-selective: {n_iprgc_also_os}")
    
    # Log group counts
    group_counts = df['group'].value_counts()
    for group_name in config.GROUP_NAMES:
        count = group_counts.get(group_name, 0)
        pct = 100 * count / len(df)
        logger.info(f"  {group_name}: {count} cells ({pct:.1f}%)")
    
    return df


def filter_group(
    df: pd.DataFrame,
    group: str,
    min_size: int | None = None,
) -> pd.DataFrame:
    """
    Filter DataFrame to a single group.
    
    Args:
        df: DataFrame with 'group' column.
        group: Group name ("DSGC", "OSGC", "Other").
        min_size: Minimum group size. Defaults to config.MIN_GROUP_SIZE.
    
    Returns:
        Filtered DataFrame for the specified group.
    
    Raises:
        ValueError: If group is too small.
    """
    min_size = min_size if min_size is not None else config.MIN_GROUP_SIZE
    
    if 'group' not in df.columns:
        raise ValueError("DataFrame must have 'group' column. Call assign_groups() first.")
    
    group_df = df[df['group'] == group].copy()
    
    if len(group_df) < min_size:
        raise ValueError(
            f"Group {group} has only {len(group_df)} cells, "
            f"minimum required is {min_size}"
        )
    
    logger.info(f"Filtered to group {group}: {len(group_df)} cells")
    
    return group_df


def get_group_stats(df: pd.DataFrame) -> dict:
    """
    Get statistics for all groups.
    
    Args:
        df: DataFrame with 'group' column.
    
    Returns:
        Dict with group statistics:
            - counts: Dict[group_name, count]
            - overlap: Number of cells meeting both DS and OS thresholds
            - total: Total number of cells
    """
    if 'group' not in df.columns:
        raise ValueError("DataFrame must have 'group' column. Call assign_groups() first.")
    
    counts = df['group'].value_counts().to_dict()
    
    # Compute overlap if boolean columns exist
    if 'is_ds' in df.columns and 'is_os' in df.columns:
        overlap = (df['is_ds'] & df['is_os']).sum()
    else:
        overlap = 0
    
    return {
        'counts': counts,
        'overlap': overlap,
        'total': len(df),
    }
