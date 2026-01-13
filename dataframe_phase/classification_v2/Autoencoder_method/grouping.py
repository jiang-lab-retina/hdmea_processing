"""
Coarse group assignment for the Autoencoder-based RGC clustering pipeline.

This module handles:
- Disjoint group assignment using configurable precedence
- Group label thresholds (ipRGC, DS, AC)
- Group mask utilities
"""

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def assign_coarse_groups(
    df: pd.DataFrame,
    precedence: list[str] | None = None,
    iprgc_threshold: float | None = None,
    ds_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Assign disjoint coarse group labels to cells.
    
    Args:
        df: DataFrame with axon_type, ds_p_value, iprgc_2hz_QI columns.
        precedence: Order of group precedence (first match wins).
            Options: ["ac", "iprgc", "ds", "nonds"]
            Defaults to config.GROUP_PRECEDENCE.
        iprgc_threshold: iprgc_2hz_QI threshold for ipRGC group.
            Defaults to config.IPRGC_QI_THRESHOLD.
        ds_threshold: ds_p_value threshold for DS group.
            Defaults to config.DS_P_THRESHOLD.
    
    Returns:
        DataFrame with added 'coarse_group' column.
        Values: "AC", "ipRGC", "DS-RGC", "nonDS-RGC"
    
    Notes:
        - NaN in iprgc_2hz_QI treated as "not ipRGC"
        - Each cell assigned to exactly one group
    """
    precedence = precedence if precedence is not None else config.GROUP_PRECEDENCE
    iprgc_threshold = iprgc_threshold if iprgc_threshold is not None else config.IPRGC_QI_THRESHOLD
    ds_threshold = ds_threshold if ds_threshold is not None else config.DS_P_THRESHOLD
    
    logger.info(f"Assigning coarse groups with precedence: {precedence}")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Initialize all cells as nonDS-RGC (default)
    df["coarse_group"] = config.GROUP_NAMES["nonds"]
    
    # Apply precedence in reverse order (later rules override earlier)
    # This way the first in precedence list wins
    for group_type in reversed(precedence):
        if group_type == "nonds":
            # Already set as default
            continue
        
        elif group_type == "ds":
            # DS-RGC: ds_p_value < threshold AND not AC
            ds_mask = (
                (df[config.DS_PVAL_COL] < ds_threshold) & 
                (df[config.AXON_COL] != "ac")
            )
            df.loc[ds_mask, "coarse_group"] = config.GROUP_NAMES["ds"]
            logger.debug(f"  DS-RGC: {ds_mask.sum()} cells")
        
        elif group_type == "iprgc":
            # ipRGC: iprgc_2hz_QI > threshold (NaN treated as not ipRGC)
            iprgc_qi = df[config.IPRGC_QI_COL].fillna(0)  # NaN -> 0 (not ipRGC)
            iprgc_mask = (
                (iprgc_qi > iprgc_threshold) & 
                (df[config.AXON_COL] != "ac")  # ipRGCs are RGCs
            )
            df.loc[iprgc_mask, "coarse_group"] = config.GROUP_NAMES["iprgc"]
            logger.debug(f"  ipRGC: {iprgc_mask.sum()} cells")
        
        elif group_type == "ac":
            # AC: axon_type == "ac"
            ac_mask = df[config.AXON_COL] == "ac"
            df.loc[ac_mask, "coarse_group"] = config.GROUP_NAMES["ac"]
            logger.debug(f"  AC: {ac_mask.sum()} cells")
    
    # Log group distribution
    group_counts = df["coarse_group"].value_counts()
    logger.info(f"Group distribution:\n{group_counts.to_string()}")
    
    return df


def get_group_mask(
    df: pd.DataFrame,
    group: str,
) -> np.ndarray:
    """
    Get boolean mask for cells in a specific group.
    
    Args:
        df: DataFrame with coarse_group column.
        group: Group name ("AC", "ipRGC", "DS-RGC", "nonDS-RGC")
    
    Returns:
        Boolean array of shape (n_cells,)
    """
    if "coarse_group" not in df.columns:
        raise ValueError("DataFrame must have 'coarse_group' column. Run assign_coarse_groups first.")
    
    return (df["coarse_group"] == group).values


def get_unique_groups(df: pd.DataFrame) -> list[str]:
    """
    Get list of unique groups present in the DataFrame.
    
    Args:
        df: DataFrame with coarse_group column.
    
    Returns:
        List of unique group names.
    """
    if "coarse_group" not in df.columns:
        raise ValueError("DataFrame must have 'coarse_group' column. Run assign_coarse_groups first.")
    
    return df["coarse_group"].unique().tolist()


def get_group_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Get indices of cells in each group.
    
    Args:
        df: DataFrame with coarse_group column.
    
    Returns:
        Dict mapping group name to array of indices.
    """
    if "coarse_group" not in df.columns:
        raise ValueError("DataFrame must have 'coarse_group' column. Run assign_coarse_groups first.")
    
    result = {}
    for group in df["coarse_group"].unique():
        mask = df["coarse_group"] == group
        result[group] = np.where(mask)[0]
    
    return result


def encode_group_labels(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """
    Encode group labels as integers for use in loss functions.
    
    Args:
        df: DataFrame with coarse_group column.
    
    Returns:
        (encoded_labels, label_map)
        - encoded_labels: (n_cells,) integer array
        - label_map: Dict mapping group name to integer
    """
    if "coarse_group" not in df.columns:
        raise ValueError("DataFrame must have 'coarse_group' column. Run assign_coarse_groups first.")
    
    groups = df["coarse_group"].unique()
    label_map = {group: i for i, group in enumerate(sorted(groups))}
    
    encoded = df["coarse_group"].map(label_map).values
    
    return encoded, label_map


def filter_groups_by_size(
    df: pd.DataFrame,
    min_size: int | None = None,
) -> pd.DataFrame:
    """
    Filter out groups with fewer than min_size cells.
    
    Args:
        df: DataFrame with coarse_group column.
        min_size: Minimum cells per group. Defaults to config.MIN_CELLS_PER_GROUP.
    
    Returns:
        Filtered DataFrame.
    """
    min_size = min_size if min_size is not None else config.MIN_CELLS_PER_GROUP
    
    group_counts = df["coarse_group"].value_counts()
    valid_groups = group_counts[group_counts >= min_size].index
    
    removed_groups = set(df["coarse_group"].unique()) - set(valid_groups)
    if removed_groups:
        logger.warning(f"Removing groups with < {min_size} cells: {removed_groups}")
    
    return df[df["coarse_group"].isin(valid_groups)]
