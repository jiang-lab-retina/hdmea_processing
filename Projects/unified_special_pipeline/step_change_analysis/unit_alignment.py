"""
Unit Alignment for Step Change Analysis Pipeline

This module tracks units across multiple recordings by matching
waveform shapes, response signatures, and electrode coordinates.

Ported from: Legacy_code/.../low_glucose/A02_pkl_aligment.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from .specific_config import (
    AlignmentConfig,
    QualityConfig,
    PipelineConfig,
    default_config,
    get_grouped_hdf5_path,
)
from .data_loader import (
    load_recording_from_hdf5,
    get_high_quality_units,
    calculate_quality_index,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Signature Extraction
# =============================================================================

def extract_unit_signature(unit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract alignment signature from unit data.
    
    Args:
        unit_data: Unit data dictionary
    
    Returns:
        Dictionary with response_signature, waveform_signature, coordinate
    """
    signature = {}
    
    # Response signature (mean step response)
    if "response_signature" in unit_data:
        signature["response_signature"] = np.array(unit_data["response_signature"])
    elif "step_responses" in unit_data:
        responses = np.array(unit_data["step_responses"])
        if responses.size > 0:
            signature["response_signature"] = responses.mean(axis=0)
        else:
            signature["response_signature"] = np.array([])
    else:
        signature["response_signature"] = np.array([])
    
    # Waveform signature
    if "waveform" in unit_data:
        signature["waveform_signature"] = np.array(unit_data["waveform"])
    else:
        signature["waveform_signature"] = np.array([])
    
    # Electrode coordinate
    row = unit_data.get("row", 0)
    col = unit_data.get("col", 0)
    signature["coordinate"] = [row, col]
    
    # Quality index
    signature["quality_index"] = unit_data.get("quality_index", 0.0)
    
    return signature


def add_signatures_to_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add alignment signatures to all units in data.
    
    Args:
        data: Recording data dictionary
    
    Returns:
        Data with signatures added to each unit
    """
    for unit_id, unit_data in data.get("units", {}).items():
        signature = extract_unit_signature(unit_data)
        unit_data["alignment"] = signature
    
    return data


# =============================================================================
# Unit Matching
# =============================================================================

def compute_alignment_score(
    ref_sig: Dict[str, Any],
    align_sig: Dict[str, Any],
    waveform_weight: float = 10.0,
) -> float:
    """
    Compute alignment score between two unit signatures.
    
    Lower score = better match.
    
    Args:
        ref_sig: Reference unit signature
        align_sig: Candidate unit signature
        waveform_weight: Weight for waveform difference
    
    Returns:
        Combined alignment score
    """
    # Response difference
    ref_resp = ref_sig.get("response_signature", np.array([]))
    align_resp = align_sig.get("response_signature", np.array([]))
    
    if len(ref_resp) > 0 and len(align_resp) > 0:
        # Ensure same length
        min_len = min(len(ref_resp), len(align_resp))
        response_diff = np.linalg.norm(ref_resp[:min_len] - align_resp[:min_len])
    else:
        response_diff = np.inf
    
    # Waveform difference
    ref_wave = ref_sig.get("waveform_signature", np.array([]))
    align_wave = align_sig.get("waveform_signature", np.array([]))
    
    if len(ref_wave) > 0 and len(align_wave) > 0:
        min_len = min(len(ref_wave), len(align_wave))
        waveform_diff = np.linalg.norm(ref_wave[:min_len] - align_wave[:min_len])
    else:
        waveform_diff = np.inf
    
    # Combined score
    total_score = response_diff + waveform_diff * waveform_weight
    
    return total_score


def compute_coordinate_distance(
    ref_sig: Dict[str, Any],
    align_sig: Dict[str, Any],
) -> float:
    """
    Compute electrode coordinate distance between units.
    
    Args:
        ref_sig: Reference unit signature
        align_sig: Candidate unit signature
    
    Returns:
        Squared Euclidean distance between coordinates
    """
    ref_coord = np.array(ref_sig.get("coordinate", [0, 0]))
    align_coord = np.array(align_sig.get("coordinate", [0, 0]))
    
    return float(np.sum((ref_coord - align_coord) ** 2))


def get_unit_pair_list(
    ref_data: Dict[str, Any],
    align_data: Dict[str, Any],
    ref_units: List[str],
    align_units: List[str],
    waveform_weight: float = 10.0,
    distance_threshold: float = 0.0,
) -> List[Tuple[str, str]]:
    """
    Find matching unit pairs between two recordings.
    
    Args:
        ref_data: Reference recording data
        align_data: Recording to align
        ref_units: List of reference unit IDs to consider
        align_units: List of candidate unit IDs to consider
        waveform_weight: Weight for waveform similarity
        distance_threshold: Maximum coordinate distance squared
    
    Returns:
        List of (ref_unit_id, align_unit_id) pairs
    """
    final_pairs = []
    
    for ref_id in ref_units:
        ref_unit = ref_data["units"].get(ref_id, {})
        ref_sig = ref_unit.get("alignment", extract_unit_signature(ref_unit))
        
        # Find candidates within distance threshold
        candidates = []
        scores = []
        
        for align_id in align_units:
            align_unit = align_data["units"].get(align_id, {})
            align_sig = align_unit.get("alignment", extract_unit_signature(align_unit))
            
            # Check coordinate distance
            coord_dist = compute_coordinate_distance(ref_sig, align_sig)
            
            if coord_dist <= distance_threshold:
                score = compute_alignment_score(ref_sig, align_sig, waveform_weight)
                candidates.append(align_id)
                scores.append(score)
        
        # Select best match
        if len(candidates) == 1:
            final_pairs.append((ref_id, candidates[0]))
        elif len(candidates) > 1:
            best_idx = np.argmin(scores)
            final_pairs.append((ref_id, candidates[best_idx]))
        # If no candidates, unit is not matched
    
    return final_pairs


def generate_alignment_links(
    ref_data: Dict[str, Any],
    align_data: Dict[str, Any],
    config: Optional[AlignmentConfig] = None,
) -> List[Tuple[str, str]]:
    """
    Generate alignment links between two recordings.
    
    Uses iterative matching with increasing distance thresholds.
    
    Args:
        ref_data: Reference recording data
        align_data: Recording to align
        config: Alignment configuration
    
    Returns:
        List of (ref_unit_id, align_unit_id) pairs
    """
    if config is None:
        config = default_config.alignment
    
    # Get high quality units
    ref_units = get_high_quality_units(ref_data, config.quality_threshold)
    align_units = get_high_quality_units(align_data, config.quality_threshold)
    
    logger.info(
        f"Aligning {len(ref_units)} ref units with {len(align_units)} target units"
    )
    
    # Add signatures if not present
    if not any("alignment" in u for u in ref_data.get("units", {}).values()):
        add_signatures_to_data(ref_data)
    if not any("alignment" in u for u in align_data.get("units", {}).values()):
        add_signatures_to_data(align_data)
    
    final_pairs = []
    remaining_ref = set(ref_units)
    remaining_align = set(align_units)
    
    # Iterate with increasing distance thresholds
    for dist_thresh in config.iteration_distances:
        if not remaining_ref or not remaining_align:
            break
        
        pairs = get_unit_pair_list(
            ref_data,
            align_data,
            list(remaining_ref),
            list(remaining_align),
            waveform_weight=config.waveform_weight,
            distance_threshold=dist_thresh,
        )
        
        # Add pairs and remove matched units
        for ref_id, align_id in pairs:
            if ref_id in remaining_ref and align_id in remaining_align:
                final_pairs.append((ref_id, align_id))
                remaining_ref.discard(ref_id)
                remaining_align.discard(align_id)
        
        logger.debug(
            f"Distance {dist_thresh}: found {len(pairs)} pairs, "
            f"{len(remaining_ref)} ref remaining"
        )
    
    logger.info(f"Total aligned pairs: {len(final_pairs)}")
    
    return final_pairs


# =============================================================================
# Chain Building
# =============================================================================

def build_alignment_chains(
    recordings: Dict[str, Dict[str, Any]],
    connections: Dict[str, List[Tuple[str, str]]],
) -> pd.DataFrame:
    """
    Build alignment chains DataFrame from pairwise connections.
    
    Each row represents a chain of matched units across recordings.
    
    Args:
        recordings: Dict mapping recording names to data
        connections: Dict mapping "recA_to_recB" to pair lists
    
    Returns:
        DataFrame with columns as recording names, rows as chains
    """
    if not connections:
        return pd.DataFrame()
    
    # Get ordered list of recording names
    rec_names = sorted(recordings.keys())
    
    # Build forward maps for each connection
    connection_keys = sorted(connections.keys())
    step_maps = []
    
    for key in connection_keys:
        fwd = {src: dst for src, dst in connections[key]}
        step_maps.append(fwd)
    
    # Extract file order from connection keys
    files_order = []
    for i, key in enumerate(connection_keys):
        src_file, _, dst_file = key.partition("_to_")
        if i == 0:
            files_order.append(src_file)
        files_order.append(dst_file)
    
    # Walk chains
    chains = []
    
    for i, fwd in enumerate(step_maps):
        sources = set(fwd.keys())
        
        # Find starting nodes (not a destination of previous step)
        if i > 0:
            prev_dests = set(step_maps[i - 1].values())
            starts = sorted(sources - prev_dests)
        else:
            starts = sorted(sources)
        
        for start in starts:
            chain = [start]
            current = start
            
            # Follow the chain forward
            for j in range(i, len(step_maps)):
                next_node = step_maps[j].get(current)
                if next_node is None:
                    break
                chain.append(next_node)
                current = next_node
            
            # Create row with NaN padding
            row = [np.nan] * len(files_order)
            for offset, node in enumerate(chain):
                col_idx = i + offset
                if 0 <= col_idx < len(files_order):
                    row[col_idx] = node
            
            chains.append(row)
    
    df = pd.DataFrame(chains, columns=files_order)
    
    return df


def build_alignment_chains_fixed_ref(
    recordings: Dict[str, Dict[str, Any]],
    connections: Dict[str, List[Tuple[str, str]]],
    ref_name: str,
) -> pd.DataFrame:
    """
    Build alignment chains with a fixed reference recording.
    
    All units are aligned to the reference recording.
    
    Args:
        recordings: Dict mapping recording names to data
        connections: Dict mapping "refName_to_targetName" to pair lists
        ref_name: Name of the reference recording
    
    Returns:
        DataFrame with reference and all aligned recordings
    """
    if not connections:
        return pd.DataFrame()
    
    # Build the dataframe from connections
    connection_keys = sorted(connections.keys())
    
    df = pd.DataFrame()
    
    for key in connection_keys:
        pairs = connections[key]
        if not pairs:
            continue
        
        src_file, _, dst_file = key.partition("_to_")
        
        # Create temporary dataframe for this connection
        pairs_array = np.array(pairs)
        temp_df = pd.DataFrame({
            src_file: pairs_array[:, 0],
            dst_file: pairs_array[:, 1],
        })
        
        if df.empty:
            df = temp_df
        else:
            # Merge on the reference column
            df = pd.concat([df, temp_df], axis=1)
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
    
    return df


# =============================================================================
# Grouped Data Operations
# =============================================================================

def create_aligned_group(
    hdf5_paths: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    use_fixed_ref: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """
    Create aligned group from multiple recordings.
    
    Args:
        hdf5_paths: List of HDF5 file paths
        output_path: Output path for grouped HDF5
        config: Pipeline configuration
        use_fixed_ref: Use fixed reference alignment (last recording)
    
    Returns:
        Tuple of (grouped_data, output_path)
    """
    if config is None:
        config = default_config
    
    if output_path is None:
        output_path = get_grouped_hdf5_path(config.output_dir)
    
    output_path = Path(output_path)
    
    # Load all recordings
    recordings = {}
    for hdf5_path in hdf5_paths:
        hdf5_path = Path(hdf5_path)
        name = hdf5_path.stem  # Use filename without extension
        recordings[name] = load_recording_from_hdf5(hdf5_path)
        add_signatures_to_data(recordings[name])
    
    rec_names = sorted(recordings.keys())
    logger.info(f"Loaded {len(rec_names)} recordings for alignment")
    
    # Generate pairwise connections (sequential)
    connections = {}
    for i in range(len(rec_names) - 1):
        ref_name = rec_names[i]
        align_name = rec_names[i + 1]
        
        pairs = generate_alignment_links(
            recordings[ref_name],
            recordings[align_name],
            config.alignment,
        )
        
        connections[f"{ref_name}_to_{align_name}"] = pairs
    
    # Build chains
    chains_df = build_alignment_chains(recordings, connections)
    
    # Fixed reference alignment if requested
    fixed_connections = {}
    fixed_chains_df = pd.DataFrame()
    
    if use_fixed_ref and len(rec_names) > 2:
        ref_idx = config.alignment.fixed_ref_index
        if ref_idx is None:
            ref_idx = -1
        ref_name = rec_names[ref_idx]
        
        fixed_config = AlignmentConfig(
            waveform_weight=config.alignment.fixed_align_waveform_weight,
            iteration_distances=config.alignment.fixed_align_iteration_distances,
            quality_threshold=config.alignment.quality_threshold,
        )
        
        for i, align_name in enumerate(rec_names):
            if align_name == ref_name:
                continue
            
            pairs = generate_alignment_links(
                recordings[ref_name],
                recordings[align_name],
                fixed_config,
            )
            
            fixed_connections[f"{ref_name}_to_{align_name}"] = pairs
        
        fixed_chains_df = build_alignment_chains_fixed_ref(
            recordings, fixed_connections, ref_name
        )
    
    # Create grouped data structure
    grouped_data = {
        "recordings": recordings,
        "connections": connections,
        "alignment_chains": chains_df,
    }
    
    if use_fixed_ref:
        grouped_data["fixed_connections"] = fixed_connections
        grouped_data["fixed_alignment_chains"] = fixed_chains_df
    
    # Save to HDF5
    save_aligned_group_to_hdf5(grouped_data, output_path)
    
    return grouped_data, output_path


def save_aligned_group_to_hdf5(
    grouped_data: Dict[str, Any],
    hdf5_path: Union[str, Path],
) -> None:
    """
    Save aligned group data to HDF5.
    
    Args:
        grouped_data: Grouped data dictionary
        hdf5_path: Output HDF5 path
    """
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving aligned group to: {hdf5_path}")
    
    with h5py.File(hdf5_path, "w") as f:
        # Save alignment chains as dataset
        chains_df = grouped_data.get("alignment_chains", pd.DataFrame())
        if not chains_df.empty:
            align_grp = f.create_group("alignment")
            
            # Save column names
            align_grp.attrs["columns"] = list(chains_df.columns)
            
            # Save data (convert to string for unit IDs)
            chains_data = chains_df.fillna("").values.astype(str)
            align_grp.create_dataset("chains", data=chains_data.astype("S50"))
        
        # Save fixed alignment chains
        fixed_chains_df = grouped_data.get("fixed_alignment_chains", pd.DataFrame())
        if not fixed_chains_df.empty:
            fixed_grp = f.create_group("fixed_alignment")
            fixed_grp.attrs["columns"] = list(fixed_chains_df.columns)
            fixed_data = fixed_chains_df.fillna("").values.astype(str)
            fixed_grp.create_dataset("chains", data=fixed_data.astype("S50"))
        
        # Save connections as datasets
        connections = grouped_data.get("connections", {})
        if connections:
            conn_grp = f.create_group("connections")
            for key, pairs in connections.items():
                if pairs:
                    pairs_array = np.array(pairs, dtype="S50")
                    conn_grp.create_dataset(key, data=pairs_array)
        
        # Save fixed connections
        fixed_connections = grouped_data.get("fixed_connections", {})
        if fixed_connections:
            fixed_conn_grp = f.create_group("fixed_connections")
            for key, pairs in fixed_connections.items():
                if pairs:
                    pairs_array = np.array(pairs, dtype="S50")
                    fixed_conn_grp.create_dataset(key, data=pairs_array)
        
        # Save recording references
        recordings = grouped_data.get("recordings", {})
        if recordings:
            rec_grp = f.create_group("recordings")
            for name, rec_data in recordings.items():
                sub_grp = rec_grp.create_group(name)
                
                # Save high-level info
                sub_grp.attrs["num_units"] = len(rec_data.get("units", {}))
                sub_grp.attrs["source_path"] = rec_data.get("source_path", "")
                
                # Save unit IDs and quality indices
                unit_ids = list(rec_data.get("units", {}).keys())
                quality_indices = [
                    rec_data["units"][uid].get("quality_index", 0.0)
                    for uid in unit_ids
                ]
                
                sub_grp.create_dataset("unit_ids", data=np.array(unit_ids, dtype="S50"))
                sub_grp.create_dataset("quality_indices", data=np.array(quality_indices))
        
        f.attrs["pipeline"] = "step_change_analysis_aligned"
        f.attrs["num_recordings"] = len(recordings)


def load_aligned_group_from_hdf5(
    hdf5_path: Union[str, Path],
    load_full_recordings: bool = False,
) -> Dict[str, Any]:
    """
    Load aligned group data from HDF5.
    
    Args:
        hdf5_path: Path to grouped HDF5 file
        load_full_recordings: Whether to load full recording data
    
    Returns:
        Grouped data dictionary
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    logger.info(f"Loading aligned group from: {hdf5_path}")
    
    grouped_data = {}
    
    with h5py.File(hdf5_path, "r") as f:
        # Load alignment chains
        if "alignment" in f:
            columns = list(f["alignment"].attrs["columns"])
            chains_data = f["alignment/chains"][:].astype(str)
            chains_data = np.where(chains_data == "", np.nan, chains_data)
            grouped_data["alignment_chains"] = pd.DataFrame(
                chains_data, columns=columns
            )
        
        # Load fixed alignment chains
        if "fixed_alignment" in f:
            columns = list(f["fixed_alignment"].attrs["columns"])
            chains_data = f["fixed_alignment/chains"][:].astype(str)
            chains_data = np.where(chains_data == "", np.nan, chains_data)
            grouped_data["fixed_alignment_chains"] = pd.DataFrame(
                chains_data, columns=columns
            )
        
        # Load connections
        if "connections" in f:
            grouped_data["connections"] = {}
            for key in f["connections"].keys():
                pairs = f[f"connections/{key}"][:].astype(str)
                grouped_data["connections"][key] = [tuple(p) for p in pairs]
        
        # Load fixed connections
        if "fixed_connections" in f:
            grouped_data["fixed_connections"] = {}
            for key in f["fixed_connections"].keys():
                pairs = f[f"fixed_connections/{key}"][:].astype(str)
                grouped_data["fixed_connections"][key] = [tuple(p) for p in pairs]
        
        # Load recordings info
        if "recordings" in f:
            grouped_data["recordings"] = {}
            for name in f["recordings"].keys():
                rec_info = {
                    "num_units": f[f"recordings/{name}"].attrs["num_units"],
                    "source_path": f[f"recordings/{name}"].attrs["source_path"],
                }
                
                if load_full_recordings and rec_info["source_path"]:
                    try:
                        rec_info.update(
                            load_recording_from_hdf5(rec_info["source_path"])
                        )
                    except Exception as e:
                        logger.warning(f"Could not load {name}: {e}")
                
                grouped_data["recordings"][name] = rec_info
    
    return grouped_data
