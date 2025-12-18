"""
Flow orchestration for HD-MEA pipeline.

Flows combine Stage 1 (Load) and Stage 2 (Extract) into named workflows.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hdmea.pipeline.config import load_flow_config, FlowConfig
from hdmea.pipeline.runner import (
    load_recording,
    extract_features,
    LoadResult,
    ExtractionResult,
    FlowResult,
)


logger = logging.getLogger(__name__)


def run_flow(
    flow_name: str,
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    dataset_id: Optional[str] = None,
    *,
    output_dir: Union[str, Path] = "artifacts",
    config_dir: Optional[Path] = None,
    force_load: bool = False,
    force_extract: bool = False,
) -> FlowResult:
    """
    Run a named pipeline flow (Stage 1 + Stage 2).
    
    Flows are defined in config/flows/{flow_name}.json and specify
    which features to extract.
    
    Args:
        flow_name: Name of flow configuration (e.g., "set6a_full").
        cmcr_path: External path to .cmcr file.
        cmtr_path: External path to .cmtr file.
        dataset_id: Unique identifier for the recording.
        output_dir: Directory for HDF5 output.
        config_dir: Directory containing flow configs.
        force_load: Force re-run of Stage 1 even if HDF5 exists.
        force_extract: Force re-extraction of features even if cached.
    
    Returns:
        FlowResult with combined results from both stages.
    
    Raises:
        ConfigurationError: If flow config not found or invalid.
    """
    logger.info(f"Starting flow: {flow_name}")
    
    # Load flow configuration
    flow_config = load_flow_config(flow_name, config_dir)
    
    logger.info(f"Flow '{flow_name}' loaded: {len(flow_config.get_feature_sets())} feature sets")
    
    # Stage 1: Load recording
    load_result = None
    load_enabled = flow_config.stages.get("load", {})
    if isinstance(load_enabled, dict):
        load_enabled = load_enabled.get("enabled", True)
    else:
        load_enabled = getattr(load_enabled, "enabled", True)
    
    if load_enabled:
        logger.info("Running Stage 1: Data Loading")
        load_result = load_recording(
            cmcr_path=cmcr_path,
            cmtr_path=cmtr_path,
            dataset_id=dataset_id,
            output_dir=output_dir,
            force=force_load,
            config=flow_config.defaults,
        )
        hdf5_path = load_result.hdf5_path
    else:
        # Assume HDF5 already exists
        output_dir = Path(output_dir)
        if dataset_id:
            hdf5_path = output_dir / f"{dataset_id}.h5"
        else:
            raise ValueError("dataset_id required when load stage is disabled")
    
    # Stage 2: Extract features
    extraction_result = None
    feature_sets = flow_config.get_feature_sets()
    
    if feature_sets:
        logger.info(f"Running Stage 2: Extracting {len(feature_sets)} feature sets")
        extraction_result = extract_features(
            hdf5_path=hdf5_path,
            features=feature_sets,
            force=force_extract,
            config_overrides=flow_config.defaults,
        )
    else:
        logger.info("No feature sets specified, skipping Stage 2")
    
    success = True
    if load_result and not load_result.stage1_completed:
        success = False
    if extraction_result and extraction_result.features_failed:
        success = False
    
    logger.info(f"Flow '{flow_name}' completed: success={success}")
    
    return FlowResult(
        hdf5_path=hdf5_path,
        load_result=load_result,
        extraction_result=extraction_result,
        success=success,
    )


def list_available_flows(config_dir: Optional[Path] = None) -> List[str]:
    """
    List available flow configurations.
    
    Args:
        config_dir: Directory containing flow configs
    
    Returns:
        List of flow names
    """
    if config_dir is None:
        config_dir = Path("config")
    
    flows_dir = config_dir / "flows"
    
    if not flows_dir.exists():
        return []
    
    return [f.stem for f in flows_dir.glob("*.json")]


def get_flow_info(flow_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about a flow.
    
    Args:
        flow_name: Name of flow
        config_dir: Directory containing flow configs
    
    Returns:
        Dictionary with flow metadata
    """
    flow_config = load_flow_config(flow_name, config_dir)
    
    return {
        "name": flow_config.name,
        "description": flow_config.description,
        "version": flow_config.version,
        "feature_sets": flow_config.get_feature_sets(),
        "stages": list(flow_config.stages.keys()),
    }

