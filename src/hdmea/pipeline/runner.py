"""
Pipeline runner for HD-MEA data processing.

Implements Stage 1 (Data Loading) and Stage 2 (Feature Extraction) with caching.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hdmea.io.cmcr import load_cmcr_data
from hdmea.io.cmtr import load_cmtr_data, validate_cmcr_cmtr_match
from hdmea.io.zarr_store import (
    create_recording_zarr,
    open_recording_zarr,
    write_units,
    write_stimulus,
    write_metadata,
    write_source_files,
    mark_stage1_complete,
    get_stage1_status,
    list_units,
    list_features,
    write_feature_to_unit,
)
from hdmea.preprocess.filtering import compute_firing_rate
from hdmea.utils.exceptions import (
    ConfigurationError,
    CacheConflictError,
    MissingInputError,
)
from hdmea.utils.hashing import hash_config, verify_hash
from hdmea.utils.validation import (
    validate_dataset_id,
    validate_input_files,
    derive_dataset_id,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class LoadResult:
    """Result of Stage 1 data loading."""
    zarr_path: Path
    dataset_id: str
    num_units: int
    stage1_completed: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of Stage 2 feature extraction."""
    zarr_path: Path
    features_extracted: List[str] = field(default_factory=list)
    features_skipped: List[str] = field(default_factory=list)
    features_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FlowResult:
    """Result of running a complete flow."""
    zarr_path: Path
    load_result: Optional[LoadResult] = None
    extraction_result: Optional[ExtractionResult] = None
    success: bool = False


# =============================================================================
# Stage 1: Data Loading
# =============================================================================

def load_recording(
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    dataset_id: Optional[str] = None,
    *,
    output_dir: Union[str, Path] = "artifacts",
    force: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> LoadResult:
    """
    Load recording from external .cmcr/.cmtr files to Zarr artifact.
    
    This is Stage 1 of the pipeline. Produces exactly ONE Zarr archive
    per recording containing all data needed for feature extraction.
    
    Args:
        cmcr_path: External path to .cmcr file (raw sensor data).
        cmtr_path: External path to .cmtr file (spike-sorted data).
        dataset_id: Unique identifier for the recording.
        output_dir: Directory for Zarr output. Default: "artifacts".
        force: If True, overwrite existing Zarr. Default: False.
        config: Optional configuration dictionary.
    
    Returns:
        LoadResult with zarr_path, dataset_id, unit count, and warnings.
    
    Raises:
        ConfigurationError: If neither cmcr_path nor cmtr_path provided.
        FileNotFoundError: If specified file(s) do not exist.
        DataLoadError: If files cannot be read.
    """
    warnings = []
    config = config or {}
    
    # Validate inputs
    cmcr_path_obj, cmtr_path_obj = validate_input_files(
        Path(cmcr_path) if cmcr_path else None,
        Path(cmtr_path) if cmtr_path else None,
    )
    
    # Derive or validate dataset_id
    if dataset_id is None:
        dataset_id = derive_dataset_id(cmcr_path_obj, cmtr_path_obj)
        logger.info(f"Derived dataset_id: {dataset_id}")
    else:
        dataset_id = validate_dataset_id(dataset_id)
    
    # Validate CMCR/CMTR match if both provided
    validate_cmcr_cmtr_match(cmcr_path_obj, cmtr_path_obj)
    
    # Prepare output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / f"{dataset_id}.zarr"
    
    # Check for existing Zarr (caching)
    if zarr_path.exists() and not force:
        root = open_recording_zarr(zarr_path, mode="r")
        status = get_stage1_status(root)
        
        if status["completed"]:
            # Check if params match
            if verify_hash(config, status["params_hash"]):
                logger.info(f"Cache hit: {zarr_path} already exists with matching params")
                num_units = len(list_units(root))
                return LoadResult(
                    zarr_path=zarr_path,
                    dataset_id=dataset_id,
                    num_units=num_units,
                    stage1_completed=True,
                    warnings=["Using cached Zarr (skipped loading)"],
                )
            else:
                logger.warning("Zarr exists but params differ. Use force=True to overwrite.")
                warnings.append("Params mismatch with cached Zarr")
    
    # Create new Zarr
    logger.info(f"Creating new Zarr: {zarr_path}")
    root = create_recording_zarr(
        zarr_path,
        dataset_id=dataset_id,
        config=config,
        overwrite=force,
    )
    
    units_data = {}
    light_reference = {}
    metadata = {"dataset_id": dataset_id}
    
    # Load CMTR data (spike-sorted)
    if cmtr_path_obj:
        try:
            cmtr_result = load_cmtr_data(cmtr_path_obj)
            units_data = cmtr_result["units"]
            metadata.update(cmtr_result.get("metadata", {}))
        except Exception as e:
            logger.error(f"Failed to load CMTR: {e}")
            warnings.append(f"CMTR load failed: {e}")
    else:
        warnings.append("No CMTR file provided - spike data unavailable")
    
    # Load CMCR data (raw sensor / light reference)
    if cmcr_path_obj:
        try:
            cmcr_result = load_cmcr_data(cmcr_path_obj)
            light_reference = cmcr_result.get("light_reference", {})
            metadata["acquisition_rate"] = cmcr_result.get("acquisition_rate", 20000)
            metadata.update(cmcr_result.get("metadata", {}))
        except Exception as e:
            logger.error(f"Failed to load CMCR: {e}")
            warnings.append(f"CMCR load failed: {e}")
    else:
        warnings.append("No CMCR file provided - light reference unavailable")
    
    # Compute firing rates for each unit
    for unit_id, unit_info in units_data.items():
        spike_times = unit_info.get("spike_times")
        if spike_times is not None and len(spike_times) > 0:
            # Get recording duration from metadata or estimate
            duration_us = metadata.get("recording_duration_s", 0) * 1e6
            if duration_us == 0:
                duration_us = spike_times[-1] + 1e6  # Add 1 second buffer
            
            unit_info["firing_rate_10hz"] = compute_firing_rate(
                spike_times, duration_us, bin_rate_hz=10
            )
    
    # Write to Zarr
    write_units(root, units_data)
    write_stimulus(root, light_reference)
    write_metadata(root, metadata)
    write_source_files(root, cmcr_path_obj, cmtr_path_obj)
    
    # Mark complete
    mark_stage1_complete(root)
    
    logger.info(f"Stage 1 complete: {len(units_data)} units loaded to {zarr_path}")
    
    return LoadResult(
        zarr_path=zarr_path,
        dataset_id=dataset_id,
        num_units=len(units_data),
        stage1_completed=True,
        warnings=warnings,
    )


# =============================================================================
# Stage 2: Feature Extraction
# =============================================================================

def extract_features(
    zarr_path: Union[str, Path],
    features: List[str],
    *,
    force: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """
    Extract features from loaded Zarr and write back to same archive.
    
    This is Stage 2 of the pipeline. Reads from Zarr, computes features
    for each unit, and writes results to units/{unit_id}/features/{feature_name}/.
    
    Args:
        zarr_path: Path to Zarr archive from Stage 1.
        features: List of feature names to extract (must be registered).
        force: If True, overwrite existing features. Default: False.
        config_overrides: Optional parameter overrides for extractors.
    
    Returns:
        ExtractionResult with lists of extracted, skipped, and failed features.
    
    Raises:
        FileNotFoundError: If zarr_path does not exist.
        KeyError: If any feature name is not registered.
        MissingInputError: If required inputs for a feature are missing.
    """
    from hdmea.features.registry import FeatureRegistry
    
    zarr_path = Path(zarr_path)
    warnings = []
    extracted = []
    skipped = []
    failed = []
    
    # Open Zarr
    root = open_recording_zarr(zarr_path, mode="r+")
    
    # Validate Stage 1 is complete
    status = get_stage1_status(root)
    if not status["completed"]:
        raise ConfigurationError(
            f"Stage 1 not complete for {zarr_path}. Run load_recording first."
        )
    
    # Get unit list
    unit_ids = list_units(root)
    if not unit_ids:
        warnings.append("No units found in Zarr")
        return ExtractionResult(
            zarr_path=zarr_path,
            features_extracted=extracted,
            features_skipped=skipped,
            features_failed=failed,
            warnings=warnings,
        )
    
    logger.info(f"Extracting features for {len(unit_ids)} units: {features}")
    
    # Process each feature
    for feature_name in features:
        try:
            # Get extractor
            extractor_class = FeatureRegistry.get(feature_name)
            extractor = extractor_class(config=config_overrides)
            
            # Check for missing inputs
            missing = extractor.validate_inputs(root)
            if missing:
                raise MissingInputError(
                    f"Feature '{feature_name}' requires missing inputs: {missing}",
                    feature_name=feature_name,
                    missing_input=", ".join(missing),
                )
            
            # Check cache for all units
            all_cached = True
            for unit_id in unit_ids:
                existing_features = list_features(root, unit_id)
                if feature_name not in existing_features:
                    all_cached = False
                    break
            
            if all_cached and not force:
                logger.info(f"Skipping {feature_name} - already extracted (cache hit)")
                skipped.append(feature_name)
                continue
            
            # Extract for each unit
            stimulus_data = root["stimulus"]
            
            for unit_id in unit_ids:
                unit_data = root["units"][unit_id]
                
                # Check if this unit already has feature
                existing = list_features(root, unit_id)
                if feature_name in existing and not force:
                    continue
                
                # Extract
                try:
                    result = extractor.extract(
                        unit_data, stimulus_data, config=config_overrides
                    )
                    
                    # Write to Zarr
                    metadata = {
                        "feature_name": feature_name,
                        "extractor_version": extractor.version,
                        "params_hash": hash_config(config_overrides or {}),
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    
                    write_feature_to_unit(
                        root, unit_id, feature_name, result, metadata
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to extract {feature_name} for {unit_id}: {e}")
                    warnings.append(f"Failed {feature_name} for {unit_id}: {e}")
            
            extracted.append(feature_name)
            logger.info(f"Extracted feature: {feature_name}")
            
        except KeyError as e:
            logger.error(f"Unknown feature: {feature_name}")
            failed.append(feature_name)
            warnings.append(str(e))
        except MissingInputError as e:
            logger.error(str(e))
            failed.append(feature_name)
            warnings.append(str(e))
        except Exception as e:
            logger.error(f"Feature extraction failed: {feature_name}: {e}")
            failed.append(feature_name)
            warnings.append(f"{feature_name}: {e}")
    
    # Update timestamp
    root.attrs["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    return ExtractionResult(
        zarr_path=zarr_path,
        features_extracted=extracted,
        features_skipped=skipped,
        features_failed=failed,
        warnings=warnings,
    )

