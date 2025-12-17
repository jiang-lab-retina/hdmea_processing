"""
Pipeline runner for HD-MEA data processing.

Implements Stage 1 (Data Loading) and Stage 2 (Feature Extraction) with caching.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

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
# Constants
# =============================================================================

DEFAULT_ACQUISITION_RATE: float = 20000.0  # Hz - typical for MaxOne/MaxTwo
MIN_TYPICAL_ACQUISITION_RATE: float = 1000.0  # Hz - warn if below
MAX_TYPICAL_ACQUISITION_RATE: float = 100000.0  # Hz - warn if above


# =============================================================================
# Helper Functions
# =============================================================================

def validate_acquisition_rate(rate: Optional[float]) -> bool:
    """
    Validate that acquisition_rate is acceptable.
    
    Args:
        rate: Proposed acquisition rate in Hz (or None)
    
    Returns:
        True if rate is valid (> 0), False otherwise.
        
    Note:
        Logs a warning if rate is outside typical range (1000-100000 Hz)
        but still returns True (value is usable, just unusual).
    """
    if rate is None:
        return False
    
    if rate <= 0:
        logger.warning(f"Invalid acquisition_rate: {rate} Hz (must be > 0)")
        return False
    
    # Warn if outside typical range, but accept the value
    if rate < MIN_TYPICAL_ACQUISITION_RATE or rate > MAX_TYPICAL_ACQUISITION_RATE:
        logger.warning(
            f"Unusual acquisition_rate: {rate:.0f} Hz "
            f"(typical range: {MIN_TYPICAL_ACQUISITION_RATE:.0f}-{MAX_TYPICAL_ACQUISITION_RATE:.0f} Hz)"
        )
    
    return True


def compute_frame_time(acquisition_rate: float) -> float:
    """
    Compute frame_time from acquisition_rate.
    
    Args:
        acquisition_rate: Sampling rate in Hz (must be > 0)
    
    Returns:
        Frame time in seconds (1 / acquisition_rate)
    
    Raises:
        ValueError: If acquisition_rate <= 0
    """
    if acquisition_rate <= 0:
        raise ValueError(f"acquisition_rate must be > 0, got {acquisition_rate}")
    return 1.0 / acquisition_rate


def get_frame_timestamps(
    light_reference: np.ndarray,
    frame_channel: int = 1,
    exclude_initial_frames: int = 4,
    normalize_percentile: float = 99.9,
    peak_height_threshold: float = 0.1,
) -> np.ndarray:
    """
    Detect frame timestamps from light reference signal.
    
    Finds frame boundaries by detecting peaks in the derivative of the
    normalized light reference signal. This matches the legacy approach
    from jianglab.common_functions.get_frame_timestamp().
    
    Args:
        light_reference: Light reference data array (raw samples).
        frame_channel: Which channel to use (0 or 1). Default: 1.
        exclude_initial_frames: Number of initial frame peaks to exclude
            (often noisy). Default: 4.
        normalize_percentile: Percentile for normalization. Default: 99.9.
        peak_height_threshold: Minimum height for peak detection in
            derivative. Default: 0.1.
    
    Returns:
        Array of frame timestamps (sample indices where frames start).
    """
    from scipy.signal import find_peaks
    
    # Normalize the signal
    percentile_val = np.percentile(light_reference, normalize_percentile)
    if percentile_val > 0:
        norm_signal = light_reference / percentile_val
    else:
        norm_signal = light_reference
    
    # Find peaks in the derivative (frame transitions)
    diff_signal = np.diff(norm_signal)
    peaks, _ = find_peaks(diff_signal, height=peak_height_threshold)
    
    # Exclude initial frames
    if exclude_initial_frames > 0 and len(peaks) > exclude_initial_frames:
        peaks = peaks[exclude_initial_frames:]
    
    logger.info(f"Detected {len(peaks)} frame timestamps from light reference")
    
    return peaks.astype(np.uint64)


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
    allow_overwrite: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> LoadResult:
    """
    Load recording from external .cmcr/.cmtr files to Zarr artifact.
    
    This is Stage 1 of the pipeline. Produces exactly ONE Zarr archive
    per recording containing all data needed for feature extraction.
    
    Timing Metadata:
        The function extracts timing parameters using a priority chain:
        
        1. CMCR file (primary): Extracts acquisition_rate from file metadata
           or estimates from data length and recording duration.
        2. CMTR file (fallback): Extracts acquisition_rate from file attributes
           if CMCR is unavailable or doesn't provide valid rate.
        3. Default (last resort): Uses 20000 Hz if neither source provides
           valid rate, and logs a warning.
        
        The frame_time (seconds per sample) is computed as 1/acquisition_rate.
        Both values are stored in the Zarr metadata group.
    
    Args:
        cmcr_path: External path to .cmcr file (raw sensor data).
        cmtr_path: External path to .cmtr file (spike-sorted data).
        dataset_id: Unique identifier for the recording.
        output_dir: Directory for Zarr output. Default: "artifacts".
        force: If True, overwrite existing Zarr. Default: False.
        allow_overwrite: If True, allow overwriting existing Zarr even if params differ. Default: False.
        config: Optional configuration dictionary.
    
    Returns:
        LoadResult with zarr_path, dataset_id, unit count, and warnings.
    
    Raises:
        ConfigurationError: If neither cmcr_path nor cmtr_path provided.
        FileNotFoundError: If specified file(s) do not exist.
        DataLoadError: If files cannot be read.
    
    Example:
        >>> result = load_recording(
        ...     cmcr_path="path/to/recording.cmcr",
        ...     cmtr_path="path/to/recording.cmtr",
        ...     dataset_id="REC_2023-12-07",
        ... )
        >>> # Access timing metadata from the created Zarr:
        >>> import zarr
        >>> root = zarr.open(str(result.zarr_path), mode="r")
        >>> acquisition_rate = root["metadata"].attrs["acquisition_rate"]  # Hz
        >>> frame_time = root["metadata"].attrs["frame_time"]  # seconds
    """
    warnings = []
    config = config or {}
    
    # Combine force and allow_overwrite - if either is True, allow overwrite
    overwrite = force or allow_overwrite
    
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
    if zarr_path.exists() and not overwrite:
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
                logger.warning("Zarr exists but params differ. Use force=True or allow_overwrite=True to overwrite.")
                raise FileExistsError(
                    f"Zarr already exists with different params: {zarr_path}. "
                    "Use force=True or allow_overwrite=True to overwrite."
                )
    
    # Create new Zarr
    logger.info(f"Creating new Zarr: {zarr_path}")
    root = create_recording_zarr(
        zarr_path,
        dataset_id=dataset_id,
        config=config,
        overwrite=overwrite,
    )
    
    units_data = {}
    light_reference = {}
    # Top-level metadata (timing + dataset info)
    metadata = {"dataset_id": dataset_id}
    # System metadata from raw files (CMCR/CMTR) goes under sys_meta
    sys_meta = {}
    
    # Track acquisition_rate sources for priority chain
    cmcr_acquisition_rate = None
    cmtr_acquisition_rate = None
    
    # Load CMTR data (spike-sorted)
    if cmtr_path_obj:
        try:
            cmtr_result = load_cmtr_data(cmtr_path_obj)
            units_data = cmtr_result["units"]
            # Raw file metadata goes to sys_meta
            sys_meta.update(cmtr_result.get("metadata", {}))
            # Extract CMTR acquisition_rate as fallback
            cmtr_acquisition_rate = cmtr_result.get("acquisition_rate")
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
            # Extract CMCR acquisition_rate as primary source
            cmcr_acquisition_rate = cmcr_result.get("acquisition_rate")
            # Raw file metadata goes to sys_meta
            sys_meta.update(cmcr_result.get("metadata", {}))
        except Exception as e:
            logger.error(f"Failed to load CMCR: {e}")
            warnings.append(f"CMCR load failed: {e}")
    else:
        warnings.append("No CMCR file provided - light reference unavailable")
    
    # Determine acquisition_rate using priority chain: CMCR -> CMTR -> default
    acquisition_rate = None
    acquisition_rate_source = None
    
    # Try CMCR first (primary)
    if validate_acquisition_rate(cmcr_acquisition_rate):
        acquisition_rate = cmcr_acquisition_rate
        acquisition_rate_source = "CMCR"
    # Try CMTR as fallback
    elif validate_acquisition_rate(cmtr_acquisition_rate):
        acquisition_rate = cmtr_acquisition_rate
        acquisition_rate_source = "CMTR"
    # Use default as last resort
    else:
        acquisition_rate = DEFAULT_ACQUISITION_RATE
        acquisition_rate_source = "default"
        logger.warning(
            f"Using default acquisition_rate ({DEFAULT_ACQUISITION_RATE:.0f} Hz) - "
            "could not extract from CMCR or CMTR"
        )
        warnings.append(f"Using default acquisition_rate: {DEFAULT_ACQUISITION_RATE:.0f} Hz")
    
    logger.info(f"acquisition_rate: {acquisition_rate:.0f} Hz (source: {acquisition_rate_source})")
    
    # Compute frame_time from acquisition_rate
    sample_interval = compute_frame_time(acquisition_rate)
    logger.debug(f"sample_interval: {sample_interval:.6e} seconds")
    
    # Detect frame timestamps from light reference signal
    # Following legacy approach from jianglab.common_functions.get_frame_timestamp()
    frame_timestamps = None
    if light_reference:
        # Prefer raw_ch2 (frame channel 1 in legacy = index 1 = channel 2)
        frame_channel_data = light_reference.get("raw_ch2")
        if frame_channel_data is None:
            frame_channel_data = light_reference.get("raw_ch1")
        if frame_channel_data is not None and len(frame_channel_data) > 0:
            try:
                frame_timestamps = get_frame_timestamps(
                    frame_channel_data,
                    exclude_initial_frames=4,
                    normalize_percentile=99.9,
                    peak_height_threshold=0.1,
                )
            except Exception as e:
                logger.warning(f"Failed to detect frame timestamps: {e}")
                warnings.append(f"Frame timestamp detection failed: {e}")
    
    # Add timing metadata (top-level, not under sys_meta)
    metadata["acquisition_rate"] = acquisition_rate
    metadata["sample_interval"] = sample_interval  # per-sample time (1/acquisition_rate)
    if frame_timestamps is not None:
        metadata["frame_timestamps"] = frame_timestamps  # Array of sample indices
        # Compute frame_time as array of timestamps in seconds
        metadata["frame_time"] = (frame_timestamps / acquisition_rate).astype(np.float64)
    
    # Add sys_meta (raw file metadata) as nested dict
    metadata["sys_meta"] = sys_meta
    
    # Compute firing rates for each unit
    for unit_id, unit_info in units_data.items():
        spike_times = unit_info.get("spike_times")
        if spike_times is not None and len(spike_times) > 0:
            # Get recording duration from sys_meta or estimate
            duration_us = sys_meta.get("recording_duration_s", 0) * 1e6
            if duration_us == 0:
                duration_us = spike_times[-1] + 1e6  # Add 1 second buffer
            
            unit_info["firing_rate_10hz"] = compute_firing_rate(
                spike_times, duration_us, bin_rate_hz=10
            )
    
    # Write to Zarr
    write_units(root, units_data)
    
    # Prepare frame_times for write_stimulus
    frame_times_dict = None
    if frame_timestamps is not None:
        frame_times_dict = {"default": frame_timestamps}
    
    write_stimulus(root, light_reference, frame_times=frame_times_dict)
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
            recording_metadata = root["metadata"] if "metadata" in root else None
            
            for unit_id in unit_ids:
                unit_data = root["units"][unit_id]
                
                # Check if this unit already has feature
                existing = list_features(root, unit_id)
                if feature_name in existing and not force:
                    continue
                
                # Extract
                try:
                    result = extractor.extract(
                        unit_data, stimulus_data, 
                        config=config_overrides,
                        metadata=recording_metadata
                    )
                    
                    # Write to Zarr
                    feature_metadata = {
                        "feature_name": feature_name,
                        "extractor_version": extractor.version,
                        "params_hash": hash_config(config_overrides or {}),
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    
                    write_feature_to_unit(
                        root, unit_id, feature_name, result, feature_metadata
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
