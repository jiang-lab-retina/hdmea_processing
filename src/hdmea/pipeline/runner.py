"""
Pipeline runner for HD-MEA data processing.

Implements Stage 1 (Data Loading) and Stage 2 (Feature Extraction) with caching.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from hdmea.io.cmcr import load_cmcr_data
from hdmea.io.cmtr import load_cmtr_data, validate_cmcr_cmtr_match
from hdmea.io.hdf5_store import (
    create_recording_hdf5,
    open_recording_hdf5,
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


def _convert_spike_times_to_samples(
    spike_times_us: np.ndarray,
    acquisition_rate: float,
) -> np.ndarray:
    """
    Convert spike timestamps from microseconds to acquisition sample indices.
    
    Args:
        spike_times_us: Array of spike timestamps in microseconds (10^-6 s).
            McsPy returns timestamps in microseconds from CMTR files.
        acquisition_rate: Sampling rate in Hz.
    
    Returns:
        Array of spike timestamps as sample indices (uint64).
    
    Formula:
        sample_index = round(timestamp_us × acquisition_rate / 10^6)
    
    Example:
        >>> spike_times_us = np.array([50_000, 100_000])  # 50ms, 100ms
        >>> _convert_spike_times_to_samples(spike_times_us, 20000.0)
        array([1000, 2000], dtype=uint64)
    """
    if len(spike_times_us) == 0:
        return np.array([], dtype=np.uint64)
    
    # Convert: sample_index = timestamp_us * acquisition_rate / 1e6
    # McsPy returns timestamps in microseconds (µs), not nanoseconds
    spike_times_samples = np.round(
        spike_times_us.astype(np.float64) * acquisition_rate / 1e6
    ).astype(np.uint64)
    
    return spike_times_samples


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
    hdf5_path: Path
    dataset_id: str
    num_units: int
    stage1_completed: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of Stage 2 feature extraction."""
    hdf5_path: Path
    features_extracted: List[str] = field(default_factory=list)
    features_skipped: List[str] = field(default_factory=list)
    features_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FlowResult:
    """Result of running a complete flow."""
    hdf5_path: Path
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
    Load recording from external .cmcr/.cmtr files to HDF5 artifact.
    
    This is Stage 1 of the pipeline. Produces exactly ONE HDF5 file
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
        Both values are stored in the HDF5 metadata group.
    
    Args:
        cmcr_path: External path to .cmcr file (raw sensor data).
        cmtr_path: External path to .cmtr file (spike-sorted data).
        dataset_id: Unique identifier for the recording.
        output_dir: Directory for HDF5 output. Default: "artifacts".
        force: If True, overwrite existing HDF5. Default: False.
        allow_overwrite: If True, allow overwriting existing HDF5 even if params differ. Default: False.
        config: Optional configuration dictionary.
    
    Returns:
        LoadResult with hdf5_path, dataset_id, unit count, and warnings.
    
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
        >>> # Access timing metadata from the created HDF5:
        >>> import h5py
        >>> with h5py.File(str(result.hdf5_path), mode="r") as f:
        ...     acquisition_rate = f["metadata/acquisition_rate"][0]  # Hz
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
    hdf5_path = output_dir / f"{dataset_id}.h5"
    
    # Check for existing HDF5 (caching)
    if hdf5_path.exists() and not overwrite:
        with open_recording_hdf5(hdf5_path, mode="r") as root:
            status = get_stage1_status(root)
            
            if status["completed"]:
                # Check if params match
                if verify_hash(config, status["params_hash"]):
                    logger.info(f"Cache hit: {hdf5_path} already exists with matching params")
                    num_units = len(list_units(root))
                    return LoadResult(
                        hdf5_path=hdf5_path,
                        dataset_id=dataset_id,
                        num_units=num_units,
                        stage1_completed=True,
                        warnings=["Using cached HDF5 (skipped loading)"],
                    )
                else:
                    logger.warning("HDF5 exists but params differ. Use force=True or allow_overwrite=True to overwrite.")
                    raise FileExistsError(
                        f"HDF5 already exists with different params: {hdf5_path}. "
                        "Use force=True or allow_overwrite=True to overwrite."
                    )
    
    # Create new HDF5
    logger.info(f"Creating new HDF5: {hdf5_path}")
    root = create_recording_hdf5(
        hdf5_path,
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
    
    # Process units in parallel: compute firing rates and convert spike_times
    # Uses 80% of CPU cores for parallel processing
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, min(int(cpu_count * 0.8), len(units_data)))
    
    # Get recording duration for firing rate computation
    recording_duration_us = sys_meta.get("recording_duration_s", 0) * 1e6
    
    def process_unit(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Process a single unit: compute firing rate and convert spike_times."""
        unit_id, unit_info = item
        spike_times = unit_info.get("spike_times")
        
        if spike_times is not None and len(spike_times) > 0:
            # Compute firing rate (using raw ns timestamps)
            duration_us = recording_duration_us
            if duration_us == 0:
                duration_us = spike_times[-1] + 1e6  # Add 1 second buffer
            
            unit_info["firing_rate_10hz"] = compute_firing_rate(
                spike_times, duration_us, bin_rate_hz=10
            )
            
            # Convert spike_times from nanoseconds to sample indices
            unit_info["spike_times"] = _convert_spike_times_to_samples(
                spike_times, acquisition_rate
            )
        
        return unit_id, unit_info
    
    # Process units in parallel with progress bar
    if len(units_data) > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_unit, item): item[0] 
                      for item in units_data.items()}
            
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc="Processing units (firing rates + spike_times)", 
                              unit="unit", leave=True):
                unit_id, processed_info = future.result()
                units_data[unit_id] = processed_info
    
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
    
    # Close the HDF5 file
    root.close()
    
    logger.info(f"Stage 1 complete: {len(units_data)} units loaded to {hdf5_path}")
    
    return LoadResult(
        hdf5_path=hdf5_path,
        dataset_id=dataset_id,
        num_units=len(units_data),
        stage1_completed=True,
        warnings=warnings,
    )


# =============================================================================
# Integrated Loading + eimage_sta Computation
# =============================================================================

@dataclass
class LoadWithEImageSTAResult:
    """Result of load_recording_with_eimage_sta."""
    hdf5_path: Path
    dataset_id: str
    num_units: int
    units_with_sta: int
    stage1_completed: bool
    elapsed_seconds: float
    filter_time_seconds: float
    warnings: List[str] = field(default_factory=list)


def load_recording_with_eimage_sta(
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
    dataset_id: Optional[str] = None,
    *,
    output_dir: Union[str, Path] = "artifacts",
    force: bool = False,
    # eimage_sta parameters
    duration_s: float = 120.0,
    spike_limit: int = 10000,
    unit_ids: Optional[List[str]] = None,
    window_range: Tuple[int, int] = (-10, 40),
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    skip_highpass: bool = False,
    chunk_duration_s: float = 30.0,
) -> LoadWithEImageSTAResult:
    """
    Load recording and compute eimage_sta in a single pass.
    
    This integrated function:
    1. Loads CMTR once (spike times)
    2. Creates HDF5 with unit data
    3. Loads CMCR sensor data in chunks
    4. Computes eimage_sta for each unit using chunk-based processing
    5. Saves everything at the end
    
    Memory efficient: Only one chunk of sensor data in memory at a time.
    
    Args:
        cmcr_path: Path to .cmcr file (raw sensor data).
        cmtr_path: Path to .cmtr file (spike-sorted data).
        dataset_id: Unique identifier for the recording.
        output_dir: Directory for HDF5 output. Default: "artifacts".
        force: If True, overwrite existing HDF5. Default: False.
        duration_s: Total duration to analyze in seconds (default 120s).
        spike_limit: Maximum spikes to use per unit (default 10000).
        unit_ids: List of unit IDs to process (default None = all units).
            Format: ["1", "2", "7"] matching CMTR numbering.
        window_range: Tuple (start, end) relative to spike in samples, e.g. (-10, 40).
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        skip_highpass: If True, skip high-pass filter and mean-center instead.
        chunk_duration_s: Duration of each chunk in seconds (default 30s).
    
    Returns:
        LoadWithEImageSTAResult with processing summary.
    
    Example:
        >>> result = load_recording_with_eimage_sta(
        ...     cmcr_path="O:/data/recording.cmcr",
        ...     cmtr_path="O:/data/recording.cmtr",
        ...     duration_s=120.0,
        ...     spike_limit=10000,
        ...     unit_ids=["1", "2", "7"],
        ...     window_range=(-10, 40),
        ...     chunk_duration_s=30.0,
        ... )
    """
    import time
    import h5py
    from scipy.signal import butter, sosfiltfilt
    from hdmea.features.eimage_sta.compute import (
        EImageSTAConfig,
        write_eimage_sta_to_hdf5,
    )
    
    start_time = time.time()
    warnings_list: List[str] = []
    
    cmcr_path = Path(cmcr_path)
    cmtr_path = Path(cmtr_path)
    
    # Validate inputs
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    if not cmtr_path.exists():
        raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
    
    # Parse window_range
    pre_samples = -window_range[0]
    post_samples = window_range[1]
    window_length = pre_samples + post_samples
    
    if pre_samples < 0 or post_samples < 0:
        raise ValueError(
            f"Invalid window_range {window_range}: start must be <= 0 and end must be >= 0"
        )
    
    logger.info(
        f"Integrated load + eimage_sta: window_range={window_range}, "
        f"duration_s={duration_s}, spike_limit={spike_limit}, chunk_duration_s={chunk_duration_s}"
    )
    
    # =========================================================================
    # STEP 1: Load CMTR data (spike times) - only once
    # =========================================================================
    logger.info("Step 1: Loading CMTR data...")
    
    try:
        from McsPy import McsCMOSMEA
        cmtr_data = McsCMOSMEA.McsData(str(cmtr_path))
        spike_sorter = cmtr_data['Spike Sorter']
        cmtr_unit_keys = [key for key in spike_sorter.keys() if key.startswith('Unit')]
        logger.info(f"Found {len(cmtr_unit_keys)} units in CMTR: {cmtr_unit_keys[:5]}...")
    except ImportError:
        raise ImportError("McsPy library not available. Install with: pip install McsPyDataTools")
    except KeyError as e:
        raise ValueError(f"Failed to access Spike Sorter in CMTR: {e}")
    
    # Also load via standard API for HDF5 creation
    cmtr_result = load_cmtr_data(cmtr_path)
    units_data = cmtr_result["units"]
    
    # =========================================================================
    # STEP 2: Load CMCR info and get sampling rate
    # =========================================================================
    logger.info("Step 2: Opening CMCR and getting sensor info...")
    
    cmcr_data_mcspy = McsCMOSMEA.McsData(str(cmcr_path))
    sensor_stream = cmcr_data_mcspy['Acquisition']['Sensor Data']["SensorData 1 1"]
    
    total_samples_available = sensor_stream.shape[0]
    n_rows = sensor_stream.shape[1]
    n_cols = sensor_stream.shape[2]
    
    # Get sampling rate from CMCR
    cmcr_result = load_cmcr_data(cmcr_path)
    sampling_rate = cmcr_result.get("acquisition_rate", 20000.0)
    
    total_samples = min(int(sampling_rate * duration_s), total_samples_available)
    logger.info(f"Sensor data: shape=({total_samples_available}, {n_rows}, {n_cols}), "
               f"processing {total_samples} samples ({total_samples/sampling_rate:.1f}s)")
    
    # =========================================================================
    # STEP 3: Derive dataset_id and create HDF5
    # =========================================================================
    if dataset_id is None:
        dataset_id = derive_dataset_id(cmcr_path, cmtr_path)
        logger.info(f"Derived dataset_id: {dataset_id}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / f"{dataset_id}.h5"
    
    logger.info(f"Step 3: Creating HDF5 file: {hdf5_path}")
    
    # Create HDF5 file
    root = create_recording_hdf5(
        hdf5_path,
        dataset_id=dataset_id,
        config={},
        overwrite=force,
    )
    
    # Process units: compute firing rates and convert spike_times
    recording_duration_us = cmtr_result.get("metadata", {}).get("recording_duration_s", 0) * 1e6
    
    for unit_id, unit_info in units_data.items():
        spike_times = unit_info.get("spike_times")
        if spike_times is not None and len(spike_times) > 0:
            duration_us = recording_duration_us if recording_duration_us > 0 else spike_times[-1] + 1e6
            unit_info["firing_rate_10hz"] = compute_firing_rate(spike_times, duration_us, bin_rate_hz=10)
            unit_info["spike_times"] = _convert_spike_times_to_samples(spike_times, sampling_rate)
    
    # Write units to HDF5
    write_units(root, units_data)
    
    # Write metadata
    metadata = {
        "dataset_id": dataset_id,
        "acquisition_rate": sampling_rate,
        "sample_interval": 1.0 / sampling_rate,
    }
    write_metadata(root, metadata)
    write_source_files(root, cmcr_path, cmtr_path)
    
    # Close root to use h5py directly
    root.close()
    
    # =========================================================================
    # STEP 4: Prepare spike samples for STA computation
    # =========================================================================
    logger.info("Step 4: Preparing spike samples for STA...")
    
    # Determine which units to process
    if unit_ids is not None:
        units_to_process = unit_ids
        logger.info(f"Processing {len(units_to_process)} specified units")
    else:
        units_to_process = []
        for key in cmtr_unit_keys:
            parts = key.split()
            if len(parts) >= 2:
                units_to_process.append(parts[1])
        logger.info(f"Processing all {len(units_to_process)} units from CMTR")
    
    # Load spike samples for each unit from CMTR (already in memory)
    unit_spike_samples: Dict[str, np.ndarray] = {}
    total_spikes_for_division: Dict[str, int] = {}
    
    for unit_num in units_to_process:
        cmtr_unit_key = f"Unit {unit_num}"
        if cmtr_unit_key not in spike_sorter:
            warnings_list.append(f"Unit {unit_num} not found in CMTR")
            continue
        
        unit_data = spike_sorter[cmtr_unit_key]
        if "Peaks" not in unit_data:
            warnings_list.append(f"No Peaks data for Unit {unit_num}")
            continue
        
        # Apply spike_limit BEFORE conversion
        if spike_limit > 0:
            spike_data = unit_data["Peaks"][:spike_limit]
        else:
            spike_data = unit_data["Peaks"][:]
        
        # Convert to sample indices
        spike_time_stamp = [spike[0] / 1_000_000 * sampling_rate for spike in spike_data]
        spike_samples_arr = np.int64(spike_time_stamp)
        
        # Filter to spikes within duration
        valid_mask = spike_samples_arr < total_samples
        spike_samples_arr = spike_samples_arr[valid_mask]
        
        unit_spike_samples[unit_num] = spike_samples_arr
        total_spikes_for_division[unit_num] = len(spike_samples_arr)
        
        logger.debug(f"Unit {unit_num}: {len(spike_samples_arr)} spikes within duration")
    
    # =========================================================================
    # STEP 5: Chunk-based STA computation
    # =========================================================================
    logger.info("Step 5: Computing eimage_sta using chunk-based processing...")
    
    # Design filter once
    if not skip_highpass:
        nyquist = 0.5 * sampling_rate
        if cutoff_hz >= nyquist:
            raise ValueError(f"cutoff_hz ({cutoff_hz}) must be less than Nyquist ({nyquist})")
        normalized_cutoff = cutoff_hz / nyquist
        sos = butter(filter_order, normalized_cutoff, btype='high', analog=False, output='sos')
    
    # Initialize accumulators
    sta_accumulators: Dict[str, np.ndarray] = {}
    spike_counts: Dict[str, int] = {}
    
    for unit_num in unit_spike_samples.keys():
        sta_accumulators[unit_num] = np.zeros((window_length, n_rows, n_cols), dtype=np.float64)
        spike_counts[unit_num] = 0
    
    # Process chunks
    chunk_samples = int(sampling_rate * chunk_duration_s)
    overlap_samples = window_length
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    filter_time_total = 0.0
    
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
        chunk_start = chunk_idx * chunk_samples
        chunk_end = min(chunk_start + chunk_samples, total_samples)
        
        # Load with overlap for filtering
        load_start = max(0, chunk_start - overlap_samples)
        load_end = min(chunk_end + overlap_samples, total_samples)
        
        valid_start_in_chunk = chunk_start - load_start
        valid_end_in_chunk = valid_start_in_chunk + (chunk_end - chunk_start)
        
        # Load chunk
        chunk_data = sensor_stream[load_start:load_end, :, :]
        chunk_data = np.array(chunk_data).astype(np.int16)
        
        # Filter chunk
        filter_start = time.time()
        
        if skip_highpass:
            chunk_float = chunk_data.astype(np.float32)
            chunk_mean = chunk_float.mean(axis=0, keepdims=True)
            filtered_chunk = chunk_float - chunk_mean
        else:
            chunk_float = chunk_data.astype(np.float32)
            original_shape = chunk_float.shape
            reshaped = chunk_float.reshape(original_shape[0], -1)
            filtered_reshaped = sosfiltfilt(sos, reshaped, axis=0).astype(np.float32)
            filtered_chunk = filtered_reshaped.reshape(original_shape)
            del reshaped, filtered_reshaped
        
        filter_time_total += time.time() - filter_start
        
        del chunk_data, chunk_float
        
        # Extract valid portion
        filtered_valid = filtered_chunk[valid_start_in_chunk:valid_end_in_chunk, :, :]
        del filtered_chunk
        
        # Accumulate STA
        for unit_num, spike_samples_arr in unit_spike_samples.items():
            min_spike = chunk_start + pre_samples
            max_spike = chunk_end - post_samples
            
            chunk_spikes_mask = (spike_samples_arr >= min_spike) & (spike_samples_arr < max_spike)
            chunk_spikes = spike_samples_arr[chunk_spikes_mask]
            
            if len(chunk_spikes) == 0:
                continue
            
            local_spikes = chunk_spikes - chunk_start
            
            for local_spike in local_spikes:
                window_start = local_spike - pre_samples
                window_end = local_spike + post_samples
                
                if window_start >= 0 and window_end <= filtered_valid.shape[0]:
                    sta_accumulators[unit_num] += filtered_valid[window_start:window_end, :, :]
                    spike_counts[unit_num] += 1
        
        del filtered_valid
    
    # =========================================================================
    # STEP 6: Finalize and write eimage_sta to HDF5
    # =========================================================================
    logger.info("Step 6: Finalizing STAs and writing to HDF5...")
    
    config = EImageSTAConfig(
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        pre_samples=pre_samples,
        post_samples=post_samples,
        spike_limit=spike_limit,
        duration_s=duration_s,
        skip_highpass=skip_highpass,
        use_cache=False,
        cache_path=None,
        force=force,
    )
    
    units_with_sta = 0
    
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        for unit_num in tqdm(unit_spike_samples.keys(), desc="Saving eimage_sta"):
            total_for_div = total_spikes_for_division.get(unit_num, 0)
            n_used = spike_counts.get(unit_num, 0)
            n_excluded = total_for_div - n_used
            
            if total_for_div == 0:
                sta = np.full_like(sta_accumulators[unit_num], np.nan)
            else:
                sta = sta_accumulators[unit_num] / total_for_div
            
            unit_id = f"unit_{int(unit_num):03d}"
            
            written = write_eimage_sta_to_hdf5(
                hdf5_file,
                unit_id,
                sta,
                n_used,
                n_excluded,
                config,
                sampling_rate,
                force=force,
            )
            
            if written:
                units_with_sta += 1
    
    # Mark stage 1 complete
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        if "pipeline" not in hdf5_file:
            hdf5_file.create_group("pipeline")
        hdf5_file["pipeline"].attrs["stage1_completed"] = True
        hdf5_file["pipeline"].attrs["stage1_timestamp"] = datetime.now(timezone.utc).isoformat()
    
    elapsed = time.time() - start_time
    
    logger.info(
        f"Integrated load + eimage_sta complete: {len(units_data)} units loaded, "
        f"{units_with_sta} units with STA, {elapsed:.1f}s total, filter_time={filter_time_total:.1f}s"
    )
    
    return LoadWithEImageSTAResult(
        hdf5_path=hdf5_path,
        dataset_id=dataset_id,
        num_units=len(units_data),
        units_with_sta=units_with_sta,
        stage1_completed=True,
        elapsed_seconds=elapsed,
        filter_time_seconds=filter_time_total,
        warnings=warnings_list,
    )


# =============================================================================
# Stage 2: Feature Extraction
# =============================================================================

def extract_features(
    hdf5_path: Union[str, Path],
    features: List[str],
    *,
    force: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """
    Extract features from loaded HDF5 and write back to same archive.
    
    This is Stage 2 of the pipeline. Reads from HDF5, computes features
    for each unit, and writes results to units/{unit_id}/features/{feature_name}/.
    
    Args:
        hdf5_path: Path to HDF5 file from Stage 1.
        features: List of feature names to extract (must be registered).
        force: If True, overwrite existing features. Default: False.
        config_overrides: Optional parameter overrides for extractors.
    
    Returns:
        ExtractionResult with lists of extracted, skipped, and failed features.
    
    Raises:
        FileNotFoundError: If hdf5_path does not exist.
        KeyError: If any feature name is not registered.
        MissingInputError: If required inputs for a feature are missing.
    """
    from hdmea.features.registry import FeatureRegistry
    
    hdf5_path = Path(hdf5_path)
    warnings = []
    extracted = []
    skipped = []
    failed = []
    
    # Open HDF5
    root = open_recording_hdf5(hdf5_path, mode="r+")
    
    # Validate Stage 1 is complete
    status = get_stage1_status(root)
    if not status["completed"]:
        root.close()
        raise ConfigurationError(
            f"Stage 1 not complete for {hdf5_path}. Run load_recording first."
        )
    
    # Get unit list
    unit_ids = list_units(root)
    if not unit_ids:
        warnings.append("No units found in HDF5")
        root.close()
        return ExtractionResult(
            hdf5_path=hdf5_path,
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
            
            # Extract features in parallel, write sequentially (Windows zarr locking issue)
            stimulus_data = root["stimulus"]
            recording_metadata = root["metadata"] if "metadata" in root else None
            
            # Use 80% of CPU cores for parallel extraction
            cpu_count = os.cpu_count() or 4
            max_workers = max(1, min(int(cpu_count * 0.8), len(unit_ids)))
            
            # Filter units that need extraction
            units_to_extract = []
            for unit_id in unit_ids:
                existing = list_features(root, unit_id)
                if feature_name not in existing or force:
                    units_to_extract.append(unit_id)
            
            if not units_to_extract:
                logger.info(f"All units already have {feature_name} (cache hit)")
                skipped.append(feature_name)
                continue
            
            def extract_for_unit(unit_id: str) -> Tuple[str, Any, Optional[str]]:
                """Extract feature for a single unit. Returns (unit_id, result, error_msg)."""
                unit_data = root["units"][unit_id]
                
                try:
                    result = extractor.extract(
                        unit_data, stimulus_data, 
                        config=config_overrides,
                        metadata=recording_metadata
                    )
                    return unit_id, result, None
                except Exception as e:
                    return unit_id, None, str(e)
            
            # Phase 1: Compute extractions in parallel
            extraction_results: List[Tuple[str, Any, Optional[str]]] = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_for_unit, uid): uid 
                          for uid in units_to_extract}
                
                for future in tqdm(as_completed(futures), total=len(futures),
                                  desc=f"Extracting {feature_name}", leave=False):
                    extraction_results.append(future.result())
            
            # Phase 2: Write results sequentially (avoids Windows zarr locking)
            for unit_id, result, error in tqdm(extraction_results, 
                                               desc=f"Writing {feature_name}", leave=False):
                if error:
                    logger.error(f"Failed to extract {feature_name} for {unit_id}: {error}")
                    warnings.append(f"Failed {feature_name} for {unit_id}: {error}")
                    continue
                
                try:
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
                    logger.error(f"Failed to write {feature_name} for {unit_id}: {e}")
                    warnings.append(f"Failed to write {feature_name} for {unit_id}: {e}")
            
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
    
    # Close the HDF5 file
    root.close()
    
    return ExtractionResult(
        hdf5_path=hdf5_path,
        features_extracted=extracted,
        features_skipped=skipped,
        features_failed=failed,
        warnings=warnings,
    )
