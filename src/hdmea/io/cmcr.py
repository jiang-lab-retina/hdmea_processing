"""
CMCR file reading via McsPy.

CMCR files contain raw sensor data and light reference from HD-MEA recordings.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import h5py
import numpy as np

from hdmea.utils.exceptions import DataLoadError


logger = logging.getLogger(__name__)


def load_cmcr_data(cmcr_path: Path) -> Dict[str, Any]:
    """
    Load raw sensor data from CMCR file.
    
    Extracts light reference data which is used for stimulus timing alignment.
    
    Args:
        cmcr_path: Path to .cmcr file (may be UNC path)
    
    Returns:
        Dictionary with keys:
            - light_reference: Dict with six entries:
                - raw_ch1, 10hz_ch1, 1khz_ch1 (channel 1 at raw/10Hz/1kHz)
                - raw_ch2, 10hz_ch2, 1khz_ch2 (channel 2 at raw/10Hz/1kHz)
            - metadata: Recording metadata
            - acquisition_rate: Sampling rate in Hz
    
    Raises:
        FileNotFoundError: If cmcr_path does not exist
        DataLoadError: If file cannot be read
    """
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    logger.info(f"Loading CMCR file: {cmcr_path}")
    
    try:
        # Import McsPy here to allow graceful failure if not installed
        from McsPy.McsCMOSMEA import McsCMOSMEAData
        
        cmcr_data = McsCMOSMEAData(str(cmcr_path))
        
        light_reference = {}
        metadata = {}
        
        # Extract file attributes as metadata
        if hasattr(cmcr_data, "attrs"):
            for key in cmcr_data.attrs.keys():
                try:
                    val = cmcr_data.attrs[key]
                    # Decode bytes to string
                    if isinstance(val, bytes):
                        metadata[key] = val.decode('utf-8', errors='ignore')
                    elif hasattr(val, '__len__') and len(val) == 1:
                        metadata[key] = val[0]
                    else:
                        metadata[key] = val
                except Exception:
                    pass
        
        # Get recording duration from attributes (in microseconds)
        recording_duration_us = metadata.get("LB.RecordingDuration", 0)
        if recording_duration_us > 0:
            metadata["recording_duration_s"] = recording_duration_us / 1e6
        
        # Default acquisition rate (20 kHz typical for MaxOne/MaxTwo)
        acquisition_rate = 20000.0
        
        # Extract light reference from Analog_Data
        # Light reference is typically stored in Acquisition.Analog_Data
        if hasattr(cmcr_data, "Acquisition") and cmcr_data.Acquisition is not None:
            acq = cmcr_data.Acquisition
            
            if hasattr(acq, "Analog_Data") and acq.Analog_Data is not None:
                analog_data = acq.Analog_Data
                
                # Try to get ChannelData_1 which contains analog channels
                if hasattr(analog_data, "ChannelData_1"):
                    try:
                        channel_data = analog_data.ChannelData_1[:]
                        
                        # channel_data shape is typically (num_channels, num_samples)
                        if channel_data is not None and len(channel_data) > 0:
                            # Use first channel as light reference
                            if channel_data.ndim == 2:
                                light_data = channel_data[0].astype(np.float32)
                            else:
                                light_data = channel_data.astype(np.float32)
                            
                            # Estimate acquisition rate from data and duration
                            if recording_duration_us > 0:
                                acquisition_rate = len(light_data) / (recording_duration_us / 1e6)
                                logger.info(f"Estimated acquisition rate: {acquisition_rate:.0f} Hz")
                            
                            # Store channel 1 at all sample rates
                            light_reference["raw_ch1"] = light_data
                            light_reference["10hz_ch1"] = _downsample_light_reference(
                                light_data, acquisition_rate, target_rate=10
                            )
                            light_reference["1khz_ch1"] = _downsample_light_reference(
                                light_data, acquisition_rate, target_rate=1000
                            )
                            
                            logger.info(f"Loaded light reference ch1: {len(light_data)} samples from Analog_Data")
                            
                            # Store channel 2 at all sample rates if available
                            if channel_data.ndim == 2 and channel_data.shape[0] > 1:
                                light_data_ch2 = channel_data[1].astype(np.float32)
                                light_reference["raw_ch2"] = light_data_ch2
                                light_reference["10hz_ch2"] = _downsample_light_reference(
                                    light_data_ch2, acquisition_rate, target_rate=10
                                )
                                light_reference["1khz_ch2"] = _downsample_light_reference(
                                    light_data_ch2, acquisition_rate, target_rate=1000
                                )
                                logger.info(f"Loaded light reference ch2: {len(light_data_ch2)} samples")
                                
                    except Exception as e:
                        logger.warning(f"Could not extract light reference from ChannelData_1: {e}")
                
                # Also check for Data attribute
                elif hasattr(analog_data, "Data"):
                    try:
                        # Data might be a McsStreamList
                        data_list = analog_data.Data
                        if hasattr(data_list, '__iter__'):
                            for i, data_item in enumerate(data_list):
                                if hasattr(data_item, 'shape'):
                                    light_data = data_item[:].astype(np.float32)
                                    light_reference[f"channel_{i}"] = light_data
                                    logger.info(f"Loaded light reference channel {i}: {len(light_data)} samples")
                    except Exception as e:
                        logger.warning(f"Could not extract light reference from Data: {e}")
        
        if not light_reference:
            logger.warning("No light reference data found in CMCR file")
        
        return {
            "light_reference": light_reference,
            "metadata": metadata,
            "acquisition_rate": acquisition_rate,
            "source_path": str(cmcr_path),
        }
        
    except ImportError:
        raise DataLoadError(
            "McsPy library not available. Install with: pip install McsPyDataTools",
            file_path=str(cmcr_path),
        )
    except Exception as e:
        logger.error(f"Failed to read CMCR: {cmcr_path}: {e}")
        raise DataLoadError(
            f"Cannot read CMCR file: {cmcr_path}",
            file_path=str(cmcr_path),
            original_error=e,
        )


def _downsample_light_reference(
    data: np.ndarray,
    source_rate: float,
    target_rate: float,
) -> np.ndarray:
    """
    Downsample light reference to target rate.
    
    Args:
        data: Source data array
        source_rate: Source sampling rate (Hz)
        target_rate: Target sampling rate (Hz)
    
    Returns:
        Downsampled array
    """
    if target_rate >= source_rate:
        return data
    
    factor = int(source_rate / target_rate)
    
    # Use reshape and mean for downsampling
    n_samples = (len(data) // factor) * factor
    reshaped = data[:n_samples].reshape(-1, factor)
    downsampled = reshaped.mean(axis=1).astype(np.float32)
    
    return downsampled


def load_sensor_data(
    cmcr_path: Path,
    duration_samples: Optional[int] = None,
    duration_s: Optional[float] = None,
) -> np.ndarray:
    """
    Load sensor data from CMCR file using direct h5py access (faster than McsPy).
    
    Uses h5py with optimized chunk cache for faster I/O. Falls back to McsPy
    if direct HDF5 access fails (for non-standard CMCR file structures).
    
    Args:
        cmcr_path: Path to CMCR file.
        duration_samples: Number of time samples to load (None = all).
        duration_s: Duration in seconds to load (alternative to duration_samples).
    
    Returns:
        3D array (time, rows, cols) as int16.
    
    Raises:
        FileNotFoundError: If CMCR file not found.
        DataLoadError: If sensor data not available in file.
    """
    cmcr_path = Path(cmcr_path)
    
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    logger.info(f"Loading sensor data from CMCR: {cmcr_path}")
    
    # Try direct h5py access first (faster than McsPy)
    try:
        return _load_sensor_data_h5py(cmcr_path, duration_samples, duration_s)
    except (KeyError, OSError) as e:
        logger.warning(f"Direct h5py access failed: {e}, falling back to McsPy")
        return _load_sensor_data_mcspy(cmcr_path, duration_samples, duration_s)


def _load_sensor_data_h5py(
    cmcr_path: Path,
    duration_samples: Optional[int] = None,
    duration_s: Optional[float] = None,
) -> np.ndarray:
    """
    Load sensor data using direct h5py access (faster than McsPy wrapper).
    
    Opens file with optimized HDF5 chunk cache for sequential reads.
    """
    # Open with optimized chunk cache (100MB) for faster sequential reading
    with h5py.File(cmcr_path, 'r', rdcc_nbytes=100*1024*1024) as f:
        # Direct HDF5 path (standard CMCR structure)
        # Try common path patterns
        dataset = None
        tried_paths = []
        
        for path in [
            'Acquisition/Sensor Data/SensorData 1 1',
            'Acquisition/Sensor Data/SensorData_1_1',
            'Acquisition/Sensor Data/SensorData1',
        ]:
            tried_paths.append(path)
            if path in f:
                dataset = f[path]
                logger.info(f"Found sensor data via h5py: {path}")
                break
        
        # If specific paths not found, search for SensorData in Sensor Data group
        if dataset is None and 'Acquisition/Sensor Data' in f:
            sensor_group = f['Acquisition/Sensor Data']
            for key in sensor_group.keys():
                if 'SensorData' in key or 'sensor' in key.lower():
                    dataset = sensor_group[key]
                    logger.info(f"Found sensor data via h5py search: Acquisition/Sensor Data/{key}")
                    break
        
        if dataset is None:
            raise KeyError(f"Sensor data not found in CMCR. Tried paths: {tried_paths}")
        
        # Get shape info
        total_samples = dataset.shape[0]
        logger.info(f"Sensor data shape: {dataset.shape}")
        
        # Determine how many samples to load
        if duration_samples is not None:
            n_samples = min(duration_samples, total_samples)
        elif duration_s is not None:
            # Default acquisition rate 20kHz
            acq_rate = 20000.0
            n_samples = min(int(duration_s * acq_rate), total_samples)
        else:
            n_samples = total_samples
        
        # Direct slice - h5py handles chunked reading efficiently
        logger.info(f"Loading {n_samples} samples of sensor data (h5py direct)...")
        sensor_array = dataset[:n_samples, :, :].astype(np.int16)
        
        logger.info(
            f"Loaded sensor data: shape={sensor_array.shape}, dtype={sensor_array.dtype}"
        )
        
        return sensor_array


def _load_sensor_data_mcspy(
    cmcr_path: Path,
    duration_samples: Optional[int] = None,
    duration_s: Optional[float] = None,
) -> np.ndarray:
    """
    Load sensor data using McsPy (fallback for non-standard CMCR files).
    """
    try:
        from McsPy.McsCMOSMEA import McsCMOSMEAData
        
        cmcr_data = McsCMOSMEAData(str(cmcr_path))
        
        # Navigate to sensor data
        if not hasattr(cmcr_data, "Acquisition") or cmcr_data.Acquisition is None:
            raise DataLoadError(
                "No Acquisition group found in CMCR file",
                file_path=str(cmcr_path),
            )
        
        acq = cmcr_data.Acquisition
        sensor_stream = None
        
        # Try dictionary-style access first
        try:
            sensor_data_group = cmcr_data['Acquisition']['Sensor Data']
            
            for key in ['SensorData 1 1', 'SensorData_1_1', 'SensorData1']:
                if key in sensor_data_group:
                    sensor_stream = sensor_data_group[key]
                    logger.info(f"Found sensor data stream via McsPy dict: {key}")
                    break
            
            if sensor_stream is None:
                available_keys = list(sensor_data_group.keys()) if hasattr(sensor_data_group, 'keys') else []
                for key in available_keys:
                    if 'sensor' in key.lower() or 'data' in key.lower():
                        sensor_stream = sensor_data_group[key]
                        logger.info(f"Found sensor data stream via McsPy: {key}")
                        break
                        
        except (KeyError, TypeError) as e:
            logger.debug(f"McsPy dictionary access failed: {e}, trying attribute access")
            
            if hasattr(acq, "Sensor_Data") and acq.Sensor_Data is not None:
                sensor_data_group = acq.Sensor_Data
                
                for attr_name in dir(sensor_data_group):
                    if attr_name.startswith("SensorData") and not attr_name.startswith("_"):
                        candidate = getattr(sensor_data_group, attr_name)
                        if hasattr(candidate, 'shape') or hasattr(candidate, '__getitem__'):
                            sensor_stream = candidate
                            logger.info(f"Found sensor data stream via McsPy attr: {attr_name}")
                            break
        
        if sensor_stream is None:
            raise DataLoadError(
                "No SensorData stream found in CMCR file.",
                file_path=str(cmcr_path),
            )
        
        # Get data shape
        if hasattr(sensor_stream, 'shape'):
            data_shape = sensor_stream.shape
        else:
            probe = sensor_stream[:10, :, :]
            data_shape = (len(sensor_stream), probe.shape[1], probe.shape[2])
        
        logger.info(f"Sensor data shape: {data_shape}")
        
        # Determine samples to load
        total_samples = data_shape[0]
        
        if duration_samples is not None:
            n_samples = min(duration_samples, total_samples)
        elif duration_s is not None:
            acq_rate = 20000.0
            n_samples = min(int(duration_s * acq_rate), total_samples)
        else:
            n_samples = total_samples
        
        logger.info(f"Loading {n_samples} samples of sensor data (McsPy)...")
        sensor_array = sensor_stream[:n_samples, :, :]
        sensor_array = np.asarray(sensor_array).astype(np.int16)
        
        logger.info(
            f"Loaded sensor data: shape={sensor_array.shape}, dtype={sensor_array.dtype}"
        )
        
        return sensor_array
        
    except ImportError:
        raise DataLoadError(
            "McsPy library not available. Install with: pip install McsPyDataTools",
            file_path=str(cmcr_path),
        )
    except DataLoadError:
        raise
    except Exception as e:
        logger.error(f"Failed to load sensor data from CMCR: {cmcr_path}: {e}")
        raise DataLoadError(
            f"Cannot load sensor data from CMCR file: {cmcr_path}",
            file_path=str(cmcr_path),
            original_error=e,
        )


def load_sensor_data_chunked(
    cmcr_path: Path,
    chunk_duration_s: float = 60.0,
    sampling_rate: float = 20000.0,
    max_samples: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, int, int, int], None, None]:
    """
    Generator that yields sensor data in memory-efficient chunks.
    
    This enables processing of large CMCR files (100GB+) that exceed RAM capacity
    by loading one chunk at a time.
    
    Args:
        cmcr_path: Path to CMCR file.
        chunk_duration_s: Duration of each chunk in seconds (default 60s).
        sampling_rate: Acquisition rate in Hz (default 20kHz).
        max_samples: Maximum total samples to yield (None = all data).
    
    Yields:
        Tuple of (chunk_data, start_sample, end_sample, total_samples):
            - chunk_data: 3D array (time, rows, cols) as int16
            - start_sample: Global sample index where this chunk starts
            - end_sample: Global sample index where this chunk ends (exclusive)
            - total_samples: Total samples to be processed (respects max_samples)
    
    Raises:
        FileNotFoundError: If CMCR file not found.
        DataLoadError: If sensor data not available in file.
    
    Example:
        >>> for chunk, start, end, total in load_sensor_data_chunked(path, max_samples=2400000):
        ...     # Process chunk (e.g., filter, compute STA)
        ...     process(chunk)
        ...     del chunk  # Free memory before next iteration
    """
    cmcr_path = Path(cmcr_path)
    
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    chunk_samples = int(chunk_duration_s * sampling_rate)
    
    logger.info(
        f"Opening CMCR for chunked reading: {cmcr_path} "
        f"(chunk_size={chunk_samples} samples = {chunk_duration_s}s)"
    )
    
    # Open with optimized chunk cache
    with h5py.File(cmcr_path, 'r', rdcc_nbytes=100*1024*1024) as f:
        # Find sensor data dataset
        dataset = None
        tried_paths = []
        
        for path in [
            'Acquisition/Sensor Data/SensorData 1 1',
            'Acquisition/Sensor Data/SensorData_1_1',
            'Acquisition/Sensor Data/SensorData1',
        ]:
            tried_paths.append(path)
            if path in f:
                dataset = f[path]
                logger.info(f"Found sensor data for chunked access: {path}")
                break
        
        # Search if specific paths not found
        if dataset is None and 'Acquisition/Sensor Data' in f:
            sensor_group = f['Acquisition/Sensor Data']
            for key in sensor_group.keys():
                if 'SensorData' in key or 'sensor' in key.lower():
                    dataset = sensor_group[key]
                    logger.info(f"Found sensor data: Acquisition/Sensor Data/{key}")
                    break
        
        if dataset is None:
            raise DataLoadError(
                f"Sensor data not found in CMCR. Tried paths: {tried_paths}",
                file_path=str(cmcr_path),
            )
        
        total_samples_available = dataset.shape[0]
        n_rows = dataset.shape[1]
        n_cols = dataset.shape[2]
        
        # Limit to max_samples if specified
        if max_samples is not None and max_samples > 0:
            total_samples = min(max_samples, total_samples_available)
            logger.info(
                f"Sensor data: ({total_samples_available}, {n_rows}, {n_cols}) available, "
                f"limiting to {total_samples} samples ({total_samples/sampling_rate:.1f}s)"
            )
        else:
            total_samples = total_samples_available
        
        n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        
        logger.info(
            f"Will yield {n_chunks} chunks of {chunk_duration_s}s each"
        )
        
        # Yield chunks
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_samples
            end = min(start + chunk_samples, total_samples)
            
            load_start = time.time()
            chunk_data = dataset[start:end, :, :].astype(np.int16)
            load_time = time.time() - load_start
            
            logger.debug(
                f"Loaded chunk {chunk_idx+1}/{n_chunks}: samples [{start}:{end}], "
                f"shape={chunk_data.shape}, load_time={load_time:.2f}s"
            )
            
            yield chunk_data, start, end, total_samples


def get_sensor_data_info(cmcr_path: Path) -> Dict[str, Any]:
    """
    Get sensor data shape and metadata without loading the full data.
    
    Args:
        cmcr_path: Path to CMCR file.
    
    Returns:
        Dictionary with keys:
            - shape: Tuple (n_samples, n_rows, n_cols)
            - dtype: Data type of the sensor data
            - n_samples: Total number of time samples
            - n_electrodes: Total number of electrodes (rows * cols)
            - duration_s: Estimated duration in seconds (at 20kHz)
    
    Raises:
        FileNotFoundError: If CMCR file not found.
        DataLoadError: If sensor data not available.
    """
    cmcr_path = Path(cmcr_path)
    
    if not cmcr_path.exists():
        raise FileNotFoundError(f"CMCR file not found: {cmcr_path}")
    
    with h5py.File(cmcr_path, 'r') as f:
        dataset = None
        
        for path in [
            'Acquisition/Sensor Data/SensorData 1 1',
            'Acquisition/Sensor Data/SensorData_1_1',
            'Acquisition/Sensor Data/SensorData1',
        ]:
            if path in f:
                dataset = f[path]
                break
        
        if dataset is None and 'Acquisition/Sensor Data' in f:
            sensor_group = f['Acquisition/Sensor Data']
            for key in sensor_group.keys():
                if 'SensorData' in key:
                    dataset = sensor_group[key]
                    break
        
        if dataset is None:
            raise DataLoadError(
                "Sensor data not found in CMCR",
                file_path=str(cmcr_path),
            )
        
        shape = dataset.shape
        return {
            "shape": shape,
            "dtype": dataset.dtype,
            "n_samples": shape[0],
            "n_rows": shape[1],
            "n_cols": shape[2],
            "n_electrodes": shape[1] * shape[2],
            "duration_s": shape[0] / 20000.0,  # Assume 20kHz
        }


def find_cmcr_file_from_cmtr(cmtr_path: Path) -> Optional[Path]:
    """
    Try to find matching CMCR file for a given CMTR file.
    
    Searches in the same directory for a file with matching base name
    but .cmcr extension.
    
    Args:
        cmtr_path: Path to .cmtr file
    
    Returns:
        Path to matching .cmcr file if found, None otherwise
    """
    # Try exact stem match
    cmcr_path = cmtr_path.with_suffix(".cmcr")
    if cmcr_path.exists():
        return cmcr_path
    
    # Try common variations
    stem = cmtr_path.stem
    directory = cmtr_path.parent
    
    for suffix in ["_sorted", "_spikes"]:
        if stem.endswith(suffix):
            base_stem = stem[: -len(suffix)]
            candidate = directory / f"{base_stem}.cmcr"
            if candidate.exists():
                return candidate
    
    # Search for any .cmcr file with similar name
    for cmcr_file in directory.glob("*.cmcr"):
        if cmcr_file.stem.startswith(stem[:10]):  # Match first 10 chars
            logger.info(f"Found potential matching CMCR: {cmcr_file}")
            return cmcr_file
    
    return None

