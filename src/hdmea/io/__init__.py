"""
I/O module for HD-MEA pipeline.

Handles:
    - CMCR/CMTR file reading via McsPy
    - HDF5 store operations (read/write)
    - Section time loading (playlist-based and analog detection)
    - Parquet export
"""

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
from hdmea.io.section_time import add_section_time, add_section_time_analog
from hdmea.io.spike_sectioning import section_spike_times, SectionResult

__all__ = [
    # HDF5 store operations
    "create_recording_hdf5",
    "open_recording_hdf5",
    "write_units",
    "write_stimulus",
    "write_metadata",
    "write_source_files",
    "mark_stage1_complete",
    "get_stage1_status",
    "list_units",
    "list_features",
    "write_feature_to_unit",
    # Section time operations
    "add_section_time",
    "add_section_time_analog",
    "section_spike_times",
    "SectionResult",
]

