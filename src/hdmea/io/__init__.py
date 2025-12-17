"""
I/O module for HD-MEA pipeline.

Handles:
    - CMCR/CMTR file reading via McsPy
    - Zarr store operations (read/write)
    - Section time loading (playlist-based and analog detection)
    - Parquet export
"""

from hdmea.io.section_time import add_section_time, add_section_time_analog

__all__ = [
    "add_section_time",
    "add_section_time_analog",
]

