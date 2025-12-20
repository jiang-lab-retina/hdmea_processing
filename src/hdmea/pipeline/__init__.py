"""
Pipeline orchestration module for HD-MEA pipeline.

Provides:
    - Pipeline runner with caching
    - Flow definitions and execution
    - Configuration loading
    - Section time loading for visual stimulation experiments
"""

from hdmea.pipeline.runner import (
    load_recording,
    load_recording_with_eimage_sta,
    extract_features,
    LoadResult,
    LoadWithEImageSTAResult,
    ExtractionResult,
    FlowResult,
)

from hdmea.pipeline.flows import (
    run_flow,
    list_available_flows,
    get_flow_info,
)

from hdmea.io.section_time import add_section_time

__all__ = [
    "load_recording",
    "load_recording_with_eimage_sta",
    "extract_features",
    "LoadResult",
    "LoadWithEImageSTAResult",
    "ExtractionResult",
    "FlowResult",
    "run_flow",
    "list_available_flows",
    "get_flow_info",
    "add_section_time",
]

