"""
Pipeline orchestration module for HD-MEA pipeline.

Provides:
    - Pipeline runner with caching
    - Flow definitions and execution
    - Configuration loading
    - Section time loading for visual stimulation experiments
    - PipelineSession for deferred HDF5 saving
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

from hdmea.pipeline.session import (
    PipelineSession,
    SaveState,
    create_session,
)

from hdmea.io.section_time import add_section_time

__all__ = [
    # Runner functions
    "load_recording",
    "load_recording_with_eimage_sta",
    "extract_features",
    # Result types
    "LoadResult",
    "LoadWithEImageSTAResult",
    "ExtractionResult",
    "FlowResult",
    # Flow functions
    "run_flow",
    "list_available_flows",
    "get_flow_info",
    # Section time
    "add_section_time",
    # Session (deferred save)
    "PipelineSession",
    "SaveState",
    "create_session",
]

