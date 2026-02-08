"""
Step Change Analysis Pipeline

This module provides tools for analyzing step responses over time,
particularly for experiments where an agonist or treatment is applied
during the recording session.

Key components:
- data_loader: Load CMCR/CMTR data and save to HDF5
- unit_alignment: Track units across multiple recordings
- response_analysis: Extract and analyze response features
- visualization: Generate analysis plots
- run_pipeline: Orchestrate the full workflow
"""

from .specific_config import (
    PipelineConfig,
    StepDetectionConfig,
    QualityConfig,
    AlignmentConfig,
    ResponseAnalysisConfig,
    VisualizationConfig,
    default_config,
    get_cmcr_cmtr_paths,
    get_all_test_file_paths,
    get_output_hdf5_path,
    get_grouped_hdf5_path,
)

from .data_loader import (
    load_and_save_recording,
    load_recording_from_hdf5,
    load_cmcr_cmtr_data,
    save_recording_to_hdf5,
    calculate_quality_index,
    detect_step_times,
    extract_step_responses,
    get_high_quality_units,
)

from .unit_alignment import (
    create_aligned_group,
    load_aligned_group_from_hdf5,
    save_aligned_group_to_hdf5,
    generate_alignment_links,
    build_alignment_chains,
    add_signatures_to_data,
)

from .response_analysis import (
    summarize_response_timecourse,
    get_all_trace_features,
    extract_response_feature,
    normalize_features,
    compute_binned_statistics,
    compute_treatment_effect,
)

from .visualization import (
    plot_analysis_summary,
    plot_response_timecourse,
    plot_step_responses_grid,
    plot_alignment_chains,
    plot_response_heatmap,
    plot_recording_summary,
)

from .run_pipeline import (
    run_full_pipeline,
    run_from_existing_hdf5,
    step1_load_recordings,
    step2_align_units,
    step3_analyze_responses,
    step4_generate_plots,
)

__all__ = [
    # Configuration
    "PipelineConfig",
    "StepDetectionConfig",
    "QualityConfig",
    "AlignmentConfig",
    "ResponseAnalysisConfig",
    "VisualizationConfig",
    "default_config",
    "get_cmcr_cmtr_paths",
    "get_all_test_file_paths",
    "get_output_hdf5_path",
    "get_grouped_hdf5_path",
    # Data loading
    "load_and_save_recording",
    "load_recording_from_hdf5",
    "load_cmcr_cmtr_data",
    "save_recording_to_hdf5",
    "calculate_quality_index",
    "detect_step_times",
    "extract_step_responses",
    "get_high_quality_units",
    # Unit alignment
    "create_aligned_group",
    "load_aligned_group_from_hdf5",
    "save_aligned_group_to_hdf5",
    "generate_alignment_links",
    "build_alignment_chains",
    "add_signatures_to_data",
    # Response analysis
    "summarize_response_timecourse",
    "get_all_trace_features",
    "extract_response_feature",
    "normalize_features",
    "compute_binned_statistics",
    "compute_treatment_effect",
    # Visualization
    "plot_analysis_summary",
    "plot_response_timecourse",
    "plot_step_responses_grid",
    "plot_alignment_chains",
    "plot_response_heatmap",
    "plot_recording_summary",
    # Pipeline
    "run_full_pipeline",
    "run_from_existing_hdf5",
    "step1_load_recordings",
    "step2_align_units",
    "step3_analyze_responses",
    "step4_generate_plots",
]
