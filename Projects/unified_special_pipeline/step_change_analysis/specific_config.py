"""
Specific Configuration for Step Change Analysis Pipeline

This module contains configuration for analyzing step responses over time,
particularly for experiments where an agonist is applied during recording.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# Path Configuration
# =============================================================================

# Directory containing this config file
CONFIG_DIR = Path(__file__).parent

# Project root (Data_Processing_2027)
PROJECT_ROOT = CONFIG_DIR.parent.parent.parent

# Output directory for processed HDF5 files
OUTPUT_DIR = CONFIG_DIR / "output"

# Figures output directory
FIGURES_DIR = CONFIG_DIR / "figures"

# =============================================================================
# Input Data Configuration
# =============================================================================

# Default data folder for test files
DATA_FOLDER = Path("N:/20251022_low_glucose")

# Test input files (12 recordings in sequence)
# Files from N:\20251022_low_glucose (low glucose experiment)
TEST_FILES: List[Dict[str, str]] = [
    {
        "cmcr": "2025.10.22-11.03.24-Rec.cmcr",
        "cmtr": "2025.10.22-11.03.24-Rec-.cmtr",
        "description": "Recording 01 - Control",
    },
    {
        "cmcr": "2025.10.22-11.24.15-Rec.cmcr",
        "cmtr": "2025.10.22-11.24.15-Rec-.cmtr",
        "description": "Recording 02",
    },
    {
        "cmcr": "2025.10.22-11.44.16-Rec.cmcr",
        "cmtr": "2025.10.22-11.44.16-Rec-.cmtr",
        "description": "Recording 03",
    },
    {
        "cmcr": "2025.10.22-12.04.17-Rec.cmcr",
        "cmtr": "2025.10.22-12.04.17-Rec-.cmtr",
        "description": "Recording 04",
    },
    {
        "cmcr": "2025.10.22-12.24.19-Rec.cmcr",
        "cmtr": "2025.10.22-12.24.19-Rec-.cmtr",
        "description": "Recording 05",
    },
    {
        "cmcr": "2025.10.22-12.44.20-Rec.cmcr",
        "cmtr": "2025.10.22-12.44.20-Rec-.cmtr",
        "description": "Recording 06",
    },
    {
        "cmcr": "2025.10.22-13.09.29-Rec.cmcr",
        "cmtr": "2025.10.22-13.09.29-Rec-.cmtr",
        "description": "Recording 07",
    },
    {
        "cmcr": "2025.10.22-13.29.30-Rec.cmcr",
        "cmtr": "2025.10.22-13.29.30-Rec-.cmtr",
        "description": "Recording 08",
    },
    {
        "cmcr": "2025.10.22-13.49.31-Rec.cmcr",
        "cmtr": "2025.10.22-13.49.31-Rec-.cmtr",
        "description": "Recording 09",
    },
    {
        "cmcr": "2025.10.22-14.09.32-Rec.cmcr",
        "cmtr": "2025.10.22-14.09.32-Rec-.cmtr",
        "description": "Recording 10",
    },
    {
        "cmcr": "2025.10.22-14.29.33-Rec.cmcr",
        "cmtr": "2025.10.22-14.29.33-Rec-.cmtr",
        "description": "Recording 11",
    },
    {
        "cmcr": "2025.10.22-14.49.34-Rec.cmcr",
        "cmtr": "2025.10.22-14.49.34-Rec-.cmtr",
        "description": "Recording 12",
    },
]

# Treatment application time (seconds from start of first recording)
# Set to 0 for now - update based on experiment protocol
AGONIST_START_TIME_S: float = 0.0

# Recording interval (approximate time between recordings in minutes)
RECORDING_INTERVAL_MINUTES: float = 20.0


# =============================================================================
# Step Detection Configuration
# =============================================================================

@dataclass
class StepDetectionConfig:
    """Configuration for detecting light step stimuli."""
    
    # Threshold for detecting step onset in light reference signal
    # For low_glucose dataset: uses height=35000 (channel 1 signal)
    threshold_height: float = 35000.0
    
    # Minimum distance between peaks (in samples at 10Hz)
    # For low_glucose dataset: distance=3 (as in legacy code)
    min_peak_distance: int = 3
    
    # Range of repeats to include [start, end] (None means all)
    # Use [1, -1] to skip first and last incomplete steps
    repeat_range: Tuple[int, int] = (1, -1)
    
    # Time window around step onset (in samples at 10Hz = 100ms/sample)
    pre_margin: int = 10  # 1 second before
    post_margin: int = 50  # 5 seconds after
    
    # Firing rate sampling frequency
    firing_rate_hz: float = 10.0
    
    # Which light reference channel to use for step detection (1 or 2)
    # For low_glucose dataset: channel 1 (index 0 in legacy code)
    light_channel: int = 1


# =============================================================================
# Quality Index Configuration
# =============================================================================

@dataclass
class QualityConfig:
    """Configuration for quality index calculation and filtering."""
    
    # Minimum quality index threshold for "high quality" units
    # Lowered from 0.05 to 0.01 for this dataset
    quality_threshold: float = 0.01
    
    # Maximum trace value to exclude artifacts
    max_trace_value: float = 400.0


# =============================================================================
# Unit Alignment Configuration
# =============================================================================

@dataclass
class AlignmentConfig:
    """Configuration for aligning units across recordings."""
    
    # Weight for waveform similarity in alignment score
    waveform_weight: float = 10.0
    
    # Iteration distances for progressive alignment
    # Start strict (0 = same electrode), then expand
    iteration_distances: Tuple[int, ...] = (0, 1, 2)
    
    # Quality threshold for units to include in alignment
    # Should match QualityConfig.quality_threshold
    quality_threshold: float = 0.01
    
    # Fixed reference index for alignment (-1 = last recording)
    fixed_ref_index: Optional[int] = -1
    
    # Waveform weight for fixed reference alignment
    fixed_align_waveform_weight: float = 100.0
    
    # Iteration distances for fixed reference alignment
    fixed_align_iteration_distances: Tuple[int, ...] = (0, 1, 2)


# =============================================================================
# Response Analysis Configuration
# =============================================================================

@dataclass
class ResponseAnalysisConfig:
    """Configuration for response feature extraction."""
    
    # Baseline range (samples at 10Hz, relative to step onset)
    baseline_range: Tuple[int, int] = (0, 5)
    
    # Peak response range for ON response
    on_peak_range: Tuple[int, int] = (10, 20)
    
    # Peak response range for OFF response  
    off_peak_range: Tuple[int, int] = (40, 50)
    
    # Trial interval in seconds
    trial_interval_s: float = 10.0
    
    # Recording file interval in minutes
    file_interval_minutes: float = 20.0
    
    # Normalization mode: "max", "first", "no_normalize"
    normalize_mode: str = "first"


# =============================================================================
# Visualization Configuration
# =============================================================================

@dataclass
class VisualizationConfig:
    """Configuration for visualization plots."""
    
    # Number of rows for unit grid plots
    grid_rows: int = 10
    
    # Figure size for grid plots
    grid_figure_size: Tuple[int, int] = (30, 10)
    
    # Figure size for timecourse plots
    timecourse_figure_size: Tuple[int, int] = (8, 5)
    
    # Y-axis limits for response plots (None for auto)
    y_lim: Tuple[Optional[float], Optional[float]] = (None, None)
    
    # X-axis limits for response plots (None for auto)
    x_lim: Tuple[Optional[float], Optional[float]] = (0, None)
    
    # Bin size for averaging responses
    bin_size: int = 3
    
    # Colors for different groups
    control_color: str = "black"
    treatment_color: str = "red"
    error_alpha: float = 0.5


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    
    # Sub-configurations
    step_detection: StepDetectionConfig = field(default_factory=StepDetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    response_analysis: ResponseAnalysisConfig = field(default_factory=ResponseAnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Data paths
    data_folder: Path = DATA_FOLDER
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    
    # Agonist timing
    agonist_start_time_s: float = AGONIST_START_TIME_S
    recording_interval_minutes: float = RECORDING_INTERVAL_MINUTES


# =============================================================================
# Helper Functions
# =============================================================================

def get_cmcr_cmtr_paths(
    data_folder: Optional[Path] = None,
    file_info: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    """
    Get full paths to CMCR and CMTR files.
    
    Args:
        data_folder: Base data folder (defaults to DATA_FOLDER)
        file_info: Dictionary with 'cmcr' and 'cmtr' keys
    
    Returns:
        Tuple of (cmcr_path, cmtr_path)
    """
    if data_folder is None:
        data_folder = DATA_FOLDER
    
    if file_info is None:
        file_info = TEST_FILES[0]
    
    cmcr_path = data_folder / file_info["cmcr"]
    cmtr_path = data_folder / file_info["cmtr"]
    
    return cmcr_path, cmtr_path


def get_all_test_file_paths(
    data_folder: Optional[Path] = None,
) -> List[Tuple[Path, Path, str]]:
    """
    Get paths for all test files.
    
    Args:
        data_folder: Base data folder (defaults to DATA_FOLDER)
    
    Returns:
        List of (cmcr_path, cmtr_path, description) tuples
    """
    if data_folder is None:
        data_folder = DATA_FOLDER
    
    paths = []
    for file_info in TEST_FILES:
        cmcr_path = data_folder / file_info["cmcr"]
        cmtr_path = data_folder / file_info["cmtr"]
        paths.append((cmcr_path, cmtr_path, file_info["description"]))
    
    return paths


def get_output_hdf5_path(
    cmcr_filename: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate output HDF5 path from CMCR filename.
    
    Args:
        cmcr_filename: Name of the CMCR file
        output_dir: Output directory (defaults to OUTPUT_DIR)
    
    Returns:
        Path for the output HDF5 file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Replace .cmcr with .h5
    h5_filename = cmcr_filename.replace(".cmcr", ".h5")
    return output_dir / h5_filename


def get_grouped_hdf5_path(
    output_dir: Optional[Path] = None,
    suffix: str = "aligned_group",
) -> Path:
    """
    Generate path for grouped/aligned HDF5 file.
    
    Args:
        output_dir: Output directory (defaults to OUTPUT_DIR)
        suffix: Suffix for the filename
    
    Returns:
        Path for the grouped HDF5 file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use first and last recording names
    first_rec = TEST_FILES[0]["cmcr"].replace(".cmcr", "")
    last_rec = TEST_FILES[-1]["cmcr"].replace(".cmcr", "")
    
    return output_dir / f"{first_rec}_{last_rec}_{suffix}.h5"


# =============================================================================
# Default Configuration Instance
# =============================================================================

# Create default configuration instance
default_config = PipelineConfig()
