"""
Configuration Constants for Unified Pipeline

This module contains all configuration constants, paths, and parameters
used by the unified pipeline.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging for the pipeline."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


# =============================================================================
# Path Configuration
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Source data paths
CSV_MAPPING_PATH = PROJECT_ROOT / "tool_box/generate_data_path_list/pkl_to_cmtr_mapping.csv"

# Manual label folder
MANUAL_LABEL_FOLDER = Path(r"M:\Python_Project\Data_Processing_2024\manual_label_data")

# Google Sheet configuration
GSHEET_CREDENTIALS_PATH = PROJECT_ROOT / "credentials" / "vibrant-epsilon-169702-467fddc26dfc.json"
GSHEET_NAME = "MEA dashboard"  # Actual sheet name accessible by service account
GSHEET_CSV_CACHE_PATH = PROJECT_ROOT / "Projects/load_gsheet/gsheet_table.csv"

# AP Tracking model
AP_TRACKING_MODEL_PATH = PROJECT_ROOT / "Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth"

# DSGC configuration
DSGC_MOVIE_NAME = "moving_h_bar_s5_d8_3x"
DSGC_ON_OFF_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)
DSGC_PADDING_FRAMES = 10

# Output directories
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "test_output"
REFERENCE_OUTPUT_DIR = PROJECT_ROOT / "Projects/dsgc_section/export_dsgc_section_20251226"


# =============================================================================
# Pipeline Parameters
# =============================================================================

@dataclass
class LoadRecordingConfig:
    """Configuration for load_recording_with_eimage_sta."""
    duration_s: float = 120.0
    spike_limit: int = 10000
    window_range: Tuple[int, int] = (-10, 40)


@dataclass
class SectionTimeConfig:
    """Configuration for section time operations."""
    playlist_name: str = "play_optimization_set6_ipRGC_manual"
    pad_margin: Tuple[float, float] = (0.0, 0.0)
    cover_range: Tuple[int, int] = (-60, 0)


@dataclass
class SectionTimeAnalogConfig:
    """
    Configuration for analog section time detection.
    
    Used by add_section_time_analog_step to detect stimulus onsets
    from raw light reference signal (raw_ch1).
    """
    threshold_value: float = 1e5  # Peak height threshold for find_peaks() on signal derivative
    movie_name: str = "iprgc_test"  # Identifier for this stimulus type
    plot_duration: float = 120.0  # Duration of each section in seconds
    min_peak_distance_margin: float = 0.8  # Fraction of plot_duration for min peak spacing
    pad_margin: Tuple[float, float] = (2.0, 0.0)  # (pre_s, post_s) padding for spike sectioning
    repeat: Optional[int] = None  # Limit to first N detected trials (None = all)
    force: bool = False  # Overwrite existing section_time if exists


@dataclass
class GeometryConfig:
    """Configuration for geometry extraction."""
    frame_range: Tuple[int, int] = (10, 14)
    threshold_fraction: float = 0.5


@dataclass
class APTrackingConfig:
    """Configuration for AP tracking."""
    model_path: Path = AP_TRACKING_MODEL_PATH
    filter_by_cell_type: bool = True
    cell_type_filter: str = "rgc"
    force_cpu: bool = False
    min_points_for_fit: int = 10
    r2_threshold: float = 0.8
    max_displacement: int = 100
    centroid_start_frame: int = 10
    max_displacement_post: float = 5.0
    centroid_exclude_fraction: float = 0.1
    min_remaining_fraction: float = 0.2
    fix_bad_lanes: bool = True
    # Enhanced intersection parameters
    direction_tolerance: float = 30.0  # degrees
    max_distance_from_center: float = 50.0  # pixels
    center_point: tuple = (32.5, 32.5)  # center of 65x65 grid


@dataclass
class DSGCConfig:
    """Configuration for DSGC section by direction."""
    movie_name: str = DSGC_MOVIE_NAME
    on_off_dict_path: Path = DSGC_ON_OFF_DICT_PATH
    padding_frames: int = DSGC_PADDING_FRAMES


# =============================================================================
# Test Configuration
# =============================================================================

# Test file for validation
TEST_DATASET_ID = "2024.08.08-10.40.20-Rec"
TEST_REFERENCE_FILE = REFERENCE_OUTPUT_DIR / f"{TEST_DATASET_ID}.h5"


# =============================================================================
# Progress Bar Utilities
# =============================================================================

def get_tqdm():
    """
    Get tqdm with fallback to simple iterator if tqdm not available.
    
    Returns:
        tqdm function or identity function
    """
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        def identity(iterable, *args, **kwargs):
            return iterable
        return identity


def progress_bar(iterable, desc: str = "", total: Optional[int] = None):
    """
    Wrap iterable with progress bar.
    
    Args:
        iterable: Iterable to wrap
        desc: Description for progress bar
        total: Total count (auto-detected if iterable has __len__)
    
    Returns:
        Wrapped iterable with progress bar
    """
    tqdm = get_tqdm()
    if total is None and hasattr(iterable, '__len__'):
        total = len(iterable)
    return tqdm(iterable, desc=desc, total=total)


# =============================================================================
# Color Output Utilities
# =============================================================================

def red_warning(message: str) -> str:
    """
    Format message with red color for warnings.
    
    Uses colorama if available, falls back to plain text.
    
    Args:
        message: Warning message
    
    Returns:
        Colored message string
    """
    try:
        from colorama import Fore, Style
        return f"{Fore.RED}{message}{Style.RESET_ALL}"
    except ImportError:
        return f"[WARNING] {message}"


def green_success(message: str) -> str:
    """
    Format message with green color for success.
    
    Args:
        message: Success message
    
    Returns:
        Colored message string
    """
    try:
        from colorama import Fore, Style
        return f"{Fore.GREEN}{message}{Style.RESET_ALL}"
    except ImportError:
        return f"[SUCCESS] {message}"

