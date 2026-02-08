"""
Specific Configuration for Set6a Image Alignment Pipeline

This module contains configuration overrides specific to the set6a_image_alignment
batch processing pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

# =============================================================================
# Path Configuration
# =============================================================================

# Directory containing this config file
CONFIG_DIR = Path(__file__).parent

# Excel file with input data
EXCEL_PATH = CONFIG_DIR / "Yan Dulce Corresponding Data.xlsx"

# Output directory for processed HDF5 files
OUTPUT_DIR = CONFIG_DIR / "export"

# =============================================================================
# Drive Mappings
# =============================================================================

# Map network share names to Windows drive letters
DRIVE_MAP: Dict[str, str] = {
    "fs_3_1_data": "O:",
    "fs_3_2_data": "N:",
}

# =============================================================================
# Excel Column Names
# =============================================================================

# Column names in the Excel file
COL_DATA_LOCATION = "MEA Data Location"
COL_DATA_FOLDER = "MEA Data Folder"
COL_DATA_FILENAME = "MEA Data File Name"

# =============================================================================
# Pipeline Configuration Overrides
# =============================================================================

# Playlist name for section time (set6a uses green_blue_3s_3i_3x_64_128_255)
PLAYLIST_NAME = "play_optimization_set6_a_ipRGC_manual"


@dataclass
class SectionTimeConfigOverride:
    """Configuration for section time operations - specific to this pipeline."""
    playlist_name: str = PLAYLIST_NAME
    pad_margin: Tuple[float, float] = (0.0, 0.0)
    cover_range: Tuple[int, int] = (-60, 0)


@dataclass
class Set6CompatibilityConfig:
    """Configuration for set6 compatibility layer.
    
    Creates green_blue_3s_3i_3x from the last N repeats of green_blue_3s_3i_3x_64_128_255.
    """
    source_movie: str = "green_blue_3s_3i_3x_64_128_255"
    target_movie: str = "green_blue_3s_3i_3x"
    repeat_slice: Tuple[Optional[int], Optional[int]] = (-3, None)  # Last 3 repeats


@dataclass
class SectionTimeAnalogConfigOverride:
    """
    Configuration for analog section time detection.
    
    Used by add_section_time_analog_step to detect stimulus onsets
    from raw light reference signal (raw_ch1).
    """
    threshold_value: float = 1e5
    movie_name: str = "iprgc_test"
    plot_duration: float = 120.0
    min_peak_distance_margin: float = 0.8
    pad_margin: Tuple[float, float] = (2.0, 0.0)
    repeat: Optional[int] = None
    force: bool = False


# =============================================================================
# Helper Functions
# =============================================================================

def _search_file_in_subfolders(base_path: Path, filename_pattern: str) -> Optional[Path]:
    """
    Search for a file in base_path and its subfolders.
    
    Args:
        base_path: Directory to start searching from
        filename_pattern: Filename to search for (e.g., "2024.02.26-10.53.19-Rec.cmcr")
    
    Returns:
        Path to the found file, or None if not found
    """
    if not base_path.exists():
        return None
    
    # First check directly in base_path
    direct_path = base_path / filename_pattern
    if direct_path.exists():
        return direct_path
    
    # Search in subfolders (one level deep)
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            candidate = subfolder / filename_pattern
            if candidate.exists():
                return candidate
    
    # Search recursively (all levels) if not found
    for candidate in base_path.rglob(filename_pattern):
        return candidate  # Return first match
    
    return None


def _normalize_filename(filename: str) -> str:
    """
    Normalize filename to standard format with dashes.
    
    Converts formats like "2025.09.09.11.17.40.Rec" to "2025.09.09-11.17.40-Rec"
    by replacing the 3rd and 4th periods with dashes.
    
    Args:
        filename: Original filename
    
    Returns:
        Normalized filename
    """
    parts = filename.split('.')
    
    # Standard format: YYYY.MM.DD-HH.MM.SS-Rec
    # If we have 6 parts (date.date.date.time.time.time) + Rec, normalize
    if len(parts) >= 6:
        # Check if this looks like all-period format (e.g., 2025.09.09.11.17.40.Rec)
        # Standard format has dashes: 2025.09.09-11.17.40-Rec
        if '-' not in filename:
            # Convert: YYYY.MM.DD.HH.MM.SS.Rec -> YYYY.MM.DD-HH.MM.SS-Rec
            date_part = '.'.join(parts[0:3])  # 2025.09.09
            time_part = '.'.join(parts[3:6])  # 11.17.40
            suffix = '.'.join(parts[6:]) if len(parts) > 6 else ""  # Rec or empty
            
            if suffix:
                return f"{date_part}-{time_part}-{suffix}"
            else:
                return f"{date_part}-{time_part}"
    
    return filename


def resolve_drive_path(location: str, folder: str, filename: str) -> Tuple[Path, Path]:
    """
    Resolve network location to local drive paths for CMCR and CMTR files.
    
    If files are not found in the expected location, searches subfolders.
    Also tries normalized filename format if the original doesn't exist.
    
    Args:
        location: Network share name (e.g., "fs_3_1_data")
        folder: Subfolder name (e.g., "20240226")
        filename: Base filename without extension (e.g., "2024.02.26-10.53.19-Rec")
    
    Returns:
        Tuple of (cmcr_path, cmtr_path)
    
    Raises:
        ValueError: If location is not in DRIVE_MAP
    """
    if location not in DRIVE_MAP:
        raise ValueError(
            f"Unknown data location: '{location}'. "
            f"Expected one of: {list(DRIVE_MAP.keys())}"
        )
    
    drive = DRIVE_MAP[location]
    base_path = Path(f"{drive}/{folder}")
    
    # Try original filename
    cmcr_filename = f"{filename}.cmcr"
    cmtr_filename = f"{filename}-.cmtr"
    
    # Also try normalized filename (periods -> dashes)
    normalized = _normalize_filename(filename)
    cmcr_filename_norm = f"{normalized}.cmcr"
    cmtr_filename_norm = f"{normalized}-.cmtr"
    
    # Try direct path first
    cmcr_path = base_path / cmcr_filename
    cmtr_path = base_path / cmtr_filename
    
    # If not found, try normalized filename
    if not cmcr_path.exists():
        cmcr_path_norm = base_path / cmcr_filename_norm
        if cmcr_path_norm.exists():
            cmcr_path = cmcr_path_norm
    
    if not cmtr_path.exists():
        cmtr_path_norm = base_path / cmtr_filename_norm
        if cmtr_path_norm.exists():
            cmtr_path = cmtr_path_norm
    
    # If still not found, search in subfolders (try both patterns)
    if not cmcr_path.exists():
        found_cmcr = _search_file_in_subfolders(base_path, cmcr_filename)
        if not found_cmcr:
            found_cmcr = _search_file_in_subfolders(base_path, cmcr_filename_norm)
        if found_cmcr:
            cmcr_path = found_cmcr
    
    if not cmtr_path.exists():
        found_cmtr = _search_file_in_subfolders(base_path, cmtr_filename)
        if not found_cmtr:
            found_cmtr = _search_file_in_subfolders(base_path, cmtr_filename_norm)
        if found_cmtr:
            cmtr_path = found_cmtr
    
    return cmcr_path, cmtr_path


def get_dataset_id(filename: str) -> str:
    """
    Extract dataset_id from filename.
    
    Args:
        filename: Base filename (e.g., "2024.02.26-10.53.19-Rec")
    
    Returns:
        Dataset ID (same as filename, stripped of trailing dash if present)
    """
    return filename.rstrip("-")
