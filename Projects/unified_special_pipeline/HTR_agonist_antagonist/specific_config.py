"""
Specific Configuration for HTR Agonist/Antagonist Pipeline

This module contains configuration for the HTR agonist/antagonist batch
processing pipeline. Key feature: dynamic playlist selection based on
gsheet "Condition" column.
"""

from pathlib import Path
from typing import List

# =============================================================================
# Path Configuration
# =============================================================================

# Directory containing this config file
CONFIG_DIR = Path(__file__).parent

# Project root (Data_Processing_2027)
PROJECT_ROOT = CONFIG_DIR.parent.parent.parent

# Data folder containing CMCR/CMTR files
DATA_FOLDER = Path("O:/20251001_HttrB_agonist")

# Output directory for processed HDF5 files
OUTPUT_DIR = CONFIG_DIR / "output"

# =============================================================================
# External Data Sources
# =============================================================================

# Google Sheet CSV cache
GSHEET_CSV_PATH = PROJECT_ROOT / "Projects/load_gsheet/gsheet_table.csv"

# Playlist CSV (network path)
PLAYLIST_CSV_PATH = Path("//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/playlist.csv")

# =============================================================================
# Test Files
# =============================================================================

# List of CMCR files to process
TEST_FILES: List[str] = [
    "2025.10.01-09.33.32-Rec.cmcr",
    "2025.10.01-09.45.32-Rec.cmcr",
    "2025.10.01-09.55.44-Rec.cmcr",
    "2025.10.01-10.05.45-Rec.cmcr",
    "2025.10.01-10.15.53-Rec.cmcr",
]

# =============================================================================
# Helper Functions
# =============================================================================

def get_dataset_id_from_cmcr(cmcr_filename: str) -> str:
    """
    Extract dataset_id from CMCR filename.
    
    Example: "2025.10.01-09.33.32-Rec.cmcr" -> "2025.10.01-09.33.32-Rec"
    
    Args:
        cmcr_filename: CMCR filename (with or without path)
    
    Returns:
        Dataset ID (filename without extension)
    """
    # Get just the filename without path and extension
    return Path(cmcr_filename).stem  # Remove .cmcr extension


def get_cmcr_cmtr_paths(cmcr_filename: str, data_folder: Path = DATA_FOLDER):
    """
    Get CMCR and CMTR file paths from CMCR filename.
    
    Args:
        cmcr_filename: CMCR filename (e.g., "2025.10.01.09.33.32.Rec.cmcr")
        data_folder: Folder containing the files
    
    Returns:
        Tuple of (cmcr_path, cmtr_path)
    """
    cmcr_path = data_folder / cmcr_filename
    
    # CMTR has a trailing dash before extension
    # e.g., "2025.10.01.09.33.32.Rec-.cmtr"
    cmtr_filename = cmcr_filename.replace(".cmcr", "-.cmtr")
    cmtr_path = data_folder / cmtr_filename
    
    return cmcr_path, cmtr_path
