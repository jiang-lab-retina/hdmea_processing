"""
Configuration for yan_pipeline batch processing.

Edit this file to set your search folders and processing options.
These values are used as defaults by batch_from_folders.py but can be
overridden via command-line arguments.
"""

from pathlib import Path

# =============================================================================
# Search Folders - Add your CMCR data folders here
# =============================================================================

# List of directories to search for *.cmcr files (searched recursively)
SEARCH_FOLDERS = [
    # Uncomment and modify these paths, or add your own:
    # Path(r"M:\data_folder_1"),
    # Path(r"M:\data_folder_2"),
    Path(r"M:\20240226_protocol_test"),
]

# =============================================================================
# Output Configuration
# =============================================================================

# Directory where output HDF5 files will be saved
OUTPUT_DIR = Path(__file__).parent / "export"

# =============================================================================
# Processing Options
# =============================================================================

# Skip RF geometry extraction with LNL fitting (faster processing)
SKIP_RF_UPDATE = False

# Overwrite existing output files (set True to reprocess all)
OVERWRITE_EXISTING = False

# =============================================================================
# Logging
# =============================================================================

# Enable debug logging for verbose output
DEBUG = False

# =============================================================================
# Manual Label Image Generation
# =============================================================================

# Generate manual label images for each processed recording
CREATE_MANUAL_LABEL = True

# Output directory for manual label images
MANUAL_LABEL_DIR = Path(__file__).parent / "manual_label_images"

# Plot types to include in each unit's image (set False to disable)
MANUAL_LABEL_PLOTS = {
    "eimage_sta": True,      # STA montage (multiple frames)
    "geometry": False,        # Soma + RF geometry
    "ap_tracking": False,     # AP spread visualization
    "dsgc": False,           # DSGC direction tuning (optional)
}
