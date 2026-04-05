"""
Specific Configuration for High Glucose JHU Analysis Pipeline

Configuration for analyzing step responses during high glucose perfusion.
Each recording is analyzed independently (no cross-recording alignment).

Data: S:\\20260226_high_glucose
Timing: hight_glucose_note.xlsx
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# sys.path setup so we can import the sibling step_change_analysis package
# ---------------------------------------------------------------------------
_USP_DIR = Path(__file__).resolve().parent.parent.parent  # unified_special_pipeline
if str(_USP_DIR) not in sys.path:
    sys.path.insert(0, str(_USP_DIR))

from step_change_analysis.specific_config import (
    StepDetectionConfig,
    QualityConfig,
    ResponseAnalysisConfig,
    VisualizationConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

CONFIG_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = CONFIG_DIR / "output"
FIGURES_DIR = CONFIG_DIR / "figures"

DATA_FOLDER = Path(r"S:\20260226_high_glucose")
XLSX_PATH = CONFIG_DIR / "hight_glucose_note.xlsx"

EXCLUDE_RECORDINGS = {
    "2026.02.26.12.09.16.Rec.cmcr",
    "2026.02.26.13.26.55.Rec.cmcr",
}


# =============================================================================
# Glucose Timing Dataclass
# =============================================================================

@dataclass
class GlucoseTimingInfo:
    """Glucose timing metadata for a single recording."""
    cmcr: str
    cmtr: str
    high_glucose_min: float
    normal_glucose_min: float
    description: str = ""


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class GlucosePipelineConfig:
    """Master configuration for the glucose analysis pipeline."""

    step_detection: StepDetectionConfig = field(default_factory=StepDetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    response_analysis: ResponseAnalysisConfig = field(default_factory=ResponseAnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    data_folder: Path = DATA_FOLDER
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    xlsx_path: Path = XLSX_PATH


default_config = GlucosePipelineConfig()


# =============================================================================
# File-Name Normalization
# =============================================================================

def normalize_cmcr_filename(xlsx_name: str) -> str:
    """
    Convert the xlsx file-name format to the actual name on disk.

    xlsx:  2026.02.26.09.25.05.Rec.cmcr   (all dots)
    disk:  2026.02.26-09.25.05-Rec.cmcr   (dashes between date/time/Rec)
    """
    parts = xlsx_name.split(".")
    # Expected: ['2026', '02', '26', '09', '25', '05', 'Rec', 'cmcr']
    if len(parts) >= 8:
        date_part = ".".join(parts[0:3])   # 2026.02.26
        time_part = ".".join(parts[3:6])   # 09.25.05
        rest = ".".join(parts[6:])         # Rec.cmcr
        return f"{date_part}-{time_part}-{rest}"
    return xlsx_name


def derive_cmtr_filename(cmcr_filename: str) -> str:
    """Derive the CMTR filename from a CMCR filename (adds dash before extension)."""
    return cmcr_filename.replace(".cmcr", "-.cmtr")


# =============================================================================
# Recording-Info Loader
# =============================================================================

def load_recording_info(
    xlsx_path: Optional[Path] = None,
    data_folder: Optional[Path] = None,
) -> List[GlucoseTimingInfo]:
    """
    Read the xlsx note file and return a list of recording metadata.

    Only recordings whose *both* CMCR and CMTR files exist on disk are
    included; others are logged and skipped.
    """
    if xlsx_path is None:
        xlsx_path = XLSX_PATH
    if data_folder is None:
        data_folder = DATA_FOLDER

    df = pd.read_excel(xlsx_path)

    recordings: List[GlucoseTimingInfo] = []
    for idx, row in df.iterrows():
        raw_name = str(row["file_name"])
        if raw_name in EXCLUDE_RECORDINGS:
            logger.info("Excluded by EXCLUDE_RECORDINGS: %s", raw_name)
            continue

        cmcr_name = normalize_cmcr_filename(raw_name)
        cmtr_name = derive_cmtr_filename(cmcr_name)

        cmcr_path = data_folder / cmcr_name
        cmtr_path = data_folder / cmtr_name

        if not cmcr_path.exists():
            logger.warning("CMCR not found, skipping: %s", cmcr_path)
            continue
        if not cmtr_path.exists():
            logger.warning("CMTR not found, skipping: %s", cmtr_path)
            continue

        recordings.append(GlucoseTimingInfo(
            cmcr=cmcr_name,
            cmtr=cmtr_name,
            high_glucose_min=float(row["high_glucose(min)"]),
            normal_glucose_min=float(row["normal_glucose(min)"]),
            description=f"Rec_{idx + 1:02d}_{cmcr_name[:19]}",
        ))

    logger.info("Loaded %d recordings from %s", len(recordings), xlsx_path.name)
    return recordings


def get_output_hdf5_path(
    cmcr_filename: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate an output HDF5 path from a CMCR filename."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / cmcr_filename.replace(".cmcr", ".h5")
