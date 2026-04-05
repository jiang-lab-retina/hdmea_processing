"""
Specific Configuration for Low Glucose Analysis Pipeline

Configuration for analyzing step responses during low glucose perfusion.
Each recording is analyzed independently (no cross-recording alignment).

Data: S:\\20260304_low_glucose and S:\\20260305_low_glucose
Timing: Parsed from MEA dashboard Google Sheet (cached CSV)

Protocol (transition recordings):
  - Normal glucose ionic AMES from start
  - 25mM high glucose onset at 2.5 min
  - 2mM low glucose onset at 15 min
"""

import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_GLUCOSE_JHU_DIR = _THIS_DIR.parent
_USP_DIR = _GLUCOSE_JHU_DIR.parent

for _p in (str(_GLUCOSE_JHU_DIR), str(_USP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step_change_analysis.specific_config import (
    StepDetectionConfig,
    QualityConfig,
    ResponseAnalysisConfig,
    VisualizationConfig,
)

_parent_cfg_path = _GLUCOSE_JHU_DIR / "high-glucose-alone" / "specific_config.py"
_spec = importlib.util.spec_from_file_location("parent_specific_config", _parent_cfg_path)
_parent_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent_cfg)
normalize_cmcr_filename = _parent_cfg.normalize_cmcr_filename
derive_cmtr_filename = _parent_cfg.derive_cmtr_filename

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

CONFIG_DIR = _THIS_DIR
OUTPUT_DIR = CONFIG_DIR / "data"
FIGURES_DIR = CONFIG_DIR / "figures"

DATA_FOLDERS = [
    Path(r"S:\20260304_low_glucose"),
    Path(r"S:\20260305_low_glucose"),
]

GSHEET_CSV_PATH = _USP_DIR.parent / "load_gsheet" / "gsheet_table.csv"

DATE_PREFIXES = ["2026.03.04", "2026.03.05"]

TRANSITION_PATTERN = re.compile(
    r"25mM\s+glucose\s*@\s*([\d.]+)\s*min.*low\s+glucose\s+2mM\s*@\s*([\d.]+)\s*min",
    re.IGNORECASE,
)


# =============================================================================
# Timing Dataclass
# =============================================================================

@dataclass
class LowGlucoseTimingInfo:
    """Glucose timing metadata for a single recording."""
    cmcr: str
    cmtr: str
    high_glucose_min: float
    low_glucose_min: float
    recording_type: str  # "transition" or "steady_state"
    data_folder: Path = Path(".")
    description: str = ""


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class LowGlucosePipelineConfig:
    """Master configuration for the low-glucose analysis pipeline."""

    step_detection: StepDetectionConfig = field(default_factory=StepDetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    response_analysis: ResponseAnalysisConfig = field(default_factory=ResponseAnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    data_folders: List[Path] = field(default_factory=lambda: list(DATA_FOLDERS))
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    gsheet_csv_path: Path = GSHEET_CSV_PATH


default_config = LowGlucosePipelineConfig()


# =============================================================================
# Google Sheet CSV Parsing
# =============================================================================

def _find_data_folder(cmcr_name: str, data_folders: List[Path]) -> Optional[Path]:
    """Find which data folder contains a given CMCR file."""
    for folder in data_folders:
        if (folder / cmcr_name).exists():
            return folder
    return None


def load_recording_info(
    gsheet_csv_path: Optional[Path] = None,
    data_folders: Optional[List[Path]] = None,
) -> List[LowGlucoseTimingInfo]:
    """Read the cached Google Sheet CSV and return recording metadata.

    Only transition recordings with both CMCR and CMTR files are included.
    """
    if gsheet_csv_path is None:
        gsheet_csv_path = GSHEET_CSV_PATH
    if data_folders is None:
        data_folders = list(DATA_FOLDERS)

    df = pd.read_csv(gsheet_csv_path)

    mask = pd.Series(False, index=df.index)
    for prefix in DATE_PREFIXES:
        mask |= df["File_name"].str.startswith(prefix, na=False)
    df_filtered = df[mask].copy()

    recordings: List[LowGlucoseTimingInfo] = []
    rec_counter = 0

    for _, row in df_filtered.iterrows():
        xlsx_name = str(row["File_name"])
        condition = str(row.get("Condition", ""))

        cmcr_name = normalize_cmcr_filename(xlsx_name)
        cmtr_name = derive_cmtr_filename(cmcr_name)

        match = TRANSITION_PATTERN.search(condition)
        if not match:
            logger.debug("Skipping steady-state recording: %s", cmcr_name)
            continue

        high_glc_min = float(match.group(1))
        low_glc_min = float(match.group(2))

        folder = _find_data_folder(cmcr_name, data_folders)
        if folder is None:
            logger.warning("CMCR not found on any drive, skipping: %s", cmcr_name)
            continue

        cmtr_path = folder / cmtr_name
        if not cmtr_path.exists():
            logger.warning("CMTR not found, skipping: %s", cmtr_path)
            continue

        rec_counter += 1
        recordings.append(LowGlucoseTimingInfo(
            cmcr=cmcr_name,
            cmtr=cmtr_name,
            high_glucose_min=high_glc_min,
            low_glucose_min=low_glc_min,
            recording_type="transition",
            data_folder=folder,
            description=f"Rec_{rec_counter:02d}_{cmcr_name[:19]}",
        ))

    logger.info("Loaded %d transition recordings from %s", len(recordings),
                gsheet_csv_path.name)
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
