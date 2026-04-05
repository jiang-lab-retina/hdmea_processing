"""
Specific Configuration for White Noise Salamander STA Pipeline

Salamander retinal ganglion cell receptive field mapping via spike-triggered
average (STA) of dense white-noise stimulation.

Data: S:\20240304_salamander, S:\20240227_salamander, S:\20260226_salamander,
      S:\20260303_salamander
Stimulus: perfect_dense_noise_15x15_5hz_r42_10min.npy
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

CONFIG_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = CONFIG_DIR / "output"
FIGURES_DIR = CONFIG_DIR / "figures"

DATA_FOLDERS: List[Path] = [
    Path(r"S:\20240227_salamander"),
    Path(r"S:\20240304_salamander"),
    Path(r"S:\20260226_salamander"),
    Path(r"S:\20260303_salamander"),
]

STIMULUS_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\perfect_dense_noise_15x15_5hz_r42_10min.npy"
)

STIMULI_DIR = STIMULUS_PATH.parent

GSHEET_CSV_PATH = CONFIG_DIR.parent.parent.parent / "Projects" / "load_gsheet" / "gsheet_table.csv"

# =============================================================================
# STA Parameters
# =============================================================================

SECTION_TIME_FRAME_NUM: Tuple[int, float] = (184, float("inf"))

STA_COVER_RANGE: Tuple[int, int] = (-60, 0)

FRAME_CHANNEL_KEY = "raw_ch2"

# Single recording for targeted analysis
TEST_FILES = [
    (
        Path(r"S:\20240304_salamander\2024.03.04-11.30.49-Rec.cmcr"),
        Path(r"S:\20240304_salamander\2024.03.04-11.30.49-Rec-.cmtr"),
    ),
]


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class STAPipelineConfig:
    """Master configuration for the white-noise STA pipeline."""

    data_folders: List[Path] = field(default_factory=lambda: list(DATA_FOLDERS))
    stimulus_path: Path = STIMULUS_PATH
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    section_time_frame_num: Tuple[int, float] = SECTION_TIME_FRAME_NUM
    cover_range: Tuple[int, int] = STA_COVER_RANGE
    frame_channel_key: str = FRAME_CHANNEL_KEY


default_config = STAPipelineConfig()


# =============================================================================
# Recording Discovery
# =============================================================================

@dataclass
class RecordingInfo:
    """Metadata for a single paired CMCR/CMTR recording."""
    cmcr_path: Path
    cmtr_path: Path
    dataset_id: str


def _derive_cmtr_path(cmcr_path: Path) -> Optional[Path]:
    """Derive the CMTR path from a CMCR path (inserts dash before extension)."""
    cmtr_name = cmcr_path.name.replace(".cmcr", "-.cmtr")
    candidate = cmcr_path.parent / cmtr_name
    if candidate.exists():
        return candidate
    return None


def discover_recordings(
    data_folders: Optional[List[Path]] = None,
) -> List[RecordingInfo]:
    """
    Scan data folders for paired CMCR + CMTR files.

    Only recordings where both files exist are returned.
    """
    if data_folders is None:
        data_folders = DATA_FOLDERS

    recordings: List[RecordingInfo] = []

    for folder in data_folders:
        if not folder.exists():
            logger.warning("Data folder not found, skipping: %s", folder)
            continue

        cmcr_files = sorted(folder.glob("*.cmcr"))
        for cmcr_path in cmcr_files:
            cmtr_path = _derive_cmtr_path(cmcr_path)
            if cmtr_path is None:
                continue

            dataset_id = cmcr_path.stem  # e.g. "2024.03.04-11.30.49-Rec"
            recordings.append(RecordingInfo(
                cmcr_path=cmcr_path,
                cmtr_path=cmtr_path,
                dataset_id=dataset_id,
            ))

    logger.info("Discovered %d paired recordings across %d folders",
                len(recordings), len(data_folders))
    return recordings


def get_output_figures_dir(
    dataset_id: str,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Return per-recording figures directory, creating it if needed."""
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    out = figures_dir / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# Google Sheet Recording Discovery
# =============================================================================

_PLAY_MOVIE_RE = re.compile(r'play_movie\("(.+?)"\)')


@dataclass
class GSheetRecordingInfo(RecordingInfo):
    """RecordingInfo augmented with per-recording stimulus metadata."""
    movie_name: str = ""
    stimulus_path: Path = STIMULUS_PATH


def gsheet_to_disk_filename(gsheet_filename: str) -> str:
    """
    Convert gsheet filename format (all dots) to disk filename format (hyphens).

    Gsheet:  2026.03.03.10.37.21.Rec.cmcr
    Disk:    2026.03.03-10.37.21-Rec.cmcr
    """
    ext = ""
    if gsheet_filename.endswith(".cmcr"):
        ext = ".cmcr"
        base = gsheet_filename[:-5]
    elif gsheet_filename.endswith(".cmtr"):
        ext = ".cmtr"
        base = gsheet_filename[:-5]
    else:
        base = gsheet_filename

    parts = base.split(".")
    if len(parts) >= 7:
        date_part = ".".join(parts[0:3])
        time_part = ".".join(parts[3:6])
        rec_part = ".".join(parts[6:])
        return f"{date_part}-{time_part}-{rec_part}{ext}"
    return gsheet_filename


def _find_cmcr_on_disk(disk_filename: str, data_folders: List[Path]) -> Optional[Path]:
    """Search DATA_FOLDERS for a CMCR file by its disk-format name."""
    for folder in data_folders:
        candidate = folder / disk_filename
        if candidate.exists():
            return candidate
    return None


def discover_recordings_from_gsheet(
    date_filter: str,
    gsheet_df: Optional[pd.DataFrame] = None,
    data_folders: Optional[List[Path]] = None,
    stimuli_dir: Optional[Path] = None,
) -> List[GSheetRecordingInfo]:
    """
    Discover recordings from Google Sheet data filtered by date and play_movie condition.

    Filters rows where File_name contains date_filter and Condition contains
    play_movie("..."), extracts the movie filename, and resolves disk paths.

    Args:
        date_filter: Date substring to match in File_name (e.g. "2026.03.03")
        gsheet_df: Pre-loaded gsheet DataFrame. Loaded from GSHEET_CSV_PATH if None.
        data_folders: Folders to search for CMCR files. Uses DATA_FOLDERS if None.
        stimuli_dir: Directory containing stimulus .npy files. Uses STIMULI_DIR if None.

    Returns:
        List of GSheetRecordingInfo for matched recordings with valid file pairs.
    """
    if gsheet_df is None:
        if not GSHEET_CSV_PATH.exists():
            logger.error("Gsheet CSV not found: %s", GSHEET_CSV_PATH)
            return []
        gsheet_df = pd.read_csv(GSHEET_CSV_PATH)

    if data_folders is None:
        data_folders = DATA_FOLDERS
    if stimuli_dir is None:
        stimuli_dir = STIMULI_DIR

    # Filter by date
    mask_date = gsheet_df["File_name"].str.contains(date_filter, na=False)
    # Filter by play_movie condition
    mask_cond = gsheet_df["Condition"].str.contains("play_movie(", na=False, regex=False)
    filtered = gsheet_df[mask_date & mask_cond].copy()

    if filtered.empty:
        logger.warning("No play_movie recordings found for date '%s'", date_filter)
        return []

    recordings: List[GSheetRecordingInfo] = []

    for _, row in filtered.iterrows():
        gsheet_name = row["File_name"]
        condition = row["Condition"]

        m = _PLAY_MOVIE_RE.search(condition)
        if m is None:
            logger.warning("Could not parse movie from condition: %s", condition)
            continue

        movie_filename = m.group(1)
        movie_name = movie_filename.replace(".npy", "")
        stim_path = stimuli_dir / movie_filename

        if not stim_path.exists():
            logger.warning("Stimulus not found: %s", stim_path)
            continue

        disk_name = gsheet_to_disk_filename(gsheet_name)
        cmcr_path = _find_cmcr_on_disk(disk_name, data_folders)
        if cmcr_path is None:
            logger.warning("CMCR not found on disk: %s", disk_name)
            continue

        cmtr_path = _derive_cmtr_path(cmcr_path)
        if cmtr_path is None:
            logger.warning("CMTR not found for: %s", cmcr_path)
            continue

        dataset_id = cmcr_path.stem
        recordings.append(GSheetRecordingInfo(
            cmcr_path=cmcr_path,
            cmtr_path=cmtr_path,
            dataset_id=dataset_id,
            movie_name=movie_name,
            stimulus_path=stim_path,
        ))

    logger.info(
        "Discovered %d play_movie recordings for date '%s'",
        len(recordings), date_filter,
    )
    return recordings
