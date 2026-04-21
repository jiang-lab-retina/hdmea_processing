"""
Specific Configuration for Blocker Alignment Analysis Pipeline

This module contains experiment profiles for different blocker conditions.
The active experiment is resolved in this order:

    1. CLI flag   --experiment _ptx
    2. Env var    BLOCKER_EXPERIMENT=_ptx
    3. Default    ACTIVE_EXPERIMENT constant below

Pipeline overview:
- discover_files.py  -> file_index.csv   (uses DATE_*, SUBFOLDER_*, CONDITION_*)
- batch_pipeline.py  -> output{postfix}/  (uses OUTPUT_DIR)
- alignment.py       -> output{postfix}/aligned/  (uses ALIGNED_OUTPUT_DIR)
- batch_update.py    -> output_export{postfix}/   (uses EXPORT_DIR)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# Experiment Profile Definition
# =============================================================================

@dataclass(frozen=True)
class ExperimentProfile:
    """All parameters that vary between blocker experiments."""
    postfix: str
    date_start: str
    date_end: str
    subfolder_before: str
    subfolder_after: str
    data_folder_pattern: str          # e.g. "{date_compact}_gaba_gly"
    before_condition: str             # gsheet Condition substring
    after_condition: str              # gsheet Condition substring
    label_root: str                   # manual axon-label folder path

_LABEL_BASE = "M:/Python_Project/Data_Processing_2024/manual_label_data/set6_a"

EXPERIMENTS: Dict[str, ExperimentProfile] = {
    "_ptx_str": ExperimentProfile(
        postfix="_ptx_str",
        date_start="2025.09.04",
        date_end="2026.02.25",
        subfolder_before="before_gaba_gly",
        subfolder_after="gaba_gly",
        data_folder_pattern="{date_compact}_gaba_gly",
        before_condition="play_optimization_set6_a_ipRGC_without_step",
        after_condition="play_optimization_set6_a_ipRGC_manual",
        label_root=f"{_LABEL_BASE}/gaba_gly_blocker",
    ),
    "_ptx": ExperimentProfile(
        postfix="_ptx",
        date_start="2025.11.06",
        date_end="2026.02.05",
        subfolder_before="before_gaba",
        subfolder_after="gaba",
        data_folder_pattern="{date_compact}_gaba",
        before_condition="play_optimization_set6_a_ipRGC_without_step",
        after_condition="play_optimization_set6_a_ipRGC_manual",
        label_root=f"{_LABEL_BASE}/gaba_gly_blocker",
    ),
    "_str": ExperimentProfile(
        postfix="_str",
        date_start="2025.12.02",
        date_end="2025.12.19",
        subfolder_before="before_gly",
        subfolder_after="gly",
        data_folder_pattern="{date_compact}_gly",
        before_condition="play_optimization_set6_a_ipRGC_without_step",
        after_condition="play_optimization_set6_a_ipRGC_manual",
        label_root=f"{_LABEL_BASE}/gaba_gly_blocker",
    ),
}

# =============================================================================
# ACTIVE EXPERIMENT  --  single switch point for the entire pipeline
# =============================================================================

_DEFAULT_EXPERIMENT = "_ptx_str"


def _resolve_active_experiment(default: str = _DEFAULT_EXPERIMENT) -> str:
    """Determine active experiment from CLI --experiment flag or env var.

    Resolution order:
        1. ``--experiment <name>`` on the command line  (pre-parsed from sys.argv)
        2. ``BLOCKER_EXPERIMENT`` environment variable
        3. *default* (the ``_DEFAULT_EXPERIMENT`` constant above)
    """
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--experiment":
            return sys.argv[i + 1]
        if arg.startswith("--experiment="):
            return arg.split("=", 1)[1]
    return os.environ.get("BLOCKER_EXPERIMENT", default)


ACTIVE_EXPERIMENT = _resolve_active_experiment()

if ACTIVE_EXPERIMENT not in EXPERIMENTS:
    raise ValueError(
        f"Unknown experiment {ACTIVE_EXPERIMENT!r}. "
        f"Available: {list(EXPERIMENTS.keys())}"
    )

_exp = EXPERIMENTS[ACTIVE_EXPERIMENT]

# =============================================================================
# Path Configuration (derived from active experiment)
# =============================================================================

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent.parent.parent

FOLDER_POSTFIX = _exp.postfix

OUTPUT_DIR = CONFIG_DIR / f"output{FOLDER_POSTFIX}"
EXPORT_DIR = CONFIG_DIR / f"output_export{FOLDER_POSTFIX}"

# =============================================================================
# External Data Sources (shared across all experiments)
# =============================================================================

GSHEET_CSV_PATH = PROJECT_ROOT / "Projects/load_gsheet/gsheet_table.csv"

PLAYLIST_CSV_PATH = Path(
    "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/playlist.csv"
)

# =============================================================================
# Experiment-specific parameters (unpacked from active profile)
# =============================================================================

DATE_START = _exp.date_start
DATE_END = _exp.date_end

SUBFOLDER_BEFORE = _exp.subfolder_before
SUBFOLDER_GABA_GLY = _exp.subfolder_after

DATA_DRIVES = [Path(f"{d}:/") for d in "L M Q R O P S T".split()]

DATA_FOLDER_PATTERN = _exp.data_folder_pattern

BEFORE_CONDITION = _exp.before_condition
AFTER_CONDITION = _exp.after_condition

LABEL_ROOT = Path(_exp.label_root)

# =============================================================================
# Output Paths (derived)
# =============================================================================

FILE_INDEX_CSV = OUTPUT_DIR / "file_index.csv"
ALIGNED_OUTPUT_DIR = OUTPUT_DIR / "aligned"

# =============================================================================
# Helper Functions
# =============================================================================

def gsheet_filename_to_disk(gsheet_filename: str) -> str:
    """
    Convert gsheet filename format (all dots) to disk filename format (hyphens).

    Gsheet:  2025.09.04.10.11.09.Rec.cmcr
    Disk:    2025.09.04-10.11.09-Rec.cmcr

    The pattern is: YYYY.MM.DD.HH.MM.SS.Rec.cmcr -> YYYY.MM.DD-HH.MM.SS-Rec.cmcr
    Dots between date-time and time-Rec become hyphens.
    """
    # Remove extension for processing
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
        # parts[0:3] = YYYY, MM, DD
        # parts[3:6] = HH, MM, SS
        # parts[6]   = Rec (and possibly more)
        date_part = ".".join(parts[0:3])
        time_part = ".".join(parts[3:6])
        rec_part = ".".join(parts[6:])  # Usually just "Rec"
        return f"{date_part}-{time_part}-{rec_part}{ext}"
    # Fallback: return as-is
    return gsheet_filename


def disk_filename_to_gsheet(disk_filename: str) -> str:
    """
    Convert disk filename format (hyphens) to gsheet filename format (all dots).

    Disk:    2025.09.04-10.11.09-Rec.cmcr
    Gsheet:  2025.09.04.10.11.09.Rec.cmcr
    """
    return disk_filename.replace("-", ".")


def get_cmtr_from_cmcr(cmcr_filename: str) -> str:
    """
    Derive CMTR filename from CMCR filename.

    CMCR: 2025.09.04-10.11.09-Rec.cmcr
    CMTR: 2025.09.04-10.11.09-Rec-.cmtr  (trailing hyphen before .cmtr)
    """
    return cmcr_filename.replace(".cmcr", "-.cmtr")


def get_dataset_id_from_cmcr(cmcr_filename: str) -> str:
    """
    Extract dataset_id from CMCR filename.

    Example: "2025.09.04-10.11.09-Rec.cmcr" -> "2025.09.04-10.11.09-Rec"
    """
    return Path(cmcr_filename).stem


def extract_date_from_gsheet_filename(gsheet_filename: str) -> str:
    """
    Extract date string from gsheet filename.

    Example: "2025.09.04.10.11.09.Rec.cmcr" -> "2025.09.04"
    """
    parts = gsheet_filename.split(".")
    if len(parts) >= 3:
        return ".".join(parts[0:3])
    return ""


def extract_time_from_gsheet_filename(gsheet_filename: str) -> str:
    """
    Extract time string from gsheet filename.

    Example: "2025.09.04.10.11.09.Rec.cmcr" -> "10.11.09"
    """
    parts = gsheet_filename.split(".")
    if len(parts) >= 6:
        return ".".join(parts[3:6])
    return ""


def date_to_compact(date_str: str) -> str:
    """
    Convert date string to compact format for folder names.

    Example: "2025.09.04" -> "20250904"
    """
    return date_str.replace(".", "")


def find_data_folder_for_date(date_str: str) -> Path:
    """
    Search all mapped network drives for a date-specific data folder.

    Tries each drive in DATA_DRIVES and returns the first existing match.
    Falls back to the first drive's path if nothing is found on disk.

    Args:
        date_str: Date string like "2025.09.04"

    Returns:
        Path to the date folder (first existing match, or first-drive
        fallback so downstream code can report a meaningful "not found").
    """
    compact = date_to_compact(date_str)
    folder_name = DATA_FOLDER_PATTERN.format(date_compact=compact)
    for drive in DATA_DRIVES:
        candidate = drive / folder_name
        if candidate.exists():
            return candidate
    return DATA_DRIVES[0] / folder_name


def parse_condition_for_playlist(condition: str) -> str:
    """
    Extract the playlist-relevant part from the Condition column.

    Takes the part before the first comma.
    Example: "play_optimization_set6_a_ipRGC_manual(), gaba, glycine"
             -> "play_optimization_set6_a_ipRGC_manual()"

    Example: "play_optimization_set6_a_ipRGC_without_step()"
             -> "play_optimization_set6_a_ipRGC_without_step()"
    """
    if not condition:
        return condition
    return condition.split(",")[0].strip()


def get_cmcr_cmtr_paths(
    cmcr_filename: str, data_folder: Path
) -> Tuple[Path, Path]:
    """
    Get CMCR and CMTR file paths from CMCR filename.

    Args:
        cmcr_filename: CMCR filename (disk format with hyphens)
        data_folder: Folder containing the files

    Returns:
        Tuple of (cmcr_path, cmtr_path)
    """
    cmcr_path = data_folder / cmcr_filename
    cmtr_filename = get_cmtr_from_cmcr(cmcr_filename)
    cmtr_path = data_folder / cmtr_filename
    return cmcr_path, cmtr_path
