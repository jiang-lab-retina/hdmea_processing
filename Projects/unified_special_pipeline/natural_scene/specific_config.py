"""
Specific Configuration for Natural Scene Pipeline

Reads file_name_natural_scene.xlsx, normalizes filenames, searches 8 network
drives for CMCR/CMTR pairs, and maps Excel protocol names to playlist.csv
playlist names.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

CONFIG_DIR = Path(__file__).parent
EXCEL_PATH = CONFIG_DIR / "file_name_natural_scene.xlsx"
OUTPUT_DIR = CONFIG_DIR / "export"

# =============================================================================
# Drive Mappings (drive letter -> root path)
# =============================================================================

DRIVE_LIST: List[str] = [
    "L:",  # \\Jiangfs1\fs_1_1_data
    "M:",  # \\Jiangfs1\fs_1_2_data
    "Q:",  # \\Jiangfs2\fs_2_1_data
    "R:",  # \\Jiangfs2\fs_2_2_data
    "O:",  # \\Jiangfs3\fs_3_1_data
    "P:",  # \\Jiangfs3\fs_3_2_data
    "S:",  # \\Jiangfs4\fs_4_1_data
    "T:",  # \\Jiangfs4\fs_4_2_data
]

# =============================================================================
# Protocol -> Playlist Mapping
# =============================================================================

PROTOCOL_PLAYLIST_MAP: Dict[str, str] = {
    "play_all_optimization_set6_manual_ipRGC()": "play_optimization_set6_ipRGC_manual",
    "play_natural_scene_movie_v1()": "play_natural_scene_movie_v1",
}

PROTOCOL_SET6 = "play_all_optimization_set6_manual_ipRGC()"
PROTOCOL_NATURAL_SCENE = "play_natural_scene_movie_v1()"

# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class NaturalSceneSectionTimeConfig:
    """Section time config for natural scene recordings."""
    playlist_name: str = "play_natural_scene_movie_v1"
    pad_margin: Tuple[float, float] = (0.0, 0.0)
    cover_range: Tuple[int, int] = (-60, 0)


@dataclass
class Set6SectionTimeConfig:
    """Section time config for set6 recordings (same as unified pipeline default)."""
    playlist_name: str = "play_optimization_set6_ipRGC_manual"
    pad_margin: Tuple[float, float] = (0.0, 0.0)
    cover_range: Tuple[int, int] = (-60, 0)


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_filename(filename: str) -> str:
    """
    Normalize all-dot filename to dash format.

    "2025.02.04.13.22.13.Rec.cmcr" -> "2025.02.04-13.22.13-Rec.cmcr"
    Works on bare stems too (no extension required).
    """
    stem = filename
    ext = ""
    if "." in filename:
        # Split off known extensions
        for known_ext in (".cmcr", ".cmtr"):
            if filename.lower().endswith(known_ext):
                stem = filename[: -len(known_ext)]
                ext = filename[-len(known_ext):]
                break

    parts = stem.split(".")
    if len(parts) >= 7 and "-" not in stem:
        # YYYY.MM.DD.HH.MM.SS.Rec -> YYYY.MM.DD-HH.MM.SS-Rec
        date_part = ".".join(parts[0:3])
        time_part = ".".join(parts[3:6])
        suffix = ".".join(parts[6:])
        return f"{date_part}-{time_part}-{suffix}{ext}"
    if len(parts) >= 6 and "-" not in stem:
        date_part = ".".join(parts[0:3])
        time_part = ".".join(parts[3:6])
        suffix = ".".join(parts[6:]) if len(parts) > 6 else ""
        if suffix:
            return f"{date_part}-{time_part}-{suffix}{ext}"
        return f"{date_part}-{time_part}{ext}"
    return filename


def _search_file_in_subfolders(base_path: Path, filename: str) -> Optional[Path]:
    """Search base_path, then one level of subfolders, then rglob."""
    if not base_path.exists():
        return None

    direct = base_path / filename
    if direct.exists():
        return direct

    try:
        for subfolder in base_path.iterdir():
            if subfolder.is_dir():
                candidate = subfolder / filename
                if candidate.exists():
                    return candidate
    except PermissionError:
        pass

    try:
        for candidate in base_path.rglob(filename):
            return candidate
    except PermissionError:
        pass

    return None


def _date_folder_from_filename(normalized_stem: str) -> str:
    """
    Derive a date-based folder name from a normalized filename stem.

    "2025.02.04-13.22.13-Rec" -> "20250204"
    """
    date_part = normalized_stem.split("-")[0]  # "2025.02.04"
    return date_part.replace(".", "")


def search_cmcr_cmtr(excel_filename: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Search all drives for a CMCR/CMTR pair given the Excel filename.

    Args:
        excel_filename: Dot-format filename from Excel
                        (e.g. "2025.02.04.13.22.13.Rec.cmcr")

    Returns:
        (cmcr_path, cmtr_path) -- either may be None if not found
    """
    normalized = _normalize_filename(excel_filename)
    stem = normalized.replace(".cmcr", "").replace(".cmtr", "")

    cmcr_filename = f"{stem}.cmcr"
    cmtr_filename = f"{stem}-.cmtr"

    date_folder = _date_folder_from_filename(stem)

    cmcr_path: Optional[Path] = None
    cmtr_path: Optional[Path] = None

    for drive in DRIVE_LIST:
        drive_root = Path(drive + "/")
        if not drive_root.exists():
            continue

        # Priority 1: date-based subfolder
        date_dir = drive_root / date_folder
        if date_dir.exists():
            found = _search_file_in_subfolders(date_dir, cmcr_filename)
            if found and cmcr_path is None:
                cmcr_path = found
            found = _search_file_in_subfolders(date_dir, cmtr_filename)
            if found and cmtr_path is None:
                cmtr_path = found

        if cmcr_path and cmtr_path:
            break

        # Priority 2: drive root (one level of subfolders)
        if cmcr_path is None:
            direct = drive_root / cmcr_filename
            if direct.exists():
                cmcr_path = direct
        if cmtr_path is None:
            direct = drive_root / cmtr_filename
            if direct.exists():
                cmtr_path = direct

        if cmcr_path and cmtr_path:
            break

    return cmcr_path, cmtr_path


def get_dataset_id(excel_filename: str) -> str:
    """Extract dataset_id from an Excel filename."""
    normalized = _normalize_filename(excel_filename)
    stem = normalized.replace(".cmcr", "").replace(".cmtr", "")
    return stem.rstrip("-")


# =============================================================================
# Excel Loading
# =============================================================================


def load_excel_data(excel_path: Path = EXCEL_PATH) -> pd.DataFrame:
    """
    Load and validate the natural scene Excel file.

    Returns:
        DataFrame with cleaned column names and non-null rows.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    required = ["file_name", "protocol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df = df.dropna(subset=required)
    df["protocol"] = df["protocol"].str.strip()
    return df


def build_recording_list(
    df: Optional[pd.DataFrame] = None,
    excel_path: Path = EXCEL_PATH,
) -> List[dict]:
    """
    Build a list of recording dicts with resolved paths and playlist names.

    Each dict has keys:
        file_name, protocol, playlist_name, dataset_id, cmcr_path, cmtr_path
    """
    if df is None:
        df = load_excel_data(excel_path)

    recordings: List[dict] = []
    for _, row in df.iterrows():
        file_name = str(row["file_name"]).strip()
        protocol = str(row["protocol"]).strip()
        playlist_name = PROTOCOL_PLAYLIST_MAP.get(protocol)

        if playlist_name is None:
            logger.warning(
                f"Unknown protocol '{protocol}' for {file_name}, skipping"
            )
            continue

        dataset_id = get_dataset_id(file_name)
        cmcr_path, cmtr_path = search_cmcr_cmtr(file_name)

        recordings.append({
            "file_name": file_name,
            "protocol": protocol,
            "playlist_name": playlist_name,
            "dataset_id": dataset_id,
            "cmcr_path": cmcr_path,
            "cmtr_path": cmtr_path,
        })

    return recordings
