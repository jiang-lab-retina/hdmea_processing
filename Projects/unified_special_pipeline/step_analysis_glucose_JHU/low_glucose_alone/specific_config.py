"""
Configuration for the low-glucose-alone pipeline.

Reproduces legacy Figure X from:
  Control:      2025.10.07 recordings  (P:\\20251007_low_glucose_control)
  Low glucose:  2025.10.10 recordings  (P:\\20251010_low_glucose)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_THIS_DIR = Path(__file__).resolve().parent

# ---- Recording pair definitions (matching legacy visualize_chains_5.py) ----

CONTROL_PAIRS: List[Tuple[str, str]] = [
    ("2025.10.07-09.49.16-Rec", "2025.10.07-11.29.21-Rec"),
    ("2025.10.07-11.54.39-Rec", "2025.10.07-13.40.38-Rec"),
    ("2025.10.07-14.07.13-Rec", "2025.10.07-15.54.17-Rec"),
]

LOW_GLUCOSE_PAIRS: List[Tuple[str, str]] = [
    ("2025.10.10-10.12.27-Rec", "2025.10.10-11.52.32-Rec"),
    ("2025.10.10-12.17.00-Rec", "2025.10.10-13.57.06-Rec"),
    ("2025.10.10-14.22.52-Rec", "2025.10.10-16.02.57-Rec"),
]

# ---- Data folder search ----

SEARCH_DRIVES = ["P:", "S:", "T:", "L:", "M:", "Q:", "R:", "O:"]
CONTROL_FOLDER_NAME = "20251007_low_glucose_control"
LOW_GLUCOSE_FOLDER_NAME = "20251010_low_glucose"


def find_data_folder(folder_name: str) -> Path:
    """Search mapped drives for a data folder by name."""
    for drive in SEARCH_DRIVES:
        candidate = Path(drive) / folder_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {folder_name} on drives {SEARCH_DRIVES}"
    )


def cmcr_path(folder: Path, stem: str) -> Path:
    return folder / f"{stem}.cmcr"


def cmtr_path(folder: Path, stem: str) -> Path:
    return folder / f"{stem}-.cmtr"


# ---- Legacy feature parameters ----

BASELINE_RANGE = (0, 5)
OFF_PEAK_RANGE = (40, 50)
ON_PEAK_RANGE = (10, 20)
REPEAT_NUM_CLIP = 83
STIMULUS_INTERVAL_S = 10
SMOOTHING_WINDOW = 5

# ---- Output paths ----

OUTPUT_DIR = _THIS_DIR / "data"
FIGURES_DIR = _THIS_DIR / "figures"
PAPER_FIGURE_DIR = _THIS_DIR / "paper_figure"


@dataclass
class LowGlucoseAloneConfig:
    control_folder_name: str = CONTROL_FOLDER_NAME
    low_glucose_folder_name: str = LOW_GLUCOSE_FOLDER_NAME
    control_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: list(CONTROL_PAIRS)
    )
    low_glucose_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: list(LOW_GLUCOSE_PAIRS)
    )
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    paper_figure_dir: Path = PAPER_FIGURE_DIR
    baseline_range: Tuple[int, int] = BASELINE_RANGE
    off_peak_range: Tuple[int, int] = OFF_PEAK_RANGE
    on_peak_range: Tuple[int, int] = ON_PEAK_RANGE
    repeat_num_clip: int = REPEAT_NUM_CLIP
    stimulus_interval_s: int = STIMULUS_INTERVAL_S
    smoothing_window: int = SMOOTHING_WINDOW


default_config = LowGlucoseAloneConfig()
