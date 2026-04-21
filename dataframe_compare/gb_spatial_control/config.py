"""
Configuration for GB Control Spatial Analysis Pipeline.

Combines "before blocker" green-blue data from all 3 experiments
(_ptx_str, _ptx, _str) for single-condition spatial analysis.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_DIR = SCRIPT_DIR.parent

EXPERIMENTS = ["_ptx_str", "_ptx", "_str"]

SOURCE_PARQUETS = {
    exp: COMPARE_DIR / f"output{exp}" / "compared_dataframe_v2_labeled_spatial.parquet"
    for exp in EXPERIMENTS
}

OUTPUT_DIR = COMPARE_DIR / "output_gb_control"
FIG_DIR_BASE = COMPARE_DIR / "figure_gb_control"

# Coordinate columns and scaling
X_COL = "improved_tx"
Y_COL = "improved_ty"
COORD_SCALE = 16
COORD_LIMIT = 100
XY_RANGE = (-COORD_LIMIT * COORD_SCALE, COORD_LIMIT * COORD_SCALE)

# Hexbin parameters
GRIDSIZE_ALL = 40
GRIDSIZE_GRP = 15
GRIDSIZE_CLUSTER = 15
MINCNT_ALL = 2
MINCNT_GRP = 1
MINCNT_CLUSTER = 1
CMAP = "coolwarm"
N_SPLINES_ALL = 30
N_SPLINES_GRP = 15
N_SPLINES_CLUSTER = 15
MIN_CELLS_CLUSTER = 10

# Step-up trace parameters
STEP_TRACE_COL = "before_step_up_5s_5i_b0_3x"
SAMPLING_RATE = 60.0  # Hz

# Group colors for step-up trace line
GROUP_COLORS = {
    "DSGC": "#1f77b4",
    "OSGC": "#2ca02c",
    "ipRGC": "#ff7f0e",
    "Other": "#7f7f7f",
}

# All before-GB feature columns (prefix will be stripped after loading).
# Ratios (8)
GB_RATIO_FEATURES = [
    "green_blue_on_ratio",
    "green_blue_on_ratio_low",
    "green_blue_on_ratio_mid",
    "green_blue_on_ratio_high",
    "green_blue_off_ratio",
    "green_blue_off_ratio_low",
    "green_blue_off_ratio_mid",
    "green_blue_off_ratio_high",
]

# Baseline (8)
GB_BASELINE_FEATURES = [
    "gb_base_mean",
    "gb_base_mean_low",
    "gb_base_mean_mid",
    "gb_base_mean_high",
    "gb_base_std",
    "gb_base_std_low",
    "gb_base_std_mid",
    "gb_base_std_high",
]

# Peak extremes (16)
GB_PEAK_FEATURES = [
    f"{color}_{phase}_peak_extreme{suffix}"
    for color in ["green", "blue"]
    for phase in ["on", "off"]
    for suffix in ["", "_low", "_mid", "_high"]
]

# Timing (4, overall only)
GB_TIMING_FEATURES = [
    "time_to_green_on_peak",
    "time_to_blue_on_peak",
    "time_to_green_off_peak",
    "time_to_blue_off_peak",
]

ALL_GB_FEATURES = (
    GB_RATIO_FEATURES + GB_BASELINE_FEATURES + GB_PEAK_FEATURES + GB_TIMING_FEATURES
)

# Metadata columns to keep alongside features
META_COLS = ["improved_tx", "improved_ty", "group", "subtype",
             "before_dataset_id", "source_experiment"]


def categorize(f):
    """Assign a feature category for plotting colors."""
    if "on_ratio" in f or "off_ratio" in f:
        return "Ratio"
    if "gb_base" in f:
        return "Baseline"
    if "peak_extreme" in f:
        return "Peak"
    if "time_to" in f:
        return "Timing"
    return "Other"


CAT_COLORS = {
    "Ratio": "#2196F3",
    "Baseline": "#4CAF50",
    "Peak": "#FF9800",
    "Timing": "#9C27B0",
    "Other": "#757575",
}


def short(f):
    """Shorten feature name for plot labels."""
    return (f.replace("green_blue_", "gb_")
             .replace("_peak_extreme", "_pk")
             .replace("_ratio", "_r")
             .replace("time_to_", "t_")
             .replace("_on_peak", "_on_pk")
             .replace("_off_peak", "_off_pk"))
