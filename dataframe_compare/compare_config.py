"""
Configuration for the Blocker Before/After Comparison Pipeline.

Defines paths, movie lists, and feature mappings adapted for the
blocker alignment data (output_export H5 files).

The active experiment is controlled by ACTIVE_EXPERIMENT in
blocker_alignment_analysis/specific_config.py.  FOLDER_POSTFIX is
imported from there so there is a single switch point for the
entire pipeline.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import FOLDER_POSTFIX from the blocker alignment config so that both
# pipelines share a single switch point (ACTIVE_EXPERIMENT).
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from Projects.unified_special_pipeline.blocker_alignment_analysis.specific_config import (
    FOLDER_POSTFIX,
    OUTPUT_DIR as _BLOCKER_OUTPUT_DIR,
    EXPORT_DIR as _BLOCKER_EXPORT_DIR,
    ALIGNED_OUTPUT_DIR as _BLOCKER_ALIGNED_DIR,
)

# =============================================================================
# DERIVED PATHS -- computed from the shared FOLDER_POSTFIX
# =============================================================================

PROJECT_ROOT = _project_root
SCRIPT_DIR = Path(__file__).parent

# Upstream H5 source directories (from blocker alignment pipeline)
ALIGNED_DIR = _BLOCKER_ALIGNED_DIR
EXPORT_DIR = _BLOCKER_EXPORT_DIR

# Local output / figure directories for the comparison pipeline
OUTPUT_DIR = SCRIPT_DIR / f"output{FOLDER_POSTFIX}"
FIG_DIR_BASE = SCRIPT_DIR / f"figure{FOLDER_POSTFIX}"

# =============================================================================
# MOVIE CONFIGURATION
# =============================================================================

# Movies available in BEFORE recordings (frame-aligned)
MOVIES_BEFORE = [
    "baseline_127",
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x_64_128_255",
    "step_up_5s_5i_b0_3x",
]

# Movies available in AFTER recordings (frame-aligned) -- superset of BEFORE
MOVIES_AFTER = [
    "baseline_127",
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x_64_128_255",
    "step_up_5s_5i_b0_3x",
    "step_up_5s_5i_b0_30x",
]

# Direction section movie (same for before and after)
MOVIE_DIRECTION_SECTION = "moving_h_bar_s5_d8_3x"

# iprgc_test -- only in AFTER recordings (sample-based, not frame-aligned)
MOVIE_SAMPLE_BASED = "iprgc_test"
IPRGC_TARGET_RATE_HZ = 60.0
IPRGC_EXPECTED_BINS = 7200
IPRGC_LENGTH_TOLERANCE = 0.10

# Excluded movies
EXCLUDED_MOVIES = ["perfect_dense_noise_15x15_15hz_r42_3min"]

# Moving bar prefix for column parsing
MOVING_BAR_PREFIX = "moving_h_bar_s5_d8_3x"

# Column names for feature extraction (blocker-specific)
GB_TRACE_COLUMN = "green_blue_3s_3i_3x_64_128_255"
STEP_TRACE_COLUMN = "step_up_5s_5i_b0_3x"
FREQ_TRACE_COLUMN = "freq_step_5st_3x"

# =============================================================================
# HDF5 FEATURE PATHS
# =============================================================================

FEATURE_PATHS = {
    # Cell type label
    "axon_type": "auto_label/axon_type",

    # AP tracking / soma polar coordinates
    "angle_correction_applied": "features/ap_tracking/soma_polar_coordinates/angle_correction_applied",
    "transformed_x": "features/ap_tracking/soma_polar_coordinates/transformed_x",
    "transformed_y": "features/ap_tracking/soma_polar_coordinates/transformed_y",
    "polar_radius": "features/ap_tracking/soma_polar_coordinates/radius",
    "polar_theta_deg": "features/ap_tracking/soma_polar_coordinates/theta_deg",
    "polar_theta_deg_raw": "features/ap_tracking/soma_polar_coordinates/theta_deg_raw",
    "cartesian_x": "features/ap_tracking/soma_polar_coordinates/cartesian_x",
    "cartesian_y": "features/ap_tracking/soma_polar_coordinates/cartesian_y",

    # STA geometry - Gaussian fit
    "gaussian_sigma_x": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/sigma_x",
    "gaussian_sigma_y": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/sigma_y",
    "gaussian_amp": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/amplitude",
    "gaussian_r2": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/gaussian_fit/r_squared",

    # STA geometry - DoG
    "dog_sigma_exc": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/sigma_exc",
    "dog_sigma_inh": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/sigma_inh",
    "dog_amp_exc": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/amp_exc",
    "dog_amp_inh": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/amp_inh",
    "dog_r2": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/DoG/r_squared",

    # STA geometry - time course (array)
    "sta_time_course": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/sta_time_course",

    # STA geometry - LNL model fit
    "lnl_a": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/a",
    "lnl_b": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/b",
    "lnl_a_norm": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/a_norm",
    "lnl_bits_per_spike": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/bits_per_spike",
    "lnl_r_squared": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/r_squared",
    "lnl_rectification_index": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/rectification_index",
    "lnl_nonlinearity_index": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/nonlinearity_index",
    "lnl_threshold_g": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/threshold_g",
    "lnl_log_likelihood": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/log_likelihood",
    "lnl_null_log_likelihood": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/null_log_likelihood",
    "lnl_n_frames": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/n_frames",
    "lnl_n_spikes": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/n_spikes",
    "lnl_g_bin_centers": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/g_bin_centers",
    "lnl_rate_vs_g": "features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/rate_vs_g",
}
