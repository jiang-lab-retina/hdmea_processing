"""
Configuration constants for the Baden-method RGC clustering pipeline.

All pipeline parameters are centralized here for easy modification.
"""

from pathlib import Path

# =============================================================================
# Paths (all absolute, based on this file's location)
# =============================================================================

# This file's directory (Baden_method folder)
_THIS_DIR = Path(__file__).parent.resolve()

# Project root (4 levels up from this file)
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent

# Default input path (absolute)
INPUT_PATH = _PROJECT_ROOT / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet"

# Output directory (this folder)
OUTPUT_DIR = _THIS_DIR

# Subdirectories
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

# =============================================================================
# Column Names
# =============================================================================

# Frequency section trace columns (replacing chirp)
# Each section contains a different frequency component
FREQ_SECTION_COLS = [
    "freq_section_0p5hz",
    "freq_section_1hz",
    "freq_section_2hz",
    "freq_section_4hz",
    "freq_section_10hz",
]

# Which frequency sections should be low-pass filtered (10 Hz is NOT filtered)
FREQ_SECTION_FILTER = {
    "freq_section_0p5hz": True,
    "freq_section_1hz": True,
    "freq_section_2hz": True,
    "freq_section_4hz": True,
    "freq_section_10hz": False,  # No filtering for 10 Hz section
}

# Legacy: Chirp stimulus trace column (kept for reference)
CHIRP_COL = "freq_step_5st_3x"

# Color stimulus trace column
COLOR_COL = "green_blue_3s_3i_3x"

# Moving bar direction columns (8 directions)
BAR_COLS = [
    "corrected_moving_h_bar_s5_d8_3x_000",
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
]

# RF time course column (STA)
RF_COL = "sta_time_course"

# Quality index column
QI_COL = "step_up_QI"

# Direction selectivity p-value column
DS_PVAL_COL = "ds_p_value"

# Axon type column
AXON_COL = "axon_type"

# Baseline trace column (used to compute baseline for all traces)
BASELINE_TRACE_COL = "step_up_5s_5i_b0_3x"

# All required trace columns (for feature extraction)
# Uses freq section columns instead of CHIRP_COL
REQUIRED_TRACE_COLS = FREQ_SECTION_COLS + [COLOR_COL, RF_COL] + BAR_COLS

# All required columns (including baseline trace)
REQUIRED_COLS = REQUIRED_TRACE_COLS + [BASELINE_TRACE_COL, QI_COL, DS_PVAL_COL, AXON_COL]

# =============================================================================
# Filter Thresholds
# =============================================================================

# Minimum quality index for cell inclusion
QI_THRESHOLD = 0.7

# P-value threshold for direction selectivity classification
DS_P_THRESHOLD = 0.05

# Valid axon types (RGC and AC)
VALID_AXON_TYPES = ["rgc", "ac"]

# Maximum baseline value for cell inclusion
# Cells with baseline (median of first 5 samples of filtered step_up trace) > this value are excluded
# Set to None to disable baseline filtering
BASELINE_MAX_THRESHOLD = 200.0

# Minimum number of good cells per batch for inclusion
# Batches with fewer than this many cells (after QI filtering) are excluded entirely
# Set to None or 0 to disable batch filtering
MIN_BATCH_GOOD_CELLS = 25

# =============================================================================
# Signal Processing
# =============================================================================

# Sampling rate in Hz (original)
SAMPLING_RATE = 60.0

# Low-pass filter cutoff frequency in Hz
LOWPASS_CUTOFF = 10.0

# Target sampling rate after downsampling (Hz)
TARGET_SAMPLING_RATE = 10.0

# Downsampling factor (SAMPLING_RATE / TARGET_SAMPLING_RATE)
DOWNSAMPLE_FACTOR = int(SAMPLING_RATE / TARGET_SAMPLING_RATE)  # = 6

# Butterworth filter order
FILTER_ORDER = 4

# Number of samples for baseline median calculation (at 10 Hz, after downsampling)
BASELINE_N_SAMPLES = 5

# Whether to apply baseline zeroing (subtraction) during preprocessing
# Set to False to skip baseline subtraction
APPLY_BASELINE_ZEROING = False

# Whether to apply max-abs normalization during preprocessing
# Set to False to skip normalization (keep original amplitude scale)
APPLY_MAX_ABS_NORMALIZATION = False

# Epsilon for numerical stability in normalization
NORMALIZE_EPS = 1e-8

# Legacy setting (kept for reference, not used)
BASELINE_SAMPLES_LEGACY = 8  # Old per-trace baseline at 60 Hz

# =============================================================================
# Sparse PCA Specifications
# =============================================================================

# Frequency section features: 4 components with 4 non-zero bins each (5 sections total)
# Replaces chirp features - extracts 20 features total (4 per section × 5 sections)
FREQ_SECTION_N_COMPONENTS = 4  # 4 components per frequency section
FREQ_SECTION_TOP_K = 4         # 4 non-zero bins per component
FREQ_SECTION_ALPHA = 1.0

# Special range for 10Hz section (skip first 60 and last 60 frames to exclude edge artifacts)
# Note: freq_section_10hz is NOT downsampled, so use original 60 Hz frame indices
FREQ_10HZ_START_OFFSET = 60
FREQ_10HZ_END_OFFSET = -60  # Negative means from end

# Legacy chirp settings (kept for reference, not used)
CHIRP_N_COMPONENTS = 20  # 20 used in Baden paper
CHIRP_TOP_K = 10
CHIRP_ALPHA = 1.0

# Color features: 6 components with 10 non-zero bins each
COLOR_N_COMPONENTS = 6
COLOR_TOP_K = 10
COLOR_ALPHA = 1.0

# Bar time course features: 8 components with 5 non-zero bins each
BAR_TC_N_COMPONENTS = 8
BAR_TC_TOP_K = 5
BAR_TC_ALPHA = 1.0

# Bar derivative features: 4 components with 6 non-zero bins each
BAR_DERIV_N_COMPONENTS = 4
BAR_DERIV_TOP_K = 6
BAR_DERIV_ALPHA = 1.0

# RF features: 2 regular PCA components
RF_N_COMPONENTS = 2

# Total feature count
# 5 freq sections × 4 components + 6 color + 8 bar TC + 4 bar deriv + 2 RF = 40
TOTAL_FEATURES = (
    len(FREQ_SECTION_COLS) * FREQ_SECTION_N_COMPONENTS +  # 5 × 4 = 20
    COLOR_N_COMPONENTS + 
    BAR_TC_N_COMPONENTS + 
    BAR_DERIV_N_COMPONENTS + 
    RF_N_COMPONENTS
)

# =============================================================================
# GMM Clustering Parameters
# =============================================================================

# Number of GMM restarts to find best solution
GMM_N_INIT = 20

# Regularization added to covariance diagonal
# Higher values = fewer clusters (penalizes complexity)
# Baden paper used 1e-5, but for over-clustering try 1e-3 or 1e-2
GMM_REG_COVAR = 1e-3

# Maximum cluster count for DS population
K_MAX_DS = 40

# Maximum cluster count for non-DS population
K_MAX_NDS = 80

# Log Bayes factor threshold for strong evidence # optional: 10.0 # default: 6.0
LOG_BF_THRESHOLD = 6.0 # 6.0 

# =============================================================================
# Bootstrap Stability Parameters
# =============================================================================

# Enable/disable bootstrap stability testing
# Set to False to skip bootstrap for faster runs (e.g., during development)
RUN_BOOTSTRAP = False

# Number of bootstrap iterations
BOOTSTRAP_N_ITERATIONS = 20

# Fraction of data to sample in each bootstrap
BOOTSTRAP_SAMPLE_FRACTION = 0.9

# Minimum stability correlation for stable clusters
STABILITY_THRESHOLD = 0.90

# =============================================================================
# GPU and Parallelization Settings
# =============================================================================

# Enable GPU acceleration (uses PyTorch CUDA for GMM fitting)
# Recommended: True if you have an NVIDIA GPU
USE_GPU = True

# Use float64 precision for GPU GMM (slower but more precise)
# Set to True for closer match to sklearn CPU results
# Set to False for faster GPU processing (default)
GPU_USE_FLOAT64 = False

# Fraction of CPU cores to use for parallel processing (0.0 to 1.0)
# Only used when N_JOBS_* != 1 and USE_GPU=False
CPU_FRACTION = 0.8

# Number of parallel jobs for BIC selection
# Set to 1 for sequential processing (recommended with GPU)
# Set to -1 for parallel CPU processing (requires working psutil)
N_JOBS_BIC = 1  # Sequential, GPU handles the speedup

# Number of parallel jobs for bootstrap
# Set to 1 for sequential processing (recommended with GPU)
# Set to -1 for parallel CPU processing (requires working psutil)
N_JOBS_BOOTSTRAP = 1  # Sequential, GPU handles the speedup

# =============================================================================
# UMAP Parameters
# =============================================================================

# Number of neighbors for UMAP
UMAP_N_NEIGHBORS = 15

# Minimum distance for UMAP
UMAP_MIN_DIST = 0.1

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration consistency."""
    # Note: Total features can vary based on component settings (Baden paper used 40)
    assert QI_THRESHOLD > 0, "QI threshold must be positive"
    assert 0 < DS_P_THRESHOLD < 1, "DS p-value threshold must be between 0 and 1"
    assert SAMPLING_RATE > 0, "Sampling rate must be positive"
    assert LOWPASS_CUTOFF > 0, "Lowpass cutoff must be positive"
    assert LOWPASS_CUTOFF < SAMPLING_RATE / 2, "Lowpass cutoff must be below Nyquist frequency"
    assert K_MAX_DS > 0, "K_MAX_DS must be positive"
    assert K_MAX_NDS > 0, "K_MAX_NDS must be positive"
    assert 0 < BOOTSTRAP_SAMPLE_FRACTION < 1, "Bootstrap sample fraction must be between 0 and 1"
    return True


# Run validation on import
validate_config()

