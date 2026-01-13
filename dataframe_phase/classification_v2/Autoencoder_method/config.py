"""
Configuration constants for the Autoencoder-based RGC clustering pipeline.

All pipeline parameters are centralized here for easy modification.
Based on research.md design decisions and spec.md requirements.
"""

from pathlib import Path

# =============================================================================
# Paths (all absolute, based on this file's location)
# =============================================================================

# This file's directory (Autoencoder_method folder)
_THIS_DIR = Path(__file__).parent.resolve()

# Project root (4 levels up from this file)
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent

# Default input path (absolute)
INPUT_PATH = _PROJECT_ROOT / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet"

# Output directory (this folder)
OUTPUT_DIR = _THIS_DIR

# Subdirectories
MODELS_DIR = OUTPUT_DIR / "models_saved"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

# =============================================================================
# Column Names
# =============================================================================

# Frequency section trace columns
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

# ipRGC test trace column
IPRGC_COL = "iprgc_test"

# Step-up stimulus column
STEP_UP_COL = "step_up_5s_5i_b0_3x"

# Quality index column
QI_COL = "step_up_QI"

# Direction selectivity p-value column
DS_PVAL_COL = "ds_p_value"

# Axon type column
AXON_COL = "axon_type"

# ipRGC quality index column
IPRGC_QI_COL = "iprgc_2hz_QI"

# Baseline trace column (used to compute baseline for all traces)
BASELINE_TRACE_COL = "step_up_5s_5i_b0_3x"

# All required trace columns for AE (10 segments)
REQUIRED_TRACE_COLS = (
    FREQ_SECTION_COLS + 
    [COLOR_COL] + 
    BAR_COLS + 
    [RF_COL, IPRGC_COL, STEP_UP_COL]
)

# All required columns (including metadata)
REQUIRED_COLS = REQUIRED_TRACE_COLS + [
    BASELINE_TRACE_COL, QI_COL, DS_PVAL_COL, AXON_COL, IPRGC_QI_COL
]

# =============================================================================
# Segment Configuration for Autoencoder
# =============================================================================

# Segment names in fixed order for concatenation
SEGMENT_NAMES = [
    "freq_section_0p5hz",
    "freq_section_1hz",
    "freq_section_2hz",
    "freq_section_4hz",
    "freq_section_10hz",
    "green_blue_3s_3i_3x",
    "bar_concat",  # 8 directions concatenated
    "sta_time_course",
    "iprgc_test",
    "step_up_5s_5i_b0_3x",
]

# Fixed latent dimensions per segment (total = 49)
SEGMENT_LATENT_DIMS = {
    "freq_section_0p5hz": 4,
    "freq_section_1hz": 4,
    "freq_section_2hz": 4,
    "freq_section_4hz": 4,
    "freq_section_10hz": 4,
    "green_blue_3s_3i_3x": 6,
    "bar_concat": 12,
    "sta_time_course": 3,
    "iprgc_test": 4,
    "step_up_5s_5i_b0_3x": 4,
}

# Total latent dimension
TOTAL_LATENT_DIM = sum(SEGMENT_LATENT_DIMS.values())  # = 49

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
BASELINE_MAX_THRESHOLD = 200.0

# Minimum number of good cells per batch for inclusion
MIN_BATCH_GOOD_CELLS = 25

# ipRGC QI threshold for group assignment
IPRGC_QI_THRESHOLD = 0.8

# Minimum cells per group for GMM fitting
MIN_CELLS_PER_GROUP = 50

# =============================================================================
# Signal Processing
# =============================================================================

# Sampling rate in Hz (original)
SAMPLING_RATE = 60.0

# Low-pass filter cutoff frequency in Hz (default for most segments)
LOWPASS_CUTOFF = 10.0

# Target sampling rate after downsampling (Hz) for most segments
TARGET_SAMPLING_RATE = 10.0

# Downsampling factor (SAMPLING_RATE / TARGET_SAMPLING_RATE)
DOWNSAMPLE_FACTOR = int(SAMPLING_RATE / TARGET_SAMPLING_RATE)  # = 6

# Butterworth filter order
FILTER_ORDER = 4

# Number of samples for baseline median calculation (at 10 Hz, after downsampling)
BASELINE_N_SAMPLES = 5

# ipRGC-specific processing
IPRGC_LOWPASS_CUTOFF = 2.0  # 2 Hz low-pass for ipRGC traces
IPRGC_TARGET_RATE = 2.0  # Downsample to 2 Hz

# 10 Hz section edge slicing (skip first/last 1 second at 60 Hz)
FREQ_10HZ_START_OFFSET = 60
FREQ_10HZ_END_OFFSET = -60

# Segments that are NOT filtered or resampled:
# - freq_section_10hz: preserves high-frequency dynamics (only edge slicing)
# - sta_time_course: already smooth, no processing needed

# =============================================================================
# Coarse Group Configuration
# =============================================================================

# Group precedence order (first match wins)
# Options: "ac", "iprgc", "ds", "nonds"
GROUP_PRECEDENCE = ["ac", "iprgc", "ds", "nonds"]

# Group names for output
GROUP_NAMES = {
    "ac": "AC",
    "iprgc": "ipRGC",
    "ds": "DS-RGC",
    "nonds": "nonDS-RGC",
}

# =============================================================================
# Autoencoder Architecture
# =============================================================================

# Encoder/decoder hidden dimensions
AE_HIDDEN_DIMS = [32, 64, 128]

# Dropout probability
AE_DROPOUT = 0.1

# =============================================================================
# Autoencoder Training
# =============================================================================

# Number of training epochs
AE_EPOCHS = 150 #100

# Training batch size
AE_BATCH_SIZE = 128 #128

# Learning rate (reduced for stability)
AE_LEARNING_RATE = 1e-4

# Weight decay (L2 regularization)
AE_WEIGHT_DECAY = 1e-5

# Supervised contrastive loss weight (β)
SUPCON_WEIGHT = 1.0 # 0.1

# SupCon temperature (τ)
SUPCON_TEMPERATURE = 0.05 # 0.1

# Early stopping patience (epochs without improvement)
EARLY_STOPPING_PATIENCE = 15

# Device for training ("cuda" or "cpu")
DEVICE = "cuda"

# =============================================================================
# Cluster Purity Loss Parameters
# =============================================================================

# Enable/disable purity loss (replaces or augments SupCon)
USE_PURITY_LOSS = True

# Purity loss weight (α) - higher = stronger purity enforcement
PURITY_LOSS_WEIGHT = 1.0

# Number of soft clusters for purity computation
PURITY_N_CLUSTERS = 100

# Temperature for soft cluster assignments (lower = sharper)
PURITY_TEMPERATURE = 1.0

# Labels to optimize purity for (column names in DataFrame)
PURITY_LABELS = ["axon_type", "ds_cell", "iprgc"]

# =============================================================================
# GMM Clustering Parameters
# =============================================================================

# Number of GMM restarts to find best solution
GMM_N_INIT = 20

# Regularization added to covariance diagonal
GMM_REG_COVAR = 1e-3

# Maximum cluster count per group
K_MAX = {
    "AC": 40,
    "ipRGC": 10,
    "DS-RGC": 40,
    "nonDS-RGC": 80,
}

# Default k_max for unknown groups
K_MAX_DEFAULT = 40

# Log Bayes factor threshold for strong evidence
LOG_BF_THRESHOLD = 6.0

# Minimum cluster size
MIN_CLUSTER_SIZE = 10

# Use GPU-accelerated GMM (PyTorch) when CUDA is available
GMM_USE_GPU = True

# =============================================================================
# Bootstrap Stability Parameters
# =============================================================================

# Enable/disable bootstrap stability testing
RUN_BOOTSTRAP = True

# Number of bootstrap iterations
BOOTSTRAP_N_ITERATIONS = 20

# Fraction of data to sample in each bootstrap
BOOTSTRAP_SAMPLE_FRACTION = 0.9

# Minimum stability correlation for stable clusters
STABILITY_THRESHOLD = 0.8

# Random seed for reproducibility
BOOTSTRAP_RANDOM_SEED = 42

# =============================================================================
# Cross-Validation Parameters
# =============================================================================

# Enable/disable CV turns
RUN_CV_TURNS = True

# CV turn definitions
CV_TURNS = [
    {"omit": "axon_type", "active": ["ds_cell", "iprgc"]},
    {"omit": "ds_cell", "active": ["axon_type", "iprgc"]},
    {"omit": "iprgc", "active": ["axon_type", "ds_cell"]},
]

# Minimum CVScore threshold for acceptable generalization
CVSCORE_THRESHOLD = 0.7

# =============================================================================
# UMAP Parameters
# =============================================================================

# Number of neighbors for UMAP
UMAP_N_NEIGHBORS = 15

# Minimum distance for UMAP
UMAP_MIN_DIST = 0.1

# Random state for UMAP reproducibility
UMAP_RANDOM_STATE = 42

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration consistency."""
    assert QI_THRESHOLD > 0, "QI threshold must be positive"
    assert 0 < DS_P_THRESHOLD < 1, "DS p-value threshold must be between 0 and 1"
    assert SAMPLING_RATE > 0, "Sampling rate must be positive"
    assert LOWPASS_CUTOFF > 0, "Lowpass cutoff must be positive"
    assert LOWPASS_CUTOFF < SAMPLING_RATE / 2, "Lowpass cutoff must be below Nyquist frequency"
    assert TOTAL_LATENT_DIM == 49, f"Total latent dim must be 49, got {TOTAL_LATENT_DIM}"
    assert 0 < BOOTSTRAP_SAMPLE_FRACTION < 1, "Bootstrap sample fraction must be between 0 and 1"
    assert SUPCON_WEIGHT >= 0, "SupCon weight must be non-negative"
    assert SUPCON_TEMPERATURE > 0, "SupCon temperature must be positive"
    return True


# Run validation on import
validate_config()
