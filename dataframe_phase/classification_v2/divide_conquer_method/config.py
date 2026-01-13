"""
Configuration for DEC-Refined RGC Subtype Clustering Pipeline.

All pipeline parameters are defined here for easy customization.
"""

from pathlib import Path
from typing import Dict, List

# ==============================================================================
# Paths
# ==============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # dataframe_phase -> ... -> project root
PACKAGE_DIR = Path(__file__).parent

# Default input file
INPUT_PATH = _PROJECT_ROOT / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260104.parquet"

# Output directories (relative to package)
OUTPUT_DIR = PACKAGE_DIR
RESULTS_DIR = PACKAGE_DIR / "results"
PLOTS_DIR = PACKAGE_DIR / "plots"
MODELS_DIR = PACKAGE_DIR / "models_saved"

# ==============================================================================
# Data Column Names
# ==============================================================================

# Required metadata columns
AXON_COL = "axon_type"
DS_PVAL_COL = "ds_p_value"
OS_PVAL_COL = "os_p_value"
IPRGC_QI_COL = "iprgc_2hz_QI"
STEP_UP_QI_COL = "step_up_QI"

# ==============================================================================
# Group Assignment Thresholds
# ==============================================================================

# DS > OS priority rule
DS_P_THRESHOLD = 0.05  # p-value threshold for DS classification
OS_P_THRESHOLD = 0.05  # p-value threshold for OS classification
IPRGC_QI_THRESHOLD = 0.8  # QI threshold for ipRGC validation

# Group names
GROUP_NAMES = ["DSGC", "OSGC", "Other"]
GROUP_PRIORITY = ["DS", "OS", "OTHER"]  # Priority order for assignment

# Minimum cells per group
MIN_GROUP_SIZE = 50

# ==============================================================================
# Preprocessing
# ==============================================================================

SAMPLING_RATE = 60.0  # Original sampling rate (Hz)

# Target rates after resampling
TARGET_RATE_DEFAULT = 10.0  # Most segments
TARGET_RATE_IPRGC = 2.0  # ipRGC test

# Low-pass filter cutoffs
LOWPASS_DEFAULT = 10.0  # Most segments (Hz)
LOWPASS_IPRGC = 4.0  # ipRGC test (Hz)

# Moving bar direction order
BAR_DIRECTIONS = ["000", "045", "090", "135", "180", "225", "270", "315"]

# Segment names (in processing order)
SEGMENT_NAMES = [
    "freq_section_0p5hz",
    "freq_section_1hz",
    "freq_section_2hz",
    "freq_section_4hz",
    "freq_section_10hz",
    "green_blue_3s_3i_3x",
    "bar_concat",  # Concatenated from 8 direction columns
    "sta_time_course",
    "iprgc_test",
    "step_up_5s_5i_b0_3x",
]

# Raw column names for moving bar (before concatenation)
BAR_COL_TEMPLATE = "corrected_moving_h_bar_s5_d8_3x_{direction}"

# ==============================================================================
# Autoencoder Architecture
# ==============================================================================

# Latent dimensions per segment (total = 49)
SEGMENT_LATENT_DIMS: Dict[str, int] = {
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

TOTAL_LATENT_DIM = sum(SEGMENT_LATENT_DIMS.values())  # 49

# CNN architecture
AE_HIDDEN_DIMS = [32, 64, 128]  # Conv channels
AE_KERNEL_SIZES = [7, 5, 3]  # Kernel sizes per layer
AE_DROPOUT = 0.1

# Training
AE_EPOCHS = 150
AE_BATCH_SIZE = 128
AE_LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15
VALIDATION_SPLIT = 0.15

# Device
DEVICE = "cuda"

# ==============================================================================
# GMM / BIC Clustering
# ==============================================================================

# Maximum k per group
K_MAX: Dict[str, int] = {
    "DSGC": 40,
    "OSGC": 20,
    "Other": 80,
}

# K range starts from 1
K_MIN = 1

# GMM fitting parameters
GMM_N_INIT = 20  # Number of random restarts
GMM_REG_COVAR = 1e-3  # Covariance regularization
GMM_MAX_ITER = 300  # Max EM iterations
GMM_TOL = 1e-4  # Convergence tolerance
GMM_USE_GPU = True  # Use GPU-accelerated GMM if available

# ==============================================================================
# DEC Refinement
# ==============================================================================

DEC_UPDATE_INTERVAL = 10  # Update target distribution every N iterations
DEC_MAX_ITERATIONS = 200  # Maximum DEC iterations
DEC_CONVERGENCE_THRESHOLD = 0.001  # Stop when assignment change < threshold
DEC_RECONSTRUCTION_WEIGHT = 0.1  # IDEC-style reconstruction term (gamma)
DEC_ALPHA = 1.0  # Student-t degrees of freedom
DEC_LEARNING_RATE = 1e-4  # Learning rate for DEC optimization

# ==============================================================================
# Visualization
# ==============================================================================

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = 42

# Plot settings
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

# ==============================================================================
# Logging
# ==============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
