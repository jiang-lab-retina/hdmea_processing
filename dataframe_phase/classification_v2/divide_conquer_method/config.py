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
INPUT_PATH = _PROJECT_ROOT / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted_area20260205.parquet"

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

# Group names (in processing order; assignment priority: ipRGC > DSGC > OSGC > Other)
GROUP_NAMES = ["ipRGC", "DSGC", "OSGC", "Other"]
GROUP_PRIORITY = ["IPRGC", "DS", "OS", "OTHER"]

# Minimum cells per group
MIN_GROUP_SIZE = 50

# ==============================================================================
# Cell Quality Filtering (matching Autoencoder_method)
# ==============================================================================

# Quality index threshold (step_up_QI) - cells below this are excluded
QI_THRESHOLD = 0.5  # Same as Autoencoder_method

# Maximum baseline firing rate (Hz) - high baseline indicates noise/artifacts
BASELINE_MAX_THRESHOLD = 200.0  # Same as Autoencoder_method

# Minimum cells per batch after filtering
MIN_BATCH_GOOD_CELLS = 25  # Same as Autoencoder_method

# Baseline trace column (used to compute baseline)
BASELINE_TRACE_COL = "step_up_5s_5i_b0_3x"

# ==============================================================================
# Preprocessing
# ==============================================================================

SAMPLING_RATE = 60.0  # Original sampling rate (Hz)

# Target rates after resampling
TARGET_RATE_DEFAULT = 10.0  # Most segments (downsampled from 60 Hz)
TARGET_RATE_IPRGC = 10.0   # ipRGC test

# Low-pass filter cutoffs
LOWPASS_DEFAULT = 10.0  # Most segments (Hz)
LOWPASS_IPRGC = 10.0  # ipRGC test (Hz)

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

# Trace normalization (applied before autoencoder)
NORMALIZE_TRACES = False  # Whether to normalize traces
NORMALIZE_METHOD = "zscore"  # Options: "zscore", "maxabs", "minmax", "baseline"
# - "zscore": Per-cell z-score (mean=0, std=1) - recommended for AE
# - "maxabs": Divide by max absolute value
# - "minmax": Scale to [0, 1]
# - "baseline": Subtract baseline only

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

# ==============================================================================
# Encoder Architecture Selection
# ==============================================================================
# Options: "tcn" (default), "cnn", "multiscale"
# - "tcn": Temporal Convolutional Network with dilated convolutions
#          Best for neural time series with multi-scale temporal patterns
# - "cnn": Standard 1D CNN (simpler, faster training)
# - "multiscale": Parallel branches with different kernel sizes
ENCODER_TYPE = "tcn"

# TCN-specific parameters (used when ENCODER_TYPE = "tcn")
# Reduced complexity: 2 blocks instead of 4, fewer channels
TCN_CHANNELS = [16, 32]  # Channels per TCN block (dilation 1,2)
TCN_KERNEL_SIZE = 5      # Slightly larger kernel to compensate for fewer blocks

# Multi-scale specific parameters (used when ENCODER_TYPE = "multiscale")
MULTISCALE_KERNEL_SIZES = [3, 7, 15]  # Fast, medium, slow timescales
MULTISCALE_CHANNELS = 32              # Channels per branch

# CNN-specific parameters (used when ENCODER_TYPE = "cnn")
AE_HIDDEN_DIMS = [32, 64, 128]   # Conv channels (was [8,16,32], increased for capacity)
AE_KERNEL_SIZES = [7, 5, 3]      # Kernel sizes per layer (large→small)

# Common parameters
AE_DROPOUT = 0.1
USE_MLP_THRESHOLD = 30  # Use MLP for segments shorter than this (samples)

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
    "ipRGC": 20,
    "DSGC": 20,
    "OSGC": 20,
    "Other": 20,
}

# K range starts from 1
K_MIN = 1

# GMM fitting parameters
GMM_N_INIT = 20  # 20 default of random restarts
GMM_REG_COVAR = 1e-1  # 1e-3 for Baden, 1e-2 for current Covariance regularization
GMM_MAX_ITER = 300  # Max EM iterations
GMM_TOL = 1e-4  # Convergence tolerance
GMM_USE_GPU = True  # Use GPU-accelerated GMM if available

# K-selection method
K_SELECTION_METHOD = "elbow"  # "min" for pure minimum BIC, "elbow" for elbow detection
ELBOW_THRESHOLD = 0.03  # Minimum relative BIC improvement to continue (3%)
                        # Higher = fewer clusters, lower = more clusters

# ==============================================================================
# DEC Refinement
# ==============================================================================

DEC_UPDATE_INTERVAL = 10  # Update target distribution every N iterations
DEC_MAX_ITERATIONS = 200  # Maximum DEC iterations (increased from 50 for convergence)
DEC_MIN_ITERATIONS = 20   # Minimum iterations before checking convergence
DEC_CONVERGENCE_THRESHOLD = 0.001  # Stop when assignment change < threshold
DEC_RECONSTRUCTION_WEIGHT = 0.01  # Low = allow embeddings to move toward cluster centers
DEC_BALANCE_WEIGHT = 5.0  # Balance regularization (increased from 1.0 to prevent collapse)
DEC_ALPHA = 1.0  # Lower = sharper assignments = tighter clusters
DEC_LEARNING_RATE = 1e-4  # Higher LR for faster tightening
DEC_ALLOW_REASSIGNMENT = True  # True = allow DEC to reassign cells between clusters

# ipRGC Enrichment (Other group only)
IPRGC_ENRICHMENT_WEIGHT = 0.0    # DEC-level enrichment loss (disabled; semi-supervised AE handles this)
IPRGC_N_TARGET_CLUSTERS = 3      # Number of target ipRGC-enriched subtypes

# Semi-supervised autoencoder (disabled; ipRGC is now its own group)
IPRGC_CLASSIFICATION_WEIGHT = 0.0  # Disabled: ipRGC cells have their own group
IPRGC_CLASSIFIER_HIDDEN = 32      # Hidden layer size for classification head

# ==============================================================================
# Visualization
# ==============================================================================

# UMAP parameters
UMAP_N_NEIGHBORS = 25  # Higher = outliers pull toward cluster; lower = more local structure
UMAP_MIN_DIST = 0.6  # Lower = points pack closer to cluster centers (0 = tightest)
UMAP_SPREAD = 1.0  # Lower = more compact clusters, fewer distant outliers
UMAP_REPULSION_STRENGTH = 5  # Lower = less push-apart, points stay near neighbors
UMAP_LOCAL_CONNECTIVITY = 5  # Higher = boundary points stay with cluster (default 1.0)
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = 42
UMAP_TARGET_WEIGHT = 0.8  # Supervised UMAP: 0 = unsupervised, 1 = strongly supervised
UMAP_SHOW_IPRGC = False  # Highlight ipRGC cells on UMAP plots

# Plot settings
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

# ==============================================================================
# Logging
# ==============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
