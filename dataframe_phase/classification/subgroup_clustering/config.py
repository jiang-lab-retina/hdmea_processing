"""
Shared Configuration for Subgroup Clustering Pipeline.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
INPUT_PARQUET = PROJECT_ROOT / "dataframe_phase/extract_feature/firing_rate_with_all_features_loaded_extracted20260102.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS_DIR = Path(__file__).parent / "models"
PLOTS_DIR = Path(__file__).parent / "validation" / "plots"
OPTUNA_DIR = Path(__file__).parent / "optuna_studies"

# =============================================================================
# SUBGROUP DEFINITIONS
# =============================================================================

SUBGROUPS = ["ipRGC", "DSGC", "OSGC", "Other"]

SUBGROUP_COLORS = {
    "ipRGC": "#E63946",   # Red
    "DSGC": "#457B9D",    # Steel blue
    "OSGC": "#2A9D8F",    # Teal
    "Other": "#8D99AE",   # Slate gray
}

# Classification thresholds
DS_P_THRESHOLD = 0.05
OS_P_THRESHOLD = 0.05
IPRGC_QI_THRESHOLD = 0.8

# =============================================================================
# EXPECTED CLUSTER RANGES (biologically motivated)
# =============================================================================

EXPECTED_K_RANGES = {
    "ipRGC": (6, 10),    # 6-10 clusters expected
    "DSGC": (8, 12),     # 8-12 clusters expected
    "OSGC": (8, 12),     # 4-12 clusters expected
    "Other": (12 , 24),   # 12-24 clusters expected
}

# =============================================================================
# MOVIE COLUMNS (Subgroup-Specific)
# =============================================================================

# Full movie columns (for ipRGC - includes iprgc_test)
# Uses corrected direction columns (angle-correction applied)
MOVIE_COLUMNS_FULL = [
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x",
    "step_up_5s_5i_b0_3x",
    "step_up_5s_5i_b0_30x",
    "corrected_moving_h_bar_s5_d8_3x_000",
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
    "iprgc_test",  # Included for ipRGC
]

# Reduced movie columns (for DSGC, OSGC, Other - excludes iprgc_test)
# Uses corrected direction columns (angle-correction applied)
MOVIE_COLUMNS_NO_IPRGC = [
    "freq_step_5st_3x",
    "green_blue_3s_3i_3x",
    "step_up_5s_5i_b0_3x",
    "step_up_5s_5i_b0_30x",
    "corrected_moving_h_bar_s5_d8_3x_000",
    "corrected_moving_h_bar_s5_d8_3x_045",
    "corrected_moving_h_bar_s5_d8_3x_090",
    "corrected_moving_h_bar_s5_d8_3x_135",
    "corrected_moving_h_bar_s5_d8_3x_180",
    "corrected_moving_h_bar_s5_d8_3x_225",
    "corrected_moving_h_bar_s5_d8_3x_270",
    "corrected_moving_h_bar_s5_d8_3x_315",
    # iprgc_test EXCLUDED for non-ipRGC subgroups
]

# Mapping subgroup to movie columns
SUBGROUP_MOVIE_COLUMNS = {
    "ipRGC": MOVIE_COLUMNS_FULL,
    "DSGC": MOVIE_COLUMNS_NO_IPRGC,
    "OSGC": MOVIE_COLUMNS_NO_IPRGC,
    "Other": MOVIE_COLUMNS_NO_IPRGC,
}

# Default (legacy compatibility)
MOVIE_COLUMNS = MOVIE_COLUMNS_FULL

# =============================================================================
# OPTIMIZED HYPERPARAMETERS (from Optuna optimization on 2025-12-30)
# =============================================================================

OPTIMIZED_PARAMS = {
    "ipRGC": {
        "latent_dim": 256,
        "learning_rate": 4.006e-05,
        "weight_decay": 5.433e-04,
        "batch_size": 256,
        "n_conv_layers": 3,
        "base_channels": 32,
        "dropout": 0.392,
        "optimal_k": 9,
        "silhouette": 0.131,
    },
    "DSGC": {
        "latent_dim": 32,
        "learning_rate": 1.284e-04,
        "weight_decay": 5.160e-06,
        "batch_size": 256,
        "n_conv_layers": 3,
        "base_channels": 16,
        "dropout": 0.228,
        "optimal_k": 8,
        "silhouette": 0.166,
    },
    "OSGC": {
        "latent_dim": 64,
        "learning_rate": 1.209e-05,
        "weight_decay": 5.233e-06,
        "batch_size": 128,
        "n_conv_layers": 2,
        "base_channels": 16,
        "dropout": 0.264,
        "optimal_k": 7,
        "silhouette": 0.291,
    },
    "Other": {
        "latent_dim": 32,
        "learning_rate": 1.590e-04,
        "weight_decay": 1.537e-05,
        "batch_size": 256,
        "n_conv_layers": 3,
        "base_channels": 16,
        "dropout": 0.435,
        "optimal_k": 12,
        "silhouette": 0.172,
    },
}

# =============================================================================
# DEFAULT MODEL HYPERPARAMETERS (fallback if not using optimized)
# =============================================================================

# Autoencoder settings
LATENT_DIM = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 300
TRAINING_PATIENCE = 30
BATCH_SIZE = 64
RANDOM_SEED = 42

# VAE settings
VAE_BETA = 1.0  # KL divergence weight

# Contrastive learning settings
CONTRASTIVE_WEIGHT = 0.3  # Lambda: weight for contrastive loss vs reconstruction
CONTRASTIVE_TEMPERATURE = 0.07  # Lower temperature = sharper similarity distribution
PROJECTION_DIM = 128  # Dimension of projection head output

# DEC settings
DEC_PRETRAIN_EPOCHS = 100
DEC_FINETUNE_EPOCHS = 200
DEC_UPDATE_INTERVAL = 10
DEC_TOL = 0.001  # Convergence tolerance

# =============================================================================
# HYPERPARAMETER OPTIMIZATION SETTINGS
# =============================================================================

OPTUNA_N_TRIALS = 50  # Number of optimization trials per subgroup
OPTUNA_TIMEOUT = 7200  # Max seconds per subgroup (2 hours)
OPTUNA_N_STARTUP_TRIALS = 10  # Random trials before Bayesian optimization

# Hyperparameter search spaces
HP_SEARCH_SPACE = {
    "latent_dim": (32, 256),           # Latent dimension range
    "learning_rate": (1e-5, 1e-3),     # Learning rate range (log scale)
    "weight_decay": (1e-6, 1e-3),      # Weight decay range (log scale)
    "batch_size": [32, 64, 128, 256],  # Batch size choices
    "n_conv_layers": (2, 4),           # Number of conv layers
    "base_channels": [16, 32, 64],     # Base channel count
    "dropout": (0.0, 0.5),             # Dropout rate
    "vae_beta": (0.01, 10.0),          # VAE KL weight (log scale)
}

# Quick optimization for testing
HP_QUICK_TRIALS = 15  # For fast testing

# =============================================================================
# CLUSTERING SETTINGS
# =============================================================================

GMM_K_MIN = 2
GMM_K_MAX = 25  # Upper bound, actual max from EXPECTED_K_RANGES
GMM_COVARIANCE_TYPE = "diag"  # 'diag' more stable for high-dim
GMM_REG_COVAR = 1e-5  # Regularization for stability

# Use K-Means for primary clustering (more stable than GMM for separation)
USE_KMEANS = True

# =============================================================================
# APPROACH NAMES
# =============================================================================

APPROACH_NAMES = {
    "ae_gmm": "Standard AE + K-Means",
    "vae_gmm": "VAE + K-Means", 
    "dec": "Deep Embedded Clustering",
    "contrastive_ae": "Contrastive AE + K-Means",
    "optimized_ae": "Optimized AE + K-Means",
}

# =============================================================================
# VISUALIZATION
# =============================================================================

# UMAP settings - optimized per subgroup for maximum cluster separation
UMAP_RANDOM_STATE = 42
UMAP_METRIC = "euclidean"

# Subgroup-specific UMAP parameters (optimized via grid search on 2D silhouette)
SUBGROUP_UMAP_PARAMS = {
    "ipRGC": {"n_neighbors": 25, "min_dist": 0.1, "spread": 0.5},
    "DSGC": {"n_neighbors": 25, "min_dist": 0.1, "spread": 0.5},
    "OSGC": {"n_neighbors": 25, "min_dist": 0.1, "spread": 0.5},
    "Other": {"n_neighbors": 25, "min_dist": 0.1, "spread": 0.5}, #30, 0.2, 0.5
}

# Default UMAP settings (fallback)
UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.1
UMAP_SPREAD = 0.5

# Cluster color palette (extended for up to 25 clusters)
CLUSTER_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700",
    "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12",
    "#1ABC9C", "#E67E22", "#34495E", "#7F8C8D", "#C0392B",
]
