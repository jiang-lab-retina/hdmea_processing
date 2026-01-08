"""
Feature extraction module for the Baden-method RGC clustering pipeline.

This module implements:
- Sparse PCA with hard sparsity enforcement
- Feature extraction from frequency sections, color, moving bar, and RF time course stimuli
- Feature matrix construction and standardization

The pipeline extracts a 40-dimensional feature vector per cell:
- 20 features from frequency sections (4 components × 4 non-zero bins × 5 sections)
  - 10 Hz section uses frames [60:-60] to exclude edge artifacts
- 6 features from color (10 non-zero bins each)
- 8 features from bar time course (5 non-zero bins each)
- 4 features from bar derivative (6 non-zero bins each)
- 2 features from RF time course (regular PCA)
"""

import logging
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import StandardScaler

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# Sparse PCA with Hard Sparsity
# =============================================================================

def _enforce_hard_sparsity(
    components: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """
    Enforce hard sparsity by keeping only top-k weights per component.
    
    Args:
        components: Component matrix of shape (n_components, n_features).
        top_k: Number of non-zero weights to keep per component.
        
    Returns:
        Sparse component matrix with exactly top_k non-zero values per row.
    """
    components = components.copy()
    n_components, n_features = components.shape
    
    for i in range(n_components):
        comp = components[i]
        # Get indices of top-k by absolute value
        abs_vals = np.abs(comp)
        if len(abs_vals) <= top_k:
            # Keep all if fewer features than top_k
            continue
        
        threshold_idx = np.argsort(abs_vals)[-top_k:]
        mask = np.zeros(n_features, dtype=bool)
        mask[threshold_idx] = True
        
        # Zero out non-top-k values
        components[i] = np.where(mask, comp, 0.0)
        
        # Renormalize to unit length
        norm = np.linalg.norm(components[i])
        if norm > 0:
            components[i] /= norm
    
    return components


def fit_sparse_pca(
    X: np.ndarray,
    n_components: int,
    top_k: int,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Tuple[SparsePCA, np.ndarray]:
    """
    Fit sparse PCA with enforced hard sparsity constraints.
    
    Args:
        X: Data matrix of shape (n_samples, n_features).
        n_components: Number of components to extract.
        top_k: Number of non-zero weights per component.
        alpha: Sparsity controlling parameter for initial fit.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (fitted SparsePCA model, transformed scores).
    """
    logger.debug(f"Fitting sparse PCA: {n_components} components, top_k={top_k}")
    
    # Fit SparsePCA
    spca = SparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=random_state,
        max_iter=500,
    )
    
    # Fit on data
    spca.fit(X)
    
    # Enforce hard sparsity
    original_components = spca.components_.copy()
    spca.components_ = _enforce_hard_sparsity(spca.components_, top_k)
    
    # Verify sparsity
    for i, comp in enumerate(spca.components_):
        n_nonzero = np.count_nonzero(comp)
        if n_nonzero != top_k and n_nonzero != len(comp):
            logger.warning(f"Component {i} has {n_nonzero} non-zero weights, expected {top_k}")
    
    # Transform data
    scores = X @ spca.components_.T
    
    return spca, scores


# =============================================================================
# Stimulus-Specific Feature Extraction
# =============================================================================

def _traces_to_matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Convert trace column to 2D matrix.
    
    Handles both flat arrays and nested trial arrays (averages trials if needed).
    Also handles variable-length traces by truncating to minimum length.
    """
    from . import preprocessing
    
    traces = df[column].values
    processed = []
    for t in traces:
        # Average trials if nested, otherwise just convert to float64
        avg_trace = preprocessing.average_trials(t)
        processed.append(avg_trace)
    
    # Handle variable length traces by truncating to minimum length
    min_len = min(len(t) for t in processed)
    processed = [t[:min_len] for t in processed]
    
    return np.vstack(processed)


def _traces_to_matrix_with_slice(
    df: pd.DataFrame, 
    column: str,
    start_offset: int = None,
    end_offset: int = None,
) -> np.ndarray:
    """
    Convert trace column to 2D matrix with optional slicing.
    
    Args:
        df: DataFrame with trace column.
        column: Column name.
        start_offset: Start index for slicing (default: 0).
        end_offset: End index for slicing (default: None = full length).
                   Negative values slice from end.
    """
    from . import preprocessing
    
    traces = df[column].values
    processed = []
    for t in traces:
        # Average trials if nested, otherwise just convert to float64
        avg_trace = preprocessing.average_trials(t)
        
        # Apply slicing if specified
        if start_offset is not None or end_offset is not None:
            start = start_offset if start_offset is not None else 0
            end = end_offset if end_offset is not None else len(avg_trace)
            avg_trace = avg_trace[start:end]
        
        processed.append(avg_trace)
    
    # Handle variable length traces by truncating to minimum length
    min_len = min(len(t) for t in processed)
    processed = [t[:min_len] for t in processed]
    
    return np.vstack(processed)


def extract_freq_section_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, SparsePCA]]:
    """
    Extract sparse PCA features from each frequency section.
    
    Each section (0.5, 1, 2, 4, 10 Hz) contributes 4 components with 4 non-zero bins,
    for a total of 20 features.
    
    For the 10 Hz section, a special range is applied (frames 60 to -60) to
    exclude edge artifacts.
    
    Args:
        df: DataFrame with preprocessed frequency section traces.
        random_state: Random seed.
        
    Returns:
        Tuple of (feature matrix of shape (N, 20), dict of fitted SparsePCA models).
    """
    logger.info(f"Extracting freq section features from {len(df)} cells...")
    
    all_scores = []
    all_models = {}
    
    for col in config.FREQ_SECTION_COLS:
        logger.debug(f"Processing {col}...")
        
        # Check if this is the 10Hz section (special range handling)
        if col == "freq_section_10hz":
            # Use special range for 10Hz: [60:-60]
            X = _traces_to_matrix_with_slice(
                df, col,
                start_offset=config.FREQ_10HZ_START_OFFSET,
                end_offset=config.FREQ_10HZ_END_OFFSET,
            )
            logger.debug(f"  {col} trace matrix shape (sliced [60:-60]): {X.shape}")
        else:
            # Build trace matrix for this section (full trace)
            X = _traces_to_matrix(df, col)
            logger.debug(f"  {col} trace matrix shape: {X.shape}")
        
        # Fit sparse PCA for this section
        spca, scores = fit_sparse_pca(
            X,
            n_components=config.FREQ_SECTION_N_COMPONENTS,
            top_k=config.FREQ_SECTION_TOP_K,
            alpha=config.FREQ_SECTION_ALPHA,
            random_state=random_state,
        )
        
        all_scores.append(scores)
        all_models[col] = spca
    
    # Concatenate all section scores
    combined_scores = np.hstack(all_scores)
    logger.info(f"Extracted {combined_scores.shape[1]} freq section features (5 sections × {config.FREQ_SECTION_N_COMPONENTS} components)")
    
    return combined_scores, all_models


# Legacy function kept for reference
def extract_chirp_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, SparsePCA]:
    """
    [LEGACY] Extract 20 sparse PCA features from chirp responses.
    
    This function is kept for reference but is no longer used.
    Use extract_freq_section_features instead.
    
    Args:
        df: DataFrame with preprocessed chirp traces.
        random_state: Random seed.
        
    Returns:
        Tuple of (feature matrix of shape (N, 20), fitted SparsePCA model).
    """
    logger.info(f"Extracting chirp features from {len(df)} cells...")
    
    # Build trace matrix
    X = _traces_to_matrix(df, config.CHIRP_COL)
    logger.debug(f"Chirp trace matrix shape: {X.shape}")
    
    # Fit sparse PCA
    spca, scores = fit_sparse_pca(
        X,
        n_components=config.CHIRP_N_COMPONENTS,
        top_k=config.CHIRP_TOP_K,
        alpha=config.CHIRP_ALPHA,
        random_state=random_state,
    )
    
    logger.info(f"Extracted {scores.shape[1]} chirp features")
    return scores, spca


def extract_color_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, SparsePCA]:
    """
    Extract 6 sparse PCA features from color responses.
    
    Args:
        df: DataFrame with preprocessed color traces.
        random_state: Random seed.
        
    Returns:
        Tuple of (feature matrix of shape (N, 6), fitted SparsePCA model).
    """
    logger.info(f"Extracting color features from {len(df)} cells...")
    
    # Build trace matrix
    X = _traces_to_matrix(df, config.COLOR_COL)
    logger.debug(f"Color trace matrix shape: {X.shape}")
    
    # Fit sparse PCA
    spca, scores = fit_sparse_pca(
        X,
        n_components=config.COLOR_N_COMPONENTS,
        top_k=config.COLOR_TOP_K,
        alpha=config.COLOR_ALPHA,
        random_state=random_state,
    )
    
    logger.info(f"Extracted {scores.shape[1]} color features")
    return scores, spca


def extract_bar_svd(
    cell_bar_traces: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract temporal component from 8-direction moving bar responses using SVD.
    
    Args:
        cell_bar_traces: Dictionary mapping column name to trace array.
                        Traces may be nested arrays (multiple trials) which will be averaged.
        
    Returns:
        Tuple of (time_course, tuning_curve).
    """
    from . import preprocessing
    
    # Stack traces: (T, 8)
    directions = ['000', '045', '090', '135', '180', '225', '270', '315']
    traces = []
    
    for d in directions:
        col_name = f"corrected_moving_h_bar_s5_d8_3x_{d}"
        if col_name in cell_bar_traces:
            trace = cell_bar_traces[col_name]
        else:
            # Try without prefix
            trace = cell_bar_traces[d]
        
        # Average trials if needed
        avg_trace = preprocessing.average_trials(trace)
        traces.append(avg_trace)
    
    # Handle variable length traces by truncating to minimum length
    min_len = min(len(t) for t in traces)
    traces = [t[:min_len] for t in traces]
    
    traces = np.column_stack(traces)  # (T, 8)
    
    # Normalize per cell (single scaling factor for entire matrix)
    # This preserves relative amplitudes across directions (directional tuning)
    max_abs = np.abs(traces).max() + 1e-8
    traces_norm = traces / max_abs
    
    # SVD
    U, S, Vt = np.linalg.svd(traces_norm, full_matrices=False)
    
    # First temporal component (scaled by singular value)
    time_course = U[:, 0] * S[0]
    tuning_curve = Vt[0, :]
    
    return time_course, tuning_curve


def extract_bar_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, SparsePCA, SparsePCA]:
    """
    Extract features from moving bar responses.
    
    Uses SVD to extract temporal component, then applies sparse PCA to:
    - Time course: 8 features with 5 non-zero bins each
    - Derivative: 4 features with 6 non-zero bins each
    
    Args:
        df: DataFrame with preprocessed bar traces.
        random_state: Random seed.
        
    Returns:
        Tuple of (combined features (N, 12), time_course_spca, derivative_spca).
    """
    logger.info(f"Extracting bar features from {len(df)} cells...")
    
    # Extract time courses via SVD for each cell
    time_courses = []
    
    for idx in df.index:
        # Get traces for this cell
        cell_traces = {}
        for col in config.BAR_COLS:
            cell_traces[col] = df.loc[idx, col]
        
        tc, _ = extract_bar_svd(cell_traces)
        time_courses.append(tc)
    
    # Stack time courses
    X_tc = np.vstack(time_courses)
    logger.debug(f"Bar time course matrix shape: {X_tc.shape}")
    
    # Compute derivative
    X_deriv = np.diff(X_tc, axis=1)
    logger.debug(f"Bar derivative matrix shape: {X_deriv.shape}")
    
    # Sparse PCA on time course
    spca_tc, scores_tc = fit_sparse_pca(
        X_tc,
        n_components=config.BAR_TC_N_COMPONENTS,
        top_k=config.BAR_TC_TOP_K,
        alpha=config.BAR_TC_ALPHA,
        random_state=random_state,
    )
    
    # Sparse PCA on derivative
    spca_deriv, scores_deriv = fit_sparse_pca(
        X_deriv,
        n_components=config.BAR_DERIV_N_COMPONENTS,
        top_k=config.BAR_DERIV_TOP_K,
        alpha=config.BAR_DERIV_ALPHA,
        random_state=random_state + 1,  # Different seed
    )
    
    # Combine
    features = np.hstack([scores_tc, scores_deriv])
    logger.info(f"Extracted {features.shape[1]} bar features (8 TC + 4 derivative)")
    
    return features, spca_tc, spca_deriv


def extract_rf_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Extract 2 regular PCA features from RF time course.
    
    Args:
        df: DataFrame with RF time course (sta_time_course).
        random_state: Random seed.
        
    Returns:
        Tuple of (feature matrix of shape (N, 2), fitted PCA model).
    """
    logger.info(f"Extracting RF features from {len(df)} cells...")
    
    # Build trace matrix
    X = _traces_to_matrix(df, config.RF_COL)
    logger.debug(f"RF trace matrix shape: {X.shape}")
    
    # Regular PCA
    pca = PCA(n_components=config.RF_N_COMPONENTS, random_state=random_state)
    scores = pca.fit_transform(X)
    
    logger.info(f"Extracted {scores.shape[1]} RF features (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    return scores, pca


# =============================================================================
# Feature Matrix Construction
# =============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    random_state: int = 42,
    return_models: bool = False,
) -> np.ndarray | Tuple[np.ndarray, list, dict]:
    """
    Build the full feature matrix from all stimuli.
    
    Args:
        df: DataFrame with preprocessed traces.
        random_state: Random seed for reproducibility.
        return_models: If True, also return feature names and fitted models.
        
    Returns:
        If return_models=False: Feature matrix of shape (N, TOTAL_FEATURES).
        If return_models=True: Tuple of (features, feature_names, models_dict).
    """
    logger.info(f"Building {config.TOTAL_FEATURES}D feature matrix for {len(df)} cells...")
    
    # Extract features from each stimulus
    freq_features, spca_freq = extract_freq_section_features(df, random_state)
    color_features, spca_color = extract_color_features(df, random_state + 100)
    bar_features, spca_bar_tc, spca_bar_deriv = extract_bar_features(df, random_state + 200)
    rf_features, pca_rf = extract_rf_features(df, random_state + 300)
    
    # Concatenate all features
    features = np.hstack([
        freq_features,    # 20 (4 per freq section × 5 sections)
        color_features,   # 6
        bar_features,     # 12 (8 + 4)
        rf_features,      # 2
    ])
    
    # Validate
    n_features = features.shape[1]
    if n_features != config.TOTAL_FEATURES:
        raise ValueError(f"Expected {config.TOTAL_FEATURES} features, got {n_features}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(features)):
        n_nan = np.isnan(features).sum()
        raise ValueError(f"Feature matrix contains {n_nan} NaN values")
    if np.any(np.isinf(features)):
        n_inf = np.isinf(features).sum()
        raise ValueError(f"Feature matrix contains {n_inf} Inf values")
    
    logger.info(f"Built feature matrix: {features.shape}")
    
    if not return_models:
        return features
    
    # Build feature names
    freq_section_names = []
    for col in config.FREQ_SECTION_COLS:
        # Extract frequency label from column name (e.g., "freq_section_0p5hz" -> "0p5hz")
        freq_label = col.replace("freq_section_", "")
        for i in range(config.FREQ_SECTION_N_COMPONENTS):
            freq_section_names.append(f"freq_{freq_label}_{i}")
    
    feature_names = (
        freq_section_names +
        [f"color_{i}" for i in range(config.COLOR_N_COMPONENTS)] +
        [f"bar_tc_{i}" for i in range(config.BAR_TC_N_COMPONENTS)] +
        [f"bar_deriv_{i}" for i in range(config.BAR_DERIV_N_COMPONENTS)] +
        [f"rf_{i}" for i in range(config.RF_N_COMPONENTS)]
    )
    
    # Pack models
    models = {
        'spca_freq': spca_freq,  # Dict of models for each freq section
        'spca_color': spca_color,
        'spca_bar_tc': spca_bar_tc,
        'spca_bar_deriv': spca_bar_deriv,
        'pca_rf': pca_rf,
    }
    
    return features, feature_names, models


def standardize_features(
    X: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Z-score standardize features.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        
    Returns:
        Tuple of (standardized features, fitted StandardScaler).
    """
    logger.info(f"Standardizing features: {X.shape}")
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Verify standardization
    means = X_std.mean(axis=0)
    stds = X_std.std(axis=0)
    
    if np.any(np.abs(means) > 1e-10):
        logger.warning(f"Feature means not zero after standardization: max={np.abs(means).max():.2e}")
    
    logger.info("Features standardized")
    return X_std, scaler


# =============================================================================
# Utilities
# =============================================================================

def transform_new_data(
    df: pd.DataFrame,
    models: Dict[str, Any],
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Transform new data using pre-fitted models.
    
    Args:
        df: DataFrame with preprocessed traces.
        models: Dictionary of fitted PCA/SparsePCA models.
        scaler: Fitted StandardScaler.
        
    Returns:
        Standardized feature matrix.
    """
    # Extract features using fitted models
    chirp_scores = _traces_to_matrix(df, config.CHIRP_COL) @ models['spca_chirp'].components_.T
    color_scores = _traces_to_matrix(df, config.COLOR_COL) @ models['spca_color'].components_.T
    
    # Bar features need SVD + transform
    time_courses = []
    for idx in df.index:
        cell_traces = {col: df.loc[idx, col] for col in config.BAR_COLS}
        tc, _ = extract_bar_svd(cell_traces)
        time_courses.append(tc)
    
    X_tc = np.vstack(time_courses)
    X_deriv = np.diff(X_tc, axis=1)
    
    bar_tc_scores = X_tc @ models['spca_bar_tc'].components_.T
    bar_deriv_scores = X_deriv @ models['spca_bar_deriv'].components_.T
    
    rf_scores = models['pca_rf'].transform(_traces_to_matrix(df, config.RF_COL))
    
    # Combine and standardize
    features = np.hstack([chirp_scores, color_scores, bar_tc_scores, bar_deriv_scores, rf_scores])
    features_std = scaler.transform(features)
    
    return features_std

