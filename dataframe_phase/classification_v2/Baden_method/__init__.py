"""
Baden-Method RGC Clustering Pipeline

This package implements an unsupervised RGC classification pipeline
reproducing the methodology described in Baden et al. The pipeline:
- Extracts a 40-dimensional feature vector from four visual stimuli
- Clusters cells using Gaussian Mixture Models with BIC-based model selection
- Evaluates cluster quality through posterior probability and bootstrap stability
- Processes DS and non-DS cells independently

Usage:
    from dataframe_phase.classification_v2.Baden_method import run_baden_pipeline
    
    results = run_baden_pipeline(random_seed=42)

Quick Start:
    >>> from dataframe_phase.classification_v2.Baden_method import run_baden_pipeline
    >>> results = run_baden_pipeline(
    ...     input_path="path/to/data.parquet",
    ...     output_dir="path/to/output",
    ...     random_seed=42
    ... )
    >>> print(f"DS clusters: {results['ds'].optimal_k}")
    >>> print(f"non-DS clusters: {results['nds'].optimal_k}")

Modules:
    - config: Configuration constants and default parameters
    - preprocessing: Data loading, filtering, and signal conditioning
    - features: Sparse PCA feature extraction (40D)
    - clustering: GMM fitting with BIC model selection
    - evaluation: Posterior curves and bootstrap stability
    - visualization: BIC, posterior, and UMAP plots
    - pipeline: Main orchestration entry point
"""

__version__ = "1.0.0"
__author__ = "Data Processing Pipeline"

# Direct imports
from . import config
from . import preprocessing
from . import features
from . import clustering
from . import evaluation
from . import visualization
from . import pipeline
from .pipeline import run_baden_pipeline

# Public API
__all__ = [
    # Main entry point
    "run_baden_pipeline",
    # Submodules
    "config",
    "preprocessing",
    "features",
    "clustering",
    "evaluation",
    "visualization",
    "pipeline",
]

