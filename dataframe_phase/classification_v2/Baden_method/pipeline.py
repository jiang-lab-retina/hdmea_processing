"""
Pipeline orchestration for the Baden-method RGC clustering pipeline.

This module provides the main entry point `run_baden_pipeline()` that:
1. Loads and preprocesses data
2. Splits DS and non-DS populations
3. Extracts 40D feature vectors for each population
4. Clusters using GMM with BIC model selection
5. Evaluates cluster quality
6. Generates visualizations
7. Saves all outputs
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config
from . import preprocessing
from . import features
from . import clustering
from . import evaluation
from . import visualization

logger = logging.getLogger(__name__)


# =============================================================================
# Result Containers
# =============================================================================

@dataclass
class PopulationResult:
    """Container for clustering results of a single population."""
    population_name: str
    cell_ids: np.ndarray
    labels: np.ndarray
    posteriors: np.ndarray
    optimal_k: int
    bic_table: pd.DataFrame
    gmm: Any
    features_std: np.ndarray
    feature_models: Dict[str, Any]
    scaler: Any
    evaluation_metrics: Dict[str, Any]


# =============================================================================
# Saving Functions
# =============================================================================

def save_results(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save clustering results as parquet file.
    
    Args:
        results_df: DataFrame with clustering results.
        output_dir: Output directory.
        
    Returns:
        Path to saved file.
    """
    results_path = output_dir / "results" / "clustering_results.parquet"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(results_path, index=False)
    logger.info(f"Saved clustering results to {results_path}")
    return results_path


def save_models(
    models_dict: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save all trained models using joblib.
    
    Args:
        models_dict: Dictionary of models to save.
        output_dir: Output directory.
        
    Returns:
        Dictionary mapping model name to saved path.
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    for name, model in models_dict.items():
        path = models_dir / f"{name}.pkl"
        joblib.dump(model, path)
        saved_paths[name] = path
        logger.debug(f"Saved model {name} to {path}")
    
    logger.info(f"Saved {len(saved_paths)} models to {models_dir}")
    return saved_paths


def save_bic_tables(
    bic_ds: pd.DataFrame,
    bic_nds: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save BIC tables for both populations.
    
    Args:
        bic_ds: BIC table for DS population.
        bic_nds: BIC table for non-DS population.
        output_dir: Output directory.
        
    Returns:
        Tuple of paths to saved files.
    """
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    ds_path = results_dir / "bic_table_ds.parquet"
    nds_path = results_dir / "bic_table_nds.parquet"
    
    bic_ds.to_parquet(ds_path, index=False)
    bic_nds.to_parquet(nds_path, index=False)
    
    logger.info(f"Saved BIC tables to {results_dir}")
    return ds_path, nds_path


def save_stability_metrics(
    metrics_dict: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Save stability metrics as JSON.
    
    Args:
        metrics_dict: Dictionary of stability metrics.
        output_dir: Output directory.
        
    Returns:
        Path to saved file.
    """
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    path = results_dir / "stability_metrics.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    metrics_dict = convert(metrics_dict)
    
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Saved stability metrics to {path}")
    return path


# =============================================================================
# Population Processing
# =============================================================================

def process_population(
    df: pd.DataFrame,
    population_name: str,
    k_max: int,
    random_state: int,
    run_bootstrap: bool = True,
    show_progress: bool = True,
) -> PopulationResult:
    """
    Process a single population (DS or non-DS) through the full pipeline.
    
    Args:
        df: DataFrame with preprocessed traces for this population.
        population_name: "DS" or "non-DS".
        k_max: Maximum number of clusters to evaluate.
        random_state: Random seed.
        run_bootstrap: Whether to run bootstrap stability analysis.
        show_progress: Whether to show progress bars.
        
    Returns:
        PopulationResult with all clustering outputs.
    """
    logger.info(f"=" * 60)
    logger.info(f"Processing {population_name} population ({len(df)} cells)")
    logger.info(f"=" * 60)
    
    cell_ids = df.index.values
    
    # Step 1: Extract features
    logger.info("Extracting features...")
    features_raw, feature_names, feature_models = features.build_feature_matrix(
        df, random_state=random_state, return_models=True
    )
    
    # Step 2: Standardize features
    features_std, scaler = features.standardize_features(features_raw)
    
    # Step 3: Cluster with BIC selection
    logger.info(f"Clustering with BIC selection (k_max={k_max})...")
    gmm, labels, posteriors, bic_table, optimal_k = clustering.run_clustering(
        features_std, k_max=k_max, random_state=random_state, show_progress=show_progress
    )
    
    # Step 4: Evaluate clustering
    logger.info("Evaluating clustering quality...")
    eval_metrics = evaluation.evaluate_clustering(
        features_std, labels, posteriors, optimal_k,
        random_state=random_state,
        run_bootstrap=run_bootstrap,
        show_progress=show_progress,
    )
    
    logger.info(f"{population_name} complete: {optimal_k} clusters")
    
    return PopulationResult(
        population_name=population_name,
        cell_ids=cell_ids,
        labels=labels,
        posteriors=posteriors,
        optimal_k=optimal_k,
        bic_table=bic_table,
        gmm=gmm,
        features_std=features_std,
        feature_models=feature_models,
        scaler=scaler,
        evaluation_metrics=eval_metrics,
    )


# =============================================================================
# Main Pipeline
# =============================================================================

def run_baden_pipeline(
    input_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    k_max_ds: int = None,
    k_max_nds: int = None,
    qi_threshold: float = None,
    ds_p_threshold: float = None,
    random_seed: int = 42,
    run_bootstrap: bool = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Run the full Baden-method RGC clustering pipeline.
    
    This is the main entry point for the pipeline. It:
    1. Loads data from parquet
    2. Filters cells (NaN, QI, axon type)
    3. Splits into DS and non-DS populations
    4. Preprocesses traces (filter, baseline, normalize)
    5. Extracts 40D feature vectors
    6. Clusters using GMM with BIC selection
    7. Evaluates cluster quality
    8. Generates visualizations
    9. Saves all outputs
    
    Args:
        input_path: Path to input parquet file. Defaults to config.INPUT_PATH.
        output_dir: Output directory. Defaults to config.OUTPUT_DIR.
        k_max_ds: Maximum clusters for DS population. Defaults to config.K_MAX_DS.
        k_max_nds: Maximum clusters for non-DS population. Defaults to config.K_MAX_NDS.
        qi_threshold: Minimum quality index. Defaults to config.QI_THRESHOLD.
        ds_p_threshold: P-value threshold for DS classification. Defaults to config.DS_P_THRESHOLD.
        random_seed: Random seed for reproducibility.
        run_bootstrap: Whether to run bootstrap stability analysis. Defaults to config.RUN_BOOTSTRAP.
        show_progress: Whether to show progress bars.
        
    Returns:
        Dictionary with keys:
            - 'ds': PopulationResult for DS population
            - 'nds': PopulationResult for non-DS population
            - 'n_input_cells': Total cells in input
            - 'n_filtered_cells': Cells after filtering
            - 'output_paths': Dictionary of saved file paths
            - 'run_info': Metadata about the run
    
    Raises:
        FileNotFoundError: If input_path doesn't exist.
        ValueError: If no valid cells remain after filtering.
    """
    # Set defaults from config
    if input_path is None:
        input_path = config.INPUT_PATH
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if k_max_ds is None:
        k_max_ds = config.K_MAX_DS
    if k_max_nds is None:
        k_max_nds = config.K_MAX_NDS
    if qi_threshold is None:
        qi_threshold = config.QI_THRESHOLD
    if ds_p_threshold is None:
        ds_p_threshold = config.DS_P_THRESHOLD
    if run_bootstrap is None:
        run_bootstrap = config.RUN_BOOTSTRAP
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BADEN-METHOD RGC CLUSTERING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"k_max: DS={k_max_ds}, non-DS={k_max_nds}")
    logger.info("")
    
    # Step 1: Load data
    logger.info("STEP 1: Loading data...")
    df = preprocessing.load_data(input_path)
    n_input_cells = len(df)
    
    # Step 2: Filter rows
    logger.info("STEP 2: Filtering cells...")
    df_filtered = preprocessing.filter_rows(df, qi_threshold=qi_threshold)
    
    # Step 2b: Filter by baseline (if threshold is set)
    if config.BASELINE_MAX_THRESHOLD is not None:
        logger.info("STEP 2b: Filtering by baseline...")
        df_filtered = preprocessing.filter_by_baseline(df_filtered)
    
    # Step 2c: Filter by batch size (if threshold is set)
    min_batch_cells = getattr(config, 'MIN_BATCH_GOOD_CELLS', None)
    if min_batch_cells is not None and min_batch_cells > 0:
        logger.info("STEP 2c: Filtering by batch size...")
        df_filtered = preprocessing.filter_by_batch_size(df_filtered)
    n_filtered_cells = len(df_filtered)
    
    # Step 3: Split DS/non-DS
    logger.info("STEP 3: Splitting DS/non-DS populations...")
    df_ds, df_nds = preprocessing.split_ds_nds(df_filtered, p_threshold=ds_p_threshold)
    
    # Step 4: Preprocess traces
    logger.info("STEP 4: Preprocessing traces...")
    df_ds = preprocessing.preprocess_traces(df_ds)
    df_nds = preprocessing.preprocess_traces(df_nds)
    
    # Step 5: Process DS population
    logger.info("STEP 5: Processing DS population...")
    result_ds = process_population(
        df_ds, "DS", k_max_ds, random_seed,
        run_bootstrap=run_bootstrap, show_progress=show_progress
    )
    
    # Step 6: Process non-DS population
    logger.info("STEP 6: Processing non-DS population...")
    result_nds = process_population(
        df_nds, "non-DS", k_max_nds, random_seed + 1000,
        run_bootstrap=run_bootstrap, show_progress=show_progress
    )
    
    # Step 7: Generate visualizations
    logger.info("STEP 7: Generating visualizations...")
    plots_dir = output_dir / "plots"
    
    ds_plots = visualization.create_all_plots(
        result_ds.bic_table,
        result_ds.labels,
        result_ds.posteriors,
        result_ds.features_std,
        "DS",
        plots_dir,
        evaluation_metrics=result_ds.evaluation_metrics,
    )
    
    nds_plots = visualization.create_all_plots(
        result_nds.bic_table,
        result_nds.labels,
        result_nds.posteriors,
        result_nds.features_std,
        "non-DS",
        plots_dir,
        evaluation_metrics=result_nds.evaluation_metrics,
    )
    
    # Step 8: Aggregate results into DataFrame
    logger.info("STEP 8: Aggregating results...")
    
    # DS results
    ds_results = pd.DataFrame({
        'cell_id': result_ds.cell_ids,
        'population': 'DS',
        'cluster_label': result_ds.labels,
        'global_cluster_id': [f"DS_{l}" for l in result_ds.labels],
        'posterior_probability': result_ds.posteriors.max(axis=1),
    })
    
    # Add feature columns
    for i in range(result_ds.features_std.shape[1]):
        ds_results[f'feature_{i}'] = result_ds.features_std[:, i]
    
    # non-DS results
    nds_results = pd.DataFrame({
        'cell_id': result_nds.cell_ids,
        'population': 'non-DS',
        'cluster_label': result_nds.labels,
        'global_cluster_id': [f"nDS_{l}" for l in result_nds.labels],
        'posterior_probability': result_nds.posteriors.max(axis=1),
    })
    
    # Add feature columns
    for i in range(result_nds.features_std.shape[1]):
        nds_results[f'feature_{i}'] = result_nds.features_std[:, i]
    
    # Combine
    results_df = pd.concat([ds_results, nds_results], ignore_index=True)
    
    # Step 9: Save outputs
    logger.info("STEP 9: Saving outputs...")
    output_paths = {}
    
    # Save results parquet
    output_paths['results'] = save_results(results_df, output_dir)
    
    # Save BIC tables
    bic_ds_path, bic_nds_path = save_bic_tables(
        result_ds.bic_table, result_nds.bic_table, output_dir
    )
    output_paths['bic_ds'] = bic_ds_path
    output_paths['bic_nds'] = bic_nds_path
    
    # Save models
    models_to_save = {
        'ds_gmm': result_ds.gmm,
        'ds_scaler': result_ds.scaler,
        'nds_gmm': result_nds.gmm,
        'nds_scaler': result_nds.scaler,
    }
    # Add feature models
    for name, model in result_ds.feature_models.items():
        models_to_save[f'ds_{name}'] = model
    for name, model in result_nds.feature_models.items():
        models_to_save[f'nds_{name}'] = model
    
    model_paths = save_models(models_to_save, output_dir)
    output_paths['models'] = model_paths
    
    # Save stability metrics
    stability_metrics = {
        'DS': {
            'optimal_k': result_ds.optimal_k,
            'bootstrap_median_correlation': result_ds.evaluation_metrics.get('bootstrap_median_correlation'),
            'bootstrap_iterations': config.BOOTSTRAP_N_ITERATIONS,
            'sample_fraction': config.BOOTSTRAP_SAMPLE_FRACTION,
            'is_stable': result_ds.evaluation_metrics.get('is_stable'),
            'all_correlations': result_ds.evaluation_metrics.get('bootstrap_all_correlations', []),
        },
        'non-DS': {
            'optimal_k': result_nds.optimal_k,
            'bootstrap_median_correlation': result_nds.evaluation_metrics.get('bootstrap_median_correlation'),
            'bootstrap_iterations': config.BOOTSTRAP_N_ITERATIONS,
            'sample_fraction': config.BOOTSTRAP_SAMPLE_FRACTION,
            'is_stable': result_nds.evaluation_metrics.get('is_stable'),
            'all_correlations': result_nds.evaluation_metrics.get('bootstrap_all_correlations', []),
        },
    }
    output_paths['stability'] = save_stability_metrics(stability_metrics, output_dir)
    
    # Add plot paths
    output_paths['plots_ds'] = ds_plots
    output_paths['plots_nds'] = nds_plots
    
    # Run info
    end_time = datetime.now()
    run_info = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'input_path': str(input_path),
        'output_dir': str(output_dir),
        'random_seed': random_seed,
        'n_input_cells': n_input_cells,
        'n_filtered_cells': n_filtered_cells,
        'n_ds_cells': len(df_ds),
        'n_nds_cells': len(df_nds),
        'ds_clusters': result_ds.optimal_k,
        'nds_clusters': result_nds.optimal_k,
    }
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total cells: {n_input_cells}")
    logger.info(f"Filtered cells: {n_filtered_cells}")
    logger.info(f"DS cells: {len(df_ds)} → {result_ds.optimal_k} clusters")
    logger.info(f"non-DS cells: {len(df_nds)} → {result_nds.optimal_k} clusters")
    logger.info(f"Duration: {run_info['duration_seconds']:.1f} seconds")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    return {
        'ds': result_ds,
        'nds': result_nds,
        'n_input_cells': n_input_cells,
        'n_filtered_cells': n_filtered_cells,
        'output_paths': output_paths,
        'run_info': run_info,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Baden-method RGC clustering pipeline")
    parser.add_argument("--input", type=str, help="Input parquet file path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap stability")
    parser.add_argument("--quiet", action="store_true", help="Hide progress bars")
    
    args = parser.parse_args()
    
    run_baden_pipeline(
        input_path=args.input,
        output_dir=args.output,
        random_seed=args.seed,
        run_bootstrap=not args.no_bootstrap,
        show_progress=not args.quiet,
    )

