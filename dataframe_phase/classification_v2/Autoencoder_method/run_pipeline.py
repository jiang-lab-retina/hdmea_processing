"""
Main entry point for the Autoencoder-based RGC clustering pipeline.

Usage:
    python -m Autoencoder_method.run_pipeline
    
    Or run directly:
    python run_pipeline.py
    
    With custom paths:
    python run_pipeline.py --input data.parquet --output results/
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Running as script directly - fix the import path
    _this_file = Path(__file__).resolve()
    _package_dir = _this_file.parent
    _parent_dir = _package_dir.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    __package__ = "Autoencoder_method"

import numpy as np
import pandas as pd

from Autoencoder_method import config
from Autoencoder_method.data_loader import load_and_filter_data
from Autoencoder_method.preprocessing import preprocess_all_segments, get_segment_lengths
from Autoencoder_method.grouping import assign_coarse_groups, encode_group_labels, filter_groups_by_size
from Autoencoder_method.train import train_autoencoder, load_model
from Autoencoder_method.embed import extract_embeddings, standardize_embeddings
from Autoencoder_method.clustering import cluster_per_group
from Autoencoder_method.stability import run_bootstrap_stability, summarize_stability, save_stability_results
from Autoencoder_method.evaluation import (
    validate_group_purity, save_embeddings, save_cluster_assignments,
    compute_silhouette_score, compute_cv_purity_posthoc, save_cv_purity_results,
    save_k_selection_data
)
from Autoencoder_method.visualization import generate_all_plots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    skip_training: bool = False,
    skip_stability: bool = False,
    skip_plots: bool = False,
    model_path: Path | str | None = None,
    subset_n: int | None = None,
    run_cv: bool = False,
) -> dict:
    """
    Run the complete AE-based clustering pipeline.
    
    Args:
        input_path: Path to input parquet file.
        output_dir: Output directory.
        skip_training: If True, load existing model instead of training.
        skip_stability: If True, skip bootstrap stability testing.
        skip_plots: If True, skip generating plots.
        model_path: Path to pre-trained model (for skip_training=True).
        subset_n: If set, use only N cells for quick testing.
        run_cv: If True, run full cross-validation (re-trains AE per turn).
    
    Returns:
        Dict with pipeline results and output paths.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Autoencoder-based RGC Clustering Pipeline")
    logger.info("=" * 60)
    
    # Apply defaults
    input_path = Path(input_path) if input_path else config.INPUT_PATH
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    
    # Create output directories
    (output_dir / "models_saved").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    results = {
        'input_path': str(input_path),
        'output_dir': str(output_dir),
        'start_time': start_time.isoformat(),
    }
    
    # ==== Step 1: Load and filter data ====
    logger.info("\n[Step 1/7] Loading and filtering data...")
    try:
        df = load_and_filter_data(input_path)
        logger.info(f"  Loaded {len(df)} cells after filtering")
        results['n_cells_after_filter'] = len(df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Apply subset if requested (for quick testing)
    if subset_n is not None and subset_n < len(df):
        logger.info(f"  Subsampling to {subset_n} cells for quick testing...")
        # Stratified subsample to keep all groups represented
        from sklearn.model_selection import train_test_split
        df_temp = assign_coarse_groups(df.copy())
        groups_temp = df_temp['coarse_group'].values
        
        # Use stratified split to maintain group proportions
        try:
            _, df = train_test_split(
                df, 
                test_size=subset_n, 
                stratify=groups_temp, 
                random_state=42
            )
        except ValueError:
            # If stratification fails (too few samples in a group), use random
            df = df.sample(n=subset_n, random_state=42)
        
        logger.info(f"  Subset: {len(df)} cells")
        results['subset_n'] = len(df)
    
    # ==== Step 2: Preprocess segments ====
    logger.info("\n[Step 2/7] Preprocessing trace segments...")
    segments = preprocess_all_segments(df)
    segment_lengths = get_segment_lengths(segments)
    logger.info(f"  Preprocessed {len(segments)} segments")
    for name, length in segment_lengths.items():
        logger.debug(f"    {name}: {length} samples")
    results['segment_lengths'] = segment_lengths
    
    # ==== Step 3: Assign coarse groups ====
    logger.info("\n[Step 3/7] Assigning coarse groups...")
    
    # Store original indices before filtering
    original_indices = np.arange(len(df))
    df = df.reset_index(drop=True)  # Reset to integer index
    
    df = assign_coarse_groups(df)
    
    # Track which rows survive filtering
    pre_filter_len = len(df)
    df = filter_groups_by_size(df)
    post_filter_len = len(df)
    
    if post_filter_len < pre_filter_len:
        logger.info(f"  Filtered {pre_filter_len - post_filter_len} cells (small groups)")
        # Filter segments to match filtered DataFrame
        keep_mask = df.index.values
        segments = {name: arr[keep_mask] for name, arr in segments.items()}
        logger.info(f"  Segments filtered to {post_filter_len} cells")
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    group_labels, group_map = encode_group_labels(df)
    groups = df['coarse_group'].values
    cell_ids = df.index.values
    
    logger.info(f"  Groups: {np.unique(groups).tolist()}")
    results['group_counts'] = df['coarse_group'].value_counts().to_dict()
    
    # Create purity labels for purity loss (if enabled)
    # Binary labels: (n_cells, 3) for [axon_type, ds_cell, iprgc]
    purity_labels = None
    if getattr(config, 'USE_PURITY_LOSS', False):
        logger.info("  Creating purity labels for purity loss...")
        axon_label = (df[config.AXON_COL] == 'ac').astype(int).values
        ds_label = (df[config.DS_PVAL_COL] < config.DS_P_THRESHOLD).astype(int).values
        iprgc_qi = df[config.IPRGC_QI_COL].fillna(0).values
        iprgc_label = (iprgc_qi > config.IPRGC_QI_THRESHOLD).astype(int)
        
        purity_labels = np.column_stack([axon_label, ds_label, iprgc_label])
        logger.info(f"  Purity labels shape: {purity_labels.shape}")
        logger.info(f"    axon_type=ac: {axon_label.sum()} cells")
        logger.info(f"    ds_cell: {ds_label.sum()} cells")
        logger.info(f"    iprgc: {iprgc_label.sum()} cells")
    
    # ==== Step 4: Train or load autoencoder ====
    logger.info("\n[Step 4/7] Training autoencoder...")
    
    if skip_training and model_path:
        logger.info(f"  Loading pre-trained model from {model_path}")
        model = load_model(model_path)
        history = None
    else:
        model, history = train_autoencoder(
            segments=segments,
            group_labels=group_labels,
            purity_labels=purity_labels,
            checkpoint_dir=output_dir / "models_saved",
        )
        results['training_history'] = {
            'final_loss': history['loss'][-1] if history else None,
            'n_epochs': len(history['loss']) if history else None,
        }
    
    # ==== Step 5: Extract and standardize embeddings ====
    logger.info("\n[Step 5/7] Extracting embeddings...")
    embeddings = extract_embeddings(model, segments)
    embeddings_std, scaler_params = standardize_embeddings(embeddings)
    
    logger.info(f"  Embedding shape: {embeddings.shape}")
    
    # Compute silhouette score to verify group separation
    silhouette = compute_silhouette_score(embeddings_std, group_labels)
    results['silhouette_score'] = silhouette
    
    # Save embeddings
    save_embeddings(
        embeddings=embeddings_std,
        cell_ids=cell_ids,
        groups=groups,
        model_version=f"v{datetime.now().strftime('%Y%m%d')}",
        output_path=output_dir / "results" / "embeddings.parquet",
    )
    
    # ==== Step 6: Cluster per group ====
    logger.info("\n[Step 6/7] Clustering per group...")
    cluster_ids, posterior_probs, group_results = cluster_per_group(
        embeddings=embeddings_std,
        groups=groups,
    )
    
    # Log cluster info per group
    for group_name, gres in group_results.items():
        logger.info(f"  {group_name}: k={gres['k_selected']} clusters")
    
    results['clusters_per_group'] = {
        g: r['k_selected'] for g, r in group_results.items()
    }
    
    # Validate group purity
    cluster_df = pd.DataFrame({
        'cell_id': cell_ids,
        'coarse_group': groups,
        'cluster_id': cluster_ids,
    })
    is_pure = validate_group_purity(cluster_df)
    results['group_pure'] = is_pure
    
    # Save cluster assignments
    save_cluster_assignments(
        cell_ids=cell_ids,
        groups=groups,
        cluster_ids=cluster_ids,
        posterior_probs=posterior_probs,
        output_path=output_dir / "results" / "cluster_assignments.parquet",
    )
    
    # Save k-selection data (BIC curves, how cluster numbers were decided)
    save_k_selection_data(
        group_results=group_results,
        output_path=output_dir / "results" / "k_selection.json",
    )
    results['k_selection_path'] = str(output_dir / "results" / "k_selection.json")
    
    # ==== Step 6b: Cross-validation purity analysis ====
    logger.info("\n[Step 6b/7] Computing cross-validation purity metrics...")
    
    # Create subtype labels for purity analysis
    subtype_labels = np.array([
        f"{group}::cluster_{cid:02d}"
        for group, cid in zip(groups, cluster_ids)
    ])
    
    cv_results = compute_cv_purity_posthoc(
        cluster_labels=subtype_labels,
        groups=groups,
        df=df,  # Pass original data for true label-based purity
    )
    save_cv_purity_results(
        cv_results=cv_results,
        output_path=output_dir / "results" / "cv_purity.json",
    )
    results['cv_purity'] = cv_results
    
    # ==== Step 7: Stability testing and visualization ====
    if not skip_stability and config.RUN_BOOTSTRAP:
        logger.info("\n[Step 7a/7] Running bootstrap stability...")
        stability_summary, bootstrap_details = run_bootstrap_stability(
            embeddings=embeddings_std,
            groups=groups,
        )
        overall_stability = summarize_stability(stability_summary)
        save_stability_results(
            stability_summary, 
            overall_stability,
            output_path=output_dir / "results" / "stability_metrics.json",
        )
        results['stability'] = overall_stability
    else:
        logger.info("\n[Step 7a/7] Skipping stability testing")
        results['stability'] = None
    
    if not skip_plots:
        logger.info("\n[Step 7b/7] Generating visualizations...")
        plot_paths = generate_all_plots(
            embeddings=embeddings_std,
            segments=segments,
            groups=groups,
            cluster_labels=cluster_ids,
            group_results=group_results,
            cv_results=cv_results,
            output_dir=output_dir / "plots",
        )
        results['plots'] = {k: str(v) for k, v in plot_paths.items()}
    else:
        logger.info("\n[Step 7b/7] Skipping visualization")
        results['plots'] = None
    
    # ==== Step 8: Full Cross-Validation (optional) ====
    if run_cv:
        from Autoencoder_method.crossval import run_full_cv
        
        logger.info("\n[Step 8/8] Running full cross-validation...")
        logger.info("  This will re-train AE for each omitted label (~25 min per turn)")
        
        cv_results_df, cv_score_full = run_full_cv(
            df=df,
            segments=segments,
            output_dir=output_dir,
            device=config.DEVICE,
        )
        results['full_cv'] = {
            'cv_score': cv_score_full,
            'turns': cv_results_df.to_dict('records'),
        }
        logger.info(f"  Full CVScore: {cv_score_full:.4f}")
    else:
        results['full_cv'] = None
    
    # ==== Complete ====
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(f"  Duration: {duration:.1f} seconds")
    logger.info(f"  Cells: {len(df)}")
    logger.info(f"  Clusters: {sum(r['k_selected'] for r in group_results.values())}")
    logger.info(f"  Outputs: {output_dir}")
    logger.info("=" * 60)
    
    results['end_time'] = end_time.isoformat()
    results['duration_seconds'] = duration
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Autoencoder-based RGC Clustering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m Autoencoder_method.run_pipeline
  python -m Autoencoder_method.run_pipeline --input data.parquet --output results/
  python -m Autoencoder_method.run_pipeline --skip-training --model models/autoencoder_best.pt
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help=f'Input parquet file path (default: {config.INPUT_PATH})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=f'Output directory (default: {config.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and load existing model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to pre-trained model (required if --skip-training)'
    )
    
    parser.add_argument(
        '--skip-stability',
        action='store_true',
        help='Skip bootstrap stability testing'
    )
    
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        metavar='N',
        help='Use only N cells for quick testing (e.g., --subset 1000)'
    )
    
    parser.add_argument(
        '--run-cv',
        action='store_true',
        help='Run full cross-validation (re-trains AE for each omitted label, ~75 min)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = main(
        input_path=args.input,
        output_dir=args.output,
        skip_training=args.skip_training,
        skip_stability=args.skip_stability,
        skip_plots=args.skip_plots,
        model_path=args.model,
        subset_n=args.subset,
        run_cv=args.run_cv,
    )
    
    # Print summary
    print("\nPipeline Results:")
    print(f"  Cells processed: {results.get('n_cells_after_filter', 'N/A')}")
    print(f"  Silhouette score: {results.get('silhouette_score', 'N/A'):.4f}" 
          if results.get('silhouette_score') else "  Silhouette score: N/A")
    print(f"  Group pure: {results.get('group_pure', 'N/A')}")
    if results.get('full_cv'):
        print(f"  Full CVScore: {results['full_cv']['cv_score']:.4f}")
    print(f"  Duration: {results.get('duration_seconds', 'N/A'):.1f}s"
          if results.get('duration_seconds') else "  Duration: N/A")
