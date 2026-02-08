"""
Main entry point for the DEC-refined RGC subtype clustering pipeline.

Usage:
    python -m divide_conquer_method.run_pipeline --group DSGC
    python -m divide_conquer_method.run_pipeline --all-groups
    python -m divide_conquer_method.run_pipeline --group DSGC --subset 500
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Support both direct execution and module execution
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent

if __name__ == "__main__":
    # Running directly or as module: ensure parent is in path
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))

# Use try/except for imports to support both execution modes
try:
    from . import config
    from .data_loader import load_and_filter_data
    from .grouping import assign_groups, filter_group, get_group_stats
    from .preprocessing import preprocess_all_segments, get_segment_lengths, extract_iprgc_last_trial
    from .train import train_autoencoder, load_model
    from .embed import extract_embeddings, standardize_embeddings
    from .clustering.gmm_bic import fit_gmm_bic, select_k_min_bic, save_k_selection, load_gmm_results
    from .clustering.dec_refine import refine_with_dec, load_dec_results
    from .validation.iprgc_metrics import compute_iprgc_metrics, get_iprgc_labels
    from .validation.mosaic_validation import (
        validate_dataframe,
        calculate_total_area,
        calculate_conversion_factor,
        calculate_subtype_coverage,
        calculate_rf_areas_for_cells,
        plot_rf_coverage_bar_chart,
        plot_validation_heatmap,
        CRITICAL_COLUMNS,
        RF_COVERAGE_THRESHOLD,
    )
    from . import evaluation
except ImportError:
    from divide_conquer_method import config
    from divide_conquer_method.data_loader import load_and_filter_data
    from divide_conquer_method.grouping import assign_groups, filter_group, get_group_stats
    from divide_conquer_method.preprocessing import preprocess_all_segments, get_segment_lengths, extract_iprgc_last_trial
    from divide_conquer_method.train import train_autoencoder, load_model
    from divide_conquer_method.embed import extract_embeddings, standardize_embeddings
    from divide_conquer_method.clustering.gmm_bic import fit_gmm_bic, select_k_min_bic, save_k_selection, load_gmm_results
    from divide_conquer_method.clustering.dec_refine import refine_with_dec, load_dec_results
    from divide_conquer_method.validation.iprgc_metrics import compute_iprgc_metrics, get_iprgc_labels
    from divide_conquer_method.validation.mosaic_validation import (
        validate_dataframe,
        calculate_total_area,
        calculate_conversion_factor,
        calculate_subtype_coverage,
        calculate_rf_areas_for_cells,
        plot_rf_coverage_bar_chart,
        plot_validation_heatmap,
        CRITICAL_COLUMNS,
        RF_COVERAGE_THRESHOLD,
    )
    from divide_conquer_method import evaluation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


def main(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    group: str | None = None,
    all_groups: bool = False,
    skip_training: bool = False,
    skip_gmm: bool = False,
    skip_dec: bool = False,
    skip_plots: bool = False,
    skip_mosaic: bool = False,
    visualize_only: bool = False,
    subset_n: int | None = None,
) -> dict:
    """
    Run the DEC-refined RGC clustering pipeline.
    
    Args:
        input_path: Path to input parquet file.
        output_dir: Output directory for results.
        group: Single group to process ("DSGC", "OSGC", "Other").
        all_groups: Process all three groups.
        skip_training: Load saved AE model instead of training.
        skip_gmm: Load cached k* instead of running GMM/BIC.
        skip_dec: Skip DEC refinement (GMM-only).
        skip_plots: Skip plot generation.
        skip_mosaic: Skip mosaic validation step.
        visualize_only: Only generate plots from saved artifacts.
        subset_n: Use subset of N cells for testing.
    
    Returns:
        Dict with pipeline results.
    """
    start_time = time.time()
    
    # Apply defaults
    input_path = Path(input_path) if input_path else config.INPUT_PATH
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    
    # Default to all_groups if no group specified
    if not group and not all_groups:
        all_groups = True
        logger.info("No group specified, defaulting to --all-groups")
    
    if group and group not in config.GROUP_NAMES:
        raise ValueError(f"Invalid group '{group}'. Must be one of {config.GROUP_NAMES}")
    
    # Determine groups to process
    groups = config.GROUP_NAMES if all_groups else [group]
    
    logger.info("=" * 60)
    logger.info("DEC-Refined RGC Subtype Clustering Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Groups: {groups}")
    
    # Visualization-only mode
    if visualize_only:
        logger.info("Visualization-only mode: regenerating plots from saved artifacts")
        from divide_conquer_method import visualization
        
        for grp in groups:
            artifacts = evaluation.load_artifacts_for_visualization(
                config.RESULTS_DIR, grp
            )
            visualization.generate_all_plots(artifacts, grp, config.PLOTS_DIR)
        
        return {'mode': 'visualize_only', 'groups': groups}
    
    # Load and filter data
    logger.info("-" * 40)
    logger.info("Step 1: Loading and filtering data")
    
    df, reject_reasons = load_and_filter_data(input_path)
    
    # Apply subset for testing
    if subset_n is not None and len(df) > subset_n:
        logger.info(f"Using subset of {subset_n} cells for testing")
        df = df.sample(n=subset_n, random_state=42).reset_index(drop=True)
    
    # Assign groups
    logger.info("-" * 40)
    logger.info("Step 2: Assigning groups (DS > OS priority)")
    
    df = assign_groups(df)
    stats = get_group_stats(df)
    
    # Process each group
    all_results = {}
    
    for grp in groups:
        logger.info("=" * 60)
        logger.info(f"Processing group: {grp}")
        logger.info("=" * 60)
        
        try:
            results = _process_single_group(
                df=df,
                group=grp,
                output_dir=output_dir,
                skip_training=skip_training,
                skip_gmm=skip_gmm,
                skip_dec=skip_dec,
                skip_plots=skip_plots,
            )
            all_results[grp] = results
            
        except ValueError as e:
            logger.warning(f"Skipping group {grp}: {e}")
            all_results[grp] = {'error': str(e)}
    
    # Generate consolidated report for all-groups mode
    if all_groups and len(all_results) > 1:
        valid_results = {g: r for g, r in all_results.items() if 'error' not in r}
        if valid_results:
            evaluation.generate_consolidated_report(valid_results, config.RESULTS_DIR)
    
    # Mosaic validation (requires cluster assignments from all processed groups)
    mosaic_results = None
    if not skip_mosaic:
        valid_groups = [g for g in groups if 'error' not in all_results.get(g, {'error': ''})]
        if valid_groups:
            mosaic_results = _run_mosaic_validation(
                df=df,
                groups=valid_groups,
                results_dir=config.RESULTS_DIR,
                skip_plots=skip_plots,
            )
            # Attach mosaic results to per-group results
            if mosaic_results is not None:
                for grp in valid_groups:
                    grp_mosaic = mosaic_results[mosaic_results['group'] == grp]
                    if len(grp_mosaic) > 0 and grp in all_results:
                        n_valid = int(grp_mosaic['mosaic_validation'].sum())
                        n_total = len(grp_mosaic)
                        all_results[grp]['mosaic_valid'] = n_valid
                        all_results[grp]['mosaic_total'] = n_total
        else:
            logger.warning("Skipping mosaic validation: no groups completed successfully")
    else:
        logger.info("Skipping mosaic validation (--skip-mosaic)")
    
    # Save labeled DataFrame with subtype + valid_mosaic columns
    valid_groups = [g for g in groups if 'error' not in all_results.get(g, {'error': ''})]
    if valid_groups:
        labeled_path = config.RESULTS_DIR / "labeled_dataframe.parquet"
        evaluation.save_labeled_dataframe(
            df=df,
            groups=valid_groups,
            results_dir=config.RESULTS_DIR,
            mosaic_results=mosaic_results,
            output_path=labeled_path,
        )
    
    # Final summary
    duration = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 60)
    
    return {
        'input_path': str(input_path),
        'output_dir': str(output_dir),
        'groups_processed': groups,
        'per_group': all_results,
        'mosaic_validation': mosaic_results.to_dict('records') if mosaic_results is not None else None,
        'duration_seconds': duration,
    }


def _run_mosaic_validation(
    df: pd.DataFrame,
    groups: list,
    results_dir: Path,
    skip_plots: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Run mosaic validation on the already-loaded DataFrame.
    
    Uses the cluster assignments saved during group processing to compute
    RF coverage for each subtype. Avoids redundant data loading by reusing
    the DataFrame already in memory.
    
    Args:
        df: Full filtered DataFrame with group assignments (from Step 1-2).
        groups: List of group names that were successfully processed.
        results_dir: Directory containing per-group cluster_assignments.parquet.
        skip_plots: Skip plot generation.
        
    Returns:
        DataFrame with mosaic validation results, or None if validation failed.
    """
    logger.info("=" * 60)
    logger.info("Step 9: Mosaic Validation")
    logger.info("=" * 60)
    
    validation_dir = Path(__file__).resolve().parent / "validation"
    output_results_dir = validation_dir / "result"
    output_figures_dir = validation_dir / "figure"
    output_results_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 9a: Validate DataFrame for NaN values
        logger.info("-" * 40)
        logger.info("Step 9a: Validating DataFrame for NaN values")
        valid_df, nan_report = validate_dataframe(df, CRITICAL_COLUMNS)
        
        nan_report_path = output_results_dir / "nan_validation_report.csv"
        nan_report.to_csv(nan_report_path, index=False)
        logger.info(f"  Saved NaN report to: {nan_report_path}")
        
        # Build group mappings from the validated DataFrame
        group_mappings = {}
        for group_name in groups:
            group_df = valid_df[valid_df['group'] == group_name]
            mapping = pd.DataFrame({
                'cell_id': range(len(group_df)),
                'original_index': group_df.index.values,
            })
            group_mappings[group_name] = mapping
            logger.info(f"  {group_name}: {len(mapping)} cells (validated)")
        
        # Step 9b: Calculate total area
        logger.info("-" * 40)
        logger.info("Step 9b: Calculating total area")
        total_area_px2, total_area_mm2, batch_areas = calculate_total_area(valid_df)
        
        # Step 9c: Calculate conversion factor
        logger.info("-" * 40)
        logger.info("Step 9c: Calculating conversion factor")
        conversion_factor, total_good_cell_count = calculate_conversion_factor(
            valid_df, total_area_mm2
        )
        
        # Step 9d: Calculate RF coverage per subtype
        logger.info("-" * 40)
        logger.info("Step 9d: Calculating RF coverage per subtype")
        
        all_group_results = []
        
        for group_name in groups:
            cluster_path = results_dir / group_name / "cluster_assignments.parquet"
            
            if not cluster_path.exists():
                logger.warning(f"  Cluster assignments not found: {cluster_path}")
                continue
            
            cluster_assignments = pd.read_parquet(cluster_path)
            index_mapping = group_mappings[group_name]
            
            group_results = calculate_subtype_coverage(
                df=valid_df,
                cluster_assignments=cluster_assignments,
                index_mapping=index_mapping,
                group_name=group_name,
                conversion_factor=conversion_factor,
                total_area_mm2=total_area_mm2,
            )
            
            all_group_results.append(group_results)
            
            n_valid = group_results['mosaic_validation'].sum()
            n_total = len(group_results)
            logger.info(f"  {group_name}: {n_valid}/{n_total} subtypes pass validation")
        
        if not all_group_results:
            logger.warning("No mosaic validation results produced")
            return None
        
        results_df = pd.concat(all_group_results, ignore_index=True)
        
        # Save results
        logger.info("-" * 40)
        logger.info("Step 9e: Saving mosaic validation results")
        
        parquet_path = output_results_dir / "mosaic_validation_results.parquet"
        results_df.to_parquet(parquet_path, index=False)
        logger.info(f"  Saved: {parquet_path}")
        
        csv_path = output_results_dir / "mosaic_summary.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")
        
        # Generate plots
        if not skip_plots:
            logger.info("-" * 40)
            logger.info("Step 9f: Generating mosaic validation plots")
            
            plot_rf_coverage_bar_chart(
                results_df,
                output_figures_dir / "rf_coverage_by_subtype.png"
            )
            
            plot_validation_heatmap(
                results_df,
                output_figures_dir / "validation_heatmap.png"
            )
        
        # Log summary
        total_valid = results_df['mosaic_validation'].sum()
        total_subtypes = len(results_df)
        logger.info(f"  Mosaic validation: {total_valid}/{total_subtypes} subtypes pass "
                     f"(threshold = {RF_COVERAGE_THRESHOLD})")
        
        return results_df
    
    except Exception as e:
        logger.error(f"Mosaic validation failed: {e}", exc_info=True)
        return None


def _process_single_group(
    df: pd.DataFrame,
    group: str,
    output_dir: Path,
    skip_training: bool = False,
    skip_gmm: bool = False,
    skip_dec: bool = False,
    skip_plots: bool = False,
) -> dict:
    """
    Process a single functional group.
    
    Returns dict with processing results.
    """
    # Filter to this group
    group_df = filter_group(df, group)
    n_cells = len(group_df)
    cell_ids = group_df.index.values
    
    logger.info(f"Group {group}: {n_cells} cells")
    
    # Create output directories
    results_dir = config.RESULTS_DIR / group
    plots_dir = config.PLOTS_DIR / group
    models_dir = config.MODELS_DIR / group
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocess traces
    logger.info("-" * 40)
    logger.info("Step 3: Preprocessing traces")
    
    segments, full_segments = preprocess_all_segments(group_df)
    
    segment_lengths = get_segment_lengths(segments)
    
    # Extract last-trial iprgc_test for prototype plots
    iprgc_last_trial = extract_iprgc_last_trial(group_df)
    
    # Extract ipRGC labels for validation metrics
    iprgc_labels = get_iprgc_labels(group_df)
    
    # Step 2: Train or load autoencoder
    logger.info("-" * 40)
    logger.info("Step 4: Autoencoder")
    
    if skip_training:
        checkpoint_path = models_dir / "autoencoder_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
        model = load_model(checkpoint_path, segment_lengths)
        history = {}
    else:
        model, history = train_autoencoder(
            segments=segments,
            checkpoint_dir=models_dir,
        )
    
    # Step 3: Extract embeddings
    logger.info("-" * 40)
    logger.info("Step 5: Extracting embeddings")
    
    embeddings = extract_embeddings(model, segments)
    embeddings_std, scaler_params = standardize_embeddings(embeddings)
    
    # Save initial embeddings
    evaluation.save_embeddings(
        embeddings_std, cell_ids, group, "initial", results_dir
    )
    
    # Step 4: GMM/BIC k-selection
    logger.info("-" * 40)
    logger.info("Step 6: GMM/BIC k-selection")
    
    k_max = config.K_MAX.get(group, 40)
    k_range = list(range(config.K_MIN, k_max + 1))
    
    if skip_gmm:
        k_selected, gmm_model, bic_values = load_gmm_results(
            config.RESULTS_DIR, group, embeddings_std
        )
    else:
        gmm_models, bic_values = fit_gmm_bic(embeddings_std, k_range)
        k_selected, gmm_model = select_k_min_bic(
            gmm_models, bic_values, k_range,
            method=getattr(config, 'K_SELECTION_METHOD', 'elbow'),
            elbow_threshold=getattr(config, 'ELBOW_THRESHOLD', 0.01),
        )
        
        evaluation.save_k_selection(k_range, bic_values, k_selected, group, results_dir)
    
    # Get GMM labels and posteriors
    gmm_labels = gmm_model.predict(embeddings_std)
    gmm_posteriors = gmm_model.predict_proba(embeddings_std).max(axis=1)
    
    # ipRGC labels already extracted above (before AE training)
    # Compute GMM metrics
    gmm_metrics = compute_iprgc_metrics(gmm_labels, iprgc_labels)
    logger.info(f"GMM purity: {gmm_metrics['purity']:.3f}")
    
    # Step 6: DEC refinement
    if skip_dec:
        logger.info("-" * 40)
        logger.info("Step 7: Skipping DEC refinement")
        
        dec_labels = gmm_labels.copy()
        dec_embeddings = embeddings_std.copy()
        dec_soft_max = gmm_posteriors.copy()
        dec_metrics = gmm_metrics.copy()
        dec_history = {}
    else:
        logger.info("-" * 40)
        logger.info("Step 7: DEC refinement")
        
        dec_labels_raw, dec_embeddings, dec_history = refine_with_dec(
            model=model,
            segments=segments,
            initial_centers=gmm_model.means_,
            k=k_selected,
            scaler_params=scaler_params,
            checkpoint_dir=models_dir,
        )
        
        # Check if reassignment is allowed
        allow_reassignment = getattr(config, 'DEC_ALLOW_REASSIGNMENT', True)
        if allow_reassignment:
            dec_labels = dec_labels_raw
            logger.info("DEC reassignment enabled: using DEC labels")
        else:
            dec_labels = gmm_labels.copy()
            logger.info("DEC reassignment disabled: keeping GMM labels, only embeddings refined")
        
        # Compute DEC soft assignments for saving
        # For now, use 1.0 as placeholder (actual soft assignments are in IDEC)
        dec_soft_max = np.ones(n_cells)
        
        # Compute DEC metrics
        dec_metrics = compute_iprgc_metrics(dec_labels, iprgc_labels)
        logger.info(f"DEC purity: {dec_metrics['purity']:.3f}")
    
    # Save DEC embeddings
    evaluation.save_embeddings(
        dec_embeddings, cell_ids, group, "dec_refined", results_dir
    )
    
    # Save cluster assignments
    evaluation.save_cluster_assignments(
        cell_ids, group, gmm_labels, gmm_posteriors,
        dec_labels, dec_soft_max, results_dir
    )
    
    # Save ipRGC validation
    evaluation.save_iprgc_validation(gmm_metrics, dec_metrics, group, results_dir)
    
    # Save comparison table
    evaluation.save_comparison_table(
        gmm_metrics, dec_metrics, group, k_selected, results_dir
    )
    
    # Generate plots
    if not skip_plots:
        logger.info("-" * 40)
        logger.info("Step 8: Generating plots")
        
        from divide_conquer_method import visualization
        
        artifacts = {
            'embeddings_initial': embeddings_std,
            'embeddings_dec': dec_embeddings,
            'gmm_labels': gmm_labels,
            'dec_labels': dec_labels,
            'iprgc_labels': iprgc_labels,
            'bic_values': bic_values,
            'k_range': k_range,
            'k_selected': k_selected,
            'gmm_metrics': gmm_metrics,
            'dec_metrics': dec_metrics,
            'segments': segments,
            'full_segments': full_segments,
            'iprgc_last_trial': iprgc_last_trial,
        }
        
        visualization.generate_all_plots(artifacts, group, plots_dir)
    
    # Return results
    return {
        'n_cells': n_cells,
        'k_selected': k_selected,
        'bic_min': float(bic_values[k_range.index(k_selected)]) if k_selected in k_range else 0,
        'gmm_metrics': gmm_metrics,
        'dec_metrics': dec_metrics,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DEC-refined RGC subtype clustering pipeline"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Input parquet file path"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory"
    )
    
    parser.add_argument(
        "--group",
        type=str,
        choices=["ipRGC", "DSGC", "OSGC", "Other"],
        default=None,
        help="Group to process"
    )
    
    parser.add_argument(
        "--all-groups",
        action="store_true",
        default=True,
        help="Process all three groups (default)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip AE training, load checkpoint"
    )
    
    parser.add_argument(
        "--skip-gmm",
        action="store_true",
        help="Skip GMM, load cached k*"
    )
    
    parser.add_argument(
        "--skip-dec",
        action="store_true",
        default=True,
        help="Skip DEC refinement (default: True)"
    )
    
    parser.add_argument(
        "--run-dec",
        action="store_true",
        help="Run DEC refinement (overrides --skip-dec default)"
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    parser.add_argument(
        "--skip-mosaic",
        action="store_true",
        help="Skip mosaic validation step"
    )
    
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only regenerate plots from saved artifacts"
    )
    
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use subset of N cells for testing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = main(
            input_path=args.input,
            output_dir=args.output,
            group=args.group,
            all_groups=args.all_groups,
            skip_training=args.skip_training,
            skip_gmm=args.skip_gmm,
            skip_dec=args.skip_dec and not getattr(args, 'run_dec', False),
            skip_plots=args.skip_plots,
            skip_mosaic=args.skip_mosaic,
            visualize_only=args.visualize_only,
            subset_n=args.subset,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        
        for group, group_results in results.get('per_group', {}).items():
            if 'error' in group_results:
                print(f"\n{group}: ERROR - {group_results['error']}")
            else:
                print(f"\n{group}:")
                print(f"  Cells: {group_results['n_cells']}")
                print(f"  k*: {group_results['k_selected']}")
                print(f"  GMM purity: {group_results['gmm_metrics']['purity']:.3f}")
                print(f"  DEC purity: {group_results['dec_metrics']['purity']:.3f}")
                if 'mosaic_valid' in group_results:
                    mv = group_results['mosaic_valid']
                    mt = group_results['mosaic_total']
                    print(f"  Mosaic: {mv}/{mt} subtypes pass")
        
        # Print mosaic summary
        if results.get('mosaic_validation'):
            print("\n" + "-" * 40)
            print("Mosaic Validation Summary")
            print("-" * 40)
            mosaic = results['mosaic_validation']
            n_pass = sum(1 for r in mosaic if r['mosaic_validation'])
            n_total = len(mosaic)
            print(f"  Total: {n_pass}/{n_total} subtypes pass (threshold = {RF_COVERAGE_THRESHOLD})")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
