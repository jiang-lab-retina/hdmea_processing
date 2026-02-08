"""
Run Pipeline for Step Change Analysis

This script orchestrates the full analysis workflow:
1. Load CMCR/CMTR recordings and save to HDF5
2. Align units across recordings
3. Extract and analyze responses
4. Generate visualization plots

Usage:
    python run_pipeline.py
    
Or import and call run_full_pipeline() from Python.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .specific_config import (
    PipelineConfig,
    default_config,
    get_all_test_file_paths,
    get_output_hdf5_path,
    get_grouped_hdf5_path,
    OUTPUT_DIR,
    FIGURES_DIR,
    DATA_FOLDER,
)
from .data_loader import (
    load_and_save_recording,
    load_recording_from_hdf5,
)
from .unit_alignment import (
    create_aligned_group,
    load_aligned_group_from_hdf5,
    add_signatures_to_data,
)
from .response_analysis import (
    summarize_response_timecourse,
    get_all_trace_features,
)
from .visualization import (
    plot_analysis_summary,
    plot_recording_summary,
    plot_step_responses_grid,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Steps
# =============================================================================

def step1_load_recordings(
    file_paths: Optional[List[Tuple[Path, Path, str]]] = None,
    config: Optional[PipelineConfig] = None,
    overwrite: bool = False,
) -> List[Path]:
    """
    Step 1: Load all recordings and save to HDF5.
    
    Args:
        file_paths: List of (cmcr_path, cmtr_path, description) tuples
        config: Pipeline configuration
        overwrite: Whether to overwrite existing HDF5 files
    
    Returns:
        List of HDF5 file paths
    """
    if config is None:
        config = default_config
    
    if file_paths is None:
        file_paths = get_all_test_file_paths(config.data_folder)
    
    logger.info("=" * 60)
    logger.info("Step 1: Loading recordings and saving to HDF5")
    logger.info("=" * 60)
    
    hdf5_paths = []
    
    for cmcr_path, cmtr_path, description in file_paths:
        logger.info(f"\nProcessing: {description}")
        logger.info(f"  CMCR: {cmcr_path}")
        logger.info(f"  CMTR: {cmtr_path}")
        
        # Check if files exist
        if not cmcr_path.exists():
            logger.error(f"  CMCR file not found: {cmcr_path}")
            continue
        if not cmtr_path.exists():
            logger.error(f"  CMTR file not found: {cmtr_path}")
            continue
        
        try:
            data, hdf5_path = load_and_save_recording(
                cmcr_path,
                cmtr_path,
                config=config,
                overwrite=overwrite,
            )
            
            hdf5_paths.append(hdf5_path)
            
            n_units = len(data.get("units", {}))
            high_quality = sum(
                1 for u in data.get("units", {}).values()
                if u.get("quality_index", 0) >= config.quality.quality_threshold
            )
            
            logger.info(f"  Saved: {hdf5_path}")
            logger.info(f"  Units: {n_units} total, {high_quality} high quality")
            
        except Exception as e:
            logger.error(f"  Error processing: {e}")
            raise
    
    logger.info(f"\nStep 1 complete: {len(hdf5_paths)} recordings saved")
    
    return hdf5_paths


def step2_align_units(
    hdf5_paths: List[Path],
    config: Optional[PipelineConfig] = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Step 2: Align units across recordings.
    
    Args:
        hdf5_paths: List of HDF5 file paths
        config: Pipeline configuration
    
    Returns:
        Tuple of (grouped_data, output_path)
    """
    if config is None:
        config = default_config
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Aligning units across recordings")
    logger.info("=" * 60)
    
    grouped_data, output_path = create_aligned_group(
        hdf5_paths,
        config=config,
        use_fixed_ref=True,
    )
    
    # Get alignment statistics
    chains_df = grouped_data.get("alignment_chains")
    if chains_df is not None and not chains_df.empty:
        n_chains = len(chains_df)
        n_complete = len(chains_df.dropna(how="any"))
        logger.info(f"  Alignment chains: {n_chains} total, {n_complete} complete")
    
    fixed_chains_df = grouped_data.get("fixed_alignment_chains")
    if fixed_chains_df is not None and not fixed_chains_df.empty:
        n_fixed = len(fixed_chains_df)
        n_fixed_complete = len(fixed_chains_df.dropna(how="any"))
        logger.info(f"  Fixed-ref chains: {n_fixed} total, {n_fixed_complete} complete")
    
    logger.info(f"\nStep 2 complete: {output_path}")
    
    return grouped_data, output_path


def step3_analyze_responses(
    grouped_data: Dict[str, Any],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Step 3: Analyze response features.
    
    Args:
        grouped_data: Grouped aligned data
        config: Pipeline configuration
    
    Returns:
        Dictionary with analysis results
    """
    if config is None:
        config = default_config
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Analyzing response features")
    logger.info("=" * 60)
    
    results = {}
    
    # ON response analysis
    logger.info("\nAnalyzing ON responses...")
    on_summary = summarize_response_timecourse(grouped_data, "ON", config)
    if on_summary:
        results["on_response"] = on_summary
        effect = on_summary.get("treatment_effect", {})
        if effect:
            logger.info(f"  Treatment effect: {effect.get('effect_percent', 0):.1f}%")
            logger.info(f"  N chains: {effect.get('n_chains', 0)}")
    
    # OFF response analysis
    logger.info("\nAnalyzing OFF responses...")
    off_summary = summarize_response_timecourse(grouped_data, "OFF", config)
    if off_summary:
        results["off_response"] = off_summary
        effect = off_summary.get("treatment_effect", {})
        if effect:
            logger.info(f"  Treatment effect: {effect.get('effect_percent', 0):.1f}%")
            logger.info(f"  N chains: {effect.get('n_chains', 0)}")
    
    logger.info("\nStep 3 complete")
    
    return results


def step4_generate_plots(
    grouped_data: Dict[str, Any],
    config: Optional[PipelineConfig] = None,
) -> List:
    """
    Step 4: Generate visualization plots.
    
    Args:
        grouped_data: Grouped aligned data
        config: Pipeline configuration
    
    Returns:
        List of figure paths
    """
    if config is None:
        config = default_config
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Generating visualization plots")
    logger.info("=" * 60)
    
    figures = plot_analysis_summary(
        grouped_data,
        config,
        save_dir=config.figures_dir,
    )
    
    # Also generate per-recording summaries
    recordings = grouped_data.get("recordings", {})
    for rec_name, rec_data in recordings.items():
        if "units" in rec_data:
            logger.info(f"\nGenerating summary for: {rec_name}")
            fig = plot_recording_summary(
                rec_data,
                rec_name,
                save_dir=config.figures_dir,
                config=config.visualization,
            )
            figures.append(fig)
    
    logger.info(f"\nStep 4 complete: {len(figures)} figures generated")
    logger.info(f"Figures saved in: {config.figures_dir}")
    
    return figures


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_full_pipeline(
    data_folder: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete step change analysis pipeline.
    
    Args:
        data_folder: Folder containing CMCR/CMTR files
        output_dir: Output directory for HDF5 files
        figures_dir: Output directory for figures
        overwrite: Whether to overwrite existing files
    
    Returns:
        Dictionary with pipeline results
    """
    # Create configuration
    config = PipelineConfig()
    
    if data_folder is not None:
        config.data_folder = Path(data_folder)
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    if figures_dir is not None:
        config.figures_dir = Path(figures_dir)
    
    # Ensure output directories exist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STEP CHANGE ANALYSIS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Data folder: {config.data_folder}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Figures dir: {config.figures_dir}")
    logger.info(f"Agonist start time: {config.agonist_start_time_s}s ({config.agonist_start_time_s/60:.1f} min)")
    
    results = {
        "config": config,
    }
    
    try:
        # Step 1: Load recordings
        hdf5_paths = step1_load_recordings(config=config, overwrite=overwrite)
        results["hdf5_paths"] = hdf5_paths
        
        if not hdf5_paths:
            logger.error("No recordings loaded. Stopping pipeline.")
            return results
        
        # Step 2: Align units
        grouped_data, grouped_path = step2_align_units(hdf5_paths, config)
        results["grouped_data"] = grouped_data
        results["grouped_path"] = grouped_path
        
        # Step 3: Analyze responses
        analysis_results = step3_analyze_responses(grouped_data, config)
        results["analysis"] = analysis_results
        
        # Step 4: Generate plots
        figures = step4_generate_plots(grouped_data, config)
        results["figures"] = figures
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"HDF5 files: {len(hdf5_paths)}")
        logger.info(f"Grouped file: {grouped_path}")
        logger.info(f"Figures: {len(figures)}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    
    return results


def run_from_existing_hdf5(
    grouped_hdf5_path: Path,
    figures_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run analysis on existing grouped HDF5 file.
    
    Args:
        grouped_hdf5_path: Path to grouped HDF5 file
        figures_dir: Output directory for figures
    
    Returns:
        Dictionary with analysis results
    """
    config = default_config
    
    if figures_dir is not None:
        config.figures_dir = Path(figures_dir)
    
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading grouped data from: {grouped_hdf5_path}")
    
    grouped_data = load_aligned_group_from_hdf5(
        grouped_hdf5_path,
        load_full_recordings=True,
    )
    
    # Add signatures to loaded recordings
    for rec_name, rec_data in grouped_data.get("recordings", {}).items():
        if "units" in rec_data:
            add_signatures_to_data(rec_data)
    
    # Run analysis
    analysis_results = step3_analyze_responses(grouped_data, config)
    
    # Generate plots
    figures = step4_generate_plots(grouped_data, config)
    
    return {
        "grouped_data": grouped_data,
        "analysis": analysis_results,
        "figures": figures,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Step Change Analysis Pipeline",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Folder containing CMCR/CMTR files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--from-hdf5",
        type=str,
        default=None,
        help="Run from existing grouped HDF5 file",
    )
    
    args = parser.parse_args()
    
    if args.from_hdf5:
        results = run_from_existing_hdf5(
            Path(args.from_hdf5),
            Path(args.figures_dir) if args.figures_dir else None,
        )
    else:
        results = run_full_pipeline(
            data_folder=Path(args.data_folder) if args.data_folder else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            figures_dir=Path(args.figures_dir) if args.figures_dir else None,
            overwrite=args.overwrite,
        )
    
    logger.info("Done!")
