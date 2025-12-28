#!/usr/bin/env python
"""
Unified Pipeline: Resume Processing from HDF5

This script loads an existing HDF5 file and resumes processing
from the last checkpoint, skipping already-completed steps.

Usage:
    python run_single_from_hdf5.py path/to/existing.h5
    
    # Resume and save to different output:
    python run_single_from_hdf5.py input.h5 --output output.h5
    
    # Run specific additional steps only:
    python run_single_from_hdf5.py input.h5 --steps ap_tracking,dsgc
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import load_session_from_hdf5

# Import step wrappers
from Projects.unified_pipeline.steps import (
    add_metadata_step,
    extract_soma_geometry_step,
    extract_rf_geometry_step,
    add_gsheet_step,
    add_cell_type_step,
    compute_ap_tracking_step,
    section_by_direction_step,
)

from Projects.unified_pipeline.config import (
    setup_logging,
    GeometryConfig,
    APTrackingConfig,
    DSGCConfig,
    green_success,
    red_warning,
)


# =============================================================================
# Configuration
# =============================================================================

# Available steps that can be run after loading from HDF5
AVAILABLE_STEPS = {
    'metadata': add_metadata_step,
    'soma_geometry': extract_soma_geometry_step,
    'rf_geometry': extract_rf_geometry_step,
    'gsheet': add_gsheet_step,
    'cell_type': add_cell_type_step,
    'ap_tracking': compute_ap_tracking_step,
    'dsgc': section_by_direction_step,
}

# Default step order for resume
DEFAULT_STEP_ORDER = [
    'metadata',
    'soma_geometry', 
    'rf_geometry',
    'gsheet',
    'cell_type',
    'ap_tracking',
    'dsgc',
]


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_pipeline_from_hdf5(
    input_path: Path,
    output_path: Path = None,
    steps: list = None,
    overwrite: bool = False,
) -> Path:
    """
    Resume pipeline from existing HDF5 file.
    
    Args:
        input_path: Path to existing HDF5 file
        output_path: Output path (defaults to input_path with overwrite=True)
        steps: List of step names to run (None = all)
        overwrite: Whether to overwrite output file
    
    Returns:
        Path to output HDF5 file
    """
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    print("=" * 70)
    print("Unified Pipeline: Resume from HDF5")
    print("=" * 70)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path or input_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load existing session
    logger.info(f"Loading session from: {input_path}")
    session = load_session_from_hdf5(input_path)
    
    print(f"\nLoaded session:")
    print(f"  Dataset ID: {session.dataset_id}")
    print(f"  Units: {session.unit_count}")
    print(f"  Completed steps: {len(session.completed_steps)}")
    
    if session.completed_steps:
        print("\n  Already completed:")
        for step in sorted(session.completed_steps):
            print(f"    ✓ {step}")
    
    # Determine output path
    if output_path is None:
        output_path = input_path
        overwrite = True  # Must overwrite if updating in place
    
    # Determine which steps to run
    if steps is None:
        steps_to_run = DEFAULT_STEP_ORDER
    else:
        steps_to_run = steps
    
    print(f"\n  Steps to run:")
    for step in steps_to_run:
        if step in AVAILABLE_STEPS:
            print(f"    • {step}")
        else:
            print(f"    ✗ {step} (unknown)")
    
    # Configuration
    geometry_config = GeometryConfig()
    ap_config = APTrackingConfig()
    dsgc_config = DSGCConfig()
    
    # =========================================================================
    # Run requested steps
    # =========================================================================
    
    print("\n" + "-" * 70)
    
    for step_name in steps_to_run:
        if step_name not in AVAILABLE_STEPS:
            logger.warning(f"Unknown step: {step_name}")
            continue
        
        step_func = AVAILABLE_STEPS[step_name]
        
        # Pass appropriate config for each step
        if step_name == 'soma_geometry':
            session = step_func(
                frame_range=geometry_config.frame_range,
                threshold_fraction=geometry_config.threshold_fraction,
                session=session,
            )
        elif step_name == 'rf_geometry':
            session = step_func(
                frame_range=geometry_config.frame_range,
                threshold_fraction=geometry_config.threshold_fraction,
                session=session,
            )
        elif step_name == 'ap_tracking':
            session = step_func(config=ap_config, session=session)
        elif step_name == 'dsgc':
            session = step_func(config=dsgc_config, session=session)
        else:
            session = step_func(session=session)
    
    # =========================================================================
    # Save if output path differs from input
    # =========================================================================
    
    if output_path != input_path:
        logger.info(f"Saving to: {output_path}")
        session.save(output_path=output_path, overwrite=overwrite)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(green_success("PIPELINE COMPLETE"))
    print("=" * 70)
    print(f"Output:   {output_path}")
    print(f"Units:    {session.unit_count}")
    print(f"Steps:    {len(session.completed_steps)}")
    print(f"Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if session.warnings:
        print(f"\nWarnings ({len(session.warnings)}):")
        for warning in session.warnings[:5]:
            print(f"  - {warning}")
        if len(session.warnings) > 5:
            print(f"  ... and {len(session.warnings) - 5} more")
    
    print("\nCompleted steps:")
    for step in sorted(session.completed_steps):
        status = "⚠️" if ":skipped" in step or ":failed" in step else "✓"
        print(f"  {status} {step}")
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Resume pipeline from existing HDF5 file"
    )
    parser.add_argument(
        "input", type=Path,
        help="Path to existing HDF5 file"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: update input file)"
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated list of steps to run (default: all)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting output file"
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="List available steps and exit"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # List steps and exit
    if args.list_steps:
        print("Available steps:")
        for name in DEFAULT_STEP_ORDER:
            print(f"  • {name}")
        return
    
    # Validate input
    if not args.input.exists():
        print(red_warning(f"Input file not found: {args.input}"))
        sys.exit(1)
    
    # Parse steps
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(',')]
    
    # Run pipeline
    try:
        output_path = run_pipeline_from_hdf5(
            input_path=args.input,
            output_path=args.output,
            steps=steps,
            overwrite=args.overwrite,
        )
        print(f"\nSuccess! Output saved to: {output_path}")
    except Exception as e:
        print(red_warning(f"\nPipeline failed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

