#!/usr/bin/env python
"""
Run Steps 6-11: Post-CMCR/CMTR Processing

This script processes steps that do NOT require CMCR/CMTR file access:
- Step 6: Extract soma geometry (from eimage_sta)
- Step 7: Extract RF geometry (from dense noise STA)
- Step 8: Add Google Sheet metadata
- Step 9: Add manual cell type labels
- Step 10: Compute AP tracking (CNN model)
- Step 11: Section by direction (DSGC)

Input: HDF5 file from load_cmcr_cmtr.py (steps 1-5 complete)
Output: Final HDF5 file with all 11 steps complete

Usage:
    python run_steps_6_to_end.py --input path/to/steps1-5.h5
    
    # Specify output file:
    python run_steps_6_to_end.py --input in.h5 --output out.h5
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import PipelineSession
from hdmea.pipeline.loader import load_session_from_hdf5

# Import step wrappers
from Projects.unified_pipeline.steps import (
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
    DEFAULT_OUTPUT_DIR,
    green_success,
    red_warning,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = DEFAULT_OUTPUT_DIR


# =============================================================================
# Main Processing Function
# =============================================================================

def run_steps_6_to_end(
    session: PipelineSession,
    geometry_config: Optional[GeometryConfig] = None,
    ap_config: Optional[APTrackingConfig] = None,
    dsgc_config: Optional[DSGCConfig] = None,
) -> PipelineSession:
    """
    Run steps 6-11 on a session that has completed steps 1-5.
    
    These steps require external resources (model files, CSV files, etc.)
    but NOT the original CMCR/CMTR files.
    
    Args:
        session: PipelineSession with steps 1-5 complete
        geometry_config: Configuration for geometry extraction
        ap_config: Configuration for AP tracking
        dsgc_config: Configuration for DSGC sectioning
    
    Returns:
        Updated session with all steps complete
    """
    logger = logging.getLogger(__name__)
    
    if geometry_config is None:
        geometry_config = GeometryConfig()
    if ap_config is None:
        ap_config = APTrackingConfig()
    if dsgc_config is None:
        dsgc_config = DSGCConfig()
    
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("Running Steps 6-11 (Post-CMCR/CMTR)")
    logger.info("=" * 60)
    logger.info(f"Dataset:  {session.dataset_id}")
    logger.info(f"Units:    {session.unit_count}")
    
    # =========================================================================
    # Step 6: Extract soma geometry
    # =========================================================================
    session = extract_soma_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )
    
    # =========================================================================
    # Step 7: Extract RF geometry
    # =========================================================================
    session = extract_rf_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )
    
    # =========================================================================
    # Step 8: Add Google Sheet metadata
    # =========================================================================
    session = add_gsheet_step(session=session)
    
    # =========================================================================
    # Step 9: Add cell type labels
    # =========================================================================
    session = add_cell_type_step(session=session)
    
    # =========================================================================
    # Step 10: Compute AP tracking
    # =========================================================================
    session = compute_ap_tracking_step(
        config=ap_config,
        session=session,
    )
    
    # =========================================================================
    # Step 11: Section by direction (DSGC)
    # =========================================================================
    session = section_by_direction_step(
        config=dsgc_config,
        session=session,
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    logger.info("-" * 60)
    logger.info(f"Steps 6-11 complete in {elapsed:.1f}s")
    logger.info(f"Total completed steps: {len(session.completed_steps)}")
    
    return session


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run steps 6-11 on a pre-loaded session"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input HDF5 file (from load_cmcr_cmtr.py)"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output HDF5 file (default: {input_stem}_final.h5)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not args.input.exists():
        print(red_warning(f"Input file not found: {args.input}"))
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: replace _steps1-5 with _final, or add _final suffix
        stem = args.input.stem
        if stem.endswith("_steps1-5"):
            new_stem = stem.replace("_steps1-5", "_final")
        else:
            new_stem = f"{stem}_final"
        output_path = args.input.parent / f"{new_stem}.h5"
    
    print("=" * 70)
    print("Run Steps 6-11 (Post-CMCR/CMTR)")
    print("=" * 70)
    print(f"Input:    {args.input}")
    print(f"Output:   {output_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Load session from HDF5
        logger.info("Loading session from HDF5...")
        session = load_session_from_hdf5(args.input)
        logger.info(f"  Loaded {session.unit_count} units, {len(session.completed_steps)} steps")
        
        # Run steps 6-11
        session = run_steps_6_to_end(session)
        
        # Save final result
        logger.info("Saving final result to HDF5...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        session.save(output_path=output_path, overwrite=args.overwrite)
        
        print("\n" + "=" * 70)
        print(green_success("STEPS 6-11 COMPLETE"))
        print("=" * 70)
        print(f"Output:   {output_path}")
        print(f"Units:    {session.unit_count}")
        print(f"Steps:    {len(session.completed_steps)}")
        
        print("\nCompleted steps:")
        for step in sorted(session.completed_steps):
            status = "[!]" if ":skipped" in step or ":failed" in step else "[+]"
            print(f"  {status} {step}")
        
        if session.warnings:
            print(f"\nWarnings ({len(session.warnings)}):")
            for warning in session.warnings[:5]:
                print(f"  - {warning}")
            if len(session.warnings) > 5:
                print(f"  ... and {len(session.warnings) - 5} more")
        
        print(f"\nSuccess! Output saved to: {output_path}")
        
    except Exception as e:
        print(red_warning(f"\nFailed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

