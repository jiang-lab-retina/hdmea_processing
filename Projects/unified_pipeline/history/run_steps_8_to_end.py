#!/usr/bin/env python
"""
Unified Pipeline Part 2: Steps 8-11

Loads an existing HDF5 file (from steps 1-7) and continues processing:
  8. Add Google Sheet metadata
  9. Add manual cell type labels
  10. Compute AP tracking
  11. Section by direction (DSGC)

Saves the final result to a new HDF5 file.

Usage:
    python run_steps_8_to_end.py --input path/to/steps1-7.h5
    python run_steps_8_to_end.py --input path/to/steps1-7.h5 --output path/to/final.h5
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

# Import step wrappers (steps 8-11 only)
from Projects.unified_pipeline.steps import (
    add_gsheet_step,
    add_cell_type_step,
    compute_ap_tracking_step,
    section_by_direction_step,
)

from Projects.unified_pipeline.config import (
    setup_logging,
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
# Pipeline Execution
# =============================================================================

def run_pipeline_steps_8_to_end(
    input_path: Path,
    output_path: Path,
) -> Path:
    """
    Load HDF5 from steps 1-7 and run steps 8-11.
    
    Args:
        input_path: Path to HDF5 file from steps 1-7
        output_path: Path to save final result
    
    Returns:
        Path to output HDF5 file
    """
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    print("=" * 70)
    print("Unified Pipeline: Steps 8-11 (GSheet to DSGC)")
    print("=" * 70)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # =========================================================================
    # Load session from HDF5
    # =========================================================================
    logger.info("Loading session from HDF5...")
    session = load_session_from_hdf5(input_path)
    logger.info(f"  Loaded {session.unit_count} units, dataset_id={session.dataset_id}")
    
    # Configuration
    ap_config = APTrackingConfig()
    dsgc_config = DSGCConfig()
    
    # =========================================================================
    # Step 8: Add Google Sheet metadata
    # =========================================================================
    session = add_gsheet_step(session=session)
    
    # =========================================================================
    # Step 9: Add manual cell type labels
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
    # Save Final Result
    # =========================================================================
    logger.info("Saving final result to HDF5...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session.save(output_path=output_path, overwrite=True)
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(green_success("STEPS 8-11 COMPLETE"))
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
        status = "[!]" if ":skipped" in step or ":failed" in step else "[+]"
        print(f"  {status} {step}")
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continue pipeline from steps 1-7 output, run steps 8-11"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to HDF5 file from steps 1-7"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (default: replace '_steps1-7.h5' with '_final.h5')"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Validate input
    if not args.input.exists():
        print(red_warning(f"Input file not found: {args.input}"))
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        # Replace suffix
        stem = args.input.stem
        if "_steps1-7" in stem:
            new_stem = stem.replace("_steps1-7", "_final")
        else:
            new_stem = stem + "_final"
        output_path = args.input.parent / f"{new_stem}.h5"
    else:
        output_path = args.output
    
    try:
        result_path = run_pipeline_steps_8_to_end(
            input_path=args.input,
            output_path=output_path,
        )
        print(f"\nSuccess! Output saved to: {result_path}")
    except Exception as e:
        print(red_warning(f"\nPipeline failed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

