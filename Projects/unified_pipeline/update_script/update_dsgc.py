#!/usr/bin/env python
"""
Batch update DSGC direction sectioning for existing HDF5 files.

Loads HDF5 files from export_all_steps/, runs the DSGC direction sectioning
step, and saves results to export_dsgc_updated/.

Usage:
    python update_dsgc.py
    python update_dsgc.py --start 0 --end 10
    python update_dsgc.py --overwrite
    python update_dsgc.py --input path/to/input --output path/to/output
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline.loader import load_session_from_hdf5
from Projects.unified_pipeline.steps import section_by_direction_step
from Projects.unified_pipeline.config import DSGCConfig, setup_logging, green_success, red_warning

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path(__file__).parent.parent / "export_all_steps"
OUTPUT_DIR = Path(__file__).parent.parent / "export_dsgc_updated"


# =============================================================================
# Processing Functions
# =============================================================================

def process_single_file(
    hdf5_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single HDF5 file through DSGC step.
    
    Args:
        hdf5_path: Path to input HDF5 file
        output_dir: Directory to save output
        overwrite: Whether to overwrite existing output
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    output_path = output_dir / hdf5_path.name
    
    # Check if output already exists
    if output_path.exists() and not overwrite:
        return False, "skipped (exists)"
    
    try:
        # Load existing HDF5 as PipelineSession
        logger.info(f"  Loading {hdf5_path.name}...")
        session = load_session_from_hdf5(hdf5_path)
        
        # Clear the DSGC step from completed_steps so it runs again
        session.completed_steps.discard("section_by_direction")
        
        # Run DSGC step
        logger.info(f"  Running DSGC direction sectioning...")
        session = section_by_direction_step(config=DSGCConfig(), session=session)
        
        # Save to output
        logger.info(f"  Saving to {output_path.name}...")
        session.save(output_path=output_path, overwrite=overwrite)
        
        return True, f"success ({session.unit_count} units)"
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        logger.debug(traceback.format_exc())
        return False, f"failed: {e}"


def get_hdf5_files(input_dir: Path) -> List[Path]:
    """Get sorted list of HDF5 files in input directory."""
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []
    
    files = sorted(input_dir.glob("*.h5"))
    return files


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch update DSGC direction sectioning for existing HDF5 files."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory containing HDF5 files (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for updated files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for batch processing (0-based, inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for batch processing (exclusive)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Get list of HDF5 files
    hdf5_files = get_hdf5_files(args.input)
    
    if not hdf5_files:
        logger.error("No HDF5 files found in input directory")
        return 1
    
    # Apply start/end range
    total_files = len(hdf5_files)
    hdf5_files = hdf5_files[args.start:args.end]
    
    logger.info("=" * 60)
    logger.info("DSGC Update Batch Processing")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Processing: {len(hdf5_files)} files (index {args.start} to {args.end or total_files})")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 60)
    
    # Track results
    successful: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    
    start_time = time.time()
    
    # Process each file
    for i, hdf5_path in enumerate(hdf5_files, args.start + 1):
        file_name = hdf5_path.stem
        
        logger.info("")
        logger.info(f"[{i}/{total_files}] {file_name}")
        logger.info("-" * 40)
        
        success, message = process_single_file(
            hdf5_path=hdf5_path,
            output_dir=args.output,
            overwrite=args.overwrite,
        )
        
        if "skipped" in message:
            skipped.append(file_name)
            logger.info(f"  -> Skipped (output already exists)")
        elif success:
            successful.append(file_name)
            green_success(f"  -> {message}")
        else:
            failed.append(file_name)
            red_warning(f"  -> {message}")
    
    # Summary
    elapsed = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(hdf5_files)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Skipped:    {len(skipped)}")
    logger.info(f"  Failed:     {len(failed)}")
    logger.info(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if failed:
        logger.info("")
        logger.info("Failed files:")
        for f in failed:
            logger.info(f"  - {f}")
    
    logger.info("=" * 60)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

