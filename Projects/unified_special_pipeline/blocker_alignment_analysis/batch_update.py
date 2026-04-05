#!/usr/bin/env python
"""
Batch Update: Add Axon Labels + AP Tracking + DSGC

Loads existing H5 files from batch_pipeline output, adds manual axon_type
labels from the folder structure, runs AP tracking (step 10) and DSGC
sectioning (step 11), then saves to a new export folder.

Label folder structure:
    LABEL_ROOT/{condition_folder}/{dataset_id}/{CellType}/*.png
    
    Where:
    - condition_folder is one of 6 blocker condition subfolders
    - dataset_id matches the H5 stem (e.g., "2025.09.04-10.11.09-Rec")
    - CellType is RGC, AC, Other, or Unknown
    - Unit number extracted from {dataset_id}_unit{N}.png

Usage:
    python batch_update.py
    python batch_update.py --start 0 --end 2
    python batch_update.py --overwrite
    python batch_update.py --debug
"""

import argparse
import logging
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import load_session_from_hdf5, PipelineSession

# Import step wrappers from unified pipeline
from Projects.unified_pipeline.steps import (
    compute_ap_tracking_step,
    section_by_direction_step,
)

# Import config classes from unified pipeline
from Projects.unified_pipeline.config import (
    setup_logging,
    APTrackingConfig,
    DSGCConfig,
    green_success,
    red_warning,
)

# Import local config (handle both module and direct execution)
try:
    from .specific_config import ACTIVE_EXPERIMENT, OUTPUT_DIR, EXPORT_DIR, LABEL_ROOT
except ImportError:
    from specific_config import ACTIVE_EXPERIMENT, OUTPUT_DIR, EXPORT_DIR, LABEL_ROOT

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Cell type labels (subfolder names under each dataset_id folder)
CELL_TYPES = ["RGC", "AC", "Other", "Unknown"]

# Step names used for tracking completion
STEP_NAME_AXON_LABEL = "add_cell_type_labels"
STEP_NAME_AP_TRACKING = "compute_ap_tracking"
STEP_NAME_DSGC = "section_by_direction"

# auto_label group and feature name (same as cell_type.py)
AUTO_LABEL_GROUP = "auto_label"
FEATURE_NAME = "axon_type"

# Input/output directories
INPUT_DIR = OUTPUT_DIR


# =============================================================================
# Axon Label Parsing
# =============================================================================

def find_dataset_label_folder(
    label_root: Path,
    dataset_id: str,
) -> Optional[Path]:
    """
    Search all condition subfolders for a dataset_id match.

    The label_root has condition subfolders, each containing
    dataset_id subfolders. We need to find which condition folder
    contains our dataset_id.

    Args:
        label_root: Root of the label folder hierarchy
        dataset_id: Dataset identifier (e.g., "2025.09.04-10.11.09-Rec")

    Returns:
        Path to the dataset folder, or None if not found
    """
    if not label_root.exists():
        logger.warning(f"Label root not found: {label_root}")
        return None

    # Search each condition subfolder
    for condition_folder in sorted(label_root.iterdir()):
        if not condition_folder.is_dir():
            continue

        dataset_folder = condition_folder / dataset_id
        if dataset_folder.exists() and dataset_folder.is_dir():
            logger.debug(
                f"  Found label folder: {condition_folder.name}/{dataset_id}"
            )
            return dataset_folder

    # Not found with exact name -- try with dot/hyphen normalization
    # H5 dataset_id uses hyphens: 2025.09.04-10.11.09-Rec
    # Some folders might use dots: 2025.09.04.10.11.09.Rec
    alt_id = dataset_id.replace("-", ".")
    for condition_folder in sorted(label_root.iterdir()):
        if not condition_folder.is_dir():
            continue

        alt_folder = condition_folder / alt_id
        if alt_folder.exists() and alt_folder.is_dir():
            logger.debug(
                f"  Found label folder (alt): {condition_folder.name}/{alt_id}"
            )
            return alt_folder

    return None


def parse_axon_labels(
    dataset_folder: Path,
    cell_types: List[str] = None,
) -> Dict[int, str]:
    """
    Parse manual axon labels from folder structure.

    Reads the cell type subfolders (RGC, AC, Other, Unknown) and extracts
    unit numbers from PNG filenames.

    Args:
        dataset_folder: Path to the dataset folder containing cell type subfolders
        cell_types: List of cell type subfolder names to search

    Returns:
        Dict mapping unit number (int) to cell type label (str, lowercase)
    """
    if cell_types is None:
        cell_types = CELL_TYPES

    labels = {}

    for cell_type in cell_types:
        cell_type_folder = dataset_folder / cell_type

        if not cell_type_folder.exists():
            continue

        # Find all PNG files (including nested)
        png_files = list(cell_type_folder.rglob("*.png"))

        for png_file in png_files:
            # Extract unit number from filename like
            # "2025.09.04-10.11.09-Rec_unit10.png"
            match = re.search(r"_unit(\d+)\.png$", png_file.name, re.IGNORECASE)
            if match:
                unit_num = int(match.group(1))
                labels[unit_num] = cell_type.lower()

    return labels


def add_axon_label_step(
    *,
    label_root: Path = LABEL_ROOT,
    session: PipelineSession,
) -> PipelineSession:
    """
    Add manual axon_type labels to units from the blocker label folder.

    Searches all condition subfolders in label_root for the dataset_id,
    then reads cell type labels from the RGC/AC/Other/Unknown subfolders.

    Args:
        label_root: Root of the blocker label folder hierarchy
        session: Pipeline session (required)

    Returns:
        Updated session with axon_type labels
    """
    logger.info(f"Adding axon labels for {session.dataset_id}...")

    # Find the label folder for this dataset
    dataset_folder = find_dataset_label_folder(label_root, session.dataset_id)

    if dataset_folder is None:
        logger.warning(f"  No label folder found for {session.dataset_id}")
        session.warnings.append(f"{STEP_NAME_AXON_LABEL}: No label folder found")
        # Still mark as complete so downstream steps can run
        session.mark_step_complete(STEP_NAME_AXON_LABEL)
        return session

    # Parse labels from folder structure
    labels = parse_axon_labels(dataset_folder)

    if not labels:
        logger.warning(f"  No labels parsed from {dataset_folder}")
        session.warnings.append(f"{STEP_NAME_AXON_LABEL}: No labels parsed")

    # Apply labels to units
    labeled_count = 0
    no_label_count = 0

    for unit_id in session.units.keys():
        # Extract unit number from unit_id (e.g., "unit_001" -> 1)
        match = re.search(r"unit_(\d+)", unit_id)
        if match:
            unit_num = int(match.group(1))

            if unit_num in labels:
                cell_type = labels[unit_num]
                labeled_count += 1
            else:
                cell_type = "no_label"
                no_label_count += 1
        else:
            cell_type = "no_label"
            no_label_count += 1

        # Add auto_label group with axon_type
        if AUTO_LABEL_GROUP not in session.units[unit_id]:
            session.units[unit_id][AUTO_LABEL_GROUP] = {}

        session.units[unit_id][AUTO_LABEL_GROUP][FEATURE_NAME] = cell_type

    logger.info(
        f"  Axon labels: {labeled_count} labeled, {no_label_count} no_label "
        f"(from {dataset_folder.parent.name}/{dataset_folder.name})"
    )

    session.mark_step_complete(STEP_NAME_AXON_LABEL)
    return session


# =============================================================================
# Single File Update
# =============================================================================

def update_single_h5(
    h5_path: Path,
    export_dir: Path,
    label_root: Path = LABEL_ROOT,
    ap_config: Optional[APTrackingConfig] = None,
    dsgc_config: Optional[DSGCConfig] = None,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Update a single H5 file: add axon labels, AP tracking, DSGC.

    Args:
        h5_path: Path to input H5 file
        export_dir: Directory to save updated file
        label_root: Root of the label folder hierarchy
        ap_config: Configuration for AP tracking
        dsgc_config: Configuration for DSGC
        overwrite: Whether to overwrite existing export

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if ap_config is None:
        ap_config = APTrackingConfig()
    if dsgc_config is None:
        dsgc_config = DSGCConfig()

    dataset_id = h5_path.stem
    export_path = export_dir / h5_path.name

    # Check if export already exists
    if export_path.exists() and not overwrite:
        return True, "skipped"

    # Validate input
    if not h5_path.exists():
        return False, f"Input H5 not found: {h5_path}"

    logger.info(f"Loading {dataset_id}...")

    # Step 1: Load existing H5 into PipelineSession
    try:
        session = load_session_from_hdf5(h5_path)
    except Exception as e:
        return False, f"Failed to load H5: {e}"

    logger.info(
        f"  Loaded: {session.unit_count} units, "
        f"{len(session.completed_steps)} completed steps"
    )

    # Clear relevant steps so they re-execute fresh
    session.completed_steps.discard(STEP_NAME_AXON_LABEL)
    session.completed_steps.discard(STEP_NAME_AP_TRACKING)
    session.completed_steps.discard(STEP_NAME_DSGC)

    # Step 2: Add axon labels
    session = add_axon_label_step(label_root=label_root, session=session)

    # Step 3: Compute AP tracking (step 10)
    session = compute_ap_tracking_step(
        config=ap_config,
        session=session,
    )

    # Step 4: Section by direction / DSGC (step 11)
    session = section_by_direction_step(
        config=dsgc_config,
        session=session,
    )

    # Step 5: Save to export folder
    export_dir.mkdir(parents=True, exist_ok=True)
    session.save(output_path=export_path, overwrite=overwrite)

    logger.info(f"  Exported: {export_path}")
    logger.info(
        f"  Units: {session.unit_count}, "
        f"Steps: {len(session.completed_steps)}"
    )

    return True, None


# =============================================================================
# Batch Processing
# =============================================================================

def discover_h5_files(input_dir: Path) -> List[Path]:
    """
    Discover all H5 files in the input directory.

    Args:
        input_dir: Directory containing H5 files

    Returns:
        Sorted list of H5 file paths
    """
    if not input_dir.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        return []

    h5_files = sorted(input_dir.glob("*.h5"))
    logger.info(f"Discovered {len(h5_files)} H5 files in {input_dir}")
    return h5_files


def run_batch_update(
    input_dir: Path = INPUT_DIR,
    export_dir: Path = EXPORT_DIR,
    label_root: Path = LABEL_ROOT,
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Run batch update on all H5 files.

    Args:
        input_dir: Directory containing input H5 files
        export_dir: Directory to save updated files
        label_root: Root of the label folder hierarchy
        start_index: Starting index (0-based)
        end_index: Ending index (exclusive, None for all)
        overwrite: Whether to overwrite existing exports

    Returns:
        Tuple of (successful, skipped, failed) lists
    """
    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)

    # Discover H5 files
    all_files = discover_h5_files(input_dir)

    if not all_files:
        logger.warning("No H5 files found to process")
        return [], [], []

    # Apply index range
    if end_index is not None:
        files_to_process = all_files[start_index:end_index]
    else:
        files_to_process = all_files[start_index:]

    logger.info(f"Processing {len(files_to_process)} of {len(all_files)} files")

    # Track results
    successful: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []

    start_time = time.time()

    for i, h5_path in enumerate(files_to_process, 1):
        dataset_id = h5_path.stem

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(files_to_process)}] {dataset_id}")
        logger.info(f"{'='*60}")

        try:
            success, error = update_single_h5(
                h5_path=h5_path,
                export_dir=export_dir,
                label_root=label_root,
                overwrite=overwrite,
            )

            if success:
                if error == "skipped":
                    logger.info(f"  Skipped - export already exists")
                    skipped.append(dataset_id)
                else:
                    successful.append(dataset_id)
            else:
                logger.error(f"  Failed: {error}")
                failed.append((dataset_id, error or "Unknown error"))

        except Exception as e:
            logger.error(f"  Exception: {e}")
            logger.debug(traceback.format_exc())
            failed.append((dataset_id, str(e)))

    # Summary
    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info("BATCH UPDATE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {len(files_to_process)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if failed:
        logger.info("\nFailed files:")
        for dataset_id, error in failed[:10]:
            logger.info(f"  - {dataset_id}: {error}")
        if len(failed) > 10:
            logger.info(f"  ... and {len(failed) - 10} more")

    return successful, skipped, failed


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch update: add axon labels from manual label folders, "
            "run AP tracking and DSGC, save to export folder"
        )
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment profile key (e.g. _ptx_str, _ptx, _str). "
             "Already resolved at import time; accepted here so argparse "
             "does not reject the flag.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DIR,
        help=f"Input H5 folder (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=EXPORT_DIR,
        help=f"Export folder (default: {EXPORT_DIR})",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=LABEL_ROOT,
        help=f"Label root folder (default: {LABEL_ROOT})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index (0-based, default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending index (exclusive, default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing export files (default: skip existing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    print("=" * 70)
    print("Batch Update: Axon Labels + AP Tracking + DSGC")
    print("=" * 70)
    print(f"Experiment: {ACTIVE_EXPERIMENT}")
    print(f"Input:      {args.input}")
    print(f"Export:     {args.export}")
    print(f"Labels:     {args.label_root}")
    print(f"Range:      {args.start} to {args.end or 'end'}")
    print(f"Overwrite:  {args.overwrite}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        successful, skipped, failed = run_batch_update(
            input_dir=args.input,
            export_dir=args.export,
            label_root=args.label_root,
            start_index=args.start,
            end_index=args.end,
            overwrite=args.overwrite,
        )

        print("\n" + "=" * 70)
        if len(failed) == 0:
            print(green_success("BATCH UPDATE COMPLETE - ALL SUCCESSFUL"))
        else:
            print(red_warning(f"BATCH UPDATE COMPLETE - {len(failed)} FAILED"))
        print("=" * 70)
        print(f"Successful: {len(successful)}")
        print(f"Skipped:    {len(skipped)}")
        print(f"Failed:     {len(failed)}")

    except Exception as e:
        print(red_warning(f"\nBatch update failed: {e}"))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
