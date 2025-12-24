"""
Pipeline Session Batch Processing Script

Processes all matched CMTR/CMCR pairs from the CSV mapping file.
"""
import csv
import logging
import traceback
from pathlib import Path

from hdmea.pipeline import create_session, load_recording_with_eimage_sta, extract_features
from hdmea.io import add_section_time, section_spike_times
from hdmea.features import compute_sta

# Enable logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_id_from_cmtr(cmtr_path: str) -> str:
    """
    Extract dataset_id from cmtr filename.
    Example: "2024.01.17-11.15.41-Rec-.cmtr" -> "2024.01.17-11.15.41-Rec"
    """
    cmtr_filename = Path(cmtr_path).stem  # "2024.01.17-11.15.41-Rec-"
    return cmtr_filename.rstrip("-")  # "2024.01.17-11.15.41-Rec"


def process_single_recording(cmcr_path: str, cmtr_path: str, output_dir: Path) -> bool:
    """
    Process a single recording through the full pipeline.
    
    Returns True if successful, False otherwise.
    """
    dataset_id = get_dataset_id_from_cmtr(cmtr_path)
    output_path = output_dir / f"{dataset_id}.h5"
    
    logger.info(f"Processing {dataset_id}...")
    logger.info(f"  CMCR: {cmcr_path}")
    logger.info(f"  CMTR: {cmtr_path}")
    
    # Create session
    session = create_session(dataset_id=dataset_id)
    
    # Load recording with eimage_sta
    session = load_recording_with_eimage_sta(
        cmcr_path=cmcr_path,
        cmtr_path=cmtr_path,
        duration_s=120.0,
        spike_limit=10000,
        window_range=(-10, 40),
        session=session,
    )
    
    # Add section time using playlist (deferred)
    session = add_section_time(
        playlist_name="play_optimization_set6_ipRGC_manual",
        session=session,
    )
    
    # Section spike times (deferred)
    session = section_spike_times(
        pad_margin=(0.0, 0.0),
        session=session,
    )
    
    # Compute STA (deferred)
    session = compute_sta(
        cover_range=(-60, 0),
        session=session,
    )
    
    # # Feature extraction (deferred) TODO: This need to be fixed
    # session = extract_features(
    #     features=["frif"],
    #     session=session,
    # )
    
    # Save to output directory
    session.save(output_path=output_path)
    
    logger.info(f"Saved {dataset_id} to {output_path}")
    logger.info(f"  Units: {session.unit_count}")
    logger.info(f"  Completed steps: {session.completed_steps}")
    logger.info(f"  Warnings: {len(session.warnings)}")
    
    return True


def main():
    """
    Main batch processing function.
    """
    # Paths
    csv_path = Path("tool_box/generate_data_path_list/pkl_to_cmtr_mapping.csv")
    output_dir = Path("Projects/pipeline_test/data")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and filter matched entries
    logger.info(f"Reading CSV: {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        matched_pairs = [row for row in reader if row["matched"] == "True"]
    
    logger.info(f"Found {len(matched_pairs)} matched pairs to process")
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    # Process each matched pair
    for i, row in enumerate(matched_pairs, 1):
        cmcr_path = row["cmcr_path"]
        cmtr_path = row["cmtr_path"]
        dataset_id = get_dataset_id_from_cmtr(cmtr_path)
        output_path = output_dir / f"{dataset_id}.h5"
        
        # Skip if output file already exists
        if output_path.exists():
            logger.info(f"[{i}/{len(matched_pairs)}] Skipping {dataset_id} - output file already exists")
            skipped.append(dataset_id)
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i}/{len(matched_pairs)}: {dataset_id}")
        logger.info(f"{'='*60}")
        
        try:
            success = process_single_recording(cmcr_path, cmtr_path, output_dir)
            if success:
                successful.append(dataset_id)
        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            logger.error(traceback.format_exc())
            failed.append((dataset_id, str(e)))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total pairs: {len(matched_pairs)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Skipped (already exists): {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.info("\nFailed recordings:")
        for dataset_id, error in failed:
            logger.info(f"  - {dataset_id}: {error}")


if __name__ == "__main__":
    main()
