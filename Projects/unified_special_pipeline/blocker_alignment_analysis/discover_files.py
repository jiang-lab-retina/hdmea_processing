#!/usr/bin/env python
"""
File Discovery and CSV Index Generation for Blocker Alignment Analysis

This script:
1. Loads the Google Sheet CSV cache
2. Filters rows for dates 2025.09.04 through 2025.09.19
3. Converts gsheet filenames (dots) to disk filenames (hyphens)
4. Derives CMTR filenames (trailing hyphen: Rec-.cmtr)
5. Parses Condition column (extract part before first comma)
6. Searches O: drive date folders for matching CMCR and CMTR files
7. Pairs before/gaba_gly recordings by Chip + date with temporal validation
8. Outputs a CSV index (output/file_index.csv)

Usage:
    python discover_files.py
    python discover_files.py --refresh-gsheet   # Re-download gsheet first
    python discover_files.py --debug             # Enable debug logging
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import local config (handle both module and direct execution)
try:
    from .specific_config import (
        ACTIVE_EXPERIMENT,
        GSHEET_CSV_PATH,
        OUTPUT_DIR,
        FILE_INDEX_CSV,
        DATE_START,
        DATE_END,
        DATA_DRIVES,
        DATA_FOLDER_PATTERN,
        SUBFOLDER_BEFORE,
        SUBFOLDER_GABA_GLY,
        BEFORE_CONDITION,
        AFTER_CONDITION,
        gsheet_filename_to_disk,
        get_cmtr_from_cmcr,
        extract_date_from_gsheet_filename,
        extract_time_from_gsheet_filename,
        date_to_compact,
        find_data_folder_for_date,
        parse_condition_for_playlist,
    )
except ImportError:
    from specific_config import (
        ACTIVE_EXPERIMENT,
        GSHEET_CSV_PATH,
        OUTPUT_DIR,
        FILE_INDEX_CSV,
        DATE_START,
        DATE_END,
        DATA_DRIVES,
        DATA_FOLDER_PATTERN,
        SUBFOLDER_BEFORE,
        SUBFOLDER_GABA_GLY,
        BEFORE_CONDITION,
        AFTER_CONDITION,
        gsheet_filename_to_disk,
        get_cmtr_from_cmcr,
        extract_date_from_gsheet_filename,
        extract_time_from_gsheet_filename,
        date_to_compact,
        find_data_folder_for_date,
        parse_condition_for_playlist,
    )

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: int = logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Google Sheet Loading
# =============================================================================

def load_gsheet(gsheet_csv_path: Path = GSHEET_CSV_PATH) -> Optional[pd.DataFrame]:
    """Load the Google Sheet CSV cache."""
    logger = logging.getLogger(__name__)

    if not gsheet_csv_path.exists():
        logger.error(f"Gsheet CSV not found: {gsheet_csv_path}")
        return None

    try:
        df = pd.read_csv(gsheet_csv_path)
        logger.info(f"Loaded gsheet with {len(df)} rows from {gsheet_csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading gsheet: {e}")
        return None


def refresh_gsheet() -> Optional[pd.DataFrame]:
    """Re-download gsheet from Google Sheets API and return DataFrame."""
    logger = logging.getLogger(__name__)
    try:
        from Projects.load_gsheet.load_gsheet import import_gsheet_v2
        df = import_gsheet_v2()
        logger.info(f"Refreshed gsheet: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to refresh gsheet: {e}")
        logger.info("Falling back to cached CSV")
        return load_gsheet()


# =============================================================================
# Date Filtering
# =============================================================================

def filter_by_date_range(
    df: pd.DataFrame,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> pd.DataFrame:
    """
    Filter gsheet rows by date range extracted from File_name column.

    Args:
        df: Full gsheet DataFrame
        date_start: Inclusive start date (e.g., "2025.09.04")
        date_end: Inclusive end date (e.g., "2025.09.19")

    Returns:
        Filtered DataFrame
    """
    logger = logging.getLogger(__name__)

    # Extract date from File_name
    df = df.copy()
    df["_date"] = df["File_name"].astype(str).apply(extract_date_from_gsheet_filename)

    # Filter by date range (string comparison works for YYYY.MM.DD format)
    mask = (df["_date"] >= date_start) & (df["_date"] <= date_end)
    filtered = df[mask].copy()

    logger.info(
        f"Filtered {len(filtered)} rows from date range {date_start} to {date_end}"
    )

    unique_dates = sorted(filtered["_date"].unique())
    logger.info(f"Unique dates: {unique_dates}")

    return filtered


# =============================================================================
# File Discovery on O: Drive
# =============================================================================

def determine_recording_type(condition: str) -> str:
    """
    Determine recording type from Condition column.

    Returns:
        'before' or 'gaba_gly' or 'unknown'
    """
    if not condition or pd.isna(condition):
        return "unknown"
    condition_str = str(condition)
    if BEFORE_CONDITION in condition_str:
        return "before"
    if AFTER_CONDITION in condition_str:
        return "gaba_gly"
    return "unknown"


def determine_subfolder(recording_type: str) -> str:
    """Map recording type to expected subfolder on O: drive."""
    if recording_type == "before":
        return SUBFOLDER_BEFORE
    elif recording_type == "gaba_gly":
        return SUBFOLDER_GABA_GLY
    return ""


def search_for_file(
    disk_filename: str,
    date_str: str,
    recording_type: str,
) -> Optional[Path]:
    """
    Search mapped network drives for a specific file.

    Searches in: {drive}:/YYYYMMDD_gaba_gly/{subfolder}/

    Args:
        disk_filename: Filename in disk format (hyphens)
        date_str: Date string (e.g., "2025.09.04")
        recording_type: 'before' or 'gaba_gly'

    Returns:
        Full path if found, None otherwise
    """
    data_folder = find_data_folder_for_date(date_str)
    subfolder = determine_subfolder(recording_type)

    if not subfolder:
        return None

    search_dir = data_folder / subfolder
    file_path = search_dir / disk_filename

    if file_path.exists():
        return file_path

    # Fallback: search the date folder directly (in case file is at top level)
    fallback_path = data_folder / disk_filename
    if fallback_path.exists():
        return fallback_path

    return None


def discover_files_for_row(row: pd.Series) -> Dict:
    """
    For a single gsheet row, find the corresponding files on O: drive.

    Returns:
        Dictionary with discovery results
    """
    logger = logging.getLogger(__name__)

    gsheet_filename = str(row["File_name"])
    condition = str(row.get("Condition", ""))
    chip = str(row.get("Chip", ""))
    genotype = str(row.get("Genotype", ""))
    note = str(row.get("Note", ""))

    # Extract date and time
    date_str = extract_date_from_gsheet_filename(gsheet_filename)
    time_str = extract_time_from_gsheet_filename(gsheet_filename)

    # Determine recording type
    recording_type = determine_recording_type(condition)

    # Convert to disk filename
    disk_cmcr = gsheet_filename_to_disk(gsheet_filename)
    disk_cmtr = get_cmtr_from_cmcr(disk_cmcr)

    # Parse condition for playlist
    playlist_condition = parse_condition_for_playlist(condition)

    # Search for files on O: drive
    cmcr_path = search_for_file(disk_cmcr, date_str, recording_type)
    cmtr_path = search_for_file(disk_cmtr, date_str, recording_type)

    # Determine data folder
    data_folder = find_data_folder_for_date(date_str)
    subfolder = determine_subfolder(recording_type)

    result = {
        "gsheet_filename": gsheet_filename,
        "disk_cmcr": disk_cmcr,
        "disk_cmtr": disk_cmtr,
        "cmcr_path": str(cmcr_path) if cmcr_path else "",
        "cmtr_path": str(cmtr_path) if cmtr_path else "",
        "cmcr_exists": cmcr_path is not None and cmcr_path.exists(),
        "cmtr_exists": cmtr_path is not None and cmtr_path.exists(),
        "date": date_str,
        "time": time_str,
        "condition": condition,
        "playlist_condition": playlist_condition,
        "recording_type": recording_type,
        "chip": chip,
        "genotype": genotype,
        "note": note,
        "data_folder": str(data_folder / subfolder) if subfolder else str(data_folder),
        "pair_id": "",  # Filled in during pairing step
    }

    if not result["cmcr_exists"]:
        logger.warning(f"CMCR not found: {disk_cmcr} in {data_folder / subfolder}")
    if not result["cmtr_exists"]:
        logger.warning(f"CMTR not found: {disk_cmtr} in {data_folder / subfolder}")

    return result


# =============================================================================
# Temporal Pairing
# =============================================================================

def parse_time_for_sorting(time_str: str) -> int:
    """
    Convert time string to integer for sorting.

    Example: "10.11.09" -> 101109
    """
    try:
        parts = time_str.split(".")
        if len(parts) == 3:
            return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0


def assign_pair_ids(records: List[Dict]) -> List[Dict]:
    """
    Pair before/gaba_gly recordings by Chip + date with temporal validation.

    For each chip+date group:
    1. Sort all recordings by time
    2. Find before/gaba_gly pairs where the gaba_gly recording immediately
       follows the before recording in time
    3. Assign matching pair_id to both recordings in the pair

    Args:
        records: List of discovery result dictionaries

    Returns:
        Updated records with pair_id assigned
    """
    logger = logging.getLogger(__name__)

    # Group by (chip, date)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for rec in records:
        key = (rec["chip"], rec["date"])
        if key not in groups:
            groups[key] = []
        groups[key].append(rec)

    pair_counter = 0
    paired_count = 0
    unpaired_before = []
    unpaired_after = []

    for (chip, date), group_records in sorted(groups.items()):
        # Sort by time
        group_records.sort(key=lambda r: parse_time_for_sorting(r["time"]))

        # Find consecutive before -> gaba_gly pairs
        i = 0
        while i < len(group_records):
            rec = group_records[i]

            if rec["recording_type"] == "before":
                # Look for the next recording in time order
                if i + 1 < len(group_records):
                    next_rec = group_records[i + 1]
                    if next_rec["recording_type"] == "gaba_gly":
                        # Valid pair: before immediately followed by gaba_gly
                        pair_id = f"pair_{pair_counter:03d}"
                        rec["pair_id"] = pair_id
                        next_rec["pair_id"] = pair_id
                        pair_counter += 1
                        paired_count += 1

                        logger.debug(
                            f"Paired {pair_id}: Chip {chip}, {date} "
                            f"{rec['time']} (before) -> {next_rec['time']} (gaba_gly)"
                        )
                        i += 2  # Skip both records
                        continue
                    else:
                        # Next record is not gaba_gly - unpaired before
                        unpaired_before.append(
                            f"Chip {chip}, {date} {rec['time']}"
                        )
                else:
                    # No next record - unpaired before
                    unpaired_before.append(
                        f"Chip {chip}, {date} {rec['time']}"
                    )

            elif rec["recording_type"] == "gaba_gly":
                # gaba_gly without a preceding before
                unpaired_after.append(
                    f"Chip {chip}, {date} {rec['time']}"
                )

            i += 1

    logger.info(f"Pairing complete: {paired_count} pairs formed")
    if unpaired_before:
        logger.warning(f"Unpaired 'before' recordings ({len(unpaired_before)}):")
        for desc in unpaired_before[:5]:
            logger.warning(f"  {desc}")
        if len(unpaired_before) > 5:
            logger.warning(f"  ... and {len(unpaired_before) - 5} more")

    if unpaired_after:
        logger.warning(f"Unpaired 'gaba_gly' recordings ({len(unpaired_after)}):")
        for desc in unpaired_after[:5]:
            logger.warning(f"  {desc}")
        if len(unpaired_after) > 5:
            logger.warning(f"  ... and {len(unpaired_after) - 5} more")

    return records


# =============================================================================
# CSV Output
# =============================================================================

def save_file_index(records: List[Dict], output_path: Path = FILE_INDEX_CSV) -> Path:
    """
    Save discovery results to CSV.

    Args:
        records: List of discovery result dictionaries
        output_path: Path to output CSV

    Returns:
        Path to saved CSV
    """
    logger = logging.getLogger(__name__)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define column order
    columns = [
        "pair_id",
        "recording_type",
        "date",
        "time",
        "chip",
        "genotype",
        "gsheet_filename",
        "disk_cmcr",
        "disk_cmtr",
        "cmcr_path",
        "cmtr_path",
        "cmcr_exists",
        "cmtr_exists",
        "condition",
        "playlist_condition",
        "data_folder",
        "note",
    ]

    df = pd.DataFrame(records)

    # Reorder columns (keep any extras at the end)
    existing_cols = [c for c in columns if c in df.columns]
    extra_cols = [c for c in df.columns if c not in columns]
    df = df[existing_cols + extra_cols]

    # Sort by date, time
    df = df.sort_values(["date", "time"]).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved file index: {output_path} ({len(df)} rows)")

    return output_path


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary(records: List[Dict]):
    """Print summary statistics of the discovery results."""
    df = pd.DataFrame(records)

    total = len(df)
    before_count = len(df[df["recording_type"] == "before"])
    gaba_gly_count = len(df[df["recording_type"] == "gaba_gly"])
    unknown_count = len(df[df["recording_type"] == "unknown"])

    cmcr_found = df["cmcr_exists"].sum()
    cmtr_found = df["cmtr_exists"].sum()
    both_found = (df["cmcr_exists"] & df["cmtr_exists"]).sum()

    paired = len(df[df["pair_id"] != ""])
    unique_pairs = df[df["pair_id"] != ""]["pair_id"].nunique()

    unique_dates = sorted(df["date"].unique())
    unique_chips = sorted(df["chip"].unique())
    unique_genotypes = sorted(df["genotype"].unique())

    print()
    print("=" * 70)
    print("FILE DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Total recordings:     {total}")
    print(f"  Before (step):      {before_count}")
    print(f"  Gaba/gly (blocker): {gaba_gly_count}")
    print(f"  Unknown:            {unknown_count}")
    print()
    print(f"Files found on O: drive:")
    print(f"  CMCR found:         {cmcr_found}/{total}")
    print(f"  CMTR found:         {cmtr_found}/{total}")
    print(f"  Both found:         {both_found}/{total}")
    print()
    print(f"Pairing:")
    print(f"  Paired recordings:  {paired}/{total}")
    print(f"  Unique pairs:       {unique_pairs}")
    print()
    print(f"Dates ({len(unique_dates)}): {', '.join(unique_dates)}")
    print(f"Chips ({len(unique_chips)}): {', '.join(unique_chips)}")
    print(f"Genotypes ({len(unique_genotypes)}): {', '.join(unique_genotypes)}")
    print("=" * 70)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_discovery(
    gsheet_csv_path: Path = GSHEET_CSV_PATH,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
    output_path: Path = FILE_INDEX_CSV,
    do_refresh: bool = False,
) -> pd.DataFrame:
    """
    Run the full file discovery pipeline.

    Args:
        gsheet_csv_path: Path to gsheet CSV cache
        date_start: Start date for filtering
        date_end: End date for filtering
        output_path: Path to save CSV index
        do_refresh: Whether to refresh gsheet from Google Sheets API

    Returns:
        DataFrame of the CSV index
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Blocker Alignment - File Discovery")
    logger.info("=" * 60)

    # Step 1: Load gsheet
    if do_refresh:
        gsheet_df = refresh_gsheet()
    else:
        gsheet_df = load_gsheet(gsheet_csv_path)

    if gsheet_df is None:
        logger.error("Cannot proceed without gsheet data")
        return pd.DataFrame()

    # Step 2: Filter by date range
    filtered_df = filter_by_date_range(gsheet_df, date_start, date_end)

    if filtered_df.empty:
        logger.warning("No rows found in date range")
        return pd.DataFrame()

    # Step 3-6: Discover files for each row
    logger.info(f"Discovering files for {len(filtered_df)} recordings...")
    records = []
    for _, row in filtered_df.iterrows():
        result = discover_files_for_row(row)
        records.append(result)

    # Step 7: Pair before/gaba_gly recordings with temporal validation
    logger.info("Pairing before/gaba_gly recordings...")
    records = assign_pair_ids(records)

    # Step 8: Save CSV index
    output_csv = save_file_index(records, output_path)

    # Print summary
    print_summary(records)

    return pd.DataFrame(records)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover and index CMCR/CMTR files for blocker alignment analysis"
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
        "--refresh-gsheet",
        action="store_true",
        help="Re-download gsheet from Google Sheets API before discovery",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FILE_INDEX_CSV,
        help=f"Output CSV path (default: {FILE_INDEX_CSV})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    print("=" * 70)
    print("Blocker Alignment Analysis - File Discovery")
    print("=" * 70)
    print(f"Experiment:  {ACTIVE_EXPERIMENT}")
    print(f"Date range:  {DATE_START} to {DATE_END}")
    print(f"Gsheet CSV:  {GSHEET_CSV_PATH}")
    print(f"Drives:      {', '.join(str(d) for d in DATA_DRIVES)}")
    print(f"Output CSV:  {args.output}")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    result_df = run_discovery(
        output_path=args.output,
        do_refresh=args.refresh_gsheet,
    )

    if result_df.empty:
        print("\nNo files discovered. Check logs for errors.")
        sys.exit(1)

    print(f"\nCSV index saved to: {args.output}")
    print(f"Total rows: {len(result_df)}")


if __name__ == "__main__":
    main()
