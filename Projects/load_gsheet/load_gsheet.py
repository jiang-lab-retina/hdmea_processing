"""
Google Sheet to HDF5 Metadata Loader

Imports MEA experiment metadata from Google Sheets and integrates it with
the HDF5 session workflow. This module provides standalone functionality
without depending on the jianglab package.

Workflow:
1. Import Google Sheets data using service account credentials
2. Match HDF5 filenames to gsheet rows
3. Add gsheet metadata to HDF5 files via session workflow

Usage:
    python load_gsheet.py

Author: Generated for experimental analysis
Date: 2024-12
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import gspread
import h5py
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Credentials path (relative to project root)
CREDENTIALS_PATH = Path(__file__).parent.parent.parent / "credentials" / "vibrant-epsilon-169702-467fddc26dfc.json"

# Google Sheet settings
SHEET_NAME = "MEA dashboard"
TAB_NAME_PREFIX = "Data_"

# Test file paths
TEST_INPUT_PATH = Path(__file__).parent.parent / "sta_quantification" / "eimage_sta_output_20251225" / "2024.03.01-14.40.14-Rec.h5"
EXPORT_DIR = Path(__file__).parent / "export"

# CSV cache path
CSV_CACHE_PATH = Path(__file__).parent / "gsheet_table.csv"


# =============================================================================
# Google Sheets Import (ported from jianglab.import_gsheet_v2)
# =============================================================================

def import_gsheet_v2(
    sheet_name: str = SHEET_NAME,
    cred: str = None,
    csv_path: str = None,
    tab_name_prefix: str = TAB_NAME_PREFIX,
) -> pd.DataFrame:
    """
    Import data from Google Sheets combining all tabs with specified prefix.
    
    Connects to Google Sheets API using service account credentials,
    reads all worksheets starting with tab_name_prefix, combines them
    into a single DataFrame, and caches to CSV.
    
    Args:
        sheet_name: Name of the Google Sheet document
        cred: Path to service account JSON credentials file
        csv_path: Path to save CSV cache (optional)
        tab_name_prefix: Only import worksheets starting with this prefix
        
    Returns:
        Combined DataFrame with all data from matching worksheets
        
    Example:
        >>> df = import_gsheet_v2(
        ...     sheet_name="MEA dashboard",
        ...     cred="credentials/service_account.json",
        ...     csv_path="cache/gsheet_table.csv"
        ... )
    """
    if cred is None:
        cred = str(CREDENTIALS_PATH)
    if csv_path is None:
        csv_path = str(CSV_CACHE_PATH)
    
    # Validate credentials file exists
    if not os.path.exists(cred):
        raise FileNotFoundError(f"Credentials file not found: {cred}")
    
    logger.info(f"Connecting to Google Sheets: {sheet_name}")
    
    # Define OAuth2 scope
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    
    # Authenticate with service account
    creds_obj = ServiceAccountCredentials.from_json_keyfile_name(cred, scope)
    client = gspread.authorize(creds_obj)
    
    # Open the spreadsheet
    sheet = client.open(sheet_name)
    
    all_data = []
    
    # Iterate through worksheets and collect data
    for worksheet in sheet.worksheets():
        if worksheet.title.startswith(tab_name_prefix):
            logger.info(f"  Reading worksheet: {worksheet.title}")
            records = worksheet.get_all_records()
            df = pd.DataFrame.from_dict(records)
            df['Sheet Name'] = worksheet.title  # Add column to identify source sheet
            all_data.append(df)
    
    if not all_data:
        logger.warning(f"No worksheets found with prefix '{tab_name_prefix}'")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Cache to CSV
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Cached {len(combined_df)} rows to: {csv_path}")
    
    return combined_df


# =============================================================================
# Filename Matching
# =============================================================================

def hdf5_to_gsheet_filename(hdf5_path: str) -> str:
    """
    Convert HDF5 filename to gsheet File_name format.
    
    Transforms filename pattern from HDF5 format to the format used in
    Google Sheets' File_name column.
    
    Args:
        hdf5_path: Path to HDF5 file (e.g., "2024.03.01-14.40.14-Rec.h5")
        
    Returns:
        Gsheet filename format (e.g., "2024.03.01.14.40.14.Rec.cmcr")
        
    Example:
        >>> hdf5_to_gsheet_filename("path/to/2024.03.01-14.40.14-Rec.h5")
        '2024.03.01.14.40.14.Rec.cmcr'
    """
    # Extract just the filename
    filename = Path(hdf5_path).name
    
    # Remove .h5 suffix only (keep -Rec as .Rec)
    if filename.endswith(".h5"):
        filename = filename[:-3]  # Remove ".h5"
    
    # Replace hyphens with dots (this turns "-Rec" into ".Rec")
    gsheet_name = filename.replace("-", ".")
    
    # Add .cmcr suffix
    gsheet_name = gsheet_name + ".cmcr"
    
    return gsheet_name


def find_gsheet_row(
    gsheet_df: pd.DataFrame,
    hdf5_path: str,
    filename_column: str = "File_name",
) -> Optional[pd.Series]:
    """
    Find the matching gsheet row for an HDF5 file.
    
    Args:
        gsheet_df: DataFrame from import_gsheet_v2()
        hdf5_path: Path to HDF5 file
        filename_column: Column name containing filenames in gsheet
        
    Returns:
        Matching row as Series, or None if not found
    """
    gsheet_filename = hdf5_to_gsheet_filename(hdf5_path)
    
    # Find matching row
    matching_rows = gsheet_df[gsheet_df[filename_column] == gsheet_filename]
    
    if matching_rows.empty:
        logger.warning(f"No gsheet row found for: {hdf5_path}")
        logger.warning(f"  Searched for: {gsheet_filename}")
        return None
    
    if len(matching_rows) > 1:
        logger.warning(f"Multiple rows found for {gsheet_filename}, using first")
    
    return matching_rows.iloc[0]


# =============================================================================
# Session Workflow (simplified, without PipelineSession dependency)
# =============================================================================

def load_hdf5_metadata(hdf5_path: Path) -> Dict[str, Any]:
    """
    Load basic metadata from an HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        Dict with dataset_id and existing metadata
    """
    result = {
        'dataset_id': hdf5_path.stem,
        'hdf5_path': hdf5_path,
        'metadata': {},
    }
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get dataset_id from attributes if available
        if 'dataset_id' in f.attrs:
            result['dataset_id'] = f.attrs['dataset_id']
        
        # Load existing metadata
        if 'metadata' in f:
            for key in f['metadata'].keys():
                try:
                    item = f[f'metadata/{key}']
                    if isinstance(item, h5py.Dataset):
                        result['metadata'][key] = item[()]
                    elif isinstance(item, h5py.Group):
                        result['metadata'][key] = _read_group_to_dict(item)
                except Exception as e:
                    logger.warning(f"Could not read metadata/{key}: {e}")
    
    return result


def _read_group_to_dict(group: h5py.Group) -> Dict[str, Any]:
    """Recursively read HDF5 group to dict."""
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = _read_group_to_dict(item)
    return result


def add_gsheet_to_hdf5(
    input_hdf5: Path,
    output_hdf5: Path,
    gsheet_row: pd.Series,
) -> Path:
    """
    Copy HDF5 file and add gsheet metadata.
    
    Copies the source HDF5 to output path and adds the gsheet row
    data under metadata/gsheet_row/.
    
    Args:
        input_hdf5: Path to source HDF5 file
        output_hdf5: Path to output HDF5 file
        gsheet_row: Series containing gsheet data for this file
        
    Returns:
        Path to output file
    """
    # Create output directory
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy source file to preserve all existing data
    logger.info(f"Copying HDF5 to: {output_hdf5}")
    shutil.copy2(input_hdf5, output_hdf5)
    
    # Add gsheet metadata
    logger.info("Adding gsheet metadata...")
    
    with h5py.File(output_hdf5, 'a') as f:
        # Ensure metadata group exists
        if 'metadata' not in f:
            f.create_group('metadata')
        
        # Remove existing gsheet_row if present
        gsheet_path = 'metadata/gsheet_row'
        if gsheet_path in f:
            del f[gsheet_path]
        
        # Create gsheet_row group
        gsheet_group = f.create_group(gsheet_path)
        
        # Write each column as a dataset
        for col_name, value in gsheet_row.items():
            try:
                _write_value_to_hdf5(gsheet_group, col_name, value)
            except Exception as e:
                logger.warning(f"Could not write column '{col_name}': {e}")
    
    logger.info(f"Added {len(gsheet_row)} metadata fields")
    return output_hdf5


def _write_value_to_hdf5(group: h5py.Group, key: str, value: Any) -> None:
    """Write a single value to HDF5 group as dataset."""
    # Sanitize key (HDF5 doesn't allow certain characters)
    safe_key = str(key).replace('/', '_').replace('\\', '_')
    
    if value is None:
        return
    
    if isinstance(value, str):
        # Store strings as variable-length string datasets
        dt = h5py.string_dtype(encoding='utf-8')
        group.create_dataset(safe_key, data=[value], dtype=dt)
    elif isinstance(value, (int, float)):
        # Store scalars as 1-element arrays
        group.create_dataset(safe_key, data=[value])
    elif isinstance(value, (list, tuple)):
        group.create_dataset(safe_key, data=value)
    else:
        # Convert to string as fallback
        dt = h5py.string_dtype(encoding='utf-8')
        group.create_dataset(safe_key, data=[str(value)], dtype=dt)


# =============================================================================
# Main Workflow
# =============================================================================

def load_gsheet_to_hdf5(
    input_hdf5: Path,
    output_hdf5: Path = None,
    gsheet_df: pd.DataFrame = None,
    cred: str = None,
) -> Optional[Path]:
    """
    Load gsheet metadata into an HDF5 file.
    
    Main entry point that combines all steps:
    1. Import gsheet data (if not provided)
    2. Match HDF5 filename to gsheet row
    3. Add gsheet metadata to output HDF5
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_hdf5: Path to output HDF5 (default: export/{input_name})
        gsheet_df: Pre-loaded gsheet DataFrame (imports if None)
        cred: Path to credentials file
        
    Returns:
        Path to output file, or None if no matching gsheet row found
    """
    input_hdf5 = Path(input_hdf5)
    
    if output_hdf5 is None:
        output_hdf5 = EXPORT_DIR / input_hdf5.name
    else:
        output_hdf5 = Path(output_hdf5)
    
    # Import gsheet if not provided
    if gsheet_df is None:
        gsheet_df = import_gsheet_v2(cred=cred)
    
    # Find matching row
    gsheet_row = find_gsheet_row(gsheet_df, str(input_hdf5))
    
    if gsheet_row is None:
        logger.error(f"Cannot add gsheet metadata: no matching row for {input_hdf5.name}")
        return None
    
    logger.info(f"Found gsheet row for: {input_hdf5.name}")
    
    # Add gsheet to HDF5
    result_path = add_gsheet_to_hdf5(input_hdf5, output_hdf5, gsheet_row)
    
    return result_path


def main():
    """
    Main workflow for testing gsheet to HDF5 integration.
    """
    print("=" * 70)
    print("Google Sheet to HDF5 Metadata Loader")
    print("=" * 70)
    print(f"Input file:  {TEST_INPUT_PATH}")
    print(f"Export dir:  {EXPORT_DIR}")
    print(f"Credentials: {CREDENTIALS_PATH}")
    print(f"Sheet name:  {SHEET_NAME}")
    
    # Validate input file
    if not TEST_INPUT_PATH.exists():
        print(f"\nError: Input file not found: {TEST_INPUT_PATH}")
        return
    
    # Validate credentials
    if not CREDENTIALS_PATH.exists():
        print(f"\nError: Credentials not found: {CREDENTIALS_PATH}")
        return
    
    # =========================================================================
    # Step 1: Import Google Sheet
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Importing Google Sheet")
    print("-" * 70)
    
    gsheet_df = import_gsheet_v2(
        sheet_name=SHEET_NAME,
        cred=str(CREDENTIALS_PATH),
        csv_path=str(CSV_CACHE_PATH),
    )
    
    print(f"Imported {len(gsheet_df)} rows")
    print(f"Columns: {list(gsheet_df.columns)[:10]}...")  # Show first 10 columns
    
    # =========================================================================
    # Step 2: Find matching gsheet row
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Finding matching gsheet row")
    print("-" * 70)
    
    gsheet_filename = hdf5_to_gsheet_filename(str(TEST_INPUT_PATH))
    print(f"HDF5 filename: {TEST_INPUT_PATH.name}")
    print(f"Gsheet format: {gsheet_filename}")
    
    gsheet_row = find_gsheet_row(gsheet_df, str(TEST_INPUT_PATH))
    
    if gsheet_row is not None:
        print(f"\nMatched row:")
        # Show first few columns
        for col in list(gsheet_row.index)[:8]:
            print(f"  {col}: {gsheet_row[col]}")
        print(f"  ... ({len(gsheet_row)} total columns)")
    else:
        print("\nNo matching row found!")
        return
    
    # =========================================================================
    # Step 3: Add gsheet to HDF5
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Adding gsheet metadata to HDF5")
    print("-" * 70)
    
    output_path = EXPORT_DIR / TEST_INPUT_PATH.name
    result_path = add_gsheet_to_hdf5(TEST_INPUT_PATH, output_path, gsheet_row)
    
    # =========================================================================
    # Step 4: Verify saved data
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Verifying saved data")
    print("-" * 70)
    
    with h5py.File(result_path, 'r') as f:
        print(f"HDF5 file: {result_path}")
        
        # Check gsheet_row group
        if 'metadata/gsheet_row' in f:
            gsheet_group = f['metadata/gsheet_row']
            print(f"\nmetadata/gsheet_row contents ({len(gsheet_group.keys())} items):")
            for key in list(gsheet_group.keys())[:10]:
                value = gsheet_group[key][()]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                elif hasattr(value, '__len__') and len(value) == 1:
                    value = value[0]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                print(f"  {key}: {value}")
            if len(gsheet_group.keys()) > 10:
                print(f"  ... and {len(gsheet_group.keys()) - 10} more")
        else:
            print("Warning: metadata/gsheet_row not found in output file!")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Input:  {TEST_INPUT_PATH}")
    print(f"Output: {result_path}")
    print(f"\nGsheet metadata successfully added to HDF5 file.")


if __name__ == "__main__":
    main()

