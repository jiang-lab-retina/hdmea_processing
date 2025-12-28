"""
Step Wrapper: Load Google Sheet Metadata

Loads metadata from Google Sheet and adds to session.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from hdmea.pipeline import PipelineSession

# Add project path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "Projects/load_gsheet"))

# Import color utilities
from Projects.unified_pipeline.config import red_warning

logger = logging.getLogger(__name__)

STEP_NAME = "add_gsheet_metadata"


def add_gsheet_step(
    *,
    sheet_name: str = None,  # Will use config default
    credentials_path: Optional[Path] = None,
    csv_cache_path: Optional[Path] = None,
    gsheet_df: Optional[pd.DataFrame] = None,
    session: PipelineSession,
) -> PipelineSession:
    """
    Load Google Sheet metadata and add to session.
    
    This is Step 8 of the pipeline.
    
    If Google Sheet is unavailable (network error, missing credentials),
    logs a RED warning and continues without the metadata.
    
    Args:
        sheet_name: Name of the Google Sheet
        credentials_path: Path to credentials JSON file
        csv_cache_path: Path to CSV cache file
        gsheet_df: Pre-loaded gsheet DataFrame (optional, loads if not provided)
        session: Pipeline session (required)
    
    Returns:
        Updated session with gsheet metadata (or unchanged if unavailable)
    """
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Step 8: Loading Google Sheet metadata...")
    
    try:
        from load_gsheet import import_gsheet_v2, find_gsheet_row, hdf5_to_gsheet_filename
        from Projects.unified_pipeline.config import GSHEET_CREDENTIALS_PATH, GSHEET_CSV_CACHE_PATH, GSHEET_NAME
        
        # Use defaults if not provided
        if sheet_name is None:
            sheet_name = GSHEET_NAME
        if credentials_path is None:
            credentials_path = GSHEET_CREDENTIALS_PATH
        if csv_cache_path is None:
            csv_cache_path = GSHEET_CSV_CACHE_PATH
        
        # Validate credentials file exists
        if not credentials_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
        
        # Load the gsheet if not provided
        if gsheet_df is None:
            try:
                logger.info(f"  Loading Google Sheet '{sheet_name}'...")
                gsheet_df = import_gsheet_v2(
                    sheet_name=sheet_name,
                    cred=str(credentials_path),
                    csv_path=str(csv_cache_path),
                )
                logger.info(f"  Loaded {len(gsheet_df)} rows from Google Sheet")
            except Exception as gsheet_error:
                # Fallback to local CSV cache
                local_csv_path = project_root / "Projects/load_gsheet/gsheet_table.csv"
                if local_csv_path.exists():
                    logger.warning(f"  Google Sheet failed ({type(gsheet_error).__name__}), falling back to local CSV")
                    gsheet_df = pd.read_csv(local_csv_path)
                    logger.info(f"  Loaded {len(gsheet_df)} rows from local CSV: {local_csv_path}")
                else:
                    # Re-raise if no fallback available
                    raise
        
        # Find matching row for this dataset
        gsheet_row = None
        
        # Try to match by hdf5_path first
        if session.hdf5_path is not None and session.hdf5_path.exists():
            gsheet_row = find_gsheet_row(gsheet_df, str(session.hdf5_path))
        
        # If no match, try to match by dataset_id
        if gsheet_row is None:
            logger.info(f"  Searching for dataset_id: {session.dataset_id}")
            
            # Normalize dataset_id for matching (handle dash vs dot format)
            # e.g., "2024.08.08-10.40.20-Rec" -> "2024.08.08.10.40.20"
            normalized_id = session.dataset_id.replace('-', '.')
            # Also try without the -Rec suffix
            base_id = session.dataset_id.replace('-Rec', '').replace('-', '.')
            
            # Look for partial match in File_name column
            for _, row in gsheet_df.iterrows():
                file_name = str(row.get('File_name', ''))
                # Try multiple matching strategies
                if (session.dataset_id in file_name or 
                    normalized_id in file_name or
                    base_id in file_name):
                    gsheet_row = row
                    logger.info(f"  Found match by dataset_id: {file_name}")
                    break
        
        if gsheet_row is None:
            logger.warning(red_warning(f"  No matching gsheet row found for {session.dataset_id}"))
            session.warnings.append(f"{STEP_NAME}: No matching row found")
            session.completed_steps.add(f"{STEP_NAME}:skipped")
            return session
        
        # Convert row to dict and store in metadata
        gsheet_dict = {}
        for col_name in gsheet_row.index:
            value = gsheet_row[col_name]
            # Sanitize key for HDF5 compatibility
            safe_key = str(col_name).replace('/', '_').replace('\\', '_')
            # Handle NaN values
            if pd.isna(value):
                gsheet_dict[safe_key] = ""
            else:
                gsheet_dict[safe_key] = value
        
        session.metadata['gsheet_row'] = gsheet_dict
        session.mark_step_complete(STEP_NAME)
        logger.info(f"  Added {len(gsheet_dict)} gsheet fields")
        
    except ImportError as e:
        logger.warning(red_warning(f"  Cannot import gsheet module: {e}"))
        session.warnings.append(f"{STEP_NAME}: Import error - {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except FileNotFoundError as e:
        logger.warning(red_warning(f"  {e}"))
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except Exception as e:
        # Log the full exception for debugging
        import traceback
        logger.debug(f"  Full exception: {traceback.format_exc()}")
        logger.warning(red_warning(f"  Google Sheet unavailable: {type(e).__name__}: {e}"))
        session.warnings.append(f"{STEP_NAME}: {type(e).__name__}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    return session
