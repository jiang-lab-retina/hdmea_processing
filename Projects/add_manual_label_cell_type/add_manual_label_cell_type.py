"""
Add Manual Label Cell Type to HDF5 Files

This module reads manual cell type labels from an image folder structure
and saves them to HDF5 files following the session-based pipeline workflow.

Manual label folder structure:
    manual_label_data/
    └── {dataset_id}/
        ├── RGC/           # Retinal ganglion cells
        ├── AC/            # Amacrine cells
        ├── Other/         # Other cell types
        └── Unknown/       # Unclassified

Labels are saved to: units/{unit_id}/auto_label/axon_type

Workflow:
    1. Load existing HDF5 into session via load_hdf5_to_session()
    2. Parse manual labels and add to session
    3. Save results to HDF5 via save_cell_type_to_hdf5()
"""

from pathlib import Path
from typing import Dict, Optional, Union
import shutil
import re
import logging

import h5py

from hdmea.pipeline import create_session, PipelineSession

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Paths
MANUAL_LABEL_FOLDER = Path(r"M:\Python_Project\Data_Processing_2024\manual_label_data")
INPUT_HDF5 = Path(r"M:/Python_Project/Data_Processing_2027/Projects/load_gsheet/export_gsheet_20251225/2024.09.04-15.22.08-Rec.h5")
OUTPUT_DIR = Path(__file__).parent / "export"

# Feature settings
FEATURE_NAME = "axon_type"
AUTO_LABEL_GROUP = "auto_label"

# Valid cell types
CELL_TYPES = ["RGC", "AC", "Other", "Unknown"]


# =============================================================================
# Session Loading
# =============================================================================

def load_hdf5_to_session(hdf5_path: Path, dataset_id: str = None) -> PipelineSession:
    """
    Load an existing HDF5 file into a PipelineSession.
    
    Loads unit IDs from the HDF5 file for cell type labeling.
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Optional dataset ID (uses filename stem if not provided)
        
    Returns:
        PipelineSession with units loaded
    """
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path  # Track source file for later saving
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'units' not in f:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        unit_ids = list(f['units'].keys())
        logger.info(f"Found {len(unit_ids)} units in HDF5 file")
        
        for unit_id in unit_ids:
            # Initialize unit with empty auto_label structure
            session.units[unit_id] = {
                AUTO_LABEL_GROUP: {}
            }
    
    session.mark_step_complete('load_hdf5')
    print(f"  Loaded {len(session.units)} units from {hdf5_path.name}")
    return session


# =============================================================================
# Pipeline Step: Parse Manual Labels
# =============================================================================

def parse_manual_labels(
    label_folder: Path,
    dataset_id: str,
    *,
    session: Optional[PipelineSession] = None,
) -> Union[Dict[int, str], PipelineSession]:
    """
    Parse manual labels from image folder structure.
    
    Args:
        label_folder: Root folder containing manual label subfolders
        dataset_id: Dataset identifier (e.g., "2024.09.04-15.22.08-Rec")
        session: If provided, operates in deferred mode and returns session.
                 If None, operates in immediate mode and returns labels dict.
    
    Returns:
        If session provided: Updated PipelineSession with labels stored in metadata
        If session is None: Dict mapping unit number (int) to cell type (str)
    """
    dataset_folder = label_folder / dataset_id
    
    if not dataset_folder.exists():
        print(f"  Warning: Manual label folder not found: {dataset_folder}")
        labels = {}
    else:
        labels = {}
        
        for cell_type in CELL_TYPES:
            cell_type_folder = dataset_folder / cell_type
            
            if not cell_type_folder.exists():
                continue
            
            # Find all PNG files in this folder (including nested)
            png_files = list(cell_type_folder.rglob("*.png"))
            
            for png_file in png_files:
                # Extract unit number from filename like "2024.09.04-15.22.08-Rec_unit10.png"
                match = re.search(r'_unit(\d+)\.png$', png_file.name, re.IGNORECASE)
                if match:
                    unit_num = int(match.group(1))
                    labels[unit_num] = cell_type.lower()
        
        print(f"  Found {len(labels)} manual labels")
    
    if session is not None:
        # Deferred mode: store labels in session metadata
        session.add_metadata({"_manual_labels": labels})
        session.mark_step_complete("parse_manual_labels")
        return session
    else:
        # Immediate mode: return labels dict directly
        return labels


# =============================================================================
# Pipeline Step: Add Cell Type Labels
# =============================================================================

def add_cell_type_labels(
    labels_dict: Optional[Dict[int, str]] = None,
    *,
    session: Optional[PipelineSession] = None,
) -> PipelineSession:
    """
    Match units with manual labels and store in session.
    
    Args:
        labels_dict: Dict mapping unit number to cell type.
                     If None, uses labels from session metadata (set by parse_manual_labels).
        session: Session with loaded units
    
    Returns:
        Updated session with cell type labels added to each unit
    """
    if session is None:
        raise ValueError("Session is required for add_cell_type_labels")
    
    # Get labels from session metadata if not provided
    if labels_dict is None:
        labels_dict = session.metadata.pop("_manual_labels", {})
    
    labeled_count = 0
    no_label_count = 0
    
    for unit_id in session.units.keys():
        # Extract unit number from unit_id (e.g., "unit_001" -> 1)
        match = re.search(r'unit_(\d+)', unit_id)
        if match:
            unit_num = int(match.group(1))
            
            if unit_num in labels_dict:
                cell_type = labels_dict[unit_num]
                labeled_count += 1
            else:
                cell_type = "no_label"
                no_label_count += 1
        else:
            cell_type = "no_label"
            no_label_count += 1
        
        # Add auto_label group with axon_type to unit
        if AUTO_LABEL_GROUP not in session.units[unit_id]:
            session.units[unit_id][AUTO_LABEL_GROUP] = {}
        
        session.units[unit_id][AUTO_LABEL_GROUP][FEATURE_NAME] = cell_type
    
    print(f"  Labeled: {labeled_count}, No label: {no_label_count}")
    
    session.mark_step_complete("add_cell_type_labels")
    return session


# =============================================================================
# HDF5 Saving
# =============================================================================

def save_cell_type_to_hdf5(session: PipelineSession, output_path: Path = None) -> Path:
    """
    Save cell type labels to HDF5 file.
    
    Copies the source HDF5 file to the output path and adds auto_label/axon_type
    to each unit, preserving all existing data.
    
    Args:
        session: PipelineSession with cell type labels
        output_path: Path to save to (uses export/{dataset_id}.h5 if not provided)
        
    Returns:
        Path to saved file
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{session.dataset_id}.h5"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy source file to preserve all existing data
    source_path = session.hdf5_path
    if source_path and source_path.exists():
        logger.info(f"Copying source HDF5 to: {output_path}")
        shutil.copy2(source_path, output_path)
    else:
        raise ValueError(f"Source HDF5 file not found: {source_path}")
    
    logger.info(f"Adding cell type labels to: {output_path}")
    
    # Open in append mode to preserve existing data
    with h5py.File(output_path, 'a') as f:
        saved_count = 0
        
        for unit_id, unit_data in session.units.items():
            # Get cell type from session
            cell_type = unit_data.get(AUTO_LABEL_GROUP, {}).get(FEATURE_NAME)
            
            if cell_type is None:
                continue
            
            # Check if unit group exists
            unit_path = f'units/{unit_id}'
            if unit_path not in f:
                logger.warning(f"Unit {unit_id} not found in HDF5, skipping")
                continue
            
            # Create or get auto_label group
            auto_label_path = f'{unit_path}/{AUTO_LABEL_GROUP}'
            if auto_label_path not in f:
                f.create_group(auto_label_path)
            
            # Remove existing axon_type dataset if present (to update)
            axon_type_path = f'{auto_label_path}/{FEATURE_NAME}'
            if axon_type_path in f:
                del f[axon_type_path]
            
            # Save cell type as string dataset (not attribute)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(axon_type_path, data=cell_type, dtype=dt)
            
            saved_count += 1
            logger.debug(f"Saved cell type for {unit_id}: {cell_type}")
    
    session.mark_step_complete('save_to_hdf5')
    print(f"  Saved cell type labels for {saved_count} units to: {output_path}")
    
    return output_path


# =============================================================================
# Single File Processing (Session Workflow)
# =============================================================================

def process_single_file(
    input_path: Path,
    output_path: Path,
    label_folder: Path = MANUAL_LABEL_FOLDER,
) -> bool:
    """
    Process a single HDF5 file using session-based workflow.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        label_folder: Root folder containing manual label subfolders
    
    Returns:
        True if successful, False otherwise
    """
    try:
        dataset_id = input_path.stem
        print(f"\nProcessing: {dataset_id}")
        
        # =================================================================
        # Session-based workflow
        # =================================================================
        
        # Step 1: Load existing HDF5 into session
        print("  Step 1: Loading HDF5 into session...")
        session = load_hdf5_to_session(input_path)
        
        # Step 2: Parse manual labels (chained to session)
        print("  Step 2: Parsing manual labels...")
        session = parse_manual_labels(label_folder, dataset_id, session=session)
        
        # Step 3: Add cell type labels to each unit
        print("  Step 3: Adding cell type labels...")
        session = add_cell_type_labels(session=session)
        
        # Step 4: Save to export HDF5
        print("  Step 4: Saving to HDF5...")
        save_cell_type_to_hdf5(session, output_path)
        
        # =================================================================
        # Summary
        # =================================================================
        label_counts = {}
        for unit_id, unit_data in session.units.items():
            auto_label = unit_data.get(AUTO_LABEL_GROUP, {})
            ct = auto_label.get(FEATURE_NAME, "unknown")
            label_counts[ct] = label_counts.get(ct, 0) + 1
        
        print("  Label distribution:")
        for ct, count in sorted(label_counts.items()):
            print(f"    {ct}: {count}")
        
        print(f"  Completed steps: {session.completed_steps}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Add Manual Label Cell Type (Session-Based Workflow)")
    print("=" * 70)
    print(f"Input:  {INPUT_HDF5}")
    print(f"Labels: {MANUAL_LABEL_FOLDER}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Process test file
    output_path = OUTPUT_DIR / INPUT_HDF5.name
    
    success = process_single_file(
        input_path=INPUT_HDF5,
        output_path=output_path,
        label_folder=MANUAL_LABEL_FOLDER,
    )
    
    if success:
        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)
        
        # Verify the output
        print("\nVerifying output...")
        with h5py.File(output_path, 'r') as f:
            # Check a few units
            units = list(f['units'].keys())[:5]
            for unit_id in units:
                path = f'units/{unit_id}/auto_label/axon_type'
                if path in f:
                    value = f[path][()]
                    # Handle both string array and scalar
                    if isinstance(value, (bytes, str)):
                        value = value.decode('utf-8') if isinstance(value, bytes) else value
                    elif hasattr(value, '__iter__'):
                        value = value[0].decode('utf-8') if isinstance(value[0], bytes) else value[0]
                    print(f"  {unit_id}: {value}")
                else:
                    print(f"  {unit_id}: NO LABEL FOUND")
    else:
        print("\n" + "=" * 70)
        print("FAILED")
        print("=" * 70)
