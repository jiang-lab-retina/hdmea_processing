"""
Step Wrapper: Add Manual Cell Type Labels

Loads manual cell type labels from image folder structure.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

from hdmea.pipeline import PipelineSession
from Projects.unified_pipeline.config import MANUAL_LABEL_FOLDER, progress_bar

logger = logging.getLogger(__name__)

STEP_NAME = "add_cell_type_labels"
AUTO_LABEL_GROUP = "auto_label"
FEATURE_NAME = "axon_type"
CELL_TYPES = ["RGC", "AC", "Other", "Unknown"]


def add_cell_type_step(
    *,
    label_folder: Optional[Path] = None,
    session: PipelineSession,
) -> PipelineSession:
    """
    Add manual cell type labels to units.
    
    This is Step 9 of the pipeline.
    
    Reads labels from folder structure:
        label_folder/{dataset_id}/RGC/*.png
        label_folder/{dataset_id}/AC/*.png
        etc.
    
    Args:
        label_folder: Root folder containing manual label subfolders
        session: Pipeline session (required)
    
    Returns:
        Updated session with cell type labels
    """
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Step 9: Adding manual cell type labels...")
    
    if label_folder is None:
        label_folder = MANUAL_LABEL_FOLDER
    
    # Parse labels from folder structure
    labels = _parse_manual_labels(label_folder, session.dataset_id)
    
    if not labels:
        logger.warning(f"  No manual labels found for {session.dataset_id}")
        session.warnings.append(f"{STEP_NAME}: No labels found")
    
    # Apply labels to units
    labeled_count = 0
    no_label_count = 0
    
    for unit_id in session.units.keys():
        # Extract unit number from unit_id (e.g., "unit_001" -> 1)
        match = re.search(r'unit_(\d+)', unit_id)
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
    
    logger.info(f"  Labeled: {labeled_count}, No label: {no_label_count}")
    session.mark_step_complete(STEP_NAME)
    
    return session


def _parse_manual_labels(label_folder: Path, dataset_id: str) -> Dict[int, str]:
    """
    Parse manual labels from folder structure.
    
    Args:
        label_folder: Root folder containing labels
        dataset_id: Dataset identifier
    
    Returns:
        Dict mapping unit number (int) to cell type (str)
    """
    dataset_folder = label_folder / dataset_id
    
    if not dataset_folder.exists():
        logger.debug(f"  Label folder not found: {dataset_folder}")
        return {}
    
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
    
    return labels

