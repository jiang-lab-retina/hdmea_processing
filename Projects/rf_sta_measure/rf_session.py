"""
RF-STA Receptive Field Measurement - Session Workflow

Performs RF geometry measurements using a session-based workflow,
saving structured results to HDF5 without generating plots.

Workflow:
1. Load existing HDF5 data into a session
2. Extract RF geometry for each unit (Gaussian, DoG, ON/OFF fits)
3. Save results to export HDF5 file with structured groups

Output structure:
units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/
    ├── center_row, center_col, size_x, size_y, area, equivalent_diameter, peak_frame
    ├── gaussian_fit/ (center_x, center_y, sigma_x, sigma_y, amplitude, theta, offset, r_squared)
    ├── DoG/ (center_x, center_y, sigma_exc, sigma_inh, amp_exc, amp_inh, offset, r_squared)
    └── ONOFF_model/ (on_*, off_* parameters)

Author: Generated for experimental analysis
Date: 2024-12
"""

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

# Import session utilities
from hdmea.pipeline import create_session, PipelineSession

# Import RF geometry extraction functions from rf_sta_measure
from rf_sta_measure import (
    extract_rf_geometry,
    RFGeometry,
    GaussianFit,
    DoGFit,
    OnOffFit,
    STA_FEATURE_NAME,
    STA_DATA_PATH,
    FRAME_RANGE,
    THRESHOLD_FRACTION,
)

# Enable logging
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Input HDF5 file (same test file as rf_sta_measure.py)
HDF5_PATH = Path(__file__).parent.parent / "sta_quantification" / "eimage_sta_output_20251225" / "2024.03.01-14.40.14-Rec.h5"
# Output directory for exported HDF5
EXPORT_DIR = Path(__file__).parent / "export"

# Geometry output path within each unit (under the STA feature)
GEOMETRY_OUTPUT_PATH = f"features/{STA_FEATURE_NAME}/sta_geometry"


# =============================================================================
# Session Loading
# =============================================================================

def load_hdf5_to_session(hdf5_path: Path, dataset_id: str = None) -> PipelineSession:
    """
    Load an existing HDF5 file into a PipelineSession.
    
    Loads STA data from each unit for RF geometry extraction.
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Optional dataset ID (uses filename stem if not provided)
        
    Returns:
        PipelineSession with STA data loaded
    """
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path  # Track source file
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load units with their STA data
        if 'units' not in f:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        unit_ids = list(f['units'].keys())
        logger.info(f"Found {len(unit_ids)} units in HDF5 file")
        
        for unit_id in unit_ids:
            unit_group = f[f'units/{unit_id}']
            unit_data = {}
            
            # Load STA data if available
            if STA_DATA_PATH in unit_group:
                if 'features' not in unit_data:
                    unit_data['features'] = {}
                if STA_FEATURE_NAME not in unit_data['features']:
                    unit_data['features'][STA_FEATURE_NAME] = {}
                unit_data['features'][STA_FEATURE_NAME]['data'] = unit_group[STA_DATA_PATH][:]
                logger.info(f"Loaded {unit_id}: shape={unit_data['features'][STA_FEATURE_NAME]['data'].shape}")
            else:
                logger.warning(f"No STA data found for {unit_id} at {STA_DATA_PATH}")
            
            if unit_data:
                session.units[unit_id] = unit_data
        
        # Load metadata if available
        if 'metadata' in f:
            for key in f['metadata'].keys():
                try:
                    session.metadata[key] = f[f'metadata/{key}'][()]
                except Exception:
                    pass
    
    session.completed_steps.add('load_hdf5')
    return session


# =============================================================================
# RF Geometry Extraction (Session Mode)
# =============================================================================

def extract_rf_geometry_session(
    session: PipelineSession,
    frame_range: tuple = FRAME_RANGE,
    threshold_fraction: float = THRESHOLD_FRACTION,
) -> PipelineSession:
    """
    Extract RF geometry for all units in the session.
    
    Processes each unit's STA data and stores geometry results in the session.
    
    Args:
        session: PipelineSession with loaded STA data
        frame_range: Range of frames to use for analysis
        threshold_fraction: Threshold fraction for size estimation
        
    Returns:
        Updated PipelineSession with geometry results
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # Fallback if tqdm not available
    
    # Filter units with STA data
    units_with_sta = [(uid, ud) for uid, ud in session.units.items() 
                      if ud.get('features', {}).get(STA_FEATURE_NAME, {}).get('data') is not None]
    
    processed_count = 0
    
    for unit_id, unit_data in tqdm(units_with_sta, desc="Extracting RF geometry"):
        # Get STA data
        sta_data = unit_data.get('features', {}).get(STA_FEATURE_NAME, {}).get('data')
        
        try:
            # Extract RF geometry using the function from rf_sta_measure.py
            geometry = extract_rf_geometry(
                sta_data,
                frame_range=frame_range,
                threshold_fraction=threshold_fraction,
            )
            
            # Store geometry in session (structured for HDF5 saving)
            if 'features' not in unit_data:
                unit_data['features'] = {}
            if STA_FEATURE_NAME not in unit_data['features']:
                unit_data['features'][STA_FEATURE_NAME] = {}
            
            # Convert geometry to structured dict for saving
            unit_data['features'][STA_FEATURE_NAME]['sta_geometry'] = _geometry_to_dict(geometry)
            
            processed_count += 1
                
        except Exception as e:
            logger.debug(f"Error processing {unit_id}: {e}")
            continue
    
    session.completed_steps.add('extract_rf_geometry')
    
    return session


def _geometry_to_dict(geometry: RFGeometry) -> Dict[str, Any]:
    """
    Convert RFGeometry to a structured dictionary for HDF5 saving.
    
    Separates results into individual items and subgroups.
    """
    result = {
        # Base geometry attributes (individual items)
        'center_row': geometry.center_row,
        'center_col': geometry.center_col,
        'size_x': geometry.size_x,
        'size_y': geometry.size_y,
        'area': geometry.area,
        'equivalent_diameter': geometry.equivalent_diameter,
        'peak_frame': geometry.peak_frame if geometry.peak_frame is not None else -1,
    }
    
    # Add diff_map if available (for visualization)
    if geometry.diff_map is not None:
        result['diff_map'] = geometry.diff_map
    
    # Gaussian fit group (individual items)
    if geometry.gaussian_fit is not None:
        gf = geometry.gaussian_fit
        result['gaussian_fit'] = {
            'center_x': gf.center_x,
            'center_y': gf.center_y,
            'sigma_x': gf.sigma_x,
            'sigma_y': gf.sigma_y,
            'amplitude': gf.amplitude,
            'theta': gf.theta,
            'offset': gf.offset,
            'r_squared': gf.r_squared,
        }
    
    # DoG fit group (individual items)
    if geometry.dog_fit is not None:
        df = geometry.dog_fit
        result['DoG'] = {
            'center_x': df.center_x,
            'center_y': df.center_y,
            'sigma_exc': df.sigma_exc,
            'sigma_inh': df.sigma_inh,
            'amp_exc': df.amp_exc,
            'amp_inh': df.amp_inh,
            'offset': df.offset,
            'r_squared': df.r_squared,
        }
    
    # ON/OFF model group (individual items)
    if geometry.on_off_fit is not None:
        oof = geometry.on_off_fit
        result['ONOFF_model'] = {
            'on_center_x': oof.on_center_x,
            'on_center_y': oof.on_center_y,
            'on_sigma_x': oof.on_sigma_x,
            'on_sigma_y': oof.on_sigma_y,
            'on_amplitude': oof.on_amplitude,
            'on_r_squared': oof.on_r_squared,
            'off_center_x': oof.off_center_x,
            'off_center_y': oof.off_center_y,
            'off_sigma_x': oof.off_sigma_x,
            'off_sigma_y': oof.off_sigma_y,
            'off_amplitude': oof.off_amplitude,
            'off_r_squared': oof.off_r_squared,
        }
    
    return result


# =============================================================================
# HDF5 Saving
# =============================================================================

def save_rf_geometry_to_hdf5(session: PipelineSession, output_path: Path = None) -> Path:
    """
    Save RF geometry results to HDF5 file.
    
    Copies the source HDF5 file to the export directory and adds sta_geometry
    groups to each unit, preserving all existing data.
    
    Args:
        session: PipelineSession with geometry data
        output_path: Path to save to (uses export/{dataset_id}.h5 if not provided)
        
    Returns:
        Path to saved file
    """
    import shutil
    
    if output_path is None:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / f"{session.dataset_id}.h5"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy source file to preserve all existing data
    source_path = session.hdf5_path
    if source_path and source_path.exists():
        logger.info(f"Copying source HDF5 to: {output_path}")
        shutil.copy2(source_path, output_path)
    else:
        logger.warning(f"Source file not found, creating new HDF5: {output_path}")
    
    logger.info(f"Adding RF geometry to: {output_path}")
    
    # Open in append mode to preserve existing data
    with h5py.File(output_path, 'a') as f:
        saved_count = 0
        
        for unit_id, unit_data in session.units.items():
            # Get geometry data from session
            geom_data = unit_data.get('features', {}).get(STA_FEATURE_NAME, {}).get('sta_geometry')
            
            if geom_data is None:
                continue
            
            # Check if unit group exists
            unit_path = f'units/{unit_id}'
            if unit_path not in f:
                logger.warning(f"Unit {unit_id} not found in HDF5, skipping")
                continue
            
            # Get or create feature group path
            feature_path = f'{unit_path}/features/{STA_FEATURE_NAME}'
            if feature_path not in f:
                f.create_group(feature_path)
            
            # Remove existing sta_geometry group if present (to update)
            geom_path = f'{feature_path}/sta_geometry'
            if geom_path in f:
                del f[geom_path]
            
            # Create sta_geometry group
            geom_group = f.create_group(geom_path)
            
            # Save base geometry attributes (individual datasets)
            for key in ['center_row', 'center_col', 'size_x', 'size_y', 
                        'area', 'equivalent_diameter', 'peak_frame']:
                if key in geom_data:
                    geom_group.create_dataset(key, data=geom_data[key])
            
            # Save gaussian_fit group (individual datasets)
            if 'gaussian_fit' in geom_data:
                gauss_group = geom_group.create_group('gaussian_fit')
                for key, value in geom_data['gaussian_fit'].items():
                    gauss_group.create_dataset(key, data=value)
            
            # Save DoG group (individual datasets)
            if 'DoG' in geom_data:
                dog_group = geom_group.create_group('DoG')
                for key, value in geom_data['DoG'].items():
                    dog_group.create_dataset(key, data=value)
            
            # Save ONOFF_model group (individual datasets)
            if 'ONOFF_model' in geom_data:
                onoff_group = geom_group.create_group('ONOFF_model')
                for key, value in geom_data['ONOFF_model'].items():
                    onoff_group.create_dataset(key, data=value)
            
            saved_count += 1
            logger.debug(f"Saved geometry for {unit_id}")
        
        # Add RF geometry metadata (create or update)
        meta_path = 'metadata/rf_sta_geometry'
        if meta_path in f:
            del f[meta_path]
        meta_group = f.create_group(meta_path)
        meta_group.create_dataset('sta_feature_name', data=STA_FEATURE_NAME)
        meta_group.create_dataset('frame_range', data=list(FRAME_RANGE))
        meta_group.create_dataset('threshold_fraction', data=THRESHOLD_FRACTION)
        meta_group.create_dataset('units_processed', data=saved_count)
    
    logger.info(f"Saved RF geometry for {saved_count} units to: {output_path}")
    return output_path


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """
    Main workflow for RF geometry extraction with session-based saving.
    """
    print("=" * 70)
    print("RF-STA Receptive Field Measurement - Session Workflow")
    print("=" * 70)
    print(f"Input file: {HDF5_PATH}")
    print(f"Export dir: {EXPORT_DIR}")
    print(f"Feature: {STA_FEATURE_NAME}")
    print(f"Frame range: {FRAME_RANGE}")
    
    if not HDF5_PATH.exists():
        print(f"Error: HDF5 file not found: {HDF5_PATH}")
        return
    
    # =========================================================================
    # Step 1: Load existing HDF5 into session
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Loading HDF5 into session")
    print("-" * 70)
    
    session = load_hdf5_to_session(HDF5_PATH)
    
    print(f"Session created: {session.dataset_id}")
    print(f"Units loaded: {len(session.units)}")
    print(f"Completed steps: {session.completed_steps}")
    
    # Count units with STA data
    units_with_sta = sum(
        1 for u in session.units.values()
        if 'features' in u and STA_FEATURE_NAME in u.get('features', {}) 
        and 'data' in u['features'][STA_FEATURE_NAME]
    )
    print(f"Units with STA data: {units_with_sta}")
    
    # =========================================================================
    # Step 2: Extract RF geometry (session mode)
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Extracting RF geometry")
    print("-" * 70)
    
    session = extract_rf_geometry_session(
        session,
        frame_range=FRAME_RANGE,
        threshold_fraction=THRESHOLD_FRACTION,
    )
    
    print(f"Completed steps: {session.completed_steps}")
    
    # =========================================================================
    # Step 3: Save to export HDF5
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Saving to export HDF5")
    print("-" * 70)
    
    output_path = save_rf_geometry_to_hdf5(session)
    
    # =========================================================================
    # Step 4: Verify saved data
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Verifying saved data")
    print("-" * 70)
    
    with h5py.File(output_path, 'r') as f:
        print(f"HDF5 structure:")
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: {obj.dtype}, shape={obj.shape}")
            else:
                print(f"{indent}{name}/")
        
        f.visititems(print_structure)
        
        # Show example unit
        if 'units' in f and len(f['units']) > 0:
            first_unit = list(f['units'].keys())[0]
            geom_path = f"units/{first_unit}/features/{STA_FEATURE_NAME}/sta_geometry"
            if geom_path in f:
                print(f"\nExample ({first_unit}):")
                geom = f[geom_path]
                print(f"  center: ({geom['center_col'][()]:.2f}, {geom['center_row'][()]:.2f})")
                if 'gaussian_fit' in geom:
                    print(f"  gaussian R2: {geom['gaussian_fit/r_squared'][()]:.4f}")
                if 'DoG' in geom:
                    print(f"  DoG R2: {geom['DoG/r_squared'][()]:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Session: {session.dataset_id}")
    print(f"Completed steps: {session.completed_steps}")
    print(f"\nResults saved to:")
    print(f"  {output_path}")
    
    # Count successful geometries
    geom_count = sum(
        1 for u in session.units.values()
        if u.get('features', {}).get(STA_FEATURE_NAME, {}).get('sta_geometry') is not None
    )
    print(f"\nUnits with RF geometry: {geom_count}/{len(session.units)}")


if __name__ == "__main__":
    main()

