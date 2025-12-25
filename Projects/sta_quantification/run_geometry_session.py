"""
Run Soma Geometry Extraction with Session Workflow

This script demonstrates using extract_eimage_sta_geometry with the 
PipelineSession workflow, following the same pattern as pipeline_session.py.

Workflow:
1. Load existing HDF5 data into a session
2. Run geometry extraction (session mode)
3. Save results back to HDF5 using session.save() or update existing file

Author: Generated for experimental analysis
Date: 2024-12
"""

from pathlib import Path
import h5py
import numpy as np

# Import session utilities
from hdmea.pipeline import create_session, PipelineSession

# Import our geometry extraction function
from ap_sta import (
    extract_eimage_sta_geometry,
    SomaGeometry,
    plot_geometry_results,
    load_geometries_from_hdf5,
)

# Enable logging
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# =============================================================================
# Configuration
# =============================================================================

# Input HDF5 file with eimage_sta data
HDF5_PATH = Path(__file__).parent.parent / "pipeline_test" / "data" / "2024.03.01-14.40.14-Rec.h5"

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / "results"

# Geometry extraction parameters
FRAME_RANGE = (10, 14)  # Frames to use for size estimation
THRESHOLD_FRACTION = 0.5  # Threshold for soma mask


# =============================================================================
# Helper: Load existing HDF5 into session
# =============================================================================

def load_hdf5_to_session(hdf5_path: Path, dataset_id: str = None) -> PipelineSession:
    """
    Load an existing HDF5 file into a PipelineSession.
    
    This allows running additional pipeline steps on existing data.
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Optional dataset ID (uses filename stem if not provided)
        
    Returns:
        PipelineSession with data loaded
    """
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path  # Track source file
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load units with their features
        if 'units' in f:
            for unit_id in f['units'].keys():
                unit_group = f[f'units/{unit_id}']
                unit_data = {}
                
                # Load eimage_sta if available
                eimage_sta_path = 'features/eimage_sta/data'
                if eimage_sta_path in unit_group:
                    if 'features' not in unit_data:
                        unit_data['features'] = {}
                    if 'eimage_sta' not in unit_data['features']:
                        unit_data['features']['eimage_sta'] = {}
                    unit_data['features']['eimage_sta']['data'] = unit_group[eimage_sta_path][:]
                
                # Load existing geometry if available
                geometry_path = 'features/eimage_sta/geometry'
                if geometry_path in unit_group:
                    geom_group = unit_group[geometry_path]
                    if 'features' not in unit_data:
                        unit_data['features'] = {}
                    if 'eimage_sta' not in unit_data['features']:
                        unit_data['features']['eimage_sta'] = {}
                    unit_data['features']['eimage_sta']['geometry'] = {
                        'center_row': float(geom_group['center_row'][()]),
                        'center_col': float(geom_group['center_col'][()]),
                        'size_x': float(geom_group['size_x'][()]),
                        'size_y': float(geom_group['size_y'][()]),
                        'area': float(geom_group['area'][()]),
                        'equivalent_diameter': float(geom_group['equivalent_diameter'][()]),
                        'diff_map': geom_group['diff_map'][:] if 'diff_map' in geom_group else None,
                    }
                
                if unit_data:
                    session.units[unit_id] = unit_data
        
        # Load metadata if available
        if 'metadata' in f:
            for key in f['metadata'].keys():
                try:
                    session.metadata[key] = f[f'metadata/{key}'][()]
                except:
                    pass
    
    session.completed_steps.add('load_hdf5')
    return session


def save_session_to_hdf5(session: PipelineSession, hdf5_path: Path = None) -> Path:
    """
    Save session geometry data back to HDF5 file.
    
    Updates the existing file with geometry data, preserving other data.
    
    Args:
        session: PipelineSession with geometry data
        hdf5_path: Path to save to (uses session.hdf5_path if not provided)
        
    Returns:
        Path to saved file
    """
    if hdf5_path is None:
        hdf5_path = session.hdf5_path
    
    if hdf5_path is None:
        raise ValueError("No HDF5 path specified")
    
    with h5py.File(hdf5_path, 'r+') as f:
        for unit_id, unit_data in session.units.items():
            geom_data = unit_data.get('features', {}).get('eimage_sta', {}).get('geometry')
            
            if geom_data is None:
                continue
            
            # Ensure unit group exists
            unit_path = f'units/{unit_id}'
            if unit_path not in f:
                continue
            
            # Create/update geometry group
            geometry_path = f'{unit_path}/features/eimage_sta/geometry'
            if geometry_path in f:
                del f[geometry_path]
            
            geom_group = f.create_group(geometry_path)
            
            # Save all geometry attributes
            for key, value in geom_data.items():
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    geom_group.create_dataset(key, data=value, compression='gzip')
                else:
                    geom_group.create_dataset(key, data=value)
    
    print(f"Geometry saved to: {hdf5_path}")
    return hdf5_path


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """
    Main workflow demonstrating session-based geometry extraction.
    """
    print("=" * 60)
    print("Soma Geometry Extraction - Session Workflow")
    print("=" * 60)
    print(f"Input file: {HDF5_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Frame range: {FRAME_RANGE}")
    
    if not HDF5_PATH.exists():
        print(f"Error: HDF5 file not found: {HDF5_PATH}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load existing HDF5 into session
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 1: Loading HDF5 into session")
    print("-" * 60)
    
    session = load_hdf5_to_session(HDF5_PATH)
    
    print(f"Session created: {session.dataset_id}")
    print(f"Units loaded: {len(session.units)}")
    print(f"Completed steps: {session.completed_steps}")
    
    # Count units with eimage_sta
    units_with_eimage_sta = sum(
        1 for u in session.units.values()
        if 'features' in u and 'eimage_sta' in u.get('features', {}) 
        and 'data' in u['features']['eimage_sta']
    )
    print(f"Units with eimage_sta: {units_with_eimage_sta}")
    
    # =========================================================================
    # Step 2: Extract soma geometry (session mode)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Extracting soma geometry")
    print("-" * 60)
    
    # This follows the same pattern as other pipeline functions:
    # session = some_function(..., session=session)
    session = extract_eimage_sta_geometry(
        frame_range=FRAME_RANGE,
        threshold_fraction=THRESHOLD_FRACTION,
        session=session,  # Pass session for deferred mode
    )
    
    print(f"Completed steps: {session.completed_steps}")
    
    # =========================================================================
    # Step 3: Save session back to HDF5
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 3: Saving session to HDF5")
    print("-" * 60)
    
    save_session_to_hdf5(session, HDF5_PATH)
    
    # =========================================================================
    # Step 4: Generate visualization plots (optional)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 4: Generating visualization plots")
    print("-" * 60)
    
    # Extract geometries from session for plotting
    geometries = {}
    for unit_id, unit_data in session.units.items():
        geom_data = unit_data.get('features', {}).get('eimage_sta', {}).get('geometry')
        if geom_data:
            geometries[unit_id] = SomaGeometry(
                center_row=geom_data['center_row'],
                center_col=geom_data['center_col'],
                size_x=geom_data['size_x'],
                size_y=geom_data['size_y'],
                area=geom_data['area'],
                equivalent_diameter=geom_data['equivalent_diameter'],
                diff_map=geom_data.get('diff_map'),
            )
    
    plot_geometry_results(
        hdf5_path=HDF5_PATH,
        geometries=geometries,
        output_dir=OUTPUT_DIR,
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"Session: {session.dataset_id}")
    print(f"Units processed: {len(geometries)}")
    print(f"Completed steps: {session.completed_steps}")
    print(f"\nResults saved to:")
    print(f"  HDF5: {HDF5_PATH}")
    print(f"  Plots: {OUTPUT_DIR}")
    
    # =========================================================================
    # Verify: Load back from HDF5 to confirm
    # =========================================================================
    print("\n" + "-" * 60)
    print("Verification: Loading geometry from saved HDF5")
    print("-" * 60)
    
    loaded_geometries = load_geometries_from_hdf5(HDF5_PATH)
    print(f"Loaded {len(loaded_geometries)} geometries from HDF5")
    
    if loaded_geometries:
        first_unit = list(loaded_geometries.keys())[0]
        g = loaded_geometries[first_unit]
        print(f"Example ({first_unit}): center=({g.center_row:.1f}, {g.center_col:.1f}), "
              f"size={g.size_x:.1f}×{g.size_y:.1f}")


if __name__ == "__main__":
    main()

