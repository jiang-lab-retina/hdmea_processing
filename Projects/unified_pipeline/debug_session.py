"""Debug script to check session data structure."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import h5py
import numpy as np

def check_hdf5_structure():
    """Check the HDF5 file structure."""
    path = project_root / "Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec_steps1-7.h5"
    
    if not path.exists():
        print(f"File not found: {path}")
        return
    
    print(f"Checking: {path}")
    print()
    
    with h5py.File(path, 'r') as f:
        print("=== Checking stimulus/section_time structure ===")
        if 'stimulus/section_time' in f:
            grp = f['stimulus/section_time']
            print(f"Type: {type(grp)}")
            if hasattr(grp, 'keys'):
                print(f"Keys: {list(grp.keys())}")
                for k in list(grp.keys())[:3]:  # First 3
                    item = grp[k]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {k}: shape={item.shape}, dtype={item.dtype}")
                        if item.shape[0] > 0:
                            print(f"    First row: {item[0][:3] if len(item.shape) > 1 else item[:3]}")
        else:
            print("section_time not found")
        
        print()
        print("=== Checking unit structure ===")
        units_grp = f.get('units')
        if units_grp:
            unit_ids = list(units_grp.keys())
            print(f"Found {len(unit_ids)} units")
            unit_id = unit_ids[0]
            print(f"Checking first unit: {unit_id}")
            unit = units_grp[unit_id]
            
            # Check spike_times_sectioned
            if 'spike_times_sectioned' in unit:
                stg = unit['spike_times_sectioned']
                print(f"  spike_times_sectioned keys: {list(stg.keys())[:5]}")
                
                # Check the movie we're looking for
                movie_name = 'moving_h_bar_s5_d8_3x'
                if movie_name in stg:
                    movie = stg[movie_name]
                    if isinstance(movie, h5py.Group):
                        print(f"  {movie_name} is a Group, keys: {list(movie.keys())}")
                        if 'full_spike_times' in movie:
                            fst = movie['full_spike_times']
                            print(f"    full_spike_times: shape={fst.shape}, first={fst[:5]}")
                    else:
                        print(f"  {movie_name} is a Dataset: shape={movie.shape}")
                else:
                    print(f"  Movie '{movie_name}' not in spike_times_sectioned")
            else:
                print("  No spike_times_sectioned")
            
            # Check features/eimage_sta
            print()
            if 'features/eimage_sta' in unit:
                print("  features/eimage_sta found")
                sta = unit['features/eimage_sta']
                
                # Check for data
                if 'data' in sta:
                    data = sta['data']
                    print(f"    data: shape={data.shape}, dtype={data.dtype}")
                
                # Check for geometry
                if 'geometry' in sta:
                    geom = sta['geometry']
                    if isinstance(geom, h5py.Group):
                        print(f"    geometry keys: {list(geom.keys())}")
                        for gk in geom.keys():
                            gv = geom[gk]
                            if isinstance(gv, h5py.Dataset):
                                print(f"      {gk}: {gv[()]}")
                    else:
                        print(f"    geometry is dataset: {geom.shape}")
            else:
                print("  No features/eimage_sta")
        
        print()
        print("=== Checking metadata/frame_timestamps ===")
        if 'metadata/frame_timestamps' in f:
            ft = f['metadata/frame_timestamps']
            print(f"frame_timestamps: shape={ft.shape}, dtype={ft.dtype}")
            print(f"  First 5: {ft[:5]}")
        else:
            print("frame_timestamps not found in metadata")
            # Check stimulus
            if 'stimulus/frame_time' in f:
                ft = f['stimulus/frame_time']
                if isinstance(ft, h5py.Group):
                    print(f"Found in stimulus/frame_time, keys: {list(ft.keys())}")
                else:
                    print(f"Found in stimulus/frame_time: shape={ft.shape}")

def check_session_loader():
    """Check what the session loader returns."""
    from hdmea.pipeline import load_session_from_hdf5
    
    path = project_root / "Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec_steps1-7.h5"
    
    print()
    print("=" * 60)
    print("=== Checking Session Loader ===")
    print("=" * 60)
    
    session = load_session_from_hdf5(path)
    
    print(f"Dataset ID: {session.dataset_id}")
    print(f"Units: {len(session.units)}")
    
    print()
    print("=== session.stimulus structure ===")
    for k, v in session.stimulus.items():
        print(f"  {k}: {type(v)}")
        if isinstance(v, dict):
            for k2, v2 in list(v.items())[:3]:
                if isinstance(v2, np.ndarray):
                    print(f"    {k2}: ndarray shape={v2.shape}")
                elif isinstance(v2, dict):
                    print(f"    {k2}: dict keys={list(v2.keys())[:5]}")
                else:
                    print(f"    {k2}: {type(v2)}")
    
    print()
    print("=== Checking section_time for moving_h_bar_s5_d8_3x ===")
    section_time_data = session.stimulus.get('section_time', {})
    print(f"section_time type: {type(section_time_data)}")
    
    movie_name = 'moving_h_bar_s5_d8_3x'
    section_time = section_time_data.get(movie_name) if isinstance(section_time_data, dict) else None
    print(f"section_time[{movie_name}] type: {type(section_time)}")
    
    if isinstance(section_time, dict):
        print(f"  It's a dict with keys: {list(section_time.keys())}")
        if 'data' in section_time:
            data = section_time['data']
            print(f"  data: shape={data.shape if hasattr(data, 'shape') else 'N/A'}")
    elif isinstance(section_time, np.ndarray):
        print(f"  It's an array with shape: {section_time.shape}")
        print(f"  First value: {section_time[0] if section_time.size > 0 else 'empty'}")
    
    print()
    print("=== Checking frame_timestamps ===")
    frame_timestamps = session.metadata.get('frame_timestamps')
    if frame_timestamps is None:
        frame_time_data = session.stimulus.get('frame_time', {})
        print(f"frame_time type: {type(frame_time_data)}")
        if isinstance(frame_time_data, dict):
            print(f"  keys: {list(frame_time_data.keys())}")
            frame_timestamps = frame_time_data.get('default')
    
    if frame_timestamps is not None:
        print(f"frame_timestamps type: {type(frame_timestamps)}")
        if isinstance(frame_timestamps, np.ndarray):
            print(f"  shape: {frame_timestamps.shape}")
        elif isinstance(frame_timestamps, dict):
            print(f"  dict keys: {list(frame_timestamps.keys())}")
    
    print()
    print("=== Checking first unit spike_times_sectioned ===")
    unit_id = list(session.units.keys())[0]
    unit_data = session.units[unit_id]
    sectioned = unit_data.get('spike_times_sectioned', {})
    print(f"spike_times_sectioned type: {type(sectioned)}")
    if isinstance(sectioned, dict):
        print(f"  keys: {list(sectioned.keys())[:5]}")
        movie_sectioned = sectioned.get(movie_name)
        if movie_sectioned is not None:
            print(f"  {movie_name} type: {type(movie_sectioned)}")
            if isinstance(movie_sectioned, dict):
                print(f"    keys: {list(movie_sectioned.keys())}")
    
    print()
    print("=== Checking STA geometry ===")
    features = unit_data.get('features', {})
    eimage_sta = features.get('eimage_sta', {})
    print(f"eimage_sta type: {type(eimage_sta)}")
    if isinstance(eimage_sta, dict):
        print(f"  keys: {list(eimage_sta.keys())}")
        geometry = eimage_sta.get('geometry', {})
        print(f"  geometry type: {type(geometry)}")
        if isinstance(geometry, dict):
            print(f"    keys: {list(geometry.keys())}")
            center_row = geometry.get('center_row')
            center_col = geometry.get('center_col')
            print(f"    center_row: {center_row}, type: {type(center_row)}")
            print(f"    center_col: {center_col}, type: {type(center_col)}")


def test_section_time_access():
    """Test the exact access pattern that fails in DSGC."""
    from hdmea.pipeline import load_session_from_hdf5
    
    path = project_root / "Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec_steps1-7.h5"
    
    print()
    print("=" * 60)
    print("=== Testing DSGC Section Time Access ===")
    print("=" * 60)
    
    session = load_session_from_hdf5(path)
    
    movie_name = 'moving_h_bar_s5_d8_3x'
    
    # Exactly replicate DSGC code:
    section_time_data = session.stimulus.get('section_time', {})
    print(f"1. section_time_data type: {type(section_time_data)}")
    
    if isinstance(section_time_data, np.ndarray):
        print("   section_time_data is an array, not dict!")
        print(f"   Shape: {section_time_data.shape}")
        return
    
    section_time = section_time_data.get(movie_name) if isinstance(section_time_data, dict) else None
    print(f"2. section_time for '{movie_name}': type={type(section_time)}")
    
    if section_time is None:
        print("   section_time is None!")
        return
    
    if isinstance(section_time, dict):
        print(f"   It's a dict with keys: {list(section_time.keys())}")
        section_time = section_time.get('data', section_time)
        print(f"   After .get('data'): type={type(section_time)}")
    
    print(f"3. Before np.asarray: type={type(section_time)}")
    if isinstance(section_time, np.ndarray):
        print(f"   Shape: {section_time.shape}")
    elif isinstance(section_time, dict):
        print(f"   Still a dict! Keys: {list(section_time.keys())}")
    
    try:
        section_time = np.asarray(section_time)
        print(f"4. After np.asarray: shape={section_time.shape}, dtype={section_time.dtype}")
        
        if section_time.ndim == 0:
            print("   WARNING: 0-dimensional array!")
            print(f"   Contents: {section_time}")
        else:
            print(f"   First row: {section_time[0] if section_time.shape[0] > 0 else 'empty'}")
        
        movie_start_sample = section_time[0, 0]
        print(f"5. movie_start_sample = {movie_start_sample}")
        
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_hdf5_structure()
    check_session_loader()
    test_section_time_access()

