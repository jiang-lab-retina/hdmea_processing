"""Check pathway data in detail for all files."""
import h5py
from pathlib import Path
import numpy as np

files = [
    ('2024.05.22-12.43.26-Rec.h5', 'FAILING'),
    ('2024.06.18-11.16.04-Rec.h5', 'FAILING'),
    ('2024.03.07-10.12.28-Rec.h5', 'WORKING'),
]

export_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')

for fname, status in files:
    f = export_dir / fname
    if not f.exists():
        print(f"Not found: {fname}")
        continue
    
    print(f"\n{'='*60}")
    print(f"{status}: {fname}")
    print('='*60)
    
    with h5py.File(f, 'r') as h:
        # Count pathways with different criteria
        total_pathways = 0
        has_r_squared = 0
        has_direction = 0
        has_centroids = 0
        r_squared_values = []
        
        for uid in h['units'].keys():
            ap_path = f'units/{uid}/features/ap_tracking'
            if ap_path not in h:
                continue
            
            pw_path = f'{ap_path}/ap_pathway'
            if pw_path not in h:
                continue
            
            total_pathways += 1
            pw = h[pw_path]
            
            # Check r_squared (new) or r_value (old)
            if 'r_squared' in pw:
                has_r_squared += 1
                r_squared_values.append(pw['r_squared'][()])
            elif 'r_value' in pw:
                r_val = pw['r_value'][()]
                r_squared_values.append(r_val ** 2)  # Convert to R²
            
            # Check direction_angle
            if 'direction_angle' in pw:
                dir_angle = pw['direction_angle'][()]
                if not np.isnan(dir_angle):
                    has_direction += 1
            
            # Check centroids in post_processed_data or post_processed
            for pp_name in ['post_processed_data', 'post_processed']:
                pp_path = f'{ap_path}/{pp_name}'
                if pp_path in h and 'axon_centroids' in h[pp_path]:
                    centroids = h[f'{pp_path}/axon_centroids'][:]
                    if len(centroids) >= 3:
                        has_centroids += 1
                    break
        
        print(f"Total pathways: {total_pathways}")
        print(f"  With R² (new style): {has_r_squared}")
        print(f"  With direction_angle: {has_direction}")
        print(f"  With centroids (>=3): {has_centroids}")
        
        if r_squared_values:
            r2_arr = np.array(r_squared_values)
            r2_arr = r2_arr[~np.isnan(r2_arr)]
            print(f"\nR² distribution (from r_value² if old format):")
            print(f"  Count: {len(r2_arr)}")
            print(f"  Min: {r2_arr.min():.4f}, Max: {r2_arr.max():.4f}")
            print(f"  Mean: {r2_arr.mean():.4f}")
            print(f"  R² >= 0.8: {np.sum(r2_arr >= 0.8)}")
            print(f"  R² >= 0.6: {np.sum(r2_arr >= 0.6)}")
            print(f"  R² >= 0.4: {np.sum(r2_arr >= 0.4)}")

