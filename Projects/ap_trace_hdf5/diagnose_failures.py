"""Diagnose why global optimization fails for certain files."""
import sys
sys.path.insert(0, 'M:/Python_Project/Data_Processing_2027')
from pathlib import Path
import h5py
import numpy as np

# Files that fell back to legacy
test_files = [
    '2024.05.22-12.43.26-Rec.h5',
    '2024.06.18-11.16.04-Rec.h5',
]

export_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')

for fname in test_files:
    hdf5_path = export_dir / fname
    if not hdf5_path.exists():
        print(f"File not found: {fname}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Diagnosing: {fname}")
    print('='*60)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Count RGCs
        rgc_units = []
        for uid in f['units'].keys():
            label_path = f'units/{uid}/auto_label/axon_type'
            if label_path in f:
                cell_type = f[label_path][()]
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode()
                if cell_type.lower() == 'rgc':
                    rgc_units.append(uid)
        
        print(f"Total RGC units: {len(rgc_units)}")
        
        # Check AP pathways
        pathways_with_data = 0
        r2_values = []
        direction_angles = []
        centroids_count = []
        no_pathway = 0
        no_r2 = 0
        no_direction = 0
        no_centroids = 0
        
        for uid in rgc_units:
            ap_path = f'units/{uid}/features/ap_tracking'
            if ap_path not in f:
                no_pathway += 1
                continue
            
            pathway_path = f'{ap_path}/ap_pathway'
            if pathway_path not in f:
                no_pathway += 1
                continue
            
            pw_grp = f[pathway_path]
            pathways_with_data += 1
            
            # Debug: print fields for first unit
            if pathways_with_data == 1:
                print(f"  First pathway fields: {list(pw_grp.keys())}")
            
            # Get R²
            if 'r_squared' in pw_grp:
                r2 = pw_grp['r_squared'][()]
                if not np.isnan(r2):
                    r2_values.append(r2)
            else:
                no_r2 += 1
            
            # Get direction
            if 'direction_angle' in pw_grp:
                dir_angle = pw_grp['direction_angle'][()]
                if not np.isnan(dir_angle):
                    direction_angles.append(dir_angle)
            else:
                no_direction += 1
            
            # Get centroids
            post_path = f'{ap_path}/post_processed'
            if post_path in f and 'axon_centroids' in f[post_path]:
                centroids = f[f'{post_path}/axon_centroids'][:]
                centroids_count.append(len(centroids))
            elif 'axon_centroids' in f[ap_path]:
                centroids = f[f'{ap_path}/axon_centroids'][:]
                centroids_count.append(len(centroids))
            else:
                no_centroids += 1
        
        print(f"  No pathway: {no_pathway}, No R²: {no_r2}, No direction: {no_direction}, No centroids: {no_centroids}")
        
        print(f"Pathways with data: {pathways_with_data}")
        
        if r2_values:
            r2_arr = np.array(r2_values)
            print(f"\nR² distribution:")
            print(f"  Min: {r2_arr.min():.4f}, Max: {r2_arr.max():.4f}")
            print(f"  Mean: {r2_arr.mean():.4f}, Median: {np.median(r2_arr):.4f}")
            print(f"  R² >= 0.8: {np.sum(r2_arr >= 0.8)}")
            print(f"  R² >= 0.6: {np.sum(r2_arr >= 0.6)}")
            print(f"  R² >= 0.4: {np.sum(r2_arr >= 0.4)}")
        
        if direction_angles:
            dir_arr = np.array(direction_angles)
            print(f"\nDirection distribution:")
            print(f"  Min: {dir_arr.min():.1f}°, Max: {dir_arr.max():.1f}°")
            print(f"  Mean: {dir_arr.mean():.1f}°, Std: {dir_arr.std():.1f}°")
            
            # Check direction spread
            # Convert to radians and calculate circular mean
            sin_sum = np.sum(np.sin(np.radians(dir_arr)))
            cos_sum = np.sum(np.cos(np.radians(dir_arr)))
            circular_mean = np.degrees(np.arctan2(sin_sum, cos_sum))
            if circular_mean < 0:
                circular_mean += 360
            print(f"  Circular mean: {circular_mean:.1f}°")
            
            # Count within 90° of circular mean
            diffs = np.abs(dir_arr - circular_mean)
            diffs = np.minimum(diffs, 360 - diffs)
            within_90 = np.sum(diffs <= 90)
            print(f"  Within ±90° of mean: {within_90}/{len(dir_arr)}")
        
        if centroids_count:
            cent_arr = np.array(centroids_count)
            print(f"\nCentroids per unit:")
            print(f"  Min: {cent_arr.min()}, Max: {cent_arr.max()}")
            print(f"  Mean: {cent_arr.mean():.1f}")
            print(f"  Units with >= 3 centroids: {np.sum(cent_arr >= 3)}")

print("\n" + "="*60)
print("Diagnosis complete!")

