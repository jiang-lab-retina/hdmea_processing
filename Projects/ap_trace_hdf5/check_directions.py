"""Check direction angles calculated from centroids."""
import h5py
from pathlib import Path
import numpy as np

files = [
    ('2024.05.22-12.43.26-Rec.h5', 'FAILING'),
    ('2024.06.18-11.16.04-Rec.h5', 'FAILING'),
    ('2024.03.07-10.12.28-Rec.h5', 'WORKING'),
]

export_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')

def calc_direction_from_centroids(centroids):
    """Calculate direction from centroid temporal order."""
    if centroids is None or len(centroids) < 3:
        return None
    
    # Sort by time
    if centroids.shape[1] == 3:
        sorted_idx = np.argsort(centroids[:, 0])
        centroids = centroids[sorted_idx]
        coords = centroids[:, 1:3]  # [row, col]
    else:
        coords = centroids
    
    # Get first and last points
    start = coords[0]
    end = coords[-1]
    
    d_row = end[0] - start[0]
    d_col = end[1] - start[1]
    
    angle = np.degrees(np.arctan2(d_row, d_col))
    if angle < 0:
        angle += 360
    return angle

for fname, status in files:
    f = export_dir / fname
    if not f.exists():
        print(f"Not found: {fname}")
        continue
    
    print(f"\n{'='*60}")
    print(f"{status}: {fname}")
    print('='*60)
    
    with h5py.File(f, 'r') as h:
        directions = []
        r2_values = []
        
        for uid in h['units'].keys():
            ap_path = f'units/{uid}/features/ap_tracking'
            if ap_path not in h:
                continue
            
            pw_path = f'{ap_path}/ap_pathway'
            if pw_path not in h:
                continue
            
            pw = h[pw_path]
            
            # Get R² value
            if 'r_value' in pw:
                r2 = pw['r_value'][()] ** 2
                if np.isnan(r2) or r2 < 0.8:  # Only consider high R²
                    continue
            else:
                continue
            
            # Get centroids
            for pp_name in ['post_processed_data', 'post_processed']:
                pp_path = f'{ap_path}/{pp_name}'
                if pp_path in h and 'axon_centroids' in h[pp_path]:
                    centroids = h[f'{pp_path}/axon_centroids'][:]
                    if len(centroids) >= 3:
                        direction = calc_direction_from_centroids(centroids)
                        if direction is not None:
                            directions.append(direction)
                            r2_values.append(r2)
                    break
        
        if directions:
            dir_arr = np.array(directions)
            print(f"High R² (>=0.8) pathways with direction: {len(directions)}")
            print(f"\nDirection distribution:")
            print(f"  Min: {dir_arr.min():.1f}°, Max: {dir_arr.max():.1f}°")
            print(f"  Mean: {dir_arr.mean():.1f}°, Std: {dir_arr.std():.1f}°")
            
            # Check for bimodal distribution (cells pointing opposite directions)
            sin_sum = np.sum(np.sin(np.radians(dir_arr)))
            cos_sum = np.sum(np.cos(np.radians(dir_arr)))
            circular_mean = np.degrees(np.arctan2(sin_sum, cos_sum))
            if circular_mean < 0:
                circular_mean += 360
            
            # Resultant length (0 = random/bimodal, 1 = all same direction)
            resultant_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(dir_arr)
            
            print(f"  Circular mean: {circular_mean:.1f}°")
            print(f"  Resultant length: {resultant_length:.3f} (1=aligned, 0=scattered)")
            
            # Count within 90° of circular mean
            diffs = np.abs(dir_arr - circular_mean)
            diffs = np.minimum(diffs, 360 - diffs)
            within_90 = np.sum(diffs <= 90)
            print(f"  Within ±90° of mean: {within_90}/{len(dir_arr)}")
            
            # Show histogram
            bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
            hist, _ = np.histogram(dir_arr, bins=bins)
            print(f"\nDirection histogram (count per 45° bin):")
            for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                bar = '*' * hist[i]
                print(f"  {lo:3d}°-{hi:3d}°: {bar} ({hist[i]})")

