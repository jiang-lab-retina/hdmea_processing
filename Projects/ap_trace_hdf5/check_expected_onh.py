"""Check where ONH would be expected based on direction."""
import h5py
from pathlib import Path
import numpy as np

files = [
    ('2024.05.22-12.43.26-Rec.h5', 'FAILING', 118.6),  # consensus direction
    ('2024.06.18-11.16.04-Rec.h5', 'FAILING', 128.5),
    ('2024.03.07-10.12.28-Rec.h5', 'WORKING', 86.8),
]

export_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')
CENTER = (33, 33)  # chip center
MAX_DIST = 98  # max distance constraint

for fname, status, consensus_dir in files:
    f = export_dir / fname
    if not f.exists():
        continue
    
    print(f"\n{'='*60}")
    print(f"{status}: {fname}")
    print(f"Consensus direction: {consensus_dir}°")
    print('='*60)
    
    # Calculate where ONH would be if at edge of constraint
    dir_rad = np.radians(consensus_dir)
    onh_at_edge_col = CENTER[0] + MAX_DIST * np.cos(dir_rad)
    onh_at_edge_row = CENTER[1] + MAX_DIST * np.sin(dir_rad)
    
    print(f"\nIf ONH at max distance (98px) from center:")
    print(f"  ONH would be at: ({onh_at_edge_col:.1f}, {onh_at_edge_row:.1f})")
    
    # Check if this is within reasonable bounds
    if onh_at_edge_col < -50 or onh_at_edge_col > 150:
        print("  WARNING: ONH col is far outside typical range!")
    if onh_at_edge_row < -50 or onh_at_edge_row > 150:
        print("  WARNING: ONH row is far outside typical range!")
    
    # Calculate expected direction for each cell's start point
    with h5py.File(f, 'r') as h:
        start_points = []
        for uid in h['units'].keys():
            ap_path = f'units/{uid}/features/ap_tracking'
            if ap_path not in h:
                continue
            
            # Get centroids
            for pp_name in ['post_processed_data', 'post_processed']:
                pp_path = f'{ap_path}/{pp_name}'
                if pp_path in h and 'axon_centroids' in h[pp_path]:
                    centroids = h[f'{pp_path}/axon_centroids'][:]
                    if len(centroids) >= 3:
                        # Sort by time
                        if centroids.shape[1] == 3:
                            sorted_idx = np.argsort(centroids[:, 0])
                            centroids = centroids[sorted_idx]
                            # Start point (row, col)
                            start_points.append((centroids[0, 1], centroids[0, 2]))
                    break
        
        if start_points:
            start_arr = np.array(start_points)
            mean_start = start_arr.mean(axis=0)
            print(f"\nMean start point (row, col): ({mean_start[0]:.1f}, {mean_start[1]:.1f})")
            
            # Check if ONH would be "forward" from start points
            to_onh_col = onh_at_edge_col - mean_start[1]
            to_onh_row = onh_at_edge_row - mean_start[0]
            
            # Direction from mean start to hypothetical ONH
            actual_dir = np.degrees(np.arctan2(to_onh_row, to_onh_col))
            if actual_dir < 0:
                actual_dir += 360
            
            print(f"Direction from mean start to edge ONH: {actual_dir:.1f}°")
            
            # Check if this matches consensus
            diff = abs(actual_dir - consensus_dir)
            if diff > 180:
                diff = 360 - diff
            print(f"Difference from consensus: {diff:.1f}°")
            
            if diff > 90:
                print("  WARNING: ONH at edge is in OPPOSITE direction from consensus!")
                print("  This causes the direction constraint penalty to block the solution!")

