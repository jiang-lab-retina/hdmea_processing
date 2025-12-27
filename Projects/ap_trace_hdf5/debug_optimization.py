"""Debug why optimization fails for certain files."""
import sys
sys.path.insert(0, 'M:/Python_Project/Data_Processing_2027')
from pathlib import Path
import shutil
import h5py
import numpy as np

# Re-process one failing file with debug output
fname = '2024.05.22-12.43.26-Rec.h5'
source_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/add_manual_label_cell_type/manual_label_20251225')
output_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')
model_path = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth')

src = source_dir / fname
dst = output_dir / fname

# Copy fresh
if dst.exists():
    dst.unlink()
shutil.copy2(src, dst)

# Now manually trace through the algorithm
from src.hdmea.features.ap_tracking.pathway_analysis import (
    prepare_cell_data_for_onh,
    calculate_consensus_direction,
    filter_pathways_by_direction,
    find_optimal_onh,
    APPathway,
)

# After compute_ap_tracking runs, check the pathway data
from src.hdmea.features.ap_tracking.core import compute_ap_tracking

print("Running compute_ap_tracking...")
compute_ap_tracking(
    dst, model_path,
    r2_threshold=0.8,
    centroid_start_frame=10,
    max_displacement=100,
    max_displacement_post=5.0,
    centroid_exclude_fraction=0.1,
)

# Now check what was stored
print("\n" + "="*60)
print("Checking stored pathway data...")
print("="*60)

with h5py.File(dst, 'r') as f:
    pathways = {}
    all_centroids = {}
    
    for uid in f['units'].keys():
        ap_path = f'units/{uid}/features/ap_tracking'
        if ap_path not in f:
            continue
        
        pw_path = f'{ap_path}/ap_pathway'
        if pw_path not in f:
            continue
        
        pw_grp = f[pw_path]
        
        # Check what fields exist
        fields = list(pw_grp.keys())
        
        # Get R² (try both names)
        r_squared = None
        if 'r_squared' in pw_grp:
            r_squared = pw_grp['r_squared'][()]
        elif 'r_value' in pw_grp:
            r_squared = pw_grp['r_value'][()] ** 2
        
        # Get direction_angle
        direction_angle = None
        if 'direction_angle' in pw_grp:
            direction_angle = pw_grp['direction_angle'][()]
        
        # Get centroids
        centroids = None
        for pp_name in ['post_processed', 'post_processed_data']:
            pp_path = f'{ap_path}/{pp_name}'
            if pp_path in f and 'axon_centroids' in f[pp_path]:
                centroids = f[f'{pp_path}/axon_centroids'][:]
                break
        
        if r_squared is not None and not np.isnan(r_squared):
            # Create APPathway-like object for analysis
            pathways[uid] = type('PW', (), {
                'r_squared': r_squared,
                'direction_angle': direction_angle,
                'direction_valid': True,  # Will be updated
                'slope': pw_grp['slope'][()] if 'slope' in pw_grp else 0,
                'intercept': pw_grp['intercept'][()] if 'intercept' in pw_grp else 0,
            })()
            
            if centroids is not None and len(centroids) >= 3:
                all_centroids[uid] = centroids

print(f"\nTotal pathways loaded: {len(pathways)}")
print(f"Pathways with centroids: {len(all_centroids)}")

# Check R² distribution
r2_values = [pw.r_squared for pw in pathways.values() if pw.r_squared is not None]
print(f"\nR² distribution:")
print(f"  Count: {len(r2_values)}")
print(f"  R² >= 0.8: {sum(1 for r in r2_values if r >= 0.8)}")
print(f"  R² >= 0.6: {sum(1 for r in r2_values if r >= 0.6)}")

# Check direction angles
dir_angles = [pw.direction_angle for pw in pathways.values() if pw.direction_angle is not None]
print(f"\nDirection angles:")
print(f"  Count with direction_angle: {len(dir_angles)}")

# Check consensus direction
consensus = calculate_consensus_direction(pathways, r2_threshold=0.8)
print(f"\nConsensus direction: {consensus}")

# Filter by direction
if consensus is not None:
    pathways = filter_pathways_by_direction(pathways, consensus, 90.0)
    valid_count = sum(1 for pw in pathways.values() if pw.r_squared >= 0.8 and pw.direction_valid)
    print(f"After direction filter: {valid_count} valid")

# Prepare cell data
cell_data, unit_ids = prepare_cell_data_for_onh(pathways, all_centroids, 0.8, 0.1)
print(f"\nCell data prepared: {len(cell_data)} cells")

if len(cell_data) > 0:
    print(f"Unit IDs: {unit_ids[:5]}...")
    print(f"First cell start_point: {cell_data[0][0]}")
    print(f"First cell n_centroids: {len(cell_data[0][1])}")
    
    # Try optimization manually to debug
    print("\nTrying find_optimal_onh manually...")
    
    from src.hdmea.features.ap_tracking.pathway_analysis import calculate_total_projection_error
    from scipy.optimize import minimize
    
    # Calculate initial guess
    start_rows = [sp[0] for sp, _ in cell_data]
    start_cols = [sp[1] for sp, _ in cell_data]
    mean_start_row = np.mean(start_rows)
    mean_start_col = np.mean(start_cols)
    print(f"Mean start point: ({mean_start_col:.1f}, {mean_start_row:.1f})")
    
    # Simple objective (no penalties)
    def objective(x):
        return calculate_total_projection_error((x[0], x[1]), cell_data)
    
    # Try a simple optimization without constraints
    initial = (mean_start_col, mean_start_row)
    print(f"Initial guess: {initial}")
    print(f"Initial error: {objective(initial):.4f}")
    
    # Try optimizing
    result = minimize(
        objective,
        x0=[initial[0], initial[1]],
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    print(f"\nUnconstrained optimization:")
    print(f"  Success: {result.success}")
    print(f"  Optimal: ({result.x[0]:.2f}, {result.x[1]:.2f})")
    print(f"  Error: {result.fun:.4f}")
    
    # Check distance from center
    dist = np.sqrt((result.x[0] - 33)**2 + (result.x[1] - 33)**2)
    print(f"  Distance from center: {dist:.2f} px")
    
    if dist > 98:
        print("  ** OUTSIDE 98px constraint - this is why it fails! **")
else:
    print("No cell data - checking why...")
    # Debug: check each pathway
    for uid, pw in list(pathways.items())[:10]:
        in_centroids = uid in all_centroids
        print(f"  {uid}: r²={pw.r_squared:.3f}, dir_valid={pw.direction_valid}, has_centroids={in_centroids}")

