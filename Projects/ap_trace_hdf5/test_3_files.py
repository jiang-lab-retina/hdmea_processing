"""Test AP tracking on 3 different files."""
import sys
sys.path.insert(0, 'M:/Python_Project/Data_Processing_2027')
from pathlib import Path
import shutil
import h5py
import numpy as np

# Test files - pick files that likely have more RGCs
test_files = [
    '2024.03.07-10.12.28-Rec.h5',    # Different retina
    '2024.04.11-11.22.37-Rec.h5',    # Different retina  
    '2024.09.17-10.09.28-Rec.h5',    # Different retina
    '2024.05.22-12.43.26-Rec.h5',    # Additional test file
    '2024.06.18-11.16.04-Rec.h5',    # Additional test file
]

source_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/add_manual_label_cell_type/manual_label_20251225')
output_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')
model_path = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth')

from src.hdmea.features.ap_tracking.core import compute_ap_tracking

for fname in test_files:
    print("\n" + "=" * 60)
    print(f"Processing: {fname}")
    print("=" * 60)
    
    src = source_dir / fname
    dst = output_dir / fname
    
    if not src.exists():
        print("  ERROR: Source file not found")
        continue
        
    # Copy source to output
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    
    # Check RGC count
    with h5py.File(dst, 'r') as f:
        units = list(f['units'].keys())
        rgc_count = 0
        for uid in units:
            label_path = f'units/{uid}/auto_label/axon_type'
            if label_path in f:
                cell_type = f[label_path][()]
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode()
                if cell_type.lower() == 'rgc':
                    rgc_count += 1
    print(f"  Total units: {len(units)}, RGCs: {rgc_count}")
    
    # Run AP tracking
    result = compute_ap_tracking(
        dst, model_path,
        r2_threshold=0.8,
        centroid_start_frame=10,
        max_displacement=100,
        max_displacement_post=5.0,
        centroid_exclude_fraction=0.1,
    )
    
    # Check results
    with h5py.File(dst, 'r') as f:
        if 'metadata/ap_tracking/all_ap_intersection' in f:
            grp = f['metadata/ap_tracking/all_ap_intersection']
            
            # Check if we have valid ONH data
            if 'x' not in grp:
                print("  No valid ONH coordinates found")
                continue
                
            x = grp['x'][()]
            y = grp['y'][()]
            method = grp['method'][()]
            if isinstance(method, bytes):
                method = method.decode()
            rmse = grp['rmse'][()] if 'rmse' in grp else 0.0
            n_cells = grp['n_cells_used'][()] if 'n_cells_used' in grp else 0
            print(f"  ONH: ({x:.2f}, {y:.2f}), RMSE={rmse:.4f}, method={method}, cells={n_cells}")
            
            # Check distance from center
            dist = np.sqrt((x-33)**2 + (y-33)**2)
            print(f"  Distance from center: {dist:.2f} px (max allowed: 98)")
            
            # Check consensus direction
            if 'consensus_direction' in grp:
                cons_dir = grp['consensus_direction'][()]
                if cons_dir is not None and not np.isnan(cons_dir):
                    print(f"  Consensus direction: {cons_dir:.1f} deg")
        else:
            print("  No ONH result found")

print("\n" + "=" * 60)
print("All tests complete!")

