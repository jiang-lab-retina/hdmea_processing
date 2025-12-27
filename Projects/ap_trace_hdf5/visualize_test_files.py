"""Visualize AP tracking results for test files."""
import sys
sys.path.insert(0, 'M:/Python_Project/Data_Processing_2027')
from pathlib import Path

# Import visualization functions
from Projects.ap_trace_hdf5.visualize_intersection import (
    load_pathway_data, visualize_intersection
)

# Test files to visualize
test_files = [
    '2024.03.07-10.12.28-Rec.h5',
    '2024.09.17-10.09.28-Rec.h5',
    '2024.05.22-12.43.26-Rec.h5',
    '2024.06.18-11.16.04-Rec.h5',
]

export_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/export')
plots_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/ap_trace_hdf5/plots')

for fname in test_files:
    hdf5_path = export_dir / fname
    if not hdf5_path.exists():
        print(f"Skipping {fname}: file not found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {fname}")
    print('='*60)
    
    # Load data to show summary
    pathways, onh_data = load_pathway_data(hdf5_path)
    
    if onh_data and onh_data.x is not None:
        print(f"  ONH: ({onh_data.x:.2f}, {onh_data.y:.2f})")
        print(f"  Method: {onh_data.method}")
        print(f"  Cells: {onh_data.n_cells_used}")
        if onh_data.consensus_direction is not None:
            print(f"  Consensus direction: {onh_data.consensus_direction:.1f} deg")
    else:
        print("  No valid ONH data")
    
    # Create output directory
    rec_name = hdf5_path.stem
    output_dir = plots_dir / rec_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    output_file = visualize_intersection(
        hdf5_path,
        output_dir,
        r2_threshold=0.8,
        direction_tolerance=90.0,
    )
    print(f"  Output: {output_file}")

print("\n" + "="*60)
print("Visualization complete!")

