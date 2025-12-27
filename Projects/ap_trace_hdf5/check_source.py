"""Check source files for potential issues."""
import h5py
from pathlib import Path
import numpy as np

files = [
    ('2024.05.22-12.43.26-Rec.h5', 'FAILING'),
    ('2024.06.18-11.16.04-Rec.h5', 'FAILING'),
    ('2024.03.07-10.12.28-Rec.h5', 'WORKING'),
]

source_dir = Path('M:/Python_Project/Data_Processing_2027/Projects/add_manual_label_cell_type/manual_label_20251225')

for fname, status in files:
    f = source_dir / fname
    if not f.exists():
        print(f"Not found: {fname}")
        continue
    
    print(f"\n{'='*60}")
    print(f"{status}: {fname}")
    print('='*60)
    
    with h5py.File(f, 'r') as h:
        # Check Center_xy metadata
        if 'metadata' in h and 'gsheet_row' in h['metadata']:
            gs = h['metadata/gsheet_row']
            if 'Center_xy' in gs:
                center = gs['Center_xy'][()]
                if isinstance(center, bytes):
                    center = center.decode()
                print(f"Center_xy: {center}")
        
        # Count RGCs
        rgc_count = 0
        for uid in h['units'].keys():
            label_path = f'units/{uid}/auto_label/axon_type'
            if label_path in h:
                cell_type = h[label_path][()]
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode()
                if cell_type.lower() == 'rgc':
                    rgc_count += 1
        print(f"RGC count: {rgc_count}")
        
        # Check a sample RGC's STA data
        for uid in h['units'].keys():
            label_path = f'units/{uid}/auto_label/axon_type'
            if label_path in h:
                cell_type = h[label_path][()]
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode()
                if cell_type.lower() == 'rgc':
                    # Check STA
                    sta_path = f'units/{uid}/features/eimage_sta/data'
                    if sta_path in h:
                        sta = h[sta_path][:]
                        print(f"\nSample RGC {uid}:")
                        print(f"  STA shape: {sta.shape}")
                        print(f"  STA range: [{sta.min():.4f}, {sta.max():.4f}]")
                        print(f"  STA std: {sta.std():.4f}")
                        
                        # Check if signal is too weak
                        if sta.std() < 0.01:
                            print("  WARNING: Very weak STA signal!")
                    break

