"""Check the HDF5 structure of the failing files."""
import h5py
from pathlib import Path

files = [
    'Projects/ap_trace_hdf5/export/2024.05.22-12.43.26-Rec.h5',
    'Projects/ap_trace_hdf5/export/2024.03.07-10.12.28-Rec.h5',  # Working file for comparison
]

for fpath in files:
    f = Path(fpath)
    if not f.exists():
        print(f"Not found: {f}")
        continue
    
    print(f"\n{'='*60}")
    print(f"File: {f.name}")
    print('='*60)
    
    with h5py.File(f, 'r') as h:
        print(f"Total units: {len(h['units'].keys())}")
        
        # Find first RGC unit
        for uid in list(h['units'].keys())[:5]:
            ap = f'units/{uid}/features/ap_tracking'
            print(f"\nChecking {uid}: ap_tracking exists = {ap in h}")
            if ap not in h:
                continue
            
            print(f"\n{uid}:")
            print(f"  ap_tracking keys: {list(h[ap].keys())}")
            
            if 'post_processed' in h[ap]:
                pp = h[f'{ap}/post_processed']
                print(f"  post_processed keys: {list(pp.keys())}")
                if 'axon_centroids' in pp:
                    centroids = pp['axon_centroids'][:]
                    print(f"    axon_centroids: {centroids.shape}")
            
            if 'ap_pathway' in h[ap]:
                pw = h[f'{ap}/ap_pathway']
                print(f"  ap_pathway keys: {list(pw.keys())}")
            
            if 'prediction' in h[ap]:
                pred = h[f'{ap}/prediction'][:]
                print(f"  prediction: shape={pred.shape}, max={pred.max():.4f}")
            
            break  # Just check first unit
        
        # Check metadata
        meta_path = 'metadata/ap_tracking/all_ap_intersection'
        if meta_path in h:
            int_grp = h[meta_path]
            print(f"\nONH metadata:")
            print(f"  x: {int_grp['x'][()]:.2f}")
            print(f"  y: {int_grp['y'][()]:.2f}")
            method = int_grp['method'][()]
            if isinstance(method, bytes):
                method = method.decode()
            print(f"  method: {method}")
            if 'n_cells_used' in int_grp:
                print(f"  n_cells_used: {int_grp['n_cells_used'][()]}")
            if 'outlier_unit_ids' in int_grp:
                outliers = [x.decode() for x in int_grp['outlier_unit_ids'][:]]
                print(f"  outliers: {len(outliers)}")
        else:
            print("\nNo ONH metadata found")

