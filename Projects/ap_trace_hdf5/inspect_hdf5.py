"""Simple script to inspect HDF5 file structure."""
import h5py

h5_path = "../load_gsheet/export/2024.02.26-11.44.42-Rec.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: {obj.shape} {obj.dtype}")
    else:
        print(f"  {name}/")

print("=== HDF5 File Structure ===")
with h5py.File(h5_path, "r") as f:
    print("\nRoot groups:", list(f.keys()))
    print("\nTotal units:", len(f['units']))
    
    # Check first unit
    unit = f['units/unit_001']
    print("\nUnit_001 structure:")
    for key in unit.keys():
        item = unit[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: {item.shape} {item.dtype}")
        else:
            print(f"  {key}/")
            for subkey in list(item.keys())[:5]:
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset):
                    print(f"    {subkey}: {subitem.shape} {subitem.dtype}")
                else:
                    print(f"    {subkey}/")
                    for subsubkey in list(subitem.keys())[:3]:
                        subsubitem = subitem[subsubkey]
                        if isinstance(subsubitem, h5py.Dataset):
                            print(f"      {subsubkey}: {subsubitem.shape} {subsubitem.dtype}")
                        else:
                            print(f"      {subsubkey}/")
    
    # Check if there's STA data
    print("\n=== Looking for STA data ===")
    if 'features' in unit:
        features = unit['features']
        print("Feature groups:", list(features.keys())[:10])
        
        # Look for noise/sta
        for feat_name in features.keys():
            if 'noise' in feat_name.lower() or 'sta' in feat_name.lower():
                print(f"\nFound: {feat_name}")
                feat = features[feat_name]
                if isinstance(feat, h5py.Group):
                    for k in feat.keys():
                        item = feat[k]
                        if isinstance(item, h5py.Dataset):
                            print(f"  {k}: {item.shape} {item.dtype}")
                        else:
                            print(f"  {k}/")

print("\n=== Metadata ===")
with h5py.File(h5_path, "r") as f:
    meta = f['metadata']
    for key in list(meta.keys())[:10]:
        item = meta[key]
        if isinstance(item, h5py.Dataset):
            if item.size == 1:
                print(f"  {key}: {item[()]}")
            else:
                print(f"  {key}: {item.shape}")
        else:
            print(f"  {key}/")

