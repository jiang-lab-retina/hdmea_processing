"""
Create minimal H5 files with only essential data for quick loading.

Extracts:
- /units/{unit_id}/spike_times_sectioned/  (all contents)
- /units/{unit_id}/features/eimage_sta/geometry/center_col
- /units/{unit_id}/features/eimage_sta/geometry/center_row
- /units/{unit_id}/unit_meta/column
- /units/{unit_id}/unit_meta/row
- /metadata/acquisition_rate  (required for iprgc_2hz_QI calculation)
- /metadata/frame_timestamps  (required for frame-aligned firing rate calculation)

Saves to export/mini/{filename}_mini.h5
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def copy_dataset(src_file, dst_file, path):
    """Copy a dataset from source to destination, creating groups as needed."""
    if path not in src_file:
        return False
    
    # Create parent groups
    parent_path = "/".join(path.split("/")[:-1])
    if parent_path and parent_path not in dst_file:
        dst_file.create_group(parent_path)
    
    # Copy the dataset
    src_data = src_file[path]
    if isinstance(src_data, h5py.Dataset):
        # Handle scalar vs array datasets
        if src_data.shape == ():
            # Scalar dataset
            dst_file.create_dataset(path, data=src_data[()])
        else:
            # Array dataset
            dst_file.create_dataset(path, data=src_data[:])
        # Copy attributes
        for key, value in src_data.attrs.items():
            dst_file[path].attrs[key] = value
    return True


def copy_group_recursive(src_file, dst_file, path):
    """Recursively copy a group and all its contents."""
    if path not in src_file:
        return False
    
    src_item = src_file[path]
    
    if isinstance(src_item, h5py.Dataset):
        # Create parent groups if needed
        parent_path = "/".join(path.split("/")[:-1])
        if parent_path and parent_path not in dst_file:
            dst_file.create_group(parent_path)
        
        # Copy dataset - handle scalar vs array
        if src_item.shape == ():
            dst_file.create_dataset(path, data=src_item[()])
        else:
            dst_file.create_dataset(path, data=src_item[:])
        for key, value in src_item.attrs.items():
            dst_file[path].attrs[key] = value
    
    elif isinstance(src_item, h5py.Group):
        # Create group
        if path not in dst_file:
            dst_file.create_group(path)
        
        # Copy attributes
        for key, value in src_item.attrs.items():
            dst_file[path].attrs[key] = value
        
        # Recursively copy children
        for child_name in src_item.keys():
            child_path = f"{path}/{child_name}"
            copy_group_recursive(src_file, dst_file, child_path)
    
    return True


def create_mini_h5(src_path: Path, dst_path: Path):
    """Create a minimal H5 file with only essential data."""
    
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy root attributes
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        
        # Copy metadata required for firing rate calculation
        # acquisition_rate: needed for sample-based firing rate (iprgc_test)
        # frame_timestamps: needed for frame-aligned firing rate (other movies)
        if "metadata" in src:
            dst.create_group("metadata")
            for field in ["acquisition_rate", "frame_timestamps"]:
                field_path = f"metadata/{field}"
                if field_path in src:
                    copy_dataset(src, dst, field_path)
        
        # Get all unit IDs
        if "units" not in src:
            print(f"  No units found in {src_path.name}")
            return 0
        
        unit_ids = list(src["units"].keys())
        
        for unit_id in unit_ids:
            unit_path = f"units/{unit_id}"
            
            # 1. Copy spike_times_sectioned (entire group)
            sectioned_path = f"{unit_path}/spike_times_sectioned"
            if sectioned_path in src:
                copy_group_recursive(src, dst, sectioned_path)
            
            # 2. Copy eimage_sta geometry (center_col, center_row)
            geometry_base = f"{unit_path}/features/eimage_sta/geometry"
            for field in ["center_col", "center_row"]:
                field_path = f"{geometry_base}/{field}"
                if field_path in src:
                    copy_dataset(src, dst, field_path)
            
            # 3. Copy unit_meta (column, row)
            meta_base = f"{unit_path}/unit_meta"
            for field in ["column", "row"]:
                field_path = f"{meta_base}/{field}"
                if field_path in src:
                    copy_dataset(src, dst, field_path)
        
        return len(unit_ids)


def main():
    export_dir = Path(__file__).parent / "export"
    mini_dir = export_dir / "mini"
    mini_dir.mkdir(exist_ok=True)
    
    h5_files = sorted(export_dir.glob("*.h5"))
    
    print(f"Found {len(h5_files)} H5 files")
    print(f"Output directory: {mini_dir}")
    print("-" * 60)
    
    for h5_file in tqdm(h5_files, desc="Processing"):
        # Create output filename
        dst_name = h5_file.stem + "_mini.h5"
        dst_path = mini_dir / dst_name
        
        try:
            unit_count = create_mini_h5(h5_file, dst_path)
            src_size = h5_file.stat().st_size / (1024 * 1024)  # MB
            dst_size = dst_path.stat().st_size / (1024 * 1024)  # MB
            reduction = (1 - dst_size / src_size) * 100
            
            tqdm.write(f"{h5_file.name}: {unit_count} units, "
                      f"{src_size:.1f}MB -> {dst_size:.1f}MB ({reduction:.1f}% reduction)")
        except Exception as e:
            tqdm.write(f"{h5_file.name}: ERROR - {e}")
    
    print("-" * 60)
    print("Done!")
    
    # Summary
    total_src = sum(f.stat().st_size for f in h5_files) / (1024 * 1024 * 1024)
    total_dst = sum(f.stat().st_size for f in mini_dir.glob("*.h5")) / (1024 * 1024 * 1024)
    print(f"Total: {total_src:.2f}GB -> {total_dst:.2f}GB")


if __name__ == "__main__":
    main()
