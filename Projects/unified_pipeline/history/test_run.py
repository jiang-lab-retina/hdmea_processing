#!/usr/bin/env python
"""
Quick test script to validate the unified pipeline.

This script:
1. Loads the reference HDF5 file to analyze its structure
2. Compares with what the pipeline would produce

Run from project root:
    python Projects/unified_pipeline/test_run.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import h5py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_hdf5_structure(path: Path, max_depth: int = 3) -> None:
    """Print the structure of an HDF5 file."""
    print(f"\n{'='*70}")
    print(f"HDF5 Structure: {path.name}")
    print(f"{'='*70}")
    
    def print_item(name, obj, depth=0):
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}/ ({len(obj)} items)")
            if depth < max_depth:
                for key in list(obj.keys())[:10]:  # Limit to first 10
                    print_item(key, obj[key], depth + 1)
                if len(obj) > 10:
                    print(f"{indent}  ... and {len(obj) - 10} more")
        elif isinstance(obj, h5py.Dataset):
            shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
            dtype_str = f"dtype={obj.dtype}"
            print(f"{indent}📄 {name} [{shape_str}, {dtype_str}]")
    
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            print_item(key, f[key], 0)


def analyze_reference_file() -> None:
    """Analyze the reference HDF5 file structure."""
    ref_path = Path("Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5")
    
    if not ref_path.exists():
        print(f"❌ Reference file not found: {ref_path}")
        return
    
    print(f"✅ Reference file found: {ref_path}")
    print(f"   Size: {ref_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print_hdf5_structure(ref_path)
    
    # Analyze units in detail
    with h5py.File(ref_path, 'r') as f:
        if 'units' in f:
            units = f['units']
            print(f"\n{'='*70}")
            print(f"Unit Analysis")
            print(f"{'='*70}")
            print(f"Total units: {len(units)}")
            
            # Check first unit's features
            first_unit = list(units.keys())[0]
            unit = units[first_unit]
            print(f"\nFirst unit: {first_unit}")
            
            if 'features' in unit:
                print(f"  Features: {list(unit['features'].keys())}")
            
            if 'auto_label' in unit:
                print(f"  auto_label: {list(unit['auto_label'].keys())}")
            
            if 'unit_meta' in unit:
                print(f"  unit_meta: exists")
        
        if 'metadata' in f:
            meta = f['metadata']
            print(f"\n{'='*70}")
            print(f"Metadata Groups")
            print(f"{'='*70}")
            for key in meta.keys():
                print(f"  📁 {key}")
        
        if 'pipeline' in f:
            pipeline = f['pipeline']
            print(f"\n{'='*70}")
            print(f"Pipeline Info")
            print(f"{'='*70}")
            if 'session_info' in pipeline:
                session_info = pipeline['session_info']
                if 'completed_steps' in session_info:
                    steps = session_info['completed_steps'][:]
                    print(f"  Completed steps:")
                    for step in steps:
                        if isinstance(step, bytes):
                            step = step.decode('utf-8')
                        print(f"    ✓ {step}")


def test_universal_loader() -> None:
    """Test that the universal loader works correctly."""
    ref_path = Path("Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5")
    
    if not ref_path.exists():
        print(f"❌ Cannot test loader - reference file not found")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing Universal Loader")
    print(f"{'='*70}")
    
    try:
        from hdmea.pipeline import load_session_from_hdf5
        
        print("Loading session from HDF5...")
        session = load_session_from_hdf5(ref_path)
        
        print(f"✅ Session loaded successfully!")
        print(f"   Dataset ID: {session.dataset_id}")
        print(f"   Unit count: {session.unit_count}")
        print(f"   Completed steps: {len(session.completed_steps)}")
        
        if session.completed_steps:
            print("   Steps:")
            for step in sorted(session.completed_steps):
                print(f"     ✓ {step}")
        
        # Check if features are loaded
        if session.units:
            first_unit_id = list(session.units.keys())[0]
            first_unit = session.units[first_unit_id]
            print(f"\n   First unit ({first_unit_id}):")
            print(f"     Keys: {list(first_unit.keys())[:10]}")
            if 'features' in first_unit:
                print(f"     Features: {list(first_unit['features'].keys())}")
        
    except Exception as e:
        print(f"❌ Error loading session: {e}")
        import traceback
        traceback.print_exc()


def test_save_overwrite_protection(tmp_dir: Path = None) -> None:
    """Test that save() with overwrite=False raises error."""
    print(f"\n{'='*70}")
    print(f"Testing Save Overwrite Protection")
    print(f"{'='*70}")
    
    ref_path = Path("Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5")
    
    if not ref_path.exists():
        print(f"❌ Cannot test - reference file not found")
        return
    
    try:
        from hdmea.pipeline import load_session_from_hdf5
        import tempfile
        
        # Load session
        session = load_session_from_hdf5(ref_path)
        
        # Create temp dir
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp())
        
        output_path = tmp_dir / "test_output.h5"
        
        # First save should work
        print(f"First save to: {output_path}")
        session.save(output_path=output_path, overwrite=True)
        print(f"✅ First save succeeded")
        
        # Second save without overwrite should fail
        print(f"Second save without overwrite=True...")
        session2 = load_session_from_hdf5(output_path)
        
        try:
            session2.save(output_path=output_path, overwrite=False)
            print(f"❌ ERROR: Should have raised FileExistsError!")
        except FileExistsError as e:
            print(f"✅ Correctly raised FileExistsError: {e}")
        
        # Clean up
        output_path.unlink()
        tmp_dir.rmdir()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("="*70)
    print("UNIFIED PIPELINE TEST")
    print("="*70)
    
    # Test 1: Analyze reference file
    analyze_reference_file()
    
    # Test 2: Test universal loader
    test_universal_loader()
    
    # Test 3: Test save protection
    test_save_overwrite_protection()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

