#!/usr/bin/env python
"""
Master Validation Plot Runner.

Runs all validation plots for a given HDF5 file:
- 01_eimage_sta: STA visualization
- 02_geometry: Soma + RF geometry plots
- 03_ap_tracking: AP pathway + ONH visualization
- 04_dsgc: Direction sectioning plots

Usage:
    python run_all.py <hdf5_path> [--output-dir <path>] [--skip <step>]
    
Example:
    python run_all.py ../test_output/2024.08.08-10.40.20-Rec_final.h5
    python run_all.py file.h5 --output-dir ./my_plots
    python run_all.py file.h5 --skip dsgc  # Skip DSGC plots
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directories to path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Import validation modules dynamically (folder names start with numbers)
import importlib.util

def _import_module(folder_name, module_name):
    """Import module from numbered folder."""
    module_path = _SCRIPT_DIR / folder_name / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Lazy imports (will be loaded when run_all_validations is called)
_modules_loaded = False
run_eimage_sta = None
run_geometry = None
run_ap_tracking = None
run_dsgc = None

def _load_modules():
    global _modules_loaded, run_eimage_sta, run_geometry, run_ap_tracking, run_dsgc
    if _modules_loaded:
        return
    
    eimage_sta_module = _import_module("01_eimage_sta", "plot_eimage_sta")
    geometry_module = _import_module("02_geometry", "plot_geometry")
    ap_tracking_module = _import_module("03_ap_tracking", "plot_ap_tracking")
    dsgc_module = _import_module("04_dsgc", "plot_dsgc")
    
    run_eimage_sta = eimage_sta_module.run_validation
    run_geometry = geometry_module.run_validation
    run_ap_tracking = ap_tracking_module.run_validation
    run_dsgc = dsgc_module.run_validation
    
    _modules_loaded = True


# =============================================================================
# Configuration
# =============================================================================

def get_steps():
    """Get steps configuration with loaded runners."""
    _load_modules()
    return {
        "eimage_sta": {
            "name": "01_eimage_sta",
            "description": "STA Visualization",
            "runner": run_eimage_sta,
        },
        "geometry": {
            "name": "02_geometry", 
            "description": "Soma + RF Geometry",
            "runner": run_geometry,
        },
        "ap_tracking": {
            "name": "03_ap_tracking",
            "description": "AP Tracking + ONH",
            "runner": run_ap_tracking,
        },
        "dsgc": {
            "name": "04_dsgc",
            "description": "DSGC Direction Sectioning",
            "runner": run_dsgc,
        },
    }

# Step names for argparse (before modules are loaded)
STEP_NAMES = ["eimage_sta", "geometry", "ap_tracking", "dsgc"]


# =============================================================================
# Main Runner
# =============================================================================

def run_all_validations(
    hdf5_path: Path,
    output_dir: Path = None,
    skip_steps: list = None,
):
    """Run all validation plots."""
    
    # Load modules
    steps = get_steps()
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _SCRIPT_DIR / "output" / f"{hdf5_path.stem}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_steps = skip_steps or []
    
    print("=" * 70)
    print("UNIFIED PIPELINE VALIDATION PLOTS")
    print("=" * 70)
    print(f"Input:    {hdf5_path}")
    print(f"Output:   {output_dir}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    total_start = time.time()
    
    for step_key, step_info in steps.items():
        if step_key in skip_steps:
            print(f"\n[SKIP] {step_info['name']}: {step_info['description']}")
            results[step_key] = "skipped"
            continue
        
        print(f"\n[RUN] {step_info['name']}: {step_info['description']}")
        print("-" * 50)
        
        step_output = output_dir / step_info["name"]
        step_start = time.time()
        
        try:
            step_info["runner"](hdf5_path, step_output)
            elapsed = time.time() - step_start
            results[step_key] = f"success ({elapsed:.1f}s)"
            print(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - step_start
            results[step_key] = f"failed: {e}"
            print(f"  FAILED: {e}")
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Output:   {output_dir}")
    print(f"Duration: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("\nResults:")
    for step_key, result in results.items():
        status = "[+]" if "success" in result else ("[-]" if "skip" in result else "[!]")
        print(f"  {status} {steps[step_key]['name']}: {result}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run all validation plots for unified pipeline output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all.py ../test_output/2024.08.08-10.40.20-Rec_final.h5
    python run_all.py file.h5 --output-dir ./plots
    python run_all.py file.h5 --skip dsgc ap_tracking
        """
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 file")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: auto-generated)")
    parser.add_argument("--skip", nargs="*", choices=STEP_NAMES,
                       help="Steps to skip (eimage_sta, geometry, ap_tracking, dsgc)")
    
    args = parser.parse_args()
    
    if not args.hdf5_path.exists():
        print(f"Error: File not found: {args.hdf5_path}")
        return 1
    
    run_all_validations(
        args.hdf5_path,
        args.output_dir,
        args.skip,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

