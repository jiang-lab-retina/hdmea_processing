"""
Validation script for analog section time detection.

Tests the add_section_time_analog() function with real Zarr data to verify:
1. Function imports correctly
2. Required data exists in Zarr
3. Peak detection produces reasonable results
4. Section times are written correctly

Usage:
    python scripts/validate_analog_section_time.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import zarr
from hdmea.io.section_time import add_section_time_analog


def validate_zarr_structure(zarr_path: Path) -> bool:
    """Check if Zarr has required structure for analog detection."""
    print(f"\n{'='*60}")
    print(f"Validating Zarr structure: {zarr_path.name}")
    print(f"{'='*60}\n")
    
    root = zarr.open(str(zarr_path), mode='r')
    
    checks = {
        "metadata group": "metadata" in root,
        "acquisition_rate": "metadata" in root and "acquisition_rate" in root["metadata"],
        "frame_timestamps": "metadata" in root and "frame_timestamps" in root["metadata"],
        "stimulus group": "stimulus" in root,
        "light_reference": "stimulus" in root and "light_reference" in root["stimulus"],
        "raw_ch1": ("stimulus" in root and 
                   "light_reference" in root["stimulus"] and 
                   "raw_ch1" in root["stimulus"]["light_reference"]),
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        # Print data info
        print("\nData dimensions:")
        acquisition_rate = float(np.array(root["metadata"]["acquisition_rate"]).flat[0])
        n_frames = root["metadata"]["frame_timestamps"].shape[0]
        raw_ch1 = root["stimulus"]["light_reference"]["raw_ch1"]
        n_samples = raw_ch1.shape[0]
        duration_s = n_samples / acquisition_rate
        
        print(f"  - Acquisition rate: {acquisition_rate} Hz")
        print(f"  - Raw samples: {n_samples:,} ({duration_s:.1f} seconds)")
        print(f"  - Frame timestamps: {n_frames:,} frames")
        print(f"  - Display rate: ~{n_frames/duration_s:.1f} Hz")
        
        # Show raw_ch1 signal stats
        preview_size = min(10000, raw_ch1.shape[0])
        raw_ch1_arr = np.array(raw_ch1[:preview_size])
        print(f"\nRaw_ch1 signal preview (first {len(raw_ch1_arr)} samples):")
        print(f"  - Min: {raw_ch1_arr.min():.2e}")
        print(f"  - Max: {raw_ch1_arr.max():.2e}")
        print(f"  - Mean: {raw_ch1_arr.mean():.2e}")
        print(f"  - Std: {raw_ch1_arr.std():.2e}")
        
        # Analyze derivative for threshold recommendation
        derivative = np.diff(raw_ch1_arr.astype(np.float64))
        if len(derivative) > 0:
            print(f"\nDerivative analysis (for threshold selection):")
            print(f"  - Max positive derivative: {derivative.max():.2e}")
            print(f"  - 99.9 percentile: {np.percentile(derivative, 99.9):.2e}")
            print(f"  - 99 percentile: {np.percentile(derivative, 99):.2e}")
            print(f"  - 95 percentile: {np.percentile(derivative, 95):.2e}")
    
    return all_passed


def test_analog_detection(zarr_path: Path, threshold: float, movie_name: str = "validation_test"):
    """Test analog section time detection with given threshold."""
    print(f"\n{'='*60}")
    print(f"Testing analog detection")
    print(f"{'='*60}\n")
    print(f"  Threshold: {threshold:.2e}")
    print(f"  Movie name: {movie_name}")
    print(f"  Plot duration: 120.0 seconds (default)")
    
    try:
        success = add_section_time_analog(
            zarr_path=zarr_path,
            threshold_value=threshold,
            movie_name=movie_name,
            plot_duration=120.0,
            force=True,  # Allow overwrite for testing
        )
        
        if success:
            print(f"\n[SUCCESS] Detection succeeded!")
            
            # Read back the results
            root = zarr.open(str(zarr_path), mode='r')
            if "section_time" in root["stimulus"] and movie_name in root["stimulus"]["section_time"]:
                section_time = np.array(root["stimulus"]["section_time"][movie_name])
                
                print(f"\nResults:")
                print(f"  - Detected {len(section_time)} stimulus trials")
                print(f"  - Section shape: {section_time.shape}")
                print(f"  - Frame range: [{section_time[:, 0].min()}, {section_time[:, 1].max()}]")
                
                if len(section_time) > 0:
                    print(f"\nFirst 5 trials:")
                    for i, (start, end) in enumerate(section_time[:5]):
                        duration_frames = end - start
                        print(f"    Trial {i+1}: frames {start} to {end} (duration: {duration_frames} frames)")
                
                return True
            else:
                print(f"[WARNING] Detection succeeded but section_time/{movie_name} not found in Zarr")
                return False
        else:
            print(f"\n[FAIL] Detection returned False (no peaks detected)")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] Detection failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run validation tests."""
    print("\n" + "="*60)
    print("Analog Section Time Detection - Validation")
    print("="*60)
    
    # Test with JIANG009_2025-04-10.zarr
    zarr_path = Path("artifacts/JIANG009_2025-04-10.zarr")
    
    if not zarr_path.exists():
        print(f"\n[FAIL] Zarr not found: {zarr_path}")
        print("\nPlease ensure the Zarr file exists or update the path in this script.")
        return 1
    
    # Step 1: Validate structure
    if not validate_zarr_structure(zarr_path):
        print("\n[FAIL] Zarr structure validation failed")
        return 1
    
    print("\n[PASS] Zarr structure validation passed")
    
    # Step 2: Test detection with reasonable threshold
    # Based on the derivative analysis, use a threshold around 99th percentile
    # This is conservative and should detect clear transitions
    test_threshold = 1e6  # Adjust based on derivative analysis output
    
    print(f"\n{'='*60}")
    print("Running detection test...")
    print(f"{'='*60}")
    print("\nNote: If no peaks are detected, try:")
    print("  1. Lower the threshold (e.g., 1e5 instead of 1e6)")
    print("  2. Inspect raw_ch1 signal to understand transitions")
    print("  3. Check derivative max value from structure validation above")
    
    success = test_analog_detection(zarr_path, test_threshold)
    
    if success:
        print(f"\n{'='*60}")
        print("[PASS] Validation PASSED")
        print(f"{'='*60}")
        print("\nThe analog section time detection is working correctly!")
        print(f"Section times written to: stimulus/section_time/validation_test")
        return 0
    else:
        print(f"\n{'='*60}")
        print("[WARNING] Validation completed with issues")
        print(f"{'='*60}")
        print("\nPossible reasons:")
        print("  - Threshold may need adjustment for this recording")
        print("  - Signal may not have clear step transitions")
        print("  - Try different threshold values based on derivative analysis")
        return 1


if __name__ == "__main__":
    sys.exit(main())

