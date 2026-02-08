"""
Test run for Step Change Analysis Pipeline
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from Projects.unified_special_pipeline.step_change_analysis.run_pipeline import (
    run_full_pipeline,
)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step Change Analysis Pipeline")
    print("=" * 60)
    
    # Run the pipeline with low_glucose dataset
    # Uses DATA_FOLDER from specific_config.py (N:/20251022_low_glucose)
    results = run_full_pipeline(
        data_folder=Path("N:/20251022_low_glucose"),
        overwrite=True,
    )
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"HDF5 files created: {len(results.get('hdf5_paths', []))}")
    
    if "grouped_path" in results:
        print(f"Grouped file: {results['grouped_path']}")
    
    if "analysis" in results:
        analysis = results["analysis"]
        if "on_response" in analysis:
            on_effect = analysis["on_response"].get("treatment_effect", {})
            on_pct = on_effect.get('effect_percent', 'N/A')
            if isinstance(on_pct, (int, float)):
                print(f"ON response effect: {on_pct:.1f}%")
            else:
                print(f"ON response effect: {on_pct}")
        if "off_response" in analysis:
            off_effect = analysis["off_response"].get("treatment_effect", {})
            off_pct = off_effect.get('effect_percent', 'N/A')
            if isinstance(off_pct, (int, float)):
                print(f"OFF response effect: {off_pct:.1f}%")
            else:
                print(f"OFF response effect: {off_pct}")
    
    print("\nTest complete!")
