# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   hdmea:
#     stage: validation
#     status: stable
#     flows: []
# ---

# %% [markdown]
# # Stage 1 Validation: Data Loading
#
# This notebook validates that Stage 1 (Data Loading) produces correct Zarr artifacts.

# %%
import zarr
from pathlib import Path

from hdmea.io.zarr_store import open_recording_zarr, list_units, get_stage1_status

# %% [markdown]
# ## Configuration

# %%
# Path to Zarr to validate
ZARR_PATH = Path("../../artifacts/YOUR_DATASET.zarr")

# %% [markdown]
# ## Load and Validate

# %%
def validate_stage1(zarr_path: Path) -> dict:
    """Validate Stage 1 output."""
    results = {
        "zarr_exists": False,
        "stage1_completed": False,
        "has_units": False,
        "has_stimulus": False,
        "has_metadata": False,
        "unit_count": 0,
        "issues": [],
    }
    
    if not zarr_path.exists():
        results["issues"].append(f"Zarr not found: {zarr_path}")
        return results
    
    results["zarr_exists"] = True
    
    root = open_recording_zarr(zarr_path)
    
    # Check stage1 status
    status = get_stage1_status(root)
    results["stage1_completed"] = status["completed"]
    
    if not status["completed"]:
        results["issues"].append("Stage 1 not marked as complete")
    
    # Check units
    units = list_units(root)
    results["unit_count"] = len(units)
    results["has_units"] = len(units) > 0
    
    if not results["has_units"]:
        results["issues"].append("No units found")
    
    # Check stimulus
    results["has_stimulus"] = "stimulus" in root
    if not results["has_stimulus"]:
        results["issues"].append("No stimulus group found")
    
    # Check metadata
    results["has_metadata"] = "metadata" in root
    
    # Validate unit structure
    for unit_id in units[:5]:  # Check first 5
        unit = root["units"][unit_id]
        if "spike_times" not in unit:
            results["issues"].append(f"Unit {unit_id} missing spike_times")
        if "features" not in unit:
            results["issues"].append(f"Unit {unit_id} missing features group")
    
    return results

# %%
# Run validation
if ZARR_PATH.exists():
    results = validate_stage1(ZARR_PATH)
    
    print("=== Stage 1 Validation Results ===")
    for key, value in results.items():
        if key != "issues":
            print(f"  {key}: {value}")
    
    if results["issues"]:
        print("\nIssues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")
    else:
        print("\n✓ All checks passed!")
else:
    print(f"Configure ZARR_PATH and run again. Current: {ZARR_PATH}")

