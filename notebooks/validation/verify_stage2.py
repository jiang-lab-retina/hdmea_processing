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
# # Stage 2 Validation: Feature Extraction
#
# This notebook validates that Stage 2 (Feature Extraction) produces correct features.

# %%
import zarr
from pathlib import Path

from hdmea.io.zarr_store import open_recording_zarr, list_units, list_features
from hdmea.features import FeatureRegistry

# %% [markdown]
# ## Configuration

# %%
# Path to Zarr to validate
ZARR_PATH = Path("../../artifacts/YOUR_DATASET.zarr")

# Expected features
EXPECTED_FEATURES = [
    "baseline_127",
    "step_up_5s_5i_3x",
]

# %% [markdown]
# ## Validate Features

# %%
def validate_stage2(zarr_path: Path, expected_features: list) -> dict:
    """Validate Stage 2 output."""
    results = {
        "zarr_exists": False,
        "features_extracted": [],
        "missing_features": [],
        "units_with_features": 0,
        "units_without_features": 0,
        "issues": [],
    }
    
    if not zarr_path.exists():
        results["issues"].append(f"Zarr not found: {zarr_path}")
        return results
    
    results["zarr_exists"] = True
    
    root = open_recording_zarr(zarr_path)
    units = list_units(root)
    
    # Check what features are extracted
    all_features = set()
    for unit_id in units:
        unit_features = list_features(root, unit_id)
        all_features.update(unit_features)
        
        if unit_features:
            results["units_with_features"] += 1
        else:
            results["units_without_features"] += 1
    
    results["features_extracted"] = sorted(all_features)
    
    # Check expected features
    for expected in expected_features:
        if expected not in all_features:
            results["missing_features"].append(expected)
            results["issues"].append(f"Missing expected feature: {expected}")
    
    # Validate feature metadata
    sample_unit = units[0] if units else None
    if sample_unit:
        for feature_name in list(all_features)[:3]:
            feature_group = root["units"][sample_unit]["features"][feature_name]
            
            if "extractor_version" not in feature_group.attrs:
                results["issues"].append(
                    f"Feature {feature_name} missing version metadata"
                )
    
    return results

# %%
# Run validation
if ZARR_PATH.exists():
    results = validate_stage2(ZARR_PATH, EXPECTED_FEATURES)
    
    print("=== Stage 2 Validation Results ===")
    print(f"  Features extracted: {results['features_extracted']}")
    print(f"  Units with features: {results['units_with_features']}")
    print(f"  Units without features: {results['units_without_features']}")
    
    if results["missing_features"]:
        print(f"\nMissing features: {results['missing_features']}")
    
    if results["issues"]:
        print("\nIssues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")
    else:
        print("\n✓ All checks passed!")
else:
    print(f"Configure ZARR_PATH and run again. Current: {ZARR_PATH}")

# %% [markdown]
# ## List Available Features

# %%
print("Registered feature extractors:")
for name in FeatureRegistry.list_all():
    metadata = FeatureRegistry.get_metadata(name)
    print(f"  - {name} (v{metadata['version']}, {metadata['runtime_class']})")

