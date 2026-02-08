"""Debug script for alignment issue"""
import h5py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Projects.unified_special_pipeline.step_change_analysis.data_loader import (
    load_recording_from_hdf5,
    load_cmcr_cmtr_data,
    get_high_quality_units,
)

# First, check raw CMCR/CMTR loading
cmcr_path = Path("O:/20251001_HttrB_agonist/2025.10.01-09.45.32-Rec.cmcr")
cmtr_path = Path("O:/20251001_HttrB_agonist/2025.10.01-09.45.32-Rec-.cmtr")

print("Loading raw CMCR/CMTR data...")
raw_data = load_cmcr_cmtr_data(cmcr_path, cmtr_path)

print(f"\nRaw data unit keys: {list(raw_data['units'].keys())[:5]}")
for uid in list(raw_data['units'].keys())[:3]:
    unit = raw_data['units'][uid]
    print(f"  {uid} keys: {list(unit.keys())}")
    print(f"    row={unit.get('row', 'N/A')}, col={unit.get('col', 'N/A')}")

# Load from HDF5
hdf5_path = Path(__file__).parent / "output" / "2025.10.01-09.45.32-Rec.h5"
print(f"\nLoading from HDF5: {hdf5_path}")

data = load_recording_from_hdf5(hdf5_path)

print(f"\nTotal units: {len(data['units'])}")

# Check a few units
for uid in list(data['units'].keys())[:3]:
    unit = data['units'][uid]
    print(f"  {uid}: keys={list(unit.keys())}")
    qi = unit.get('quality_index', None)
    row = unit.get('row', None)
    col = unit.get('col', None)
    print(f"    qi={qi}, row={row}, col={col}")

# Check get_high_quality_units
hq_units = get_high_quality_units(data, threshold=0.01)
print(f"\nHigh quality units (threshold=0.01): {len(hq_units)}")
