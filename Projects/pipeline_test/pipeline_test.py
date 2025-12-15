import sys
import zarr
import numpy as np

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from hdmea.pipeline import load_recording
from hdmea.pipeline import extract_features

print("=" * 60)
print("Running pipeline test...")
print("=" * 60)

# Provide external paths to raw files
result = load_recording(
    cmcr_path="M:\\20231207_WT_Blocking\\2023.12.07-09.37.02-Rec.cmcr",
    cmtr_path="M:\\20231207_WT_Blocking\\2023.12.07-09.37.02-Rec-.cmtr",
    dataset_id="TESTPIPELINE01_2023-12-07",
    allow_overwrite=True
)


# Open the Zarr archive and check metadata
root = zarr.open(str(result.zarr_path), mode="r")

# Show the tree structure
print("\nZarr tree structure:")
print(root.tree())


# After loading recording (which detects frame_timestamps)
result = extract_features(
    zarr_path=result.zarr_path,
    features=["frif"]
)