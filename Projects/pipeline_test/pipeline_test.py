import sys
import zarr
import numpy as np

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from hdmea.pipeline import load_recording
from hdmea.pipeline import extract_features
from hdmea.pipeline import add_section_time

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


# Extract FRIF features
print("\n" + "=" * 60)
print("Extracting FRIF features...")
print("=" * 60)
extract_result = extract_features(
    zarr_path=result.zarr_path,
    features=["frif"],
    force=True
)

add_section_time(
    zarr_path=result.zarr_path,
    playlist_name="play_all_simple",
    repeats=3,
)