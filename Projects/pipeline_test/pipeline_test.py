import sys
import zarr
import numpy as np

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from hdmea.pipeline import load_recording
from hdmea.pipeline import extract_features
from hdmea.pipeline import add_section_time
from hdmea.io.section_time import add_section_time_analog


print("=" * 60)
print("Running pipeline test...")
print("=" * 60)

# # Provide external paths to raw files
# result = load_recording(
#     cmcr_path="O:\\20250410\\set6\\2025.04.10-11.12.57-Rec.cmcr",
#     cmtr_path="O:\\20250410\\set6\\2025.04.10-11.12.57-Rec-.cmtr",
#     dataset_id="JIANG009_2025-04-10",
#     allow_overwrite=True
# )


# # Extract FRIF features
# print("\n" + "=" * 60)
# print("Extracting FRIF features...")
# print("=" * 60)
# extract_result = extract_features(
#     zarr_path=result.zarr_path,
#     features=["frif"],
#     force=True
# )


add_section_time(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    # zarr_path=result.zarr_path,
    playlist_name="play_optimization_set6_ipRGC_manual",
    repeats=1,
    force=True,
)

# Detect ipRGC stimulus onsets from recorded light signal
success = add_section_time_analog(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    threshold_value=3e4,  # Must inspect signal to determine
    movie_name="iprgc_test",
    plot_duration=120.0,  # 2 minute windows
    repeat=2,  # Use first 3 trials only
    force=True,
)