"""
Pipeline Session Test Script

Tests the deferred save mode with various pipeline functions.
"""
from hdmea.pipeline import create_session, load_recording, load_recording_with_eimage_sta, extract_features
from hdmea.io import add_section_time, section_spike_times
from hdmea.io.section_time import add_section_time_analog
from hdmea.features import compute_sta

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# =============================================================================
# Example 1: Deferred mode with load_recording (recommended workflow)
# =============================================================================

# 1. Create a session
session = create_session(dataset_id="2025.04.10-11.12.57-Rec")

# 2. Run pipeline steps (data accumulates in memory)
# session = load_recording(
#     cmcr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr",
#     cmtr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr",
#     session=session,
# )

session = load_recording_with_eimage_sta(
    cmcr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr",
    cmtr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr",
    duration_s=120.0,
    spike_limit=10000,
    window_range=(-10, 40),
    session=session,  # Pass session for deferred mode
    unit_ids=["2", "7", "9"],
    # load_unit_meta=True,  # Default: loads extended CMTR unit metadata -> units/*/unit_meta
    # load_sys_meta=True,   # Default: loads file metadata -> metadata/cmcr_meta, cmtr_meta
)


# Add section time using playlist (deferred)
session = add_section_time(
    playlist_name="play_optimization_set6_ipRGC_manual",
    session=session,
)

# Section spike times (deferred)
session = section_spike_times(
    pad_margin=(0.0, 0.0),
    session=session,
)

# Compute STA (deferred) - requires section_spike_times first
session = compute_sta(
    cover_range=(-60, 0),
    session=session,
)

# # Feature extraction in deferred mode - now fully supported!
# # FRIF extractor works with session data via DictAdapter
# session = extract_features(
#     features=["frif"],
#     session=session,
# )

# # Verify FRIF was extracted in session
# units_with_frif = sum(
#     1 for u in session.units.values()
#     if "features" in u and "frif" in u.get("features", {})
# )
# print(f"\nFRIF extraction in session mode: {units_with_frif}/{session.unit_count} units")

# 3. Save once at the end
hdf5_path = session.save()
print(f"\n{'='*60}")
print(f"Session saved to: {hdf5_path}")
print(f"Completed steps: {session.completed_steps}")
print(f"Warnings: {len(session.warnings)}")
print(f"{'='*60}\n")


# =============================================================================
# Example 2: Verify features are saved in HDF5
# =============================================================================
print("\nVerifying features in saved HDF5...")
import h5py
with h5py.File(hdf5_path, "r") as f:
    unit_ids = list(f["units"].keys())
    if unit_ids:
        first_unit = unit_ids[0]
        if "features" in f[f"units/{first_unit}"]:
            features_in_hdf5 = list(f[f"units/{first_unit}/features"].keys())
            print(f"Features in HDF5 for {first_unit}: {features_in_hdf5}")
        else:
            print(f"No features found in HDF5 for {first_unit}")


# # =============================================================================
# # Example 3: load_recording_with_eimage_sta with session (deferred mode)
# # This function now supports deferred mode! It computes eimage_sta for all
# # units and stores the results in the session. Save when ready.
# # =============================================================================
# print("\n" + "="*60)
# print("Example 3: load_recording_with_eimage_sta with session")
# print("="*60)

# # Create a new session for this example
# session_eimage = create_session(dataset_id="2025.04.10-11.12.57-Rec-eimage")

# # Load recording with eimage_sta computation in deferred mode
# session_eimage = load_recording_with_eimage_sta(
#     cmcr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec.cmcr",
#     cmtr_path="O:/20250410/set6/2025.04.10-11.12.57-Rec-.cmtr",
#     duration_s=120.0,
#     spike_limit=10000,
#     window_range=(-10, 40),
#     session=session_eimage,  # Pass session for deferred mode
# )

# print(f"Session has {session_eimage.unit_count} units")
# print(f"Completed steps: {session_eimage.completed_steps}")
# print(f"Estimated memory usage: {session_eimage.memory_estimate_gb:.2f} GB")

# # Check if eimage_sta was stored
# units_with_eimage_sta = sum(
#     1 for u in session_eimage.units.values()
#     if "features" in u and "eimage_sta" in u.get("features", {})
# )
# print(f"Units with eimage_sta: {units_with_eimage_sta}")



# =============================================================================
# Example 4: add_section_time_analog (immediate mode only for now)
# NOTE: This function can detect stimulus onsets from the light reference
#       signal using peak detection. Currently immediate mode only.
# =============================================================================
# add_section_time_analog(
#     hdf5_path=hdf5_path,
#     threshold_value=1e6,  # Determined from inspecting np.diff(raw_ch1)
#     movie_name="custom_stim",
#     plot_duration=120.0,
#     force=True,
# )


