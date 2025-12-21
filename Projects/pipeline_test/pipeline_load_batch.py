import sys
import h5py
import numpy as np

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from hdmea.pipeline import load_recording, load_recording_with_eimage_sta
from hdmea.pipeline import extract_features
from hdmea.pipeline import add_section_time
from hdmea.io.section_time import add_section_time_analog
from hdmea.io import section_spike_times
from hdmea.features import compute_sta
from hdmea.features.eimage_sta import compute_eimage_sta


# IMPORTANT: On Windows, multiprocessing requires the main code to be inside
# if __name__ == '__main__': block to prevent infinite recursion during spawn
if __name__ == '__main__':
    print("=" * 60)
    print("Running pipeline test...")
    print("=" * 60)

    # # Original separate load_recording call (kept for reference)
    # result = load_recording(
    #     cmcr_path="O:\\20250410\\set6\\2025.04.10-11.12.57-Rec.cmcr",
    #     cmtr_path="O:\\20250410\\set6\\2025.04.10-11.12.57-Rec-.cmtr",
    #     force=True,  # Force regeneration to apply spike_times conversion
    # )
    
    # Integrated loading + eimage_sta computation (loads CMTR and CMCR only once)
    result = load_recording_with_eimage_sta(
        cmcr_path="O:\\20250410\\set6\\2025.04.10-11.12.57-Rec.cmcr",
        cmtr_path="O:\\20250410\\set6\\2025.04.10-11.12.57-Rec-.cmtr",
        force=True,
        # eimage_sta parameters
        duration_s=120.0,
        spike_limit=10000,
        unit_ids= None, #["2", "7"],  # Match CMTR numbering (1-based)
        window_range=(-10, 40),
        skip_highpass=False,
        chunk_duration_s=30.0,
    )
    
    # print(f"\nResult: {result.num_units} units loaded, {result.units_with_sta} with STA")
    # print(f"Elapsed: {result.elapsed_seconds:.1f}s, Filter time: {result.filter_time_seconds:.1f}s")


    # # # # Extract FRIF features
    # # # print("\n" + "=" * 60)
    # # # print("Extracting FRIF features...")
    # # # print("=" * 60)
    # # # extract_result = extract_features(
    # # #     hdf5_path=result.hdf5_path,
    # # #     features=["frif"],
    # # #     force=True
    # # # )


    add_section_time(
        hdf5_path="artifacts/2025.04.10-11.12.57-Rec.h5",
        playlist_name="play_optimization_set6_ipRGC_manual",
        force=True,
    )

    # Check what movies have section_time BEFORE sectioning
    print("\n" + "=" * 60)
    print("Movies with section_time BEFORE sectioning:")
    with h5py.File("artifacts/JIANG009_2025-04-10.h5", mode='r') as root:
        if 'stimulus' in root and 'section_time' in root['stimulus']:
            movies_with_section = list(root['stimulus']['section_time'].keys())
            print(f"  Movies: {movies_with_section}")
            for m in movies_with_section:
                st = root['stimulus']['section_time'][m][:]
                print(f"    {m}: {st.shape[0]} trials, range=[{st[:,0].min():,} - {st[:,1].max():,}]")
        else:
            print("  No section_time found!")

        # Also check spike_times range
        print("\n  First unit spike_times range:")
        if 'units' in root:
            first_unit = list(root['units'].keys())[0]
            spk = root['units'][first_unit]['spike_times'][:]
            print(f"    {first_unit}: range=[{spk.min():,} - {spk.max():,}], count={len(spk)}")

    print("=" * 60)

    # Section spike times by trial boundaries
    section_result = section_spike_times(
        hdf5_path="artifacts/2025.04.10-11.12.57-Rec.h5",
        trial_repeats=3,           # Process first 3 trials
        pad_margin=(0.0, 0.0),     # 0s padding for now
        force=True,               # Force regeneration
    )

    # Print section result details
    print("\n" + "=" * 60)
    print("Section result:")
    print(f"  Units processed: {section_result.units_processed}")
    print(f"  Movies processed: {section_result.movies_processed}")
    print(f"  Warnings: {section_result.warnings}")
    print("=" * 60)

    # Detect ipRGC stimulus onsets from recorded light signal
    success = add_section_time_analog(
        hdf5_path="artifacts/2025.04.10-11.12.57-Rec.h5",
        threshold_value=3e4,  # Must inspect signal to determine
        movie_name="iprgc_test",
        plot_duration=120.0,  # 2 minute windows
        repeat=2,  # Use first 3 trials only
        force=True,
    )
    result = compute_sta(
        "artifacts/2025.04.10-11.12.57-Rec.h5",
        cover_range=(-60, 0),
        use_multiprocessing=True,
        force=True,
    )

