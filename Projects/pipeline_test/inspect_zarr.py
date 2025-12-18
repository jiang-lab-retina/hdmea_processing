"""Quick script to inspect spike_times and section_time in zarr."""
import zarr
import numpy as np

zarr_path = 'artifacts/JIANG009_2025-04-10.zarr'
root = zarr.open(zarr_path, mode='r')

# Check for TickUnit in metadata
print('='*60)
print('METADATA - TICK UNIT CHECK')
print('='*60)
if 'metadata' in root and 'sys_meta' in root['metadata']:
    sys_meta = root['metadata']['sys_meta']
    for key in sys_meta.keys():
        if 'tick' in key.lower() or 'time' in key.lower() or 'unit' in key.lower():
            val = sys_meta[key][...]
            print(f'{key}: {val}')
print()

print('='*60)
print('SPIKE TIMES EXAMINATION')
print('='*60)

# Check a few units
units = root['units']
unit_ids = list(units.keys())[:5]

for unit_id in unit_ids:
    unit = units[unit_id]
    if 'spike_times' in unit:
        st = unit['spike_times'][:]
        attrs = dict(unit['spike_times'].attrs) if hasattr(unit['spike_times'], 'attrs') else {}
        if len(st) > 0:
            print(f'{unit_id}: range=[{st.min():,} - {st.max():,}], count={len(st)}, attrs={attrs}')
        else:
            print(f'{unit_id}: empty')

print()
print('='*60)
print('SECTION TIME EXAMINATION (ALL MOVIES)')
print('='*60)

all_movies = []
if 'stimulus' in root and 'section_time' in root['stimulus']:
    st_group = root['stimulus']['section_time']
    all_movies = list(st_group.keys())
    print(f'Total movies with section_time: {len(all_movies)}')
    print(f'Movies: {all_movies}')
    for movie in all_movies:
        section = st_group[movie][:]
        print(f'\n{movie}: {section.shape[0]} trials')
        if len(section) > 0:
            print(f'  Trial 0: [{section[0,0]:,} - {section[0,1]:,}]')
            if len(section) > 1:
                print(f'  Trial 1: [{section[1,0]:,} - {section[1,1]:,}]')
            if len(section) > 2:
                print(f'  Trial 2: [{section[2,0]:,} - {section[2,1]:,}]')

print()
print('='*60)
print('ACQUISITION RATE & INTERPRETATION')
print('='*60)
if 'metadata' in root and 'acquisition_rate' in root['metadata']:
    acq = float(root['metadata']['acquisition_rate'][...])
    print(f'acquisition_rate: {acq} Hz')
    
    # Get first unit's spike range for interpretation
    first_unit = units[unit_ids[0]]
    st = first_unit['spike_times'][:]
    if len(st) > 0:
        max_spike = st.max()
        print(f'\nFirst unit max spike sample: {max_spike:,}')
        print(f'  = {max_spike / acq:.2f} seconds into recording')
        
        # Check if this looks like nanoseconds or samples
        if max_spike > 1e12:
            print(f'\n⚠️  WARNING: Values look like NANOSECONDS (too large for samples)')
            print(f'   Expected max for 20min recording: ~24,000,000 samples')
            print(f'   Re-run load_recording with force=True to apply conversion!')
        else:
            print(f'\n✓  Values appear to be in SAMPLE INDICES (correct)')

print()
print('='*60)
print('SPIKE TIMES SECTIONED CHECK')
print('='*60)

# Check sectioned data
first_unit_id = unit_ids[0]
first_unit = units[first_unit_id]
if 'spike_times_sectioned' in first_unit:
    sectioned = first_unit['spike_times_sectioned']
    movies_in_sectioned = list(sectioned.keys())
    print(f'Movies in spike_times_sectioned: {movies_in_sectioned}')
    
    for movie in movies_in_sectioned[:3]:
        movie_group = sectioned[movie]
        if 'full_spike_times' in movie_group:
            full = movie_group['full_spike_times'][:]
            print(f'  {movie}/full_spike_times: count={len(full)}')
            if len(full) > 0:
                print(f'    range=[{full.min():,} - {full.max():,}]')
        if 'trials_spike_times' in movie_group:
            trials = movie_group['trials_spike_times']
            trial_ids = list(trials.keys())
            print(f'  {movie}/trials_spike_times: {len(trial_ids)} trials')
            for tid in trial_ids[:2]:
                trial_data = trials[tid][:]
                print(f'    trial {tid}: count={len(trial_data)}')
else:
    print(f'{first_unit_id} has no spike_times_sectioned group')

