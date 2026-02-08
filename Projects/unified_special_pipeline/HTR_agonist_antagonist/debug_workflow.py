"""Debug script to examine the alignment and cell type workflow."""
import h5py
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path('output')

# Check the before file structure and step response data
before_file = OUTPUT_DIR / '2025.10.01-09.33.32-Rec.h5'
print('=== Before File Structure ===')
with h5py.File(before_file, 'r') as f:
    unit = f['units/unit_001']
    print('Unit attrs:', list(unit.attrs.keys()))
    print('Unit groups:', list(unit.keys()))
    
    step_path = 'spike_times_sectioned/step_up_5s_5i_b0_3x'
    if step_path in unit:
        step = unit[step_path]
        print(f'Step group keys: {list(step.keys())}')
        if 'trials_start_end' in step:
            tse = step['trials_start_end'][:]
            print(f'Trials start/end shape: {tse.shape}')
            print(f'Trial 0: start={tse[0,0]}, end={tse[0,1]}')
            print(f'Trial duration (frames): {tse[0,1] - tse[0,0]}')
            print(f'Trial duration (seconds): {(tse[0,1] - tse[0,0]) / 20000:.2f}')
        
        if 'trials_spike_times' in step:
            trials = step['trials_spike_times']
            print(f'Number of trials: {len(trials.keys())}')
            t0 = trials['0'][:]
            print(f'Trial 0 spikes: {len(t0)}')
            if len(t0) > 0:
                print(f'  First spikes (frames): {t0[:5]}')
                rel_ms = (t0[:5] - tse[0,0]) / 20
                print(f'  Relative to start (ms): {rel_ms}')
    else:
        print(f'Step response NOT FOUND at {step_path}')

# Check aligned middle file
print('\n=== Aligned Middle File ===')
aligned_file = OUTPUT_DIR / 'aligned_middle/chip_4121_middle_aligned.h5'
with h5py.File(aligned_file, 'r') as f:
    print('Root attrs:')
    for k, v in f.attrs.items():
        print(f'  {k}: {v}')
    
    print(f'\nNum chains: {f.attrs["num_chains"]}')
    
    # Count cell types
    cell_types = {}
    for chain_key in f['aligned_units'].keys():
        chain = f['aligned_units'][chain_key]
        ct = chain.attrs['cell_type']
        cell_types[ct] = cell_types.get(ct, 0) + 1
    print(f'Cell type distribution: {cell_types}')
    
    # Check first few chains
    print('\nFirst 5 chains:')
    for i, chain_key in enumerate(list(f['aligned_units'].keys())[:5]):
        chain = f['aligned_units'][chain_key]
        print(f'  {chain_key}: ref={chain.attrs["reference_unit"]}, cell_type={chain.attrs["cell_type"]}')
        print(f'    unit_ids: {list(chain.attrs["unit_ids"])}')
        for rec_key in chain.keys():
            rec = chain[rec_key]
            print(f'    {rec_key}: file={rec.attrs["file"]}, unit={rec.attrs["unit_id"]}')

# Check middle file structure (spike times)
print('\n=== Middle File Structure ===')
middle_file = OUTPUT_DIR / '2025.10.01-09.45.32-Rec.h5'
with h5py.File(middle_file, 'r') as f:
    unit = f['units/unit_001']
    print('Unit groups:', list(unit.keys()))
    if 'spike_times' in unit:
        st = unit['spike_times'][:]
        print(f'Spike times: {len(st)} spikes')
        print(f'  First: {st[0]}, Last: {st[-1]}')
        print(f'  Duration (min): {st[-1] / 20000 / 60:.2f}')
    
    # Check if step response exists
    step_path = 'spike_times_sectioned/step_up_5s_5i_b0_3x'
    if step_path in unit:
        print(f'Step response exists in middle file')
    else:
        print(f'Step response NOT in middle file (expected)')
