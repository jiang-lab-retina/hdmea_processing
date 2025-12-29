"""Quick summary of on_off_dict problems."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
from collections import defaultdict

ON_OFF_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8

with open(ON_OFF_DICT_PATH, 'rb') as f:
    on_off_dict = pickle.load(f)

# Count problems by direction
problems_by_dir = defaultdict(int)
pixels_by_dir = defaultdict(set)

for key, pixel_data in on_off_dict.items():
    on_times = pixel_data['on_peak_location']
    off_times = pixel_data['off_peak_location']
    
    for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
        if off < on:
            direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
            problems_by_dir[direction] += 1
            pixels_by_dir[direction].add(key)

print('='*60)
print('ON/OFF DICT PROBLEM SUMMARY')
print('='*60)
print(f'Total pixels in dict: {len(on_off_dict)}')
print()
print('Problematic entries by direction:')
print('-'*60)
for d in DIRECTION_LIST:
    n_entries = problems_by_dir[d]
    n_pixels = len(pixels_by_dir[d])
    if n_entries > 0:
        print(f'  Direction {d:>3}°: {n_entries:>5} entries across {n_pixels:>4} pixels')
    else:
        print(f'  Direction {d:>3}°: ✓ No problems')

total = sum(problems_by_dir.values())
total_pixels = len(set.union(*pixels_by_dir.values()) if pixels_by_dir else set())
print('-'*60)
print(f'  TOTAL: {total} entries across {total_pixels} unique pixels')

# Check pixel (270, 30) specifically
print()
print('='*60)
print('PIXEL (270, 30) ANALYSIS')
print('='*60)

key = (270, 30)
if key not in on_off_dict:
    key = (30, 270)
    
if key in on_off_dict:
    pd = on_off_dict[key]
    print(f'Found pixel with key: {key}')
    print()
    print('Trial-by-trial on/off times:')
    print('-'*60)
    on_times = pd['on_peak_location']
    off_times = pd['off_peak_location']
    
    for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
        direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
        rep = trial_idx // N_DIRECTIONS
        dur = off - on
        status = "✓" if dur > 0 else "✗ SWAPPED!"
        print(f'  Trial {trial_idx:>2} (Dir {direction:>3}°, Rep {rep}): on={on:>5}, off={off:>5}, dur={dur:>4} {status}')
else:
    print('Pixel NOT FOUND!')

# What's the pattern?
print()
print('='*60)
print('PATTERN ANALYSIS')
print('='*60)
print()
print('Directions with problems (diagonal movement):')
for d in [45, 135, 225, 315]:
    n = len(pixels_by_dir[d])
    if n > 0:
        print(f'  {d}°: {n} pixels affected')
        
print()
print('Directions without problems (horizontal/vertical):')
for d in [0, 90, 180, 270]:
    n = len(pixels_by_dir[d])
    if n > 0:
        print(f'  {d}°: {n} pixels affected')
    else:
        print(f'  {d}°: ✓ No problems')

