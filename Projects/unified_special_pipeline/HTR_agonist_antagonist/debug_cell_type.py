"""Debug cell type classification."""
import h5py
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path('output')
BIN_SIZE_MS = 50
TRIAL_DURATION_MS = 10000

def compute_psth(spike_times_trials, trial_start_end):
    """Compute PSTH from trial spike times."""
    n_bins = TRIAL_DURATION_MS // BIN_SIZE_MS  # 200 bins
    bin_counts = np.zeros(n_bins)
    n_trials = len(spike_times_trials)
    
    for trial_idx, trial_spikes in enumerate(spike_times_trials):
        if trial_idx >= len(trial_start_end):
            continue
        start_time = trial_start_end[trial_idx, 0]
        
        for spike in trial_spikes:
            relative_time_ms = (spike - start_time) / 20.0  # frames to ms at 20kHz
            if 0 <= relative_time_ms < TRIAL_DURATION_MS:
                bin_idx = int(relative_time_ms // BIN_SIZE_MS)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
    
    # Convert to Hz
    firing_rate = bin_counts / (n_trials * (BIN_SIZE_MS / 1000.0))
    return firing_rate

# Check a few units from the before file
before_file = OUTPUT_DIR / '2025.10.01-09.33.32-Rec.h5'

print("=== PSTH Analysis for Cell Type Classification ===\n")

with h5py.File(before_file, 'r') as f:
    for unit_id in ['unit_003', 'unit_008', 'unit_005']:  # OFF, ON, ON_OFF examples
        unit = f['units'][unit_id]
        step_path = 'spike_times_sectioned/step_up_5s_5i_b0_3x'
        
        if step_path not in unit:
            print(f"{unit_id}: No step response data")
            continue
        
        step = unit[step_path]
        trial_start_end = step['trials_start_end'][:]
        
        trial_spikes = []
        for trial_key in sorted(step['trials_spike_times'].keys(), key=int):
            trial_spikes.append(step['trials_spike_times'][trial_key][:])
        
        psth = compute_psth(trial_spikes, trial_start_end)
        
        # Analyze response
        # ON: bins 20-60 (1-3s), baseline bins 0-10 (0-0.5s)
        # OFF: bins 120-160 (6-8s), baseline bins 100-110 (5-5.5s)
        
        on_response = psth[20:60]
        on_baseline = psth[0:10]
        off_response = psth[120:160]
        off_baseline = psth[100:110]
        
        on_mean = np.mean(on_response)
        on_base_mean = np.mean(on_baseline)
        off_mean = np.mean(off_response)
        off_base_mean = np.mean(off_baseline)
        
        print(f"{unit_id}:")
        print(f"  ON:  baseline={on_base_mean:.2f} Hz, response={on_mean:.2f} Hz, diff={on_mean - on_base_mean:.2f}")
        print(f"  OFF: baseline={off_base_mean:.2f} Hz, response={off_mean:.2f} Hz, diff={off_mean - off_base_mean:.2f}")
        
        # Show PSTH summary
        print(f"  PSTH bins 0-10 (0-0.5s):   {psth[0:10].mean():.2f} Hz")
        print(f"  PSTH bins 10-20 (0.5-1s):  {psth[10:20].mean():.2f} Hz")
        print(f"  PSTH bins 20-60 (1-3s):    {psth[20:60].mean():.2f} Hz")
        print(f"  PSTH bins 100-110 (5-5.5s): {psth[100:110].mean():.2f} Hz")
        print(f"  PSTH bins 120-160 (6-8s):  {psth[120:160].mean():.2f} Hz")
        print()

# Check what the alignment file says about cell types
print("=== Cell Types from Aligned File ===")
aligned_file = OUTPUT_DIR / 'aligned_middle/chip_4121_middle_aligned.h5'
with h5py.File(aligned_file, 'r') as f:
    for chain_key in list(f['aligned_units'].keys())[:10]:
        chain = f['aligned_units'][chain_key]
        print(f"{chain_key}: ref={chain.attrs['reference_unit']}, cell_type={chain.attrs['cell_type']}")
