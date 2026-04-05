# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Add Freq-Step Trace Segments & Sine-Fit Features
#
# Reads the enriched dataframe, computes the mean trace from the 3 reps in
# `freq_step_5st_3x`, extracts each frequency segment, fits a sine wave,
# and stores amplitude + phase_shift (degrees, -180 to 180) as new columns.
#
# New columns added:
#   - freq_section_0p5hz .. freq_section_10hz   (trace segments, stored as JSON)
#   - freq_sinefit_{f}hz_amplitude              (fitted amplitude)
#   - freq_sinefit_{f}hz_phase_deg              (fitted phase in degrees, -180 to 180)

# %%
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit

print("Imports done", flush=True)

# %%
# ---------- configuration (mirrors step_config.py) ----------
SAMPLING_RATE = 60.0  # Hz
FREQ_STEP_FREQUENCIES = [0.5, 1, 2, 4, 10]  # Hz
FREQ_STEP_BOUNDS = {
    0.5: (30, 270),
    1:   (330, 570),
    2:   (630, 870),
    4:   (930, 1170),
    10:  (1230, 1470),
}
FREQ_FIT_SKIP_FRAMES = 60   # skip transient (except 0.5 Hz)
FREQ_AMP_GUESS = 50
FREQ_AMP_UPPER_LIM = 400
FREQ_OFFSET_UPPER_LIM = 200
FREQ_OFFSET_LOWER_LIM = -200
FREQ_MAXFEV = 2000
FREQ_R_SQUARED_THRESHOLD = 0.1

TRACE_COLUMN = "freq_step_5st_3x"

# ---------- paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SPATIAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PARQUET = RESULTS_DIR / "labeled_dataframe_with_legacy_coords.parquet"
OUTPUT_PARQUET = RESULTS_DIR / "labeled_dataframe_with_legacy_coords_freq.parquet"


# %%
# ==========================================================================
# Helpers
# ==========================================================================

def freq_label(freq):
    """0.5 -> '0p5', 1 -> '1', 10 -> '10'."""
    s = str(freq)
    return s.replace(".", "p")


def compute_mean_trace(trials_data):
    """Average 3 reps into one mean trace (no filtering)."""
    if trials_data is None:
        return None
    if isinstance(trials_data, float) and np.isnan(trials_data):
        return None
    try:
        valid = [np.asarray(t, dtype=float) for t in trials_data if t is not None]
        if len(valid) == 0:
            return None
        return np.vstack(valid).mean(axis=0)
    except Exception:
        return None


def fit_sine_fixed_freq(x, y, freq_fixed):
    """
    Fit  y = A * sin(2*pi*f*x + phi) + offset.
    Returns (amplitude, phase_deg, r_squared).
    phase_deg in (-180, 180).
    """
    def model(x, amplitude, phase, offset):
        return amplitude * np.sin(2 * np.pi * freq_fixed * x + phase) + offset

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    try:
        p0 = [FREQ_AMP_GUESS, 0.0, 0.0]
        bounds = (
            [0, -np.pi, FREQ_OFFSET_LOWER_LIM],
            [FREQ_AMP_UPPER_LIM, np.pi, FREQ_OFFSET_UPPER_LIM],
        )
        params, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=FREQ_MAXFEV)
        amp, phase_rad, offset = params

        residuals = y - model(x, *params)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        if not np.isfinite(r2):
            r2 = 0.0

        # Convert phase to degrees
        phase_deg = float(np.degrees(phase_rad))

        # Apply R^2 threshold
        if r2 < FREQ_R_SQUARED_THRESHOLD:
            amp = 0.0
            phase_deg = np.nan

        return float(amp), phase_deg, float(r2)

    except Exception:
        return 0.0, np.nan, 0.0


# %%
# ==========================================================================
# Main
# ==========================================================================

print("Loading dataframe ...", flush=True)
df = pd.read_parquet(INPUT_PARQUET)
print(f"  Shape: {df.shape}", flush=True)

# Pre-allocate new columns
section_cols = {}
for freq in FREQ_STEP_FREQUENCIES:
    fl = freq_label(freq)
    section_cols[freq] = f"freq_section_{fl}hz"
    df[f"freq_section_{fl}hz"] = None
    df[f"freq_sinefit_{fl}hz_amplitude"] = np.nan
    df[f"freq_sinefit_{fl}hz_phase_deg"] = np.nan
    df[f"freq_sinefit_{fl}hz_r_squared"] = np.nan

max_end = max(end for _, end in FREQ_STEP_BOUNDS.values())

print("Processing cells ...", flush=True)
n = len(df)
valid_count = 0

for i, idx in enumerate(df.index):
    trials_data = df.at[idx, TRACE_COLUMN]
    mean_trace = compute_mean_trace(trials_data)

    if mean_trace is None or len(mean_trace) < max_end:
        continue

    for freq in FREQ_STEP_FREQUENCIES:
        fl = freq_label(freq)
        start, end = FREQ_STEP_BOUNDS[freq]

        # Store segment as JSON list
        segment_full = mean_trace[start:end]
        df.at[idx, f"freq_section_{fl}hz"] = json.dumps(segment_full.tolist())

        # Fit: skip transient for all except 0.5 Hz
        fit_start = start + FREQ_FIT_SKIP_FRAMES if freq != 0.5 else start
        segment_fit = mean_trace[fit_start:end]
        t = np.arange(len(segment_fit)) / SAMPLING_RATE

        amp, phase_deg, r2 = fit_sine_fixed_freq(t, segment_fit, freq)
        df.at[idx, f"freq_sinefit_{fl}hz_amplitude"] = amp
        df.at[idx, f"freq_sinefit_{fl}hz_phase_deg"] = phase_deg
        df.at[idx, f"freq_sinefit_{fl}hz_r_squared"] = r2

    valid_count += 1
    if (i + 1) % 2000 == 0 or (i + 1) == n:
        print(f"  [{i+1}/{n}] valid={valid_count}", flush=True)

# %%
# ---------- summary ----------
print(f"\nSummary:", flush=True)
print(f"  Total cells:  {n}", flush=True)
print(f"  Valid traces:  {valid_count}", flush=True)
for freq in FREQ_STEP_FREQUENCIES:
    fl = freq_label(freq)
    amp_col = f"freq_sinefit_{fl}hz_amplitude"
    phase_col = f"freq_sinefit_{fl}hz_phase_deg"
    r2_col = f"freq_sinefit_{fl}hz_r_squared"
    valid_amp = df[amp_col].dropna()
    valid_phase = df[phase_col].dropna()
    print(f"\n  {freq} Hz:", flush=True)
    print(f"    amplitude: mean={valid_amp.mean():.2f}, std={valid_amp.std():.2f}", flush=True)
    print(f"    phase_deg: mean={valid_phase.mean():.1f}, range=[{valid_phase.min():.1f}, {valid_phase.max():.1f}]", flush=True)
    print(f"    valid phase (R2>={FREQ_R_SQUARED_THRESHOLD}): {len(valid_phase)}/{len(valid_amp)}", flush=True)

# %%
print(f"\nSaving -> {OUTPUT_PARQUET}", flush=True)
df.to_parquet(OUTPUT_PARQUET, index=True)
print("Done.", flush=True)
