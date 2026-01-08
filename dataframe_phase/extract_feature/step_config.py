"""
Configuration for step response feature extraction.

Timing parameters adapted for 60 Hz data with trace length 599 frames.
Legacy code was designed for 10 Hz, so base values are multiplied by 6.
"""

# =============================================================================
# Timing Parameters (60 Hz)
# =============================================================================

# Baseline period before stimulus (frames 0 to PRE_MARGIN-1)
PRE_MARGIN = 60  # 1 second at 60 Hz

# Window size for transient peak detection
TRANSIENT_ZONE = 60  # 1 second at 60 Hz

# =============================================================================
# Frame Indices for Feature Extraction
# =============================================================================

# ON transient: peak detection window after light ON
TRANSIENT_START = 60   # Frame index where ON phase begins
TRANSIENT_END = 120    # End of ON transient detection window

# ON sustained: late phase of ON response
SUSTAINED_START = 239  # Start of ON sustained window
SUSTAINED_END = 299    # End of ON sustained window

# OFF transient: peak detection window after light OFF
OFF_TRANSIENT_START = 360  # Frame index where OFF phase begins
OFF_TRANSIENT_END = 420    # End of OFF transient detection window

# OFF sustained: late phase of OFF response
OFF_SUSTAINED_START = 539  # Start of OFF sustained window
OFF_SUSTAINED_END = 599    # End of OFF sustained window (trace end)

# =============================================================================
# Quality Control
# =============================================================================

# Maximum valid firing rate (Hz) - rows exceeding this are filtered out
VALID_MAX_FIRING_RATE = 400

# =============================================================================
# Peak Detection
# =============================================================================

# Prominence threshold as multiple of baseline standard deviation
# prominence = PROMINENCE_STD_THRESHOLD * baseline_std
# Higher values filter out more noise peaks
PROMINENCE_STD_THRESHOLD = 2

# Low-pass filter settings
LOWPASS_CUTOFF_FREQ = 10.0  # Hz
LOWPASS_FILTER_ORDER = 5

# =============================================================================
# Data Configuration
# =============================================================================

# Column containing step response trials
STEP_TRACE_COLUMN = "step_up_5s_5i_b0_3x"

# Sampling rate
SAMPLING_RATE = 60.0  # Hz

# =============================================================================
# Green-Blue Timing Parameters (60 Hz)
# =============================================================================
# Legacy code was designed for 10 Hz with pre_margin=10, step_duration=30,
# transient_duration=10. Values scaled by 6 for 60 Hz.

GB_PRE_MARGIN = 60          # 1 second at 60 Hz (baseline period)
GB_STEP_DURATION = 180      # 3 seconds at 60 Hz (each color step duration)
GB_TRANSIENT_DURATION = 60  # 1 second at 60 Hz (transient detection window)

# Column containing green-blue response trials
GB_TRACE_COLUMN = "green_blue_3s_3i_3x"

# Maximum valid firing rate for green-blue features (Hz)
GB_VALID_MAX_FIRING_RATE = 400

# =============================================================================
# Frequency Step Timing Parameters (60 Hz)
# =============================================================================
# Legacy code was designed for 10 Hz with pre_margin=10, gap_duration=10,
# step_duration=40. Values adapted for 60 Hz with pre_margin=30 (user specified).
# Note: 0.5 Hz starts at pre_margin with NO gap before it.

FREQ_PRE_MARGIN = 30           # Start of 0.5 Hz (no gap before first freq)
FREQ_GAP_DURATION = 60         # 1 second gap between frequencies at 60 Hz
FREQ_STEP_DURATION = 240       # 4 seconds per frequency at 60 Hz

# Frequencies to analyze
FREQ_STEP_FREQUENCIES = [0.5, 1, 2, 4, 10]  # Hz

# Pre-calculated frame indices (0.5 Hz has no gap before it)
FREQ_STEP_BOUNDS = {
    0.5: (30, 270),
    1: (330, 570),
    2: (630, 870),
    4: (930, 1170),
    10: (1230, 1470),
}

# Column containing frequency step response trials
FREQ_TRACE_COLUMN = "freq_step_5st_3x"

# Sine fit parameters
FREQ_AMP_GUESS = 50
FREQ_AMP_UPPER_LIM = 400       # Upper bound for amplitude
FREQ_OFFSET_UPPER_LIM = 200    # Upper bound for offset
FREQ_OFFSET_LOWER_LIM = -200   # Lower bound for offset
FREQ_MAXFEV = 2000             # Max function evaluations for curve_fit

# Quality threshold - fits with R^2 below this get amplitude set to 0
FREQ_R_SQUARED_THRESHOLD = 0.1

# Number of frames to skip at start of each frequency step for fitting
# (to avoid transient response). Applied to all frequencies except 0.5 Hz.
FREQ_FIT_SKIP_FRAMES = 60

