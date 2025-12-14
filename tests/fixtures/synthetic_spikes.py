"""
Synthetic spike data generators for testing.

These generators create controlled spike patterns for testing feature extractors.
"""

import numpy as np
from typing import Optional, Tuple


def generate_poisson_spikes(
    duration_s: float,
    rate_hz: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Poisson spike train for testing.
    
    Args:
        duration_s: Duration in seconds
        rate_hz: Mean firing rate in Hz
        seed: Random seed for reproducibility
    
    Returns:
        Spike times in microseconds
    """
    rng = np.random.default_rng(seed)
    
    # Generate expected number of spikes (with buffer)
    n_expected = int(duration_s * rate_hz * 1.5)
    if n_expected < 10:
        n_expected = 10
    
    # Generate inter-spike intervals from exponential distribution
    isi = rng.exponential(1.0 / rate_hz, n_expected)
    
    # Cumulative sum to get spike times
    times_s = np.cumsum(isi)
    
    # Keep only spikes within duration
    times_s = times_s[times_s < duration_s]
    
    # Convert to microseconds
    times_us = (times_s * 1e6).astype(np.uint64)
    
    return times_us


def generate_on_off_response(
    baseline_rate: float = 5.0,
    on_rate: float = 50.0,
    off_rate: float = 30.0,
    stim_onset_s: float = 5.0,
    stim_offset_s: float = 10.0,
    duration_s: float = 15.0,
    transient_duration_s: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate spike train with ON/OFF response pattern.
    
    Args:
        baseline_rate: Baseline firing rate (Hz)
        on_rate: Peak ON response rate (Hz)
        off_rate: Peak OFF response rate (Hz)
        stim_onset_s: Time of stimulus onset (seconds)
        stim_offset_s: Time of stimulus offset (seconds)
        duration_s: Total duration (seconds)
        transient_duration_s: Duration of transient response
        seed: Random seed
    
    Returns:
        Spike times in microseconds
    """
    rng = np.random.default_rng(seed)
    
    # Generate time-varying rate profile
    dt = 0.001  # 1ms resolution
    times = np.arange(0, duration_s, dt)
    rates = np.full_like(times, baseline_rate)
    
    # ON response (transient increase at stimulus onset)
    on_start_idx = int(stim_onset_s / dt)
    on_end_idx = int((stim_onset_s + transient_duration_s) / dt)
    if on_end_idx <= len(rates):
        rates[on_start_idx:on_end_idx] = on_rate
    
    # OFF response (transient increase at stimulus offset)
    off_start_idx = int(stim_offset_s / dt)
    off_end_idx = int((stim_offset_s + transient_duration_s) / dt)
    if off_end_idx <= len(rates):
        rates[off_start_idx:off_end_idx] = off_rate
    
    # Generate inhomogeneous Poisson spikes
    spikes = []
    for t, rate in zip(times, rates):
        if rng.random() < rate * dt:
            spikes.append(t * 1e6)  # Convert to microseconds
    
    return np.array(spikes, dtype=np.uint64)


def generate_direction_selective_response(
    preferred_direction: float = 0.0,
    baseline_rate: float = 5.0,
    peak_rate: float = 50.0,
    dsi: float = 0.8,
    n_directions: int = 8,
    stim_duration_s: float = 2.0,
    isi_s: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate direction-selective spike response.
    
    Args:
        preferred_direction: Preferred direction in degrees
        baseline_rate: Baseline rate (Hz)
        peak_rate: Peak response rate for preferred direction (Hz)
        dsi: Direction selectivity index [0, 1]
        n_directions: Number of directions tested
        stim_duration_s: Duration of each direction stimulus
        isi_s: Inter-stimulus interval
        seed: Random seed
    
    Returns:
        Tuple of (spike_times, direction_times)
        - spike_times: Spike times in microseconds
        - direction_times: Start times for each direction presentation
    """
    rng = np.random.default_rng(seed)
    
    # Calculate response rate for each direction
    directions = np.linspace(0, 360, n_directions, endpoint=False)
    direction_times = []
    
    # Von Mises-like tuning curve
    pref_rad = np.deg2rad(preferred_direction)
    dir_rad = np.deg2rad(directions)
    
    # Tuning = baseline + amplitude * (1 + cos(theta - pref)) / 2
    amplitude = (peak_rate - baseline_rate) * dsi
    tuning = baseline_rate + amplitude * (1 + np.cos(dir_rad - pref_rad)) / 2
    
    spikes = []
    current_time = 0.0
    
    for i, (direction, rate) in enumerate(zip(directions, tuning)):
        direction_times.append(current_time * 1e6)
        
        # Generate spikes for this direction
        dt = 0.001
        for t_offset in np.arange(0, stim_duration_s, dt):
            t = current_time + t_offset
            if rng.random() < rate * dt:
                spikes.append(t * 1e6)
        
        current_time += stim_duration_s + isi_s
    
    return np.array(spikes, dtype=np.uint64), np.array(direction_times, dtype=np.uint64)


def generate_synthetic_waveform(
    n_samples: int = 50,
    peak_amplitude: float = -100.0,
    trough_amplitude: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic spike waveform.
    
    Args:
        n_samples: Number of samples
        peak_amplitude: Negative peak amplitude (μV)
        trough_amplitude: Positive trough amplitude (μV)
        seed: Random seed
    
    Returns:
        Waveform array (float32)
    """
    rng = np.random.default_rng(seed)
    
    t = np.linspace(0, 1, n_samples)
    
    # Simple spike shape: fast negative peak, slower positive trough
    waveform = peak_amplitude * np.exp(-((t - 0.3) ** 2) / 0.01)
    waveform += trough_amplitude * np.exp(-((t - 0.5) ** 2) / 0.03)
    
    # Add small noise
    waveform += rng.normal(0, 2, n_samples)
    
    return waveform.astype(np.float32)


def generate_synthetic_light_reference(
    duration_s: float,
    sample_rate_hz: float = 20000,
    n_flashes: int = 3,
    flash_duration_s: float = 5.0,
    flash_interval_s: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic light reference signal.
    
    Args:
        duration_s: Total duration in seconds
        sample_rate_hz: Sample rate in Hz
        n_flashes: Number of light flashes
        flash_duration_s: Duration of each flash
        flash_interval_s: Interval between flashes
        seed: Random seed
    
    Returns:
        Light reference signal (float32)
    """
    rng = np.random.default_rng(seed)
    
    n_samples = int(duration_s * sample_rate_hz)
    signal = np.zeros(n_samples, dtype=np.float32)
    
    # Add flashes
    samples_per_flash = int(flash_duration_s * sample_rate_hz)
    samples_per_interval = int(flash_interval_s * sample_rate_hz)
    
    for i in range(n_flashes):
        start = i * (samples_per_flash + samples_per_interval)
        end = start + samples_per_flash
        if end <= n_samples:
            signal[start:end] = 1.0
    
    # Add noise
    signal += rng.normal(0, 0.02, n_samples)
    
    return signal.astype(np.float32)

