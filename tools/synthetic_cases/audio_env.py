"""Audio environment simulation for synthetic case generation.

Applies deterministic audio transformations to simulate different
recording conditions:
  - clean_room: no processing
  - telephone: bandpass filter (300-3400 Hz)
  - noisy_room: additive noise
  - distance_near / distance_far: volume attenuation + mild low-pass

All transforms operate on float32 numpy arrays at 16 kHz mono.
Uses scipy.signal for filtering — no external audio libraries needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt


TARGET_RATE = 16000  # SCRIBE standard sample rate


@dataclass
class AudioEnvConfig:
    """Configuration for audio environment simulation."""

    mode: str = "clean"
    # "clean" | "telephone" | "noisy" | "distance_near" | "distance_far"

    noise_level: float = 0.005
    # RMS of additive Gaussian noise (only for mode="noisy")

    telephone_low_hz: float = 300.0
    telephone_high_hz: float = 3400.0

    distance_far_gain: float = 0.35
    distance_far_lpf_hz: float = 4000.0

    distance_near_gain: float = 0.85


def apply_environment(
    audio: np.ndarray,
    config: AudioEnvConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply audio environment effects to a float32 waveform.

    Args:
        audio: float32 mono waveform at 16 kHz.
        config: environment configuration.
        rng: random generator for reproducible noise.

    Returns:
        Processed float32 waveform (same length).
    """
    if config.mode == "clean":
        return audio.copy()

    if config.mode == "telephone":
        return _apply_telephone(audio, config)

    if config.mode == "noisy":
        return _apply_noise(audio, config, rng)

    if config.mode == "distance_near":
        return audio * config.distance_near_gain

    if config.mode == "distance_far":
        return _apply_distance_far(audio, config)

    # Unknown mode — return clean copy
    return audio.copy()


def _apply_telephone(audio: np.ndarray, config: AudioEnvConfig) -> np.ndarray:
    """Bandpass filter simulating telephone audio."""
    nyq = TARGET_RATE / 2
    low = config.telephone_low_hz / nyq
    high = config.telephone_high_hz / nyq
    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfilt(sos, audio).astype(np.float32)
    return filtered


def _apply_noise(
    audio: np.ndarray,
    config: AudioEnvConfig,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Add Gaussian noise at configured level."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, config.noise_level, len(audio)).astype(np.float32)
    return audio + noise


def _apply_distance_far(audio: np.ndarray, config: AudioEnvConfig) -> np.ndarray:
    """Attenuate + low-pass to simulate far microphone placement."""
    attenuated = audio * config.distance_far_gain
    nyq = TARGET_RATE / 2
    cutoff = config.distance_far_lpf_hz / nyq
    sos = butter(2, cutoff, btype="low", output="sos")
    return sosfilt(sos, attenuated).astype(np.float32)
