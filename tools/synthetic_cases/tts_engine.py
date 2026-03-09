"""TTS engine wrapper for synthetic case audio generation.

Uses pyttsx3 (Windows SAPI) for local, offline, deterministic speech
synthesis.  Falls back to a sine-tone placeholder if pyttsx3 is not
available, so the test framework still works without TTS.

All output is 16 kHz mono float32 numpy arrays.
"""

from __future__ import annotations

import io
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample

from tools.synthetic_cases.audio_env import TARGET_RATE


@dataclass
class VoiceConfig:
    """Per-speaker voice configuration."""

    voice_id: str | None = None
    # pyttsx3 voice ID; None = engine default.

    rate: int = 160
    # Words per minute.

    volume: float = 1.0
    # 0.0 – 1.0.


# ── voice presets ────────────────────────────────────────────────

# Map voice_hint from scenario participants to pyttsx3 voice IDs.
# These are the standard Windows SAPI voices; will be ignored if
# the voice is not installed.
_VOICE_HINTS: dict[str, str] = {
    "male": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
    "female": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
}


def voice_config_from_hint(hint: str, rate: int = 160, volume: float = 1.0) -> VoiceConfig:
    """Create a VoiceConfig from a voice_hint string."""
    return VoiceConfig(
        voice_id=_VOICE_HINTS.get(hint),
        rate=rate,
        volume=volume,
    )


# ── TTS synthesis ────────────────────────────────────────────────


def synthesize_utterance(
    text: str,
    voice: VoiceConfig | None = None,
) -> np.ndarray:
    """Synthesize a single utterance to a 16 kHz mono float32 array.

    Args:
        text: the text to speak.
        voice: voice configuration.  ``None`` uses defaults.

    Returns:
        float32 numpy array at 16 kHz.
    """
    if voice is None:
        voice = VoiceConfig()

    try:
        return _synthesize_pyttsx3(text, voice)
    except Exception:
        return _synthesize_placeholder(text)


def _synthesize_pyttsx3(text: str, voice: VoiceConfig) -> np.ndarray:
    """Synthesize using pyttsx3 (Windows SAPI)."""
    import pyttsx3  # type: ignore[import-untyped]

    engine = pyttsx3.init()

    if voice.voice_id:
        try:
            engine.setProperty("voice", voice.voice_id)
        except Exception:
            pass  # Fall back to default voice

    engine.setProperty("rate", voice.rate)
    engine.setProperty("volume", voice.volume)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        return _load_and_resample(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_and_resample(wav_path: str) -> np.ndarray:
    """Load a WAV file and resample to 16 kHz mono float32."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        orig_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        # 24-bit or other — try to parse as 16-bit best-effort
        samples = np.frombuffer(raw[:n_frames * 2], dtype=np.int16).astype(np.float32) / 32768.0

    # Mix to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16 kHz
    if orig_rate != TARGET_RATE:
        target_len = int(len(samples) * TARGET_RATE / orig_rate)
        samples = resample(samples, target_len).astype(np.float32)

    return samples


def _synthesize_placeholder(text: str) -> np.ndarray:
    """Generate a sine-tone placeholder when TTS is unavailable.

    Produces a fixed-length tone proportional to text length so tests
    can still verify audio assembly without real TTS.
    """
    # ~0.05s per character, minimum 0.5s
    duration = max(0.5, len(text) * 0.05)
    t = np.linspace(0, duration, int(TARGET_RATE * duration), dtype=np.float32)
    # 440 Hz tone at low volume
    return 0.1 * np.sin(2 * np.pi * 440 * t)
