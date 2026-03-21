"""TTS provider abstraction — synthetic audio generation for testing.

Provides a small provider contract for text-to-speech synthesis, with
a factory for selecting backends by name.  The first (and only v1)
backend is ``edge`` (Microsoft Edge TTS, cloud-based).

**This module is for synthetic/test audio only.**  TTS outputs are
clearly marked with ``synthetic=True`` and must not be confused with
real patient recordings.

Design principles:
  - Provider interface is generic — no edge-tts specifics leak out.
  - Adding a new provider = one class + one registry entry.
  - Cloud providers require explicit opt-in by name.
  - No silent fallback between providers.

Pure I/O layer — no clinical logic, no ASR, no pipeline coupling.
"""

from __future__ import annotations

import struct
import wave
from abc import ABC, abstractmethod
from pathlib import Path


# ── result contract ────────────────────────────────────────────────


def _tts_result(
    audio_path: str,
    provider: str,
    voice: str,
    text: str,
    success: bool,
    error: str | None = None,
) -> dict:
    """Build a standardised TTS result dict.

    Every provider returns this shape — downstream code can rely on it.
    """
    return {
        "audio_path": audio_path,
        "provider": provider,
        "voice": voice,
        "text": text,
        "success": success,
        "error": error,
        "synthetic": True,
    }


_TTS_RESULT_KEYS = frozenset(_tts_result("", "", "", "", True).keys())


# ── provider ABC ───────────────────────────────────────────────────


class TTSProvider(ABC):
    """Abstract base for TTS providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. ``"edge"``)."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str | None = None,
        rate: str | None = None,
        lang: str | None = None,
    ) -> dict:
        """Synthesize *text* to a 16 kHz mono 16-bit PCM WAV file.

        Args:
            text: input text to synthesize.
            output_path: destination ``.wav`` path.
            voice: provider-specific voice identifier (optional).
            rate: speech rate adjustment (optional, provider-specific).
            lang: language code hint (optional).

        Returns:
            Standardised result dict (use :func:`_tts_result`).
        """


# ── audio helper ───────────────────────────────────────────────────


def write_pcm16_wav(
    samples: bytes,
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write raw PCM-16 samples to a WAV file.

    Args:
        samples: raw signed 16-bit little-endian PCM bytes.
        path: output file path.
        sample_rate: samples per second.
        channels: number of audio channels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)


def read_wav_float32(path: Path) -> tuple:
    """Read a WAV file and return (float32_samples, sample_rate).

    Converts 16-bit PCM to float32 in [-1.0, 1.0].  Takes only the
    first channel if stereo.

    Raises:
        ValueError: if the file is not 16-bit PCM.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV, got {sampwidth * 8}-bit: {path}"
        )

    # Unpack all samples.
    total_samples = n_frames * n_channels
    samples = struct.unpack(f"<{total_samples}h", raw)

    # Take first channel only.
    if n_channels > 1:
        samples = samples[::n_channels]

    # Convert to float32.
    import numpy as np

    audio = np.array(samples, dtype=np.float32) / 32768.0
    return audio, framerate


# ── edge-tts provider ─────────────────────────────────────────────


_EDGE_DEFAULT_VOICES: dict[str, str] = {
    "en": "en-US-GuyNeural",
    "da": "da-DK-JeppeNeural",
    "sv": "sv-SE-MattiasNeural",
}


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS provider (cloud-based, requires internet)."""

    @property
    def name(self) -> str:
        return "edge"

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str | None = None,
        rate: str | None = None,
        lang: str | None = None,
    ) -> dict:
        resolved_voice = voice or _EDGE_DEFAULT_VOICES.get(lang or "en", _EDGE_DEFAULT_VOICES["en"])

        try:
            import asyncio
            import edge_tts
        except ImportError as exc:
            return _tts_result(
                audio_path="",
                provider=self.name,
                voice=resolved_voice,
                text=text,
                success=False,
                error=f"edge-tts not installed: {exc}",
            )

        try:
            mp3_path = output_path.with_suffix(".mp3")
            communicate = edge_tts.Communicate(text, resolved_voice, rate=rate)
            asyncio.run(communicate.save(str(mp3_path)))

            # Convert MP3 → 16 kHz mono PCM WAV.
            _convert_mp3_to_wav(mp3_path, output_path)

            # Clean up intermediate MP3.
            mp3_path.unlink(missing_ok=True)

            return _tts_result(
                audio_path=str(output_path),
                provider=self.name,
                voice=resolved_voice,
                text=text,
                success=True,
            )
        except Exception as exc:
            return _tts_result(
                audio_path="",
                provider=self.name,
                voice=resolved_voice,
                text=text,
                success=False,
                error=str(exc),
            )


def _convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    """Convert an MP3 file to 16 kHz mono 16-bit PCM WAV.

    Uses the ``pydub`` library if available, otherwise falls back to
    a basic ``audioop``-free approach via the stdlib.

    Raises:
        RuntimeError: if conversion fails.
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(str(wav_path), format="wav")
        return
    except ImportError:
        pass

    raise RuntimeError(
        "Cannot convert MP3 to WAV: install pydub (pip install pydub) "
        "and ensure ffmpeg is available on PATH."
    )


# ── provider factory ──────────────────────────────────────────────


_PROVIDERS: dict[str, type[TTSProvider]] = {
    "edge": EdgeTTSProvider,
}


def get_tts_provider(name: str) -> TTSProvider:
    """Get a TTS provider by name.

    Args:
        name: provider identifier (e.g. ``"edge"``).

    Returns:
        An instance of the requested provider.

    Raises:
        ValueError: if the provider name is not registered.
    """
    cls = _PROVIDERS.get(name)
    if cls is None:
        available = sorted(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown TTS provider: {name!r}. Available: {available}"
        )
    return cls()


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_PROVIDERS.keys())
