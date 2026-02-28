"""Audio quality pre-check: metrics, reporting, and warnings."""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass

import numpy as np


DBFS_FLOOR = -120.0
SNR_CAP = 60.0
NOISE_FLOOR = 1e-8


@dataclass
class AudioPrecheckMetrics:
    duration_sec: float
    peak: float
    peak_dbfs: float
    rms: float
    rms_dbfs: float
    clipping_rate: float
    snr_db: float


def _to_dbfs(value: float) -> float:
    if value <= 0.0:
        return DBFS_FLOOR
    db = 20.0 * math.log10(value)
    return max(db, DBFS_FLOOR)


def compute_audio_precheck_metrics(
    audio: np.ndarray,
    sr: int,
    config,
) -> AudioPrecheckMetrics:
    """Compute audio quality metrics from a 1-D float32 array in [-1, 1]."""
    duration_sec = len(audio) / sr if sr > 0 else 0.0

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio)) if len(audio) > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    clip_level = config.audio.precheck_clip_level
    clipping_rate = float(np.mean(abs_audio >= clip_level)) if len(audio) > 0 else 0.0

    # SNR: split into frames, compute per-frame RMS
    frame_samples = int(sr * config.audio.precheck_frame_ms / 1000)
    if frame_samples > 0 and len(audio) >= frame_samples:
        n_frames = len(audio) // frame_samples
        frames = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        signal = float(np.percentile(frame_rms, 95))
        noise = float(np.percentile(frame_rms, 10))
        if noise <= NOISE_FLOOR:
            snr_db = SNR_CAP
        else:
            snr_db = 20.0 * math.log10(signal / noise)
        snr_db = max(0.0, min(snr_db, SNR_CAP))
    else:
        snr_db = 0.0

    return AudioPrecheckMetrics(
        duration_sec=duration_sec,
        peak=peak,
        peak_dbfs=_to_dbfs(peak),
        rms=rms,
        rms_dbfs=_to_dbfs(rms),
        clipping_rate=clipping_rate,
        snr_db=snr_db,
    )


def format_precheck_report(metrics: AudioPrecheckMetrics) -> str:
    """Format metrics into a single printable block."""
    lines = [
        "Audio Pre-check Results",
        f"  Duration     : {metrics.duration_sec:.1f} s",
        f"  Peak         : {metrics.peak:.4f}  ({metrics.peak_dbfs:.1f} dBFS)",
        f"  RMS          : {metrics.rms:.4f}  ({metrics.rms_dbfs:.1f} dBFS)",
        f"  Clipping     : {metrics.clipping_rate:.4%}",
        f"  SNR estimate : {metrics.snr_db:.1f} dB",
    ]
    return "\n".join(lines)


def detect_warnings(
    metrics: AudioPrecheckMetrics,
    config,
) -> list[str]:
    """Return warning strings for metrics outside acceptable ranges."""
    warnings: list[str] = []
    if metrics.snr_db < config.audio.precheck_snr_warn_db:
        warnings.append(
            f"Low estimated SNR ({metrics.snr_db:.1f} dB "
            f"< {config.audio.precheck_snr_warn_db:.1f} dB)"
        )
    if metrics.rms_dbfs < config.audio.precheck_rms_warn_dbfs:
        warnings.append(
            f"Low signal level ({metrics.rms_dbfs:.1f} dBFS "
            f"< {config.audio.precheck_rms_warn_dbfs:.1f} dBFS)"
        )
    if metrics.clipping_rate > config.audio.precheck_clip_warn_rate:
        warnings.append(
            f"Clipping detected ({metrics.clipping_rate:.4%} "
            f"> {config.audio.precheck_clip_warn_rate:.4%})"
        )
    return warnings


def record_precheck(config) -> np.ndarray | None:
    """Record a short audio buffer for pre-check using a temporary AudioCapture.

    Creates its own AudioCapture instance so the main session capture
    is never started, stopped, or otherwise affected.

    Returns the concatenated 1-D float32 audio, or None on failure.
    """
    from app.audio import AudioCapture

    precheck_audio = AudioCapture(config)
    chunks: list[np.ndarray] = []
    samples_needed = int(config.audio.precheck_seconds * config.audio.sample_rate)
    collected = 0

    precheck_audio.start()
    try:
        while collected < samples_needed:
            try:
                chunk = precheck_audio.get_chunk(timeout=2.0)
                chunks.append(chunk)
                collected += len(chunk)
            except queue.Empty:
                break
    finally:
        precheck_audio.stop()

    if not chunks:
        return None
    return np.concatenate(chunks)
