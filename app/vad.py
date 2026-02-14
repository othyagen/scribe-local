"""RMS-based Voice Activity Detection."""

from __future__ import annotations

from enum import Enum

import numpy as np

from app.config import AppConfig


class VadResult(Enum):
    """Per-chunk classification returned by the VAD."""
    SPEECH = "speech"
    SILENCE = "silence"


class VoiceActivityDetector:
    """Deterministic, threshold-based VAD using RMS energy."""

    def __init__(self, config: AppConfig) -> None:
        self.threshold: float = config.vad.speech_threshold_rms

    def process(self, chunk: np.ndarray) -> VadResult:
        """Classify a single audio chunk as speech or silence."""
        rms = self.compute_rms(chunk)
        if rms >= self.threshold:
            return VadResult.SPEECH
        return VadResult.SILENCE

    @staticmethod
    def compute_rms(chunk: np.ndarray) -> float:
        """Compute the Root Mean Square energy of an audio chunk."""
        return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
