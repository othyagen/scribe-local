"""Speaker diarization — metadata-only, never modifies text.

The diarization layer determines *who* is speaking for a given audio
segment.  It runs BEFORE ASR and only attaches a ``speaker_id`` string.

Architecture:
    * ``Diarizer`` — abstract base class defining the interface.
    * ``DefaultDiarizer`` — assigns every segment to ``spk_0``.
    * ``create_diarizer()`` — factory that returns the correct backend
      based on config.  Drop-in replacements (e.g. pyannote) only need
      to subclass ``Diarizer`` and register in the factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.config import AppConfig


class Diarizer(ABC):
    """Base class for all diarization backends."""

    @abstractmethod
    def identify_speaker(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return a speaker_id string for the given audio segment.

        Must NEVER inspect or modify transcribed text.
        Must NEVER assign roles (patient / clinician / etc.).
        """

    def detect_speaker_change(
        self,
        previous_audio: np.ndarray | None,
        current_audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Return True if the speaker changed between two audio segments.

        Default implementation always returns False (single speaker).
        """
        return False


class DefaultDiarizer(Diarizer):
    """Single-speaker stub — assigns spk_0 to all audio."""

    def identify_speaker(self, audio: np.ndarray, sample_rate: int) -> str:
        return "spk_0"

    def detect_speaker_change(
        self,
        previous_audio: np.ndarray | None,
        current_audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        return False


def create_diarizer(config: AppConfig) -> Diarizer:
    """Factory — returns the diarization backend specified in config."""
    backend = config.diarization.backend
    if backend == "default":
        return DefaultDiarizer()
    raise ValueError(
        f"Unknown diarization backend: {backend!r}. "
        "Available: 'default'."
    )
