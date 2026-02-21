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

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

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
    """Factory — returns the diarization backend specified in config.

    When backend is "pyannote", real-time diarization still uses the
    default single-speaker stub.  Post-session diarization on the full
    WAV is handled separately by ``run_pyannote_diarization()``.
    """
    backend = config.diarization.backend
    if backend in ("default", "pyannote"):
        return DefaultDiarizer()
    raise ValueError(
        f"Unknown diarization backend: {backend!r}. "
        "Available: 'default', 'pyannote'."
    )


def run_pyannote_diarization(wav_path: Path, output_dir: str) -> Path:
    """Run pyannote speaker diarization on a WAV file.

    Requires ``HF_TOKEN`` environment variable for Hugging Face auth.
    Writes ``diarization_<timestamp>.json`` to *output_dir*.

    Returns the path to the written JSON file.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required for pyannote diarization"
        )

    from pyannote.audio import Pipeline  # lazy import — heavy dependency

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    diarization = pipeline(str(wav_path))

    # Map pyannote labels to spk_0, spk_1, ... (ordered by first appearance)
    label_map: dict[str, str] = {}
    turns: list[dict] = []

    for turn, _, label in diarization.itertracks(yield_label=True):
        if label not in label_map:
            label_map[label] = f"spk_{len(label_map)}"
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": label_map[label],
        })

    # Extract timestamp from WAV filename: audio_<timestamp>.wav
    ts = wav_path.stem.removeprefix("audio_")
    out = Path(output_dir) / f"diarization_{ts}.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"turns": turns}, f, ensure_ascii=False, indent=2)

    return out
