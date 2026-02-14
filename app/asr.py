"""Streaming ASR using faster-whisper / CTranslate2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.config import AppConfig


@dataclass
class AsrResult:
    """A single transcription segment returned by the ASR engine."""
    start: float   # seconds, relative to audio chunk start
    end: float     # seconds, relative to audio chunk start
    text: str


class ASREngine:
    """Wraps faster-whisper for near–real-time transcription."""

    def __init__(self, config: AppConfig) -> None:
        from faster_whisper import WhisperModel

        device = self._resolve_device(config.asr.device)
        compute_type = config.asr.compute_type

        # float16 is unsupported on CPU — fall back to int8
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        self.model = WhisperModel(
            config.asr.model,
            device=device,
            compute_type=compute_type,
        )
        self.language: str = config.language
        self.model_name: str = config.asr.model
        self._device: str = device
        self._compute_type: str = compute_type

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device_str: str) -> str:
        """Determine the compute device (cuda / cpu)."""
        if device_str in ("cuda", "cpu"):
            return device_str
        # auto-detect
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> List[AsrResult]:
        """Transcribe a float32 mono audio buffer at 16 kHz.

        Returns a list of ``AsrResult`` segments.  Empty segments are
        silently dropped.
        """
        segments, _info = self.model.transcribe(
            audio.squeeze(),
            language=self.language,
            beam_size=5,
            vad_filter=False,      # we handle VAD ourselves
            word_timestamps=False,
        )

        results: List[AsrResult] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append(AsrResult(start=seg.start, end=seg.end, text=text))
        return results

    @property
    def device(self) -> str:
        return self._device
