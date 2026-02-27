"""Streaming ASR using faster-whisper / CTranslate2."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.config import AppConfig


def _register_nvidia_dll_dirs() -> None:
    """Add nvidia package DLL dirs to the search path (Windows only).

    Pip-installed nvidia-cublas-cu12 (and similar) place DLLs under
    site-packages/nvidia/<lib>/bin/ which Python doesn't search by default.
    We prepend to PATH so that both Python and native C extensions (ctranslate2)
    can find them.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
        nvidia_root = os.path.dirname(nvidia.__path__[0])
        nvidia_pkg = os.path.join(nvidia_root, "nvidia")
        dirs_to_add = []
        for pkg in os.listdir(nvidia_pkg):
            bin_dir = os.path.join(nvidia_pkg, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                dirs_to_add.append(bin_dir)
        if dirs_to_add:
            os.environ["PATH"] = os.pathsep.join(dirs_to_add) + os.pathsep + os.environ.get("PATH", "")
    except (ImportError, OSError):
        pass


_register_nvidia_dll_dirs()


@dataclass
class AsrResult:
    """A single transcription segment returned by the ASR engine."""
    start: float   # seconds, relative to audio chunk start
    end: float     # seconds, relative to audio chunk start
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None


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
                results.append(AsrResult(
                    start=seg.start, end=seg.end, text=text,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                    compression_ratio=getattr(seg, "compression_ratio", None),
                ))
        return results

    @property
    def device(self) -> str:
        return self._device
