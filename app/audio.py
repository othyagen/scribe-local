"""Microphone audio capture using sounddevice."""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from app.config import AppConfig


class AudioCapture:
    """Streams audio from a microphone into a thread-safe queue."""

    def __init__(self, config: AppConfig) -> None:
        self.sample_rate: int = config.audio.sample_rate
        self.device: Optional[int] = config.audio.device
        self.channels: int = config.audio.channels
        self.chunk_samples: int = int(
            self.sample_rate * config.vad.chunk_duration_ms / 1000
        )
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[audio] status: {status}")
        # indata shape: (frames, channels) – copy to decouple from driver buffer
        self._queue.put(indata[:, 0].copy())

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the microphone stream."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._callback,
        )
        self._stream.start()

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray:
        """Block until the next audio chunk is available.

        Returns a 1-D float32 numpy array of *chunk_samples* samples.
        Raises ``queue.Empty`` on timeout.
        """
        return self._queue.get(timeout=timeout)

    def stop(self, timeout: float = 3.0) -> None:
        """Close the microphone stream with a bounded timeout.

        Runs abort/stop/close in a helper thread so that PortAudio driver
        hangs cannot block shutdown indefinitely.
        """
        if self._stream is None:
            return

        def _close() -> None:
            try:
                self._stream.abort()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self._stream.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self._stream.close()  # type: ignore[union-attr]
            except Exception:
                pass

        t = threading.Thread(target=_close, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            print(
                f"[shutdown] WARNING: audio.stop() did not complete within "
                f"{timeout}s — continuing shutdown"
            )
        self._stream = None


def list_devices() -> None:
    """Print all available audio devices to stdout."""
    print(sd.query_devices())
