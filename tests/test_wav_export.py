"""Tests for _write_session_wav."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from app.main import _write_session_wav


class TestWriteSessionWav:
    """Unit tests for the WAV export helper."""

    def test_returns_none_for_empty_chunks(self, tmp_path):
        result = _write_session_wav([], 16000, str(tmp_path))
        assert result is None

    def test_creates_wav_file(self, tmp_path):
        chunks = [np.zeros(1600, dtype=np.float32)]
        path = _write_session_wav(chunks, 16000, str(tmp_path))
        assert path is not None
        assert path.exists()
        assert path.suffix == ".wav"

    def test_wav_filename_format(self, tmp_path):
        chunks = [np.zeros(1600, dtype=np.float32)]
        path = _write_session_wav(chunks, 16000, str(tmp_path))
        assert path.name.startswith("audio_")
        assert path.name.endswith(".wav")

    def test_wav_header_params(self, tmp_path):
        chunks = [np.zeros(1600, dtype=np.float32)]
        path = _write_session_wav(chunks, 16000, str(tmp_path))
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_wav_frame_count_single_chunk(self, tmp_path):
        chunks = [np.zeros(1600, dtype=np.float32)]
        path = _write_session_wav(chunks, 16000, str(tmp_path))
        with wave.open(str(path), "rb") as wf:
            assert wf.getnframes() == 1600

    def test_wav_frame_count_multiple_chunks(self, tmp_path):
        chunks = [np.zeros(1600, dtype=np.float32) for _ in range(5)]
        path = _write_session_wav(chunks, 16000, str(tmp_path))
        with wave.open(str(path), "rb") as wf:
            assert wf.getnframes() == 8000

    def test_audio_content_roundtrip(self, tmp_path):
        """Verify audio samples survive the float32 → int16 → read-back cycle."""
        rng = np.random.default_rng(42)
        original = rng.uniform(-0.5, 0.5, size=3200).astype(np.float32)
        chunks = [original[:1600], original[1600:]]

        path = _write_session_wav(chunks, 16000, str(tmp_path))
        with wave.open(str(path), "rb") as wf:
            raw_bytes = wf.readframes(wf.getnframes())

        read_back = np.frombuffer(raw_bytes, dtype=np.int16)
        expected = (original * 32767).clip(-32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(read_back, expected)

    def test_clipping_at_boundaries(self, tmp_path):
        """Values outside [-1, 1] should be clipped, not wrap around."""
        audio = np.array([1.5, -1.5, 1.0, -1.0, 0.0], dtype=np.float32)
        path = _write_session_wav([audio], 16000, str(tmp_path))
        with wave.open(str(path), "rb") as wf:
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        assert samples[0] == 32767   # clipped max
        assert samples[1] == -32768  # clipped min
        assert samples[4] == 0

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        chunks = [np.zeros(1600, dtype=np.float32)]
        path = _write_session_wav(chunks, 16000, str(nested))
        assert path.exists()
        assert nested.exists()
