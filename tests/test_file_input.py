"""Tests for file-input mode (--input-file).

Tests mock ASREngine to avoid loading a real Whisper model.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.asr import AsrResult


# ── helpers ─────────────────────────────────────────────────────────


def _make_wav(path: Path, duration: float = 1.0, rate: int = 16000) -> None:
    """Create a minimal valid WAV file."""
    samples = int(rate * duration)
    audio = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


def _mock_asr(results: list[AsrResult] | None = None):
    """Create a mock ASREngine instance."""
    mock = MagicMock()
    mock.model_name = "test-model"
    mock.device = "cpu"
    mock.transcribe.return_value = results or []
    return mock


def _base_args(**overrides):
    """Create a minimal args namespace for _run_file_input."""
    defaults = dict(
        export_clinical_note=False,
        export_srt=False,
        export_vtt=False,
        export_summary=False,
        export_notes=None,
        note_template="soap",
        template_alias=None,
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


# ══════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════


class TestFileInputValidation:
    def test_file_not_found(self, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        config = AppConfig(output_dir=str(tmp_path / "out"))
        with pytest.raises(SystemExit):
            _run_file_input(config, _base_args(), str(tmp_path / "no.wav"))

    def test_non_wav_rejected(self, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        txt = tmp_path / "test.mp3"
        txt.write_bytes(b"\x00" * 100)
        config = AppConfig(output_dir=str(tmp_path / "out"))
        with pytest.raises(SystemExit):
            _run_file_input(config, _base_args(), str(txt))


# ══════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFileInputPipeline:
    @patch("app.main.ASREngine")
    def test_creates_raw_and_normalized(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=0.5, text="Hello"),
            AsrResult(start=0.5, end=1.0, text="World"),
        ])

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        raw_files = list(out.glob("raw_*.json"))
        assert len(raw_files) == 1
        norm_files = list(out.glob("normalized_*.json"))
        assert len(norm_files) == 1

        with open(norm_files[0], encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        texts = [s["normalized_text"] for s in data]
        assert "Hello" in texts
        assert "World" in texts

    @patch("app.main.ASREngine")
    def test_copies_audio(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav, duration=2.0)
        mock_cls.return_value = _mock_asr()

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        audio_files = list(out.glob("audio_*.wav"))
        assert len(audio_files) == 1
        with wave.open(str(audio_files[0]), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000

    @patch("app.main.ASREngine")
    def test_session_report_with_file_input_metadata(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=1.0, text="Test segment"),
        ])

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        reports = list(out.glob("session_report_*.json"))
        assert len(reports) == 1
        with open(reports[0], encoding="utf-8") as f:
            sr = json.load(f)
        assert sr["stats"]["segment_count"] == 1
        assert "file_input" in sr
        assert sr["file_input"]["sample_rate"] == 16000
        assert sr["file_input"]["source"].endswith("input.wav")

    @patch("app.main.ASREngine")
    def test_confidence_report_created(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([
            AsrResult(
                start=0.0, end=1.0, text="Hello",
                avg_logprob=-0.5, no_speech_prob=0.1, compression_ratio=1.2,
            ),
        ])

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        conf = list(out.glob("confidence_report_*.json"))
        assert len(conf) == 1

    @patch("app.main.ASREngine")
    def test_clinical_note_export(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=1.0, text="Patient reports chest pain"),
        ])

        config = AppConfig(output_dir=str(out))
        args = _base_args(export_clinical_note=True)
        _run_file_input(config, args, str(wav))

        notes = list(out.glob("clinical_note_*_soap.md"))
        assert len(notes) == 1

    @patch("app.main.ASREngine")
    def test_empty_transcription(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([])  # silence → no segments

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        raw_files = list(out.glob("raw_*.json"))
        assert len(raw_files) == 1
        norm_files = list(out.glob("normalized_*.json"))
        assert len(norm_files) == 1
        with open(norm_files[0], encoding="utf-8") as f:
            assert json.load(f) == []

    @patch("app.main.ASREngine")
    def test_stereo_wav_accepted(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "stereo.wav"
        out = tmp_path / "out"
        stereo = np.zeros(16000 * 2, dtype=np.int16)
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(stereo.tobytes())

        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=1.0, text="Stereo test"),
        ])

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        raw_files = list(out.glob("raw_*.json"))
        assert len(raw_files) == 1

    @patch("app.main.ASREngine")
    def test_changes_json_created(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=1.0, text="Test"),
        ])

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        changes = list(out.glob("changes_*.json"))
        assert len(changes) == 1

    @patch("app.main.ASREngine")
    def test_segments_have_correct_timestamps(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav, duration=5.0)
        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=2.0, text="First"),
            AsrResult(start=2.5, end=4.5, text="Second"),
        ])

        config = AppConfig(output_dir=str(out))
        _run_file_input(config, _base_args(), str(wav))

        norm_files = list(out.glob("normalized_*.json"))
        with open(norm_files[0], encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["t0"] == 0.0
        assert data[0]["t1"] == 2.0
        assert data[1]["t0"] == 2.5
        assert data[1]["t1"] == 4.5

    @patch("app.main.ASREngine")
    def test_export_summary(self, mock_cls, tmp_path):
        from app.main import _run_file_input
        from app.config import AppConfig

        wav = tmp_path / "input.wav"
        out = tmp_path / "out"
        _make_wav(wav)
        mock_cls.return_value = _mock_asr([
            AsrResult(start=0.0, end=1.0, text="Hello"),
        ])

        config = AppConfig(output_dir=str(out))
        args = _base_args(export_summary=True)
        _run_file_input(config, args, str(wav))

        summaries = list(out.glob("session_summary_*.md"))
        assert len(summaries) == 1
