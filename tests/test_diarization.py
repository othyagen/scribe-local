"""Tests for diarization module."""

from __future__ import annotations

import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.config import AppConfig, DiarizationConfig
from app.diarization import (
    DefaultDiarizer,
    create_diarizer,
    run_pyannote_diarization,
)


# ── mock helpers ─────────────────────────────────────────────────────

@dataclass
class FakeTurn:
    """Mimics a pyannote Segment with start/end attributes."""
    start: float
    end: float


def _make_fake_annotation(tracks: list[tuple[float, float, str]]):
    """Build a mock Annotation with itertracks from (start, end, label) tuples."""
    mock = MagicMock()
    mock.itertracks.return_value = [
        (FakeTurn(start=s, end=e), None, label)
        for s, e, label in tracks
    ]
    return mock


def _make_fake_diarization(tracks: list[tuple[float, float, str]]):
    """Build a mock DiarizeOutput from (start, end, label) tuples."""
    annotation = _make_fake_annotation(tracks)
    mock = MagicMock()
    mock.speaker_diarization = annotation
    return mock


def _make_wav_path(tmp_path: Path, ts: str = "2026-01-01_12-00-00") -> Path:
    """Create a valid 16kHz mono 16-bit WAV file with silence."""
    p = tmp_path / f"audio_{ts}.wav"
    samples = np.zeros(16000, dtype=np.int16)  # 1 second of silence
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return p


@pytest.fixture()
def mock_pyannote_pipeline(monkeypatch):
    """Inject a fake pyannote.audio module with a mock Pipeline class.

    Returns the mock Pipeline class so tests can configure it.
    """
    mock_pipeline_cls = MagicMock()

    # Build a fake pyannote.audio module with Pipeline attribute
    fake_module = ModuleType("pyannote.audio")
    fake_module.Pipeline = mock_pipeline_cls

    monkeypatch.setitem(sys.modules, "pyannote", ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    return mock_pipeline_cls


# ── DefaultDiarizer ─────────────────────────────────────────────────

class TestDefaultDiarizer:
    def test_always_returns_spk_0(self):
        d = DefaultDiarizer()
        audio = np.zeros(1600, dtype=np.float32)
        assert d.identify_speaker(audio, 16000) == "spk_0"

    def test_no_speaker_change(self):
        d = DefaultDiarizer()
        audio = np.zeros(1600, dtype=np.float32)
        assert d.detect_speaker_change(None, audio, 16000) is False
        assert d.detect_speaker_change(audio, audio, 16000) is False


# ── create_diarizer factory ──────────────────────────────────────────

class TestCreateDiarizer:
    def test_default_backend(self):
        config = AppConfig(diarization=DiarizationConfig(backend="default"))
        d = create_diarizer(config)
        assert isinstance(d, DefaultDiarizer)

    def test_pyannote_backend_returns_default_diarizer(self):
        config = AppConfig(diarization=DiarizationConfig(backend="pyannote"))
        d = create_diarizer(config)
        assert isinstance(d, DefaultDiarizer)

    def test_unknown_backend_raises(self):
        config = AppConfig(diarization=DiarizationConfig(backend="unknown"))
        with pytest.raises(ValueError, match="Unknown diarization backend"):
            create_diarizer(config)


# ── run_pyannote_diarization ─────────────────────────────────────────

class TestRunPyannoteDiarization:
    def test_raises_without_hf_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        wav = _make_wav_path(tmp_path)
        with pytest.raises(ValueError, match="HF_TOKEN"):
            run_pyannote_diarization(wav, str(tmp_path))

    def test_writes_json_file(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([
            (0.0, 3.2, "SPEAKER_00"),
            (3.2, 5.1, "SPEAKER_01"),
        ])

        wav = _make_wav_path(tmp_path, "2026-01-15_10-30-00")
        result = run_pyannote_diarization(wav, str(tmp_path))

        assert result.exists()
        assert result.name == "diarization_2026-01-15_10-30-00.json"

    def test_json_structure(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([
            (0.0, 3.2, "SPEAKER_00"),
            (3.2, 5.1, "SPEAKER_01"),
        ])

        wav = _make_wav_path(tmp_path)
        result = run_pyannote_diarization(wav, str(tmp_path))

        data = json.loads(result.read_text(encoding="utf-8"))
        assert "turns" in data
        assert len(data["turns"]) == 2
        for turn in data["turns"]:
            assert "start" in turn
            assert "end" in turn
            assert "speaker" in turn

    def test_speaker_labels_ordered_by_appearance(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([
            (0.0, 2.0, "SPEAKER_02"),
            (2.0, 4.0, "SPEAKER_00"),
            (4.0, 6.0, "SPEAKER_02"),
        ])

        wav = _make_wav_path(tmp_path)
        result = run_pyannote_diarization(wav, str(tmp_path))
        data = json.loads(result.read_text(encoding="utf-8"))

        assert data["turns"][0]["speaker"] == "spk_0"
        assert data["turns"][1]["speaker"] == "spk_1"
        assert data["turns"][2]["speaker"] == "spk_0"  # same as first

    def test_timestamps_rounded_to_3_decimals(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([
            (0.12345678, 3.98765432, "SPEAKER_00"),
        ])

        wav = _make_wav_path(tmp_path)
        result = run_pyannote_diarization(wav, str(tmp_path))
        data = json.loads(result.read_text(encoding="utf-8"))

        assert data["turns"][0]["start"] == 0.123
        assert data["turns"][0]["end"] == 3.988

    def test_empty_diarization(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([])

        wav = _make_wav_path(tmp_path)
        result = run_pyannote_diarization(wav, str(tmp_path))
        data = json.loads(result.read_text(encoding="utf-8"))

        assert data["turns"] == []

    def test_single_speaker_session(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_00"),
        ])

        wav = _make_wav_path(tmp_path)
        result = run_pyannote_diarization(wav, str(tmp_path))
        data = json.loads(result.read_text(encoding="utf-8"))

        assert all(t["speaker"] == "spk_0" for t in data["turns"])

    def test_pipeline_called_with_waveform_dict(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([])

        wav = _make_wav_path(tmp_path)
        run_pyannote_diarization(wav, str(tmp_path))

        mock_pipeline.assert_called_once()
        call_arg = mock_pipeline.call_args[0][0]
        assert "waveform" in call_arg
        assert "sample_rate" in call_arg
        assert call_arg["sample_rate"] == 16000

    def test_pipeline_uses_hf_token(
        self, mock_pyannote_pipeline, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_TOKEN", "my-secret-token")
        mock_pipeline = MagicMock()
        mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = _make_fake_diarization([])

        wav = _make_wav_path(tmp_path)
        run_pyannote_diarization(wav, str(tmp_path))

        mock_pyannote_pipeline.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            token="my-secret-token",
        )
