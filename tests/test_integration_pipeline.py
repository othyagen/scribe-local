"""End-to-end integration test for the full pipeline.

Exercises: commit → normalize → OutputWriter → WAV → pyannote diarization
→ segment relabeling → speaker tagging, all without a live microphone.
"""

from __future__ import annotations

import json
import sys
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import AppConfig, NormalizationConfig, DiarizationConfig
from app.commit import SegmentCommitter
from app.normalize import Normalizer
from app.io import OutputWriter
from app.diarization import relabel_segments, run_pyannote_diarization
from app.tagging import (
    load_or_create_tags,
    apply_auto_tags,
    save_tags,
    generate_tag_labeled_txt,
)


# ── constants ────────────────────────────────────────────────────────

TS = "2026-01-01_12-00-00"
MODEL = "test-model"
LANGUAGE = "en"
SAMPLE_RATE = 16000


# ── mock helpers (same pattern as test_diarization.py) ───────────────

@dataclass
class FakeTurn:
    start: float
    end: float


def _make_fake_annotation(tracks: list[tuple[float, float, str]]):
    mock = MagicMock()
    mock.itertracks.return_value = [
        (FakeTurn(start=s, end=e), None, label)
        for s, e, label in tracks
    ]
    return mock


def _make_fake_diarization(tracks: list[tuple[float, float, str]]):
    annotation = _make_fake_annotation(tracks)
    mock = MagicMock()
    mock.speaker_diarization = annotation
    return mock


@pytest.fixture()
def mock_pyannote_pipeline(monkeypatch):
    mock_pipeline_cls = MagicMock()
    fake_module = ModuleType("pyannote.audio")
    fake_module.Pipeline = mock_pipeline_cls
    monkeypatch.setitem(sys.modules, "pyannote", ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    return mock_pipeline_cls


# ── helpers ──────────────────────────────────────────────────────────

def _write_wav(tmp_path: Path) -> Path:
    """Write a 10-second silence WAV."""
    p = tmp_path / f"audio_{TS}.wav"
    samples = np.zeros(SAMPLE_RATE * 10, dtype=np.int16)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    return p


def _create_lexicons(tmp_path: Path) -> Path:
    """Create minimal English lexicons with one testable replacement."""
    lex_dir = tmp_path / "lexicons" / "en"
    lex_dir.mkdir(parents=True)
    for name in ("custom", "medical"):
        (lex_dir / f"{name}.json").write_text(
            json.dumps({"replacements": {}}), encoding="utf-8"
        )
    (lex_dir / "general.json").write_text(
        json.dumps({"replacements": {"teh": "the"}}), encoding="utf-8"
    )
    return tmp_path / "lexicons"


# ── the integration test ─────────────────────────────────────────────

class _FixedDatetime(datetime):
    """datetime subclass that returns a fixed timestamp from now().

    Only used to patch app.io.datetime so OutputWriter generates
    predictable filenames.  Does NOT affect the global datetime.
    """
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 12, 0, 0)


def test_full_pipeline(tmp_path, mock_pyannote_pipeline, monkeypatch):
    """Full pipeline: commit → normalize → write → WAV → diarize → relabel → tag."""

    output_dir = str(tmp_path)
    lex_dir = _create_lexicons(tmp_path)

    # ── config ────────────────────────────────────────────────────
    config = AppConfig(
        language=LANGUAGE,
        normalization=NormalizationConfig(
            enabled=True,
            fuzzy_threshold=0.92,
            lexicon_dir=str(lex_dir),
        ),
        diarization=DiarizationConfig(backend="pyannote"),
        output_dir=output_dir,
    )

    # Patch only the datetime in app.io so OutputWriter generates
    # predictable filenames — does NOT touch the global datetime.
    monkeypatch.setattr("app.io.datetime", _FixedDatetime)

    # ── 1. commit + normalize + write ─────────────────────────────
    committer = SegmentCommitter(MODEL, LANGUAGE)
    normalizer = Normalizer(config)
    writer = OutputWriter(output_dir, MODEL)

    # 4 segments: first two spoken by spk_0, last two by spk_0
    # (all spk_0 during recording; diarization relabels later)
    segments_data = [
        (0.0, 3.0, "spk_0", "hello world"),
        (3.0, 5.0, "spk_0", "teh quick fox"),      # "teh" → "the"
        (5.0, 8.0, "spk_0", "good morning"),
        (8.0, 10.0, "spk_0", "nice to meet you"),
    ]

    for t0, t1, spk, text in segments_data:
        raw_seg = committer.commit(t0, t1, spk, text)
        writer.append_raw(raw_seg)
        norm_seg, changes = normalizer.normalize(raw_seg)
        writer.add_normalized(norm_seg, changes)

    # Snapshot RAW content before finalize (to verify immutability later)
    raw_json_content = writer.raw_json_path.read_text(encoding="utf-8")

    writer.finalize(timeout=5.0)

    # ── 2. WAV ────────────────────────────────────────────────────
    wav_path = _write_wav(tmp_path)

    # ── 3. Pyannote diarization (mocked) ──────────────────────────
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    mock_pipeline = MagicMock()
    mock_pyannote_pipeline.from_pretrained.return_value = mock_pipeline
    mock_pipeline.return_value = _make_fake_diarization([
        (0.0, 5.0, "SPEAKER_00"),
        (5.0, 10.0, "SPEAKER_01"),
    ])

    diar_path = run_pyannote_diarization(wav_path, output_dir)

    # ── 4. Relabel segments ───────────────────────────────────────
    diarized_json, diarized_txt = relabel_segments(
        writer.normalized_json_path, diar_path, output_dir
    )

    # ── 5. Speaker tagging ────────────────────────────────────────
    diar_ts = diarized_txt.stem.removeprefix("diarized_")
    tags = load_or_create_tags(output_dir, diar_ts)
    speakers = sorted(set(
        s["new_speaker_id"]
        for s in json.loads(diarized_json.read_text(encoding="utf-8"))
    ))
    apply_auto_tags(tags, "alphabetical", speakers)
    tags_path = save_tags(tags, output_dir, diar_ts)
    tagged_txt = generate_tag_labeled_txt(
        diarized_txt, tags, output_dir, diar_ts
    )

    # ═════════════════════════════════════════════════════════════
    # ASSERTIONS
    # ═════════════════════════════════════════════════════════════

    # ── all output files exist ────────────────────────────────────
    tag = MODEL
    expected_files = [
        f"raw_{TS}_{tag}.json",
        f"raw_{TS}_{tag}.txt",
        f"normalized_{TS}_{tag}.json",
        f"normalized_{TS}_{tag}.txt",
        f"changes_{TS}_{tag}.json",
        f"audio_{TS}.wav",
        f"diarization_{TS}.json",
        f"diarized_segments_{TS}.json",
        f"diarized_{TS}.txt",
        f"speaker_tags_{TS}.json",
        f"tag_labeled_{TS}.txt",
    ]
    for fname in expected_files:
        assert (tmp_path / fname).exists(), f"Missing: {fname}"

    # ── RAW JSON Lines: 4 lines, each valid JSON ─────────────────
    raw_lines = raw_json_content.strip().split("\n")
    assert len(raw_lines) == 4
    for line in raw_lines:
        obj = json.loads(line)
        assert "seg_id" in obj
        assert "raw_text" in obj
        assert "speaker_id" in obj
        assert "t0" in obj and "t1" in obj

    # ── RAW immutability: content unchanged after pipeline ────────
    raw_after = writer.raw_json_path.read_text(encoding="utf-8")
    assert raw_after == raw_json_content

    # ── Normalized JSON: array of 4, has normalized_text ──────────
    norm_data = json.loads(
        (tmp_path / f"normalized_{TS}_{tag}.json").read_text(encoding="utf-8")
    )
    assert isinstance(norm_data, list)
    assert len(norm_data) == 4
    for seg in norm_data:
        assert "normalized_text" in seg
        assert "raw_text" in seg

    # ── Normalization applied: "teh" → "the" ─────────────────────
    seg_with_fix = norm_data[1]  # second segment had "teh quick fox"
    assert seg_with_fix["raw_text"] == "teh quick fox"
    assert seg_with_fix["normalized_text"] == "the quick fox"

    # ── Changes log: at least one change recorded ─────────────────
    changes_data = json.loads(
        (tmp_path / f"changes_{TS}_{tag}.json").read_text(encoding="utf-8")
    )
    assert len(changes_data) >= 1
    assert any(c["from_text"] == "teh" and c["to_text"] == "the" for c in changes_data)

    # ── Diarization JSON: 2 turns, 2 distinct speakers ───────────
    diar_data = json.loads(diar_path.read_text(encoding="utf-8"))
    assert len(diar_data["turns"]) == 2
    diar_speakers = set(t["speaker"] for t in diar_data["turns"])
    assert len(diar_speakers) >= 2

    # ── Diarized segments: both spk_0 and spk_1 present ──────────
    diarized_seg_data = json.loads(diarized_json.read_text(encoding="utf-8"))
    relabeled_speakers = set(s["new_speaker_id"] for s in diarized_seg_data)
    assert "spk_0" in relabeled_speakers
    assert "spk_1" in relabeled_speakers

    # ── Diarized txt: has both speaker tokens ─────────────────────
    diarized_content = diarized_txt.read_text(encoding="utf-8")
    assert "[spk_0]" in diarized_content
    assert "[spk_1]" in diarized_content

    # ── Speaker tags: both speakers tagged ────────────────────────
    tags_data = json.loads(tags_path.read_text(encoding="utf-8"))
    assert "spk_0" in tags_data
    assert "spk_1" in tags_data
    assert tags_data["spk_0"]["tag"] == "Speaker A"
    assert tags_data["spk_1"]["tag"] == "Speaker B"

    # ── Tag-labeled txt: tags applied, no raw spk_N ──────────────
    tagged_content = tagged_txt.read_text(encoding="utf-8")
    assert "[Speaker A]" in tagged_content
    assert "[Speaker B]" in tagged_content
    assert "[spk_0]" not in tagged_content
    assert "[spk_1]" not in tagged_content

    # ── WAV file is valid ─────────────────────────────────────────
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
