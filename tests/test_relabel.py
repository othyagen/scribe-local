"""Tests for segment relabeling based on diarization overlap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.diarization import relabel_segments


# ── helpers ──────────────────────────────────────────────────────────

def _write_normalized(tmp_path: Path, segments: list[dict]) -> Path:
    p = tmp_path / "normalized_2026-01-01_12-00-00_large-v3.json"
    p.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return p


def _write_diarization(tmp_path: Path, turns: list[dict]) -> Path:
    p = tmp_path / "diarization_2026-01-01_12-00-00.json"
    p.write_text(json.dumps({"turns": turns}, ensure_ascii=False), encoding="utf-8")
    return p


def _seg(seg_id: str, t0: float, t1: float, text: str = "hello",
         speaker: str = "spk_0", para: str = "para_0000") -> dict:
    return {
        "seg_id": seg_id, "t0": t0, "t1": t1,
        "speaker_id": speaker, "raw_text": text,
        "normalized_text": text, "model_name": "test",
        "language": "en", "paragraph_id": para,
    }


def _turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


# ── overlap assignment ───────────────────────────────────────────────

class TestOverlapAssignment:
    def test_full_overlap_assigns_speaker(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 0.0, 3.0)])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_1")])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["new_speaker_id"] == "spk_1"

    def test_partial_overlap_picks_largest(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 2.0, 6.0)])
        diar = _write_diarization(tmp_path, [
            _turn(0.0, 3.0, "spk_0"),   # overlap: 1.0
            _turn(3.0, 8.0, "spk_1"),   # overlap: 3.0
        ])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["new_speaker_id"] == "spk_1"

    def test_no_overlap_keeps_original(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 10.0, 12.0)])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_1")])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["new_speaker_id"] == "spk_0"

    def test_multiple_turns_same_speaker_accumulated(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 0.0, 10.0)])
        diar = _write_diarization(tmp_path, [
            _turn(0.0, 2.0, "spk_0"),   # overlap: 2.0
            _turn(2.0, 5.0, "spk_1"),   # overlap: 3.0
            _turn(5.0, 10.0, "spk_0"),  # overlap: 5.0  (total spk_0: 7.0)
        ])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["new_speaker_id"] == "spk_0"

    def test_empty_turns_keeps_original(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 0.0, 3.0)])
        diar = _write_diarization(tmp_path, [])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["new_speaker_id"] == "spk_0"

    def test_empty_segments(self, tmp_path):
        norm = _write_normalized(tmp_path, [])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_0")])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data == []

    def test_multiple_segments_relabeled_independently(self, tmp_path):
        norm = _write_normalized(tmp_path, [
            _seg("seg_0001", 0.0, 3.0),
            _seg("seg_0002", 5.0, 8.0),
        ])
        diar = _write_diarization(tmp_path, [
            _turn(0.0, 4.0, "spk_1"),
            _turn(4.0, 9.0, "spk_2"),
        ])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["new_speaker_id"] == "spk_1"
        assert data[1]["new_speaker_id"] == "spk_2"


# ── JSON output format ──────────────────────────────────────────────

class TestJsonOutput:
    def test_json_file_created(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 0.0, 3.0)])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_1")])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        assert json_out.exists()
        assert json_out.name == "diarized_segments_2026-01-01_12-00-00.json"

    def test_json_fields(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 1.5, 4.2)])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_1")])
        json_out, _ = relabel_segments(norm, diar, str(tmp_path))
        data = json.loads(json_out.read_text(encoding="utf-8"))
        entry = data[0]
        assert entry["seg_id"] == "seg_0001"
        assert entry["t0"] == 1.5
        assert entry["t1"] == 4.2
        assert entry["old_speaker_id"] == "spk_0"
        assert entry["new_speaker_id"] == "spk_1"


# ── TXT output format ───────────────────────────────────────────────

class TestTxtOutput:
    def test_txt_file_created(self, tmp_path):
        norm = _write_normalized(tmp_path, [_seg("seg_0001", 0.0, 3.0)])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_1")])
        _, txt_out = relabel_segments(norm, diar, str(tmp_path))
        assert txt_out.exists()
        assert txt_out.name == "diarized_2026-01-01_12-00-00.txt"

    def test_txt_uses_new_speaker_id(self, tmp_path):
        norm = _write_normalized(tmp_path, [
            _seg("seg_0001", 0.0, 3.0, text="hello world"),
        ])
        diar = _write_diarization(tmp_path, [_turn(0.0, 5.0, "spk_1")])
        _, txt_out = relabel_segments(norm, diar, str(tmp_path))
        content = txt_out.read_text(encoding="utf-8")
        assert "[spk_1]" in content
        assert "hello world" in content

    def test_txt_paragraph_breaks(self, tmp_path):
        norm = _write_normalized(tmp_path, [
            _seg("seg_0001", 0.0, 3.0, para="para_0000"),
            _seg("seg_0002", 5.0, 8.0, para="para_0001"),
            _seg("seg_0003", 10.0, 13.0, para="para_0002"),
        ])
        diar = _write_diarization(tmp_path, [_turn(0.0, 14.0, "spk_0")])
        _, txt_out = relabel_segments(norm, diar, str(tmp_path))
        content = txt_out.read_text(encoding="utf-8")
        lines = content.split("\n")
        # Expect: line1, empty (paragraph break), line2, empty (paragraph break), line3
        assert lines[1] == ""
        assert lines[3] == ""

    def test_txt_empty_segments(self, tmp_path):
        norm = _write_normalized(tmp_path, [])
        diar = _write_diarization(tmp_path, [])
        _, txt_out = relabel_segments(norm, diar, str(tmp_path))
        content = txt_out.read_text(encoding="utf-8")
        assert content == ""
