"""Tests for --reprocess session mode."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import AppConfig, NormalizationConfig
from app.commit import RawSegment
from app.main import _reprocess_session, _atomic_write_json, _atomic_write_text, ReprocessError


TS = "2026-03-01_10-00-00"
MODEL_TAG = "large-v3"


def _raw_seg(seg_num: int, text: str, t0: float = 0.0, t1: float = 1.0,
             speaker: str = "spk_0") -> dict:
    """Create a RAW segment dict."""
    return {
        "seg_id": f"seg_{seg_num:04d}",
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker,
        "raw_text": text,
        "model_name": MODEL_TAG,
        "language": "en",
        "paragraph_id": "para_0000",
    }


def _write_raw(output_dir: Path, segments: list[dict]) -> Path:
    """Write a RAW JSONL file."""
    raw_path = output_dir / f"raw_{TS}_{MODEL_TAG}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    return raw_path


def _write_lexicon(lexicon_dir: Path, lang: str, replacements: dict) -> Path:
    """Create a custom.json lexicon."""
    path = lexicon_dir / lang / "custom.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"replacements": replacements}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_config(output_dir: Path, lexicon_dir: Path) -> AppConfig:
    """Build config pointing at test dirs."""
    config = AppConfig()
    config.output_dir = str(output_dir)
    config.normalization = NormalizationConfig(
        enabled=True, fuzzy_threshold=0.92,
        lexicon_dir=str(lexicon_dir),
    )
    config.language = "en"
    return config


def _write_diarization(output_dir: Path, turns: list[dict]) -> Path:
    """Write a diarization JSON file."""
    path = output_dir / f"diarization_{TS}.json"
    data = {"turns": turns}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class _FakeArgs:
    """Minimal args namespace for reprocess tests."""
    export_srt = False
    export_vtt = False
    export_summary = False


# ── normalization tests ──────────────────────────────────────────────


class TestReprocessNormalization:
    def test_renormalize_with_updated_lexicon(self, tmp_path):
        """New lexicon term is applied during reprocess."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {"pt": "patient"})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "The pt arrived.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        norm = json.loads((output_dir / f"normalized_{TS}_{MODEL_TAG}.json").read_text("utf-8"))
        assert norm[0]["normalized_text"] == "The patient arrived."

    def test_raw_unchanged_after_reprocess(self, tmp_path):
        """RAW JSONL is byte-identical before and after reprocess."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {"pt": "patient"})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        raw_path = _write_raw(output_dir, [_raw_seg(1, "The pt arrived.")])
        original_bytes = raw_path.read_bytes()

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        assert raw_path.read_bytes() == original_bytes

    def test_changes_log_reflects_new_normalization(self, tmp_path):
        """Changes JSON contains entries from new lexicon."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {"pt": "patient"})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "The pt is here.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        changes = json.loads((output_dir / f"changes_{TS}_{MODEL_TAG}.json").read_text("utf-8"))
        assert len(changes) >= 1
        assert changes[0]["from_text"] == "pt"
        assert changes[0]["to_text"] == "patient"

    def test_segment_count_preserved(self, tmp_path):
        """Same number of segments in normalized as in RAW."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        segs = [_raw_seg(i, f"Segment {i}.", t0=float(i), t1=float(i+1)) for i in range(1, 6)]
        _write_raw(output_dir, segs)

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        norm = json.loads((output_dir / f"normalized_{TS}_{MODEL_TAG}.json").read_text("utf-8"))
        assert len(norm) == 5

    def test_normalized_txt_written(self, tmp_path):
        """Plain text output exists and contains expected text."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "Hello world.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        txt_path = output_dir / f"normalized_{TS}_{MODEL_TAG}.txt"
        assert txt_path.exists()
        content = txt_path.read_text("utf-8")
        assert "Hello world." in content


# ── relabeling tests ─────────────────────────────────────────────────


class TestReprocessRelabeling:
    def test_relabeling_rerun_with_diarization(self, tmp_path):
        """Diarized segments regenerated from new normalized output."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [
            _raw_seg(1, "Hello.", t0=0.0, t1=2.0, speaker="spk_0"),
            _raw_seg(2, "World.", t0=3.0, t1=5.0, speaker="spk_0"),
        ])
        _write_diarization(output_dir, [
            {"speaker": "spk_0", "start": 0.0, "end": 2.5},
            {"speaker": "spk_1", "start": 2.5, "end": 6.0},
        ])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        diarized_path = output_dir / f"diarized_segments_{TS}.json"
        assert diarized_path.exists()
        diar_segs = json.loads(diarized_path.read_text("utf-8"))
        assert len(diar_segs) == 2

    def test_no_diarization_still_normalizes(self, tmp_path):
        """Missing diarization file does not block normalization."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {"pt": "patient"})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "The pt.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        norm = json.loads((output_dir / f"normalized_{TS}_{MODEL_TAG}.json").read_text("utf-8"))
        assert norm[0]["normalized_text"] == "The patient."
        assert not (output_dir / f"diarized_segments_{TS}.json").exists()

    def test_tagging_rerun_after_relabeling(self, tmp_path):
        """Speaker tags applied to refreshed relabeling."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [
            _raw_seg(1, "Hello.", t0=0.0, t1=2.0, speaker="spk_0"),
        ])
        _write_diarization(output_dir, [
            {"speaker": "spk_0", "start": 0.0, "end": 3.0},
        ])
        # Pre-existing speaker tags
        tags = {"spk_0": {"tag": "Doctor", "label": "Dr. Smith"}}
        tags_path = output_dir / f"speaker_tags_{TS}.json"
        tags_path.write_text(json.dumps(tags), encoding="utf-8")

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        tagged_path = output_dir / f"tag_labeled_{TS}.txt"
        assert tagged_path.exists()


# ── export tests ─────────────────────────────────────────────────────


class TestReprocessExports:
    def test_srt_export_after_reprocess(self, tmp_path):
        """--export-srt produces updated subtitle file."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [
            _raw_seg(1, "Hello.", t0=0.0, t1=2.0, speaker="spk_0"),
        ])
        _write_diarization(output_dir, [
            {"speaker": "spk_0", "start": 0.0, "end": 3.0},
        ])

        config = _make_config(output_dir, lexicon_dir)
        args = _FakeArgs()
        args.export_srt = True
        _reprocess_session(config, args, TS)

        srt_files = list(output_dir.glob(f"subtitles_{TS}.srt"))
        assert len(srt_files) == 1

    def test_summary_export_after_reprocess(self, tmp_path):
        """--export-summary produces updated summary file."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "Hello.")])

        config = _make_config(output_dir, lexicon_dir)
        args = _FakeArgs()
        args.export_summary = True
        _reprocess_session(config, args, TS)

        summary_files = list(output_dir.glob(f"session_summary_{TS}.md"))
        assert len(summary_files) == 1

    def test_no_export_flags_no_export_files(self, tmp_path):
        """Without export flags, no subtitle/summary written."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "Hello.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        assert not list(output_dir.glob("subtitles_*.srt"))
        assert not list(output_dir.glob("subtitles_*.vtt"))
        assert not list(output_dir.glob("session_summary_*.md"))


# ── report tests ─────────────────────────────────────────────────────


class TestReprocessReport:
    def test_session_report_has_reprocess_block(self, tmp_path):
        """Report contains 'reprocess' key with timestamp."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "Hello.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        report = json.loads((output_dir / f"session_report_{TS}.json").read_text("utf-8"))
        assert "reprocess" in report
        assert "reprocess_ts" in report["reprocess"]

    def test_report_lists_regenerated_outputs(self, tmp_path):
        """outputs_regenerated lists correct files."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "Hello.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        report = json.loads((output_dir / f"session_report_{TS}.json").read_text("utf-8"))
        regen = report["reprocess"]["outputs_regenerated"]
        assert "normalized" in regen
        assert "changes" in regen

    def test_report_segment_count_correct(self, tmp_path):
        """stats.segment_count matches RAW."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        segs = [_raw_seg(i, f"S{i}.", t0=float(i), t1=float(i+1)) for i in range(1, 4)]
        _write_raw(output_dir, segs)

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        report = json.loads((output_dir / f"session_report_{TS}.json").read_text("utf-8"))
        assert report["stats"]["segment_count"] == 3


# ── error tests ──────────────────────────────────────────────────────


class TestReprocessErrors:
    def test_missing_raw_file(self, tmp_path):
        """Clear error for missing RAW file."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})

        config = _make_config(output_dir, lexicon_dir)
        with pytest.raises(ReprocessError, match="no RAW file found"):
            _reprocess_session(config, _FakeArgs(), "1999-01-01_00-00-00")

    def test_multiple_raw_files(self, tmp_path):
        """Clear error for ambiguous session (multiple RAW files)."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        # Create two RAW files for same timestamp
        _write_raw(output_dir, [_raw_seg(1, "A.")])
        raw2 = output_dir / f"raw_{TS}_other-model.json"
        raw2.write_text(json.dumps(_raw_seg(1, "B.")) + "\n", encoding="utf-8")

        config = _make_config(output_dir, lexicon_dir)
        with pytest.raises(ReprocessError, match="multiple RAW files"):
            _reprocess_session(config, _FakeArgs(), TS)

    def test_corrupt_raw_jsonl(self, tmp_path):
        """Handles malformed line gracefully (skip + warn)."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        raw_path = output_dir / f"raw_{TS}_{MODEL_TAG}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_raw_seg(1, "Good.")) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(_raw_seg(2, "Also good.", t0=2.0, t1=3.0)) + "\n")

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        norm = json.loads((output_dir / f"normalized_{TS}_{MODEL_TAG}.json").read_text("utf-8"))
        assert len(norm) == 2  # corrupt line skipped


# ── atomic write tests ───────────────────────────────────────────────


class TestReprocessAtomicWrite:
    def test_normalized_written_atomically(self, tmp_path):
        """No .tmp file left after success."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        _write_raw(output_dir, [_raw_seg(1, "Hello.")])

        config = _make_config(output_dir, lexicon_dir)
        _reprocess_session(config, _FakeArgs(), TS)

        tmp_files = list(output_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_raw_not_opened_for_writing(self, tmp_path):
        """RAW file is only read, never written."""
        lexicon_dir = tmp_path / "lexicons"
        _write_lexicon(lexicon_dir, "en", {})
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        raw_path = _write_raw(output_dir, [_raw_seg(1, "Hello.")])
        original_content = raw_path.read_bytes()

        config = _make_config(output_dir, lexicon_dir)

        # Track all open() calls
        original_open = open
        write_opens_to_raw = []

        raw_filename = raw_path.name

        def tracked_open(path, *a, **kw):
            path_str = str(path)
            if path_str.endswith(raw_filename):
                mode = a[0] if a else kw.get("mode", "r")
                if "w" in mode or "a" in mode:
                    write_opens_to_raw.append(path_str)
            return original_open(path, *a, **kw)

        with patch("builtins.open", side_effect=tracked_open):
            _reprocess_session(config, _FakeArgs(), TS)

        assert len(write_opens_to_raw) == 0
        assert raw_path.read_bytes() == original_content


# ── atomic helper unit tests ─────────────────────────────────────────


class TestAtomicHelpers:
    def test_atomic_write_json(self, tmp_path):
        path = tmp_path / "test.json"
        _atomic_write_json(path, {"key": "value"})
        assert path.exists()
        assert json.loads(path.read_text("utf-8")) == {"key": "value"}
        assert not (tmp_path / "test.tmp").exists()

    def test_atomic_write_text(self, tmp_path):
        path = tmp_path / "test.txt"
        _atomic_write_text(path, "hello\nworld\n")
        assert path.exists()
        assert path.read_text("utf-8") == "hello\nworld\n"
        assert not (tmp_path / "test.tmp").exists()
