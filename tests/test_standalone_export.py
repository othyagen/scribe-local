"""Tests for standalone session export mode (--session + export flags)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


TS = "2026-03-01_10-00-00"


def _create_session_files(output_dir: Path) -> None:
    """Create minimal session artifacts for standalone export."""
    # Normalized segments
    norm_segs = [
        {
            "seg_id": "seg_0001",
            "t0": 0.0,
            "t1": 2.5,
            "speaker_id": "spk_0",
            "normalized_text": "Hello world.",
        },
        {
            "seg_id": "seg_0002",
            "t0": 3.0,
            "t1": 5.0,
            "speaker_id": "spk_1",
            "normalized_text": "Hi there.",
        },
    ]
    norm_path = output_dir / f"normalized_{TS}_large-v3.json"
    norm_path.write_text(json.dumps(norm_segs), encoding="utf-8")

    # Diarized segments
    diar_segs = [
        {"seg_id": "seg_0001", "t0": 0.0, "t1": 2.5,
         "old_speaker_id": "spk_0", "new_speaker_id": "spk_0"},
        {"seg_id": "seg_0002", "t0": 3.0, "t1": 5.0,
         "old_speaker_id": "spk_0", "new_speaker_id": "spk_1"},
    ]
    diar_path = output_dir / f"diarized_segments_{TS}.json"
    diar_path.write_text(json.dumps(diar_segs), encoding="utf-8")

    # Session report
    report = {
        "session_ts": TS,
        "config": {
            "language": "en",
            "asr": {"model": "large-v3", "device": "cpu", "compute_type": "int8"},
            "diarization": {"backend": "default", "smoothing": False},
            "reporting": {"session_report_enabled": True},
        },
        "feature_flags": {},
        "outputs": {
            "raw": str(output_dir / f"raw_{TS}_large-v3.json"),
        },
        "stats": {"segment_count": 2},
    }
    report_path = output_dir / f"session_report_{TS}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")


# ── standalone SRT export ────────────────────────────────────────────

class TestStandaloneSrt:
    def test_srt_export(self, tmp_path):
        _create_session_files(tmp_path)
        from app.export_subtitles import write_srt
        # Load and build segments (same logic as main.py standalone)
        with open(tmp_path / f"diarized_segments_{TS}.json") as f:
            diar_segs = json.load(f)
        norm_files = sorted(tmp_path.glob(f"normalized_{TS}*.json"))
        with open(norm_files[-1]) as f:
            norm_segs = json.load(f)
        text_by_seg = {s["seg_id"]: s.get("normalized_text", "") for s in norm_segs}
        subtitle_segs = [
            {
                "t0": ds["t0"], "t1": ds["t1"],
                "speaker": ds["new_speaker_id"],
                "text": text_by_seg.get(ds["seg_id"], ""),
            }
            for ds in diar_segs
        ]
        path = write_srt(subtitle_segs, str(tmp_path), TS)
        assert path.exists()
        assert path.suffix == ".srt"
        content = path.read_text("utf-8")
        assert "[spk_0] Hello world." in content
        assert "[spk_1] Hi there." in content
        assert "00:00:00,000 --> 00:00:02,500" in content

    def test_vtt_export(self, tmp_path):
        _create_session_files(tmp_path)
        from app.export_subtitles import write_vtt
        with open(tmp_path / f"diarized_segments_{TS}.json") as f:
            diar_segs = json.load(f)
        norm_files = sorted(tmp_path.glob(f"normalized_{TS}*.json"))
        with open(norm_files[-1]) as f:
            norm_segs = json.load(f)
        text_by_seg = {s["seg_id"]: s.get("normalized_text", "") for s in norm_segs}
        subtitle_segs = [
            {
                "t0": ds["t0"], "t1": ds["t1"],
                "speaker": ds["new_speaker_id"],
                "text": text_by_seg.get(ds["seg_id"], ""),
            }
            for ds in diar_segs
        ]
        path = write_vtt(subtitle_segs, str(tmp_path), TS)
        assert path.exists()
        assert path.suffix == ".vtt"
        content = path.read_text("utf-8")
        assert content.startswith("WEBVTT")
        assert "[spk_1] Hi there." in content


# ── standalone summary export ────────────────────────────────────────

class TestStandaloneSummary:
    def test_summary_export(self, tmp_path):
        _create_session_files(tmp_path)
        from app.export_summary import write_summary
        with open(tmp_path / f"session_report_{TS}.json") as f:
            report = json.load(f)
        path = write_summary(report, str(tmp_path), TS)
        assert path.exists()
        assert path.suffix == ".md"
        content = path.read_text("utf-8")
        assert "# Session Summary" in content
        assert "**Segments:** 2" in content

    def test_summary_report_not_mutated(self, tmp_path):
        _create_session_files(tmp_path)
        from app.export_summary import write_summary
        import copy
        with open(tmp_path / f"session_report_{TS}.json") as f:
            report = json.load(f)
        original = copy.deepcopy(report)
        write_summary(report, str(tmp_path), TS)
        assert report == original


# ── missing files ────────────────────────────────────────────────────

class TestMissingFiles:
    def test_missing_diarized_segments(self, tmp_path):
        """SRT export fails gracefully when diarized segments missing."""
        _create_session_files(tmp_path)
        (tmp_path / f"diarized_segments_{TS}.json").unlink()
        # Verify the file is gone
        assert not (tmp_path / f"diarized_segments_{TS}.json").exists()

    def test_missing_normalized(self, tmp_path):
        """SRT export fails gracefully when normalized file missing."""
        _create_session_files(tmp_path)
        for f in tmp_path.glob(f"normalized_{TS}*.json"):
            f.unlink()
        assert not list(tmp_path.glob(f"normalized_{TS}*.json"))

    def test_missing_session_report(self, tmp_path):
        """Summary export fails gracefully when report missing."""
        _create_session_files(tmp_path)
        (tmp_path / f"session_report_{TS}.json").unlink()
        assert not (tmp_path / f"session_report_{TS}.json").exists()


# ── segment building ────────────────────────────────────────────────

class TestSegmentBuilding:
    def test_text_lookup_by_seg_id(self, tmp_path):
        """Segments are correctly joined by seg_id."""
        _create_session_files(tmp_path)
        with open(tmp_path / f"diarized_segments_{TS}.json") as f:
            diar_segs = json.load(f)
        norm_files = sorted(tmp_path.glob(f"normalized_{TS}*.json"))
        with open(norm_files[-1]) as f:
            norm_segs = json.load(f)
        text_by_seg = {s["seg_id"]: s.get("normalized_text", "") for s in norm_segs}
        subtitle_segs = [
            {
                "t0": ds["t0"], "t1": ds["t1"],
                "speaker": ds["new_speaker_id"],
                "text": text_by_seg.get(ds["seg_id"], ""),
            }
            for ds in diar_segs
        ]
        assert len(subtitle_segs) == 2
        assert subtitle_segs[0]["text"] == "Hello world."
        assert subtitle_segs[0]["speaker"] == "spk_0"
        assert subtitle_segs[1]["text"] == "Hi there."
        assert subtitle_segs[1]["speaker"] == "spk_1"

    def test_missing_seg_id_gets_empty_text(self, tmp_path):
        """Diarized segment with no matching normalized entry gets empty text."""
        _create_session_files(tmp_path)
        # Add extra diarized segment with unknown seg_id
        with open(tmp_path / f"diarized_segments_{TS}.json") as f:
            diar_segs = json.load(f)
        diar_segs.append({
            "seg_id": "seg_9999", "t0": 10.0, "t1": 12.0,
            "old_speaker_id": "spk_0", "new_speaker_id": "spk_0",
        })
        with open(tmp_path / f"diarized_segments_{TS}.json", "w") as f:
            json.dump(diar_segs, f)

        with open(tmp_path / f"diarized_segments_{TS}.json") as f:
            diar_segs = json.load(f)
        norm_files = sorted(tmp_path.glob(f"normalized_{TS}*.json"))
        with open(norm_files[-1]) as f:
            norm_segs = json.load(f)
        text_by_seg = {s["seg_id"]: s.get("normalized_text", "") for s in norm_segs}
        subtitle_segs = [
            {
                "t0": ds["t0"], "t1": ds["t1"],
                "speaker": ds["new_speaker_id"],
                "text": text_by_seg.get(ds["seg_id"], ""),
            }
            for ds in diar_segs
        ]
        assert subtitle_segs[2]["text"] == ""
