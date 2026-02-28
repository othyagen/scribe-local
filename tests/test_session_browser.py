"""Tests for session browser — read-only session listing and inspection."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from app.config import build_arg_parser
from app.session_browser import scan_sessions, show_session


# ── helpers ──────────────────────────────────────────────────────────

def _seg_dict(
    seg_id: str = "seg_0001",
    t0: float = 0.0,
    t1: float = 1.0,
    model_name: str = "large-v3",
    language: str = "en",
) -> dict:
    return {
        "seg_id": seg_id, "t0": t0, "t1": t1,
        "speaker_id": "spk_0", "raw_text": "hello",
        "model_name": model_name, "language": language,
        "paragraph_id": "para_0000",
    }


def _write_raw(tmp_path: Path, ts: str, segments: list[dict],
               model_tag: str = "large-v3") -> Path:
    p = tmp_path / f"raw_{ts}_{model_tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    return p


def _write_wav(path: Path, n_samples: int = 16000) -> Path:
    pcm = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    return path


def _write_confidence(tmp_path: Path, ts: str, flagged: int = 2,
                      total: int = 10) -> Path:
    p = tmp_path / f"confidence_report_{ts}.json"
    p.write_text(json.dumps({
        "thresholds": {"no_speech_prob": 0.6, "avg_logprob": -1.0,
                       "compression_ratio_high": 2.4},
        "segments": [],
        "flagged_count": flagged,
        "total_count": total,
    }), encoding="utf-8")
    return p


def _write_diarization(tmp_path: Path, ts: str) -> Path:
    p = tmp_path / f"diarization_{ts}.json"
    p.write_text(json.dumps({
        "turns": [
            {"start": 0.0, "end": 3.0, "speaker": "spk_0"},
            {"start": 3.0, "end": 6.0, "speaker": "spk_1"},
        ]
    }), encoding="utf-8")
    return p


# ── scan_sessions ────────────────────────────────────────────────────

class TestScanSessions:
    def test_finds_and_sorts_newest_first(self, tmp_path):
        _write_raw(tmp_path, "2026-01-01_10-00-00", [_seg_dict()])
        _write_raw(tmp_path, "2026-01-02_10-00-00", [_seg_dict()])
        _write_raw(tmp_path, "2026-01-01_15-00-00", [_seg_dict()])

        sessions = scan_sessions(tmp_path)
        timestamps = [s.ts for s in sessions]
        assert timestamps == [
            "2026-01-02_10-00-00",
            "2026-01-01_15-00-00",
            "2026-01-01_10-00-00",
        ]

    def test_extracts_duration_and_count(self, tmp_path):
        segs = [
            _seg_dict("seg_0001", 0.0, 1.5),
            _seg_dict("seg_0002", 2.0, 3.5),
            _seg_dict("seg_0003", 5.0, 7.2),
        ]
        _write_raw(tmp_path, "2026-01-01_12-00-00", segs)

        sessions = scan_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].segment_count == 3
        assert sessions[0].duration_sec == 7.2
        assert sessions[0].model_tag == "large-v3"
        assert sessions[0].language == "en"

    def test_detects_companion_files(self, tmp_path):
        ts = "2026-01-01_12-00-00"
        _write_raw(tmp_path, ts, [_seg_dict()])
        _write_wav(tmp_path / f"audio_{ts}.wav")
        _write_diarization(tmp_path, ts)
        (tmp_path / f"speaker_tags_{ts}.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"normalized_{ts}_large-v3.json").write_text("[]", encoding="utf-8")

        sessions = scan_sessions(tmp_path)
        s = sessions[0]
        assert s.has_audio is True
        assert s.audio_parts_count == 1
        assert s.has_diarization is True
        assert s.has_tags is True
        assert s.has_normalized is True
        assert s.resume_possible is True

    def test_detects_audio_parts(self, tmp_path):
        ts = "2026-01-01_12-00-00"
        _write_raw(tmp_path, ts, [_seg_dict()])
        _write_wav(tmp_path / f"audio_{ts}.wav")
        _write_wav(tmp_path / f"audio_{ts}_part2.wav")

        sessions = scan_sessions(tmp_path)
        assert sessions[0].audio_parts_count == 2
        assert sessions[0].has_audio is True

    def test_reads_confidence_flagged_count(self, tmp_path):
        ts = "2026-01-01_12-00-00"
        _write_raw(tmp_path, ts, [_seg_dict()])
        _write_confidence(tmp_path, ts, flagged=3, total=10)

        sessions = scan_sessions(tmp_path)
        assert sessions[0].has_confidence is True
        assert sessions[0].confidence_flagged_count == 3

    def test_skips_corrupt_raw_without_crashing(self, tmp_path, capsys):
        ts_good = "2026-01-02_12-00-00"
        ts_bad = "2026-01-01_12-00-00"

        _write_raw(tmp_path, ts_good, [_seg_dict()])
        # Write corrupt JSONL
        p = tmp_path / f"raw_{ts_bad}_large-v3.json"
        p.write_text('{"seg_id":"seg_0001","t0":0,"t1":1}\n{bad\n', encoding="utf-8")

        sessions = scan_sessions(tmp_path)
        # Only the good session should be returned
        assert len(sessions) == 1
        assert sessions[0].ts == ts_good

        # Warning should be printed to stderr
        captured = capsys.readouterr()
        assert "corrupt JSONL" in captured.err
        assert ts_bad in captured.err

    def test_no_audio_means_no_resume(self, tmp_path):
        _write_raw(tmp_path, "2026-01-01_12-00-00", [_seg_dict()])

        sessions = scan_sessions(tmp_path)
        assert sessions[0].has_audio is False
        assert sessions[0].resume_possible is False

    def test_empty_directory(self, tmp_path):
        sessions = scan_sessions(tmp_path)
        assert sessions == []

    def test_empty_raw_file_skipped(self, tmp_path):
        (tmp_path / "raw_2026-01-01_12-00-00_large-v3.json").write_text(
            "", encoding="utf-8"
        )
        sessions = scan_sessions(tmp_path)
        assert sessions == []


# ── show_session ─────────────────────────────────────────────────────

class TestShowSession:
    def test_returns_expected_fields(self, tmp_path):
        ts = "2026-01-01_12-00-00"
        _write_raw(tmp_path, ts, [
            _seg_dict("seg_0001", 0.0, 2.0),
            _seg_dict("seg_0002", 3.0, 5.0),
        ])
        _write_wav(tmp_path / f"audio_{ts}.wav")
        _write_diarization(tmp_path, ts)
        _write_confidence(tmp_path, ts, flagged=1, total=2)

        info = show_session(tmp_path, ts)
        assert info["ts"] == ts
        assert info["segment_count"] == 2
        assert info["duration_sec"] == 5.0
        assert info["has_audio"] is True
        assert info["has_diarization"] is True
        assert info["speaker_count"] == 2
        assert info["has_confidence"] is True
        assert info["confidence_flagged_count"] == 1
        assert info["confidence_total"] == 2
        assert info["resume_possible"] is True
        assert "raw" in info["files"]
        assert "diarization" in info["files"]

    def test_missing_session_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no session found"):
            show_session(tmp_path, "2026-99-99_00-00-00")

    def test_corrupt_session_raises(self, tmp_path):
        ts = "2026-01-01_12-00-00"
        p = tmp_path / f"raw_{ts}_large-v3.json"
        p.write_text("{bad json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="corrupt"):
            show_session(tmp_path, ts)

    def test_resume_reason_when_no_audio(self, tmp_path):
        ts = "2026-01-01_12-00-00"
        _write_raw(tmp_path, ts, [_seg_dict()])

        info = show_session(tmp_path, ts)
        assert info["resume_possible"] is False
        assert "no audio" in info["resume_reason"]


# ── CLI parsing ──────────────────────────────────────────────────────

class TestBrowserCli:
    def test_list_sessions_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--list-sessions"])
        assert args.list_sessions is True

    def test_show_session_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--show-session", "2026-01-01_12-00-00"])
        assert args.show_session == "2026-01-01_12-00-00"

    def test_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.list_sessions is False
        assert args.show_session is None
