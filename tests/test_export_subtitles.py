"""Tests for SRT/VTT subtitle export."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.export_subtitles import (
    build_srt_cues,
    build_vtt_cues,
    format_srt_time,
    format_vtt_time,
    write_srt,
    write_vtt,
)


# ── time formatting ──────────────────────────────────────────────────

class TestFormatTime:
    def test_srt_zero(self):
        assert format_srt_time(0.0) == "00:00:00,000"

    def test_srt_fractional(self):
        assert format_srt_time(3661.5) == "01:01:01,500"

    def test_srt_negative_clamped(self):
        assert format_srt_time(-1.0) == "00:00:00,000"

    def test_vtt_zero(self):
        assert format_vtt_time(0.0) == "00:00:00.000"

    def test_vtt_fractional(self):
        assert format_vtt_time(3661.5) == "01:01:01.500"

    def test_vtt_negative_clamped(self):
        assert format_vtt_time(-1.0) == "00:00:00.000"

    def test_srt_milliseconds(self):
        assert format_srt_time(1.234) == "00:00:01,234"

    def test_vtt_milliseconds(self):
        assert format_vtt_time(1.234) == "00:00:01.234"


# ── cue building ─────────────────────────────────────────────────────

SEGMENTS = [
    {"t0": 0.0, "t1": 2.5, "speaker": "spk_0", "text": "Hello world."},
    {"t0": 3.0, "t1": 5.0, "speaker": "spk_1", "text": "Hi there."},
]


class TestBuildSrt:
    def test_basic_cues(self):
        result = build_srt_cues(SEGMENTS)
        lines = result.split("\n")
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,500" in lines[1]
        assert "[spk_0] Hello world." in lines[2]
        assert lines[4] == "2"
        assert "[spk_1] Hi there." in lines[6]

    def test_empty_segments(self):
        assert build_srt_cues([]) == ""

    def test_skips_empty_text(self):
        segs = [
            {"t0": 0.0, "t1": 1.0, "speaker": "spk_0", "text": "Good."},
            {"t0": 1.0, "t1": 2.0, "speaker": "spk_0", "text": ""},
            {"t0": 2.0, "t1": 3.0, "speaker": "spk_0", "text": "   "},
        ]
        result = build_srt_cues(segs)
        assert result.count("-->") == 1

    def test_fix_end_before_start(self):
        segs = [{"t0": 5.0, "t1": 5.0, "speaker": "spk_0", "text": "Fixed."}]
        result = build_srt_cues(segs)
        assert "00:00:05,000 --> 00:00:05,010" in result

    def test_negative_times_clamped(self):
        segs = [{"t0": -1.0, "t1": 1.0, "speaker": "spk_0", "text": "Clamped."}]
        result = build_srt_cues(segs)
        assert "00:00:00,000 --> 00:00:01,000" in result

    def test_sequential_numbering(self):
        segs = [
            {"t0": 0.0, "t1": 1.0, "speaker": "spk_0", "text": "One."},
            {"t0": 1.0, "t1": 2.0, "speaker": "spk_0", "text": "Two."},
            {"t0": 2.0, "t1": 3.0, "speaker": "spk_0", "text": "Three."},
        ]
        result = build_srt_cues(segs)
        lines = result.split("\n")
        assert lines[0] == "1"
        assert lines[4] == "2"
        assert lines[8] == "3"


class TestBuildVtt:
    def test_basic_cues(self):
        result = build_vtt_cues(SEGMENTS)
        lines = result.split("\n")
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""
        assert "00:00:00.000 --> 00:00:02.500" in lines[2]
        assert "[spk_0] Hello world." in lines[3]

    def test_empty_segments(self):
        result = build_vtt_cues([])
        assert result.startswith("WEBVTT")
        assert result.count("-->") == 0

    def test_skips_whitespace_text(self):
        segs = [
            {"t0": 0.0, "t1": 1.0, "speaker": "spk_0", "text": "  \n  "},
        ]
        result = build_vtt_cues(segs)
        assert result.count("-->") == 0

    def test_fix_end_equals_start(self):
        segs = [{"t0": 2.0, "t1": 2.0, "speaker": "spk_0", "text": "Edge."}]
        result = build_vtt_cues(segs)
        assert "00:00:02.000 --> 00:00:02.010" in result


# ── file writing ─────────────────────────────────────────────────────

class TestWriteFiles:
    def test_write_srt(self, tmp_path):
        path = write_srt(SEGMENTS, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.suffix == ".srt"
        assert path.name == "subtitles_2026-01-01_12-00-00.srt"
        content = path.read_text("utf-8")
        assert "[spk_0]" in content
        assert "[spk_1]" in content

    def test_write_vtt(self, tmp_path):
        path = write_vtt(SEGMENTS, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.suffix == ".vtt"
        assert path.name == "subtitles_2026-01-01_12-00-00.vtt"
        content = path.read_text("utf-8")
        assert content.startswith("WEBVTT")

    def test_missing_text_key(self):
        segs = [{"t0": 0.0, "t1": 1.0, "speaker": "spk_0"}]
        result = build_srt_cues(segs)
        assert result == ""

    def test_missing_speaker_key(self):
        segs = [{"t0": 0.0, "t1": 1.0, "text": "No speaker."}]
        result = build_srt_cues(segs)
        assert "[] No speaker." in result
