"""Tests for session summary Markdown export."""

from __future__ import annotations

import copy

import pytest

from app.export_summary import build_summary_markdown, write_summary


# ── fixtures ─────────────────────────────────────────────────────────

FULL_REPORT = {
    "session_ts": "2026-01-01_12-00-00",
    "config": {
        "language": "da",
        "asr": {"model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "diarization": {
            "backend": "pyannote",
            "smoothing": True,
            "calibration_profile": "my_clinic",
            "calibration_enabled": True,
            "overlap_stabilizer_enabled": True,
            "prototype_matching_enabled": True,
            "min_duration_filter_enabled": True,
        },
        "reporting": {"session_report_enabled": True},
    },
    "feature_flags": {
        "calibration_enabled": True,
        "overlap_stabilizer_enabled": True,
        "prototype_matching_enabled": True,
        "min_duration_filter_enabled": True,
    },
    "outputs": {
        "raw": "outputs/raw_2026-01-01_12-00-00_large-v3.json",
        "normalized": "outputs/normalized_2026-01-01_12-00-00_large-v3.json",
        "audio": "outputs/audio_2026-01-01_12-00-00.wav",
        "diarization": "outputs/diarization_2026-01-01_12-00-00.json",
        "srt": None,
        "vtt": None,
    },
    "stats": {
        "segment_count": 42,
        "turns_before_smoothing": 60,
        "turns_after_smoothing": 50,
        "overlaps_marked": 3,
        "embeddings_computed": 20,
        "clusters_total": 3,
        "clusters_assigned": 2,
        "clusters_unknown": 1,
    },
    "audio_precheck": {
        "enabled": True,
        "duration_sec": 4.0,
        "peak_dbfs": -6.0,
        "rms_dbfs": -20.0,
        "snr_db_est": 25.0,
        "clipping_rate": 0.0,
        "warnings": [],
        "passed": True,
    },
}


MINIMAL_REPORT = {
    "session_ts": "2026-02-01_08-00-00",
    "config": {
        "language": "en",
        "asr": {"model": "small", "device": "cpu", "compute_type": "int8"},
        "diarization": {"backend": "default", "smoothing": False},
        "reporting": {"session_report_enabled": True},
    },
    "feature_flags": {},
    "outputs": {
        "raw": "outputs/raw_2026-02-01_08-00-00_small.json",
    },
    "stats": {"segment_count": 5},
}


# ── sections ─────────────────────────────────────────────────────────

class TestBuildSummary:
    def test_title_includes_timestamp(self):
        md = build_summary_markdown(FULL_REPORT)
        assert "# Session Summary — 2026-01-01_12-00-00" in md

    def test_configuration_section(self):
        md = build_summary_markdown(FULL_REPORT)
        assert "## Configuration" in md
        assert "**Language:** da" in md
        assert "**ASR model:** large-v3" in md
        assert "**Compute device:** cuda" in md
        assert "**Calibration profile:** my_clinic" in md

    def test_feature_flags_section(self):
        md = build_summary_markdown(FULL_REPORT)
        assert "## Feature Flags" in md
        assert "`calibration_enabled`: on" in md

    def test_audio_precheck_section(self):
        md = build_summary_markdown(FULL_REPORT)
        assert "## Audio Pre-check" in md
        assert "**Peak:** -6.0 dBFS" in md
        assert "**SNR estimate:** 25.0 dB" in md
        assert "**Passed:** yes" in md

    def test_precheck_with_warnings(self):
        report = copy.deepcopy(FULL_REPORT)
        report["audio_precheck"]["passed"] = False
        report["audio_precheck"]["warnings"] = ["Low SNR (10.0 dB < 15.0 dB)"]
        md = build_summary_markdown(report)
        assert "**Passed:** no" in md
        assert "Low SNR" in md

    def test_statistics_section(self):
        md = build_summary_markdown(FULL_REPORT)
        assert "## Statistics" in md
        assert "**Segments:** 42" in md
        assert "60" in md
        assert "50" in md
        assert "3 total" in md
        assert "2 assigned" in md
        assert "1 unknown" in md

    def test_output_files_section(self):
        md = build_summary_markdown(FULL_REPORT)
        assert "## Output Files" in md
        assert "**raw:**" in md
        assert "**audio:**" in md
        # None values should be excluded
        assert "**srt:**" not in md
        assert "**vtt:**" not in md

    def test_missing_precheck_omitted(self):
        md = build_summary_markdown(MINIMAL_REPORT)
        assert "Audio Pre-check" not in md

    def test_missing_diarization_stats(self):
        md = build_summary_markdown(MINIMAL_REPORT)
        assert "Diarization turns" not in md
        assert "Clusters" not in md

    def test_no_calibration_profile_omitted(self):
        md = build_summary_markdown(MINIMAL_REPORT)
        assert "Calibration profile" not in md

    def test_does_not_mutate_report(self):
        original = copy.deepcopy(FULL_REPORT)
        build_summary_markdown(FULL_REPORT)
        assert FULL_REPORT == original

    def test_empty_feature_flags(self):
        md = build_summary_markdown(MINIMAL_REPORT)
        # Empty flags dict — section still present but no items
        # (or section omitted; either is acceptable)
        assert "## Statistics" in md


# ── file writing ─────────────────────────────────────────────────────

class TestWriteSummary:
    def test_write_creates_file(self, tmp_path):
        path = write_summary(FULL_REPORT, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.name == "session_summary_2026-01-01_12-00-00.md"
        content = path.read_text("utf-8")
        assert "# Session Summary" in content

    def test_write_minimal_report(self, tmp_path):
        path = write_summary(MINIMAL_REPORT, str(tmp_path), "2026-02-01_08-00-00")
        assert path.exists()
        content = path.read_text("utf-8")
        assert "**Segments:** 5" in content
