"""Tests for confidence report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.confidence import (
    AVG_LOGPROB_THRESHOLD,
    COMPRESSION_RATIO_HIGH,
    NO_SPEECH_THRESHOLD,
    build_confidence_report,
    write_confidence_report,
)


# ── helpers ──────────────────────────────────────────────────────────

def _entry(
    seg_id: str = "seg_0001",
    t0: float = 0.0,
    t1: float = 1.0,
    avg_logprob: float | None = -0.5,
    no_speech_prob: float | None = 0.1,
    compression_ratio: float | None = 1.5,
) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
    }


# ── build_confidence_report ──────────────────────────────────────────

class TestBuildConfidenceReport:
    def test_clean_segments_no_flags(self):
        report = build_confidence_report([_entry()])
        assert report["flagged_count"] == 0
        assert report["segments"][0]["flags"] == []

    def test_no_speech_flagged(self):
        report = build_confidence_report([_entry(no_speech_prob=0.8)])
        assert "no_speech" in report["segments"][0]["flags"]
        assert report["flagged_count"] == 1

    def test_low_confidence_flagged(self):
        report = build_confidence_report([_entry(avg_logprob=-1.5)])
        assert "low_confidence" in report["segments"][0]["flags"]

    def test_repetitive_flagged(self):
        report = build_confidence_report([_entry(compression_ratio=3.0)])
        assert "repetitive" in report["segments"][0]["flags"]

    def test_multiple_flags(self):
        report = build_confidence_report([_entry(
            no_speech_prob=0.9,
            avg_logprob=-2.0,
            compression_ratio=5.0,
        )])
        flags = report["segments"][0]["flags"]
        assert "no_speech" in flags
        assert "low_confidence" in flags
        assert "repetitive" in flags
        assert report["flagged_count"] == 1

    def test_none_metric_not_flagged(self):
        report = build_confidence_report([_entry(
            avg_logprob=None, no_speech_prob=0.1, compression_ratio=1.0,
        )])
        assert "low_confidence" not in report["segments"][0]["flags"]
        assert report["flagged_count"] == 0

    def test_all_none_missing_metrics(self):
        report = build_confidence_report([_entry(
            avg_logprob=None, no_speech_prob=None, compression_ratio=None,
        )])
        assert report["segments"][0]["flags"] == ["missing_metrics"]
        assert report["flagged_count"] == 1

    def test_partial_none_no_missing(self):
        report = build_confidence_report([_entry(
            avg_logprob=-0.5, no_speech_prob=None, compression_ratio=None,
        )])
        flags = report["segments"][0]["flags"]
        assert "missing_metrics" not in flags
        assert report["flagged_count"] == 0

    def test_report_structure(self):
        report = build_confidence_report([_entry(), _entry(seg_id="seg_0002")])
        assert "thresholds" in report
        assert report["thresholds"]["no_speech_prob"] == NO_SPEECH_THRESHOLD
        assert report["thresholds"]["avg_logprob"] == AVG_LOGPROB_THRESHOLD
        assert report["thresholds"]["compression_ratio_high"] == COMPRESSION_RATIO_HIGH
        assert len(report["segments"]) == 2
        assert report["total_count"] == 2

    def test_empty_input(self):
        report = build_confidence_report([])
        assert report["flagged_count"] == 0
        assert report["total_count"] == 0
        assert report["segments"] == []


# ── write_confidence_report ──────────────────────────────────────────

class TestWriteConfidenceReport:
    def test_writes_json_file(self, tmp_path):
        report = build_confidence_report([_entry()])
        path = write_confidence_report(report, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.name == "confidence_report_2026-01-01_12-00-00.json"

    def test_roundtrip(self, tmp_path):
        report = build_confidence_report([
            _entry(no_speech_prob=0.9),
            _entry(seg_id="seg_0002"),
        ])
        path = write_confidence_report(report, str(tmp_path), "2026-01-01_12-00-00")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["flagged_count"] == report["flagged_count"]
        assert loaded["total_count"] == report["total_count"]
        assert len(loaded["segments"]) == len(report["segments"])
