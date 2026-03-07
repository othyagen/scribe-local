"""Tests for symptom timeline extraction and clinical note rendering."""

from __future__ import annotations

import pytest

from app.symptom_timeline import extract_symptom_timeline
from app.export_clinical_note import build_clinical_note


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── extraction tests ─────────────────────────────────────────────────


class TestSymptomTimeline:
    def test_symptom_with_numeric_duration(self):
        segs = [_seg("I have had a headache for 3 days.")]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "headache"
        assert result[0]["time_expression"] == "3 days"

    def test_symptom_with_relative_time(self):
        segs = [_seg("dizziness since yesterday.")]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "dizziness"
        assert result[0]["time_expression"] == "since yesterday"

    def test_symptom_without_duration(self):
        segs = [_seg("patient reports nausea.")]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "nausea"
        assert result[0]["time_expression"] is None

    def test_multiple_symptoms_different_segments(self):
        segs = [
            _seg("headache for 2 days.", seg_id="seg_0001", t0=0.0, t1=1.0),
            _seg("also has nausea.", seg_id="seg_0002", t0=1.0, t1=2.0),
        ]
        result = extract_symptom_timeline(segs)
        assert len(result) == 2
        assert result[0]["symptom"] == "headache"
        assert result[0]["time_expression"] == "2 days"
        assert result[1]["symptom"] == "nausea"
        assert result[1]["time_expression"] is None

    def test_empty_input(self):
        assert extract_symptom_timeline([]) == []

    def test_deduplication(self):
        segs = [
            _seg("headache for 3 days.", seg_id="seg_0001", t0=0.0),
            _seg("headache again today.", seg_id="seg_0002", t0=5.0),
        ]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "headache"
        assert result[0]["time_expression"] == "3 days"

    def test_started_today(self):
        segs = [_seg("headache started today.")]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["time_expression"] == "started today"

    def test_since_last_night(self):
        segs = [_seg("pain since last night.")]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "pain"
        assert result[0]["time_expression"] == "since last night"

    def test_written_number_duration(self):
        segs = [_seg("cough for two weeks.")]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "cough"
        assert result[0]["time_expression"] == "for two weeks"

    def test_missing_segment_fields(self):
        segs = [{"normalized_text": "headache for 3 days."}]
        result = extract_symptom_timeline(segs)
        assert len(result) == 1
        assert result[0]["symptom"] == "headache"
        assert result[0]["seg_id"] is None
        assert result[0]["speaker_id"] is None
        assert result[0]["t_start"] is None


# ── rendering tests ──────────────────────────────────────────────────

_TIMELINE_TEMPLATE = {
    "name": "Test Note",
    "format": "markdown",
    "sections": [],
    "show_symptom_timeline": True,
}


class TestTimelineRendering:
    def test_timeline_rendered_in_note(self):
        timeline = [
            {"symptom": "headache", "time_expression": "3 days"},
            {"symptom": "nausea", "time_expression": None},
        ]
        result = build_clinical_note([], _TIMELINE_TEMPLATE, symptom_timeline=timeline)
        assert "## Symptom Timeline" in result
        assert "- headache \u2014 3 days" in result
        assert "- nausea" in result

    def test_timeline_not_rendered_without_flag(self):
        template = {**_TIMELINE_TEMPLATE, "show_symptom_timeline": False}
        timeline = [{"symptom": "headache", "time_expression": "3 days"}]
        result = build_clinical_note([], template, symptom_timeline=timeline)
        assert "## Symptom Timeline" not in result

    def test_timeline_empty_no_section(self):
        result = build_clinical_note([], _TIMELINE_TEMPLATE, symptom_timeline=[])
        assert "## Symptom Timeline" not in result

    def test_timeline_format_no_time(self):
        timeline = [{"symptom": "dizziness", "time_expression": None}]
        result = build_clinical_note([], _TIMELINE_TEMPLATE, symptom_timeline=timeline)
        assert "- dizziness" in result
        assert "\u2014" not in result
