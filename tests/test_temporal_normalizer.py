"""Tests for deterministic temporal normalizer."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.temporal_normalizer import normalize_time_expression, normalize_timeline
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

# Fixed reference: Sunday 2026-03-08
REF = datetime(2026, 3, 8)


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _tl_entry(symptom: str, time_expression: str | None = None,
              seg_id: str | None = None, speaker_id: str | None = None,
              t_start: float | None = None) -> dict:
    return {
        "symptom": symptom,
        "time_expression": time_expression,
        "seg_id": seg_id,
        "speaker_id": speaker_id,
        "t_start": t_start,
    }


# ══════════════════════════════════════════════════════════════════
# Relative days
# ══════════════════════════════════════════════════════════════════


class TestRelativeDays:
    def test_today(self):
        assert normalize_time_expression("today", REF) == "2026-03-08"

    def test_yesterday(self):
        assert normalize_time_expression("yesterday", REF) == "2026-03-07"

    def test_day_before_yesterday(self):
        assert normalize_time_expression("day before yesterday", REF) == "2026-03-06"

    def test_tomorrow(self):
        assert normalize_time_expression("tomorrow", REF) == "2026-03-09"

    def test_case_insensitive(self):
        assert normalize_time_expression("Yesterday", REF) == "2026-03-07"
        assert normalize_time_expression("TODAY", REF) == "2026-03-08"

    def test_whitespace_stripped(self):
        assert normalize_time_expression("  yesterday  ", REF) == "2026-03-07"


# ══════════════════════════════════════════════════════════════════
# Weekday normalization
# ══════════════════════════════════════════════════════════════════


class TestWeekdays:
    def test_bare_monday(self):
        # REF is Sunday 2026-03-08; most recent Monday = 2026-03-02
        assert normalize_time_expression("monday", REF) == "2026-03-02"

    def test_bare_friday(self):
        # Most recent Friday before Sunday 2026-03-08 = 2026-03-06
        assert normalize_time_expression("friday", REF) == "2026-03-06"

    def test_bare_saturday(self):
        # Most recent Saturday before Sunday 2026-03-08 = 2026-03-07
        assert normalize_time_expression("saturday", REF) == "2026-03-07"

    def test_bare_sunday_goes_back_one_week(self):
        # REF is Sunday; bare "sunday" should mean last Sunday = 2026-03-01
        assert normalize_time_expression("sunday", REF) == "2026-03-01"

    def test_case_insensitive(self):
        assert normalize_time_expression("Monday", REF) == "2026-03-02"
        assert normalize_time_expression("FRIDAY", REF) == "2026-03-06"

    def test_weekday_from_different_ref(self):
        # Wednesday 2026-03-04 as reference; most recent Monday = 2026-03-02
        wed_ref = datetime(2026, 3, 4)
        assert normalize_time_expression("monday", wed_ref) == "2026-03-02"


# ══════════════════════════════════════════════════════════════════
# Since expressions
# ══════════════════════════════════════════════════════════════════


class TestSinceExpressions:
    def test_since_monday(self):
        assert normalize_time_expression("since Monday", REF) == "2026-03-02"

    def test_since_friday(self):
        assert normalize_time_expression("since Friday", REF) == "2026-03-06"

    def test_since_sunday(self):
        # "since sunday" on a Sunday → last Sunday
        assert normalize_time_expression("since Sunday", REF) == "2026-03-01"

    def test_case_insensitive(self):
        assert normalize_time_expression("since monday", REF) == "2026-03-02"
        assert normalize_time_expression("SINCE MONDAY", REF) == "2026-03-02"


# ══════════════════════════════════════════════════════════════════
# Last week
# ══════════════════════════════════════════════════════════════════


class TestLastWeek:
    def test_last_week(self):
        assert normalize_time_expression("last week", REF) == "2026-03-01"

    def test_case_insensitive(self):
        assert normalize_time_expression("Last Week", REF) == "2026-03-01"


# ══════════════════════════════════════════════════════════════════
# Duration normalization
# ══════════════════════════════════════════════════════════════════


class TestDurations:
    def test_days(self):
        assert normalize_time_expression("3 days", REF) == "P3D"

    def test_for_days(self):
        assert normalize_time_expression("for 3 days", REF) == "P3D"

    def test_weeks(self):
        assert normalize_time_expression("2 weeks", REF) == "P2W"

    def test_hours(self):
        assert normalize_time_expression("5 hours", REF) == "PT5H"

    def test_minutes(self):
        assert normalize_time_expression("10 minutes", REF) == "PT10M"

    def test_seconds(self):
        assert normalize_time_expression("30 seconds", REF) == "PT30S"

    def test_singular_units(self):
        assert normalize_time_expression("1 day", REF) == "P1D"
        assert normalize_time_expression("1 week", REF) == "P1W"
        assert normalize_time_expression("1 hour", REF) == "PT1H"
        assert normalize_time_expression("1 minute", REF) == "PT1M"

    def test_months(self):
        assert normalize_time_expression("3 months", REF) == "P3M"

    def test_years(self):
        assert normalize_time_expression("2 years", REF) == "P2Y"

    def test_for_prefix_with_weeks(self):
        assert normalize_time_expression("for 2 weeks", REF) == "P2W"

    def test_case_insensitive(self):
        assert normalize_time_expression("For 3 Days", REF) == "P3D"


# ══════════════════════════════════════════════════════════════════
# Unsupported / edge cases
# ══════════════════════════════════════════════════════════════════


class TestUnsupported:
    def test_empty_string(self):
        assert normalize_time_expression("", REF) is None

    def test_whitespace_only(self):
        assert normalize_time_expression("   ", REF) is None

    def test_unrecognized_phrase(self):
        assert normalize_time_expression("a long time ago", REF) is None

    def test_partial_match_no_number(self):
        assert normalize_time_expression("some days", REF) is None

    def test_random_text(self):
        assert normalize_time_expression("the patient said", REF) is None

    def test_ambiguous_phrase(self):
        assert normalize_time_expression("a while back", REF) is None

    def test_vague_relative(self):
        assert normalize_time_expression("recently", REF) is None

    def test_mention_order_not_used(self):
        """Mention-only order must NOT produce a normalized time."""
        assert normalize_time_expression("first symptom", REF) is None
        assert normalize_time_expression("then", REF) is None


# ══════════════════════════════════════════════════════════════════
# normalize_timeline
# ══════════════════════════════════════════════════════════════════


class TestNormalizeTimeline:
    def test_adds_normalized_time(self):
        tl = [_tl_entry("headache", "yesterday")]
        result = normalize_timeline(tl, REF)
        assert len(result) == 1
        assert result[0]["normalized_time"] == "2026-03-07"
        assert result[0]["time_expression"] == "yesterday"
        assert result[0]["symptom"] == "headache"

    def test_preserves_original_entries(self):
        tl = [_tl_entry("headache", "yesterday")]
        result = normalize_timeline(tl, REF)
        # Original entry must NOT have normalized_time
        assert "normalized_time" not in tl[0]

    def test_none_time_expression(self):
        tl = [_tl_entry("headache", None)]
        result = normalize_timeline(tl, REF)
        assert result[0]["normalized_time"] is None

    def test_unsupported_expression(self):
        tl = [_tl_entry("headache", "a while back")]
        result = normalize_timeline(tl, REF)
        assert result[0]["normalized_time"] is None

    def test_multiple_entries(self):
        tl = [
            _tl_entry("headache", "yesterday"),
            _tl_entry("fever", "3 days"),
            _tl_entry("cough", None),
        ]
        result = normalize_timeline(tl, REF)
        assert result[0]["normalized_time"] == "2026-03-07"
        assert result[1]["normalized_time"] == "P3D"
        assert result[2]["normalized_time"] is None

    def test_empty_timeline(self):
        assert normalize_timeline([], REF) == []

    def test_all_original_fields_preserved(self):
        tl = [_tl_entry("headache", "today", "seg_0001", "spk_0", 1.5)]
        result = normalize_timeline(tl, REF)
        assert result[0]["symptom"] == "headache"
        assert result[0]["time_expression"] == "today"
        assert result[0]["seg_id"] == "seg_0001"
        assert result[0]["speaker_id"] == "spk_0"
        assert result[0]["t_start"] == 1.5
        assert result[0]["normalized_time"] == "2026-03-08"

    def test_does_not_infer_order_from_position(self):
        """Two entries without time expressions — both get None, no ordering."""
        tl = [
            _tl_entry("headache", None, t_start=0.0),
            _tl_entry("fever", None, t_start=5.0),
        ]
        result = normalize_timeline(tl, REF)
        assert result[0]["normalized_time"] is None
        assert result[1]["normalized_time"] is None


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        tl = [
            _tl_entry("headache", "yesterday"),
            _tl_entry("fever", "3 days"),
        ]
        r1 = normalize_timeline(tl, REF)
        r2 = normalize_timeline(tl, REF)
        assert r1 == r2

    def test_same_expression_same_result(self):
        r1 = normalize_time_expression("since Monday", REF)
        r2 = normalize_time_expression("since Monday", REF)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_normalized_timeline_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "normalized_timeline" in state["derived"]
        assert isinstance(state["derived"]["normalized_timeline"], list)

    def test_normalized_timeline_entries_have_field(self):
        state = build_clinical_state([
            _seg("patient has headache for 3 days."),
        ])
        ntl = state["derived"]["normalized_timeline"]
        for entry in ntl:
            assert "normalized_time" in entry

    def test_original_timeline_unchanged(self):
        state = build_clinical_state([
            _seg("patient has headache for 3 days."),
        ])
        for entry in state["timeline"]:
            assert "normalized_time" not in entry

    def test_structured_data_unchanged(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert isinstance(state["derived"]["problem_representation"], dict)
        assert isinstance(state["derived"]["symptom_representations"], list)
        assert isinstance(state["derived"]["problem_summary"], str)
        assert isinstance(state["derived"]["ontology_concepts"], list)
        assert isinstance(state["derived"]["clinical_patterns"], list)
        assert isinstance(state["derived"]["running_summary"], dict)

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_ntl = list(state["derived"]["normalized_timeline"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["normalized_timeline"] == original_ntl
        assert "ai_overlay" in state["derived"]

    def test_empty_input(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["normalized_timeline"] == []
