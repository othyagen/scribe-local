"""Tests for clinical interaction — next-step question derivation."""

from __future__ import annotations

import pytest

from app.clinical_interaction import derive_next_questions, _MAX_QUESTIONS
from app.clinical_state import build_clinical_state


# ── helpers ──────────────────────────────────────────────────────────


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _state_with_insights(suggested=None, missing=None):
    """Build a minimal state dict with clinical_insights populated."""
    return {
        "clinical_insights": {
            "missing_information": missing or [],
            "uncertainties": [],
            "suggested_questions": suggested or [],
            "data_quality_issues": [],
        },
    }


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_returns_list(self):
        state = _state_with_insights()
        result = derive_next_questions(state)
        assert isinstance(result, list)

    def test_empty_when_no_insights(self):
        result = derive_next_questions({})
        assert result == []

    def test_empty_when_no_suggested_questions(self):
        state = _state_with_insights(suggested=[])
        result = derive_next_questions(state)
        assert result == []

    def test_question_has_required_fields(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert len(result) == 1
        q = result[0]
        assert "question" in q
        assert "reason" in q
        assert "priority" in q

    def test_no_internal_sort_key_in_output(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        for q in result:
            assert "_sort_key" not in q


# ── priority ordering ────────────────────────────────────────────────


class TestPriority:
    def test_allergy_before_duration(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long have you had headache?", "related": "headache"},
            {"category": "allergy", "question": "Do you have any known drug allergies?", "related": None},
        ])
        result = derive_next_questions(state)
        assert result[0]["question"] == "Do you have any known drug allergies?"

    def test_duration_before_severity(self):
        state = _state_with_insights(suggested=[
            {"category": "severity", "question": "How severe is headache?", "related": "headache"},
            {"category": "duration", "question": "How long have you had headache?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert result[0]["question"] == "How long have you had headache?"

    def test_allergy_is_high_priority(self):
        state = _state_with_insights(suggested=[
            {"category": "allergy", "question": "Any allergies?", "related": None},
        ])
        result = derive_next_questions(state)
        assert result[0]["priority"] == "high"

    def test_duration_is_medium_priority(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert result[0]["priority"] == "medium"

    def test_severity_is_low_priority(self):
        state = _state_with_insights(suggested=[
            {"category": "severity", "question": "How severe?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert result[0]["priority"] == "low"

    def test_boost_from_important_gap(self):
        """Duration question for a symptom with missing_duration gap gets boosted."""
        state = _state_with_insights(
            suggested=[
                {"category": "duration", "question": "How long have you had headache?", "related": "headache"},
            ],
            missing=[
                {"category": "missing_duration", "detail": "No duration for headache", "related": "headache"},
            ],
        )
        result = derive_next_questions(state)
        assert result[0]["priority"] == "high"


# ── deduplication ────────────────────────────────────────────────────


class TestDeduplication:
    def test_duplicate_questions_removed(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long have you had headache?", "related": "headache"},
            {"category": "duration", "question": "How long have you had headache?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long?", "related": "headache"},
            {"category": "duration", "question": "how long?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert len(result) == 1


# ── cap ──────────────────────────────────────────────────────────────


class TestCap:
    def test_capped_at_max(self):
        suggested = [
            {"category": "severity", "question": f"Question {i}?", "related": f"sym_{i}"}
            for i in range(_MAX_QUESTIONS + 3)
        ]
        state = _state_with_insights(suggested=suggested)
        result = derive_next_questions(state)
        assert len(result) == _MAX_QUESTIONS


# ── reason text ──────────────────────────────────────────────────────


class TestReason:
    def test_allergy_reason(self):
        state = _state_with_insights(suggested=[
            {"category": "allergy", "question": "Any allergies?", "related": None},
        ])
        result = derive_next_questions(state)
        assert "allergy" in result[0]["reason"].lower()

    def test_duration_reason_includes_symptom(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long?", "related": "headache"},
        ])
        result = derive_next_questions(state)
        assert "headache" in result[0]["reason"]

    def test_severity_reason_includes_symptom(self):
        state = _state_with_insights(suggested=[
            {"category": "severity", "question": "How severe?", "related": "nausea"},
        ])
        result = derive_next_questions(state)
        assert "nausea" in result[0]["reason"]

    def test_unknown_category_reason(self):
        state = _state_with_insights(suggested=[
            {"category": "other", "question": "Anything else?", "related": None},
        ])
        result = derive_next_questions(state)
        assert result[0]["reason"]  # non-empty


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_state(self):
        state = _state_with_insights(suggested=[
            {"category": "duration", "question": "How long?", "related": "headache"},
        ])
        original_count = len(state["clinical_insights"]["suggested_questions"])
        derive_next_questions(state)
        assert len(state["clinical_insights"]["suggested_questions"]) == original_count

    def test_deterministic(self):
        state = _state_with_insights(suggested=[
            {"category": "allergy", "question": "Allergies?", "related": None},
            {"category": "duration", "question": "How long?", "related": "headache"},
        ])
        r1 = derive_next_questions(state)
        r2 = derive_next_questions(state)
        assert r1 == r2


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_next_questions_in_clinical_state(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        assert "next_questions" in state
        assert isinstance(state["next_questions"], list)

    def test_questions_generated_for_symptom_without_duration(self):
        state = build_clinical_state([
            _seg("patient has nausea.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        questions = state["next_questions"]
        q_texts = [q["question"].lower() for q in questions]
        assert any("nausea" in t for t in q_texts)

    def test_allergy_question_with_medication(self):
        state = build_clinical_state([
            _seg("prescribed ibuprofen 400 mg.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        questions = state["next_questions"]
        q_texts = [q["question"].lower() for q in questions]
        assert any("allerg" in t for t in q_texts)

    def test_empty_state_no_questions(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["next_questions"] == []

    def test_full_scenario(self):
        segments = [
            _seg("patient reports headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("prescribed ibuprofen.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ]
        state = build_clinical_state(segments)
        questions = state["next_questions"]
        assert isinstance(questions, list)
        for q in questions:
            assert "question" in q
            assert "reason" in q
            assert "priority" in q
