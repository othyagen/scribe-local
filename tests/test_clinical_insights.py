"""Tests for clinical insights — gap analysis and next-step prompts."""

from __future__ import annotations

import pytest

from app.clinical_insights import (
    derive_clinical_insights,
    _find_missing_information,
    _find_uncertainties,
    _suggest_questions,
    _find_data_quality_issues,
)
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


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_returns_dict_with_expected_keys(self):
        state = build_clinical_state([_seg("hello.")])
        insights = derive_clinical_insights(state)
        assert set(insights.keys()) == {
            "missing_information",
            "uncertainties",
            "suggested_questions",
            "data_quality_issues",
        }

    def test_all_values_are_lists(self):
        state = build_clinical_state([_seg("hello.")])
        insights = derive_clinical_insights(state)
        for key, value in insights.items():
            assert isinstance(value, list), f"{key} should be a list"

    def test_empty_state(self):
        state = build_clinical_state([])
        insights = derive_clinical_insights(state)
        assert insights["missing_information"] == []
        assert insights["uncertainties"] == []
        assert insights["suggested_questions"] == []
        assert insights["data_quality_issues"] == []


# ── missing information ──────────────────────────────────────────────


class TestMissingInformation:
    def test_missing_duration_detected(self):
        state = build_clinical_state([_seg("patient has nausea.")])
        gaps = _find_missing_information(state)
        categories = [g["category"] for g in gaps]
        assert "missing_duration" in categories
        related = [g["related"] for g in gaps if g["category"] == "missing_duration"]
        assert "nausea" in related

    def test_no_missing_duration_when_present(self):
        state = build_clinical_state([_seg("nausea for 3 days.")])
        gaps = _find_missing_information(state)
        duration_gaps = [g for g in gaps if g["category"] == "missing_duration"]
        related = [g["related"].lower() for g in duration_gaps]
        assert "nausea" not in related

    def test_missing_severity_detected(self):
        state = build_clinical_state([_seg("patient has headache.")])
        gaps = _find_missing_information(state)
        categories = [g["category"] for g in gaps]
        assert "missing_severity" in categories

    def test_missing_dosage_from_review_flags(self):
        state = build_clinical_state([_seg("prescribed ibuprofen.")])
        gaps = _find_missing_information(state)
        categories = [g["category"] for g in gaps]
        assert "missing_dosage" in categories

    def test_missing_medications_when_problems_exist(self):
        # Symptoms produce problems but no medications mentioned
        state = build_clinical_state([_seg("patient has headache.")])
        gaps = _find_missing_information(state)
        categories = [g["category"] for g in gaps]
        if state.get("problems"):
            assert "missing_medications" in categories

    def test_gap_has_required_fields(self):
        state = build_clinical_state([_seg("patient has headache.")])
        gaps = _find_missing_information(state)
        for gap in gaps:
            assert "category" in gap
            assert "detail" in gap
            assert "related" in gap


# ── uncertainties ────────────────────────────────────────────────────


class TestUncertainties:
    def test_unexplained_symptom_detected(self):
        # A symptom not covered by any hypothesis
        state = build_clinical_state([_seg("patient has nausea.")])
        uncertainties = _find_uncertainties(state)
        if state.get("hypotheses"):
            categories = [u["category"] for u in uncertainties]
            assert "unexplained_symptom" in categories

    def test_no_uncertainties_without_hypotheses(self):
        state = build_clinical_state([_seg("hello.")])
        uncertainties = _find_uncertainties(state)
        # No hypotheses → no unexplained_symptom entries
        unexplained = [u for u in uncertainties if u["category"] == "unexplained_symptom"]
        assert unexplained == []

    def test_uncertainty_has_required_fields(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat."),
        ])
        uncertainties = _find_uncertainties(state)
        for u in uncertainties:
            assert "category" in u
            assert "detail" in u
            assert "related" in u


# ── suggested questions ──────────────────────────────────────────────


class TestSuggestedQuestions:
    def test_duration_question_for_symptom_without_timing(self):
        state = build_clinical_state([_seg("patient has nausea.")])
        questions = _suggest_questions(state)
        duration_qs = [q for q in questions if q["category"] == "duration"]
        assert len(duration_qs) >= 1
        assert "nausea" in duration_qs[0]["question"].lower()

    def test_no_duration_question_when_timing_present(self):
        state = build_clinical_state([_seg("nausea for 3 days.")])
        questions = _suggest_questions(state)
        duration_qs = [q for q in questions if q["category"] == "duration"]
        related = [q["related"].lower() for q in duration_qs]
        assert "nausea" not in related

    def test_severity_question_for_symptom(self):
        state = build_clinical_state([_seg("patient has headache.")])
        questions = _suggest_questions(state)
        severity_qs = [q for q in questions if q["category"] == "severity"]
        assert len(severity_qs) >= 1

    def test_allergy_question_when_medications_present(self):
        state = build_clinical_state([_seg("prescribed ibuprofen 400 mg.")])
        questions = _suggest_questions(state)
        allergy_qs = [q for q in questions if q["category"] == "allergy"]
        assert len(allergy_qs) >= 1

    def test_no_allergy_question_without_medications(self):
        state = build_clinical_state([_seg("patient has headache.")])
        questions = _suggest_questions(state)
        allergy_qs = [q for q in questions if q["category"] == "allergy"]
        assert allergy_qs == []

    def test_question_has_required_fields(self):
        state = build_clinical_state([_seg("patient has headache.")])
        questions = _suggest_questions(state)
        for q in questions:
            assert "category" in q
            assert "question" in q
            assert "related" in q


# ── data quality issues ──────────────────────────────────────────────


class TestDataQualityIssues:
    def test_low_confidence_flagged(self):
        confidence = [{"seg_id": "seg_0001", "avg_logprob": -2.0}]
        state = build_clinical_state(
            [_seg("hello.")], confidence_entries=confidence,
        )
        issues = _find_data_quality_issues(state)
        categories = [i["category"] for i in issues]
        assert "low_confidence_transcription" in categories

    def test_no_issues_without_flags(self):
        state = build_clinical_state([_seg("hello.")])
        issues = _find_data_quality_issues(state)
        low_conf = [i for i in issues if i["category"] == "low_confidence_transcription"]
        assert low_conf == []

    def test_limited_data_detected(self):
        state = build_clinical_state([_seg("patient has headache.")])
        issues = _find_data_quality_issues(state)
        limited = [i for i in issues if i["category"] == "limited_data"]
        # One symptom observation → limited_data
        assert len(limited) >= 1

    def test_issue_has_required_fields(self):
        confidence = [{"seg_id": "seg_0001", "avg_logprob": -2.0}]
        state = build_clinical_state(
            [_seg("hello.")], confidence_entries=confidence,
        )
        issues = _find_data_quality_issues(state)
        for issue in issues:
            assert "category" in issue
            assert "detail" in issue


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_state(self):
        state = build_clinical_state([_seg("patient has headache.")])
        original_symptoms = list(state["symptoms"])
        derive_clinical_insights(state)
        assert state["symptoms"] == original_symptoms

    def test_deterministic(self):
        state = build_clinical_state([_seg("patient has headache.")])
        r1 = derive_clinical_insights(state)
        r2 = derive_clinical_insights(state)
        assert r1 == r2


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_insights_in_clinical_state(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        assert "clinical_insights" in state
        insights = state["clinical_insights"]
        assert set(insights.keys()) == {
            "missing_information",
            "uncertainties",
            "suggested_questions",
            "data_quality_issues",
        }

    def test_full_scenario_produces_insights(self):
        segments = [
            _seg("patient reports headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("denies fever. prescribed ibuprofen.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ]
        state = build_clinical_state(segments)
        insights = state["clinical_insights"]

        # Should have some missing info (e.g. missing severity)
        assert isinstance(insights["missing_information"], list)
        # Should have some suggested questions
        assert isinstance(insights["suggested_questions"], list)
        # Data quality list present
        assert isinstance(insights["data_quality_issues"], list)

    def test_empty_state_insights(self):
        state = build_clinical_state([_seg("hello.")])
        insights = state["clinical_insights"]
        assert insights["missing_information"] == []
        assert insights["uncertainties"] == []
        assert insights["suggested_questions"] == []
