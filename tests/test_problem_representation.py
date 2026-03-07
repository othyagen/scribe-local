"""Tests for deterministic structured problem representation."""

from __future__ import annotations

import pytest

from app.problem_representation import build_problem_representation
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────


def _state(**overrides) -> dict:
    """Minimal clinical_state dict with overrides."""
    base: dict = {
        "symptoms": [],
        "qualifiers": [],
        "timeline": [],
        "durations": [],
        "negations": [],
        "medications": [],
        "diagnostic_hints": [],
    }
    base.update(overrides)
    return base


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ══════════════════════════════════════════════════════════════════
# Core symptom selection
# ══════════════════════════════════════════════════════════════════


class TestCoreSymptomSelection:
    def test_single_symptom(self):
        pr = build_problem_representation(_state(symptoms=["headache"]))
        assert pr["core_symptom"] == "headache"

    def test_multiple_symptoms_first_wins(self):
        pr = build_problem_representation(_state(
            symptoms=["headache", "nausea", "fever"],
        ))
        assert pr["core_symptom"] == "headache"

    def test_timeline_earliest_wins(self):
        pr = build_problem_representation(_state(
            symptoms=["nausea", "headache"],
            timeline=[
                {"symptom": "nausea", "time_expression": None, "t_start": 5.0},
                {"symptom": "headache", "time_expression": None, "t_start": 1.0},
            ],
        ))
        assert pr["core_symptom"] == "headache"

    def test_no_symptoms(self):
        pr = build_problem_representation(_state())
        assert pr["core_symptom"] is None

    def test_timeline_with_none_t_start(self):
        pr = build_problem_representation(_state(
            symptoms=["headache", "nausea"],
            timeline=[
                {"symptom": "headache", "time_expression": None, "t_start": None},
                {"symptom": "nausea", "time_expression": None, "t_start": 2.0},
            ],
        ))
        assert pr["core_symptom"] == "nausea"


# ══════════════════════════════════════════════════════════════════
# Qualifier population
# ══════════════════════════════════════════════════════════════════


class TestQualifierPopulation:
    def test_qualifiers_for_core_symptom(self):
        pr = build_problem_representation(_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {
                    "severity": "severe",
                    "onset": "sudden",
                    "pattern": "constant",
                },
            }],
        ))
        assert pr["severity"] == "severe"
        assert pr["onset"] == "sudden"
        assert pr["pattern"] == "constant"

    def test_qualifiers_missing(self):
        pr = build_problem_representation(_state(symptoms=["headache"]))
        assert pr["severity"] is None
        assert pr["onset"] is None
        assert pr["pattern"] is None
        assert pr["progression"] is None
        assert pr["laterality"] is None
        assert pr["radiation"] is None

    def test_qualifiers_no_match(self):
        pr = build_problem_representation(_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "nausea",
                "qualifiers": {"severity": "mild"},
            }],
        ))
        assert pr["severity"] is None

    def test_radiation_from_qualifiers(self):
        pr = build_problem_representation(_state(
            symptoms=["chest pain"],
            qualifiers=[{
                "symptom": "chest pain",
                "qualifiers": {"radiation": "to left arm"},
            }],
        ))
        assert pr["radiation"] == "to left arm"


# ══════════════════════════════════════════════════════════════════
# Duration
# ══════════════════════════════════════════════════════════════════


class TestDuration:
    def test_duration_from_timeline(self):
        pr = build_problem_representation(_state(
            symptoms=["headache"],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
            ],
            durations=["5 weeks"],
        ))
        assert pr["duration"] == "3 days"

    def test_duration_from_durations_fallback(self):
        pr = build_problem_representation(_state(
            symptoms=["headache"],
            durations=["2 weeks"],
        ))
        assert pr["duration"] == "2 weeks"

    def test_duration_missing(self):
        pr = build_problem_representation(_state(symptoms=["headache"]))
        assert pr["duration"] is None


# ══════════════════════════════════════════════════════════════════
# Associated symptoms
# ══════════════════════════════════════════════════════════════════


class TestAssociatedSymptoms:
    def test_associated_excludes_core(self):
        pr = build_problem_representation(_state(
            symptoms=["headache", "nausea", "fever"],
        ))
        assert "headache" not in pr["associated_symptoms"]
        assert "nausea" in pr["associated_symptoms"]
        assert "fever" in pr["associated_symptoms"]

    def test_associated_preserves_order(self):
        pr = build_problem_representation(_state(
            symptoms=["headache", "fever", "nausea"],
        ))
        assert pr["associated_symptoms"] == ["fever", "nausea"]

    def test_single_symptom_no_associated(self):
        pr = build_problem_representation(_state(symptoms=["headache"]))
        assert pr["associated_symptoms"] == []


# ══════════════════════════════════════════════════════════════════
# Factors
# ══════════════════════════════════════════════════════════════════


class TestFactors:
    def test_aggravating_from_core_symptom(self):
        pr = build_problem_representation(_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"aggravating_factors": ["movement", "light"]},
            }],
        ))
        assert pr["aggravating_factors"] == ["movement", "light"]

    def test_relieving_from_core_symptom(self):
        pr = build_problem_representation(_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"relieving_factors": ["rest", "darkness"]},
            }],
        ))
        assert pr["relieving_factors"] == ["rest", "darkness"]

    def test_factors_fallback_to_all(self):
        pr = build_problem_representation(_state(
            symptoms=["headache", "nausea"],
            qualifiers=[
                {"symptom": "headache", "qualifiers": {"severity": "severe"}},
                {"symptom": "nausea", "qualifiers": {
                    "aggravating_factors": ["eating"],
                }},
            ],
        ))
        # Core symptom (headache) has no aggravating_factors → fall back to all
        assert pr["aggravating_factors"] == ["eating"]

    def test_factors_deduplicated(self):
        pr = build_problem_representation(_state(
            symptoms=["headache", "nausea"],
            qualifiers=[
                {"symptom": "headache", "qualifiers": {"severity": "severe"}},
                {"symptom": "nausea", "qualifiers": {
                    "aggravating_factors": ["movement"],
                }},
                {"symptom": "fever", "qualifiers": {
                    "aggravating_factors": ["movement", "stress"],
                }},
            ],
        ))
        # Fallback; "movement" appears in two entries → only once
        assert pr["aggravating_factors"].count("movement") == 1
        assert "stress" in pr["aggravating_factors"]

    def test_no_factors(self):
        pr = build_problem_representation(_state(symptoms=["headache"]))
        assert pr["aggravating_factors"] == []
        assert pr["relieving_factors"] == []


# ══════════════════════════════════════════════════════════════════
# Negatives and hints
# ══════════════════════════════════════════════════════════════════


class TestNegativesAndHints:
    def test_pertinent_negatives(self):
        negations = ["no fever", "denies chest pain"]
        pr = build_problem_representation(_state(negations=negations))
        assert pr["pertinent_negatives"] == negations

    def test_diagnostic_hints_names(self):
        hints = [
            {"condition": "Pharyngitis", "snomed_code": "363746003", "evidence": ["fever"]},
            {"condition": "Migraine", "snomed_code": "37796009", "evidence": ["headache"]},
        ]
        pr = build_problem_representation(_state(diagnostic_hints=hints))
        assert pr["diagnostic_hints"] == ["Pharyngitis", "Migraine"]

    def test_empty_hints(self):
        pr = build_problem_representation(_state())
        assert pr["diagnostic_hints"] == []


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state(
            symptoms=["headache", "nausea"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"severity": "severe", "onset": "sudden"},
            }],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
            ],
            durations=["3 days"],
            negations=["no fever"],
            diagnostic_hints=[
                {"condition": "Migraine", "snomed_code": "37796009", "evidence": ["headache"]},
            ],
        )
        pr1 = build_problem_representation(state)
        pr2 = build_problem_representation(state)
        assert pr1 == pr2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_derived_key_present_in_clinical_state(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "derived" in state
        assert "problem_representation" in state["derived"]
        assert isinstance(state["derived"]["problem_representation"], dict)

    def test_problem_focus_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert state["derived"]["problem_focus"] == "headache"

    def test_problem_focus_none_when_no_symptoms(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["problem_focus"] is None

    def test_derived_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_pr = state["derived"]["problem_representation"]

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)

        # problem_representation preserved
        assert state["derived"]["problem_representation"] == original_pr
        # ai_overlay added alongside
        assert state["derived"]["ai_overlay"]["soap_draft"] == "Draft."

    def test_full_example(self):
        segments = [
            _seg("patient reports severe headache for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("also has nausea and dizziness.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
            _seg("denies fever, no chest pain.",
                 seg_id="seg_0003", t0=5.0, t1=7.0),
        ]
        state = build_clinical_state(segments)
        pr = state["derived"]["problem_representation"]

        assert pr["core_symptom"] is not None
        assert isinstance(pr["associated_symptoms"], list)
        assert isinstance(pr["pertinent_negatives"], list)
        assert isinstance(pr["diagnostic_hints"], list)
        assert isinstance(pr["timeline"], list)
