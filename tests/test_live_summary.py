"""Tests for deterministic running clinical summary generator."""

from __future__ import annotations

import pytest

from app.live_summary import build_running_summary
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _derived(**overrides) -> dict:
    """Build a minimal clinical_state with derived fields."""
    base: dict = {
        "derived": {
            "problem_summary": "",
            "problem_focus": None,
            "symptom_representations": [],
            "clinical_patterns": [],
        },
    }
    base["derived"].update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════
# Core problem
# ══════════════════════════════════════════════════════════════════


class TestCoreProblem:
    def test_core_problem_from_summary(self):
        state = _derived(problem_summary="Headache for 3 days.")
        result = build_running_summary(state)
        assert result["core_problem"] == "Headache for 3 days."

    def test_core_problem_empty_when_no_summary(self):
        state = _derived()
        result = build_running_summary(state)
        assert result["core_problem"] == ""

    def test_core_problem_empty_when_no_derived(self):
        result = build_running_summary({})
        assert result["core_problem"] == ""


# ══════════════════════════════════════════════════════════════════
# Additional symptoms
# ══════════════════════════════════════════════════════════════════


class TestAdditionalSymptoms:
    def test_excludes_core_symptom(self):
        state = _derived(
            problem_focus="headache",
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": "nausea"},
                {"symptom": "fever"},
            ],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == ["nausea", "fever"]

    def test_case_insensitive_exclusion(self):
        state = _derived(
            problem_focus="Headache",
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": "nausea"},
            ],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == ["nausea"]

    def test_single_symptom_no_additional(self):
        state = _derived(
            problem_focus="headache",
            symptom_representations=[{"symptom": "headache"}],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == []

    def test_no_symptoms_empty(self):
        state = _derived()
        result = build_running_summary(state)
        assert result["additional_symptoms"] == []

    def test_no_focus_all_are_additional(self):
        state = _derived(
            problem_focus=None,
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": "nausea"},
            ],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == ["headache", "nausea"]

    def test_preserves_order(self):
        state = _derived(
            problem_focus="headache",
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": "fever"},
                {"symptom": "cough"},
                {"symptom": "nausea"},
            ],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == ["fever", "cough", "nausea"]

    def test_no_duplicates(self):
        state = _derived(
            problem_focus="headache",
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": "nausea"},
                {"symptom": "nausea"},
            ],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == ["nausea"]

    def test_empty_symptom_name_skipped(self):
        state = _derived(
            problem_focus="headache",
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": ""},
                {"symptom": "nausea"},
            ],
        )
        result = build_running_summary(state)
        assert result["additional_symptoms"] == ["nausea"]


# ══════════════════════════════════════════════════════════════════
# Patterns detected
# ══════════════════════════════════════════════════════════════════


class TestPatternsDetected:
    def test_extracts_labels(self):
        state = _derived(
            clinical_patterns=[
                {"pattern": "migraine_like", "label": "Migraine-like pattern",
                 "evidence": ["headache", "nausea"]},
            ],
        )
        result = build_running_summary(state)
        assert result["patterns_detected"] == ["Migraine-like pattern"]

    def test_multiple_patterns(self):
        state = _derived(
            clinical_patterns=[
                {"pattern": "migraine_like", "label": "Migraine-like pattern",
                 "evidence": []},
                {"pattern": "gastroenteritis_like",
                 "label": "Gastroenteritis-like pattern", "evidence": []},
            ],
        )
        result = build_running_summary(state)
        assert result["patterns_detected"] == [
            "Migraine-like pattern",
            "Gastroenteritis-like pattern",
        ]

    def test_no_patterns(self):
        state = _derived()
        result = build_running_summary(state)
        assert result["patterns_detected"] == []

    def test_pattern_without_label_skipped(self):
        state = _derived(
            clinical_patterns=[
                {"pattern": "test"},
            ],
        )
        result = build_running_summary(state)
        assert result["patterns_detected"] == []


# ══════════════════════════════════════════════════════════════════
# Alerts
# ══════════════════════════════════════════════════════════════════


class TestAlerts:
    def test_alerts_empty(self):
        result = build_running_summary({})
        assert result["alerts"] == []

    def test_alerts_is_list(self):
        result = build_running_summary(_derived())
        assert isinstance(result["alerts"], list)


# ══════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_state(self):
        result = build_running_summary({})
        assert result == {
            "core_problem": "",
            "additional_symptoms": [],
            "patterns_detected": [],
            "alerts": [],
        }

    def test_all_keys_present(self):
        result = build_running_summary({})
        assert set(result.keys()) == {
            "core_problem", "additional_symptoms",
            "patterns_detected", "alerts",
        }


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _derived(
            problem_summary="Severe headache for 3 days.",
            problem_focus="headache",
            symptom_representations=[
                {"symptom": "headache"},
                {"symptom": "nausea"},
            ],
            clinical_patterns=[
                {"pattern": "migraine_like", "label": "Migraine-like pattern",
                 "evidence": ["headache", "nausea"]},
            ],
        )
        r1 = build_running_summary(state)
        r2 = build_running_summary(state)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_running_summary_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "running_summary" in state["derived"]
        summary = state["derived"]["running_summary"]
        assert isinstance(summary, dict)

    def test_core_problem_populated(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        summary = state["derived"]["running_summary"]
        assert summary["core_problem"] != ""
        assert "headache" in summary["core_problem"].lower()

    def test_additional_symptoms_populated(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        summary = state["derived"]["running_summary"]
        assert len(summary["additional_symptoms"]) >= 1

    def test_patterns_detected_populated(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        summary = state["derived"]["running_summary"]
        assert "Migraine-like pattern" in summary["patterns_detected"]

    def test_empty_input(self):
        state = build_clinical_state([_seg("hello.")])
        summary = state["derived"]["running_summary"]
        assert summary["core_problem"] == ""
        assert summary["additional_symptoms"] == []
        assert summary["patterns_detected"] == []
        assert summary["alerts"] == []

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

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        original_summary = dict(state["derived"]["running_summary"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["running_summary"] == original_summary
        assert "ai_overlay" in state["derived"]
