"""Tests for deterministic clinical problem summary generator."""

from __future__ import annotations

import pytest

from app.problem_summary import summarize_problem
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────


def _derived(
    core: str | None,
    symptom_reps: list[dict] | None = None,
) -> dict:
    """Build a minimal clinical_state with derived fields."""
    if symptom_reps is None and core is not None:
        symptom_reps = [_rep(core)]
    return {
        "derived": {
            "problem_focus": core,
            "symptom_representations": symptom_reps or [],
        },
    }


def _rep(
    symptom: str,
    severity: str | None = None,
    duration: str | None = None,
    onset: str | None = None,
    pattern: str | None = None,
    progression: str | None = None,
    laterality: str | None = None,
    radiation: str | None = None,
    aggravating_factors: list[str] | None = None,
    relieving_factors: list[str] | None = None,
) -> dict:
    return {
        "symptom": symptom,
        "severity": severity,
        "duration": duration,
        "onset": onset,
        "pattern": pattern,
        "progression": progression,
        "laterality": laterality,
        "radiation": radiation,
        "aggravating_factors": aggravating_factors or [],
        "relieving_factors": relieving_factors or [],
    }


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
# Missing core symptom
# ══════════════════════════════════════════════════════════════════


class TestMissingCore:
    def test_empty_string_when_no_focus(self):
        assert summarize_problem(_derived(None)) == ""

    def test_empty_string_when_no_derived(self):
        assert summarize_problem({}) == ""

    def test_empty_string_when_focus_empty_string(self):
        state = {"derived": {"problem_focus": "", "symptom_representations": []}}
        assert summarize_problem(state) == ""

    def test_empty_string_when_no_matching_rep(self):
        state = _derived("headache", symptom_reps=[_rep("nausea")])
        assert summarize_problem(state) == ""


# ══════════════════════════════════════════════════════════════════
# Single symptom — basic fields
# ══════════════════════════════════════════════════════════════════


class TestSingleSymptom:
    def test_symptom_only(self):
        result = summarize_problem(_derived("headache"))
        assert result == "Headache."

    def test_symptom_with_severity(self):
        state = _derived("headache", [_rep("headache", severity="severe")])
        result = summarize_problem(state)
        assert result == "Severe headache."

    def test_symptom_with_duration(self):
        state = _derived("headache", [_rep("headache", duration="3 days")])
        result = summarize_problem(state)
        assert result == "Headache, for 3 days."

    def test_symptom_with_severity_and_duration(self):
        state = _derived("headache", [
            _rep("headache", severity="severe", duration="3 days"),
        ])
        result = summarize_problem(state)
        assert result == "Severe headache, for 3 days."

    def test_symptom_with_onset(self):
        state = _derived("headache", [_rep("headache", onset="sudden")])
        result = summarize_problem(state)
        assert result == "Headache, sudden onset."

    def test_symptom_with_pattern(self):
        state = _derived("headache", [_rep("headache", pattern="intermittent")])
        result = summarize_problem(state)
        assert result == "Headache, intermittent."

    def test_symptom_with_progression(self):
        state = _derived("headache", [_rep("headache", progression="worsening")])
        result = summarize_problem(state)
        assert result == "Headache, worsening."

    def test_symptom_with_laterality(self):
        state = _derived("headache", [_rep("headache", laterality="left")])
        result = summarize_problem(state)
        assert result == "Headache, left side."

    def test_symptom_with_radiation(self):
        state = _derived("chest pain", [
            _rep("chest pain", radiation="to left arm"),
        ])
        result = summarize_problem(state)
        assert result == "Chest pain, radiating to left arm."


# ══════════════════════════════════════════════════════════════════
# Factors
# ══════════════════════════════════════════════════════════════════


class TestFactors:
    def test_aggravating_factors(self):
        state = _derived("headache", [
            _rep("headache", aggravating_factors=["movement", "light"]),
        ])
        result = summarize_problem(state)
        assert result == "Headache, worse with movement, light."

    def test_relieving_factors(self):
        state = _derived("headache", [
            _rep("headache", relieving_factors=["rest", "darkness"]),
        ])
        result = summarize_problem(state)
        assert result == "Headache, relieved by rest, darkness."

    def test_both_factors(self):
        state = _derived("headache", [
            _rep("headache",
                 aggravating_factors=["movement"],
                 relieving_factors=["rest"]),
        ])
        result = summarize_problem(state)
        assert result == "Headache, worse with movement, relieved by rest."


# ══════════════════════════════════════════════════════════════════
# Full ordering
# ══════════════════════════════════════════════════════════════════


class TestFullOrdering:
    def test_all_fields(self):
        state = _derived("headache", [
            _rep("headache",
                 severity="severe",
                 duration="3 days",
                 onset="sudden",
                 pattern="constant",
                 progression="worsening",
                 laterality="left",
                 radiation="to left arm",
                 aggravating_factors=["movement"],
                 relieving_factors=["rest"]),
        ])
        result = summarize_problem(state)
        assert result == (
            "Severe headache, for 3 days, sudden onset, constant, "
            "worsening, left side, radiating to left arm, "
            "worse with movement, relieved by rest."
        )

    def test_severity_duration_progression(self):
        state = _derived("headache", [
            _rep("headache",
                 severity="severe",
                 duration="3 days",
                 progression="worsening"),
        ])
        result = summarize_problem(state)
        assert result == "Severe headache, for 3 days, worsening."


# ══════════════════════════════════════════════════════════════════
# Multiple symptoms — qualifier isolation
# ══════════════════════════════════════════════════════════════════


class TestMultipleSymptoms:
    def test_additional_symptoms_appended(self):
        state = _derived("headache", [
            _rep("headache", severity="severe", duration="3 days"),
            _rep("nausea"),
        ])
        result = summarize_problem(state)
        assert result == "Severe headache, for 3 days, with additional nausea."

    def test_additional_symptoms_no_qualifier_inheritance(self):
        """Core symptom attributes must not appear on additional symptoms."""
        state = _derived("headache", [
            _rep("headache",
                 severity="severe",
                 duration="3 days",
                 progression="worsening"),
            _rep("nausea", duration="since this morning"),
        ])
        result = summarize_problem(state)
        # Correct: core attributes on headache, nausea listed by name only
        assert result == (
            "Severe headache, for 3 days, worsening, "
            "with additional nausea."
        )
        # Must NOT say "severe headache and nausea for 3 days"
        assert "nausea for 3 days" not in result
        assert "severe headache and nausea" not in result.lower()

    def test_multiple_additional_symptoms(self):
        state = _derived("headache", [
            _rep("headache"),
            _rep("nausea"),
            _rep("dizziness"),
            _rep("fever"),
        ])
        result = summarize_problem(state)
        assert result == "Headache, with additional nausea, dizziness, fever."

    def test_additional_symptoms_own_qualifiers_ignored_in_summary(self):
        """Additional symptoms may have their own qualifiers, but the
        summary only names them — their qualifiers stay in
        symptom_representations for structured access."""
        state = _derived("headache", [
            _rep("headache", severity="severe"),
            _rep("nausea", severity="mild", pattern="intermittent"),
        ])
        result = summarize_problem(state)
        # nausea's qualifiers do not appear in the summary
        assert "mild" not in result
        assert "intermittent" not in result
        assert "with additional nausea" in result


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _derived("headache", [
            _rep("headache", severity="severe", duration="3 days"),
            _rep("nausea"),
        ])
        r1 = summarize_problem(state)
        r2 = summarize_problem(state)
        assert r1 == r2

    def test_capitalization(self):
        state = _derived("headache")
        result = summarize_problem(state)
        assert result[0].isupper()
        assert result.endswith(".")


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_problem_summary_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "problem_summary" in state["derived"]
        assert isinstance(state["derived"]["problem_summary"], str)

    def test_problem_summary_empty_when_no_symptoms(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["problem_summary"] == ""

    def test_problem_summary_mentions_core(self):
        state = build_clinical_state([
            _seg("patient has severe headache for 3 days."),
        ])
        summary = state["derived"]["problem_summary"]
        assert "headache" in summary.lower()

    def test_structured_data_unchanged(self):
        """problem_summary must not alter any structured fields."""
        state = build_clinical_state([
            _seg("patient has headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=2.0),
        ])
        # Structured fields still intact
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert isinstance(state["derived"]["problem_representation"], dict)
        assert isinstance(state["derived"]["symptom_representations"], list)
        assert state["derived"]["problem_focus"] is not None

    def test_summary_preserved_after_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_summary = state["derived"]["problem_summary"]

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["problem_summary"] == original_summary

    def test_end_to_end_multi_symptom(self):
        state = build_clinical_state([
            _seg("patient reports severe headache for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("also has nausea and dizziness.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ])
        summary = state["derived"]["problem_summary"]
        assert summary  # non-empty
        assert summary[0].isupper()
        assert summary.endswith(".")
        # Core symptom attributes present
        assert "headache" in summary.lower()
        # Additional symptoms mentioned
        assert "nausea" in summary.lower() or "dizziness" in summary.lower()
