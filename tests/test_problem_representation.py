"""Tests for deterministic structured problem representation."""

from __future__ import annotations

import pytest

from app.problem_representation import (
    build_problem_representation,
    build_symptom_representations,
)
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
# Symptom representations — per-symptom qualifier isolation
# ══════════════════════════════════════════════════════════════════


_SYMPTOM_REP_KEYS = {
    "symptom", "severity", "duration", "onset", "pattern",
    "progression", "laterality", "radiation",
    "aggravating_factors", "relieving_factors",
}


class TestSymptomRepresentationStructure:
    def test_empty_symptoms(self):
        reps = build_symptom_representations(_state())
        assert reps == []

    def test_one_per_symptom(self):
        reps = build_symptom_representations(_state(
            symptoms=["headache", "nausea", "fever"],
        ))
        assert len(reps) == 3
        assert [r["symptom"] for r in reps] == ["headache", "nausea", "fever"]

    def test_has_expected_keys(self):
        reps = build_symptom_representations(_state(symptoms=["headache"]))
        assert set(reps[0].keys()) == _SYMPTOM_REP_KEYS

    def test_preserves_symptom_order(self):
        reps = build_symptom_representations(_state(
            symptoms=["fever", "headache", "nausea"],
        ))
        assert [r["symptom"] for r in reps] == ["fever", "headache", "nausea"]


class TestSymptomRepQualifiers:
    def test_qualifiers_linked_to_correct_symptom(self):
        reps = build_symptom_representations(_state(
            symptoms=["headache", "nausea"],
            qualifiers=[
                {"symptom": "headache", "qualifiers": {
                    "severity": "severe", "onset": "sudden",
                }},
                {"symptom": "nausea", "qualifiers": {
                    "severity": "mild", "pattern": "intermittent",
                }},
            ],
        ))
        headache = reps[0]
        nausea = reps[1]

        assert headache["severity"] == "severe"
        assert headache["onset"] == "sudden"
        assert headache["pattern"] is None  # not from nausea

        assert nausea["severity"] == "mild"
        assert nausea["pattern"] == "intermittent"
        assert nausea["onset"] is None  # not from headache

    def test_no_qualifier_inheritance(self):
        """Core symptom qualifiers must NOT leak to other symptoms."""
        reps = build_symptom_representations(_state(
            symptoms=["headache", "nausea"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {
                    "severity": "severe",
                    "onset": "sudden",
                    "laterality": "left",
                    "radiation": "to left arm",
                    "aggravating_factors": ["movement"],
                    "relieving_factors": ["rest"],
                },
            }],
        ))
        nausea = reps[1]
        assert nausea["severity"] is None
        assert nausea["onset"] is None
        assert nausea["laterality"] is None
        assert nausea["radiation"] is None
        assert nausea["aggravating_factors"] == []
        assert nausea["relieving_factors"] == []

    def test_symptom_without_qualifiers_all_none(self):
        reps = build_symptom_representations(_state(symptoms=["dizziness"]))
        rep = reps[0]
        assert rep["severity"] is None
        assert rep["onset"] is None
        assert rep["pattern"] is None
        assert rep["progression"] is None
        assert rep["laterality"] is None
        assert rep["radiation"] is None
        assert rep["aggravating_factors"] == []
        assert rep["relieving_factors"] == []

    def test_progression_and_laterality(self):
        reps = build_symptom_representations(_state(
            symptoms=["knee pain"],
            qualifiers=[{
                "symptom": "knee pain",
                "qualifiers": {
                    "progression": "worsening",
                    "laterality": "right",
                },
            }],
        ))
        assert reps[0]["progression"] == "worsening"
        assert reps[0]["laterality"] == "right"


class TestSymptomRepDuration:
    def test_duration_from_timeline(self):
        reps = build_symptom_representations(_state(
            symptoms=["headache", "nausea"],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
                {"symptom": "nausea", "time_expression": "since yesterday", "t_start": 1.0},
            ],
        ))
        assert reps[0]["duration"] == "3 days"
        assert reps[1]["duration"] == "since yesterday"

    def test_duration_not_shared_across_symptoms(self):
        """Duration for one symptom must NOT leak to another."""
        reps = build_symptom_representations(_state(
            symptoms=["headache", "nausea"],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
            ],
        ))
        assert reps[0]["duration"] == "3 days"
        assert reps[1]["duration"] is None

    def test_duration_none_when_no_timeline(self):
        reps = build_symptom_representations(_state(symptoms=["headache"]))
        assert reps[0]["duration"] is None

    def test_duration_none_when_timeline_has_no_expression(self):
        reps = build_symptom_representations(_state(
            symptoms=["headache"],
            timeline=[
                {"symptom": "headache", "time_expression": None, "t_start": 0.0},
            ],
        ))
        assert reps[0]["duration"] is None


class TestSymptomRepFactors:
    def test_factors_per_symptom(self):
        reps = build_symptom_representations(_state(
            symptoms=["headache", "back pain"],
            qualifiers=[
                {"symptom": "headache", "qualifiers": {
                    "aggravating_factors": ["light", "noise"],
                    "relieving_factors": ["darkness"],
                }},
                {"symptom": "back pain", "qualifiers": {
                    "aggravating_factors": ["bending"],
                    "relieving_factors": ["lying down"],
                }},
            ],
        ))
        assert reps[0]["aggravating_factors"] == ["light", "noise"]
        assert reps[0]["relieving_factors"] == ["darkness"]
        assert reps[1]["aggravating_factors"] == ["bending"]
        assert reps[1]["relieving_factors"] == ["lying down"]

    def test_no_factor_fallback_across_symptoms(self):
        """Factors must NOT fall back to other symptoms' factors."""
        reps = build_symptom_representations(_state(
            symptoms=["headache", "nausea"],
            qualifiers=[{
                "symptom": "nausea",
                "qualifiers": {"aggravating_factors": ["eating"]},
            }],
        ))
        assert reps[0]["aggravating_factors"] == []  # headache: no fallback
        assert reps[1]["aggravating_factors"] == ["eating"]


class TestSymptomRepDeterminism:
    def test_identical_input_identical_output(self):
        state = _state(
            symptoms=["headache", "nausea"],
            qualifiers=[
                {"symptom": "headache", "qualifiers": {"severity": "severe"}},
                {"symptom": "nausea", "qualifiers": {"pattern": "intermittent"}},
            ],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
            ],
        )
        r1 = build_symptom_representations(state)
        r2 = build_symptom_representations(state)
        assert r1 == r2


class TestSymptomRepCaseInsensitive:
    def test_case_insensitive_qualifier_match(self):
        reps = build_symptom_representations(_state(
            symptoms=["Headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"severity": "severe"},
            }],
        ))
        assert reps[0]["severity"] == "severe"

    def test_case_insensitive_timeline_match(self):
        reps = build_symptom_representations(_state(
            symptoms=["Headache"],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
            ],
        ))
        assert reps[0]["duration"] == "3 days"


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

    def test_symptom_representations_in_derived(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        assert "symptom_representations" in state["derived"]
        reps = state["derived"]["symptom_representations"]
        assert isinstance(reps, list)
        assert len(reps) >= 2
        syms = [r["symptom"] for r in reps]
        assert "headache" in syms
        assert "nausea" in syms

    def test_symptom_representations_empty_when_no_symptoms(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["symptom_representations"] == []

    def test_symptom_reps_qualifiers_not_shared(self):
        """End-to-end: qualifiers detected for one symptom stay on that symptom."""
        state = build_clinical_state([
            _seg("severe headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=2.0),
        ])
        reps = state["derived"]["symptom_representations"]
        by_sym = {r["symptom"]: r for r in reps}
        # "severe" should only be on headache (if detected), never on nausea
        if "headache" in by_sym and by_sym["headache"]["severity"]:
            assert by_sym.get("nausea", {}).get("severity") is None

    def test_symptom_reps_preserved_after_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_reps = list(state["derived"]["symptom_representations"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["symptom_representations"] == original_reps

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

        # symptom_representations also populated
        reps = state["derived"]["symptom_representations"]
        assert len(reps) >= 2
        # Each symptom has its own entry
        rep_syms = {r["symptom"] for r in reps}
        assert pr["core_symptom"] in rep_syms
