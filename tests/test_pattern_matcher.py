"""Tests for deterministic clinical pattern matcher."""

from __future__ import annotations

import pytest

from app.pattern_matcher import match_clinical_patterns
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────


def _rep(
    symptom: str,
    severity: str | None = None,
    aggravating_factors: list[str] | None = None,
    relieving_factors: list[str] | None = None,
    **kwargs,
) -> dict:
    return {
        "symptom": symptom,
        "severity": severity,
        "duration": kwargs.get("duration"),
        "onset": kwargs.get("onset"),
        "pattern": kwargs.get("pattern"),
        "progression": kwargs.get("progression"),
        "laterality": kwargs.get("laterality"),
        "radiation": kwargs.get("radiation"),
        "aggravating_factors": aggravating_factors or [],
        "relieving_factors": relieving_factors or [],
    }


def _state_with_reps(*reps: dict) -> dict:
    """Build a minimal clinical_state with symptom_representations."""
    return {
        "derived": {
            "symptom_representations": list(reps),
        },
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


def _pattern_names(patterns: list[dict]) -> list[str]:
    return [p["pattern"] for p in patterns]


# ══════════════════════════════════════════════════════════════════
# Angina-like
# ══════════════════════════════════════════════════════════════════


class TestAnginaLike:
    def test_matches(self):
        state = _state_with_reps(
            _rep("chest pain",
                 aggravating_factors=["exertion"],
                 relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" in _pattern_names(patterns)
        p = patterns[0]
        assert p["label"] == "Angina-like pattern"
        assert "chest pain" in p["evidence"]
        assert "aggravating factor: exertion" in p["evidence"]
        assert "relieving factor: rest" in p["evidence"]

    def test_exercise_variant(self):
        state = _state_with_reps(
            _rep("chest pain",
                 aggravating_factors=["exercise"],
                 relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" in _pattern_names(patterns)

    def test_walking_variant(self):
        state = _state_with_reps(
            _rep("chest pain",
                 aggravating_factors=["walking"],
                 relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" in _pattern_names(patterns)

    def test_no_match_without_chest_pain(self):
        state = _state_with_reps(
            _rep("headache",
                 aggravating_factors=["exertion"],
                 relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" not in _pattern_names(patterns)

    def test_no_match_without_aggravating(self):
        state = _state_with_reps(
            _rep("chest pain", relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" not in _pattern_names(patterns)

    def test_no_match_without_relieving(self):
        state = _state_with_reps(
            _rep("chest pain", aggravating_factors=["exertion"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" not in _pattern_names(patterns)

    def test_factors_must_be_on_chest_pain(self):
        """Factors on a different symptom must not trigger angina pattern."""
        state = _state_with_reps(
            _rep("chest pain"),
            _rep("headache",
                 aggravating_factors=["exertion"],
                 relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" not in _pattern_names(patterns)


# ══════════════════════════════════════════════════════════════════
# Lower respiratory
# ══════════════════════════════════════════════════════════════════


class TestLowerRespiratory:
    def test_matches(self):
        state = _state_with_reps(
            _rep("cough"),
            _rep("fever"),
            _rep("shortness of breath"),
        )
        patterns = match_clinical_patterns(state)
        assert "lower_respiratory_pattern" in _pattern_names(patterns)
        p = [x for x in patterns if x["pattern"] == "lower_respiratory_pattern"][0]
        assert "cough" in p["evidence"]
        assert "fever" in p["evidence"]
        assert "shortness of breath" in p["evidence"]

    def test_dyspnea_variant(self):
        state = _state_with_reps(
            _rep("cough"),
            _rep("fever"),
            _rep("dyspnea"),
        )
        patterns = match_clinical_patterns(state)
        assert "lower_respiratory_pattern" in _pattern_names(patterns)

    def test_no_match_without_cough(self):
        state = _state_with_reps(
            _rep("fever"),
            _rep("shortness of breath"),
        )
        patterns = match_clinical_patterns(state)
        assert "lower_respiratory_pattern" not in _pattern_names(patterns)

    def test_no_match_without_fever(self):
        state = _state_with_reps(
            _rep("cough"),
            _rep("shortness of breath"),
        )
        patterns = match_clinical_patterns(state)
        assert "lower_respiratory_pattern" not in _pattern_names(patterns)

    def test_no_match_without_dyspnea(self):
        state = _state_with_reps(
            _rep("cough"),
            _rep("fever"),
        )
        patterns = match_clinical_patterns(state)
        assert "lower_respiratory_pattern" not in _pattern_names(patterns)


# ══════════════════════════════════════════════════════════════════
# Migraine-like
# ══════════════════════════════════════════════════════════════════


class TestMigraineLike:
    def test_matches(self):
        state = _state_with_reps(
            _rep("headache"),
            _rep("nausea"),
        )
        patterns = match_clinical_patterns(state)
        assert "migraine_like" in _pattern_names(patterns)
        p = [x for x in patterns if x["pattern"] == "migraine_like"][0]
        assert "headache" in p["evidence"]
        assert "nausea" in p["evidence"]

    def test_matches_with_severity(self):
        state = _state_with_reps(
            _rep("headache", severity="severe"),
            _rep("nausea"),
        )
        patterns = match_clinical_patterns(state)
        assert "migraine_like" in _pattern_names(patterns)
        p = [x for x in patterns if x["pattern"] == "migraine_like"][0]
        assert "severity: severe" in p["evidence"]

    def test_no_severity_in_evidence_when_mild(self):
        state = _state_with_reps(
            _rep("headache", severity="mild"),
            _rep("nausea"),
        )
        patterns = match_clinical_patterns(state)
        assert "migraine_like" in _pattern_names(patterns)
        p = [x for x in patterns if x["pattern"] == "migraine_like"][0]
        assert "severity: severe" not in p["evidence"]

    def test_no_match_without_headache(self):
        state = _state_with_reps(
            _rep("nausea"),
        )
        patterns = match_clinical_patterns(state)
        assert "migraine_like" not in _pattern_names(patterns)

    def test_no_match_without_nausea(self):
        state = _state_with_reps(
            _rep("headache"),
        )
        patterns = match_clinical_patterns(state)
        assert "migraine_like" not in _pattern_names(patterns)


# ══════════════════════════════════════════════════════════════════
# Urinary irritative
# ══════════════════════════════════════════════════════════════════


class TestUrinaryIrritative:
    def test_matches(self):
        state = _state_with_reps(
            _rep("dysuria"),
            _rep("frequency"),
        )
        patterns = match_clinical_patterns(state)
        assert "urinary_irritative_pattern" in _pattern_names(patterns)

    def test_painful_urination_variant(self):
        state = _state_with_reps(
            _rep("painful urination"),
            _rep("urinary frequency"),
        )
        patterns = match_clinical_patterns(state)
        assert "urinary_irritative_pattern" in _pattern_names(patterns)

    def test_no_match_without_dysuria(self):
        state = _state_with_reps(
            _rep("frequency"),
        )
        patterns = match_clinical_patterns(state)
        assert "urinary_irritative_pattern" not in _pattern_names(patterns)

    def test_no_match_without_frequency(self):
        state = _state_with_reps(
            _rep("dysuria"),
        )
        patterns = match_clinical_patterns(state)
        assert "urinary_irritative_pattern" not in _pattern_names(patterns)


# ══════════════════════════════════════════════════════════════════
# Gastroenteritis-like
# ══════════════════════════════════════════════════════════════════


class TestGastroenteritisLike:
    def test_matches_with_nausea(self):
        state = _state_with_reps(
            _rep("diarrhea"),
            _rep("nausea"),
        )
        patterns = match_clinical_patterns(state)
        assert "gastroenteritis_like" in _pattern_names(patterns)
        p = [x for x in patterns if x["pattern"] == "gastroenteritis_like"][0]
        assert "diarrhea" in p["evidence"]
        assert "nausea" in p["evidence"]

    def test_matches_with_vomiting(self):
        state = _state_with_reps(
            _rep("diarrhea"),
            _rep("vomiting"),
        )
        patterns = match_clinical_patterns(state)
        assert "gastroenteritis_like" in _pattern_names(patterns)

    def test_optional_fever(self):
        state = _state_with_reps(
            _rep("diarrhea"),
            _rep("nausea"),
            _rep("fever"),
        )
        patterns = match_clinical_patterns(state)
        p = [x for x in patterns if x["pattern"] == "gastroenteritis_like"][0]
        assert "fever" in p["evidence"]

    def test_no_match_without_diarrhea(self):
        state = _state_with_reps(
            _rep("nausea"),
            _rep("vomiting"),
        )
        patterns = match_clinical_patterns(state)
        assert "gastroenteritis_like" not in _pattern_names(patterns)

    def test_no_match_without_nausea_or_vomiting(self):
        state = _state_with_reps(
            _rep("diarrhea"),
        )
        patterns = match_clinical_patterns(state)
        assert "gastroenteritis_like" not in _pattern_names(patterns)


# ══════════════════════════════════════════════════════════════════
# Factor isolation
# ══════════════════════════════════════════════════════════════════


class TestFactorIsolation:
    def test_factors_on_wrong_symptom_no_match(self):
        """Aggravating/relieving on headache must not trigger angina."""
        state = _state_with_reps(
            _rep("chest pain"),
            _rep("headache",
                 aggravating_factors=["exertion"],
                 relieving_factors=["rest"]),
        )
        patterns = match_clinical_patterns(state)
        assert "angina_like" not in _pattern_names(patterns)

    def test_severity_on_wrong_symptom_no_migraine_evidence(self):
        """Severity on nausea must not appear in migraine evidence."""
        state = _state_with_reps(
            _rep("headache"),
            _rep("nausea", severity="severe"),
        )
        patterns = match_clinical_patterns(state)
        p = [x for x in patterns if x["pattern"] == "migraine_like"][0]
        assert "severity: severe" not in p["evidence"]


# ══════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_state(self):
        assert match_clinical_patterns({}) == []

    def test_empty_derived(self):
        assert match_clinical_patterns({"derived": {}}) == []

    def test_empty_reps(self):
        assert match_clinical_patterns(_state_with_reps()) == []

    def test_no_duplicate_patterns(self):
        state = _state_with_reps(
            _rep("headache"),
            _rep("nausea"),
        )
        patterns = match_clinical_patterns(state)
        names = _pattern_names(patterns)
        assert len(names) == len(set(names))

    def test_multiple_patterns_can_match(self):
        state = _state_with_reps(
            _rep("headache"),
            _rep("nausea"),
            _rep("diarrhea"),
        )
        patterns = match_clinical_patterns(state)
        names = _pattern_names(patterns)
        assert "migraine_like" in names
        assert "gastroenteritis_like" in names


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state_with_reps(
            _rep("headache", severity="severe"),
            _rep("nausea"),
            _rep("diarrhea"),
        )
        r1 = match_clinical_patterns(state)
        r2 = match_clinical_patterns(state)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_clinical_patterns_in_derived(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        assert "clinical_patterns" in state["derived"]
        patterns = state["derived"]["clinical_patterns"]
        assert isinstance(patterns, list)

    def test_migraine_detected_end_to_end(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        names = _pattern_names(state["derived"]["clinical_patterns"])
        assert "migraine_like" in names

    def test_empty_patterns_when_no_match(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["clinical_patterns"] == []

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

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        original_patterns = list(state["derived"]["clinical_patterns"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["clinical_patterns"] == original_patterns
        assert "ai_overlay" in state["derived"]
