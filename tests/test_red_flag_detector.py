"""Tests for deterministic red flag detector."""

from __future__ import annotations

import pytest

from app.red_flag_detector import detect_red_flags
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


def _rep(
    symptom: str,
    severity: str | None = None,
    onset: str | None = None,
    **kw,
) -> dict:
    return {
        "symptom": symptom,
        "severity": severity,
        "onset": onset,
        "duration": kw.get("duration"),
        "pattern": kw.get("pattern"),
        "progression": kw.get("progression"),
        "laterality": kw.get("laterality"),
        "radiation": kw.get("radiation"),
        "aggravating_factors": kw.get("aggravating_factors", []),
        "relieving_factors": kw.get("relieving_factors", []),
    }


def _state_with_reps(*reps: dict) -> dict:
    """Build a minimal clinical_state with symptom_representations."""
    return {
        "derived": {
            "symptom_representations": list(reps),
        },
    }


def _flag_names(flags: list[dict]) -> list[str]:
    return [f["flag"] for f in flags]


# ══════════════════════════════════════════════════════════════════
# Sudden severe headache
# ══════════════════════════════════════════════════════════════════


class TestSuddenSevereHeadache:
    def test_matches(self):
        state = _state_with_reps(
            _rep("headache", severity="severe", onset="sudden"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" in _flag_names(flags)
        f = flags[0]
        assert f["label"] == "Sudden severe headache"
        assert f["severity"] == "high"
        assert "headache" in f["evidence"]
        assert "severity: severe" in f["evidence"]
        assert "onset: sudden" in f["evidence"]

    def test_acute_onset_variant(self):
        state = _state_with_reps(
            _rep("headache", severity="severe", onset="acute"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" in _flag_names(flags)

    def test_no_match_without_severe(self):
        state = _state_with_reps(
            _rep("headache", severity="mild", onset="sudden"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" not in _flag_names(flags)

    def test_no_match_without_sudden_onset(self):
        state = _state_with_reps(
            _rep("headache", severity="severe", onset="gradual"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" not in _flag_names(flags)

    def test_no_match_without_headache(self):
        state = _state_with_reps(
            _rep("chest pain", severity="severe", onset="sudden"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" not in _flag_names(flags)

    def test_no_match_without_onset(self):
        state = _state_with_reps(
            _rep("headache", severity="severe"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" not in _flag_names(flags)

    def test_no_match_without_severity(self):
        state = _state_with_reps(
            _rep("headache", onset="sudden"),
        )
        flags = detect_red_flags(state)
        assert "sudden_severe_headache" not in _flag_names(flags)


# ══════════════════════════════════════════════════════════════════
# Chest pain with dyspnea
# ══════════════════════════════════════════════════════════════════


class TestChestPainWithDyspnea:
    def test_matches_with_dyspnea(self):
        state = _state_with_reps(
            _rep("chest pain"),
            _rep("dyspnea"),
        )
        flags = detect_red_flags(state)
        assert "chest_pain_with_dyspnea" in _flag_names(flags)
        f = [x for x in flags if x["flag"] == "chest_pain_with_dyspnea"][0]
        assert "chest pain" in f["evidence"]
        assert "dyspnea" in f["evidence"]

    def test_matches_with_shortness_of_breath(self):
        state = _state_with_reps(
            _rep("chest pain"),
            _rep("shortness of breath"),
        )
        flags = detect_red_flags(state)
        assert "chest_pain_with_dyspnea" in _flag_names(flags)

    def test_no_match_without_chest_pain(self):
        state = _state_with_reps(
            _rep("dyspnea"),
        )
        flags = detect_red_flags(state)
        assert "chest_pain_with_dyspnea" not in _flag_names(flags)

    def test_no_match_without_dyspnea(self):
        state = _state_with_reps(
            _rep("chest pain"),
        )
        flags = detect_red_flags(state)
        assert "chest_pain_with_dyspnea" not in _flag_names(flags)


# ══════════════════════════════════════════════════════════════════
# Hemoptysis
# ══════════════════════════════════════════════════════════════════


class TestHemoptysis:
    def test_matches(self):
        state = _state_with_reps(_rep("hemoptysis"))
        flags = detect_red_flags(state)
        assert "hemoptysis_flag" in _flag_names(flags)
        f = [x for x in flags if x["flag"] == "hemoptysis_flag"][0]
        assert f["severity"] == "high"
        assert "hemoptysis" in f["evidence"]

    def test_no_match_without_hemoptysis(self):
        state = _state_with_reps(_rep("cough"))
        flags = detect_red_flags(state)
        assert "hemoptysis_flag" not in _flag_names(flags)


# ══════════════════════════════════════════════════════════════════
# Suicidal ideation
# ══════════════════════════════════════════════════════════════════


class TestSuicidalIdeation:
    def test_matches(self):
        state = _state_with_reps(_rep("suicidal ideation"))
        flags = detect_red_flags(state)
        assert "suicidal_ideation_flag" in _flag_names(flags)
        f = [x for x in flags if x["flag"] == "suicidal_ideation_flag"][0]
        assert f["severity"] == "high"

    def test_no_match_without_ideation(self):
        state = _state_with_reps(_rep("depressed mood"))
        flags = detect_red_flags(state)
        assert "suicidal_ideation_flag" not in _flag_names(flags)


# ══════════════════════════════════════════════════════════════════
# Systemic malignancy pattern
# ══════════════════════════════════════════════════════════════════


class TestSystemicMalignancyPattern:
    def test_matches(self):
        state = _state_with_reps(
            _rep("weight loss"),
            _rep("night sweats"),
            _rep("lymphadenopathy"),
        )
        flags = detect_red_flags(state)
        assert "systemic_malignancy_pattern" in _flag_names(flags)
        f = [x for x in flags if x["flag"] == "systemic_malignancy_pattern"][0]
        assert "weight loss" in f["evidence"]
        assert "night sweats" in f["evidence"]
        assert "lymphadenopathy" in f["evidence"]

    def test_lymph_node_swelling_variant(self):
        state = _state_with_reps(
            _rep("weight loss"),
            _rep("night sweats"),
            _rep("lymph node swelling"),
        )
        flags = detect_red_flags(state)
        assert "systemic_malignancy_pattern" in _flag_names(flags)

    def test_no_match_without_weight_loss(self):
        state = _state_with_reps(
            _rep("night sweats"),
            _rep("lymphadenopathy"),
        )
        flags = detect_red_flags(state)
        assert "systemic_malignancy_pattern" not in _flag_names(flags)

    def test_no_match_without_night_sweats(self):
        state = _state_with_reps(
            _rep("weight loss"),
            _rep("lymphadenopathy"),
        )
        flags = detect_red_flags(state)
        assert "systemic_malignancy_pattern" not in _flag_names(flags)

    def test_no_match_without_lymph(self):
        state = _state_with_reps(
            _rep("weight loss"),
            _rep("night sweats"),
        )
        flags = detect_red_flags(state)
        assert "systemic_malignancy_pattern" not in _flag_names(flags)


# ══════════════════════════════════════════════════════════════════
# No duplicate flags
# ══════════════════════════════════════════════════════════════════


class TestNoDuplicates:
    def test_no_duplicates(self):
        state = _state_with_reps(
            _rep("hemoptysis"),
            _rep("hemoptysis"),
        )
        flags = detect_red_flags(state)
        names = _flag_names(flags)
        assert len(names) == len(set(names))


# ══════════════════════════════════════════════════════════════════
# Multiple flags can match
# ══════════════════════════════════════════════════════════════════


class TestMultipleFlags:
    def test_multiple_flags(self):
        state = _state_with_reps(
            _rep("headache", severity="severe", onset="sudden"),
            _rep("hemoptysis"),
        )
        flags = detect_red_flags(state)
        names = _flag_names(flags)
        assert "sudden_severe_headache" in names
        assert "hemoptysis_flag" in names


# ══════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_state(self):
        assert detect_red_flags({}) == []

    def test_empty_derived(self):
        assert detect_red_flags({"derived": {}}) == []

    def test_empty_reps(self):
        assert detect_red_flags(_state_with_reps()) == []

    def test_all_flags_have_required_keys(self):
        state = _state_with_reps(
            _rep("headache", severity="severe", onset="sudden"),
            _rep("hemoptysis"),
            _rep("suicidal ideation"),
        )
        flags = detect_red_flags(state)
        for f in flags:
            assert "flag" in f
            assert "label" in f
            assert "severity" in f
            assert "evidence" in f
            assert isinstance(f["evidence"], list)


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state_with_reps(
            _rep("headache", severity="severe", onset="sudden"),
            _rep("hemoptysis"),
        )
        r1 = detect_red_flags(state)
        r2 = detect_red_flags(state)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_red_flags_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "red_flags" in state["derived"]
        assert isinstance(state["derived"]["red_flags"], list)

    def test_empty_input(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["red_flags"] == []

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
        assert isinstance(state["derived"]["normalized_timeline"], list)
        assert isinstance(state["derived"]["temporal_context"], dict)

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_flags = list(state["derived"]["red_flags"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["red_flags"] == original_flags
        assert "ai_overlay" in state["derived"]
