"""Tests for deterministic temporal reasoner."""

from __future__ import annotations

import pytest

from app.temporal_reasoner import derive_temporal_context
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


def _ntl_entry(symptom: str, normalized_time: str | None = None) -> dict:
    """Minimal normalized-timeline entry."""
    return {
        "symptom": symptom,
        "time_expression": normalized_time or "",
        "normalized_time": normalized_time,
        "seg_id": None,
        "speaker_id": None,
        "t_start": None,
    }


def _rep(symptom: str, progression: str | None = None, **kw) -> dict:
    """Minimal symptom representation."""
    return {
        "symptom": symptom,
        "severity": kw.get("severity"),
        "duration": kw.get("duration"),
        "onset": kw.get("onset"),
        "pattern": kw.get("pattern"),
        "progression": progression,
        "laterality": kw.get("laterality"),
        "radiation": kw.get("radiation"),
        "aggravating_factors": kw.get("aggravating_factors", []),
        "relieving_factors": kw.get("relieving_factors", []),
    }


def _state(
    symptoms: list[str] | None = None,
    ntl: list[dict] | None = None,
    reps: list[dict] | None = None,
) -> dict:
    """Build a minimal clinical state for temporal reasoning tests."""
    return {
        "symptoms": symptoms or [],
        "derived": {
            "normalized_timeline": ntl or [],
            "symptom_representations": reps or [],
        },
    }


# ══════════════════════════════════════════════════════════════════
# Clinical onset order from explicit dates
# ══════════════════════════════════════════════════════════════════


class TestClinicalOnsetOrder:
    def test_ordered_by_date(self):
        state = _state(
            symptoms=["headache", "fever"],
            ntl=[
                _ntl_entry("headache", "2026-03-05"),
                _ntl_entry("fever", "2026-03-07"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == ["headache", "fever"]

    def test_reverse_order(self):
        state = _state(
            symptoms=["fever", "headache"],
            ntl=[
                _ntl_entry("fever", "2026-03-07"),
                _ntl_entry("headache", "2026-03-03"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == ["headache", "fever"]

    def test_same_date_both_included(self):
        state = _state(
            symptoms=["headache", "fever"],
            ntl=[
                _ntl_entry("headache", "2026-03-05"),
                _ntl_entry("fever", "2026-03-05"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert set(ctx["clinical_onset_order"]) == {"headache", "fever"}

    def test_earliest_date_wins_for_same_symptom(self):
        state = _state(
            symptoms=["headache"],
            ntl=[
                _ntl_entry("headache", "2026-03-07"),
                _ntl_entry("headache", "2026-03-03"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == ["headache"]

    def test_empty_when_no_dates(self):
        state = _state(symptoms=["headache", "fever"])
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == []


# ══════════════════════════════════════════════════════════════════
# No onset order from mention order alone
# ══════════════════════════════════════════════════════════════════


class TestNoMentionOrderInference:
    def test_no_order_from_mention_position(self):
        """Symptoms with no dates must NOT be ordered by mention position."""
        state = _state(
            symptoms=["headache", "fever", "cough"],
            ntl=[
                _ntl_entry("headache", None),
                _ntl_entry("fever", None),
                _ntl_entry("cough", None),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == []

    def test_no_order_from_duration_only(self):
        """Duration-only symptoms cannot be date-ordered."""
        state = _state(
            symptoms=["headache", "fever"],
            ntl=[
                _ntl_entry("headache", "P3D"),
                _ntl_entry("fever", "P1D"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == []

    def test_partial_dates_only_dated_ordered(self):
        """Only dated symptoms appear in onset order."""
        state = _state(
            symptoms=["headache", "fever", "cough"],
            ntl=[
                _ntl_entry("headache", "2026-03-05"),
                _ntl_entry("fever", None),
                _ntl_entry("cough", "P3D"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == ["headache"]


# ══════════════════════════════════════════════════════════════════
# Progression events
# ══════════════════════════════════════════════════════════════════


class TestProgressionEvents:
    def test_worsening(self):
        state = _state(
            symptoms=["headache"],
            reps=[_rep("headache", progression="worsening")],
        )
        ctx = derive_temporal_context(state)
        assert len(ctx["progression_events"]) == 1
        assert ctx["progression_events"][0]["symptom"] == "headache"
        assert ctx["progression_events"][0]["progression"] == "worsening"

    def test_improving(self):
        state = _state(
            symptoms=["headache"],
            reps=[_rep("headache", progression="improving")],
        )
        ctx = derive_temporal_context(state)
        assert len(ctx["progression_events"]) == 1
        assert ctx["progression_events"][0]["progression"] == "improving"

    def test_stable(self):
        state = _state(
            symptoms=["headache"],
            reps=[_rep("headache", progression="stable")],
        )
        ctx = derive_temporal_context(state)
        assert len(ctx["progression_events"]) == 1
        assert ctx["progression_events"][0]["progression"] == "stable"

    def test_no_progression(self):
        state = _state(
            symptoms=["headache"],
            reps=[_rep("headache")],
        )
        ctx = derive_temporal_context(state)
        assert ctx["progression_events"] == []

    def test_multiple_symptoms_with_progression(self):
        state = _state(
            symptoms=["headache", "fever"],
            reps=[
                _rep("headache", progression="worsening"),
                _rep("fever", progression="improving"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert len(ctx["progression_events"]) == 2

    def test_ignores_unknown_progression_values(self):
        state = _state(
            symptoms=["headache"],
            reps=[_rep("headache", progression="fluctuating")],
        )
        ctx = derive_temporal_context(state)
        assert ctx["progression_events"] == []

    def test_case_insensitive_progression(self):
        state = _state(
            symptoms=["headache"],
            reps=[_rep("headache", progression="Worsening")],
        )
        ctx = derive_temporal_context(state)
        assert len(ctx["progression_events"]) == 1
        assert ctx["progression_events"][0]["progression"] == "worsening"


# ══════════════════════════════════════════════════════════════════
# New symptom detection
# ══════════════════════════════════════════════════════════════════


class TestNewSymptoms:
    def test_later_onset_is_new(self):
        state = _state(
            symptoms=["headache", "fever"],
            ntl=[
                _ntl_entry("headache", "2026-03-03"),
                _ntl_entry("fever", "2026-03-07"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["new_symptoms"] == ["fever"]

    def test_same_date_not_new(self):
        state = _state(
            symptoms=["headache", "fever"],
            ntl=[
                _ntl_entry("headache", "2026-03-05"),
                _ntl_entry("fever", "2026-03-05"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["new_symptoms"] == []

    def test_single_symptom_not_new(self):
        state = _state(
            symptoms=["headache"],
            ntl=[_ntl_entry("headache", "2026-03-05")],
        )
        ctx = derive_temporal_context(state)
        assert ctx["new_symptoms"] == []

    def test_no_dates_no_new(self):
        state = _state(symptoms=["headache", "fever"])
        ctx = derive_temporal_context(state)
        assert ctx["new_symptoms"] == []

    def test_multiple_new_symptoms(self):
        state = _state(
            symptoms=["headache", "fever", "cough"],
            ntl=[
                _ntl_entry("headache", "2026-03-01"),
                _ntl_entry("fever", "2026-03-05"),
                _ntl_entry("cough", "2026-03-07"),
            ],
        )
        ctx = derive_temporal_context(state)
        assert ctx["new_symptoms"] == ["fever", "cough"]


# ══════════════════════════════════════════════════════════════════
# Temporal uncertainty
# ══════════════════════════════════════════════════════════════════


class TestTemporalUncertainty:
    def test_symptom_without_any_temporal(self):
        state = _state(symptoms=["headache"])
        ctx = derive_temporal_context(state)
        assert len(ctx["temporal_uncertainty"]) >= 1
        assert any("headache" in u for u in ctx["temporal_uncertainty"])
        assert any("onset unknown" in u for u in ctx["temporal_uncertainty"])

    def test_symptom_with_duration_only(self):
        state = _state(
            symptoms=["headache"],
            ntl=[_ntl_entry("headache", "P3D")],
        )
        ctx = derive_temporal_context(state)
        assert any(
            "duration" in u.lower() and "headache" in u
            for u in ctx["temporal_uncertainty"]
        )

    def test_no_uncertainty_when_date_present(self):
        state = _state(
            symptoms=["headache"],
            ntl=[_ntl_entry("headache", "2026-03-05")],
        )
        ctx = derive_temporal_context(state)
        assert not any("headache" in u for u in ctx["temporal_uncertainty"])

    def test_no_symptoms_no_uncertainty(self):
        state = _state()
        ctx = derive_temporal_context(state)
        assert ctx["temporal_uncertainty"] == []

    def test_mixed_dated_and_undated(self):
        state = _state(
            symptoms=["headache", "fever"],
            ntl=[_ntl_entry("headache", "2026-03-05")],
        )
        ctx = derive_temporal_context(state)
        assert any("fever" in u for u in ctx["temporal_uncertainty"])
        assert not any("headache" in u for u in ctx["temporal_uncertainty"])


# ══════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_state(self):
        ctx = derive_temporal_context({})
        assert ctx == {
            "clinical_onset_order": [],
            "progression_events": [],
            "new_symptoms": [],
            "temporal_uncertainty": [],
        }

    def test_all_keys_present(self):
        ctx = derive_temporal_context({})
        assert set(ctx.keys()) == {
            "clinical_onset_order", "progression_events",
            "new_symptoms", "temporal_uncertainty",
        }

    def test_invalid_date_ignored(self):
        state = _state(
            symptoms=["headache"],
            ntl=[_ntl_entry("headache", "not-a-date")],
        )
        ctx = derive_temporal_context(state)
        assert ctx["clinical_onset_order"] == []


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state(
            symptoms=["headache", "fever", "cough"],
            ntl=[
                _ntl_entry("headache", "2026-03-03"),
                _ntl_entry("fever", "2026-03-07"),
                _ntl_entry("cough", None),
            ],
            reps=[
                _rep("headache", progression="worsening"),
                _rep("fever"),
                _rep("cough"),
            ],
        )
        r1 = derive_temporal_context(state)
        r2 = derive_temporal_context(state)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_temporal_context_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "temporal_context" in state["derived"]
        ctx = state["derived"]["temporal_context"]
        assert isinstance(ctx, dict)

    def test_temporal_context_has_all_keys(self):
        state = build_clinical_state([_seg("patient has headache.")])
        ctx = state["derived"]["temporal_context"]
        assert set(ctx.keys()) == {
            "clinical_onset_order", "progression_events",
            "new_symptoms", "temporal_uncertainty",
        }

    def test_empty_input(self):
        state = build_clinical_state([_seg("hello.")])
        ctx = state["derived"]["temporal_context"]
        assert ctx["clinical_onset_order"] == []
        assert ctx["progression_events"] == []
        assert ctx["new_symptoms"] == []

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

    def test_does_not_modify_timeline(self):
        state = build_clinical_state([
            _seg("patient has headache for 3 days."),
        ])
        for entry in state["timeline"]:
            assert "normalized_time" not in entry

    def test_does_not_modify_normalized_timeline(self):
        state = build_clinical_state([
            _seg("patient has headache for 3 days."),
        ])
        ntl = state["derived"]["normalized_timeline"]
        for entry in ntl:
            assert "clinical_onset_order" not in entry

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_ctx = dict(state["derived"]["temporal_context"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["temporal_context"] == original_ctx
        assert "ai_overlay" in state["derived"]
