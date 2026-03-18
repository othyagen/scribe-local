"""Tests for clinical flow — manual update flow orchestration."""

from __future__ import annotations

import pytest

from app.clinical_flow import (
    handle_answers,
    apply_pending_update,
    build_app_state,
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


_SEGMENTS = [
    _seg("patient has headache and nausea.", seg_id="seg_0001", t0=0.0, t1=3.0),
    _seg("prescribed ibuprofen.", seg_id="seg_0002", t0=3.0, t1=5.0),
]

_ANSWERS = [
    {"type": "duration", "value": "3 days", "related": "headache"},
    {"type": "allergy", "value": "penicillin"},
]


# ── handle_answers structure ─────────────────────────────────────────


class TestHandleAnswersStructure:
    def test_returns_expected_keys(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS)
        assert set(result.keys()) == {
            "new_observations", "unparsed_answers",
            "should_update", "pending_observations",
        }

    def test_new_observations_is_list(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS)
        assert isinstance(result["new_observations"], list)

    def test_unparsed_answers_is_list(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS)
        assert isinstance(result["unparsed_answers"], list)

    def test_should_update_is_bool(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS)
        assert isinstance(result["should_update"], bool)


# ── handle_answers manual mode ───────────────────────────────────────


class TestHandleAnswersManual:
    def test_manual_does_not_trigger_update(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS, {"update_strategy": "manual"})
        assert result["should_update"] is False

    def test_manual_populates_pending(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS, {"update_strategy": "manual"})
        assert len(result["pending_observations"]) == len(result["new_observations"])
        assert result["pending_observations"] == result["new_observations"]

    def test_default_is_manual(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS)
        assert result["should_update"] is False
        assert len(result["pending_observations"]) > 0


# ── handle_answers automatic mode ────────────────────────────────────


class TestHandleAnswersAutomatic:
    def test_automatic_triggers_update(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS, {"update_strategy": "automatic"})
        assert result["should_update"] is True

    def test_automatic_clears_pending(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS, {"update_strategy": "automatic"})
        assert result["pending_observations"] == []

    def test_automatic_still_returns_new_observations(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS, {"update_strategy": "automatic"})
        assert len(result["new_observations"]) == 2


# ── handle_answers edge cases ────────────────────────────────────────


class TestHandleAnswersEdgeCases:
    def test_empty_answers(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, [])
        assert result["new_observations"] == []
        assert result["unparsed_answers"] == []
        assert result["should_update"] is False
        assert result["pending_observations"] == []

    def test_all_unparsed(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, [{"type": "unknown", "value": "foo"}])
        assert result["new_observations"] == []
        assert len(result["unparsed_answers"]) == 1
        assert result["should_update"] is False

    def test_mixed_valid_and_invalid(self):
        state = build_clinical_state(_SEGMENTS)
        answers = [
            {"type": "duration", "value": "3 days", "related": "headache"},
            {"type": "unknown", "value": "foo"},
        ]
        result = handle_answers(state, answers)
        assert len(result["new_observations"]) == 1
        assert len(result["unparsed_answers"]) == 1

    def test_automatic_no_new_obs_does_not_trigger(self):
        """Even automatic mode doesn't trigger when nothing was parsed."""
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(
            state, [{"type": "unknown", "value": "x"}],
            {"update_strategy": "automatic"},
        )
        assert result["should_update"] is False


# ── apply_pending_update ─────────────────────────────────────────────


class TestApplyPendingUpdate:
    def test_returns_complete_state(self):
        state = build_clinical_state(_SEGMENTS)
        result = handle_answers(state, _ANSWERS)
        updated = apply_pending_update(
            state, result["pending_observations"], _SEGMENTS,
        )
        assert isinstance(updated, dict)
        assert set(updated.keys()) == set(state.keys())

    def test_observations_include_pending(self):
        state = build_clinical_state(_SEGMENTS)
        base_count = len(state["observations"])
        result = handle_answers(state, _ANSWERS)
        updated = apply_pending_update(
            state, result["pending_observations"], _SEGMENTS,
        )
        assert len(updated["observations"]) == base_count + len(result["pending_observations"])

    def test_empty_pending_is_noop(self):
        state = build_clinical_state(_SEGMENTS)
        updated = apply_pending_update(state, [], _SEGMENTS)
        assert len(updated["observations"]) == len(state["observations"])

    def test_does_not_mutate_state(self):
        state = build_clinical_state(_SEGMENTS)
        original_count = len(state["observations"])
        result = handle_answers(state, _ANSWERS)
        apply_pending_update(state, result["pending_observations"], _SEGMENTS)
        assert len(state["observations"]) == original_count

    def test_speaker_roles_forwarded(self):
        roles = {"spk_0": {"role": "patient", "confidence": 0.8, "evidence": []}}
        state = build_clinical_state(_SEGMENTS, speaker_roles=roles)
        result = handle_answers(state, _ANSWERS)
        updated = apply_pending_update(
            state, result["pending_observations"], _SEGMENTS,
            speaker_roles=roles,
        )
        assert updated["speaker_roles"] == roles


# ── build_app_state ──────────────────────────────────────────────────


class TestBuildAppState:
    def test_returns_expected_keys(self):
        state = build_clinical_state(_SEGMENTS)
        app = build_app_state(state)
        assert set(app.keys()) == {"orchestrated", "has_pending", "pending_count"}

    def test_orchestrated_is_dict(self):
        state = build_clinical_state(_SEGMENTS)
        app = build_app_state(state)
        assert isinstance(app["orchestrated"], dict)
        assert "mode" in app["orchestrated"]
        assert "visible_outputs" in app["orchestrated"]

    def test_no_pending_by_default(self):
        state = build_clinical_state(_SEGMENTS)
        app = build_app_state(state)
        assert app["has_pending"] is False
        assert app["pending_count"] == 0

    def test_with_pending_observations(self):
        state = build_clinical_state(_SEGMENTS)
        pending = [{"observation_id": "obs_9001"}]
        app = build_app_state(state, pending_observations=pending)
        assert app["has_pending"] is True
        assert app["pending_count"] == 1

    def test_config_forwarded_to_orchestration(self):
        state = build_clinical_state(_SEGMENTS)
        app = build_app_state(state, config={"mode": "assist", "show_questions": True})
        assert app["orchestrated"]["mode"] == "assist"
        assert app["orchestrated"]["visible_outputs"]["clinical_questions"] is not None

    def test_does_not_mutate_state(self):
        state = build_clinical_state(_SEGMENTS)
        original_keys = set(state.keys())
        build_app_state(state)
        assert set(state.keys()) == original_keys


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_handle_answers_does_not_mutate_state(self):
        state = build_clinical_state(_SEGMENTS)
        original_obs = len(state["observations"])
        handle_answers(state, _ANSWERS)
        assert len(state["observations"]) == original_obs

    def test_handle_answers_does_not_mutate_answers(self):
        answers = [dict(a) for a in _ANSWERS]
        originals = [dict(a) for a in answers]
        state = build_clinical_state(_SEGMENTS)
        handle_answers(state, answers)
        assert answers == originals

    def test_handle_answers_deterministic(self):
        state = build_clinical_state(_SEGMENTS)
        r1 = handle_answers(state, _ANSWERS)
        r2 = handle_answers(state, _ANSWERS)
        assert r1 == r2

    def test_build_app_state_deterministic(self):
        state = build_clinical_state(_SEGMENTS)
        r1 = build_app_state(state, config={"mode": "assist"})
        r2 = build_app_state(state, config={"mode": "assist"})
        assert r1 == r2


# ── end-to-end flow ──────────────────────────────────────────────────


class TestEndToEndFlow:
    def test_manual_flow(self):
        """Full manual flow: build → answer → pending → apply → rebuild."""
        segments = _SEGMENTS
        state = build_clinical_state(segments)

        # Step 1: handle answers (manual mode).
        result = handle_answers(state, _ANSWERS, {"update_strategy": "manual"})
        assert result["should_update"] is False
        assert len(result["pending_observations"]) == 2

        # Step 2: app state shows pending.
        app = build_app_state(state, pending_observations=result["pending_observations"])
        assert app["has_pending"] is True
        assert app["pending_count"] == 2

        # Step 3: user triggers manual apply.
        updated = apply_pending_update(
            state, result["pending_observations"], segments,
        )
        assert len(updated["observations"]) > len(state["observations"])

        # Step 4: app state after apply — no more pending.
        app2 = build_app_state(updated)
        assert app2["has_pending"] is False

    def test_automatic_flow(self):
        """Full automatic flow: build → answer → immediate apply."""
        segments = _SEGMENTS
        state = build_clinical_state(segments)

        # Step 1: handle answers (automatic mode).
        result = handle_answers(
            state, _ANSWERS, {"update_strategy": "automatic"},
        )
        assert result["should_update"] is True
        assert result["pending_observations"] == []

        # Step 2: caller applies immediately using new_observations.
        updated = apply_pending_update(
            state, result["new_observations"], segments,
        )
        assert len(updated["observations"]) > len(state["observations"])
        assert "clinical_summary" in updated
        assert "next_questions" in updated
