"""Tests for clinical session — app-facing facade."""

from __future__ import annotations

import pytest

from app.clinical_session import (
    initialize_session,
    get_app_view,
    submit_answers,
    apply_manual_update,
)


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

_SESSION_KEYS = {
    "clinical_state", "segments", "config",
    "pending_observations", "speaker_roles", "confidence_entries",
}


# ── initialize_session ───────────────────────────────────────────────


class TestInitializeSession:
    def test_returns_expected_keys(self):
        session = initialize_session(_SEGMENTS)
        assert set(session.keys()) == _SESSION_KEYS

    def test_clinical_state_is_dict(self):
        session = initialize_session(_SEGMENTS)
        assert isinstance(session["clinical_state"], dict)
        assert "symptoms" in session["clinical_state"]

    def test_segments_preserved(self):
        session = initialize_session(_SEGMENTS)
        assert len(session["segments"]) == len(_SEGMENTS)

    def test_segments_are_copy(self):
        session = initialize_session(_SEGMENTS)
        assert session["segments"] is not _SEGMENTS

    def test_empty_pending(self):
        session = initialize_session(_SEGMENTS)
        assert session["pending_observations"] == []

    def test_config_stored(self):
        cfg = {"mode": "assist", "show_questions": True}
        session = initialize_session(_SEGMENTS, config=cfg)
        assert session["config"]["mode"] == "assist"

    def test_config_is_copy(self):
        cfg = {"mode": "assist"}
        session = initialize_session(_SEGMENTS, config=cfg)
        assert session["config"] is not cfg

    def test_default_config_is_empty_dict(self):
        session = initialize_session(_SEGMENTS)
        assert session["config"] == {}

    def test_speaker_roles_forwarded(self):
        roles = {"spk_0": {"role": "patient", "confidence": 0.8, "evidence": []}}
        session = initialize_session(_SEGMENTS, speaker_roles=roles)
        assert session["speaker_roles"] == roles
        assert session["clinical_state"]["speaker_roles"] == roles

    def test_confidence_entries_forwarded(self):
        conf = [{"seg_id": "seg_0001", "avg_logprob": -0.5}]
        session = initialize_session(_SEGMENTS, confidence_entries=conf)
        assert session["confidence_entries"] == conf

    def test_empty_segments(self):
        session = initialize_session([])
        assert session["clinical_state"]["symptoms"] == []


# ── get_app_view ─────────────────────────────────────────────────────


class TestGetAppView:
    def test_returns_expected_keys(self):
        session = initialize_session(_SEGMENTS)
        view = get_app_view(session)
        assert set(view.keys()) == {"orchestrated", "has_pending", "pending_count"}

    def test_no_pending_initially(self):
        session = initialize_session(_SEGMENTS)
        view = get_app_view(session)
        assert view["has_pending"] is False
        assert view["pending_count"] == 0

    def test_shows_pending_after_submit(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        view = get_app_view(session2)
        assert view["has_pending"] is True
        assert view["pending_count"] > 0

    def test_config_affects_orchestration(self):
        session = initialize_session(
            _SEGMENTS, config={"mode": "assist", "show_questions": True},
        )
        view = get_app_view(session)
        assert view["orchestrated"]["mode"] == "assist"
        assert view["orchestrated"]["visible_outputs"]["clinical_questions"] is not None

    def test_does_not_mutate_session(self):
        session = initialize_session(_SEGMENTS)
        original_keys = set(session.keys())
        get_app_view(session)
        assert set(session.keys()) == original_keys


# ── submit_answers (manual) ──────────────────────────────────────────


class TestSubmitAnswersManual:
    def test_returns_session_keys(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        assert _SESSION_KEYS.issubset(set(session2.keys()))

    def test_has_answer_result(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        assert "answer_result" in session2
        assert "new_observations" in session2["answer_result"]

    def test_pending_accumulated(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        assert len(session2["pending_observations"]) == 2

    def test_multiple_submits_accumulate(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, [_ANSWERS[0]])
        session3 = submit_answers(session2, [_ANSWERS[1]])
        assert len(session3["pending_observations"]) == 2

    def test_clinical_state_unchanged(self):
        session = initialize_session(_SEGMENTS)
        original_obs = len(session["clinical_state"]["observations"])
        session2 = submit_answers(session, _ANSWERS)
        # In manual mode, state is not rebuilt.
        assert len(session2["clinical_state"]["observations"]) == original_obs

    def test_does_not_mutate_input_session(self):
        session = initialize_session(_SEGMENTS)
        original_pending = len(session["pending_observations"])
        submit_answers(session, _ANSWERS)
        assert len(session["pending_observations"]) == original_pending

    def test_unparsed_in_result(self):
        session = initialize_session(_SEGMENTS)
        answers = [{"type": "unknown", "value": "foo"}]
        session2 = submit_answers(session, answers)
        assert len(session2["answer_result"]["unparsed_answers"]) == 1
        assert session2["pending_observations"] == []


# ── submit_answers (automatic) ───────────────────────────────────────


class TestSubmitAnswersAutomatic:
    def test_state_rebuilt_immediately(self):
        session = initialize_session(
            _SEGMENTS, config={"update_strategy": "automatic"},
        )
        original_obs = len(session["clinical_state"]["observations"])
        session2 = submit_answers(session, _ANSWERS)
        assert len(session2["clinical_state"]["observations"]) > original_obs

    def test_pending_cleared(self):
        session = initialize_session(
            _SEGMENTS, config={"update_strategy": "automatic"},
        )
        session2 = submit_answers(session, _ANSWERS)
        assert session2["pending_observations"] == []

    def test_answer_result_shows_should_update(self):
        session = initialize_session(
            _SEGMENTS, config={"update_strategy": "automatic"},
        )
        session2 = submit_answers(session, _ANSWERS)
        assert session2["answer_result"]["should_update"] is True


# ── apply_manual_update ──────────────────────────────────────────────


class TestApplyManualUpdate:
    def test_returns_session_keys(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        session3 = apply_manual_update(session2)
        assert _SESSION_KEYS.issubset(set(session3.keys()))

    def test_pending_drained(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        assert len(session2["pending_observations"]) > 0
        session3 = apply_manual_update(session2)
        assert session3["pending_observations"] == []

    def test_state_rebuilt(self):
        session = initialize_session(_SEGMENTS)
        original_obs = len(session["clinical_state"]["observations"])
        session2 = submit_answers(session, _ANSWERS)
        session3 = apply_manual_update(session2)
        assert len(session3["clinical_state"]["observations"]) > original_obs

    def test_noop_when_no_pending(self):
        session = initialize_session(_SEGMENTS)
        session2 = apply_manual_update(session)
        assert len(session2["clinical_state"]["observations"]) == \
            len(session["clinical_state"]["observations"])

    def test_does_not_mutate_input(self):
        session = initialize_session(_SEGMENTS)
        session2 = submit_answers(session, _ANSWERS)
        pending_count = len(session2["pending_observations"])
        apply_manual_update(session2)
        assert len(session2["pending_observations"]) == pending_count

    def test_speaker_roles_preserved(self):
        roles = {"spk_0": {"role": "patient", "confidence": 0.8, "evidence": []}}
        session = initialize_session(_SEGMENTS, speaker_roles=roles)
        session2 = submit_answers(session, _ANSWERS)
        session3 = apply_manual_update(session2)
        assert session3["speaker_roles"] == roles
        assert session3["clinical_state"]["speaker_roles"] == roles


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_initialize_deterministic(self):
        s1 = initialize_session(_SEGMENTS)
        s2 = initialize_session(_SEGMENTS)
        assert s1["clinical_state"]["symptoms"] == s2["clinical_state"]["symptoms"]
        assert s1["clinical_state"]["observations"] == s2["clinical_state"]["observations"]

    def test_submit_deterministic(self):
        session = initialize_session(_SEGMENTS)
        s1 = submit_answers(session, _ANSWERS)
        s2 = submit_answers(session, _ANSWERS)
        assert s1["pending_observations"] == s2["pending_observations"]
        assert s1["answer_result"] == s2["answer_result"]

    def test_get_app_view_deterministic(self):
        session = initialize_session(_SEGMENTS)
        v1 = get_app_view(session)
        v2 = get_app_view(session)
        assert v1 == v2


# ── end-to-end ───────────────────────────────────────────────────────


class TestEndToEnd:
    def test_full_manual_lifecycle(self):
        # 1. Initialize.
        session = initialize_session(_SEGMENTS)
        view = get_app_view(session)
        assert view["has_pending"] is False

        # 2. Submit answers.
        session = submit_answers(session, _ANSWERS)
        view = get_app_view(session)
        assert view["has_pending"] is True
        assert view["pending_count"] == 2

        # 3. Apply manual update.
        session = apply_manual_update(session)
        view = get_app_view(session)
        assert view["has_pending"] is False
        assert len(session["clinical_state"]["observations"]) > 0

    def test_full_automatic_lifecycle(self):
        # 1. Initialize with automatic mode.
        session = initialize_session(
            _SEGMENTS, config={"update_strategy": "automatic"},
        )
        original_obs = len(session["clinical_state"]["observations"])

        # 2. Submit answers — auto-applied.
        session = submit_answers(session, _ANSWERS)
        assert session["pending_observations"] == []
        assert len(session["clinical_state"]["observations"]) > original_obs

        # 3. View reflects updated state.
        view = get_app_view(session)
        assert view["has_pending"] is False

    def test_multiple_answer_rounds(self):
        session = initialize_session(_SEGMENTS)

        # Round 1.
        session = submit_answers(session, [_ANSWERS[0]])
        assert len(session["pending_observations"]) == 1

        # Round 2.
        session = submit_answers(session, [_ANSWERS[1]])
        assert len(session["pending_observations"]) == 2

        # Apply all.
        session = apply_manual_update(session)
        assert session["pending_observations"] == []
        assert "clinical_summary" in session["clinical_state"]
