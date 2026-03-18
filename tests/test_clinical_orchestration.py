"""Tests for clinical orchestration — output visibility and update strategy."""

from __future__ import annotations

import pytest

from app.clinical_orchestration import (
    orchestrate_outputs,
    should_apply_update,
    _resolve_config,
    _DEFAULT_CONFIG,
    _VALID_MODES,
    _VALID_STRATEGIES,
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


def _minimal_state():
    """A minimal clinical state with the keys orchestration reads."""
    return {
        "clinical_summary": {"key_findings": []},
        "summary_views": {"overview_summary": {}},
        "clinical_insights": {"missing_information": []},
        "next_questions": [{"question": "How long?", "reason": "test", "priority": "medium"}],
    }


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_returns_dict_with_expected_keys(self):
        result = orchestrate_outputs(_minimal_state())
        assert set(result.keys()) == {"mode", "visible_outputs", "update_behavior"}

    def test_visible_outputs_has_expected_keys(self):
        result = orchestrate_outputs(_minimal_state())
        assert set(result["visible_outputs"].keys()) == {
            "clinical_summary",
            "summary_views",
            "clinical_insights",
            "clinical_questions",
        }

    def test_update_behavior_has_strategy(self):
        result = orchestrate_outputs(_minimal_state())
        assert "update_strategy" in result["update_behavior"]


# ── default config ───────────────────────────────────────────────────


class TestDefaultConfig:
    def test_default_mode_is_scribe(self):
        result = orchestrate_outputs(_minimal_state())
        assert result["mode"] == "scribe"

    def test_default_shows_summary_views(self):
        result = orchestrate_outputs(_minimal_state())
        assert result["visible_outputs"]["summary_views"] is not None

    def test_default_shows_insights(self):
        result = orchestrate_outputs(_minimal_state())
        assert result["visible_outputs"]["clinical_insights"] is not None

    def test_default_hides_questions(self):
        result = orchestrate_outputs(_minimal_state())
        assert result["visible_outputs"]["clinical_questions"] is None

    def test_default_update_strategy_is_manual(self):
        result = orchestrate_outputs(_minimal_state())
        assert result["update_behavior"]["update_strategy"] == "manual"

    def test_clinical_summary_always_visible(self):
        result = orchestrate_outputs(_minimal_state())
        assert result["visible_outputs"]["clinical_summary"] is not None


# ── mode selection ───────────────────────────────────────────────────


class TestMode:
    def test_scribe_mode(self):
        result = orchestrate_outputs(_minimal_state(), {"mode": "scribe"})
        assert result["mode"] == "scribe"

    def test_assist_mode(self):
        result = orchestrate_outputs(_minimal_state(), {"mode": "assist"})
        assert result["mode"] == "assist"

    def test_invalid_mode_falls_back_to_default(self):
        result = orchestrate_outputs(_minimal_state(), {"mode": "invalid"})
        assert result["mode"] == "scribe"


# ── visibility toggles ──────────────────────────────────────────────


class TestVisibility:
    def test_show_questions_true(self):
        state = _minimal_state()
        result = orchestrate_outputs(state, {"show_questions": True})
        assert result["visible_outputs"]["clinical_questions"] is not None
        assert result["visible_outputs"]["clinical_questions"] == state["next_questions"]

    def test_show_questions_false(self):
        result = orchestrate_outputs(_minimal_state(), {"show_questions": False})
        assert result["visible_outputs"]["clinical_questions"] is None

    def test_hide_summary_views(self):
        result = orchestrate_outputs(_minimal_state(), {"show_summary_views": False})
        assert result["visible_outputs"]["summary_views"] is None

    def test_hide_insights(self):
        result = orchestrate_outputs(_minimal_state(), {"show_insights": False})
        assert result["visible_outputs"]["clinical_insights"] is None

    def test_show_all(self):
        result = orchestrate_outputs(_minimal_state(), {
            "show_summary_views": True,
            "show_insights": True,
            "show_questions": True,
        })
        vo = result["visible_outputs"]
        assert vo["clinical_summary"] is not None
        assert vo["summary_views"] is not None
        assert vo["clinical_insights"] is not None
        assert vo["clinical_questions"] is not None

    def test_hide_all_optional(self):
        result = orchestrate_outputs(_minimal_state(), {
            "show_summary_views": False,
            "show_insights": False,
            "show_questions": False,
        })
        vo = result["visible_outputs"]
        # clinical_summary always present
        assert vo["clinical_summary"] is not None
        assert vo["summary_views"] is None
        assert vo["clinical_insights"] is None
        assert vo["clinical_questions"] is None

    def test_missing_state_key_produces_none(self):
        """If state doesn't have a key, visible output is None even if enabled."""
        result = orchestrate_outputs({}, {
            "show_summary_views": True,
            "show_insights": True,
            "show_questions": True,
        })
        vo = result["visible_outputs"]
        assert vo["clinical_summary"] is None
        assert vo["summary_views"] is None
        assert vo["clinical_insights"] is None
        assert vo["clinical_questions"] is None


# ── update strategy ──────────────────────────────────────────────────


class TestUpdateStrategy:
    def test_manual_strategy(self):
        result = orchestrate_outputs(_minimal_state(), {"update_strategy": "manual"})
        assert result["update_behavior"]["update_strategy"] == "manual"

    def test_automatic_strategy(self):
        result = orchestrate_outputs(_minimal_state(), {"update_strategy": "automatic"})
        assert result["update_behavior"]["update_strategy"] == "automatic"

    def test_invalid_strategy_falls_back(self):
        result = orchestrate_outputs(_minimal_state(), {"update_strategy": "invalid"})
        assert result["update_behavior"]["update_strategy"] == "manual"


# ── resolve config ───────────────────────────────────────────────────


class TestResolveConfig:
    def test_none_returns_defaults(self):
        cfg = _resolve_config(None)
        assert cfg == _DEFAULT_CONFIG

    def test_partial_config_fills_defaults(self):
        cfg = _resolve_config({"mode": "assist"})
        assert cfg["mode"] == "assist"
        assert cfg["show_questions"] == _DEFAULT_CONFIG["show_questions"]

    def test_boolean_coercion(self):
        cfg = _resolve_config({"show_questions": 1})
        assert cfg["show_questions"] is True
        cfg2 = _resolve_config({"show_questions": 0})
        assert cfg2["show_questions"] is False

    def test_does_not_mutate_input(self):
        original = {"mode": "assist"}
        _resolve_config(original)
        assert original == {"mode": "assist"}


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_state(self):
        state = _minimal_state()
        original_summary = state["clinical_summary"]
        orchestrate_outputs(state, {"show_questions": True})
        assert state["clinical_summary"] is original_summary

    def test_deterministic(self):
        state = _minimal_state()
        cfg = {"mode": "assist", "show_questions": True}
        r1 = orchestrate_outputs(state, cfg)
        r2 = orchestrate_outputs(state, cfg)
        assert r1 == r2

    def test_visible_outputs_are_references_not_copies(self):
        """Orchestration selects, not copies — returns same objects."""
        state = _minimal_state()
        result = orchestrate_outputs(state, {"show_questions": True})
        assert result["visible_outputs"]["clinical_summary"] is state["clinical_summary"]
        assert result["visible_outputs"]["clinical_questions"] is state["next_questions"]


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_with_full_clinical_state_default(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        result = orchestrate_outputs(state)
        assert result["mode"] == "scribe"
        assert result["visible_outputs"]["clinical_summary"] is not None
        assert result["visible_outputs"]["clinical_questions"] is None

    def test_with_full_clinical_state_assist(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        result = orchestrate_outputs(state, {
            "mode": "assist",
            "show_questions": True,
        })
        assert result["mode"] == "assist"
        assert result["visible_outputs"]["clinical_questions"] is not None

    def test_empty_state(self):
        state = build_clinical_state([_seg("hello.")])
        result = orchestrate_outputs(state)
        assert result["mode"] == "scribe"
        assert result["visible_outputs"]["clinical_summary"] is not None

    def test_full_scenario(self):
        segments = [
            _seg("patient reports headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("prescribed ibuprofen.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ]
        state = build_clinical_state(segments)
        result = orchestrate_outputs(state, {
            "mode": "assist",
            "show_summary_views": True,
            "show_insights": True,
            "show_questions": True,
            "update_strategy": "automatic",
        })
        vo = result["visible_outputs"]
        assert vo["clinical_summary"] is not None
        assert vo["summary_views"] is not None
        assert vo["clinical_insights"] is not None
        assert vo["clinical_questions"] is not None
        assert result["update_behavior"]["update_strategy"] == "automatic"


# ── should_apply_update ──────────────────────────────────────────────


class TestShouldApplyUpdate:
    def test_automatic_returns_true(self):
        assert should_apply_update({"update_strategy": "automatic"}, "new_answers") is True

    def test_manual_returns_false(self):
        assert should_apply_update({"update_strategy": "manual"}, "new_answers") is False

    def test_default_config_returns_false(self):
        assert should_apply_update(None, "new_answers") is False

    def test_invalid_strategy_falls_back_to_manual(self):
        assert should_apply_update({"update_strategy": "invalid"}, "new_answers") is False

    def test_automatic_with_different_events(self):
        cfg = {"update_strategy": "automatic"}
        assert should_apply_update(cfg, "new_answers") is True
        assert should_apply_update(cfg, "session_end") is True
        assert should_apply_update(cfg, "manual_trigger") is True

    def test_manual_with_different_events(self):
        cfg = {"update_strategy": "manual"}
        assert should_apply_update(cfg, "new_answers") is False
        assert should_apply_update(cfg, "session_end") is False

    def test_does_not_mutate_config(self):
        cfg = {"update_strategy": "automatic"}
        original = dict(cfg)
        should_apply_update(cfg, "new_answers")
        assert cfg == original

    def test_deterministic(self):
        cfg = {"update_strategy": "automatic"}
        r1 = should_apply_update(cfg, "new_answers")
        r2 = should_apply_update(cfg, "new_answers")
        assert r1 == r2

    def test_partial_config(self):
        """Config with mode but no strategy uses default (manual)."""
        assert should_apply_update({"mode": "assist"}, "new_answers") is False

    def test_assist_mode_automatic(self):
        cfg = {"mode": "assist", "update_strategy": "automatic"}
        assert should_apply_update(cfg, "new_answers") is True
