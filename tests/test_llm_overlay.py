"""Tests for LLM overlay module."""

from __future__ import annotations

import json
import os
import time

import pytest

from app.config import AiConfig
from app.llm_overlay import (
    apply_ai_overlay,
    generate_ai_overlay,
    get_provider,
    load_prompt,
    render_prompt,
)
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "prompts")


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _sample_state() -> dict:
    return build_clinical_state([
        _seg("patient reports headache for 3 days.", seg_id="seg_0001"),
        _seg("no fever.", seg_id="seg_0002", t0=1.0, t1=2.0),
    ])


def _disabled_config(**overrides) -> AiConfig:
    defaults = dict(
        enabled=False,
        provider="openai",
        model="test-model",
        temperature=0.2,
        prompts={
            "soap": "soap/v1.txt",
            "summary": "summary/v1.txt",
            "follow_up": "follow_up/v1.txt",
            "problem_representation": "problem_representation/v1.txt",
        },
        prompts_dir=_PROMPTS_DIR,
    )
    defaults.update(overrides)
    return AiConfig(**defaults)


def _enabled_config(**overrides) -> AiConfig:
    return _disabled_config(enabled=True, **overrides)


def _mock_provider(prompt: str, config: AiConfig) -> str:
    """Deterministic mock LLM provider that echoes back a summary."""
    return f"[AI output for model={config.model}]"


def _failing_provider(prompt: str, config: AiConfig) -> str:
    """Mock provider that always raises."""
    raise RuntimeError("LLM service unavailable")


# ══════════════════════════════════════════════════════════════════
# Prompt loading
# ══════════════════════════════════════════════════════════════════


class TestPromptLoading:
    def test_load_soap_prompt(self):
        text = load_prompt("soap/v1.txt", _PROMPTS_DIR)
        assert "{{clinical_state}}" in text
        assert "SOAP" in text

    def test_load_clinical_summary_prompt(self):
        text = load_prompt("summary/v1.txt", _PROMPTS_DIR)
        assert "{{clinical_state}}" in text
        assert "summary" in text.lower()

    def test_load_follow_up_prompt(self):
        text = load_prompt("follow_up/v1.txt", _PROMPTS_DIR)
        assert "{{clinical_state}}" in text

    def test_load_problem_representation_prompt(self):
        text = load_prompt("problem_representation/v1.txt", _PROMPTS_DIR)
        assert "{{clinical_state}}" in text

    def test_missing_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent.txt", _PROMPTS_DIR)

    def test_prompts_not_hardcoded(self):
        """Each prompt is loaded from a file — verify content differs."""
        soap = load_prompt("soap/v1.txt", _PROMPTS_DIR)
        summary = load_prompt("summary/v1.txt", _PROMPTS_DIR)
        assert soap != summary

    def test_custom_prompts_dir(self, tmp_path):
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text("Custom prompt: {{clinical_state}}")
        text = load_prompt("custom.txt", str(tmp_path))
        assert "Custom prompt" in text


# ══════════════════════════════════════════════════════════════════
# Prompt rendering
# ══════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    def test_substitution(self):
        template = "State: {{clinical_state}}"
        state = {"symptoms": ["headache"]}
        rendered = render_prompt(template, state)
        assert '"headache"' in rendered
        assert "{{clinical_state}}" not in rendered

    def test_state_is_json(self):
        template = "Data: {{clinical_state}}"
        state = {"key": "value"}
        rendered = render_prompt(template, state)
        # Verify the inserted text is valid JSON
        json_part = rendered.replace("Data: ", "")
        parsed = json.loads(json_part)
        assert parsed == {"key": "value"}

    def test_no_placeholder_unchanged(self):
        template = "No placeholder here."
        rendered = render_prompt(template, {"a": 1})
        assert rendered == "No placeholder here."

    def test_empty_state(self):
        template = "State: {{clinical_state}}"
        rendered = render_prompt(template, {})
        assert "{}" in rendered


# ══════════════════════════════════════════════════════════════════
# Provider abstraction
# ══════════════════════════════════════════════════════════════════


class TestProviderAbstraction:
    def test_openai_provider_registered(self):
        fn = get_provider("openai")
        assert callable(fn)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_provider("unknown_provider")

    def test_custom_provider_fn(self):
        config = _enabled_config()
        result = _mock_provider("test", config)
        assert "AI output" in result


# ══════════════════════════════════════════════════════════════════
# AI disabled
# ══════════════════════════════════════════════════════════════════


class TestAiDisabled:
    def test_no_llm_calls_when_disabled(self):
        calls: list[str] = []

        def tracking_provider(prompt, config):
            calls.append(prompt)
            return "should not be called"

        config = _disabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=tracking_provider)
        assert calls == []

    def test_overlay_empty_when_disabled(self):
        config = _disabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=_mock_provider)
        assert result["ai_overlay"] == {}

    def test_meta_present_when_disabled(self):
        config = _disabled_config()
        result = generate_ai_overlay({}, config, provider_fn=_mock_provider)
        assert "ai_overlay_meta" in result
        assert result["ai_overlay_meta"]["model"] == "test-model"

    def test_pipeline_unchanged_when_disabled(self):
        state = _sample_state()
        original_symptoms = list(state["symptoms"])
        original_negations = list(state["negations"])
        original_pr = dict(state["derived"]["problem_representation"])

        config = _disabled_config()
        overlay = generate_ai_overlay(state, config, provider_fn=_mock_provider)
        apply_ai_overlay(state, overlay)

        assert state["symptoms"] == original_symptoms
        assert state["negations"] == original_negations
        assert state["derived"]["problem_representation"] == original_pr


# ══════════════════════════════════════════════════════════════════
# AI enabled
# ══════════════════════════════════════════════════════════════════


class TestAiEnabled:
    def test_overlay_populated(self):
        config = _enabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=_mock_provider)
        overlay = result["ai_overlay"]
        assert "soap_draft" in overlay
        assert "clinical_summary" in overlay
        assert "follow_up_questions" in overlay
        assert "problem_representation_refined" in overlay

    def test_overlay_values_from_provider(self):
        config = _enabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=_mock_provider)
        for key, value in result["ai_overlay"].items():
            assert "AI output" in value

    def test_meta_includes_model(self):
        config = _enabled_config(model="custom-model")
        result = generate_ai_overlay({}, config, provider_fn=_mock_provider)
        assert result["ai_overlay_meta"]["model"] == "custom-model"

    def test_meta_includes_provider(self):
        config = _enabled_config()
        result = generate_ai_overlay({}, config, provider_fn=_mock_provider)
        assert result["ai_overlay_meta"]["provider"] == "openai"

    def test_meta_includes_prompt_files(self):
        config = _enabled_config()
        result = generate_ai_overlay({}, config, provider_fn=_mock_provider)
        assert "soap" in result["ai_overlay_meta"]["prompt_files"]

    def test_meta_includes_timestamp(self):
        config = _enabled_config()
        before = time.time()
        result = generate_ai_overlay({}, config, provider_fn=_mock_provider)
        after = time.time()
        ts = result["ai_overlay_meta"]["timestamp"]
        assert before <= ts <= after

    def test_provider_receives_rendered_prompt(self):
        received: list[str] = []

        def capturing_provider(prompt, config):
            received.append(prompt)
            return "ok"

        config = _enabled_config()
        state = {"symptoms": ["headache"]}
        generate_ai_overlay(state, config, provider_fn=capturing_provider)
        # All prompts should have had clinical_state substituted
        for prompt in received:
            assert "headache" in prompt
            assert "{{clinical_state}}" not in prompt

    def test_subset_of_prompts(self):
        config = _enabled_config(prompts={"soap": "soap/v1.txt"})
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=_mock_provider)
        assert "soap_draft" in result["ai_overlay"]
        assert "clinical_summary" not in result["ai_overlay"]


# ══════════════════════════════════════════════════════════════════
# Failure handling
# ══════════════════════════════════════════════════════════════════


class TestFailureHandling:
    def test_llm_error_does_not_crash(self):
        config = _enabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=_failing_provider)
        # Should complete without raising
        assert isinstance(result, dict)

    def test_llm_error_leaves_overlay_empty(self):
        config = _enabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=_failing_provider)
        assert result["ai_overlay"] == {}

    def test_llm_error_preserves_meta(self):
        config = _enabled_config()
        result = generate_ai_overlay({}, config, provider_fn=_failing_provider)
        assert "ai_overlay_meta" in result
        assert result["ai_overlay_meta"]["model"] == "test-model"

    def test_missing_prompt_file_does_not_crash(self):
        config = _enabled_config(prompts={"soap": "nonexistent.txt"})
        result = generate_ai_overlay({}, config, provider_fn=_mock_provider)
        assert isinstance(result, dict)
        assert "soap_draft" not in result["ai_overlay"]

    def test_partial_failure(self):
        """One prompt fails, others succeed."""
        call_count = [0]

        def sometimes_failing(prompt, config):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("transient failure")
            return "success"

        config = _enabled_config()
        state = _sample_state()
        result = generate_ai_overlay(state, config, provider_fn=sometimes_failing)
        overlay = result["ai_overlay"]
        # At least some should succeed
        success_count = sum(1 for v in overlay.values() if v == "success")
        assert success_count >= 1
        # The failed one should be missing
        assert len(overlay) < 4


# ══════════════════════════════════════════════════════════════════
# apply_ai_overlay — nesting under derived
# ══════════════════════════════════════════════════════════════════


class TestApplyOverlay:
    def test_overlay_added_to_derived(self):
        state = _sample_state()
        overlay = {
            "ai_overlay": {"soap_draft": "Draft note."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["ai_overlay"]["soap_draft"] == "Draft note."
        assert state["derived"]["ai_overlay_meta"]["model"] == "test"

    def test_deterministic_keys_unchanged(self):
        state = _sample_state()
        original_symptoms = list(state["symptoms"])
        original_negations = list(state["negations"])
        original_durations = list(state["durations"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)

        assert state["symptoms"] == original_symptoms
        assert state["negations"] == original_negations
        assert state["durations"] == original_durations

    def test_empty_overlay(self):
        state = _sample_state()
        apply_ai_overlay(state, {"ai_overlay": {}, "ai_overlay_meta": {}})
        assert state["derived"]["ai_overlay"] == {}

    def test_overlay_replaces_previous(self):
        state = _sample_state()
        state["derived"] = {"ai_overlay": {"old": "data"}}
        apply_ai_overlay(state, {
            "ai_overlay": {"new": "data"},
            "ai_overlay_meta": {},
        })
        assert "old" not in state["derived"]["ai_overlay"]
        assert state["derived"]["ai_overlay"]["new"] == "data"


# ══════════════════════════════════════════════════════════════════
# Compatibility — deterministic outputs identical with AI on/off
# ══════════════════════════════════════════════════════════════════


class TestCompatibility:
    def test_deterministic_identical_ai_on_vs_off(self):
        segments = [
            _seg("severe headache for 3 days.", seg_id="seg_0001"),
            _seg("no fever, prescribed ibuprofen.", seg_id="seg_0002", t0=1.0, t1=2.0),
        ]

        state_off = build_clinical_state(segments)
        state_on = build_clinical_state(segments)

        # Apply overlay to state_on
        config = _enabled_config()
        overlay = generate_ai_overlay(state_on, config, provider_fn=_mock_provider)
        apply_ai_overlay(state_on, overlay)

        # All deterministic keys should be identical
        deterministic_keys = [
            "symptoms", "durations", "negations", "medications",
            "timeline", "review_flags", "diagnostic_hints",
            "speaker_roles", "history", "qualifiers",
        ]
        for key in deterministic_keys:
            assert state_on[key] == state_off[key], f"{key} differs with AI on"

    def test_ai_overlay_absent_without_overlay(self):
        state = _sample_state()
        # derived exists (problem_representation), but ai_overlay does not
        assert "ai_overlay" not in state.get("derived", {})

    def test_derived_present_after_overlay(self):
        state = _sample_state()
        overlay = generate_ai_overlay(
            state, _enabled_config(), provider_fn=_mock_provider,
        )
        apply_ai_overlay(state, overlay)
        assert "ai_overlay" in state["derived"]
        assert "ai_overlay_meta" in state["derived"]


# ══════════════════════════════════════════════════════════════════
# Config integration
# ══════════════════════════════════════════════════════════════════


class TestConfigIntegration:
    def test_default_ai_disabled(self):
        from app.config import AppConfig
        config = AppConfig()
        assert config.ai.enabled is False

    def test_ai_config_from_dict(self):
        from app.config import _build_ai
        ai = _build_ai({"enabled": True, "model": "custom"})
        assert ai.enabled is True
        assert ai.model == "custom"

    def test_ai_config_defaults(self):
        from app.config import _build_ai
        ai = _build_ai({})
        assert ai.enabled is False
        assert ai.provider == "openai"
        assert ai.model == "gpt-4.1-mini"
        assert ai.temperature == 0.2
        assert "soap" in ai.prompts

    def test_ai_config_prompt_paths(self):
        from app.config import _build_ai
        ai = _build_ai({})
        assert ai.prompts["soap"] == "soap/v1.txt"
        assert ai.prompts["summary"] == "summary/v1.txt"
        assert ai.prompts["follow_up"] == "follow_up/v1.txt"
        assert ai.prompts["problem_representation"] == "problem_representation/v1.txt"

    def test_cli_ai_flag(self):
        from app.config import AppConfig, apply_cli_overrides, build_arg_parser
        config = AppConfig()
        parser = build_arg_parser()
        args = parser.parse_args(["--ai"])
        apply_cli_overrides(config, args)
        assert config.ai.enabled is True

    def test_cli_no_ai_flag(self):
        from app.config import AppConfig, apply_cli_overrides, build_arg_parser
        config = AppConfig()
        config.ai.enabled = True
        parser = build_arg_parser()
        args = parser.parse_args(["--no-ai"])
        apply_cli_overrides(config, args)
        assert config.ai.enabled is False

    def test_custom_prompts_in_config(self):
        from app.config import _build_ai
        ai = _build_ai({
            "prompts": {"soap": "my_soap.txt"},
            "prompts_dir": "/custom/prompts",
        })
        assert ai.prompts == {"soap": "my_soap.txt"}
        assert ai.prompts_dir == "/custom/prompts"
