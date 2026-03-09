"""LLM overlay — optional AI-generated clinical text from structured state.

Reads the deterministic clinical_state and generates optional draft outputs
(SOAP notes, summaries, follow-up questions, problem representations)
using an external LLM provider.

Core rules:
  - The LLM overlay is fully optional.  If disabled, the system behaves
    exactly as before.
  - LLM output is an **overlay only** — it never modifies or replaces
    deterministic extraction results.
  - Prompts are loaded from files, never hardcoded.
  - Failures are logged and swallowed; the deterministic pipeline is
    never interrupted.

Provider abstraction:
  The :func:`generate_ai_overlay` entry point delegates to a provider
  function selected by ``config.ai.provider``.  Adding a new provider
  requires only registering a new callable in ``_PROVIDERS``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import AiConfig


# ── prompt loading ─────────────────────────────────────────────────

def load_prompt(name: str, prompts_dir: str = "prompts") -> str:
    """Load a prompt template from the prompts directory.

    Args:
        name: filename of the prompt (e.g. ``"soap_draft.txt"``).
        prompts_dir: directory containing prompt files.

    Returns:
        The prompt text.

    Raises:
        FileNotFoundError: if the prompt file does not exist.
    """
    path = Path(prompts_dir) / name
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, clinical_state: dict) -> str:
    """Substitute ``{{clinical_state}}`` in *template* with JSON state."""
    state_json = json.dumps(clinical_state, indent=2, ensure_ascii=False)
    return template.replace("{{clinical_state}}", state_json)


# ── provider abstraction ──────────────────────────────────────────

# Type alias for provider callables.
# Signature: (prompt: str, config: AiConfig) -> str
ProviderFn = Callable[[str, AiConfig], str]


def _call_openai(prompt: str, config: AiConfig) -> str:
    """Call the OpenAI-compatible API.

    Lazy-imports ``openai`` so the package is only required when AI is
    actually enabled.
    """
    import openai  # type: ignore[import-untyped]

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


# Provider registry — extend by adding entries here.
_PROVIDERS: dict[str, ProviderFn] = {
    "openai": _call_openai,
}


def get_provider(name: str) -> ProviderFn:
    """Look up a provider function by name.

    Raises ``ValueError`` for unknown providers.
    """
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown AI provider: {name!r}. "
            f"Available: {', '.join(sorted(_PROVIDERS))}"
        )
    return _PROVIDERS[name]


# ── overlay generation ────────────────────────────────────────────

# Mapping from overlay key to prompt config key.
_OVERLAY_KEYS: list[tuple[str, str]] = [
    ("soap_draft", "soap"),
    ("clinical_summary", "summary"),
    ("follow_up_questions", "follow_up"),
    ("problem_representation_refined", "problem_representation"),
]


def generate_ai_overlay(
    clinical_state: dict,
    config: AiConfig,
    provider_fn: Optional[ProviderFn] = None,
) -> dict:
    """Generate the AI overlay from structured clinical state.

    Args:
        clinical_state: the fully assembled deterministic state dict.
        config: AI configuration (provider, model, prompts, etc.).
        provider_fn: optional override for the LLM call function
            (useful for testing / dependency injection).

    Returns:
        dict with ``"ai_overlay"`` and ``"ai_overlay_meta"`` keys.
        If AI is disabled or all calls fail, the overlay values will
        be empty / ``None``.
    """
    result: dict[str, Any] = {
        "ai_overlay": {},
        "ai_overlay_meta": {
            "model": config.model,
            "provider": config.provider,
            "prompt_files": dict(config.prompts),
            "timestamp": time.time(),
        },
    }

    if not config.enabled:
        return result

    call_fn = provider_fn or get_provider(config.provider)

    for overlay_key, prompt_config_key in _OVERLAY_KEYS:
        prompt_file = config.prompts.get(prompt_config_key)
        if not prompt_file:
            continue

        try:
            template = load_prompt(prompt_file, config.prompts_dir)
            rendered = render_prompt(template, clinical_state)
            output = call_fn(rendered, config)
            result["ai_overlay"][overlay_key] = output
        except FileNotFoundError:
            print(
                f"WARNING: AI prompt file not found: {prompt_file}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"WARNING: AI overlay failed for {overlay_key}: {exc}",
                file=sys.stderr,
            )

    return result


def apply_ai_overlay(clinical_state: dict, overlay: dict) -> dict:
    """Merge AI overlay into clinical_state under ``derived.ai_overlay``.

    Does **not** modify deterministic keys — only adds the overlay
    and metadata under the ``derived`` namespace.

    Returns the (mutated) clinical_state for convenience.
    """
    derived = clinical_state.setdefault("derived", {})
    derived["ai_overlay"] = overlay.get("ai_overlay", {})
    derived["ai_overlay_meta"] = overlay.get("ai_overlay_meta", {})
    return clinical_state
