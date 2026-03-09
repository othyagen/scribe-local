"""Output selector — centralized control of optional output layers.

Applies optional outputs (classification, FHIR export, AI overlay) based
on configuration.  The deterministic core pipeline always runs regardless
of these settings.

Each optional output is independently controllable and additive — it
never modifies extraction results or other derived fields.

No ML, no LLM (except when AI overlay is explicitly enabled).
"""

from __future__ import annotations

from typing import Any

from app.classification_router import apply_classification
from app.fhir_exporter import build_fhir_bundle


def apply_optional_outputs(
    clinical_state: dict,
    config: Any = None,
) -> dict:
    """Apply optional output layers based on configuration.

    Runs in order:
      1. Classification (if enabled)
      2. FHIR export (if enabled)
      3. AI overlay (if enabled)

    The AI overlay is prepared but not executed here — it requires
    async provider calls that are handled by the caller.  This function
    sets a flag so callers know whether to run it.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
            Must already contain the ``derived`` namespace from the
            deterministic core pipeline.
        config: an ``AppConfig`` instance, a plain dict, or ``None``.

    Returns:
        The (mutated) clinical_state with optional outputs added under
        ``derived``.
    """
    cls_enabled, cls_system = _resolve_classification(config)
    fhir_enabled = _resolve_fhir(config)

    derived = clinical_state.setdefault("derived", {})

    # 1. Classification
    if cls_enabled and cls_system != "none":
        classification = apply_classification(clinical_state, config)
        if classification:
            derived["classification"] = classification

    # 2. FHIR export
    if fhir_enabled:
        derived["fhir_bundle"] = build_fhir_bundle(clinical_state)

    return clinical_state


def should_run_ai_overlay(config: Any = None) -> bool:
    """Check whether AI overlay should be executed.

    Args:
        config: an ``AppConfig`` instance, a plain dict, or ``None``.

    Returns:
        ``True`` if AI overlay is enabled in the configuration.
    """
    return _resolve_ai(config)


# ── config resolution ────────────────────────────────────────────


def _resolve_classification(config: Any) -> tuple[bool, str]:
    """Extract (enabled, system) for classification."""
    if config is None:
        return False, "none"

    if isinstance(config, dict):
        cls = config.get("classification", {})
        if isinstance(cls, dict):
            return cls.get("enabled", False), cls.get("system", "none")
        return False, "none"

    cls_cfg = getattr(config, "classification", None)
    if cls_cfg is not None:
        return (
            getattr(cls_cfg, "enabled", False),
            getattr(cls_cfg, "system", "none"),
        )

    return False, "none"


def _resolve_fhir(config: Any) -> bool:
    """Extract fhir_enabled from config."""
    if config is None:
        return False

    if isinstance(config, dict):
        exp = config.get("export", {})
        if isinstance(exp, dict):
            return exp.get("fhir_enabled", False)
        return False

    exp_cfg = getattr(config, "export", None)
    if exp_cfg is not None:
        return getattr(exp_cfg, "fhir_enabled", False)

    return False


def _resolve_ai(config: Any) -> bool:
    """Extract ai.enabled from config."""
    if config is None:
        return False

    if isinstance(config, dict):
        ai = config.get("ai", {})
        if isinstance(ai, dict):
            return ai.get("enabled", False)
        return False

    ai_cfg = getattr(config, "ai", None)
    if ai_cfg is not None:
        return getattr(ai_cfg, "enabled", False)

    return False
