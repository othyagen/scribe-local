"""Classification system router for clinical coding suggestions.

Selects the appropriate classification system based on configuration
and applies it to the clinical state.  Currently supports ICPC-2 and
ICD-10; ICD-11 is reserved for future implementation.

Classification is an optional output layer — it never modifies
extraction results or other derived fields.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

from typing import Any

from app.icpc_mapper import suggest_icpc_codes
from app.icd_mapper import suggest_icd10_codes


def apply_classification(
    clinical_state: dict,
    config: Any = None,
) -> dict:
    """Apply the configured classification system to clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
        config: an ``AppConfig`` instance, a plain dict with a
            ``classification`` key, or ``None`` (disabled by default).

    Returns:
        Classification result dict with ``system`` and ``suggestions``
        keys, or an empty dict if classification is disabled.
    """
    enabled, system = _resolve_config(config)

    if not enabled or system == "none":
        return {}

    if system == "icpc":
        suggestions = suggest_icpc_codes(clinical_state)
        return {
            "system": "ICPC",
            "suggestions": suggestions,
        }

    if system == "icd10":
        suggestions = suggest_icd10_codes(clinical_state)
        return {
            "system": "ICD-10",
            "suggestions": suggestions,
        }

    # Future placeholder
    if system == "icd11":
        return {
            "system": "ICD-11",
            "suggestions": [],
        }

    # Unknown system — return empty
    return {}


def _resolve_config(config: Any) -> tuple[bool, str]:
    """Extract (enabled, system) from various config formats."""
    if config is None:
        return False, "none"

    # Support plain dict with "classification" key
    if isinstance(config, dict):
        cls = config.get("classification", {})
        if isinstance(cls, dict):
            return cls.get("enabled", False), cls.get("system", "none")
        return False, "none"

    # Support AppConfig with ClassificationConfig attribute
    cls_cfg = getattr(config, "classification", None)
    if cls_cfg is not None:
        return (
            getattr(cls_cfg, "enabled", False),
            getattr(cls_cfg, "system", "none"),
        )

    return False, "none"
