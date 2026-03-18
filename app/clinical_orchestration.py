"""Clinical orchestration — output visibility and update strategy.

Selects, filters, and organises already-computed clinical outputs
based on a simple configuration dict.  Does not recompute, mutate,
or duplicate logic from any upstream layer.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


# ── defaults ─────────────────────────────────────────────────────────

_VALID_MODES = frozenset({"scribe", "assist"})
_VALID_STRATEGIES = frozenset({"manual", "automatic"})

_DEFAULT_CONFIG: dict = {
    "mode": "scribe",
    "show_summary_views": True,
    "show_insights": True,
    "show_questions": False,
    "update_strategy": "manual",
}


def orchestrate_outputs(
    clinical_state: dict,
    config: dict | None = None,
) -> dict:
    """Select visible outputs and update behaviour from clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
        config: orchestration configuration.  Missing keys fall back to
            :data:`_DEFAULT_CONFIG`.

    Returns:
        dict with ``mode``, ``visible_outputs``, ``update_behavior``.
    """
    cfg = _resolve_config(config)

    mode = cfg["mode"]
    show_summary_views = cfg["show_summary_views"]
    show_insights = cfg["show_insights"]
    show_questions = cfg["show_questions"]
    update_strategy = cfg["update_strategy"]

    visible: dict = {
        "clinical_summary": clinical_state.get("clinical_summary"),
        "summary_views": (
            clinical_state.get("summary_views") if show_summary_views else None
        ),
        "clinical_insights": (
            clinical_state.get("clinical_insights") if show_insights else None
        ),
        "clinical_questions": (
            clinical_state.get("next_questions") if show_questions else None
        ),
    }

    return {
        "mode": mode,
        "visible_outputs": visible,
        "update_behavior": {
            "update_strategy": update_strategy,
        },
    }


def _resolve_config(config: dict | None) -> dict:
    """Merge caller config with defaults, validating enum fields."""
    if config is None:
        return dict(_DEFAULT_CONFIG)

    resolved = dict(_DEFAULT_CONFIG)
    resolved.update(config)

    # Clamp invalid values to defaults.
    if resolved["mode"] not in _VALID_MODES:
        resolved["mode"] = _DEFAULT_CONFIG["mode"]
    if resolved["update_strategy"] not in _VALID_STRATEGIES:
        resolved["update_strategy"] = _DEFAULT_CONFIG["update_strategy"]

    # Booleans: coerce truthy/falsy.
    for key in ("show_summary_views", "show_insights", "show_questions"):
        resolved[key] = bool(resolved[key])

    return resolved
