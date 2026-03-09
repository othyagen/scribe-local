"""Deterministic running clinical summary generator.

Aggregates derived clinical state into a compact overview dict suitable
for live display.  No ML, no LLM, no external API calls.
"""

from __future__ import annotations


def build_running_summary(clinical_state: dict) -> dict:
    """Build a running clinical summary from structured clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with keys: ``core_problem``, ``additional_symptoms``,
        ``patterns_detected``, ``alerts``.
    """
    derived = clinical_state.get("derived", {})

    # Core problem from problem_summary (human-readable sentence)
    core_problem = derived.get("problem_summary", "")

    # Additional symptoms: all symptom representations except core
    problem_focus = derived.get("problem_focus")
    symptom_reps: list[dict] = derived.get("symptom_representations", [])
    additional_symptoms: list[str] = []
    for rep in symptom_reps:
        name = rep.get("symptom", "")
        if name and name.lower() != (problem_focus or "").lower():
            if name not in additional_symptoms:
                additional_symptoms.append(name)

    # Pattern labels from clinical_patterns
    patterns: list[dict] = derived.get("clinical_patterns", [])
    patterns_detected: list[str] = [
        p["label"] for p in patterns if "label" in p
    ]

    # Alerts: initially empty, extensible
    alerts: list = []

    return {
        "core_problem": core_problem,
        "additional_symptoms": additional_symptoms,
        "patterns_detected": patterns_detected,
        "alerts": alerts,
    }
