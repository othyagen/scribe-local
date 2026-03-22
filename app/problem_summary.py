"""Deterministic clinical problem summary generator.

Produces a concise human-readable sentence describing the primary
clinical problem.  Uses only the core symptom's own attributes —
additional symptoms are mentioned separately without inheriting
any qualifiers from the core symptom.

Pure function — no LLM, no I/O, no side effects.
"""

from __future__ import annotations


def summarize_problem(clinical_state: dict) -> str:
    """Build a one-sentence problem summary from clinical state.

    Uses ``derived.problem_focus`` to identify the core symptom, then
    reads that symptom's entry from ``derived.symptom_representations``
    to populate the sentence.  Additional symptoms are appended without
    qualifiers.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        A summary string, or ``""`` if there is no core symptom.
    """
    derived = clinical_state.get("derived", {})
    core = derived.get("problem_focus")
    if not core:
        return ""

    symptom_reps: list[dict] = derived.get("symptom_representations", [])

    # Find the core symptom's representation
    core_rep = _find_core_rep(core, symptom_reps)
    if core_rep is None:
        return ""

    parts: list[str] = []

    # 1. severity + character + core_symptom
    severity = core_rep.get("severity")
    character = core_rep.get("character")
    descriptors = [d for d in (severity, character) if d]
    if descriptors:
        parts.append(f"{' '.join(descriptors)} {core}")
    else:
        parts.append(core)

    # 2. duration
    duration = core_rep.get("duration")
    if duration:
        parts.append(f"for {duration}")

    # 3. onset
    onset = core_rep.get("onset")
    if onset:
        parts.append(f"{onset} onset")

    # 4. pattern
    pattern = core_rep.get("pattern")
    if pattern:
        parts.append(pattern)

    # 5. progression
    progression = core_rep.get("progression")
    if progression:
        parts.append(progression)

    # 6. laterality
    laterality = core_rep.get("laterality")
    if laterality:
        parts.append(f"{laterality} side")

    # 7. radiation
    radiation = core_rep.get("radiation")
    if radiation:
        parts.append(f"radiating {radiation}")

    # 8. aggravating_factors
    aggravating: list[str] = core_rep.get("aggravating_factors", [])
    if aggravating:
        parts.append(f"worse with {', '.join(aggravating)}")

    # 9. relieving_factors
    relieving: list[str] = core_rep.get("relieving_factors", [])
    if relieving:
        parts.append(f"relieved by {', '.join(relieving)}")

    # Build core sentence
    summary = ", ".join(parts)

    # Append additional symptoms (names only, no qualifiers)
    others = [
        r["symptom"] for r in symptom_reps
        if r["symptom"].lower() != core.lower()
    ]
    if others:
        summary += f", with additional {', '.join(others)}"

    # Capitalise first letter, end with period
    summary = summary[0].upper() + summary[1:]
    if not summary.endswith("."):
        summary += "."

    return summary


def _find_core_rep(
    core: str,
    symptom_reps: list[dict],
) -> dict | None:
    """Find the symptom representation matching *core* (case-insensitive)."""
    for rep in symptom_reps:
        if rep.get("symptom", "").lower() == core.lower():
            return rep
    return None
