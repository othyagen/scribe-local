"""Deterministic structured problem representation.

Reads the assembled clinical_state and produces a formal problem
representation dict that identifies the primary clinical problem and
organises its attributes.  Pure function — no extraction logic, no
side effects, no I/O.
"""

from __future__ import annotations


def build_problem_representation(clinical_state: dict) -> dict:
    """Build a structured problem representation from clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with keys: ``core_symptom``, ``severity``, ``duration``,
        ``onset``, ``pattern``, ``progression``, ``laterality``,
        ``radiation``, ``associated_symptoms``, ``aggravating_factors``,
        ``relieving_factors``, ``pertinent_negatives``, ``timeline``,
        ``diagnostic_hints``.
    """
    symptoms: list[str] = clinical_state.get("symptoms", [])
    qualifiers: list[dict] = clinical_state.get("qualifiers", [])
    timeline: list[dict] = clinical_state.get("timeline", [])
    durations: list[str] = clinical_state.get("durations", [])
    negations: list[str] = clinical_state.get("negations", [])
    diagnostic_hints: list[dict] = clinical_state.get("diagnostic_hints", [])

    # ── core symptom selection ────────────────────────────────────
    core_symptom = _select_core_symptom(symptoms, timeline)

    # ── qualifier lookup for core symptom ─────────────────────────
    core_quals = _find_qualifiers(core_symptom, qualifiers)

    severity = core_quals.get("severity") if core_quals else None
    onset = core_quals.get("onset") if core_quals else None
    pattern = core_quals.get("pattern") if core_quals else None
    progression = core_quals.get("progression") if core_quals else None
    laterality = core_quals.get("laterality") if core_quals else None
    radiation = core_quals.get("radiation") if core_quals else None

    # ── duration ──────────────────────────────────────────────────
    duration = _select_duration(core_symptom, timeline, durations)

    # ── associated symptoms ───────────────────────────────────────
    associated = [s for s in symptoms if s != core_symptom] if core_symptom else []

    # ── aggravating / relieving factors ───────────────────────────
    aggravating = _collect_factors("aggravating_factors", core_symptom, qualifiers)
    relieving = _collect_factors("relieving_factors", core_symptom, qualifiers)

    return {
        "core_symptom": core_symptom,
        "severity": severity,
        "duration": duration,
        "onset": onset,
        "pattern": pattern,
        "progression": progression,
        "laterality": laterality,
        "radiation": radiation,
        "associated_symptoms": associated,
        "aggravating_factors": aggravating,
        "relieving_factors": relieving,
        "pertinent_negatives": list(negations),
        "timeline": list(timeline),
        "diagnostic_hints": [h["condition"] for h in diagnostic_hints],
    }


# ── helpers ───────────────────────────────────────────────────────


def _select_core_symptom(
    symptoms: list[str],
    timeline: list[dict],
) -> str | None:
    """Pick the primary symptom — earliest by timeline t_start, else first."""
    if not symptoms:
        return None

    # Try timeline entries with non-None t_start
    timed = [
        e for e in timeline
        if e.get("t_start") is not None
        and e.get("symptom") in symptoms
    ]
    if timed:
        earliest = min(timed, key=lambda e: e["t_start"])
        return earliest["symptom"]

    return symptoms[0]


def _find_qualifiers(
    core_symptom: str | None,
    qualifiers: list[dict],
) -> dict | None:
    """Find the qualifier entry matching *core_symptom*."""
    if core_symptom is None:
        return None
    for entry in qualifiers:
        if entry.get("symptom", "").lower() == core_symptom.lower():
            return entry.get("qualifiers", {})
    return None


def _select_duration(
    core_symptom: str | None,
    timeline: list[dict],
    durations: list[str],
) -> str | None:
    """Pick a duration string for the core symptom."""
    if core_symptom is not None:
        for entry in timeline:
            if (entry.get("symptom", "").lower() == core_symptom.lower()
                    and entry.get("time_expression")):
                return entry["time_expression"]
    return durations[0] if durations else None


def _collect_factors(
    factor_key: str,
    core_symptom: str | None,
    qualifiers: list[dict],
) -> list[str]:
    """Collect factors — prefer core symptom, fall back to all."""
    if not qualifiers:
        return []

    # Try core symptom first
    if core_symptom is not None:
        for entry in qualifiers:
            if entry.get("symptom", "").lower() == core_symptom.lower():
                factors = entry.get("qualifiers", {}).get(factor_key, [])
                if factors:
                    return list(factors)

    # Fall back: union across all qualifier entries, deduplicated
    seen: set[str] = set()
    result: list[str] = []
    for entry in qualifiers:
        for f in entry.get("qualifiers", {}).get(factor_key, []):
            key = f.lower()
            if key not in seen:
                seen.add(key)
                result.append(f)
    return result
