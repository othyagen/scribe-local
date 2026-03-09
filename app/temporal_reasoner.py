"""Deterministic temporal reasoning over normalized timeline data.

Derives clinical temporal context — onset ordering, progression events,
new-symptom detection, and uncertainty notes — from explicit temporal
evidence only.

**Does NOT infer clinical onset order from transcript mention order.**
Mention order and clinical onset order are separate concepts.  Only
normalized ISO dates allow safe comparison; durations and missing
expressions produce uncertainty.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

import re
from datetime import date


# ── helpers ──────────────────────────────────────────────────────

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso_date(value: str | None) -> date | None:
    """Parse a ``YYYY-MM-DD`` string.  Returns ``None`` on failure."""
    if not value or not _ISO_DATE_RE.match(value):
        return None
    try:
        parts = value.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def _is_duration(value: str | None) -> bool:
    """Return True if *value* looks like an ISO 8601 duration (P…)."""
    return bool(value and value.startswith("P"))


# ── public API ───────────────────────────────────────────────────


def derive_temporal_context(clinical_state: dict) -> dict:
    """Derive clinical temporal context from structured clinical state.

    Uses only explicit temporal evidence (normalized dates, progression
    qualifiers).  Never infers order from mention position.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with keys ``clinical_onset_order``, ``progression_events``,
        ``new_symptoms``, ``temporal_uncertainty``.
    """
    derived = clinical_state.get("derived", {})
    ntl: list[dict] = derived.get("normalized_timeline", [])
    symptom_reps: list[dict] = derived.get("symptom_representations", [])
    all_symptoms: list[str] = clinical_state.get("symptoms", [])

    # ── 1. Build symptom → earliest ISO date mapping ─────────
    symptom_dates: dict[str, date] = {}
    symptoms_with_duration_only: set[str] = set()

    for entry in ntl:
        sym = entry.get("symptom", "").lower()
        norm = entry.get("normalized_time")
        if not sym:
            continue

        dt = _parse_iso_date(norm)
        if dt is not None:
            if sym not in symptom_dates or dt < symptom_dates[sym]:
                symptom_dates[sym] = dt
        elif _is_duration(norm):
            if sym not in symptom_dates:
                symptoms_with_duration_only.add(sym)

    # ── 2. Clinical onset order (dates only) ─────────────────
    # Only symptoms with explicit ISO dates can be ordered.
    dated_pairs = sorted(symptom_dates.items(), key=lambda kv: kv[1])
    clinical_onset_order: list[str] = [sym for sym, _ in dated_pairs]

    # ── 3. Progression events ────────────────────────────────
    _PROGRESSION_VALUES = {"worsening", "improving", "stable"}
    progression_events: list[dict] = []
    for rep in symptom_reps:
        prog = rep.get("progression")
        if prog and prog.lower() in _PROGRESSION_VALUES:
            progression_events.append({
                "symptom": rep["symptom"],
                "progression": prog.lower(),
            })

    # ── 4. New symptoms ──────────────────────────────────────
    # A symptom is "new" if its earliest date is strictly after
    # at least one other symptom's earliest date.
    new_symptoms: list[str] = []
    if len(symptom_dates) >= 2:
        earliest_overall = min(symptom_dates.values())
        for sym, dt in dated_pairs:
            if dt > earliest_overall:
                new_symptoms.append(sym)

    # ── 5. Temporal uncertainty ──────────────────────────────
    temporal_uncertainty: list[str] = []

    # Symptoms with no normalized time at all
    dated_or_duration = set(symptom_dates.keys()) | symptoms_with_duration_only
    for sym in all_symptoms:
        if sym.lower() not in dated_or_duration:
            temporal_uncertainty.append(
                f"No temporal evidence for '{sym}'; onset unknown"
            )

    # Symptoms with duration only (no date)
    for sym in sorted(symptoms_with_duration_only):
        temporal_uncertainty.append(
            f"Only duration available for '{sym}'; "
            f"onset date cannot be determined"
        )

    return {
        "clinical_onset_order": clinical_onset_order,
        "progression_events": progression_events,
        "new_symptoms": new_symptoms,
        "temporal_uncertainty": temporal_uncertainty,
    }
