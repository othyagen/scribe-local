"""Deterministic structured problem representation.

Reads the assembled clinical_state and produces a formal problem
representation dict that identifies the primary clinical problem and
organises its attributes.  Pure functions — no extraction logic, no
side effects, no I/O.

``build_symptom_representations`` produces per-symptom representations
so that qualifiers remain linked to their correct symptom and are never
incorrectly inherited across symptoms.
"""

from __future__ import annotations


def build_problem_representation(clinical_state: dict) -> dict:
    """Build a structured problem representation from clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with keys: ``core_symptom``, ``severity``, ``duration``,
        ``onset``, ``character``, ``pattern``, ``progression``,
        ``laterality``, ``radiation``, ``associated_symptoms``,
        ``aggravating_factors``, ``relieving_factors``,
        ``pertinent_negatives``, ``timeline``, ``diagnostic_hints``.
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
    character = core_quals.get("character") if core_quals else None
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
        "character": character,
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


def build_symptom_representations(clinical_state: dict) -> list[dict]:
    """Build per-symptom representations preserving qualifier–symptom links.

    Each symptom gets its own representation with only the qualifiers
    that were actually detected for *that* symptom.  No qualifier
    inheritance across symptoms.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of dicts, one per symptom, preserving symptom list order.
        Each dict has keys: ``symptom``, ``severity``, ``duration``,
        ``onset``, ``character``, ``pattern``, ``progression``,
        ``laterality``, ``radiation``, ``aggravating_factors``,
        ``relieving_factors``.
    """
    symptoms: list[str] = clinical_state.get("symptoms", [])
    qualifiers: list[dict] = clinical_state.get("qualifiers", [])
    timeline: list[dict] = clinical_state.get("timeline", [])

    # Index qualifiers by symptom (case-insensitive)
    qual_index: dict[str, dict] = {}
    for entry in qualifiers:
        key = entry.get("symptom", "").lower()
        if key and key not in qual_index:
            qual_index[key] = entry.get("qualifiers", {})

    # Index timeline time_expression by symptom (first match)
    time_index: dict[str, str] = {}
    for entry in timeline:
        sym = entry.get("symptom", "").lower()
        expr = entry.get("time_expression")
        if sym and expr and sym not in time_index:
            time_index[sym] = expr

    result: list[dict] = []
    for symptom in symptoms:
        key = symptom.lower()
        quals = qual_index.get(key, {})

        result.append({
            "symptom": symptom,
            "severity": quals.get("severity"),
            "duration": time_index.get(key),
            "onset": quals.get("onset"),
            "character": quals.get("character"),
            "pattern": quals.get("pattern"),
            "progression": quals.get("progression"),
            "laterality": quals.get("laterality"),
            "radiation": quals.get("radiation"),
            "aggravating_factors": list(quals.get("aggravating_factors", [])),
            "relieving_factors": list(quals.get("relieving_factors", [])),
        })

    return result


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


# ── problem narrative ─────────────────────────────────────────────


def build_problem_narrative(clinical_state: dict) -> dict:
    """Build a problem narrative with positive/negative features.

    Assembles positive features from core symptom qualifiers and
    negative features from negations into a structured narrative.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with keys: ``positive_features``, ``negative_features``,
        ``narrative``.
    """
    derived = clinical_state.get("derived", {})
    core = derived.get("problem_focus")
    negations: list[str] = clinical_state.get("negations", [])

    if not core:
        return {
            "positive_features": [],
            "negative_features": list(negations),
            "narrative": _build_narrative_text([], negations),
        }

    symptom_reps: list[dict] = derived.get("symptom_representations", [])
    structured_symptoms: list[dict] = derived.get("structured_symptoms", [])

    # Find core symptom representation
    core_rep = None
    for rep in symptom_reps:
        if rep.get("symptom", "").lower() == core.lower():
            core_rep = rep
            break

    positive = _build_positive_features(core, core_rep, structured_symptoms)
    negative = list(negations)

    return {
        "positive_features": positive,
        "negative_features": negative,
        "narrative": _build_narrative_text(positive, negative),
    }


def _build_positive_features(
    core: str,
    core_rep: dict | None,
    structured_symptoms: list[dict],
) -> list[str]:
    """Build human-readable positive feature strings."""
    features: list[str] = []

    if core_rep is None:
        features.append(core)
        return features

    # Severity
    severity = core_rep.get("severity")
    if severity:
        features.append(severity)

    # Character
    character = core_rep.get("character")
    if character:
        features.append(f"{character} character")

    # Core symptom name
    features.append(core)

    # Duration
    duration = core_rep.get("duration")
    if duration:
        features.append(f"for {duration}")

    # Onset
    onset = core_rep.get("onset")
    if onset:
        features.append(f"{onset} onset")

    # Pattern
    pattern = core_rep.get("pattern")
    if pattern:
        features.append(pattern)

    # Progression
    progression = core_rep.get("progression")
    if progression:
        features.append(progression)

    # Laterality
    laterality = core_rep.get("laterality")
    if laterality:
        features.append(f"{laterality} side")

    # Radiation
    radiation = core_rep.get("radiation")
    if radiation:
        features.append(f"radiating {radiation}")

    # Aggravating factors
    aggravating: list[str] = core_rep.get("aggravating_factors", [])
    if aggravating:
        features.append(f"worse with {', '.join(aggravating)}")

    # Relieving factors
    relieving: list[str] = core_rep.get("relieving_factors", [])
    if relieving:
        features.append(f"relieved by {', '.join(relieving)}")

    # Associated symptoms from structured_symptoms if available
    for ss in structured_symptoms:
        if ss.get("symptom", "").lower() != core.lower():
            associated = ss.get("context", {}).get("associated_present", [])
            if associated:
                for a in associated:
                    feat = f"with {a}"
                    if feat not in features:
                        features.append(feat)

    return features


def _build_narrative_text(
    positive: list[str],
    negative: list[str],
) -> str:
    """Assemble narrative sentence from features."""
    parts: list[str] = []

    if positive:
        parts.append(", ".join(positive))

    if negative:
        neg_text = ". " + ", ".join(negative)
        parts.append(neg_text)

    if not parts:
        return ""

    narrative = "".join(parts)

    # Capitalise first letter
    narrative = narrative[0].upper() + narrative[1:]

    # End with period
    if not narrative.endswith("."):
        narrative += "."

    return narrative
