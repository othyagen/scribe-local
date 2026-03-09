"""Deterministic red flag detection for clinical constellations.

Identifies high-risk clinical patterns from structured data using
rule-based exact matching against symptom representations.

Additive only — does not modify any existing structured fields.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations


def detect_red_flags(clinical_state: dict) -> list[dict]:
    """Detect clinically important red flags from structured clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of red flag dicts, each with ``flag``, ``label``,
        ``severity``, and ``evidence`` keys.  No duplicates.
        Empty list if nothing matches.
    """
    derived = clinical_state.get("derived", {})
    symptom_reps: list[dict] = derived.get("symptom_representations", [])

    sym_set, rep_index = _build_indexes(symptom_reps)

    matched: list[dict] = []
    seen: set[str] = set()

    for rule in _RULES:
        result = rule(sym_set, rep_index)
        if result is not None and result["flag"] not in seen:
            seen.add(result["flag"])
            matched.append(result)

    return matched


# ── indexes ──────────────────────────────────────────────────────


def _build_indexes(
    symptom_reps: list[dict],
) -> tuple[set[str], dict[str, dict]]:
    """Build lowercase symptom set and rep lookup from representations."""
    sym_set: set[str] = set()
    rep_index: dict[str, dict] = {}
    for rep in symptom_reps:
        key = rep.get("symptom", "").lower()
        if key:
            sym_set.add(key)
            if key not in rep_index:
                rep_index[key] = rep
    return sym_set, rep_index


# ── rule functions ───────────────────────────────────────────────


def _sudden_severe_headache(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "headache" not in sym_set:
        return None
    rep = rep_index["headache"]

    severity = (rep.get("severity") or "").lower()
    onset = (rep.get("onset") or "").lower()

    if severity != "severe":
        return None
    if onset not in ("sudden", "acute"):
        return None

    evidence = ["headache", f"severity: {severity}", f"onset: {onset}"]
    return {
        "flag": "sudden_severe_headache",
        "label": "Sudden severe headache",
        "severity": "high",
        "evidence": evidence,
    }


def _chest_pain_with_dyspnea(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "chest pain" not in sym_set:
        return None

    dyspnea_terms = {"dyspnea", "shortness of breath"}
    dyspnea_match = dyspnea_terms & sym_set
    if not dyspnea_match:
        return None

    evidence = ["chest pain", sorted(dyspnea_match)[0]]
    return {
        "flag": "chest_pain_with_dyspnea",
        "label": "Chest pain with dyspnea",
        "severity": "high",
        "evidence": evidence,
    }


def _hemoptysis_flag(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "hemoptysis" not in sym_set:
        return None
    return {
        "flag": "hemoptysis_flag",
        "label": "Hemoptysis",
        "severity": "high",
        "evidence": ["hemoptysis"],
    }


def _suicidal_ideation_flag(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "suicidal ideation" not in sym_set:
        return None
    return {
        "flag": "suicidal_ideation_flag",
        "label": "Suicidal ideation",
        "severity": "high",
        "evidence": ["suicidal ideation"],
    }


def _systemic_malignancy_pattern(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "weight loss" not in sym_set:
        return None
    if "night sweats" not in sym_set:
        return None

    lymph_terms = {"lymphadenopathy", "lymph node swelling"}
    lymph_match = lymph_terms & sym_set
    if not lymph_match:
        return None

    evidence = ["weight loss", "night sweats", sorted(lymph_match)[0]]
    return {
        "flag": "systemic_malignancy_pattern",
        "label": "Systemic malignancy pattern",
        "severity": "high",
        "evidence": evidence,
    }


# Rule registry — order determines output order when multiple match.
_RULES = [
    _sudden_severe_headache,
    _chest_pain_with_dyspnea,
    _hemoptysis_flag,
    _suicidal_ideation_flag,
    _systemic_malignancy_pattern,
]
