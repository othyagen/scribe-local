"""Deterministic red flag detection for clinical constellations.

Identifies high-risk clinical patterns from structured data using
rule-based exact matching against symptom representations.

Simple single-symptom flags are driven by
:mod:`app.clinical_terminology`; compound multi-symptom patterns
remain as explicit rule functions.

Additive only — does not modify any existing structured fields.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

from app.clinical_terminology import get_term as _get_term, is_red_flag as _is_red_flag


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

    # Compound / multi-symptom pattern rules
    for rule in _COMPOUND_RULES:
        result = rule(sym_set, rep_index)
        if result is not None and result["flag"] not in seen:
            seen.add(result["flag"])
            matched.append(result)

    # Simple single-symptom red flags from terminology registry
    for result in _terminology_red_flags(sym_set):
        if result["flag"] not in seen:
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


# ── terminology-driven simple flags ──────────────────────────────

# Terms that warrant "high" severity when present alone.
_HIGH_SEVERITY_TERMS: frozenset[str] = frozenset({"hemoptysis", "syncope"})


def _terminology_red_flags(sym_set: set[str]) -> list[dict]:
    """Generate simple red flag dicts for symptoms flagged by clinical_terminology."""
    results: list[dict] = []
    for sym in sorted(sym_set):  # sorted for determinism
        if not _is_red_flag(sym):
            continue
        term = _get_term(sym)
        label = term["display"] if term else sym.title()
        severity = "high" if sym in _HIGH_SEVERITY_TERMS else "moderate"
        results.append({
            "flag": sym.replace(" ", "_") + "_flag",
            "label": label,
            "severity": severity,
            "evidence": [sym],
        })
    return results


# ── compound rule functions ──────────────────────────────────────


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


def _suicidal_ideation_flag(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    # Not yet in clinical_terminology registry — kept as explicit rule.
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


# Compound rules — order determines output order when multiple match.
_COMPOUND_RULES = [
    _sudden_severe_headache,
    _chest_pain_with_dyspnea,
    _suicidal_ideation_flag,
    _systemic_malignancy_pattern,
]
