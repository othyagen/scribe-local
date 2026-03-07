"""Deterministic clinical pattern matcher.

Detects common clinical patterns from structured data using rule-based
exact matching.  Reads primarily from ``derived.symptom_representations``
to keep qualifiers linked to the correct symptom.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations


def match_clinical_patterns(clinical_state: dict) -> list[dict]:
    """Match clinical patterns against structured clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of pattern dicts, each with ``pattern``, ``label``, and
        ``evidence`` keys.  Empty list if nothing matches.
        No duplicates.
    """
    derived = clinical_state.get("derived", {})
    symptom_reps: list[dict] = derived.get("symptom_representations", [])

    # Build indexes from symptom representations
    sym_set, rep_index = _build_indexes(symptom_reps)

    matched: list[dict] = []
    seen: set[str] = set()

    for rule in _RULES:
        result = rule(sym_set, rep_index)
        if result is not None and result["pattern"] not in seen:
            seen.add(result["pattern"])
            matched.append(result)

    return matched


# ── indexes ───────────────────────────────────────────────────────


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


def _has_factor(rep: dict, factor_key: str, terms: set[str]) -> str | None:
    """Check if a symptom rep has a matching factor.  Returns the match."""
    for f in rep.get(factor_key, []):
        if f.lower() in terms:
            return f
    return None


# ── pattern rules ─────────────────────────────────────────────────


def _angina_like(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "chest pain" not in sym_set:
        return None
    rep = rep_index["chest pain"]

    aggravating_terms = {"exertion", "exercise", "walking", "physical activity"}
    relieving_terms = {"rest"}

    agg_match = _has_factor(rep, "aggravating_factors", aggravating_terms)
    rel_match = _has_factor(rep, "relieving_factors", relieving_terms)

    if agg_match is None or rel_match is None:
        return None

    return {
        "pattern": "angina_like",
        "label": "Angina-like pattern",
        "evidence": [
            "chest pain",
            f"aggravating factor: {agg_match}",
            f"relieving factor: {rel_match}",
        ],
    }


def _lower_respiratory(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    required = {"cough", "fever"}
    dyspnea_terms = {"shortness of breath", "dyspnea"}

    if not required.issubset(sym_set):
        return None

    dyspnea_match = dyspnea_terms & sym_set
    if not dyspnea_match:
        return None

    evidence = ["cough", "fever", sorted(dyspnea_match)[0]]
    return {
        "pattern": "lower_respiratory_pattern",
        "label": "Lower respiratory pattern",
        "evidence": evidence,
    }


def _migraine_like(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "headache" not in sym_set:
        return None
    if "nausea" not in sym_set:
        return None

    evidence: list[str] = ["headache", "nausea"]

    rep = rep_index.get("headache", {})
    severity = rep.get("severity")
    if severity and severity.lower() == "severe":
        evidence.append("severity: severe")

    return {
        "pattern": "migraine_like",
        "label": "Migraine-like pattern",
        "evidence": evidence,
    }


def _urinary_irritative(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    dysuria_terms = {"dysuria", "painful urination"}
    frequency_terms = {"frequency", "urinary frequency"}

    dysuria_match = dysuria_terms & sym_set
    frequency_match = frequency_terms & sym_set

    if not dysuria_match or not frequency_match:
        return None

    return {
        "pattern": "urinary_irritative_pattern",
        "label": "Urinary irritative pattern",
        "evidence": [sorted(dysuria_match)[0], sorted(frequency_match)[0]],
    }


def _gastroenteritis_like(
    sym_set: set[str],
    rep_index: dict[str, dict],
) -> dict | None:
    if "diarrhea" not in sym_set:
        return None

    nv_match = {"nausea", "vomiting"} & sym_set
    if not nv_match:
        return None

    evidence: list[str] = ["diarrhea", sorted(nv_match)[0]]

    if "fever" in sym_set:
        evidence.append("fever")

    return {
        "pattern": "gastroenteritis_like",
        "label": "Gastroenteritis-like pattern",
        "evidence": evidence,
    }


# Rule registry — order determines output order when multiple match.
_RULES = [
    _angina_like,
    _lower_respiratory,
    _migraine_like,
    _urinary_irritative,
    _gastroenteritis_like,
]
