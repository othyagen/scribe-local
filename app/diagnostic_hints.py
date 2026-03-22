"""Deterministic diagnostic hints from extracted symptoms.

Pure rule-based matching against a curated rule table.
No ML, no LLM, no external APIs.

Each rule requires ALL listed symptoms to be present and NONE
to be negated.  Returns matched conditions with SNOMED CT codes.
"""

from __future__ import annotations

import re

# ── rule table ───────────────────────────────────────────────────────

RULES: list[dict] = [
    {
        "symptoms": ["fever", "cough", "dyspnea"],
        "condition": "Pneumonia",
        "snomed": "233604007",
        "min_required": 2,
    },
    {
        "symptoms": ["headache", "neck stiffness", "fever"],
        "condition": "Meningitis",
        "snomed": "7180009",
    },
    {
        "symptoms": ["chest pain", "dyspnea", "swelling"],
        "condition": "Pulmonary embolism",
        "snomed": "59282003",
    },
    {
        "symptoms": ["chest pain"],
        "condition": "Pericarditis",
        "snomed": "3238004",
        "qualifiers": {
            "chest pain": {
                "character": ["sharp"],
                "aggravating_factors": ["deep breathing"],
                "relieving_factors": ["leaning forward"],
            },
        },
    },
    {
        "symptoms": ["chest pain", "dyspnea"],
        "condition": "Acute coronary syndrome",
        "snomed": "394659003",
    },
    {
        "symptoms": ["headache", "nausea", "vomiting"],
        "condition": "Migraine",
        "snomed": "37796009",
    },
    {
        "symptoms": ["fever", "sore throat"],
        "condition": "Pharyngitis",
        "snomed": "405737000",
    },
    {
        "symptoms": ["cough", "fever", "fatigue"],
        "condition": "Upper respiratory tract infection",
        "snomed": "54150009",
    },
    {
        "symptoms": ["diarrhea", "nausea", "vomiting"],
        "condition": "Gastroenteritis",
        "snomed": "25374005",
    },
    {
        "symptoms": ["swelling", "pain", "stiffness"],
        "condition": "Arthritis",
        "snomed": "3723001",
    },
    {
        "symptoms": ["dizziness", "nausea"],
        "condition": "Vertigo",
        "snomed": "399153001",
    },
    {
        "symptoms": ["anxiety", "palpitations", "insomnia"],
        "condition": "Generalised anxiety disorder",
        "snomed": "21897009",
    },
    {
        "symptoms": ["dysuria", "urinary frequency"],
        "condition": "Urinary tract infection",
        "snomed": "68566005",
    },
]

# Pre-compile patterns to detect negated symptoms from negation strings.
# Negation strings look like "No fever", "Denies chest pain", etc.
_NEGATION_PREFIXES = re.compile(
    r"^(no|not|denies|denied|without|absent|negative for|rules? out|ruled out)\s+",
    re.IGNORECASE,
)


def _extract_negated_symptoms(negations: list[str]) -> set[str]:
    """Extract the symptom portion from negation strings.

    E.g. "No fever" → "fever", "Denies chest pain" → "chest pain".
    Returns a set of lowercase symptom strings.
    """
    result: set[str] = set()
    for neg in negations:
        m = _NEGATION_PREFIXES.match(neg)
        if m:
            symptom = neg[m.end():].strip().rstrip(".,;!?").lower()
            if symptom:
                result.add(symptom)
    return result


def generate_diagnostic_hints(
    symptoms: list[str],
    negations: list[str] | None = None,
    qualifiers: list[dict] | None = None,
) -> list[dict]:
    """Generate diagnostic hints from symptom and negation lists.

    Args:
        symptoms: list of symptom strings (e.g. from extract_symptoms)
        negations: optional list of negation strings (e.g. from extract_negations)
        qualifiers: optional list of qualifier dicts from
            :func:`app.qualifier_extraction.extract_qualifiers`.

    Returns:
        list of dicts with ``condition``, ``snomed_code``, ``evidence``.
        Sorted by number of evidence symptoms (descending), then alphabetically.
    """
    symptom_set = {s.lower() for s in symptoms}

    negated = set()
    if negations:
        negated = _extract_negated_symptoms(negations)

    # Remove negated symptoms from the active set
    active_symptoms = symptom_set - negated

    # Index qualifiers by symptom name for fast lookup.
    qual_index: dict[str, dict] = {}
    for q in qualifiers or []:
        key = q.get("symptom", "").lower()
        if key:
            qual_index[key] = q.get("qualifiers", {})

    results: list[dict] = []
    for rule in RULES:
        required = {s.lower() for s in rule["symptoms"]}
        matched = required & active_symptoms
        min_req = rule.get("min_required", len(required))
        if len(matched) < min_req:
            continue

        # Check qualifier constraints if the rule defines them.
        rule_quals = rule.get("qualifiers")
        if rule_quals and not _check_qualifiers(rule_quals, qual_index):
            continue

        evidence = sorted(matched)
        results.append({
            "condition": rule["condition"],
            "snomed_code": rule["snomed"],
            "evidence": evidence,
        })

    # Sort: most evidence first, then alphabetical by condition name
    results.sort(key=lambda r: (-len(r["evidence"]), r["condition"]))
    return results


def _check_qualifiers(
    rule_quals: dict[str, dict],
    qual_index: dict[str, dict],
) -> bool:
    """Check whether extracted qualifiers satisfy rule constraints.

    Each key in *rule_quals* is a symptom name.  Its value is a dict
    mapping qualifier fields (e.g. ``"character"``, ``"aggravating_factors"``)
    to lists of acceptable values.

    For scalar fields (e.g. ``character``), the extracted value must be
    in the acceptable list.  For list fields (e.g. ``aggravating_factors``),
    at least one extracted value must be in the acceptable list.

    All qualifier constraints must be satisfied (AND logic).
    """
    for symptom, constraints in rule_quals.items():
        actual = qual_index.get(symptom.lower())
        if not actual:
            return False
        for field, acceptable in constraints.items():
            acceptable_lower = {v.lower() for v in acceptable}
            value = actual.get(field)
            if value is None:
                return False
            if isinstance(value, list):
                if not any(v.lower() in acceptable_lower for v in value):
                    return False
            else:
                if value.lower() not in acceptable_lower:
                    return False
    return True
