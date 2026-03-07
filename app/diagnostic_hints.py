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
        "symptoms": ["fever", "cough", "shortness of breath"],
        "condition": "Pneumonia",
        "snomed": "233604007",
    },
    {
        "symptoms": ["headache", "neck stiffness", "fever"],
        "condition": "Meningitis",
        "snomed": "7180009",
    },
    {
        "symptoms": ["chest pain", "shortness of breath"],
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
) -> list[dict]:
    """Generate diagnostic hints from symptom and negation lists.

    Args:
        symptoms: list of symptom strings (e.g. from extract_symptoms)
        negations: optional list of negation strings (e.g. from extract_negations)

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

    results: list[dict] = []
    for rule in RULES:
        required = {s.lower() for s in rule["symptoms"]}
        if required <= active_symptoms:
            evidence = sorted(required & active_symptoms)
            results.append({
                "condition": rule["condition"],
                "snomed_code": rule["snomed"],
                "evidence": evidence,
            })

    # Sort: most evidence first, then alphabetical by condition name
    results.sort(key=lambda r: (-len(r["evidence"]), r["condition"]))
    return results
