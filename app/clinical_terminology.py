"""Centralized clinical terminology registry.

Single source of truth for clinical concepts: canonical labels, synonyms,
red flag status, and display names.

Pure data + lookup functions.  No I/O, no ML, no LLM.
"""

from __future__ import annotations


# ── term registry ────────────────────────────────────────────────────

CLINICAL_TERMS: dict[str, dict] = {
    "dyspnea": {
        "display": "Dyspnea",
        "type": "symptom",
        "synonyms": [
            "shortness of breath",
            "short of breath",
            "breathlessness",
            "difficulty breathing",
            "sob",
        ],
        "red_flag": True,
        "snomed_code": "267036007",
    },
    "chest pain": {
        "display": "Chest pain",
        "type": "symptom",
        "synonyms": [
            "chest discomfort",
            "chest tightness",
        ],
        "red_flag": True,
        "snomed_code": "29857009",
    },
    "dysuria": {
        "display": "Dysuria",
        "type": "symptom",
        "synonyms": [
            "painful urination",
            "burning urination",
        ],
        "red_flag": False,
        "snomed_code": "49650001",
    },
    "abdominal pain": {
        "display": "Abdominal pain",
        "type": "symptom",
        "synonyms": [],
        "red_flag": False,
        "snomed_code": "21522001",
    },
    "fever": {
        "display": "Fever",
        "type": "symptom",
        "synonyms": [],
        "red_flag": False,
        "snomed_code": "386661006",
    },
    "syncope": {
        "display": "Syncope",
        "type": "symptom",
        "synonyms": [
            "fainting",
            "loss of consciousness",
        ],
        "red_flag": True,
        "snomed_code": "271594007",
    },
    "cough": {
        "display": "Cough",
        "type": "symptom",
        "synonyms": [],
        "red_flag": False,
        "snomed_code": "49727002",
    },
    "hemoptysis": {
        "display": "Hemoptysis",
        "type": "symptom",
        "synonyms": [
            "coughing up blood",
        ],
        "red_flag": True,
        "snomed_code": "66857006",
    },
}


# ── reverse index (built once at import time) ────────────────────────

_SYNONYM_TO_CANONICAL: dict[str, str] = {}
for _label, _term in CLINICAL_TERMS.items():
    for _syn in _term["synonyms"]:
        _SYNONYM_TO_CANONICAL[_syn.lower()] = _label


# ── public API ───────────────────────────────────────────────────────


def get_canonical_label(text: str) -> str:
    """Return the canonical label for *text*.

    Case-insensitive, whitespace-trimmed.  If *text* matches a known
    synonym the canonical label is returned.  If *text* is itself a
    canonical label it is returned as-is (lowercased).  Unknown labels
    are returned stripped but with original casing preserved.
    """
    stripped = text.strip()
    key = stripped.lower()

    # Direct canonical match
    if key in CLINICAL_TERMS:
        return key

    # Synonym match
    if key in _SYNONYM_TO_CANONICAL:
        return _SYNONYM_TO_CANONICAL[key]

    # Unknown — preserve original casing
    return stripped


def get_term(label: str) -> dict | None:
    """Return the term dict for a canonical label, or ``None``."""
    key = label.strip().lower()
    return CLINICAL_TERMS.get(key)


def is_red_flag(label: str) -> bool:
    """Return whether *label* (or its synonym) is a red flag.

    Returns ``False`` for unknown labels.
    """
    canonical = get_canonical_label(label)
    term = CLINICAL_TERMS.get(canonical)
    if term is None:
        return False
    return term["red_flag"]


def get_all_labels() -> list[str]:
    """Return all canonical labels in registry order."""
    return list(CLINICAL_TERMS.keys())
