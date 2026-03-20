"""Deterministic text extractors for clinical note generation.

Each extractor is a pure function: text -> list[str].
No ML, no LLM — regex + curated keyword lists only.

Vocabularies are loaded from resources/extractors/*.json at import time,
with hardcoded defaults as fallback.
"""

from __future__ import annotations

import re
from typing import Callable

from app.extractor_vocab import load_vocab

# ── symptom extraction ────────────────────────────────────────────────

_DEFAULT_SYMPTOMS: list[str] = [
    "headache",
    "nausea",
    "vomiting",
    "dizziness",
    "fatigue",
    "fever",
    "cough",
    "pain",
    "shortness of breath",
    "chest pain",
    "sore throat",
    "diarrhea",
    "numbness",
    "tingling",
    "swelling",
    "bleeding",
    "rash",
    "insomnia",
    "anxiety",
    "weakness",
    "chills",
    "congestion",
    "palpitations",
    "cramps",
    "stiffness",
]

SYMPTOM_KEYWORDS: list[str] = load_vocab("symptoms", _DEFAULT_SYMPTOMS)

# Synonym map: colloquial term → canonical name.
# Applied at extraction time so downstream modules always see canonical names.
_SYMPTOM_SYNONYMS: dict[str, str] = {
    "painful urination": "dysuria",
    "shortness of breath": "dyspnea",
    "short of breath": "dyspnea",
    "breathlessness": "dyspnea",
    "difficulty breathing": "dyspnea",
}

# Pre-compile symptom patterns (longest first for multi-word matches)
_SYMPTOM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), kw)
    for kw in sorted(SYMPTOM_KEYWORDS, key=len, reverse=True)
]


def extract_symptoms(text: str) -> list[str]:
    """Extract symptom mentions from text.

    Returns unique symptoms in first-occurrence order, lowercase.
    Synonym terms are canonicalised via ``_SYMPTOM_SYNONYMS``.
    """
    seen: set[str] = set()
    results: list[str] = []
    for pattern, keyword in _SYMPTOM_PATTERNS:
        canonical = _SYMPTOM_SYNONYMS.get(keyword, keyword)
        if canonical in seen:
            continue
        if pattern.search(text):
            seen.add(canonical)
            results.append(canonical)
    return results


# ── negation extraction ───────────────────────────────────────────────

# Negation triggers — matched case-insensitively
_NEGATION_TRIGGERS = [
    "no",
    "not",
    "denies",
    "denied",
    "without",
    "absent",
    "negative for",
    "rules out",
    "ruled out",
]

# Build pattern: (trigger)\s+(clinical phrase up to boundary)
# Boundary = period, comma, semicolon, newline, or end of string.
# Phrase: 1-6 words captured greedily up to the boundary.
_NEGATION_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(t) for t in sorted(_NEGATION_TRIGGERS, key=len, reverse=True))
    + r")\s+([^.,;!?\n]{1,50})",
    re.IGNORECASE,
)


def _normalize_negation(trigger: str, phrase: str) -> str:
    """Format a negation match as a clean, readable bullet item.

    Capitalises the trigger and strips trailing whitespace from the phrase.
    Examples:
        ("denies", "chest pain")  -> "Denies chest pain"
        ("no", "fever")           -> "No fever"
        ("without", "nausea")     -> "Without nausea"
    """
    trigger_cap = trigger.capitalize()
    phrase_clean = phrase.strip().rstrip(".,;!?")
    return f"{trigger_cap} {phrase_clean}"


def extract_negations(text: str) -> list[str]:
    """Extract negation statements as clean bullet items.

    Returns normalised strings like "No fever", "Denies chest pain".
    Unique items only, preserving first-occurrence order.
    """
    seen: set[str] = set()
    results: list[str] = []
    for m in _NEGATION_PATTERN.finditer(text):
        trigger = m.group(1)
        phrase = m.group(2)
        bullet = _normalize_negation(trigger, phrase)
        key = bullet.lower()
        if key not in seen:
            seen.add(key)
            results.append(bullet)
    return results


# ── duration extraction ───────────────────────────────────────────────

_DURATION_PATTERN = re.compile(
    r"\b(\d+)\s*(days?|weeks?|months?|years?|hours?|minutes?|min)\b",
    re.IGNORECASE,
)

_WORD_NUMBERS: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "twenty": "20", "thirty": "30",
}

_SPELLED_DURATION_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True))
    + r")\s+(days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)


def extract_durations(text: str) -> list[str]:
    """Extract duration phrases like '3 days', 'three days', '2 weeks'.

    Matches both numeric (``3 days``) and spelled-out (``three days``)
    forms.  Spelled-out durations are returned as-is (e.g. ``three days``).

    Returns unique items in first-occurrence order.
    """
    seen: set[str] = set()
    results: list[str] = []

    # Collect all matches with their positions for first-occurrence ordering
    matches: list[tuple[int, str]] = []

    for m in _DURATION_PATTERN.finditer(text):
        value = m.group(1)
        unit = m.group(2).lower()
        matches.append((m.start(), f"{value} {unit}"))

    for m in _SPELLED_DURATION_PATTERN.finditer(text):
        word = m.group(1).lower()
        unit = m.group(2).lower()
        matches.append((m.start(), f"{word} {unit}"))

    # Sort by position for first-occurrence order
    matches.sort(key=lambda x: x[0])

    for _, item in matches:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            results.append(item)

    return results


# ── medication extraction ─────────────────────────────────────────────

_DEFAULT_MEDICATIONS: list[str] = [
    "ibuprofen",
    "acetaminophen",
    "paracetamol",
    "amoxicillin",
    "metformin",
    "lisinopril",
    "omeprazole",
    "aspirin",
    "prednisone",
    "metoprolol",
    "atorvastatin",
    "amlodipine",
    "albuterol",
    "gabapentin",
    "losartan",
    "pantoprazole",
    "sertraline",
    "furosemide",
    "warfarin",
    "insulin",
]

MEDICATION_KEYWORDS: list[str] = load_vocab("medications", _DEFAULT_MEDICATIONS)

# Pattern: medication name followed by dosage
_MED_DOSAGE_PATTERN = re.compile(
    r"\b(\w+)\s+(\d+\s*(?:mg|mcg|ml|g|units?))\b",
    re.IGNORECASE,
)

_MED_KEYWORD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), kw)
    for kw in MEDICATION_KEYWORDS
]


def extract_medications(text: str) -> list[str]:
    """Extract medication references from text.

    Two-pass approach:
      1. Medication + dosage patterns (e.g. 'ibuprofen 400 mg')
      2. Keyword-only fallback for known medication names

    Returns unique items in first-occurrence order.
    """
    seen: set[str] = set()
    results: list[str] = []

    # Pass 1: dosage patterns
    for m in _MED_DOSAGE_PATTERN.finditer(text):
        med_name = m.group(1).lower()
        dosage = m.group(2).strip()
        item = f"{med_name} {dosage}"
        if med_name not in seen:
            seen.add(med_name)
            results.append(item)

    # Pass 2: keyword fallback (skip if already found via dosage)
    for pattern, keyword in _MED_KEYWORD_PATTERNS:
        if keyword in seen:
            continue
        if pattern.search(text):
            seen.add(keyword)
            results.append(keyword)

    return results


# ── extractor registry ────────────────────────────────────────────────

EXTRACTORS: dict[str, Callable[[str], list[str]]] = {
    "symptoms": extract_symptoms,
    "negations": extract_negations,
    "durations": extract_durations,
    "medications": extract_medications,
}


def get_extractor(name: str) -> Callable[[str], list[str]]:
    """Look up an extractor by name.

    Raises KeyError if the extractor is not registered.
    """
    return EXTRACTORS[name]
