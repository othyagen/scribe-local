"""Deterministic speaker role detection from normalized transcript text.

Assigns roles (clinician, patient, other, unknown) to each speaker_id
based on linguistic signal patterns.  Supports multiple clinicians and
multiple patients — roles are not hardcoded 1:1.

No ML, no LLM — regex + keyword density heuristics only.
"""

from __future__ import annotations

import re
from typing import Optional


# ── role constants ────────────────────────────────────────────────────

ROLE_CLINICIAN = "clinician"
ROLE_PATIENT = "patient"
ROLE_OTHER = "other"
ROLE_UNKNOWN = "unknown"


# ── signal patterns (per language) ────────────────────────────────────

# Clinician signals: questions, directives, medical instructions
_CLINICIAN_SIGNALS: dict[str, list[str]] = {
    "en": [
        "how long",
        "do you",
        "are you",
        "have you",
        "any history",
        "are you taking",
        "let me",
        "I recommend",
        "we should",
        "prescribe",
        "refer you",
        "follow up",
        "your symptoms",
        "examination",
        "diagnosis",
        "treatment",
    ],
    "da": [
        "har du",
        "hvor længe",
        "oplever du",
        "tager du",
        "nogen historie",
        "jeg anbefaler",
        "vi skal",
        "lad mig",
        "dine symptomer",
        "undersøgelse",
        "behandling",
    ],
    "sv": [
        "har du",
        "hur länge",
        "upplever du",
        "tar du",
        "jag rekommenderar",
        "vi bör",
        "låt mig",
        "dina symptom",
        "undersökning",
        "behandling",
    ],
}

# Patient signals: first-person narrative, symptom descriptions
_PATIENT_SIGNALS: dict[str, list[str]] = {
    "en": [
        "I have",
        "I feel",
        "I've been",
        "I noticed",
        "I can't",
        "I'm",
        "it started",
        "it hurts",
        "my pain",
        "my head",
        "my stomach",
        "woke up",
        "I think",
        "I was",
    ],
    "da": [
        "jeg har",
        "jeg føler",
        "det startede",
        "jeg bliver",
        "jeg kan ikke",
        "det gør ondt",
        "mit hoved",
        "min mave",
        "jeg vågnede",
        "jeg tror",
    ],
    "sv": [
        "jag har",
        "jag känner",
        "det började",
        "jag blir",
        "jag kan inte",
        "det gör ont",
        "mitt huvud",
        "min mage",
        "jag vaknade",
        "jag tror",
    ],
}

# Profile name substrings that hint at clinician role
_CLINICIAN_PROFILE_HINTS = [
    "doctor", "dr", "clinician", "physician", "nurse",
    "læge", "doktor", "kliniker", "sygeplejerske",       # Danish
    "läkare", "kliniker", "sjuksköterska",                # Swedish
]


def _compile_patterns(
    signals: dict[str, list[str]], language: str
) -> list[re.Pattern[str]]:
    """Compile signal keyword lists into regex patterns for a language."""
    phrases = signals.get(language, signals.get("en", []))
    return [
        re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
        for p in phrases
    ]


# ── core detection ────────────────────────────────────────────────────

def detect_speaker_roles(
    segments: list[dict],
    language: str = "en",
    confidence_threshold: float = 0.6,
    profile_hints: Optional[dict[str, str]] = None,
) -> dict[str, dict]:
    """Detect speaker roles from normalized segment text.

    Args:
        segments: list of normalized segment dicts (must have speaker_id,
                  normalized_text)
        language: session language ("en", "da", "sv")
        confidence_threshold: minimum score to assign a role
        profile_hints: optional {speaker_id: profile_name} for hint bonuses

    Returns:
        {speaker_id: {"role": str, "confidence": float, "evidence": list[str]}}
    """
    clinician_patterns = _compile_patterns(_CLINICIAN_SIGNALS, language)
    patient_patterns = _compile_patterns(_PATIENT_SIGNALS, language)

    # Group text by speaker
    speaker_texts: dict[str, list[str]] = {}
    for seg in segments:
        spk = seg.get("speaker_id", "")
        text = seg.get("normalized_text", "")
        if spk:
            speaker_texts.setdefault(spk, []).append(text)

    roles: dict[str, dict] = {}

    for speaker_id, texts in speaker_texts.items():
        combined = " ".join(texts)
        word_count = max(len(combined.split()), 1)

        # Count signal hits
        clinician_hits: list[str] = []
        for pat in clinician_patterns:
            matches = pat.findall(combined)
            if matches:
                clinician_hits.extend(matches)

        patient_hits: list[str] = []
        for pat in patient_patterns:
            matches = pat.findall(combined)
            if matches:
                patient_hits.extend(matches)

        # Question marks are a strong clinician signal
        question_count = combined.count("?")
        clinician_hits.extend(["?"] * question_count)

        # Compute density scores (hits per 10 words for reasonable scale)
        clinician_score = len(clinician_hits) / word_count * 10
        patient_score = len(patient_hits) / word_count * 10

        # Profile hint bonus
        hint_bonus = 0.0
        if profile_hints and speaker_id in profile_hints:
            pname = profile_hints[speaker_id].lower()
            if any(h in pname for h in _CLINICIAN_PROFILE_HINTS):
                hint_bonus = 0.2

        clinician_score += hint_bonus

        # Clamp scores to [0, 1]
        clinician_score = min(clinician_score, 1.0)
        patient_score = min(patient_score, 1.0)

        # Assign role
        evidence: list[str] = []
        if clinician_hits:
            evidence.append(
                f"clinician_signals={len(clinician_hits)}"
            )
        if patient_hits:
            evidence.append(
                f"patient_signals={len(patient_hits)}"
            )
        if hint_bonus > 0:
            evidence.append("profile_hint=clinician")

        if (
            clinician_score > patient_score
            and clinician_score >= confidence_threshold
        ):
            role = ROLE_CLINICIAN
            confidence = clinician_score
        elif (
            patient_score > clinician_score
            and patient_score >= confidence_threshold
        ):
            role = ROLE_PATIENT
            confidence = patient_score
        else:
            role = ROLE_UNKNOWN
            confidence = max(clinician_score, patient_score)

        roles[speaker_id] = {
            "role": role,
            "confidence": round(confidence, 4),
            "evidence": evidence,
        }

    return roles


# ── display labels ────────────────────────────────────────────────────

def get_role_label(role: str, speaker_id: str = "") -> str:
    """Map role to a human-readable display label.

    - clinician → "Clinician"
    - patient   → "Patient"
    - unknown/other → original speaker_id (or "Speaker" if empty)
    """
    if role == ROLE_CLINICIAN:
        return "Clinician"
    if role == ROLE_PATIENT:
        return "Patient"
    return speaker_id or "Speaker"
