"""Layer 1: Observation records — source evidence traceability.

Builds observation records from segments post-extraction.  Multiple
observations per finding — every segment where a finding appears
produces an observation (not just first match).  This preserves
the full evidence trail.

Pure function — no I/O, no side effects.
"""

from __future__ import annotations

import re

from app.extractors import (
    _SYMPTOM_PATTERNS,
    _MED_KEYWORD_PATTERNS,
    _NEGATION_PATTERN,
    _DURATION_PATTERN,
)


def build_observation_layer(
    segments: list[dict],
    symptoms: list[str],
    negations: list[str],
    durations: list[str],
    medications: list[str],
) -> list[dict]:
    """Build observation records from segments and extracted findings.

    For each finding, scans ALL segments using compiled patterns from
    ``app/extractors``.  Every segment match produces an observation
    record.  Observations ordered by t_start, then by finding_type.

    Args:
        segments: list of normalized segment dicts.
        symptoms: extracted symptom strings.
        negations: extracted negation strings.
        durations: extracted duration strings.
        medications: extracted medication strings.

    Returns:
        list of observation dicts with keys: ``observation_id``,
        ``finding_type``, ``value``, ``seg_id``, ``speaker_id``,
        ``t_start``, ``t_end``, ``source_text``.
    """
    if not segments:
        return []

    observations: list[dict] = []

    # Build lookup sets for fast membership check
    symptom_set = {s.lower() for s in symptoms}
    medication_set = {m.lower().split()[0] if " " in m else m.lower() for m in medications}
    # Also keep full medication strings for multi-word matching
    medication_full_set = {m.lower() for m in medications}
    negation_set = {n.lower() for n in negations}
    duration_set = {d.lower() for d in durations}

    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        seg_id = seg.get("seg_id")
        speaker_id = seg.get("speaker_id")
        t_start = seg.get("t0")
        t_end = seg.get("t1")

        # Symptoms
        for pattern, keyword in _SYMPTOM_PATTERNS:
            if keyword.lower() not in symptom_set:
                continue
            if pattern.search(text):
                observations.append(_obs(
                    "symptom", keyword, seg_id, speaker_id,
                    t_start, t_end, text,
                ))

        # Negations
        for m in _NEGATION_PATTERN.finditer(text):
            trigger = m.group(1)
            phrase = m.group(2).strip().rstrip(".,;!?")
            bullet = f"{trigger.capitalize()} {phrase}"
            if bullet.lower() in negation_set:
                observations.append(_obs(
                    "negation", bullet, seg_id, speaker_id,
                    t_start, t_end, text,
                ))

        # Durations
        for m in _DURATION_PATTERN.finditer(text):
            value = m.group(1)
            unit = m.group(2).lower()
            item = f"{value} {unit}"
            if item in duration_set:
                observations.append(_obs(
                    "duration", item, seg_id, speaker_id,
                    t_start, t_end, text,
                ))

        # Medications
        for pattern, keyword in _MED_KEYWORD_PATTERNS:
            if keyword.lower() not in medication_set and keyword.lower() not in medication_full_set:
                continue
            if pattern.search(text):
                observations.append(_obs(
                    "medication", keyword, seg_id, speaker_id,
                    t_start, t_end, text,
                ))

    # Sort by t_start, then finding_type
    _TYPE_ORDER = {"symptom": 0, "negation": 1, "duration": 2, "medication": 3}
    observations.sort(key=lambda o: (
        o["t_start"] if o["t_start"] is not None else float("inf"),
        _TYPE_ORDER.get(o["finding_type"], 99),
    ))

    # Assign sequential IDs
    for i, obs in enumerate(observations):
        obs["observation_id"] = f"obs_{i + 1:04d}"

    return observations


def _obs(
    finding_type: str,
    value: str,
    seg_id: str | None,
    speaker_id: str | None,
    t_start: float | None,
    t_end: float | None,
    source_text: str,
) -> dict:
    """Create an observation record (without ID — assigned after sorting)."""
    return {
        "observation_id": "",  # placeholder
        "finding_type": finding_type,
        "value": value,
        "seg_id": seg_id,
        "speaker_id": speaker_id,
        "t_start": t_start,
        "t_end": t_end,
        "source_text": source_text,
    }
