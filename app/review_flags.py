"""Review flags — deterministic safety checks on extracted clinical findings.

Flags conditions that should be manually verified before a note is finalized.
Pure diagnostic layer — never modifies RAW, normalized, or extractor outputs.

Pipeline position:
    extractors → review flags → templates / reports
"""

from __future__ import annotations

import re
from typing import Optional

from app.extractors import (
    extract_medications,
    extract_symptoms,
    extract_durations,
)

# ── dosage detection ─────────────────────────────────────────────────

_DOSAGE_PATTERN = re.compile(
    r"\d+\s*(?:mg|mcg|ml|g|units?)\b",
    re.IGNORECASE,
)


def _has_dosage(medication_item: str) -> bool:
    """Return True if the medication item string includes a dosage."""
    return bool(_DOSAGE_PATTERN.search(medication_item))


# ── low-confidence threshold ─────────────────────────────────────────

LOW_CONFIDENCE_THRESHOLD = -1.0


# ── flag generation ──────────────────────────────────────────────────

def generate_review_flags(
    segments: list[dict],
    confidence_entries: Optional[list[dict]] = None,
) -> list[dict]:
    """Generate review flags from normalized segments and optional confidence data.

    Args:
        segments: list of normalized segment dicts (must have normalized_text,
                  and optionally seg_id, speaker_id, t0).
        confidence_entries: optional list of per-segment ASR quality dicts
                  (each with seg_id, avg_logprob, no_speech_prob, compression_ratio).

    Returns:
        list of flag dicts, each with: type, message, severity, and optional evidence.
    """
    flags: list[dict] = []

    # Collect all medications and durations across segments, per-segment
    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        evidence = _build_evidence(seg)

        # Rule 1: medication without dosage
        meds = extract_medications(text)
        for med in meds:
            if not _has_dosage(med):
                flag: dict = {
                    "type": "medication_without_dosage",
                    "message": f"Medication mentioned without dosage: {med}",
                    "severity": "warning",
                }
                if evidence:
                    flag["evidence"] = evidence
                flags.append(flag)

        # Rule 2: symptom without duration
        symptoms = extract_symptoms(text)
        durations = extract_durations(text)
        if symptoms and not durations:
            for symptom in symptoms:
                flag = {
                    "type": "symptom_without_duration",
                    "message": f"Symptom mentioned without duration: {symptom}",
                    "severity": "info",
                }
                if evidence:
                    flag["evidence"] = evidence
                flags.append(flag)

    # Rule 3: low-confidence segments
    if confidence_entries:
        for entry in confidence_entries:
            alp = entry.get("avg_logprob")
            if alp is not None and alp < LOW_CONFIDENCE_THRESHOLD:
                seg_id = entry.get("seg_id", "")
                flag = {
                    "type": "low_confidence_segment",
                    "message": f"Low ASR confidence on segment {seg_id} "
                               f"(avg_logprob={alp:.2f})",
                    "severity": "warning",
                }
                evidence = {}
                if seg_id:
                    evidence["segment_id"] = seg_id
                t0 = entry.get("t0")
                if t0 is not None:
                    evidence["t_start"] = t0
                if evidence:
                    flag["evidence"] = evidence
                flags.append(flag)

    return flags


def _build_evidence(seg: dict) -> dict:
    """Build evidence dict from segment, returning empty dict if no fields."""
    ev: dict = {}
    seg_id = seg.get("seg_id")
    if seg_id:
        ev["segment_id"] = seg_id
    speaker_id = seg.get("speaker_id")
    if speaker_id:
        ev["speaker_id"] = speaker_id
    t0 = seg.get("t0")
    if t0 is not None:
        ev["t_start"] = t0
    return ev
