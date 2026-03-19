"""Synthea import — convert Synthea-style patient data to SCRIBE cases.

Minimal v1 importer for testing and case generation.  Converts
simplified Synthea-like patient dicts into SCRIBE-compatible case
format.  No FHIR parsing, no advanced clinical mapping.

Pure functions — no I/O mutation, no ML, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path


# ── constants ───────────────────────────────────────────────────────


_RED_FLAG_SYMPTOMS = frozenset({
    "shortness of breath",
    "chest pain",
    "syncope",
    "hemoptysis",
    "severe headache",
    "altered mental status",
    "high fever",
})


# ── segment building ───────────────────────────────────────────────


def build_segments(patient: dict) -> list[dict]:
    """Build SCRIBE-format segments from a Synthea patient dict.

    Args:
        patient: simplified Synthea-like patient dict.

    Returns:
        List of segment dicts with seg_id, t0, t1, speaker_id,
        normalized_text.
    """
    segments: list[dict] = []
    t = 0.0
    seg_n = 0

    # Opening segment: demographics + primary symptoms + duration.
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    symptoms = patient.get("symptoms") or []
    duration = patient.get("duration_days")

    if age or gender or symptoms:
        parts: list[str] = []
        if age and gender:
            parts.append(f"{age}-year-old {gender}")
        elif age:
            parts.append(f"{age}-year-old patient")
        elif gender:
            parts.append(f"{gender} patient")
        else:
            parts.append("Patient")

        if symptoms:
            symptom_str = " and ".join(symptoms[:2])
            parts.append(f"presenting with {symptom_str}")

        if duration and isinstance(duration, (int, float)) and duration > 0:
            parts.append(f"for {_format_duration(duration)}")

        text = " ".join(parts) + "."
        segments.append(_make_segment(seg_n, t, text))
        seg_n += 1
        t += 3.0

    # Additional symptom segments (beyond first 2).
    for symptom in symptoms[2:]:
        text = f"Reports {symptom}."
        segments.append(_make_segment(seg_n, t, text))
        seg_n += 1
        t += 3.0

    # Conditions as history segments.
    conditions = patient.get("conditions") or []
    # Skip first condition (used as hypothesis) for history.
    history_conditions = conditions[1:] if len(conditions) > 1 else []
    for condition in history_conditions:
        text = f"Has history of {condition}."
        segments.append(_make_segment(seg_n, t, text))
        seg_n += 1
        t += 3.0

    # Medications.
    medications = patient.get("medications") or []
    if medications:
        med_str = ", ".join(medications)
        text = f"Currently taking {med_str}."
        segments.append(_make_segment(seg_n, t, text))
        seg_n += 1
        t += 3.0

    # Allergies.
    allergies = patient.get("allergies") or []
    if allergies:
        allergy_str = ", ".join(allergies)
        text = f"Known allergies: {allergy_str}."
        segments.append(_make_segment(seg_n, t, text))
        seg_n += 1
        t += 3.0

    # Ensure at least one segment.
    if not segments:
        segments.append(_make_segment(0, 0.0, "Patient presents for evaluation."))

    return segments


def _make_segment(n: int, t: float, text: str) -> dict:
    return {
        "seg_id": f"seg_{n + 1:04d}",
        "t0": t,
        "t1": t + 3.0,
        "speaker_id": "spk_0",
        "normalized_text": text,
    }


def _format_duration(days: int | float) -> str:
    days = int(days)
    if days == 1:
        return "1 day"
    if days < 7:
        return f"{days} days"
    weeks = days // 7
    remainder = days % 7
    if remainder == 0:
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    return f"{days} days"


# ── ground truth building ──────────────────────────────────────────


def build_ground_truth(patient: dict) -> dict:
    """Build ground truth expectations from a Synthea patient dict.

    Args:
        patient: simplified Synthea-like patient dict.

    Returns:
        Ground truth dict with expected_hypotheses, key_findings,
        red_flags.
    """
    conditions = patient.get("conditions") or []
    symptoms = patient.get("symptoms") or []

    # First condition as expected hypothesis.
    expected_hypotheses = []
    if conditions:
        expected_hypotheses.append(conditions[0])

    # All symptoms as key findings.
    key_findings = list(symptoms)

    # Red flags from symptom list.
    red_flags = [
        s for s in symptoms
        if s.lower() in _RED_FLAG_SYMPTOMS
    ]

    return {
        "expected_hypotheses": expected_hypotheses,
        "key_findings": key_findings,
        "red_flags": red_flags,
    }


# ── patient to case conversion ─────────────────────────────────────


def synthea_patient_to_case(patient: dict) -> dict:
    """Convert a Synthea-style patient dict to a SCRIBE case.

    Args:
        patient: simplified Synthea-like patient dict.

    Returns:
        SCRIBE-compatible case dict.
    """
    patient_id = patient.get("id", "unknown")
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    symptoms = patient.get("symptoms") or []

    # Build title.
    title_parts: list[str] = []
    if age and gender:
        title_parts.append(f"{age}-year-old {gender}")
    elif age:
        title_parts.append(f"{age}-year-old patient")
    if symptoms:
        title_parts.append(f"with {' and '.join(symptoms[:2])}")
    title = " ".join(title_parts) if title_parts else f"Patient {patient_id}"

    segments = build_segments(patient)
    ground_truth = build_ground_truth(patient)

    return {
        "case_id": f"synthea_{patient_id}",
        "title": title,
        "segments": segments,
        "ground_truth": ground_truth,
        "config": {
            "mode": "assist",
            "update_strategy": "manual",
            "show_questions": True,
        },
        "meta": {
            "source": "synthea",
            "original_id": patient_id,
            "tags": list(patient.get("conditions") or []),
        },
    }


# ── batch operations ───────────────────────────────────────────────


def load_synthea_patients(path: str | Path) -> list[dict]:
    """Load Synthea patient data from a JSON file.

    Args:
        path: path to a JSON file containing a list of patient dicts.

    Returns:
        List of patient dicts.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file does not contain a list.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of patients, got {type(data).__name__}"
        )
    return data


def convert_patients_to_cases(patients: list[dict]) -> list[dict]:
    """Convert a list of Synthea patients to SCRIBE cases.

    Args:
        patients: list of simplified Synthea-like patient dicts.

    Returns:
        List of SCRIBE-compatible case dicts.
    """
    return [synthea_patient_to_case(p) for p in patients]
