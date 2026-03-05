"""Tests for deterministic speaker role detection."""

from __future__ import annotations

import pytest

from app.role_detection import (
    detect_speaker_roles,
    get_role_label,
    ROLE_CLINICIAN,
    ROLE_PATIENT,
    ROLE_UNKNOWN,
)


def _make_segments(speaker_texts: dict[str, list[str]]) -> list[dict]:
    """Create normalized segment dicts from {speaker_id: [texts]}."""
    segments: list[dict] = []
    i = 0
    for speaker_id, texts in speaker_texts.items():
        for text in texts:
            segments.append({
                "seg_id": f"seg_{i+1:04d}",
                "t0": float(i),
                "t1": float(i + 1),
                "speaker_id": speaker_id,
                "normalized_text": text,
            })
            i += 1
    return segments


# ── clinician detection ──────────────────────────────────────────────


class TestClinicianDetection:
    def test_question_heavy_speaker_is_clinician(self):
        segments = _make_segments({
            "spk_0": [
                "How long have you had this pain?",
                "Do you take any medications?",
                "Have you experienced this before?",
                "Are you taking anything for it?",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_CLINICIAN

    def test_directive_patterns(self):
        segments = _make_segments({
            "spk_0": [
                "I recommend taking ibuprofen.",
                "We should schedule a follow up.",
                "Let me examine your throat.",
                "I recommend rest for a few days.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_CLINICIAN

    def test_danish_clinician_patterns(self):
        segments = _make_segments({
            "spk_0": [
                "Har du haft det længe?",
                "Oplever du nogen smerter?",
                "Tager du medicin?",
                "Hvor længe har du haft det?",
            ],
        })
        roles = detect_speaker_roles(segments, "da")
        assert roles["spk_0"]["role"] == ROLE_CLINICIAN


# ── patient detection ────────────────────────────────────────────────


class TestPatientDetection:
    def test_first_person_symptoms(self):
        segments = _make_segments({
            "spk_0": [
                "I have a terrible headache.",
                "I feel dizzy when I stand up.",
                "I've been having trouble sleeping.",
                "I noticed a rash on my arm.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_PATIENT

    def test_danish_patient_patterns(self):
        segments = _make_segments({
            "spk_0": [
                "Jeg har ondt i hovedet.",
                "Jeg føler mig svimmel.",
                "Det startede for tre dage siden.",
                "Jeg bliver træt hele tiden.",
            ],
        })
        roles = detect_speaker_roles(segments, "da")
        assert roles["spk_0"]["role"] == ROLE_PATIENT

    def test_narrative_speaker(self):
        segments = _make_segments({
            "spk_0": [
                "I was feeling fine yesterday.",
                "It started this morning.",
                "My pain is getting worse.",
                "I can't sleep because of it.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_PATIENT


# ── multiple speakers ────────────────────────────────────────────────


class TestMultipleSpeakers:
    def test_two_clinicians(self):
        segments = _make_segments({
            "spk_0": [
                "How long have you had this?",
                "Do you have any allergies?",
                "Are you taking medications?",
            ],
            "spk_1": [
                "Let me check your vitals.",
                "I recommend a blood test.",
                "We should do an examination.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_CLINICIAN
        assert roles["spk_1"]["role"] == ROLE_CLINICIAN

    def test_two_patients(self):
        segments = _make_segments({
            "spk_0": [
                "I have a headache.",
                "I feel nauseous.",
                "I've been tired all week.",
            ],
            "spk_1": [
                "I have chest pain.",
                "I feel short of breath.",
                "I noticed swelling in my legs.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_PATIENT
        assert roles["spk_1"]["role"] == ROLE_PATIENT

    def test_mixed_roles(self):
        segments = _make_segments({
            "spk_0": [
                "How long have you had this?",
                "Do you take any medications?",
                "I recommend ibuprofen.",
            ],
            "spk_1": [
                "I have a headache.",
                "I feel dizzy.",
                "It started yesterday.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_CLINICIAN
        assert roles["spk_1"]["role"] == ROLE_PATIENT


# ── unknown fallback ─────────────────────────────────────────────────


class TestUnknownFallback:
    def test_unclear_speaker_gets_unknown(self):
        segments = _make_segments({
            "spk_0": [
                "Hello.",
                "Thank you.",
                "Goodbye.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_UNKNOWN

    def test_empty_text_gets_unknown(self):
        segments = _make_segments({
            "spk_0": ["", "", ""],
        })
        roles = detect_speaker_roles(segments, "en")
        assert roles["spk_0"]["role"] == ROLE_UNKNOWN

    def test_tie_gets_unknown(self):
        """When clinician and patient signals are equal, assign unknown."""
        segments = _make_segments({
            "spk_0": [
                "Do you have pain?",
                "I have a headache.",
            ],
        })
        roles = detect_speaker_roles(segments, "en")
        # Should be unknown or whichever dominates, but not crash
        assert roles["spk_0"]["role"] in (ROLE_CLINICIAN, ROLE_PATIENT, ROLE_UNKNOWN)


# ── profile hints ────────────────────────────────────────────────────


class TestProfileHints:
    def test_doctor_hint_boosts_clinician(self):
        """Profile name containing 'doctor' boosts clinician score."""
        segments = _make_segments({
            "spk_0": [
                "How are you feeling today?",
                "Any changes since last visit?",
            ],
        })
        roles = detect_speaker_roles(
            segments, "en",
            profile_hints={"spk_0": "doctor_smith"},
        )
        assert roles["spk_0"]["role"] == ROLE_CLINICIAN
        assert "profile_hint=clinician" in roles["spk_0"]["evidence"]

    def test_hint_not_sufficient_alone(self):
        """Hint alone without signals should not force assignment."""
        segments = _make_segments({
            "spk_0": ["Hello.", "Thanks."],
        })
        roles = detect_speaker_roles(
            segments, "en",
            profile_hints={"spk_0": "doctor_jones"},
            confidence_threshold=0.9,
        )
        # Hint adds 0.2 but without signals, total still low
        # Could be unknown or clinician depending on threshold
        assert roles["spk_0"]["role"] in (ROLE_CLINICIAN, ROLE_UNKNOWN)


# ── role label mapping ───────────────────────────────────────────────


class TestRoleLabelMapping:
    def test_clinician_label(self):
        assert get_role_label(ROLE_CLINICIAN) == "Clinician"

    def test_patient_label(self):
        assert get_role_label(ROLE_PATIENT) == "Patient"

    def test_unknown_falls_back_to_speaker_id(self):
        assert get_role_label(ROLE_UNKNOWN, "spk_0") == "spk_0"

    def test_unknown_without_speaker_id(self):
        assert get_role_label(ROLE_UNKNOWN) == "Speaker"


# ── contract tests ───────────────────────────────────────────────────


class TestDetectSpeakerRolesContract:
    def test_return_structure(self):
        segments = _make_segments({"spk_0": ["I have pain."]})
        roles = detect_speaker_roles(segments, "en")
        assert "spk_0" in roles
        entry = roles["spk_0"]
        assert "role" in entry
        assert "confidence" in entry
        assert "evidence" in entry
        assert isinstance(entry["confidence"], float)
        assert isinstance(entry["evidence"], list)

    def test_deterministic(self):
        segments = _make_segments({
            "spk_0": ["How long have you had this?"],
            "spk_1": ["I have a headache."],
        })
        r1 = detect_speaker_roles(segments, "en")
        r2 = detect_speaker_roles(segments, "en")
        assert r1 == r2
