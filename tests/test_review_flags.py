"""Tests for review flags — deterministic safety checks on extracted findings."""

from __future__ import annotations

import pytest

from app.review_flags import generate_review_flags, _has_dosage
from app.export_clinical_note import build_clinical_note


# ── fixtures ─────────────────────────────────────────────────────────

def _seg(
    seg_id: str = "seg_0001",
    t0: float = 0.0,
    t1: float = 1.5,
    speaker_id: str = "spk_0",
    text: str = "hello",
) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── _has_dosage tests ────────────────────────────────────────────────


class TestHasDosage:
    def test_with_dosage(self):
        assert _has_dosage("ibuprofen 400 mg") is True

    def test_without_dosage(self):
        assert _has_dosage("metformin") is False

    def test_mcg(self):
        assert _has_dosage("fentanyl 50 mcg") is True

    def test_ml(self):
        assert _has_dosage("saline 10 ml") is True

    def test_units(self):
        assert _has_dosage("insulin 20 units") is True


# ── medication_without_dosage ────────────────────────────────────────


class TestMedicationWithoutDosage:
    def test_keyword_only_medication_flagged(self):
        segs = [_seg(text="the patient takes metformin daily")]
        flags = generate_review_flags(segs)
        med_flags = [f for f in flags if f["type"] == "medication_without_dosage"]
        assert len(med_flags) == 1
        assert "metformin" in med_flags[0]["message"]
        assert med_flags[0]["severity"] == "warning"

    def test_medication_with_dosage_not_flagged(self):
        segs = [_seg(text="ibuprofen 400 mg twice daily")]
        flags = generate_review_flags(segs)
        med_flags = [f for f in flags if f["type"] == "medication_without_dosage"]
        assert len(med_flags) == 0

    def test_multiple_medications_mixed(self):
        segs = [_seg(text="metformin and ibuprofen 400 mg")]
        flags = generate_review_flags(segs)
        med_flags = [f for f in flags if f["type"] == "medication_without_dosage"]
        # metformin has no dosage, ibuprofen does
        assert len(med_flags) == 1
        assert "metformin" in med_flags[0]["message"]

    def test_no_medications_no_flags(self):
        segs = [_seg(text="the weather is nice today")]
        flags = generate_review_flags(segs)
        med_flags = [f for f in flags if f["type"] == "medication_without_dosage"]
        assert len(med_flags) == 0

    def test_evidence_attached(self):
        segs = [_seg(
            seg_id="seg_0005", t0=12.3, speaker_id="spk_1",
            text="started aspirin",
        )]
        flags = generate_review_flags(segs)
        med_flags = [f for f in flags if f["type"] == "medication_without_dosage"]
        assert len(med_flags) == 1
        ev = med_flags[0]["evidence"]
        assert ev["segment_id"] == "seg_0005"
        assert ev["speaker_id"] == "spk_1"
        assert ev["t_start"] == 12.3


# ── symptom_without_duration ─────────────────────────────────────────


class TestSymptomWithoutDuration:
    def test_symptom_without_duration_flagged(self):
        segs = [_seg(text="patient reports headache")]
        flags = generate_review_flags(segs)
        sym_flags = [f for f in flags if f["type"] == "symptom_without_duration"]
        assert len(sym_flags) == 1
        assert "headache" in sym_flags[0]["message"]
        assert sym_flags[0]["severity"] == "info"

    def test_symptom_with_duration_not_flagged(self):
        segs = [_seg(text="headache for 3 days")]
        flags = generate_review_flags(segs)
        sym_flags = [f for f in flags if f["type"] == "symptom_without_duration"]
        assert len(sym_flags) == 0

    def test_multiple_symptoms_without_duration(self):
        segs = [_seg(text="headache and nausea")]
        flags = generate_review_flags(segs)
        sym_flags = [f for f in flags if f["type"] == "symptom_without_duration"]
        assert len(sym_flags) == 2
        messages = [f["message"] for f in sym_flags]
        assert any("headache" in m for m in messages)
        assert any("nausea" in m for m in messages)

    def test_symptom_with_duration_in_same_segment(self):
        segs = [_seg(text="nausea for 2 weeks")]
        flags = generate_review_flags(segs)
        sym_flags = [f for f in flags if f["type"] == "symptom_without_duration"]
        assert len(sym_flags) == 0

    def test_no_symptoms_no_flags(self):
        segs = [_seg(text="everything looks good")]
        flags = generate_review_flags(segs)
        sym_flags = [f for f in flags if f["type"] == "symptom_without_duration"]
        assert len(sym_flags) == 0

    def test_evidence_attached(self):
        segs = [_seg(
            seg_id="seg_0003", t0=5.0, speaker_id="spk_0",
            text="I have a fever",
        )]
        flags = generate_review_flags(segs)
        sym_flags = [f for f in flags if f["type"] == "symptom_without_duration"]
        assert len(sym_flags) == 1
        ev = sym_flags[0]["evidence"]
        assert ev["segment_id"] == "seg_0003"
        assert ev["t_start"] == 5.0


# ── low_confidence_segment ───────────────────────────────────────────


class TestLowConfidenceSegment:
    def test_low_confidence_flagged(self):
        segs = [_seg(text="hello")]
        conf = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.5,
             "avg_logprob": -1.5, "no_speech_prob": 0.1, "compression_ratio": 1.0},
        ]
        flags = generate_review_flags(segs, confidence_entries=conf)
        lc_flags = [f for f in flags if f["type"] == "low_confidence_segment"]
        assert len(lc_flags) == 1
        assert "seg_0001" in lc_flags[0]["message"]
        assert "-1.50" in lc_flags[0]["message"]
        assert lc_flags[0]["severity"] == "warning"

    def test_good_confidence_not_flagged(self):
        segs = [_seg(text="hello")]
        conf = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.5,
             "avg_logprob": -0.5, "no_speech_prob": 0.1, "compression_ratio": 1.0},
        ]
        flags = generate_review_flags(segs, confidence_entries=conf)
        lc_flags = [f for f in flags if f["type"] == "low_confidence_segment"]
        assert len(lc_flags) == 0

    def test_none_avg_logprob_not_flagged(self):
        segs = [_seg(text="hello")]
        conf = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.5,
             "avg_logprob": None, "no_speech_prob": 0.1, "compression_ratio": 1.0},
        ]
        flags = generate_review_flags(segs, confidence_entries=conf)
        lc_flags = [f for f in flags if f["type"] == "low_confidence_segment"]
        assert len(lc_flags) == 0

    def test_no_confidence_data(self):
        segs = [_seg(text="headache")]
        flags = generate_review_flags(segs, confidence_entries=None)
        lc_flags = [f for f in flags if f["type"] == "low_confidence_segment"]
        assert len(lc_flags) == 0

    def test_evidence_on_low_confidence(self):
        segs = [_seg(text="hello")]
        conf = [
            {"seg_id": "seg_0002", "t0": 3.5, "t1": 5.0,
             "avg_logprob": -2.0, "no_speech_prob": 0.1, "compression_ratio": 1.0},
        ]
        flags = generate_review_flags(segs, confidence_entries=conf)
        lc_flags = [f for f in flags if f["type"] == "low_confidence_segment"]
        assert len(lc_flags) == 1
        ev = lc_flags[0]["evidence"]
        assert ev["segment_id"] == "seg_0002"
        assert ev["t_start"] == 3.5


# ── graceful handling ────────────────────────────────────────────────


class TestGracefulHandling:
    def test_empty_segments(self):
        flags = generate_review_flags([])
        assert flags == []

    def test_segment_missing_text(self):
        segs = [{"seg_id": "seg_0001"}]
        flags = generate_review_flags(segs)
        assert flags == []

    def test_segment_missing_optional_fields(self):
        """Segment with text but no seg_id/speaker_id/t0 should not crash."""
        segs = [{"normalized_text": "metformin daily"}]
        flags = generate_review_flags(segs)
        med_flags = [f for f in flags if f["type"] == "medication_without_dosage"]
        assert len(med_flags) == 1
        # No evidence attached (no fields to build it from)
        assert "evidence" not in med_flags[0]

    def test_confidence_entry_missing_fields(self):
        """Confidence entry missing optional fields should not crash."""
        segs = [_seg(text="hello")]
        conf = [{"avg_logprob": -1.5}]  # missing seg_id, t0, etc.
        flags = generate_review_flags(segs, confidence_entries=conf)
        lc_flags = [f for f in flags if f["type"] == "low_confidence_segment"]
        assert len(lc_flags) == 1


# ── combined rules ───────────────────────────────────────────────────


class TestCombinedRules:
    def test_multiple_flag_types(self):
        segs = [
            _seg(seg_id="seg_0001", text="headache and metformin daily"),
        ]
        conf = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.5,
             "avg_logprob": -1.5, "no_speech_prob": 0.1, "compression_ratio": 1.0},
        ]
        flags = generate_review_flags(segs, confidence_entries=conf)
        types = {f["type"] for f in flags}
        assert "medication_without_dosage" in types
        assert "symptom_without_duration" in types
        assert "low_confidence_segment" in types

    def test_clean_segment_no_flags(self):
        """Segment with dosage + duration → no flags."""
        segs = [_seg(text="ibuprofen 400 mg for headache lasting 3 days")]
        flags = generate_review_flags(segs)
        assert flags == []


# ── clinical note integration ────────────────────────────────────────


class TestClinicalNoteReviewFlags:
    def test_review_flags_rendered_in_note(self):
        template = {
            "name": "Test Note",
            "format": "markdown",
            "show_review_flags": True,
            "sections": [
                {"title": "Findings", "extractors": ["symptoms"]},
            ],
        }
        segs = [_seg(text="patient reports headache")]
        flags = [
            {
                "type": "symptom_without_duration",
                "message": "Symptom mentioned without duration: headache",
                "severity": "info",
            },
        ]
        note = build_clinical_note(segs, template, review_flags=flags)
        assert "## Review Flags" in note
        assert "INFO" in note
        assert "headache" in note

    def test_review_flags_not_rendered_without_template_flag(self):
        template = {
            "name": "Test Note",
            "format": "markdown",
            "sections": [
                {"title": "Findings", "extractors": ["symptoms"]},
            ],
        }
        segs = [_seg(text="patient reports headache")]
        flags = [
            {
                "type": "symptom_without_duration",
                "message": "Symptom mentioned without duration: headache",
                "severity": "info",
            },
        ]
        note = build_clinical_note(segs, template, review_flags=flags)
        assert "Review Flags" not in note

    def test_review_flags_empty_list_no_section(self):
        template = {
            "name": "Test Note",
            "format": "markdown",
            "show_review_flags": True,
            "sections": [
                {"title": "Findings", "extractors": ["symptoms"]},
            ],
        }
        segs = [_seg(text="headache for 3 days")]
        note = build_clinical_note(segs, template, review_flags=[])
        assert "Review Flags" not in note

    def test_review_flags_none_no_section(self):
        template = {
            "name": "Test Note",
            "format": "markdown",
            "show_review_flags": True,
            "sections": [
                {"title": "Findings", "extractors": ["symptoms"]},
            ],
        }
        segs = [_seg(text="headache for 3 days")]
        note = build_clinical_note(segs, template, review_flags=None)
        assert "Review Flags" not in note

    def test_review_flags_text_format(self):
        template = {
            "name": "Test Note",
            "format": "text",
            "show_review_flags": True,
            "sections": [
                {"title": "Findings", "extractors": ["symptoms"]},
            ],
        }
        segs = [_seg(text="patient reports headache")]
        flags = [
            {
                "type": "symptom_without_duration",
                "message": "Symptom mentioned without duration: headache",
                "severity": "warning",
            },
        ]
        note = build_clinical_note(segs, template, review_flags=flags)
        assert "Review Flags" in note
        assert "[WARNING]" in note

    def test_backward_compat_no_review_flags_param(self):
        """Existing callers without review_flags still work."""
        template = {
            "name": "Test Note",
            "format": "markdown",
            "sections": [
                {"title": "Findings", "extractors": ["symptoms"]},
            ],
        }
        segs = [_seg(text="patient reports headache")]
        note = build_clinical_note(segs, template)
        assert "headache" in note
        assert "Review Flags" not in note
