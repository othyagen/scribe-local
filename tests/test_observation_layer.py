"""Tests for Layer 1: Observation records."""

from __future__ import annotations

import pytest

from app.observation_layer import build_observation_layer


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── empty / basic ────────────────────────────────────────────────


class TestEmpty:
    def test_empty_segments(self):
        assert build_observation_layer([], [], [], [], []) == []

    def test_no_findings(self):
        result = build_observation_layer(
            [_seg("hello world.")], [], [], [], [],
        )
        assert result == []

    def test_finding_not_in_segment(self):
        result = build_observation_layer(
            [_seg("hello world.")],
            symptoms=["headache"],
            negations=[],
            durations=[],
            medications=[],
        )
        assert result == []


# ── symptom observations ─────────────────────────────────────────


class TestSymptomObservations:
    def test_single_finding_one_segment(self):
        result = build_observation_layer(
            [_seg("patient has headache.")],
            symptoms=["headache"],
            negations=[],
            durations=[],
            medications=[],
        )
        assert len(result) == 1
        obs = result[0]
        assert obs["finding_type"] == "symptom"
        assert obs["value"] == "headache"
        assert obs["seg_id"] == "seg_0001"
        assert obs["speaker_id"] == "spk_0"
        assert obs["t_start"] == 0.0
        assert obs["t_end"] == 1.0
        assert obs["source_text"] == "patient has headache."

    def test_same_finding_multiple_segments(self):
        result = build_observation_layer(
            [
                _seg("headache started yesterday.", seg_id="seg_0001", t0=0.0, t1=2.0),
                _seg("headache is getting worse.", seg_id="seg_0002", t0=2.0, t1=4.0),
            ],
            symptoms=["headache"],
            negations=[],
            durations=[],
            medications=[],
        )
        assert len(result) == 2
        assert result[0]["seg_id"] == "seg_0001"
        assert result[1]["seg_id"] == "seg_0002"

    def test_multi_word_finding(self):
        result = build_observation_layer(
            [_seg("patient reports chest pain.")],
            symptoms=["chest pain"],
            negations=[],
            durations=[],
            medications=[],
        )
        assert len(result) == 1
        assert result[0]["value"] == "chest pain"

    def test_multiple_symptoms_in_same_segment(self):
        result = build_observation_layer(
            [_seg("headache and nausea reported.")],
            symptoms=["headache", "nausea"],
            negations=[],
            durations=[],
            medications=[],
        )
        symptom_obs = [o for o in result if o["finding_type"] == "symptom"]
        assert len(symptom_obs) == 2
        values = {o["value"] for o in symptom_obs}
        assert values == {"headache", "nausea"}


# ── observation IDs ──────────────────────────────────────────────


class TestObservationIds:
    def test_sequential_ids(self):
        result = build_observation_layer(
            [
                _seg("headache and nausea.", seg_id="seg_0001", t0=0.0, t1=2.0),
                _seg("fever too.", seg_id="seg_0002", t0=2.0, t1=4.0),
            ],
            symptoms=["headache", "nausea", "fever"],
            negations=[],
            durations=[],
            medications=[],
        )
        ids = [o["observation_id"] for o in result]
        assert ids[0] == "obs_0001"
        # All IDs should be sequential
        for i, obs_id in enumerate(ids):
            assert obs_id == f"obs_{i + 1:04d}"


# ── negation observations ────────────────────────────────────────


class TestNegationObservations:
    def test_negation_captured(self):
        result = build_observation_layer(
            [_seg("no fever reported.")],
            symptoms=[],
            negations=["No fever reported"],
            durations=[],
            medications=[],
        )
        neg_obs = [o for o in result if o["finding_type"] == "negation"]
        assert len(neg_obs) == 1
        assert "fever" in neg_obs[0]["value"].lower()

    def test_denies_negation(self):
        result = build_observation_layer(
            [_seg("denies chest pain.")],
            symptoms=[],
            negations=["Denies chest pain"],
            durations=[],
            medications=[],
        )
        neg_obs = [o for o in result if o["finding_type"] == "negation"]
        assert len(neg_obs) == 1


# ── duration observations ────────────────────────────────────────


class TestDurationObservations:
    def test_duration_captured(self):
        result = build_observation_layer(
            [_seg("headache for 3 days.")],
            symptoms=["headache"],
            negations=[],
            durations=["3 days"],
            medications=[],
        )
        dur_obs = [o for o in result if o["finding_type"] == "duration"]
        assert len(dur_obs) == 1
        assert dur_obs[0]["value"] == "3 days"


# ── medication observations ──────────────────────────────────────


class TestMedicationObservations:
    def test_medication_captured(self):
        result = build_observation_layer(
            [_seg("prescribed ibuprofen.")],
            symptoms=[],
            negations=[],
            durations=[],
            medications=["ibuprofen"],
        )
        med_obs = [o for o in result if o["finding_type"] == "medication"]
        assert len(med_obs) == 1
        assert med_obs[0]["value"] == "ibuprofen"


# ── ordering ─────────────────────────────────────────────────────


class TestOrdering:
    def test_ordered_by_t_start(self):
        result = build_observation_layer(
            [
                _seg("nausea reported.", seg_id="seg_0002", t0=3.0, t1=5.0),
                _seg("headache started.", seg_id="seg_0001", t0=0.0, t1=2.0),
            ],
            symptoms=["headache", "nausea"],
            negations=[],
            durations=[],
            medications=[],
        )
        t_starts = [o["t_start"] for o in result]
        assert t_starts == sorted(t_starts)


# ── multiple finding types in same segment ───────────────────────


class TestMixedTypes:
    def test_multiple_types_same_segment(self):
        result = build_observation_layer(
            [_seg("headache for 3 days, prescribed ibuprofen, no fever.")],
            symptoms=["headache"],
            negations=["No fever"],
            durations=["3 days"],
            medications=["ibuprofen"],
        )
        types = {o["finding_type"] for o in result}
        assert "symptom" in types
        assert "duration" in types
        assert "medication" in types
