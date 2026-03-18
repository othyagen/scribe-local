"""Tests for clinical input — structured answer ingestion."""

from __future__ import annotations

import pytest

from app.clinical_input import (
    ingest_structured_answers,
    _SUPPORTED_TYPES,
)
from app.clinical_state import build_clinical_state


# ── helpers ──────────────────────────────────────────────────────────


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _state_with_obs(count: int = 3):
    """Minimal state with a given number of existing observations."""
    return {
        "observations": [{"observation_id": f"obs_{i+1:04d}"} for i in range(count)],
    }


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_returns_dict_with_expected_keys(self):
        result = ingest_structured_answers(_state_with_obs(), [])
        assert set(result.keys()) == {"new_observations", "unparsed_answers"}

    def test_both_lists(self):
        result = ingest_structured_answers(_state_with_obs(), [])
        assert isinstance(result["new_observations"], list)
        assert isinstance(result["unparsed_answers"], list)

    def test_empty_answers(self):
        result = ingest_structured_answers(_state_with_obs(), [])
        assert result["new_observations"] == []
        assert result["unparsed_answers"] == []

    def test_empty_state(self):
        result = ingest_structured_answers({}, [
            {"type": "duration", "value": "3 days", "related": "headache"},
        ])
        assert len(result["new_observations"]) == 1
        assert result["new_observations"][0]["observation_id"] == "obs_0001"


# ── supported types ──────────────────────────────────────────────────


class TestSupportedTypes:
    def test_supported_types_defined(self):
        assert "duration" in _SUPPORTED_TYPES
        assert "severity" in _SUPPORTED_TYPES
        assert "allergy" in _SUPPORTED_TYPES
        assert "dosage" in _SUPPORTED_TYPES

    def test_unsupported_type_goes_to_unparsed(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "unknown_type", "value": "something"},
        ])
        assert result["new_observations"] == []
        assert len(result["unparsed_answers"]) == 1


# ── duration answers ─────────────────────────────────────────────────


class TestDuration:
    def test_duration_creates_observation(self):
        result = ingest_structured_answers(_state_with_obs(3), [
            {"type": "duration", "value": "3 days", "related": "headache"},
        ])
        obs = result["new_observations"]
        assert len(obs) == 1
        assert obs[0]["finding_type"] == "duration"
        assert obs[0]["value"] == "3 days"
        assert obs[0]["category"] is None
        assert obs[0]["attributes"]["related_symptom"] == "headache"
        assert obs[0]["source"] == "structured_answer"

    def test_duration_without_related(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "duration", "value": "2 weeks"},
        ])
        obs = result["new_observations"][0]
        assert obs["attributes"] == {}

    def test_duration_value_stripped(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "duration", "value": "  5 days  ", "related": "nausea"},
        ])
        assert result["new_observations"][0]["value"] == "5 days"


# ── severity answers ─────────────────────────────────────────────────


class TestSeverity:
    def test_severity_creates_symptom_observation(self):
        result = ingest_structured_answers(_state_with_obs(3), [
            {"type": "severity", "value": "moderate", "related": "headache"},
        ])
        obs = result["new_observations"]
        assert len(obs) == 1
        assert obs[0]["finding_type"] == "symptom"
        assert obs[0]["category"] == "symptom"
        assert obs[0]["value"] == "headache"
        assert obs[0]["attributes"]["severity"] == "moderate"

    def test_severity_without_related(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "severity", "value": "severe"},
        ])
        obs = result["new_observations"][0]
        assert obs["value"] == ""
        assert obs["attributes"]["severity"] == "severe"


# ── allergy answers ──────────────────────────────────────────────────


class TestAllergy:
    def test_allergy_creates_allergy_observation(self):
        result = ingest_structured_answers(_state_with_obs(3), [
            {"type": "allergy", "value": "penicillin"},
        ])
        obs = result["new_observations"]
        assert len(obs) == 1
        assert obs[0]["finding_type"] == "allergy"
        assert obs[0]["category"] == "allergy"
        assert obs[0]["value"] == "penicillin"

    def test_allergy_value_stripped(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "allergy", "value": "  sulfa drugs  "},
        ])
        assert result["new_observations"][0]["value"] == "sulfa drugs"


# ── dosage answers ───────────────────────────────────────────────────


class TestDosage:
    def test_dosage_creates_medication_observation(self):
        result = ingest_structured_answers(_state_with_obs(3), [
            {"type": "dosage", "value": "400 mg twice daily", "related": "ibuprofen"},
        ])
        obs = result["new_observations"]
        assert len(obs) == 1
        assert obs[0]["finding_type"] == "medication"
        assert obs[0]["category"] == "medication"
        assert obs[0]["value"] == "ibuprofen"
        assert obs[0]["attributes"]["dosage"] == "400 mg twice daily"

    def test_dosage_without_related(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "dosage", "value": "200 mg"},
        ])
        obs = result["new_observations"][0]
        assert obs["value"] == ""


# ── observation ID sequencing ────────────────────────────────────────


class TestIdSequencing:
    def test_ids_continue_from_existing(self):
        result = ingest_structured_answers(_state_with_obs(5), [
            {"type": "duration", "value": "3 days"},
            {"type": "allergy", "value": "penicillin"},
        ])
        ids = [o["observation_id"] for o in result["new_observations"]]
        assert ids == ["obs_0006", "obs_0007"]

    def test_ids_start_at_1_when_no_observations(self):
        result = ingest_structured_answers({}, [
            {"type": "allergy", "value": "aspirin"},
        ])
        assert result["new_observations"][0]["observation_id"] == "obs_0001"


# ── observation fields ───────────────────────────────────────────────


class TestObservationFields:
    def test_has_all_expected_fields(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "duration", "value": "3 days", "related": "headache"},
        ])
        obs = result["new_observations"][0]
        expected_fields = {
            "observation_id", "finding_type", "value", "seg_id",
            "speaker_id", "t_start", "t_end", "source_text",
            "source", "category", "attributes", "confidence",
        }
        assert set(obs.keys()) == expected_fields

    def test_temporal_fields_are_none(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "allergy", "value": "penicillin"},
        ])
        obs = result["new_observations"][0]
        assert obs["seg_id"] is None
        assert obs["speaker_id"] is None
        assert obs["t_start"] is None
        assert obs["t_end"] is None
        assert obs["source_text"] is None

    def test_confidence_is_none(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "allergy", "value": "penicillin"},
        ])
        assert result["new_observations"][0]["confidence"] is None


# ── unparsed answers ─────────────────────────────────────────────────


class TestUnparsed:
    def test_missing_type_goes_to_unparsed(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"value": "something"},
        ])
        assert result["new_observations"] == []
        assert len(result["unparsed_answers"]) == 1

    def test_empty_type_goes_to_unparsed(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "", "value": "something"},
        ])
        assert len(result["unparsed_answers"]) == 1

    def test_missing_value_goes_to_unparsed(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "duration"},
        ])
        assert len(result["unparsed_answers"]) == 1

    def test_empty_value_goes_to_unparsed(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "duration", "value": ""},
        ])
        assert len(result["unparsed_answers"]) == 1

    def test_mixed_valid_and_invalid(self):
        result = ingest_structured_answers(_state_with_obs(), [
            {"type": "duration", "value": "3 days"},
            {"type": "unknown", "value": "foo"},
            {"type": "allergy", "value": "penicillin"},
        ])
        assert len(result["new_observations"]) == 2
        assert len(result["unparsed_answers"]) == 1


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_state(self):
        state = _state_with_obs(3)
        original_count = len(state["observations"])
        ingest_structured_answers(state, [
            {"type": "allergy", "value": "penicillin"},
        ])
        assert len(state["observations"]) == original_count

    def test_does_not_mutate_answers(self):
        answers = [{"type": "duration", "value": "3 days", "related": "headache"}]
        original = dict(answers[0])
        ingest_structured_answers(_state_with_obs(), answers)
        assert answers[0] == original

    def test_deterministic(self):
        state = _state_with_obs(3)
        answers = [
            {"type": "duration", "value": "3 days", "related": "headache"},
            {"type": "allergy", "value": "penicillin"},
        ]
        r1 = ingest_structured_answers(state, answers)
        r2 = ingest_structured_answers(state, answers)
        assert r1 == r2


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_with_full_clinical_state(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        result = ingest_structured_answers(state, [
            {"type": "duration", "value": "3 days", "related": "headache"},
            {"type": "severity", "value": "moderate", "related": "headache"},
        ])
        assert len(result["new_observations"]) == 2
        assert result["unparsed_answers"] == []
        # IDs continue from existing observations
        existing_count = len(state["observations"])
        first_new_id = int(result["new_observations"][0]["observation_id"].split("_")[1])
        assert first_new_id == existing_count + 1

    def test_with_medication_state(self):
        state = build_clinical_state([
            _seg("prescribed ibuprofen.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        result = ingest_structured_answers(state, [
            {"type": "dosage", "value": "400 mg twice daily", "related": "ibuprofen"},
            {"type": "allergy", "value": "no known allergies"},
        ])
        assert len(result["new_observations"]) == 2

    def test_full_scenario(self):
        state = build_clinical_state([
            _seg("patient reports headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("prescribed ibuprofen.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ])
        answers = [
            {"type": "duration", "value": "3 days", "related": "headache"},
            {"type": "severity", "value": "severe", "related": "headache"},
            {"type": "dosage", "value": "400 mg", "related": "ibuprofen"},
            {"type": "allergy", "value": "penicillin"},
            {"type": "unknown", "value": "something"},
        ]
        result = ingest_structured_answers(state, answers)
        assert len(result["new_observations"]) == 4
        assert len(result["unparsed_answers"]) == 1
        # All new observations have source="structured_answer"
        for obs in result["new_observations"]:
            assert obs["source"] == "structured_answer"
