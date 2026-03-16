"""Tests for observation normalization layer."""

from __future__ import annotations

import pytest

from app.observation_normalization import normalize_observations
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _obs(finding_type: str, value: str, seg_id: str = "seg_0001",
         category: str | None = None) -> dict:
    return {
        "observation_id": "obs_0001",
        "finding_type": finding_type,
        "value": value,
        "seg_id": seg_id,
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": "test sentence.",
        "category": category,
        "attributes": {},
        "confidence": None,
    }


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── negation linking ────────────────────────────────────────────────


class TestNegationAttribute:
    def test_symptom_gets_negated_true(self):
        obs_list = [
            _obs("symptom", "chest pain", "seg_0001", "symptom"),
            _obs("negation", "No chest pain", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        symptom = [o for o in result if o["finding_type"] == "symptom"][0]
        assert symptom["attributes"]["negated"] is True

    def test_no_negation_no_attribute(self):
        obs_list = [_obs("symptom", "headache", "seg_0001", "symptom")]
        result = normalize_observations(obs_list)
        assert "negated" not in result[0]["attributes"]

    def test_negation_in_different_segment_not_linked(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("negation", "No fever", "seg_0002"),
        ]
        result = normalize_observations(obs_list)
        symptom = [o for o in result if o["finding_type"] == "symptom"][0]
        assert "negated" not in symptom["attributes"]


# ── duration linking ────────────────────────────────────────────────


class TestDurationAttribute:
    def test_symptom_gets_duration(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        symptom = [o for o in result if o["finding_type"] == "symptom"][0]
        assert symptom["attributes"]["duration"] == "3 days"

    def test_no_duration_no_attribute(self):
        obs_list = [_obs("symptom", "headache", "seg_0001", "symptom")]
        result = normalize_observations(obs_list)
        assert "duration" not in result[0]["attributes"]

    def test_duration_in_different_segment_not_linked(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("duration", "3 days", "seg_0002"),
        ]
        result = normalize_observations(obs_list)
        symptom = [o for o in result if o["finding_type"] == "symptom"][0]
        assert "duration" not in symptom["attributes"]

    def test_first_duration_per_segment_wins(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("duration", "3 days", "seg_0001"),
            _obs("duration", "2 weeks", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        symptom = [o for o in result if o["finding_type"] == "symptom"][0]
        assert symptom["attributes"]["duration"] == "3 days"


# ── combined ────────────────────────────────────────────────────────


class TestCombinedAttributes:
    def test_both_negation_and_duration(self):
        obs_list = [
            _obs("symptom", "chest pain", "seg_0001", "symptom"),
            _obs("negation", "No chest pain", "seg_0001"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        symptom = [o for o in result if o["finding_type"] == "symptom"][0]
        assert symptom["attributes"]["negated"] is True
        assert symptom["attributes"]["duration"] == "3 days"

    def test_multiple_symptoms_same_segment(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("symptom", "nausea", "seg_0001", "symptom"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        symptoms = [o for o in result if o["finding_type"] == "symptom"]
        assert all(s["attributes"]["duration"] == "3 days" for s in symptoms)


# ── preservation ────────────────────────────────────────────────────


class TestPreservation:
    def test_original_observations_preserved(self):
        obs_list = [
            _obs("symptom", "chest pain", "seg_0001", "symptom"),
            _obs("negation", "No chest pain", "seg_0001"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        assert len(result) == 3
        types = [o["finding_type"] for o in result]
        assert types == ["symptom", "negation", "duration"]

    def test_qualifier_observations_unchanged(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("negation", "No fever", "seg_0001"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        neg = [o for o in result if o["finding_type"] == "negation"][0]
        dur = [o for o in result if o["finding_type"] == "duration"][0]
        assert neg["attributes"] == {}
        assert dur["attributes"] == {}

    def test_does_not_mutate_input(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("negation", "No fever", "seg_0001"),
        ]
        original_attrs = dict(obs_list[0]["attributes"])
        normalize_observations(obs_list)
        assert obs_list[0]["attributes"] == original_attrs
        assert "negated" not in obs_list[0]["attributes"]

    def test_existing_attributes_preserved(self):
        obs = _obs("symptom", "headache", "seg_0001", "symptom")
        obs["attributes"]["custom"] = "value"
        result = normalize_observations([obs])
        assert result[0]["attributes"]["custom"] == "value"

    def test_all_original_fields_present(self):
        obs = _obs("symptom", "headache", "seg_0001", "symptom")
        [result] = normalize_observations([obs])
        for key in obs:
            assert key in result


# ── edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_list(self):
        assert normalize_observations([]) == []

    def test_deterministic(self):
        obs_list = [
            _obs("symptom", "headache", "seg_0001", "symptom"),
            _obs("negation", "No fever", "seg_0001"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        r1 = normalize_observations(obs_list)
        r2 = normalize_observations(obs_list)
        assert r1 == r2

    def test_only_qualifiers_no_symptom(self):
        obs_list = [
            _obs("negation", "No fever", "seg_0001"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        assert len(result) == 2
        assert result[0]["attributes"] == {}
        assert result[1]["attributes"] == {}

    def test_medication_not_enriched(self):
        obs_list = [
            _obs("medication", "ibuprofen", "seg_0001", "medication"),
            _obs("duration", "3 days", "seg_0001"),
        ]
        result = normalize_observations(obs_list)
        med = [o for o in result if o["finding_type"] == "medication"][0]
        assert "duration" not in med["attributes"]


# ── integration with clinical_state ─────────────────────────────────


class TestIntegration:
    def test_negated_symptom_in_clinical_state(self):
        state = build_clinical_state([
            _seg("no chest pain.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        symptom_obs = [o for o in state["observations"]
                       if o["finding_type"] == "symptom"
                       and "chest pain" in o["value"].lower()]
        neg_obs = [o for o in state["observations"]
                   if o["finding_type"] == "negation"]
        if symptom_obs and neg_obs:
            assert symptom_obs[0]["attributes"].get("negated") is True

    def test_duration_linked_in_clinical_state(self):
        state = build_clinical_state([
            _seg("headache for 3 days.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        symptom_obs = [o for o in state["observations"]
                       if o["finding_type"] == "symptom"
                       and o["value"].lower() == "headache"]
        dur_obs = [o for o in state["observations"]
                   if o["finding_type"] == "duration"]
        if symptom_obs and dur_obs:
            assert symptom_obs[0]["attributes"].get("duration") == "3 days"

    def test_all_observations_still_present(self):
        state = build_clinical_state([
            _seg("headache for 3 days. no fever.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        types = {o["finding_type"] for o in state["observations"]}
        assert "symptom" in types
