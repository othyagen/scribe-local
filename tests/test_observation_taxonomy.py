"""Tests for observation taxonomy enrichment."""

from __future__ import annotations

import pytest

from app.observation_taxonomy import (
    OBSERVATION_CATEGORIES,
    FINDING_TYPE_TO_CATEGORY,
    enrich_observations,
)
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _obs(finding_type: str, value: str = "test") -> dict:
    return {
        "observation_id": "obs_0001",
        "finding_type": finding_type,
        "value": value,
        "seg_id": "seg_0001",
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": "test sentence.",
    }


# ── TestCategories ──────────────────────────────────────────────────


class TestCategories:
    def test_has_17_categories(self):
        assert len(OBSERVATION_CATEGORIES) == 17

    def test_no_duplicates(self):
        as_list = list(OBSERVATION_CATEGORIES)
        assert len(as_list) == len(set(as_list))

    def test_all_lowercase(self):
        for cat in OBSERVATION_CATEGORIES:
            assert cat == cat.lower(), f"{cat!r} is not lowercase"

    def test_expected_categories_present(self):
        expected = {
            "symptom", "clinical_sign", "vital", "laboratory",
            "microbiology", "imaging", "waveform", "functional_test",
            "diagnosis", "risk_factor", "medication", "allergy",
            "family_history", "social_history", "device",
            "pregnancy_status", "administrative",
        }
        assert OBSERVATION_CATEGORIES == expected


# ── TestFindingTypeMapping ──────────────────────────────────────────


class TestFindingTypeMapping:
    def test_symptom_maps_to_symptom(self):
        assert FINDING_TYPE_TO_CATEGORY["symptom"] == "symptom"

    def test_medication_maps_to_medication(self):
        assert FINDING_TYPE_TO_CATEGORY["medication"] == "medication"

    def test_negation_maps_to_none(self):
        assert FINDING_TYPE_TO_CATEGORY["negation"] is None

    def test_duration_maps_to_none(self):
        assert FINDING_TYPE_TO_CATEGORY["duration"] is None

    def test_all_mapped_categories_in_taxonomy(self):
        for cat in FINDING_TYPE_TO_CATEGORY.values():
            if cat is not None:
                assert cat in OBSERVATION_CATEGORIES


# ── TestEnrichObservations ──────────────────────────────────────────


class TestEnrichObservations:
    def test_preserves_all_old_fields(self):
        obs = _obs("symptom", "headache")
        [enriched] = enrich_observations([obs])
        for key in obs:
            assert enriched[key] == obs[key]

    def test_adds_category(self):
        [enriched] = enrich_observations([_obs("symptom")])
        assert enriched["category"] == "symptom"

    def test_adds_attributes(self):
        [enriched] = enrich_observations([_obs("symptom")])
        assert enriched["attributes"] == {}

    def test_adds_confidence(self):
        [enriched] = enrich_observations([_obs("symptom")])
        assert enriched["confidence"] is None

    def test_negation_category_is_none(self):
        [enriched] = enrich_observations([_obs("negation")])
        assert enriched["category"] is None

    def test_duration_category_is_none(self):
        [enriched] = enrich_observations([_obs("duration")])
        assert enriched["category"] is None

    def test_medication_category(self):
        [enriched] = enrich_observations([_obs("medication")])
        assert enriched["category"] == "medication"

    def test_unknown_finding_type_category_is_none(self):
        [enriched] = enrich_observations([_obs("unknown_type")])
        assert enriched["category"] is None

    def test_does_not_mutate_input(self):
        obs = _obs("symptom")
        original_keys = set(obs.keys())
        enrich_observations([obs])
        assert set(obs.keys()) == original_keys
        assert "category" not in obs

    def test_empty_list(self):
        assert enrich_observations([]) == []

    def test_deterministic(self):
        obs_list = [_obs("symptom"), _obs("negation"), _obs("medication")]
        result1 = enrich_observations(obs_list)
        result2 = enrich_observations(obs_list)
        assert result1 == result2

    def test_multiple_observations(self):
        obs_list = [
            _obs("symptom", "headache"),
            _obs("negation", "No fever"),
            _obs("duration", "3 days"),
            _obs("medication", "ibuprofen"),
        ]
        enriched = enrich_observations(obs_list)
        assert len(enriched) == 4
        assert enriched[0]["category"] == "symptom"
        assert enriched[1]["category"] is None
        assert enriched[2]["category"] is None
        assert enriched[3]["category"] == "medication"


# ── TestIntegration ─────────────────────────────────────────────────


class TestIntegration:
    def test_clinical_state_observations_have_category(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        obs = state["observations"]
        assert len(obs) >= 1
        for o in obs:
            assert "category" in o
            assert "attributes" in o
            assert "confidence" in o

    def test_symptom_obs_has_category_symptom(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        symptom_obs = [o for o in state["observations"]
                       if o["finding_type"] == "symptom"]
        assert len(symptom_obs) >= 1
        for o in symptom_obs:
            assert o["category"] == "symptom"

    def test_negation_obs_has_category_none(self):
        state = build_clinical_state([
            _seg("denies fever.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        neg_obs = [o for o in state["observations"]
                   if o["finding_type"] == "negation"]
        assert len(neg_obs) >= 1
        for o in neg_obs:
            assert o["category"] is None

    def test_medication_obs_has_category_medication(self):
        state = build_clinical_state([
            _seg("prescribed ibuprofen 400 mg.", seg_id="seg_0001",
                 t0=0.0, t1=3.0),
        ])
        med_obs = [o for o in state["observations"]
                   if o["finding_type"] == "medication"]
        assert len(med_obs) >= 1
        for o in med_obs:
            assert o["category"] == "medication"

    def test_enriched_obs_still_have_observation_id(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for o in state["observations"]:
            assert o["observation_id"].startswith("obs_")
