"""Tests for symptom group layer."""

from __future__ import annotations

import pytest

from app.symptom_groups import (
    build_symptom_groups,
    _classify_temporal_bucket,
    _assign_systems,
    _load_symptom_systems,
)
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _obs(obs_id: str, value: str, seg_id: str = "seg_0001",
         duration: str | None = None) -> dict:
    attrs = {}
    if duration is not None:
        attrs["duration"] = duration
    return {
        "observation_id": obs_id,
        "finding_type": "symptom",
        "value": value,
        "seg_id": seg_id,
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": f"patient has {value}.",
        "category": "symptom",
        "attributes": attrs,
        "confidence": None,
    }


def _neg_obs(obs_id: str, value: str, seg_id: str = "seg_0001") -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": "negation",
        "value": value,
        "seg_id": seg_id,
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": f"no {value}.",
        "category": None,
        "attributes": {},
        "confidence": None,
    }


def _enc(enc_id: str, obs_ids: list[str]) -> dict:
    return {
        "id": enc_id,
        "observations": obs_ids,
    }


def _state(observations: list[dict], encounters: list[dict] | None = None) -> dict:
    return {
        "observations": observations,
        "encounters": encounters or [],
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


# ── temporal bucket ────────────────────────────────────────────────


class TestTemporalBucket:
    def test_acute_days(self):
        assert _classify_temporal_bucket("3 days") == "acute"

    def test_acute_1_day(self):
        assert _classify_temporal_bucket("1 day") == "acute"

    def test_subacute_weeks(self):
        assert _classify_temporal_bucket("2 weeks") == "subacute"

    def test_subacute_boundary(self):
        assert _classify_temporal_bucket("7 days") == "subacute"

    def test_chronic_months(self):
        assert _classify_temporal_bucket("3 months") == "chronic"

    def test_chronic_years(self):
        assert _classify_temporal_bucket("2 years") == "chronic"

    def test_unknown_none(self):
        assert _classify_temporal_bucket(None) == "unknown"

    def test_unknown_empty(self):
        assert _classify_temporal_bucket("") == "unknown"

    def test_keyword_today(self):
        assert _classify_temporal_bucket("today") == "acute"

    def test_keyword_yesterday(self):
        assert _classify_temporal_bucket("yesterday") == "acute"

    def test_1_month_subacute(self):
        assert _classify_temporal_bucket("1 month") == "subacute"

    def test_chronic_31_days(self):
        assert _classify_temporal_bucket("31 days") == "chronic"


# ── system assignment ──────────────────────────────────────────────


class TestSystemAssignment:
    def setup_method(self):
        self.systems = _load_symptom_systems()

    def test_respiratory_cough(self):
        assert "respiratory" in _assign_systems("cough", self.systems)

    def test_respiratory_shortness_of_breath(self):
        assert "respiratory" in _assign_systems("shortness of breath", self.systems)

    def test_cardiac_chest_pain(self):
        assert "cardiac" in _assign_systems("chest pain", self.systems)

    def test_gastrointestinal_nausea(self):
        assert "gastrointestinal" in _assign_systems("nausea", self.systems)

    def test_gastrointestinal_diarrhea(self):
        assert "gastrointestinal" in _assign_systems("diarrhea", self.systems)

    def test_neurological_headache(self):
        assert "neurological" in _assign_systems("headache", self.systems)

    def test_general_fever(self):
        assert "general" in _assign_systems("fever", self.systems)

    def test_fallback_to_general(self):
        assert _assign_systems("something unknown", self.systems) == ["general"]

    def test_case_insensitive(self):
        assert "respiratory" in _assign_systems("Cough", self.systems)


# ── grouping by system ─────────────────────────────────────────────


class TestGroupingBySystem:
    def test_respiratory_symptoms_grouped(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough", duration="3 days"),
                _obs("obs_0002", "dyspnea", duration="3 days"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        assert len(groups) == 1
        assert groups[0]["systems"] == ["respiratory"]
        assert set(groups[0]["observations"]) == {"obs_0001", "obs_0002"}

    def test_cardiac_symptoms_grouped(self):
        state = _state(
            observations=[
                _obs("obs_0001", "chest pain", duration="2 days"),
                _obs("obs_0002", "palpitations", duration="2 days"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        assert len(groups) == 1
        assert groups[0]["systems"] == ["cardiac"]

    def test_gastrointestinal_symptoms_grouped(self):
        state = _state(
            observations=[
                _obs("obs_0001", "nausea"),
                _obs("obs_0002", "vomiting"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        gi_groups = [g for g in groups if "gastrointestinal" in g["systems"]]
        assert len(gi_groups) == 1

    def test_different_systems_separate_groups(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough"),
                _obs("obs_0002", "chest pain"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        systems = [g["systems"][0] for g in groups]
        assert "respiratory" in systems
        assert "cardiac" in systems

    def test_fallback_general_system(self):
        state = _state(
            observations=[_obs("obs_0001", "something rare")],
            encounters=[_enc("enc_0001", ["obs_0001"])],
        )
        groups = build_symptom_groups(state)
        assert groups[0]["systems"] == ["general"]


# ── grouping by encounter ──────────────────────────────────────────


class TestGroupingByEncounter:
    def test_same_system_different_encounters_separate(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough"),
                _obs("obs_0002", "cough"),
            ],
            encounters=[
                _enc("enc_0001", ["obs_0001"]),
                _enc("enc_0002", ["obs_0002"]),
            ],
        )
        groups = build_symptom_groups(state)
        assert len(groups) == 2
        assert groups[0]["encounters"] == ["enc_0001"]
        assert groups[1]["encounters"] == ["enc_0002"]


# ── grouping by temporal bucket ────────────────────────────────────


class TestGroupingByTemporalBucket:
    def test_different_buckets_separate_groups(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough", duration="3 days"),
                _obs("obs_0002", "cough", duration="3 months"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        buckets = {g["temporal_bucket"] for g in groups}
        assert "acute" in buckets
        assert "chronic" in buckets


# ── negation exclusion ─────────────────────────────────────────────


class TestNonSymptomExclusion:
    def test_negation_observations_excluded(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough"),
                _neg_obs("obs_0002", "No fever"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        all_obs = []
        for g in groups:
            all_obs.extend(g["observations"])
        assert "obs_0002" not in all_obs

    def test_empty_when_no_symptoms(self):
        state = _state(
            observations=[_neg_obs("obs_0001", "No fever")],
            encounters=[_enc("enc_0001", ["obs_0001"])],
        )
        groups = build_symptom_groups(state)
        assert groups == []


# ── IDs and titles ─────────────────────────────────────────────────


class TestIDsAndTitles:
    def test_sequential_ids(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough"),
                _obs("obs_0002", "chest pain"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        groups = build_symptom_groups(state)
        ids = [g["id"] for g in groups]
        assert ids[0] == "grp_0001"
        if len(ids) > 1:
            assert ids[1] == "grp_0002"

    def test_title_format(self):
        state = _state(
            observations=[_obs("obs_0001", "cough", duration="3 days")],
            encounters=[_enc("enc_0001", ["obs_0001"])],
        )
        groups = build_symptom_groups(state)
        assert groups[0]["title"] == "acute respiratory symptom group"


# ── preservation and determinism ────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_state(self):
        observations = [_obs("obs_0001", "cough")]
        encounters = [_enc("enc_0001", ["obs_0001"])]
        state = _state(observations, encounters)
        original_obs_count = len(state["observations"])
        build_symptom_groups(state)
        assert len(state["observations"]) == original_obs_count
        assert "symptom_groups" not in state

    def test_deterministic(self):
        state = _state(
            observations=[
                _obs("obs_0001", "cough"),
                _obs("obs_0002", "nausea"),
            ],
            encounters=[_enc("enc_0001", ["obs_0001", "obs_0002"])],
        )
        r1 = build_symptom_groups(state)
        r2 = build_symptom_groups(state)
        assert r1 == r2

    def test_empty_observations(self):
        state = _state(observations=[], encounters=[])
        assert build_symptom_groups(state) == []


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_symptom_groups_in_clinical_state(self):
        state = build_clinical_state([
            _seg("patient has cough and fever for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        assert "symptom_groups" in state
        assert isinstance(state["symptom_groups"], list)

    def test_groups_have_required_fields(self):
        state = build_clinical_state([
            _seg("patient has cough and fever for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for grp in state["symptom_groups"]:
            assert grp["id"].startswith("grp_")
            assert "title" in grp
            assert grp["kind"] == "symptom_group"
            assert isinstance(grp["systems"], list)
            assert grp["temporal_bucket"] in ("acute", "subacute", "chronic", "unknown")
            assert isinstance(grp["observations"], list)
            assert isinstance(grp["encounters"], list)

    def test_no_groups_without_symptoms(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["symptom_groups"] == []

    def test_groups_reference_valid_observation_ids(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        valid_ids = {o["observation_id"] for o in state["observations"]}
        for grp in state["symptom_groups"]:
            for obs_id in grp["observations"]:
                assert obs_id in valid_ids
