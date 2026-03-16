"""Tests for problem evidence annotation."""

from __future__ import annotations

import pytest

from app.problem_evidence import annotate_problem_evidence
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _prob(prob_id: str = "prob_0001", obs_ids: list[str] | None = None,
          kind: str = "symptom_problem") -> dict:
    return {
        "id": prob_id,
        "title": "headache",
        "kind": kind,
        "status": "active",
        "onset": None,
        "observations": obs_ids or [],
        "encounters": ["enc_0001"],
        "actions": [],
        "documents": [],
        "priority": "normal",
    }


def _obs(obs_id: str = "obs_0001", finding_type: str = "symptom") -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": "headache",
        "seg_id": "seg_0001",
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": "patient has headache.",
        "category": "symptom" if finding_type == "symptom" else None,
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


# ── supporting observations ────────────────────────────────────────


class TestSupportingObservations:
    def test_mirrors_problem_observation_ids(self):
        problems = [_prob(obs_ids=["obs_0001", "obs_0002"])]
        observations = [_obs("obs_0001"), _obs("obs_0002")]
        result = annotate_problem_evidence(problems, observations)
        assert result[0]["supporting_observations"] == ["obs_0001", "obs_0002"]

    def test_single_observation(self):
        problems = [_prob(obs_ids=["obs_0001"])]
        observations = [_obs("obs_0001")]
        result = annotate_problem_evidence(problems, observations)
        assert result[0]["supporting_observations"] == ["obs_0001"]

    def test_empty_observations(self):
        problems = [_prob(obs_ids=[])]
        observations = []
        result = annotate_problem_evidence(problems, observations)
        assert result[0]["supporting_observations"] == []

    def test_validates_against_observation_layer(self):
        problems = [_prob(obs_ids=["obs_0001", "obs_9999"])]
        observations = [_obs("obs_0001")]
        result = annotate_problem_evidence(problems, observations)
        assert result[0]["supporting_observations"] == ["obs_0001"]

    def test_preserves_order(self):
        problems = [_prob(obs_ids=["obs_0003", "obs_0001", "obs_0002"])]
        observations = [_obs("obs_0001"), _obs("obs_0002"), _obs("obs_0003")]
        result = annotate_problem_evidence(problems, observations)
        assert result[0]["supporting_observations"] == [
            "obs_0003", "obs_0001", "obs_0002",
        ]


# ── conflicting observations ───────────────────────────────────────


class TestConflictingObservations:
    def test_defaults_to_empty_list(self):
        problems = [_prob(obs_ids=["obs_0001"])]
        observations = [_obs("obs_0001")]
        result = annotate_problem_evidence(problems, observations)
        assert result[0]["conflicting_observations"] == []

    def test_empty_problem_list(self):
        result = annotate_problem_evidence([], [])
        assert result == []


# ── preservation ────────────────────────────────────────────────────


class TestPreservation:
    def test_all_original_fields_present(self):
        prob = _prob(obs_ids=["obs_0001"])
        observations = [_obs("obs_0001")]
        [result] = annotate_problem_evidence([prob], observations)
        for key in prob:
            assert key in result
            assert result[key] == prob[key]

    def test_does_not_mutate_input(self):
        prob = _prob(obs_ids=["obs_0001"])
        original_keys = set(prob.keys())
        annotate_problem_evidence([prob], [_obs("obs_0001")])
        assert set(prob.keys()) == original_keys
        assert "supporting_observations" not in prob

    def test_multiple_problems(self):
        problems = [
            _prob("prob_0001", obs_ids=["obs_0001"]),
            _prob("prob_0002", obs_ids=["obs_0002"]),
        ]
        observations = [_obs("obs_0001"), _obs("obs_0002")]
        result = annotate_problem_evidence(problems, observations)
        assert len(result) == 2
        assert result[0]["supporting_observations"] == ["obs_0001"]
        assert result[1]["supporting_observations"] == ["obs_0002"]


# ── determinism ─────────────────────────────────────────────────────


class TestDeterminism:
    def test_deterministic(self):
        problems = [_prob(obs_ids=["obs_0001", "obs_0002"])]
        observations = [_obs("obs_0001"), _obs("obs_0002")]
        r1 = annotate_problem_evidence(problems, observations)
        r2 = annotate_problem_evidence(problems, observations)
        assert r1 == r2


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_problems_have_evidence_fields(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["problems"]:
            assert "supporting_observations" in prob
            assert "conflicting_observations" in prob
            assert isinstance(prob["supporting_observations"], list)
            assert isinstance(prob["conflicting_observations"], list)

    def test_supporting_matches_observations(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["problems"]:
            assert prob["supporting_observations"] == prob["observations"]

    def test_conflicting_empty_in_v1(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["problems"]:
            assert prob["conflicting_observations"] == []

    def test_existing_problem_fields_unchanged(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["problems"]:
            assert "id" in prob
            assert "title" in prob
            assert "kind" in prob
            assert "observations" in prob
            assert "encounters" in prob
