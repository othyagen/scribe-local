"""Tests for diagnostic hypotheses layer."""

from __future__ import annotations

import pytest

from app.diagnostic_hypotheses import build_diagnostic_hypotheses
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


def _obs(obs_id: str, value: str, finding_type: str = "symptom") -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": "seg_0001",
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": f"patient has {value}.",
        "category": "symptom" if finding_type == "symptom" else None,
        "attributes": {},
        "confidence": None,
    }


def _state(hints, observations, problems=None):
    return {
        "diagnostic_hints": hints,
        "observations": observations,
        "problems": problems or [],
    }


# ── hypothesis generation ───────────────────────────────────────────


class TestHypothesisGeneration:
    def test_generates_from_hints(self):
        state = _state(
            hints=[{"condition": "Pneumonia", "snomed": "233604007",
                    "evidence": ["fever", "cough"]}],
            observations=[_obs("obs_0001", "fever"), _obs("obs_0002", "cough")],
        )
        result = build_diagnostic_hypotheses(state)
        assert len(result) == 1
        assert result[0]["title"] == "Pneumonia"
        assert result[0]["status"] == "candidate"

    def test_sequential_ids(self):
        state = _state(
            hints=[
                {"condition": "Pneumonia", "snomed": "x", "evidence": ["fever"]},
                {"condition": "Pharyngitis", "snomed": "y", "evidence": ["fever"]},
            ],
            observations=[_obs("obs_0001", "fever")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["id"] == "hyp_0001"
        assert result[1]["id"] == "hyp_0002"

    def test_no_hints_empty_result(self):
        state = _state(hints=[], observations=[])
        assert build_diagnostic_hypotheses(state) == []

    def test_empty_condition_skipped(self):
        state = _state(
            hints=[{"condition": "", "snomed": "x", "evidence": ["fever"]}],
            observations=[_obs("obs_0001", "fever")],
        )
        assert build_diagnostic_hypotheses(state) == []


# ── supporting observations ────────────────────────────────────────


class TestSupportingObservations:
    def test_linked_from_evidence(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": ["fever", "cough"]}],
            observations=[_obs("obs_0001", "fever"), _obs("obs_0002", "cough")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["supporting_observations"] == ["obs_0001", "obs_0002"]

    def test_missing_evidence_excluded(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": ["fever", "rash"]}],
            observations=[_obs("obs_0001", "fever")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["supporting_observations"] == ["obs_0001"]

    def test_case_insensitive_matching(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": ["Fever"]}],
            observations=[_obs("obs_0001", "fever")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["supporting_observations"] == ["obs_0001"]

    def test_no_duplicates(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": ["fever", "fever"]}],
            observations=[_obs("obs_0001", "fever")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["supporting_observations"] == ["obs_0001"]


# ── conflicting observations ───────────────────────────────────────


class TestConflictingObservations:
    def test_defaults_to_empty(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x", "evidence": ["fever"]}],
            observations=[_obs("obs_0001", "fever")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["conflicting_observations"] == []


# ── related problems ───────────────────────────────────────────────


class TestRelatedProblems:
    def test_linked_via_shared_observations(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x", "evidence": ["fever"]}],
            observations=[_obs("obs_0001", "fever")],
            problems=[{
                "id": "prob_0001",
                "observations": ["obs_0001"],
            }],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["related_problems"] == ["prob_0001"]

    def test_no_related_if_no_shared_obs(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x", "evidence": ["fever"]}],
            observations=[_obs("obs_0001", "fever")],
            problems=[{
                "id": "prob_0001",
                "observations": ["obs_9999"],
            }],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["related_problems"] == []

    def test_no_duplicate_problem_ids(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": ["fever", "cough"]}],
            observations=[_obs("obs_0001", "fever"), _obs("obs_0002", "cough")],
            problems=[{
                "id": "prob_0001",
                "observations": ["obs_0001", "obs_0002"],
            }],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["related_problems"] == ["prob_0001"]


# ── confidence logic ───────────────────────────────────────────────


class TestConfidence:
    def test_low_with_few_observations(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x", "evidence": ["fever"]}],
            observations=[_obs("obs_0001", "fever")],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["confidence"] == "low"

    def test_moderate_with_3_observations(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": ["fever", "cough", "dyspnea"]}],
            observations=[
                _obs("obs_0001", "fever"),
                _obs("obs_0002", "cough"),
                _obs("obs_0003", "dyspnea"),
            ],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["confidence"] == "moderate"

    def test_strong_with_5_observations(self):
        symptoms = ["fever", "cough", "dyspnea", "fatigue", "headache"]
        state = _state(
            hints=[{"condition": "Test", "snomed": "x",
                    "evidence": symptoms}],
            observations=[
                _obs(f"obs_{i+1:04d}", s) for i, s in enumerate(symptoms)
            ],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["confidence"] == "strong"

    def test_zero_observations_is_low(self):
        state = _state(
            hints=[{"condition": "Test", "snomed": "x", "evidence": ["rash"]}],
            observations=[],
        )
        result = build_diagnostic_hypotheses(state)
        assert result[0]["confidence"] == "low"


# ── determinism ─────────────────────────────────────────────────────


class TestDeterminism:
    def test_deterministic(self):
        state = _state(
            hints=[
                {"condition": "A", "snomed": "x", "evidence": ["fever"]},
                {"condition": "B", "snomed": "y", "evidence": ["cough"]},
            ],
            observations=[_obs("obs_0001", "fever"), _obs("obs_0002", "cough")],
        )
        r1 = build_diagnostic_hypotheses(state)
        r2 = build_diagnostic_hypotheses(state)
        assert r1 == r2


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_hypotheses_in_clinical_state(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        assert "hypotheses" in state
        assert isinstance(state["hypotheses"], list)

    def test_hypothesis_from_pharyngitis_hint(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        titles = [h["title"] for h in state["hypotheses"]]
        assert "Pharyngitis" in titles

    def test_hypothesis_structure(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for hyp in state["hypotheses"]:
            assert hyp["id"].startswith("hyp_")
            assert hyp["status"] == "candidate"
            assert isinstance(hyp["supporting_observations"], list)
            assert isinstance(hyp["conflicting_observations"], list)
            assert isinstance(hyp["related_problems"], list)
            assert hyp["confidence"] in ("low", "moderate", "strong")

    def test_no_hypotheses_without_hints(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["hypotheses"] == []
