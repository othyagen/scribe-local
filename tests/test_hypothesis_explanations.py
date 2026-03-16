"""Tests for hypothesis explanation layer."""

from __future__ import annotations

import pytest

from app.hypothesis_explanations import build_hypothesis_explanations
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _ev(obs_id: str, kind: str = "supporting", strength: str = "weak") -> dict:
    return {"observation_id": obs_id, "kind": kind, "strength": strength}


def _obs(obs_id: str, value: str) -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": "symptom",
        "value": value,
        "seg_id": "seg_0001",
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": f"patient has {value}.",
        "category": "symptom",
        "attributes": {},
        "confidence": None,
    }


def _hyp(hyp_id: str, rank: int = 1, score: int = 2,
         supporting: list[dict] | None = None,
         conflicting: list[dict] | None = None) -> dict:
    return {
        "id": hyp_id,
        "title": f"Condition {hyp_id}",
        "status": "candidate",
        "supporting_observations": supporting or [],
        "conflicting_observations": conflicting or [],
        "related_problems": [],
        "confidence": "low",
        "score": score,
        "rank": rank,
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


# ── supporting evidence values ─────────────────────────────────────


class TestSupportingEvidence:
    def test_values_extracted(self):
        hyps = [_hyp("hyp_0001", supporting=[_ev("obs_0001")])]
        obs = [_obs("obs_0001", "headache")]
        result = build_hypothesis_explanations(hyps, obs)
        assert result[0]["explanation"]["supporting_evidence"] == ["headache"]

    def test_multiple_values(self):
        hyps = [_hyp("hyp_0001", supporting=[
            _ev("obs_0001"), _ev("obs_0002"),
        ])]
        obs = [_obs("obs_0001", "fever"), _obs("obs_0002", "cough")]
        result = build_hypothesis_explanations(hyps, obs)
        assert result[0]["explanation"]["supporting_evidence"] == [
            "fever", "cough",
        ]

    def test_missing_observation_skipped(self):
        hyps = [_hyp("hyp_0001", supporting=[_ev("obs_9999")])]
        result = build_hypothesis_explanations(hyps, [])
        assert result[0]["explanation"]["supporting_evidence"] == []


# ── conflicting evidence values ────────────────────────────────────


class TestConflictingEvidence:
    def test_empty_when_no_conflicts(self):
        hyps = [_hyp("hyp_0001")]
        result = build_hypothesis_explanations(hyps, [])
        assert result[0]["explanation"]["conflicting_evidence"] == []

    def test_values_extracted_when_present(self):
        hyps = [_hyp("hyp_0001",
                      conflicting=[_ev("obs_0001", "conflicting")])]
        obs = [_obs("obs_0001", "normal temperature")]
        result = build_hypothesis_explanations(hyps, obs)
        assert result[0]["explanation"]["conflicting_evidence"] == [
            "normal temperature",
        ]


# ── explanation object ─────────────────────────────────────────────


class TestExplanationObject:
    def test_has_all_fields(self):
        hyps = [_hyp("hyp_0001", rank=1, score=3,
                      supporting=[_ev("obs_0001")])]
        obs = [_obs("obs_0001", "headache")]
        result = build_hypothesis_explanations(hyps, obs)
        explanation = result[0]["explanation"]
        assert set(explanation.keys()) == {
            "summary", "supporting_evidence", "conflicting_evidence",
            "score", "rank",
        }

    def test_score_and_rank_copied(self):
        hyps = [_hyp("hyp_0001", rank=2, score=5)]
        result = build_hypothesis_explanations(hyps, [])
        assert result[0]["explanation"]["score"] == 5
        assert result[0]["explanation"]["rank"] == 2


# ── summary text ───────────────────────────────────────────────────


class TestSummaryText:
    def test_deterministic_format(self):
        hyps = [_hyp("hyp_0001", rank=1)]
        result = build_hypothesis_explanations(hyps, [])
        assert result[0]["explanation"]["summary"] == (
            "Hypothesis ranked #1 based on supporting clinical evidence."
        )

    def test_rank_reflected_in_summary(self):
        hyps = [_hyp("hyp_0001", rank=3)]
        result = build_hypothesis_explanations(hyps, [])
        assert "#3" in result[0]["explanation"]["summary"]


# ── preservation and determinism ────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_input(self):
        hyp = _hyp("hyp_0001", supporting=[_ev("obs_0001")])
        original_keys = set(hyp.keys())
        build_hypothesis_explanations([hyp], [_obs("obs_0001", "headache")])
        assert set(hyp.keys()) == original_keys
        assert "explanation" not in hyp

    def test_preserves_existing_fields(self):
        hyp = _hyp("hyp_0001", rank=1, score=2,
                    supporting=[_ev("obs_0001")])
        [result] = build_hypothesis_explanations(
            [hyp], [_obs("obs_0001", "headache")],
        )
        assert result["id"] == hyp["id"]
        assert result["title"] == hyp["title"]
        assert result["score"] == hyp["score"]
        assert result["rank"] == hyp["rank"]

    def test_deterministic(self):
        hyps = [_hyp("hyp_0001", supporting=[_ev("obs_0001")])]
        obs = [_obs("obs_0001", "headache")]
        r1 = build_hypothesis_explanations(hyps, obs)
        r2 = build_hypothesis_explanations(hyps, obs)
        assert r1 == r2

    def test_empty_hypotheses(self):
        assert build_hypothesis_explanations([], []) == []


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_hypotheses_have_explanation(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for hyp in state["hypotheses"]:
            assert "explanation" in hyp
            exp = hyp["explanation"]
            assert "summary" in exp
            assert "supporting_evidence" in exp
            assert "conflicting_evidence" in exp
            assert "score" in exp
            assert "rank" in exp

    def test_supporting_evidence_has_values(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for hyp in state["hypotheses"]:
            if hyp["supporting_observations"]:
                assert len(hyp["explanation"]["supporting_evidence"]) > 0

    def test_no_hypotheses_no_explanations(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["hypotheses"] == []
