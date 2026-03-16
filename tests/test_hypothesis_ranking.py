"""Tests for hypothesis ranking layer."""

from __future__ import annotations

import pytest

from app.hypothesis_ranking import get_strength_weight, rank_hypotheses
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _ev(obs_id: str, kind: str = "supporting", strength: str = "weak") -> dict:
    return {"observation_id": obs_id, "kind": kind, "strength": strength}


def _hyp(hyp_id: str, supporting: list[dict] | None = None,
         conflicting: list[dict] | None = None) -> dict:
    return {
        "id": hyp_id,
        "title": f"Condition {hyp_id}",
        "status": "candidate",
        "supporting_observations": supporting or [],
        "conflicting_observations": conflicting or [],
        "related_problems": [],
        "confidence": "low",
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


# ── weight mapping ─────────────────────────────────────────────────


class TestStrengthWeights:
    def test_weak_is_1(self):
        assert get_strength_weight("weak") == 1

    def test_moderate_is_2(self):
        assert get_strength_weight("moderate") == 2

    def test_strong_is_3(self):
        assert get_strength_weight("strong") == 3

    def test_unknown_is_0(self):
        assert get_strength_weight("unknown") == 0

    def test_empty_is_0(self):
        assert get_strength_weight("") == 0


# ── score calculation ──────────────────────────────────────────────


class TestScoreCalculation:
    def test_supporting_only(self):
        hyp = _hyp("hyp_0001", supporting=[
            _ev("obs_0001", strength="weak"),
            _ev("obs_0002", strength="strong"),
        ])
        [result] = rank_hypotheses([hyp])
        assert result["score"] == 4  # 1 + 3

    def test_conflicting_subtracted(self):
        hyp = _hyp("hyp_0001",
                    supporting=[_ev("obs_0001", strength="strong")],
                    conflicting=[_ev("obs_0002", "conflicting", "moderate")])
        [result] = rank_hypotheses([hyp])
        assert result["score"] == 1  # 3 - 2

    def test_empty_evidence_score_zero(self):
        hyp = _hyp("hyp_0001")
        [result] = rank_hypotheses([hyp])
        assert result["score"] == 0

    def test_negative_score_possible(self):
        hyp = _hyp("hyp_0001",
                    supporting=[_ev("obs_0001", strength="weak")],
                    conflicting=[_ev("obs_0002", "conflicting", "strong")])
        [result] = rank_hypotheses([hyp])
        assert result["score"] == -2  # 1 - 3

    def test_multiple_supporting(self):
        hyp = _hyp("hyp_0001", supporting=[
            _ev("obs_0001", strength="moderate"),
            _ev("obs_0002", strength="moderate"),
            _ev("obs_0003", strength="weak"),
        ])
        [result] = rank_hypotheses([hyp])
        assert result["score"] == 5  # 2 + 2 + 1


# ── ranking order ──────────────────────────────────────────────────


class TestRankingOrder:
    def test_higher_score_ranks_first(self):
        hyps = [
            _hyp("hyp_0001", supporting=[_ev("obs_0001", strength="weak")]),
            _hyp("hyp_0002", supporting=[_ev("obs_0002", strength="strong")]),
        ]
        result = rank_hypotheses(hyps)
        assert result[0]["id"] == "hyp_0002"
        assert result[0]["rank"] == 1
        assert result[1]["id"] == "hyp_0001"
        assert result[1]["rank"] == 2

    def test_equal_scores_preserve_input_order(self):
        hyps = [
            _hyp("hyp_0001", supporting=[_ev("obs_0001", strength="weak")]),
            _hyp("hyp_0002", supporting=[_ev("obs_0002", strength="weak")]),
        ]
        result = rank_hypotheses(hyps)
        assert result[0]["id"] == "hyp_0001"
        assert result[0]["rank"] == 1
        assert result[1]["id"] == "hyp_0002"
        assert result[1]["rank"] == 2

    def test_rank_starts_at_1(self):
        hyps = [_hyp("hyp_0001")]
        result = rank_hypotheses(hyps)
        assert result[0]["rank"] == 1

    def test_three_hypotheses_ranked(self):
        hyps = [
            _hyp("hyp_0001", supporting=[_ev("obs_0001", strength="weak")]),
            _hyp("hyp_0002", supporting=[
                _ev("obs_0002", strength="strong"),
                _ev("obs_0003", strength="moderate"),
            ]),
            _hyp("hyp_0003", supporting=[_ev("obs_0004", strength="moderate")]),
        ]
        result = rank_hypotheses(hyps)
        assert [h["id"] for h in result] == ["hyp_0002", "hyp_0003", "hyp_0001"]
        assert [h["rank"] for h in result] == [1, 2, 3]

    def test_empty_list(self):
        assert rank_hypotheses([]) == []


# ── preservation and determinism ────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_input(self):
        hyp = _hyp("hyp_0001", supporting=[_ev("obs_0001", strength="weak")])
        original_keys = set(hyp.keys())
        rank_hypotheses([hyp])
        assert set(hyp.keys()) == original_keys
        assert "score" not in hyp
        assert "rank" not in hyp

    def test_preserves_existing_fields(self):
        hyp = _hyp("hyp_0001", supporting=[_ev("obs_0001", strength="weak")])
        [result] = rank_hypotheses([hyp])
        assert result["id"] == hyp["id"]
        assert result["title"] == hyp["title"]
        assert result["status"] == hyp["status"]
        assert result["confidence"] == hyp["confidence"]
        assert result["related_problems"] == hyp["related_problems"]

    def test_deterministic(self):
        hyps = [
            _hyp("hyp_0001", supporting=[_ev("obs_0001", strength="strong")]),
            _hyp("hyp_0002", supporting=[_ev("obs_0002", strength="weak")]),
        ]
        r1 = rank_hypotheses(hyps)
        r2 = rank_hypotheses(hyps)
        assert r1 == r2


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_hypotheses_have_score_and_rank(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for hyp in state["hypotheses"]:
            assert "score" in hyp
            assert "rank" in hyp
            assert isinstance(hyp["score"], int)
            assert isinstance(hyp["rank"], int)
            assert hyp["rank"] >= 1

    def test_no_hypotheses_no_ranking(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["hypotheses"] == []

    def test_ranks_are_sequential(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        if state["hypotheses"]:
            ranks = [h["rank"] for h in state["hypotheses"]]
            assert ranks == list(range(1, len(ranks) + 1))
