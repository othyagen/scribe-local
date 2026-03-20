"""Tests for optional hypothesis prioritization layer."""

from __future__ import annotations

import copy

import pytest

from app.hypothesis_prioritization import (
    DANGEROUS_CONDITIONS,
    prioritize_hypotheses,
)


# ── helpers ──────────────────────────────────────────────────────────


def _hyp(
    title: str,
    rank: int = 1,
    hyp_id: str = "",
    supporting: list | None = None,
    conflicting: list | None = None,
    score: int = 0,
) -> dict:
    """Build a minimal hypothesis dict."""
    return {
        "id": hyp_id or f"hyp_{rank:04d}",
        "title": title,
        "status": "candidate",
        "supporting_observations": supporting or [],
        "conflicting_observations": conflicting or [],
        "related_problems": [],
        "confidence": "low",
        "score": score,
        "rank": rank,
    }


def _evidence(obs_id: str, strength: str = "weak") -> dict:
    """Build a minimal structured evidence dict."""
    return {
        "observation_id": obs_id,
        "kind": "supporting",
        "strength": strength,
    }


def _red_flag(label: str, evidence: list[str] | None = None) -> dict:
    """Build a minimal red flag dict."""
    return {
        "flag": label.replace(" ", "_") + "_flag",
        "label": label,
        "severity": "high",
        "evidence": evidence or [label.lower()],
    }


# ── empty input ──────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_hypotheses(self):
        result = prioritize_hypotheses([])
        assert result == []

    def test_empty_hypotheses_with_red_flags(self):
        result = prioritize_hypotheses([], [_red_flag("Chest pain")])
        assert result == []


# ── most_likely ──────────────────────────────────────────────────────


class TestMostLikely:
    def test_safe_rank_1_is_most_likely(self):
        hyps = [_hyp("Pneumonia", rank=1, supporting=[_evidence("fever")])]
        result = prioritize_hypotheses(hyps)
        assert len(result) == 1
        assert result[0]["priority_class"] == "most_likely"
        assert result[0]["rank"] == 1

    def test_reason_mentions_top_ranked(self):
        hyps = [_hyp("Pneumonia", rank=1, supporting=[_evidence("fever")])]
        result = prioritize_hypotheses(hyps)
        assert "top-ranked" in result[0]["reason"]


# ── must_not_miss ────────────────────────────────────────────────────


class TestMustNotMiss:
    def test_dangerous_with_evidence_is_must_not_miss(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2,
                 supporting=[_evidence("chest pain")]),
        ]
        result = prioritize_hypotheses(hyps)
        acs = [r for r in result if r["title"] == "Acute coronary syndrome"]
        assert len(acs) == 1
        assert acs[0]["priority_class"] == "must_not_miss"

    def test_dangerous_with_red_flag_is_must_not_miss(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Meningitis", rank=2),
        ]
        flags = [_red_flag("Sudden severe headache", ["headache", "meningitis"])]
        result = prioritize_hypotheses(hyps, flags)
        men = [r for r in result if r["title"] == "Meningitis"]
        assert men[0]["priority_class"] == "must_not_miss"

    def test_dangerous_without_evidence_or_flag_is_NOT_must_not_miss(self):
        """Key requirement: name alone is not sufficient."""
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2),  # No evidence, no flags
        ]
        result = prioritize_hypotheses(hyps)
        acs = [r for r in result if r["title"] == "Acute coronary syndrome"]
        assert acs[0]["priority_class"] == "less_likely"

    def test_must_not_miss_reason_includes_basis(self):
        hyps = [_hyp("Sepsis", rank=1, supporting=[_evidence("fever")])]
        result = prioritize_hypotheses(hyps)
        assert result[0]["priority_class"] == "must_not_miss"
        assert "supporting evidence" in result[0]["reason"]

    def test_must_not_miss_reason_includes_red_flag_basis(self):
        hyps = [_hyp("Meningitis", rank=2)]
        flags = [_red_flag("Neck stiffness", ["meningitis"])]
        result = prioritize_hypotheses(hyps, flags)
        men = result[0]
        assert men["priority_class"] == "must_not_miss"
        assert "red flag" in men["reason"]

    def test_dangerous_rank_1_with_evidence_is_must_not_miss_not_most_likely(self):
        """must_not_miss takes precedence over most_likely for rank 1."""
        hyps = [_hyp("Acute coronary syndrome", rank=1,
                      supporting=[_evidence("chest pain")])]
        result = prioritize_hypotheses(hyps)
        assert result[0]["priority_class"] == "must_not_miss"

    def test_case_insensitive_condition_matching(self):
        hyps = [_hyp("ACUTE CORONARY SYNDROME", rank=1,
                      supporting=[_evidence("chest pain")])]
        result = prioritize_hypotheses(hyps)
        assert result[0]["priority_class"] == "must_not_miss"


# ── less_likely ──────────────────────────────────────────────────────


class TestLessLikely:
    def test_non_dangerous_rank_2_is_less_likely(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Pharyngitis", rank=2, supporting=[_evidence("sore throat")]),
        ]
        result = prioritize_hypotheses(hyps)
        phar = [r for r in result if r["title"] == "Pharyngitis"]
        assert phar[0]["priority_class"] == "less_likely"


# ── ranking preservation ─────────────────────────────────────────────


class TestRankingPreservation:
    def test_original_hypotheses_not_mutated(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2,
                 supporting=[_evidence("chest pain")]),
        ]
        original = copy.deepcopy(hyps)
        prioritize_hypotheses(hyps)
        assert hyps == original

    def test_original_rank_preserved_in_output(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2,
                 supporting=[_evidence("chest pain")]),
        ]
        result = prioritize_hypotheses(hyps)
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2

    def test_output_order_matches_input_order(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2,
                 supporting=[_evidence("chest pain")]),
            _hyp("Migraine", rank=3, supporting=[_evidence("headache")]),
        ]
        result = prioritize_hypotheses(hyps)
        titles = [r["title"] for r in result]
        assert titles == ["Pneumonia", "Acute coronary syndrome", "Migraine"]


# ── both views coexist ───────────────────────────────────────────────


class TestBothViews:
    def test_prioritization_and_ranking_independent(self):
        """Simulate reading both state['hypotheses'] and
        state['hypothesis_prioritization'] — they coexist."""
        hyps = [
            _hyp("Pneumonia", rank=1, score=4,
                 supporting=[_evidence("cough"), _evidence("fever")]),
            _hyp("Acute coronary syndrome", rank=2, score=3,
                 supporting=[_evidence("chest pain")]),
        ]
        # Standard ranking view.
        assert hyps[0]["rank"] == 1
        assert hyps[0]["title"] == "Pneumonia"
        assert hyps[0]["score"] == 4

        # Prioritization view.
        prio = prioritize_hypotheses(hyps)
        assert prio[0]["priority_class"] == "most_likely"
        assert prio[1]["priority_class"] == "must_not_miss"

        # Both available simultaneously; ranking unchanged.
        assert hyps[0]["rank"] == 1
        assert hyps[0]["score"] == 4

    def test_one_to_one_correspondence(self):
        hyps = [
            _hyp("A", rank=1, supporting=[_evidence("x")]),
            _hyp("B", rank=2),
        ]
        prio = prioritize_hypotheses(hyps)
        assert len(prio) == len(hyps)
        for h, p in zip(hyps, prio):
            assert p["title"] == h["title"]
            assert p["rank"] == h["rank"]


# ── output structure ─────────────────────────────────────────────────


class TestOutputStructure:
    def test_required_keys(self):
        hyps = [_hyp("Pneumonia", rank=1, supporting=[_evidence("cough")])]
        result = prioritize_hypotheses(hyps)
        entry = result[0]
        assert "hypothesis_id" in entry
        assert "title" in entry
        assert "rank" in entry
        assert "priority_class" in entry
        assert "reason" in entry
        assert "evidence" in entry

    def test_priority_class_values(self):
        valid = {"most_likely", "must_not_miss", "less_likely"}
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2,
                 supporting=[_evidence("chest pain")]),
            _hyp("Migraine", rank=3),
        ]
        result = prioritize_hypotheses(hyps)
        for entry in result:
            assert entry["priority_class"] in valid


# ── determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_identical_input_identical_output(self):
        hyps = [
            _hyp("Pneumonia", rank=1, supporting=[_evidence("cough")]),
            _hyp("Acute coronary syndrome", rank=2,
                 supporting=[_evidence("chest pain")]),
        ]
        flags = [_red_flag("Chest pain")]
        r1 = prioritize_hypotheses(hyps, flags)
        r2 = prioritize_hypotheses(hyps, flags)
        assert r1 == r2


# ── dangerous conditions registry ───────────────────────────────────


class TestDangerousConditions:
    def test_registry_is_non_empty(self):
        assert len(DANGEROUS_CONDITIONS) > 0

    def test_all_keys_lowercase(self):
        for key in DANGEROUS_CONDITIONS:
            assert key == key.lower()

    def test_all_values_non_empty(self):
        for key, reason in DANGEROUS_CONDITIONS.items():
            assert len(reason) > 0, f"empty reason for {key}"
