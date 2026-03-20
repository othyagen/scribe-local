"""Tests for the case scoring layer."""

from __future__ import annotations

import copy

import pytest

from app.case_scoring import (
    score_result_against_ground_truth,
    score_case_run,
    score_case_script_run,
    summarize_score,
)
from app.case_system import run_case, run_case_script


# ── helpers ──────────────────────────────────────────────────────────


def _minimal_case(**overrides) -> dict:
    base = {
        "case_id": "test_01",
        "segments": [
            {
                "seg_id": "seg_0001",
                "t0": 0.0,
                "t1": 3.0,
                "speaker_id": "spk_0",
                "normalized_text": "Patient has headache and nausea for 3 days.",
            },
        ],
    }
    base.update(overrides)
    return base


def _case_with_gt(**gt_overrides) -> dict:
    gt = {
        "expected_hypotheses": [],
        "red_flags": [],
        "key_findings": ["headache", "nausea"],
    }
    gt.update(gt_overrides)
    return _minimal_case(ground_truth=gt)


def _scripted_case() -> dict:
    return {
        "case_id": "scripted_01",
        "segments": [
            {
                "seg_id": "seg_0001",
                "t0": 0.0,
                "t1": 3.0,
                "speaker_id": "spk_0",
                "normalized_text": "Patient has headache and fever for 3 days.",
            },
            {
                "seg_id": "seg_0002",
                "t0": 3.0,
                "t1": 6.0,
                "speaker_id": "spk_0",
                "normalized_text": "Denies nausea. Prescribed ibuprofen.",
            },
        ],
        "config": {"mode": "assist", "update_strategy": "manual"},
        "ground_truth": {
            "expected_hypotheses": [],
            "red_flags": [],
            "key_findings": ["headache", "fever"],
        },
        "answer_script": [
            {"question_type": "duration", "value": "3 days", "related": "headache"},
        ],
    }


# ── score structure tests ────────────────────────────────────────────


class TestScoreStructure:
    def test_top_level_keys(self):
        result = run_case(_minimal_case())
        score = score_result_against_ground_truth(result)
        assert set(score.keys()) == {
            "case_id", "has_ground_truth", "gt_validation",
            "hypotheses", "red_flags", "key_findings", "summary",
        }

    def test_hypotheses_keys(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        hyp = score["hypotheses"]
        assert set(hyp.keys()) == {
            "expected", "present", "missing",
            "expected_count", "matched_count",
            "top_hypothesis", "top_hypothesis_expected",
            "expected_ranks", "hit_rate",
        }

    def test_red_flags_keys(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        rf = score["red_flags"]
        assert set(rf.keys()) == {
            "expected", "present", "missing",
            "expected_count", "matched_count", "hit_rate",
        }

    def test_key_findings_keys(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert set(kf.keys()) == {
            "expected", "present", "missing",
            "expected_count", "matched_count", "hit_rate",
        }

    def test_summary_keys(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        assert set(score["summary"].keys()) == {
            "hypothesis_hit_rate",
            "hypothesis_expected_count",
            "hypothesis_matched_count",
            "red_flag_hit_rate",
            "red_flag_expected_count",
            "red_flag_matched_count",
            "key_finding_hit_rate",
            "key_finding_expected_count",
            "key_finding_matched_count",
            "top_hypothesis_expected",
        }


# ── no ground truth ─────────────────────────────────────────────────


class TestNoGroundTruth:
    def test_has_ground_truth_false(self):
        result = run_case(_minimal_case())
        score = score_result_against_ground_truth(result)
        assert score["has_ground_truth"] is False

    def test_empty_lists(self):
        result = run_case(_minimal_case())
        score = score_result_against_ground_truth(result)
        assert score["hypotheses"]["expected"] == []
        assert score["red_flags"]["expected"] == []
        assert score["key_findings"]["expected"] == []

    def test_hit_rates_zero(self):
        result = run_case(_minimal_case())
        score = score_result_against_ground_truth(result)
        assert score["summary"]["hypothesis_hit_rate"] == 0.0
        assert score["summary"]["red_flag_hit_rate"] == 0.0
        assert score["summary"]["key_finding_hit_rate"] == 0.0

    def test_counts_zero(self):
        result = run_case(_minimal_case())
        score = score_result_against_ground_truth(result)
        assert score["summary"]["hypothesis_expected_count"] == 0
        assert score["summary"]["red_flag_expected_count"] == 0
        assert score["summary"]["key_finding_expected_count"] == 0


# ── key finding scoring ─────────────────────────────────────────────


class TestKeyFindingScoring:
    def test_found_findings(self):
        result = run_case(_case_with_gt(key_findings=["headache", "nausea"]))
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert "headache" in kf["present"]
        assert "nausea" in kf["present"]
        assert kf["missing"] == []
        assert kf["hit_rate"] == 1.0

    def test_missing_finding(self):
        result = run_case(_case_with_gt(key_findings=["headache", "rash"]))
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert "headache" in kf["present"]
        assert "rash" in kf["missing"]
        assert kf["hit_rate"] == 0.5

    def test_all_missing(self):
        result = run_case(_case_with_gt(key_findings=["rash", "edema"]))
        score = score_result_against_ground_truth(result)
        assert score["key_findings"]["hit_rate"] == 0.0

    def test_case_insensitive(self):
        result = run_case(_case_with_gt(key_findings=["Headache"]))
        score = score_result_against_ground_truth(result)
        assert score["key_findings"]["matched_count"] == 1

    def test_whitespace_tolerance(self):
        result = run_case(_case_with_gt(key_findings=["  headache  "]))
        score = score_result_against_ground_truth(result)
        assert score["key_findings"]["matched_count"] == 1

    def test_counts(self):
        result = run_case(_case_with_gt(key_findings=["headache", "nausea", "rash"]))
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert kf["expected_count"] == 3
        assert kf["matched_count"] == 2


# ── hypothesis scoring ──────────────────────────────────────────────


class TestHypothesisScoring:
    def test_no_expected_hypotheses(self):
        result = run_case(_case_with_gt(expected_hypotheses=[]))
        score = score_result_against_ground_truth(result)
        assert score["hypotheses"]["hit_rate"] == 0.0
        assert score["hypotheses"]["expected_count"] == 0

    def test_expected_ranks_none_when_missing(self):
        result = run_case(_case_with_gt(expected_hypotheses=["Nonexistent"]))
        score = score_result_against_ground_truth(result)
        assert score["hypotheses"]["expected_ranks"]["Nonexistent"] is None

    def test_top_hypothesis_expected_false_when_empty(self):
        result = run_case(_case_with_gt(expected_hypotheses=[]))
        score = score_result_against_ground_truth(result)
        assert score["hypotheses"]["top_hypothesis_expected"] is False

    def test_top_hypothesis_field(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        # May or may not have hypotheses; field should always be a string.
        assert isinstance(score["hypotheses"]["top_hypothesis"], str)


# ── red flag scoring ────────────────────────────────────────────────


class TestRedFlagScoring:
    def test_no_expected_red_flags(self):
        result = run_case(_case_with_gt(red_flags=[]))
        score = score_result_against_ground_truth(result)
        assert score["red_flags"]["hit_rate"] == 0.0
        assert score["red_flags"]["expected_count"] == 0

    def test_missing_red_flag(self):
        result = run_case(_case_with_gt(red_flags=["Sepsis"]))
        score = score_result_against_ground_truth(result)
        assert "Sepsis" in score["red_flags"]["missing"]

    def test_counts(self):
        result = run_case(_case_with_gt(red_flags=["A", "B"]))
        score = score_result_against_ground_truth(result)
        assert score["red_flags"]["expected_count"] == 2


# ── score_case_run / score_case_script_run ──────────────────────────


class TestScoreCaseRun:
    def test_returns_expected_keys(self):
        scored = score_case_run(_case_with_gt())
        assert set(scored.keys()) == {"case_id", "result_bundle", "score"}

    def test_case_id(self):
        scored = score_case_run(_case_with_gt())
        assert scored["case_id"] == "test_01"

    def test_score_has_structure(self):
        scored = score_case_run(_case_with_gt())
        assert "hypotheses" in scored["score"]
        assert "summary" in scored["score"]

    def test_does_not_mutate_input(self):
        case = _case_with_gt()
        original = copy.deepcopy(case)
        score_case_run(case)
        assert case == original


class TestScoreCaseScriptRun:
    def test_returns_expected_keys(self):
        scored = score_case_script_run(_scripted_case())
        assert set(scored.keys()) == {"case_id", "result_bundle", "score"}

    def test_case_id(self):
        scored = score_case_script_run(_scripted_case())
        assert scored["case_id"] == "scripted_01"

    def test_does_not_mutate_input(self):
        case = _scripted_case()
        original = copy.deepcopy(case)
        score_case_script_run(case)
        assert case == original


# ── summarize_score ─────────────────────────────────────────────────


class TestSummarizeScore:
    def test_keys(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        summary = summarize_score(score)
        assert set(summary.keys()) == {
            "case_id", "has_ground_truth",
            "hypothesis_hit_rate", "red_flag_hit_rate",
            "key_finding_hit_rate", "top_hypothesis_expected",
        }

    def test_values_match_score(self):
        result = run_case(_case_with_gt())
        score = score_result_against_ground_truth(result)
        summary = summarize_score(score)
        assert summary["case_id"] == score["case_id"]
        assert summary["has_ground_truth"] == score["has_ground_truth"]
        assert summary["hypothesis_hit_rate"] == score["summary"]["hypothesis_hit_rate"]

    def test_empty_score(self):
        summary = summarize_score({})
        assert summary["case_id"] == ""
        assert summary["has_ground_truth"] is False
        assert summary["hypothesis_hit_rate"] == 0.0


# ── determinism and preservation ─────────────────────────────────────


class TestPreservation:
    def test_deterministic(self):
        case = _case_with_gt(key_findings=["headache", "nausea"])
        r1 = run_case(case)
        r2 = run_case(case)
        s1 = score_result_against_ground_truth(r1)
        s2 = score_result_against_ground_truth(r2)
        assert s1 == s2

    def test_does_not_mutate_result(self):
        result = run_case(_case_with_gt())
        original_gt = copy.deepcopy(result["ground_truth"])
        score_result_against_ground_truth(result)
        assert result["ground_truth"] == original_gt


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_full_scoring_cycle(self):
        case = _scripted_case()
        scored = score_case_script_run(case)
        score = scored["score"]
        summary = summarize_score(score)

        assert score["has_ground_truth"] is True
        assert score["key_findings"]["expected_count"] == 2
        assert summary["key_finding_hit_rate"] >= 0.0
        assert isinstance(summary["top_hypothesis_expected"], bool)

    def test_score_with_no_gt(self):
        scored = score_case_run(_minimal_case())
        score = scored["score"]
        assert score["has_ground_truth"] is False
        summary = summarize_score(score)
        assert summary["hypothesis_hit_rate"] == 0.0
