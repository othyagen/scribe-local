"""Tests for the case analysis layer."""

from __future__ import annotations

import copy

import pytest

from app.case_analysis import (
    analyze_case_results,
    safe_get_score_fields,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_result(
    case_id: str = "test_01",
    hyp_rate: float = 0.5,
    rf_rate: float = 1.0,
    kf_rate: float = 0.67,
    missing_hyp: list | None = None,
    missing_rf: list | None = None,
    missing_kf: list | None = None,
    has_gt: bool = True,
    strategy: str = "",
) -> dict:
    result = {
        "case_id": case_id,
        "score": {
            "case_id": case_id,
            "has_ground_truth": has_gt,
            "hypotheses": {
                "missing": missing_hyp or [],
            },
            "red_flags": {
                "missing": missing_rf or [],
            },
            "key_findings": {
                "missing": missing_kf or [],
            },
            "summary": {
                "hypothesis_hit_rate": hyp_rate,
                "red_flag_hit_rate": rf_rate,
                "key_finding_hit_rate": kf_rate,
                "top_hypothesis_expected": False,
            },
        },
    }
    if strategy:
        result["adversarial"] = {"strategy": strategy}
    return result


# ── safe_get_score_fields ────────────────────────────────────────────


class TestSafeGetScoreFields:
    def test_full_result(self):
        r = _make_result(hyp_rate=0.5, kf_rate=0.8)
        fields = safe_get_score_fields(r)
        assert fields["case_id"] == "test_01"
        assert fields["hypothesis_hit_rate"] == 0.5
        assert fields["key_finding_hit_rate"] == 0.8
        assert fields["has_ground_truth"] is True

    def test_empty_result(self):
        fields = safe_get_score_fields({})
        assert fields["case_id"] == ""
        assert fields["hypothesis_hit_rate"] == 0.0
        assert fields["has_ground_truth"] is False
        assert fields["missing_hypotheses"] == []

    def test_missing_score(self):
        fields = safe_get_score_fields({"case_id": "x"})
        assert fields["case_id"] == "x"
        assert fields["hypothesis_hit_rate"] == 0.0

    def test_strategy_extracted(self):
        r = _make_result(strategy="noise_injection")
        fields = safe_get_score_fields(r)
        assert fields["strategy"] == "noise_injection"

    def test_no_strategy(self):
        fields = safe_get_score_fields(_make_result())
        assert fields["strategy"] == ""


# ── analyze_case_results structure ──────────────────────────────────


class TestAnalysisStructure:
    def test_top_level_keys(self):
        analysis = analyze_case_results([])
        assert set(analysis.keys()) == {
            "overall", "worst_cases", "strategy_breakdown",
            "hypothesis_failures", "red_flag_failures",
            "key_finding_failures", "score_distribution",
        }

    def test_overall_keys(self):
        analysis = analyze_case_results([])
        assert set(analysis["overall"].keys()) == {
            "total_cases", "scored_cases", "avg_score",
            "avg_hypothesis_hit_rate", "avg_red_flag_hit_rate",
            "avg_key_finding_hit_rate",
        }

    def test_distribution_buckets(self):
        analysis = analyze_case_results([_make_result()])
        dist = analysis["score_distribution"]
        assert len(dist) == 5
        for d in dist:
            assert "range" in d
            assert "count" in d


# ── empty input ─────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_list(self):
        analysis = analyze_case_results([])
        assert analysis["overall"]["total_cases"] == 0
        assert analysis["overall"]["scored_cases"] == 0
        assert analysis["overall"]["avg_score"] == 0.0
        assert analysis["worst_cases"] == []
        assert analysis["strategy_breakdown"] == []
        assert analysis["hypothesis_failures"] == []


# ── overall metrics ─────────────────────────────────────────────────


class TestOverallMetrics:
    def test_single_case(self):
        results = [_make_result(hyp_rate=1.0, rf_rate=1.0, kf_rate=1.0)]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["total_cases"] == 1
        assert analysis["overall"]["scored_cases"] == 1
        assert analysis["overall"]["avg_score"] == 1.0

    def test_multiple_cases(self):
        results = [
            _make_result(case_id="a", hyp_rate=1.0, rf_rate=1.0, kf_rate=1.0),
            _make_result(case_id="b", hyp_rate=0.0, rf_rate=0.0, kf_rate=0.0),
        ]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["scored_cases"] == 2
        assert analysis["overall"]["avg_score"] == 0.5

    def test_no_ground_truth_excluded(self):
        results = [
            _make_result(case_id="a", hyp_rate=1.0, rf_rate=1.0, kf_rate=1.0),
            _make_result(case_id="b", has_gt=False),
        ]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["total_cases"] == 2
        assert analysis["overall"]["scored_cases"] == 1
        assert analysis["overall"]["avg_score"] == 1.0


# ── worst cases ─────────────────────────────────────────────────────


class TestWorstCases:
    def test_ordered_by_score(self):
        results = [
            _make_result(case_id="good", hyp_rate=1.0, rf_rate=1.0, kf_rate=1.0),
            _make_result(case_id="bad", hyp_rate=0.0, rf_rate=0.0, kf_rate=0.0),
            _make_result(case_id="mid", hyp_rate=0.5, rf_rate=0.5, kf_rate=0.5),
        ]
        analysis = analyze_case_results(results)
        worst = analysis["worst_cases"]
        assert worst[0]["case_id"] == "bad"
        assert worst[1]["case_id"] == "mid"

    def test_max_five(self):
        results = [
            _make_result(case_id=f"case_{i}", hyp_rate=i * 0.1, rf_rate=0.0, kf_rate=0.0)
            for i in range(10)
        ]
        analysis = analyze_case_results(results)
        assert len(analysis["worst_cases"]) == 5

    def test_includes_missing_info(self):
        results = [_make_result(
            missing_hyp=["Pneumonia"],
            missing_kf=["cough"],
        )]
        analysis = analyze_case_results(results)
        w = analysis["worst_cases"][0]
        assert "Pneumonia" in w["missing_hypotheses"]
        assert "cough" in w["missing_key_findings"]


# ── strategy breakdown ──────────────────────────────────────────────


class TestStrategyBreakdown:
    def test_no_strategies(self):
        results = [_make_result()]
        analysis = analyze_case_results(results)
        assert analysis["strategy_breakdown"] == []

    def test_grouped_by_strategy(self):
        results = [
            _make_result(case_id="a", strategy="noise", hyp_rate=1.0, rf_rate=1.0, kf_rate=1.0),
            _make_result(case_id="b", strategy="noise", hyp_rate=0.0, rf_rate=0.0, kf_rate=0.0),
            _make_result(case_id="c", strategy="flip", hyp_rate=0.5, rf_rate=0.5, kf_rate=0.5),
        ]
        analysis = analyze_case_results(results)
        breakdown = {s["strategy"]: s for s in analysis["strategy_breakdown"]}
        assert breakdown["noise"]["count"] == 2
        assert breakdown["noise"]["avg_score"] == 0.5
        assert breakdown["flip"]["count"] == 1
        assert breakdown["flip"]["avg_score"] == 0.5

    def test_sorted_alphabetically(self):
        results = [
            _make_result(case_id="a", strategy="z_strat"),
            _make_result(case_id="b", strategy="a_strat"),
        ]
        analysis = analyze_case_results(results)
        names = [s["strategy"] for s in analysis["strategy_breakdown"]]
        assert names == sorted(names)


# ── failure frequency ───────────────────────────────────────────────


class TestFailureFrequency:
    def test_hypothesis_failures(self):
        results = [
            _make_result(case_id="a", missing_hyp=["Pneumonia"]),
            _make_result(case_id="b", missing_hyp=["Pneumonia", "PE"]),
            _make_result(case_id="c", missing_hyp=["PE"]),
        ]
        analysis = analyze_case_results(results)
        failures = analysis["hypothesis_failures"]
        # pneumonia: 2, pe: 2 — both should appear, sorted alpha for ties.
        items = {f["item"]: f["count"] for f in failures}
        assert items["pneumonia"] == 2
        assert items["pe"] == 2

    def test_key_finding_failures(self):
        results = [
            _make_result(case_id="a", missing_kf=["cough", "fever"]),
            _make_result(case_id="b", missing_kf=["cough"]),
        ]
        analysis = analyze_case_results(results)
        failures = analysis["key_finding_failures"]
        items = {f["item"]: f["count"] for f in failures}
        assert items["cough"] == 2
        assert items["fever"] == 1

    def test_max_five(self):
        results = [_make_result(missing_hyp=[f"h{i}" for i in range(10)])]
        analysis = analyze_case_results(results)
        assert len(analysis["hypothesis_failures"]) == 5

    def test_empty_when_no_failures(self):
        results = [_make_result(hyp_rate=1.0)]
        analysis = analyze_case_results(results)
        assert analysis["hypothesis_failures"] == []


# ── score distribution ──────────────────────────────────────────────


class TestScoreDistribution:
    def test_single_perfect_score(self):
        results = [_make_result(hyp_rate=1.0, rf_rate=1.0, kf_rate=1.0)]
        analysis = analyze_case_results(results)
        dist = analysis["score_distribution"]
        # Score 1.0 should be in 0.8-1.0 bucket.
        last_bucket = dist[-1]
        assert last_bucket["range"] == "0.8-1.0"
        assert last_bucket["count"] == 1

    def test_single_zero_score(self):
        results = [_make_result(hyp_rate=0.0, rf_rate=0.0, kf_rate=0.0)]
        analysis = analyze_case_results(results)
        dist = analysis["score_distribution"]
        first_bucket = dist[0]
        assert first_bucket["range"] == "0.0-0.2"
        assert first_bucket["count"] == 1

    def test_total_matches_scored(self):
        results = [
            _make_result(case_id="a", hyp_rate=0.1, rf_rate=0.1, kf_rate=0.1),
            _make_result(case_id="b", hyp_rate=0.9, rf_rate=0.9, kf_rate=0.9),
            _make_result(case_id="c", hyp_rate=0.5, rf_rate=0.5, kf_rate=0.5),
        ]
        analysis = analyze_case_results(results)
        total = sum(d["count"] for d in analysis["score_distribution"])
        assert total == 3


# ── determinism and preservation ─────────────────────────────────────


class TestPreservation:
    def test_deterministic(self):
        results = [
            _make_result(case_id="a", missing_hyp=["Pneumonia"]),
            _make_result(case_id="b", strategy="noise"),
        ]
        a1 = analyze_case_results(results)
        a2 = analyze_case_results(results)
        assert a1 == a2

    def test_does_not_mutate_input(self):
        results = [
            _make_result(case_id="a", missing_hyp=["Pneumonia"]),
        ]
        original = copy.deepcopy(results)
        analyze_case_results(results)
        assert results == original


# ── partial data tolerance ──────────────────────────────────────────


class TestTolerance:
    def test_missing_score_key(self):
        results = [{"case_id": "x"}]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["total_cases"] == 1
        assert analysis["overall"]["scored_cases"] == 0

    def test_empty_score(self):
        results = [{"case_id": "x", "score": {}}]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["total_cases"] == 1

    def test_none_score(self):
        results = [{"case_id": "x", "score": None}]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["total_cases"] == 1

    def test_mixed_valid_invalid(self):
        results = [
            _make_result(case_id="valid"),
            {"case_id": "invalid"},
            _make_result(case_id="also_valid"),
        ]
        analysis = analyze_case_results(results)
        assert analysis["overall"]["total_cases"] == 3
        assert analysis["overall"]["scored_cases"] == 2
