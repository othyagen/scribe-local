"""Tests for mismatch explanation and aggregation module."""

from __future__ import annotations

import pytest

from app.mismatch_explainer import explain_mismatches, summarize_mismatches


# ── helpers ──────────────────────────────────────────────────────────


def _bundle(gt: dict, symptoms: list[str] | None = None,
            red_flags: list[dict] | None = None,
            hypotheses: list[dict] | None = None) -> dict:
    """Build a minimal result_bundle."""
    return {
        "case_id": "test",
        "ground_truth": gt,
        "session": {
            "clinical_state": {
                "symptoms": symptoms or [],
                "hypotheses": hypotheses or [],
                "derived": {
                    "red_flags": red_flags or [],
                },
            },
        },
    }


def _score(hyp_missing: list[str] | None = None,
           kf_missing: list[str] | None = None,
           rf_missing: list[str] | None = None) -> dict:
    """Build a minimal score dict with missing lists."""
    return {
        "hypotheses": {"missing": hyp_missing or []},
        "key_findings": {"missing": kf_missing or []},
        "red_flags": {"missing": rf_missing or []},
    }


# ── no mismatches ───────────────────────────────────────────────────


class TestNoMismatches:
    def test_all_matched(self):
        gt = {"key_findings": ["fever"], "red_flags": [], "expected_hypotheses": []}
        bundle = _bundle(gt, symptoms=["fever"])
        score = _score()
        result = explain_mismatches(bundle, score)
        assert result == []

    def test_empty_ground_truth(self):
        bundle = _bundle({})
        score = _score()
        result = explain_mismatches(bundle, score)
        assert result == []

    def test_none_ground_truth(self):
        bundle = {"case_id": "test", "session": {"clinical_state": {}}}
        score = _score()
        result = explain_mismatches(bundle, score)
        assert result == []


# ── not_detected ─────────────────────────────────────────────────────


class TestNotDetected:
    def test_key_finding_not_detected(self):
        gt = {"key_findings": ["fever"]}
        bundle = _bundle(gt, symptoms=["cough"])
        score = _score(kf_missing=["fever"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        assert result[0]["field"] == "key_findings"
        assert result[0]["label"] == "fever"
        assert result[0]["reason"] == "not_detected"

    def test_hypothesis_not_detected(self):
        gt = {"expected_hypotheses": ["Pneumonia"]}
        bundle = _bundle(gt, hypotheses=[])
        score = _score(hyp_missing=["Pneumonia"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        assert result[0]["field"] == "expected_hypotheses"
        assert result[0]["reason"] == "not_detected"
        assert "no hypothesis" in result[0]["detail"]

    def test_red_flag_not_detected(self):
        gt = {"red_flags": ["dyspnea"]}
        bundle = _bundle(gt, red_flags=[])
        score = _score(rf_missing=["dyspnea"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        assert result[0]["field"] == "red_flags"
        assert result[0]["reason"] == "not_detected"
        assert "no red_flag" in result[0]["detail"]

    def test_not_detected_with_other_symptoms(self):
        gt = {"key_findings": ["fever"]}
        bundle = _bundle(gt, symptoms=["cough", "headache"])
        score = _score(kf_missing=["fever"])
        result = explain_mismatches(bundle, score)
        assert result[0]["reason"] == "not_detected"
        assert "cough" in result[0]["detail"]


# ── canonical_mismatch ───────────────────────────────────────────────


class TestCanonicalMismatch:
    def test_synonym_not_found_after_canonicalization(self):
        gt = {"key_findings": ["shortness of breath"]}
        bundle = _bundle(gt, symptoms=["cough"])
        score = _score(kf_missing=["shortness of breath"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        assert result[0]["reason"] == "canonical_mismatch"
        assert result[0]["canonical"] == "dyspnea"
        assert "shortness of breath" in result[0]["detail"]
        assert "dyspnea" in result[0]["detail"]

    def test_synonym_red_flag_canonical_mismatch(self):
        gt = {"red_flags": ["chest discomfort"]}
        bundle = _bundle(gt, red_flags=[])
        score = _score(rf_missing=["chest discomfort"])
        result = explain_mismatches(bundle, score)
        assert result[0]["reason"] == "canonical_mismatch"
        assert result[0]["canonical"] == "chest pain"


# ── partial_overlap ──────────────────────────────────────────────────


class TestPartialOverlap:
    def test_substring_overlap(self):
        gt = {"key_findings": ["pain"]}
        bundle = _bundle(gt, symptoms=["chest pain", "abdominal pain"])
        score = _score(kf_missing=["pain"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        assert result[0]["reason"] == "partial_overlap"
        assert "chest pain" in result[0]["detail"] or "abdominal pain" in result[0]["detail"]

    def test_partial_overlap_red_flag(self):
        gt = {"red_flags": ["chest"]}
        bundle = _bundle(gt, red_flags=[{"label": "Chest pain", "flag": "chest_pain_flag"}])
        score = _score(rf_missing=["chest"])
        result = explain_mismatches(bundle, score)
        assert result[0]["reason"] == "partial_overlap"


# ── output structure ─────────────────────────────────────────────────


class TestOutputStructure:
    def test_required_keys(self):
        gt = {"key_findings": ["fever"]}
        bundle = _bundle(gt, symptoms=[])
        score = _score(kf_missing=["fever"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        entry = result[0]
        assert "field" in entry
        assert "label" in entry
        assert "canonical" in entry
        assert "reason" in entry
        assert "detail" in entry

    def test_reason_is_known_value(self):
        known_reasons = {"not_detected", "synonym_mismatch", "canonical_mismatch", "partial_overlap"}
        gt = {"key_findings": ["fever", "shortness of breath"]}
        bundle = _bundle(gt, symptoms=["cough"])
        score = _score(kf_missing=["fever", "shortness of breath"])
        result = explain_mismatches(bundle, score)
        for entry in result:
            assert entry["reason"] in known_reasons


# ── multiple fields ──────────────────────────────────────────────────


class TestMultipleFields:
    def test_mismatches_across_fields(self):
        gt = {
            "expected_hypotheses": ["Pneumonia"],
            "red_flags": ["dyspnea"],
            "key_findings": ["fever"],
        }
        bundle = _bundle(gt, symptoms=[], red_flags=[], hypotheses=[])
        score = _score(
            hyp_missing=["Pneumonia"],
            rf_missing=["dyspnea"],
            kf_missing=["fever"],
        )
        result = explain_mismatches(bundle, score)
        fields = {e["field"] for e in result}
        assert "expected_hypotheses" in fields
        assert "red_flags" in fields
        assert "key_findings" in fields

    def test_mixed_match_and_miss(self):
        gt = {"key_findings": ["fever", "cough"]}
        bundle = _bundle(gt, symptoms=["fever"])
        score = _score(kf_missing=["cough"])
        result = explain_mismatches(bundle, score)
        assert len(result) == 1
        assert result[0]["label"] == "cough"


# ── determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_identical_input_identical_output(self):
        gt = {"key_findings": ["fever", "shortness of breath"]}
        bundle = _bundle(gt, symptoms=["cough"])
        score = _score(kf_missing=["fever", "shortness of breath"])
        r1 = explain_mismatches(bundle, score)
        r2 = explain_mismatches(bundle, score)
        assert r1 == r2


# ── summarize_mismatches ─────────────────────────────────────────────


def _m(field: str, label: str, reason: str, canonical: str = "",
       detail: str = "") -> dict:
    """Build a mismatch entry for aggregation tests."""
    return {
        "field": field,
        "label": label,
        "canonical": canonical or label,
        "reason": reason,
        "detail": detail or f"{label} not found",
    }


class TestSummarizeMismatches:
    def test_empty_input(self):
        result = summarize_mismatches([])
        assert result["total_mismatches"] == 0
        assert result["cases_with_mismatches"] == 0
        assert result["cases_total"] == 0

    def test_no_mismatches(self):
        result = summarize_mismatches([[], [], []])
        assert result["total_mismatches"] == 0
        assert result["cases_with_mismatches"] == 0
        assert result["cases_total"] == 3

    def test_single_case_single_mismatch(self):
        mismatches = [[_m("key_findings", "fever", "not_detected")]]
        result = summarize_mismatches(mismatches)
        assert result["total_mismatches"] == 1
        assert result["cases_with_mismatches"] == 1
        assert result["by_reason"] == {"not_detected": 1}
        assert result["by_field"] == {"key_findings": 1}

    def test_counts_aggregate_across_cases(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected")],
            [_m("key_findings", "fever", "not_detected"),
             _m("red_flags", "dyspnea", "not_detected")],
            [],
        ]
        result = summarize_mismatches(mismatches)
        assert result["total_mismatches"] == 3
        assert result["cases_with_mismatches"] == 2
        assert result["cases_total"] == 3
        assert result["by_reason"]["not_detected"] == 3
        assert result["by_field"]["key_findings"] == 2
        assert result["by_field"]["red_flags"] == 1

    def test_top_missed_labels(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected"),
             _m("key_findings", "cough", "not_detected")],
            [_m("key_findings", "fever", "not_detected")],
            [_m("key_findings", "fever", "not_detected")],
        ]
        result = summarize_mismatches(mismatches)
        top = result["top_missed_labels"]
        assert top[0]["label"] == "fever"
        assert top[0]["count"] == 3
        assert top[1]["label"] == "cough"
        assert top[1]["count"] == 1

    def test_top_synonym_issues(self):
        mismatches = [
            [_m("key_findings", "shortness of breath", "canonical_mismatch",
                canonical="dyspnea")],
            [_m("key_findings", "shortness of breath", "canonical_mismatch",
                canonical="dyspnea")],
            [_m("key_findings", "chest discomfort", "synonym_mismatch",
                canonical="chest pain")],
            [_m("key_findings", "fever", "not_detected")],
        ]
        result = summarize_mismatches(mismatches)
        syn = result["top_synonym_issues"]
        assert syn[0]["label"] == "shortness of breath"
        assert syn[0]["count"] == 2
        assert syn[1]["label"] == "chest discomfort"
        assert syn[1]["count"] == 1
        # "fever" is not_detected, not a synonym issue
        labels = [s["label"] for s in syn]
        assert "fever" not in labels

    def test_top_reasons(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected"),
             _m("key_findings", "sob", "canonical_mismatch")],
            [_m("key_findings", "cough", "not_detected"),
             _m("red_flags", "chest", "partial_overlap")],
        ]
        result = summarize_mismatches(mismatches)
        top = result["top_reasons"]
        assert top[0]["label"] == "not_detected"
        assert top[0]["count"] == 2

    def test_top_n_limits(self):
        mismatches = [[
            _m("key_findings", f"label_{i}", "not_detected")
            for i in range(10)
        ]]
        result = summarize_mismatches(mismatches, top_n=3)
        assert len(result["top_missed_labels"]) == 3

    def test_top_n_alphabetical_tiebreak(self):
        mismatches = [
            [_m("key_findings", "beta", "not_detected"),
             _m("key_findings", "alpha", "not_detected")],
        ]
        result = summarize_mismatches(mismatches, top_n=5)
        top = result["top_missed_labels"]
        assert top[0]["label"] == "alpha"
        assert top[1]["label"] == "beta"

    def test_summary_keys(self):
        result = summarize_mismatches([[_m("key_findings", "fever", "not_detected")]])
        expected_keys = {
            "total_mismatches", "cases_with_mismatches", "cases_total",
            "by_reason", "by_field", "top_missed_labels",
            "top_synonym_issues", "top_reasons",
        }
        assert set(result.keys()) == expected_keys

    def test_deterministic(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected"),
             _m("key_findings", "sob", "canonical_mismatch")],
        ]
        r1 = summarize_mismatches(mismatches)
        r2 = summarize_mismatches(mismatches)
        assert r1 == r2
