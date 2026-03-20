"""Tests for mismatch explanation and aggregation module."""

from __future__ import annotations

import pytest

import copy

from app.clinical_terminology import CLINICAL_TERMS, _SYNONYM_TO_CANONICAL
from app.mismatch_explainer import (
    apply_suggestions,
    explain_mismatches,
    summarize_mismatches,
    suggest_improvements,
)


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


# ── suggest_improvements ────────────────────────────────────────────


class TestSuggestImprovements:
    def test_empty_summary(self):
        result = suggest_improvements({})
        assert result == []

    def test_no_mismatches(self):
        summary = summarize_mismatches([[], []])
        result = suggest_improvements(summary)
        assert result == []

    def test_not_detected_suggestion(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected"),
             _m("key_findings", "rash", "not_detected")],
        ]
        summary = summarize_mismatches(mismatches)
        result = suggest_improvements(summary)
        assert len(result) >= 1
        nd = [s for s in result if s["issue"] == "not_detected"]
        assert len(nd) == 1
        assert "fever" in nd[0]["affected_labels"]
        assert "rash" in nd[0]["affected_labels"]
        assert "extraction" in nd[0]["suggested_fix"].lower() or "symptoms.json" in nd[0]["suggested_fix"]

    def test_synonym_suggestion(self):
        mismatches = [
            [_m("key_findings", "shortness of breath", "canonical_mismatch",
                canonical="dyspnea")],
            [_m("key_findings", "chest discomfort", "synonym_mismatch",
                canonical="chest pain")],
        ]
        summary = summarize_mismatches(mismatches)
        result = suggest_improvements(summary)
        syn = [s for s in result if s["issue"] == "synonym_or_canonical_mismatch"]
        assert len(syn) == 1
        assert "shortness of breath" in syn[0]["affected_labels"]
        assert "chest discomfort" in syn[0]["affected_labels"]
        assert "synonym" in syn[0]["suggested_fix"].lower()

    def test_partial_overlap_suggestion(self):
        mismatches = [
            [_m("key_findings", "pain", "partial_overlap")],
        ]
        summary = summarize_mismatches(mismatches)
        result = suggest_improvements(summary)
        po = [s for s in result if s["issue"] == "partial_overlap"]
        assert len(po) == 1
        assert "pain" in po[0]["affected_labels"]

    def test_sorted_by_affected_count_desc(self):
        mismatches = [
            [_m("key_findings", "a", "not_detected"),
             _m("key_findings", "b", "not_detected"),
             _m("key_findings", "c", "not_detected"),
             _m("key_findings", "x", "canonical_mismatch")],
        ]
        summary = summarize_mismatches(mismatches)
        result = suggest_improvements(summary)
        if len(result) >= 2:
            assert len(result[0]["affected_labels"]) >= len(result[1]["affected_labels"])

    def test_output_structure(self):
        mismatches = [[_m("key_findings", "fever", "not_detected")]]
        summary = summarize_mismatches(mismatches)
        result = suggest_improvements(summary)
        for s in result:
            assert "issue" in s
            assert "suggested_fix" in s
            assert "affected_labels" in s
            assert isinstance(s["affected_labels"], list)

    def test_mixed_reasons(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected"),
             _m("key_findings", "sob", "canonical_mismatch",
                canonical="dyspnea"),
             _m("key_findings", "pain", "partial_overlap")],
        ]
        summary = summarize_mismatches(mismatches)
        result = suggest_improvements(summary)
        issues = {s["issue"] for s in result}
        assert "not_detected" in issues or "partial_overlap" in issues
        assert "synonym_or_canonical_mismatch" in issues

    def test_deterministic(self):
        mismatches = [
            [_m("key_findings", "fever", "not_detected"),
             _m("key_findings", "sob", "canonical_mismatch")],
        ]
        summary = summarize_mismatches(mismatches)
        r1 = suggest_improvements(summary)
        r2 = suggest_improvements(summary)
        assert r1 == r2


# ── apply_suggestions ───────────────────────────────────────────────


def _suggestion(issue: str, labels: list[str]) -> dict:
    """Build a minimal suggestion dict for apply_suggestions tests."""
    return {
        "issue": issue,
        "suggested_fix": "test fix",
        "affected_labels": labels,
    }


class TestApplySuggestions:
    @pytest.fixture(autouse=True)
    def _snapshot_terminology(self):
        """Save and restore CLINICAL_TERMS + reverse index after each test."""
        terms_backup = copy.deepcopy(CLINICAL_TERMS)
        index_backup = dict(_SYNONYM_TO_CANONICAL)
        yield
        CLINICAL_TERMS.clear()
        CLINICAL_TERMS.update(terms_backup)
        _SYNONYM_TO_CANONICAL.clear()
        _SYNONYM_TO_CANONICAL.update(index_backup)

    def test_empty_suggestions(self):
        result = apply_suggestions([])
        assert result["proposed_changes"] == []
        assert result["applied_changes"] == []
        assert result["skipped_changes"] == []

    def test_already_registered_synonym_skipped(self):
        result = apply_suggestions([
            _suggestion("synonym_or_canonical_mismatch",
                        ["shortness of breath"]),
        ])
        assert result["proposed_changes"] == []
        assert len(result["skipped_changes"]) == 1
        assert result["skipped_changes"][0]["reason"] == "already_registered"

    def test_already_registered_canonical_skipped(self):
        result = apply_suggestions([
            _suggestion("not_detected", ["fever"]),
        ])
        assert result["proposed_changes"] == []
        assert len(result["skipped_changes"]) == 1
        assert result["skipped_changes"][0]["reason"] == "already_registered"

    def test_unknown_label_no_match_skipped(self):
        result = apply_suggestions([
            _suggestion("not_detected", ["rash"]),
        ])
        assert result["proposed_changes"] == []
        assert len(result["skipped_changes"]) == 1
        assert result["skipped_changes"][0]["reason"] == "no_safe_mapping"

    def test_modifier_match_proposed(self):
        # "severe chest pain" → "chest pain" with modifier "severe"
        result = apply_suggestions([
            _suggestion("not_detected", ["severe chest pain"]),
        ])
        assert len(result["proposed_changes"]) == 1
        assert result["proposed_changes"][0]["canonical_target"] == "chest pain"
        assert result["proposed_changes"][0]["action"] == "add_synonym"

    def test_dry_run_does_not_apply(self):
        result = apply_suggestions([
            _suggestion("not_detected", ["severe chest pain"]),
        ], dry_run=True)
        assert len(result["proposed_changes"]) == 1
        assert result["applied_changes"] == []
        # Synonym not actually added.
        from app.clinical_terminology import get_canonical_label
        assert get_canonical_label("severe chest pain") == "severe chest pain"

    def test_apply_adds_synonym(self):
        result = apply_suggestions([
            _suggestion("not_detected", ["severe chest pain"]),
        ], dry_run=False)
        assert len(result["proposed_changes"]) == 1
        assert len(result["applied_changes"]) == 1
        assert result["applied_changes"][0]["canonical_target"] == "chest pain"
        # Synonym now resolves.
        from app.clinical_terminology import get_canonical_label
        assert get_canonical_label("severe chest pain") == "chest pain"

    def test_postfix_modifier_match(self):
        # "chest pain acute" → "chest pain" with modifier "acute"
        result = apply_suggestions([
            _suggestion("partial_overlap", ["chest pain acute"]),
        ])
        assert len(result["proposed_changes"]) == 1
        assert result["proposed_changes"][0]["canonical_target"] == "chest pain"

    def test_short_label_skipped(self):
        # Labels shorter than 4 chars never get synonym proposals.
        result = apply_suggestions([
            _suggestion("not_detected", ["sob"]),
        ])
        # "sob" is already a synonym of "dyspnea" → already_registered
        assert result["proposed_changes"] == []

    def test_ambiguous_multiple_matches_skipped(self):
        # "severe pain" contains "pain" which appears in both
        # "chest pain" and "abdominal pain" — ambiguous.
        result = apply_suggestions([
            _suggestion("not_detected", ["severe pain"]),
        ])
        # Neither "chest pain" nor "abdominal pain" would match
        # because "severe pain" doesn't end with or start with either.
        assert result["proposed_changes"] == []
        assert len(result["skipped_changes"]) == 1
        assert result["skipped_changes"][0]["reason"] == "no_safe_mapping"

    def test_single_word_modifier_only(self):
        # "very severe chest pain" has multi-word modifier → no match.
        result = apply_suggestions([
            _suggestion("not_detected", ["very severe chest pain"]),
        ])
        assert result["proposed_changes"] == []

    def test_output_structure(self):
        result = apply_suggestions([
            _suggestion("not_detected", ["severe chest pain", "rash"]),
        ])
        assert "proposed_changes" in result
        assert "applied_changes" in result
        assert "skipped_changes" in result
        for change in result["proposed_changes"]:
            assert "label" in change
            assert "action" in change
            assert "detail" in change
        for change in result["skipped_changes"]:
            assert "label" in change
            assert "reason" in change

    def test_deterministic(self):
        suggestions = [
            _suggestion("not_detected", ["severe chest pain", "rash"]),
            _suggestion("synonym_or_canonical_mismatch", ["sob"]),
        ]
        r1 = apply_suggestions(suggestions)
        r2 = apply_suggestions(suggestions)
        assert r1 == r2
