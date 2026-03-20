"""Tests for case ground-truth validation."""

from __future__ import annotations

import pytest

from app.case_validation import validate_ground_truth, validate_case_ground_truth


# ── valid cases ──────────────────────────────────────────────────────


class TestValidGroundTruth:
    def test_fully_canonical(self):
        gt = {
            "expected_hypotheses": ["Pneumonia"],
            "red_flags": ["dyspnea"],
            "key_findings": ["cough", "fever", "dyspnea"],
        }
        result = validate_ground_truth(gt)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_empty_red_flags_no_red_flag_findings(self):
        gt = {
            "expected_hypotheses": ["Urinary tract infection"],
            "red_flags": [],
            "key_findings": ["dysuria", "urinary frequency"],
        }
        result = validate_ground_truth(gt)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_minimal_valid(self):
        gt = {"key_findings": ["fever"]}
        result = validate_ground_truth(gt)
        assert result["valid"] is True

    def test_returns_required_keys(self):
        result = validate_ground_truth({})
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result


# ── warnings ─────────────────────────────────────────────────────────


class TestWarnings:
    def test_synonym_label_warns(self):
        gt = {"key_findings": ["shortness of breath"]}
        result = validate_ground_truth(gt)
        assert result["valid"] is True
        assert any("synonym" in w for w in result["warnings"])
        assert any("dyspnea" in w for w in result["warnings"])

    def test_unknown_key_finding_warns(self):
        gt = {"key_findings": ["exploding head syndrome"]}
        result = validate_ground_truth(gt)
        assert result["valid"] is True
        assert any("not in clinical terminology" in w for w in result["warnings"])

    def test_unknown_hypothesis_no_warning(self):
        """Hypotheses are condition names, not symptoms — no unknown warning."""
        gt = {"expected_hypotheses": ["Pneumonia"]}
        result = validate_ground_truth(gt)
        assert not any("not in clinical terminology" in w for w in result["warnings"])

    def test_missing_red_flags_for_red_flag_findings(self):
        gt = {
            "red_flags": [],
            "key_findings": ["dyspnea", "cough"],
        }
        result = validate_ground_truth(gt)
        assert result["valid"] is True
        assert any("red-flag terms" in w for w in result["warnings"])
        assert any("dyspnea" in w for w in result["warnings"])

    def test_no_red_flag_warning_when_red_flags_present(self):
        gt = {
            "red_flags": ["dyspnea"],
            "key_findings": ["dyspnea", "cough"],
        }
        result = validate_ground_truth(gt)
        assert not any("red-flag terms" in w for w in result["warnings"])

    def test_no_scoring_fields_warns(self):
        result = validate_ground_truth({})
        assert any("no scoring fields" in w for w in result["warnings"])

    def test_synonym_in_red_flags_warns(self):
        gt = {"red_flags": ["shortness of breath"]}
        result = validate_ground_truth(gt)
        assert any("synonym" in w for w in result["warnings"])

    def test_absent_red_flags_field_triggers_consistency_warning(self):
        gt = {"key_findings": ["chest pain"]}
        result = validate_ground_truth(gt)
        assert any("red-flag terms" in w for w in result["warnings"])


# ── errors ───────────────────────────────────────────────────────────


class TestErrors:
    def test_non_dict_ground_truth(self):
        result = validate_ground_truth("not a dict")
        assert result["valid"] is False
        assert any("must be a dict" in e for e in result["errors"])

    def test_non_list_field(self):
        gt = {"key_findings": "fever"}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("must be a list" in e for e in result["errors"])

    def test_non_string_label(self):
        gt = {"key_findings": [123]}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("must be a string" in e for e in result["errors"])

    def test_empty_label(self):
        gt = {"key_findings": [""]}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("empty" in e for e in result["errors"])

    def test_whitespace_only_label(self):
        gt = {"key_findings": ["  "]}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("empty" in e for e in result["errors"])

    def test_duplicate_after_canonicalization(self):
        gt = {"key_findings": ["shortness of breath", "dyspnea"]}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("duplicates" in e for e in result["errors"])

    def test_duplicate_same_label(self):
        gt = {"key_findings": ["fever", "fever"]}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("duplicates" in e for e in result["errors"])

    def test_duplicate_case_variants(self):
        gt = {"key_findings": ["Dyspnea", "dyspnea"]}
        result = validate_ground_truth(gt)
        assert result["valid"] is False
        assert any("duplicates" in e for e in result["errors"])


# ── validate_case_ground_truth ───────────────────────────────────────


class TestValidateCaseGroundTruth:
    def test_case_with_valid_gt(self):
        case = {
            "case_id": "test",
            "segments": [],
            "ground_truth": {
                "key_findings": ["fever"],
            },
        }
        result = validate_case_ground_truth(case)
        assert result["valid"] is True

    def test_case_without_gt(self):
        case = {"case_id": "test", "segments": []}
        result = validate_case_ground_truth(case)
        assert result["valid"] is True
        assert any("no ground_truth" in w for w in result["warnings"])

    def test_case_with_invalid_gt(self):
        case = {
            "case_id": "test",
            "segments": [],
            "ground_truth": {"key_findings": "not a list"},
        }
        result = validate_case_ground_truth(case)
        assert result["valid"] is False


# ── scoring integration ──────────────────────────────────────────────


class TestScoringIntegration:
    def test_gt_validation_in_score_output(self):
        from app.case_scoring import score_result_against_ground_truth

        result_bundle = {
            "case_id": "test",
            "ground_truth": {
                "expected_hypotheses": [],
                "red_flags": [],
                "key_findings": ["fever"],
            },
            "session": {
                "clinical_state": {
                    "symptoms": ["fever"],
                    "derived": {},
                },
            },
        }
        score = score_result_against_ground_truth(result_bundle)
        assert "gt_validation" in score
        assert score["gt_validation"]["valid"] is True

    def test_gt_validation_reports_synonym_warning(self):
        from app.case_scoring import score_result_against_ground_truth

        result_bundle = {
            "case_id": "test",
            "ground_truth": {
                "expected_hypotheses": [],
                "red_flags": [],
                "key_findings": ["shortness of breath"],
            },
            "session": {
                "clinical_state": {
                    "symptoms": ["dyspnea"],
                    "derived": {},
                },
            },
        }
        score = score_result_against_ground_truth(result_bundle)
        assert "gt_validation" in score
        assert any("synonym" in w for w in score["gt_validation"]["warnings"])
