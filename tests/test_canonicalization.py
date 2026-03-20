"""Tests for the canonical label alignment module."""

from __future__ import annotations

import copy

import pytest

from app.canonicalization import (
    canonicalize_label,
    canonicalize_labels,
    canonicalize_ground_truth,
    LABEL_SYNONYMS,
)


# ── canonicalize_label ──────────────────────────────────────────────


class TestCanonicalizeLabel:
    def test_known_synonym(self):
        assert canonicalize_label("shortness of breath") == "dyspnea"

    def test_sob_abbreviation(self):
        assert canonicalize_label("SOB") == "dyspnea"

    def test_painful_urination(self):
        assert canonicalize_label("painful urination") == "dysuria"

    def test_frequent_urination(self):
        assert canonicalize_label("frequent urination") == "urinary frequency"

    def test_chest_discomfort(self):
        assert canonicalize_label("chest discomfort") == "chest pain"

    def test_unknown_label_passthrough(self):
        assert canonicalize_label("headache") == "headache"

    def test_unknown_label_preserves_case(self):
        assert canonicalize_label("Headache") == "Headache"
        assert canonicalize_label("Pneumonia") == "Pneumonia"

    def test_whitespace_stripped(self):
        assert canonicalize_label("  shortness of breath  ") == "dyspnea"

    def test_case_insensitive(self):
        assert canonicalize_label("Shortness of Breath") == "dyspnea"
        assert canonicalize_label("SHORTNESS OF BREATH") == "dyspnea"

    def test_empty_string(self):
        assert canonicalize_label("") == ""

    def test_already_canonical(self):
        assert canonicalize_label("dyspnea") == "dyspnea"
        assert canonicalize_label("dysuria") == "dysuria"

    def test_breathlessness(self):
        assert canonicalize_label("breathlessness") == "dyspnea"

    def test_difficulty_breathing(self):
        assert canonicalize_label("difficulty breathing") == "dyspnea"

    def test_burning_urination(self):
        assert canonicalize_label("burning urination") == "dysuria"

    def test_chest_tightness(self):
        assert canonicalize_label("chest tightness") == "chest pain"


# ── canonicalize_labels ─────────────────────────────────────────────


class TestCanonicalizeLabels:
    def test_list_of_synonyms(self):
        result = canonicalize_labels(["shortness of breath", "chest pain"])
        assert result == ["dyspnea", "chest pain"]

    def test_preserves_order(self):
        result = canonicalize_labels(["cough", "SOB", "fever"])
        assert result == ["cough", "dyspnea", "fever"]

    def test_empty_list(self):
        assert canonicalize_labels([]) == []

    def test_mixed_known_unknown(self):
        result = canonicalize_labels(["painful urination", "rash", "SOB"])
        assert result == ["dysuria", "rash", "dyspnea"]

    def test_deterministic(self):
        labels = ["shortness of breath", "fever", "chest discomfort"]
        assert canonicalize_labels(labels) == canonicalize_labels(labels)


# ── canonicalize_ground_truth ───────────────────────────────────────


class TestCanonicalizeGroundTruth:
    def test_canonicalizes_all_fields(self):
        gt = {
            "expected_hypotheses": ["Pneumonia"],
            "red_flags": ["shortness of breath"],
            "key_findings": ["painful urination", "fever"],
        }
        result = canonicalize_ground_truth(gt)
        assert result["expected_hypotheses"] == ["Pneumonia"]  # no synonym, preserved
        assert result["red_flags"] == ["dyspnea"]  # synonym mapped
        assert result["key_findings"] == ["dysuria", "fever"]  # synonym + passthrough

    def test_does_not_mutate_input(self):
        gt = {
            "expected_hypotheses": ["Pneumonia"],
            "red_flags": ["shortness of breath"],
            "key_findings": ["cough"],
        }
        original = copy.deepcopy(gt)
        canonicalize_ground_truth(gt)
        assert gt == original

    def test_empty_ground_truth(self):
        result = canonicalize_ground_truth({})
        assert result == {}

    def test_missing_fields_preserved(self):
        gt = {"key_findings": ["fever"]}
        result = canonicalize_ground_truth(gt)
        assert result == {"key_findings": ["fever"]}
        assert "expected_hypotheses" not in result
        assert "red_flags" not in result

    def test_none_fields_preserved(self):
        gt = {"key_findings": None, "red_flags": ["SOB"]}
        result = canonicalize_ground_truth(gt)
        assert result["key_findings"] is None
        assert result["red_flags"] == ["dyspnea"]

    def test_extra_fields_preserved(self):
        gt = {
            "key_findings": ["cough"],
            "custom_field": "some_value",
        }
        result = canonicalize_ground_truth(gt)
        assert result["custom_field"] == "some_value"

    def test_returns_new_dict(self):
        gt = {"key_findings": ["fever"]}
        result = canonicalize_ground_truth(gt)
        assert result is not gt
        assert result["key_findings"] is not gt["key_findings"]


# ── synonym map properties ──────────────────────────────────────────


class TestSynonymMap:
    def test_keys_are_lowercase(self):
        for key in LABEL_SYNONYMS:
            assert key == key.lower(), f"key not lowercase: {key!r}"

    def test_values_are_lowercase(self):
        for value in LABEL_SYNONYMS.values():
            assert value == value.lower(), f"value not lowercase: {value!r}"

    def test_no_identity_mappings(self):
        for key, value in LABEL_SYNONYMS.items():
            assert key != value, f"identity mapping: {key!r}"

    def test_no_duplicate_keys(self):
        # dict enforces this, but verify explicitly
        assert len(LABEL_SYNONYMS) == len(set(LABEL_SYNONYMS.keys()))


# ── integration with scoring ────────────────────────────────────────


class TestScoringIntegration:
    def test_shortness_of_breath_red_flag_matches_dyspnea(self):
        """Ground truth 'shortness of breath' should match red flag label
        'Dyspnea' after canonicalization aligns ground truth to canonical form."""
        from app.case_scoring import score_result_against_ground_truth

        result_bundle = {
            "case_id": "test",
            "ground_truth": {
                "expected_hypotheses": [],
                "red_flags": ["shortness of breath"],
                "key_findings": ["shortness of breath", "cough"],
            },
            "session": {
                "clinical_state": {
                    "symptoms": ["dyspnea", "cough"],
                    "derived": {
                        "red_flags": [
                            {"label": "Dyspnea", "flag": "dyspnea_flag"},
                        ],
                    },
                },
            },
        }
        score = score_result_against_ground_truth(result_bundle)
        # Ground truth "shortness of breath" canonicalized to "dyspnea"
        # matches extracted symptom "dyspnea"
        assert score["key_findings"]["hit_rate"] == 1.0
        # Red flag: gt "shortness of breath" → "dyspnea" (canonicalized),
        # detector label "Dyspnea" → normalize → "dyspnea" → MATCH
        assert score["red_flags"]["hit_rate"] == 1.0

    def test_sob_key_finding_matches_dyspnea(self):
        from app.case_scoring import score_result_against_ground_truth

        result_bundle = {
            "case_id": "test",
            "ground_truth": {
                "expected_hypotheses": [],
                "red_flags": [],
                "key_findings": ["SOB"],
            },
            "session": {
                "clinical_state": {
                    "symptoms": ["dyspnea"],
                    "derived": {},
                },
            },
        }
        score = score_result_against_ground_truth(result_bundle)
        assert score["key_findings"]["hit_rate"] == 1.0
