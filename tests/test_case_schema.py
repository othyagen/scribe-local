"""Tests for extended case schema validation and metadata extraction."""

from __future__ import annotations

import pytest

from app.case_schema import extract_case_metadata, validate_extended_schema


# ── helpers ────────────────────────────────────────────────────────


def _minimal_case():
    return {
        "case_id": "min_01",
        "segments": [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
             "speaker_id": "spk_0", "normalized_text": "Headache."},
        ],
    }


def _full_case():
    return {
        **_minimal_case(),
        "title": "Test case",
        "description": "A test case.",
        "classification": {
            "organ_systems": ["cardiovascular"],
            "presenting_complaints": ["chest pain"],
            "diagnosis_targets": {
                "icd10": [{"code": "I20.9", "display": "Angina"}],
                "icpc": [{"code": "K74", "display": "Angina pectoris"}],
                "snomed": [{"code": "194828000", "display": "Angina"}],
            },
        },
        "patient": {"age": 58, "sex": "male"},
        "meta": {
            "tags": ["cardiac", "emergency"],
            "difficulty": "moderate",
            "source": "synthetic",
        },
        "provenance": {"origin": "synthetic", "created": "2026-03-22"},
        "ground_truth": {"expected_hypotheses": [], "red_flags": []},
    }


# ── validate_extended_schema ──────────────────────────────────────


class TestValidateExtendedSchema:
    def test_full_classification_valid(self):
        result = validate_extended_schema(_full_case())
        assert result["errors"] == []

    def test_missing_classification_warns(self):
        case = _minimal_case()
        result = validate_extended_schema(case)
        assert any("classification" in w for w in result["warnings"])

    def test_missing_patient_warns(self):
        case = _minimal_case()
        result = validate_extended_schema(case)
        assert any("patient" in w for w in result["warnings"])

    def test_classification_not_dict_errors(self):
        case = _minimal_case()
        case["classification"] = "bad"
        result = validate_extended_schema(case)
        assert any("classification must be a dict" in e for e in result["errors"])

    def test_patient_not_dict_errors(self):
        case = _minimal_case()
        case["patient"] = "bad"
        result = validate_extended_schema(case)
        assert any("patient must be a dict" in e for e in result["errors"])

    def test_organ_systems_not_list_errors(self):
        case = _minimal_case()
        case["classification"] = {"organ_systems": "cardiovascular"}
        result = validate_extended_schema(case)
        assert any("organ_systems must be a list" in e for e in result["errors"])

    def test_organ_systems_empty_string_errors(self):
        case = _minimal_case()
        case["classification"] = {"organ_systems": [""]}
        result = validate_extended_schema(case)
        assert any("non-empty strings" in e for e in result["errors"])

    def test_presenting_complaints_not_list_errors(self):
        case = _minimal_case()
        case["classification"] = {"presenting_complaints": 42}
        result = validate_extended_schema(case)
        assert any("presenting_complaints must be a list" in e for e in result["errors"])

    def test_diagnosis_targets_not_dict_errors(self):
        case = _minimal_case()
        case["classification"] = {"diagnosis_targets": []}
        result = validate_extended_schema(case)
        assert any("diagnosis_targets must be a dict" in e for e in result["errors"])

    def test_diagnosis_target_missing_code_errors(self):
        case = _minimal_case()
        case["classification"] = {
            "diagnosis_targets": {
                "icd10": [{"display": "Angina"}],
            },
        }
        result = validate_extended_schema(case)
        assert any("requires code and display" in e for e in result["errors"])

    def test_diagnosis_target_unknown_system_warns(self):
        case = _minimal_case()
        case["classification"] = {
            "diagnosis_targets": {
                "custom": [{"code": "X", "display": "Y"}],
            },
        }
        result = validate_extended_schema(case)
        assert any("unknown system" in w for w in result["warnings"])

    def test_patient_age_not_number_errors(self):
        case = _minimal_case()
        case["patient"] = {"age": "old"}
        result = validate_extended_schema(case)
        assert any("patient.age must be a number" in e for e in result["errors"])

    def test_patient_sex_not_string_errors(self):
        case = _minimal_case()
        case["patient"] = {"sex": 42}
        result = validate_extended_schema(case)
        assert any("patient.sex must be a string" in e for e in result["errors"])

    def test_valid_patient_only(self):
        case = _minimal_case()
        case["patient"] = {"age": 30, "sex": "female"}
        case["classification"] = {}
        result = validate_extended_schema(case)
        assert result["errors"] == []

    def test_deterministic(self):
        case = _full_case()
        r1 = validate_extended_schema(case)
        r2 = validate_extended_schema(case)
        assert r1 == r2


# ── extract_case_metadata ─────────────────────────────────────────


class TestExtractCaseMetadata:
    def test_full_case_shape(self):
        meta = extract_case_metadata(_full_case())
        assert meta["case_id"] == "min_01"
        assert meta["title"] == "Test case"
        assert meta["origin"] == "synthetic"
        assert meta["difficulty"] == "moderate"
        assert meta["organ_systems"] == ["cardiovascular"]
        assert meta["presenting_complaints"] == ["chest pain"]
        assert meta["tags"] == ["cardiac", "emergency"]
        assert meta["patient_age"] == 58
        assert meta["patient_sex"] == "male"
        assert meta["icd10_codes"] == ["I20.9"]
        assert meta["icpc_codes"] == ["K74"]
        assert meta["snomed_codes"] == ["194828000"]
        assert meta["has_ground_truth"] is True
        assert meta["segment_count"] == 1

    def test_minimal_case_defaults(self):
        meta = extract_case_metadata(_minimal_case())
        assert meta["case_id"] == "min_01"
        assert meta["title"] == ""
        assert meta["origin"] == "unknown"
        assert meta["difficulty"] == "unspecified"
        assert meta["organ_systems"] == []
        assert meta["presenting_complaints"] == []
        assert meta["tags"] == []
        assert meta["patient_age"] is None
        assert meta["patient_sex"] is None
        assert meta["icd10_codes"] == []
        assert meta["has_ground_truth"] is False


# ── precedence rules ──────────────────────────────────────────────


class TestPrecedenceRules:
    def test_origin_from_provenance(self):
        """provenance.origin is canonical when present."""
        case = _minimal_case()
        case["provenance"] = {"origin": "synthea", "created": "2026-01-01"}
        case["meta"] = {"source": "imported"}
        meta = extract_case_metadata(case)
        assert meta["origin"] == "synthea"

    def test_origin_fallback_to_meta_source(self):
        """meta.source used only when provenance is absent."""
        case = _minimal_case()
        case["meta"] = {"source": "imported"}
        meta = extract_case_metadata(case)
        assert meta["origin"] == "imported"

    def test_origin_default_when_both_missing(self):
        """Default to 'unknown' when neither provenance nor meta.source."""
        meta = extract_case_metadata(_minimal_case())
        assert meta["origin"] == "unknown"

    def test_origin_provenance_present_but_no_origin_key(self):
        """provenance dict exists but has no origin key → fallback."""
        case = _minimal_case()
        case["provenance"] = {"created": "2026-01-01"}
        case["meta"] = {"source": "imported"}
        meta = extract_case_metadata(case)
        # provenance.origin is falsy (None) → falls through to meta.source
        assert meta["origin"] == "imported"

    def test_difficulty_from_meta(self):
        case = _minimal_case()
        case["meta"] = {"difficulty": "hard"}
        meta = extract_case_metadata(case)
        assert meta["difficulty"] == "hard"

    def test_difficulty_default(self):
        meta = extract_case_metadata(_minimal_case())
        assert meta["difficulty"] == "unspecified"

    def test_tags_from_meta_only(self):
        """Tags come from meta.tags only — no merge from other sources."""
        case = _minimal_case()
        case["meta"] = {"tags": ["cardiac"]}
        meta = extract_case_metadata(case)
        assert meta["tags"] == ["cardiac"]

    def test_tags_default_empty(self):
        meta = extract_case_metadata(_minimal_case())
        assert meta["tags"] == []

    def test_tags_not_mutated(self):
        """Returned tags list is a copy, not a reference."""
        case = _minimal_case()
        case["meta"] = {"tags": ["a", "b"]}
        meta = extract_case_metadata(case)
        meta["tags"].append("c")
        meta2 = extract_case_metadata(case)
        assert meta2["tags"] == ["a", "b"]

    def test_title_from_top_level(self):
        case = _minimal_case()
        case["title"] = "My Title"
        meta = extract_case_metadata(case)
        assert meta["title"] == "My Title"

    def test_title_default_empty(self):
        meta = extract_case_metadata(_minimal_case())
        assert meta["title"] == ""


# ── backward compatibility ────────────────────────────────────────


class TestBackwardCompat:
    def test_case_with_only_required_fields(self):
        """Minimal case with only case_id + segments extracts without error."""
        meta = extract_case_metadata(_minimal_case())
        assert meta["case_id"] == "min_01"
        assert meta["segment_count"] == 1

    def test_empty_classification_no_crash(self):
        case = _minimal_case()
        case["classification"] = {}
        meta = extract_case_metadata(case)
        assert meta["organ_systems"] == []

    def test_empty_patient_no_crash(self):
        case = _minimal_case()
        case["patient"] = {}
        meta = extract_case_metadata(case)
        assert meta["patient_age"] is None
