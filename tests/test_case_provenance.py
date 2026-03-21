"""Tests for case provenance and safety metadata."""

from __future__ import annotations

import pytest

from app.case_provenance import (
    VALID_ORIGINS,
    build_provenance,
    default_safety,
    validate_provenance,
)
from app.case_system import validate_case


# ── helpers ────────────────────────────────────────────────────────


def _case_with_provenance(prov, safety=None):
    case = {
        "case_id": "test_01",
        "segments": [{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient has headache.",
        }],
        "provenance": prov,
    }
    if safety is not None:
        case["safety"] = safety
    return case


def _derived_case(safety_overrides=None):
    safety = {
        "contains_real_data": False,
        "anonymized": True,
        "approved_for_evaluation": True,
    }
    if safety_overrides:
        safety.update(safety_overrides)
    return _case_with_provenance(
        {"origin": "derived", "created": "2026-03-22", "author": "timothy"},
        safety=safety,
    )


# ── build_provenance ──────────────────────────────────────────────


class TestBuildProvenance:
    def test_valid_origin(self):
        prov = build_provenance("synthetic")
        assert prov["origin"] == "synthetic"
        assert "created" in prov

    def test_all_origins_accepted(self):
        for origin in VALID_ORIGINS:
            prov = build_provenance(origin)
            assert prov["origin"] == origin

    def test_invalid_origin_raises(self):
        with pytest.raises(ValueError, match="invalid origin"):
            build_provenance("unknown")

    def test_kwargs_passed_through(self):
        prov = build_provenance("synthea", source_ref="synthea:p1", author="x")
        assert prov["source_ref"] == "synthea:p1"
        assert prov["author"] == "x"

    def test_created_override(self):
        prov = build_provenance("synthetic", created="2025-01-01")
        assert prov["created"] == "2025-01-01"


# ── default_safety ────────────────────────────────────────────────


class TestDefaultSafety:
    def test_shape(self):
        s = default_safety()
        assert s["contains_real_data"] is False
        assert s["approved_for_evaluation"] is True

    def test_no_anonymized_key(self):
        s = default_safety()
        assert "anonymized" not in s


# ── validate_provenance — valid cases ─────────────────────────────


class TestValidateProvenanceValid:
    def test_synthetic(self):
        case = _case_with_provenance({"origin": "synthetic", "created": "2026-03-22"})
        result = validate_provenance(case)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_synthea(self):
        case = _case_with_provenance(
            {"origin": "synthea", "created": "2026-03-22", "source_ref": "synthea:p1"}
        )
        result = validate_provenance(case)
        assert result["valid"] is True

    def test_derived_with_full_safety(self):
        case = _derived_case()
        result = validate_provenance(case)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_imported(self):
        case = _case_with_provenance({"origin": "imported", "created": "2026-03-22"})
        result = validate_provenance(case)
        assert result["valid"] is True


# ── validate_provenance — missing provenance ──────────────────────


class TestValidateProvenanceMissing:
    def test_missing_provenance_is_warning_not_error(self):
        case = {"case_id": "test_01", "segments": []}
        result = validate_provenance(case)
        assert result["valid"] is True
        assert any("missing provenance" in w for w in result["warnings"])

    def test_provenance_not_dict(self):
        case = _case_with_provenance("not_a_dict")
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("must be a dict" in e for e in result["errors"])


# ── validate_provenance — errors ──────────────────────────────────


class TestValidateProvenanceErrors:
    def test_invalid_origin(self):
        case = _case_with_provenance({"origin": "magic", "created": "2026-03-22"})
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("invalid" in e for e in result["errors"])

    def test_missing_origin(self):
        case = _case_with_provenance({"created": "2026-03-22"})
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("origin is required" in e for e in result["errors"])

    def test_missing_created(self):
        case = _case_with_provenance({"origin": "synthetic"})
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("created is required" in e for e in result["errors"])

    def test_derived_without_safety(self):
        case = _case_with_provenance(
            {"origin": "derived", "created": "2026-03-22"}
        )
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("safety block required" in e for e in result["errors"])

    def test_derived_not_anonymized(self):
        case = _derived_case({"anonymized": False})
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("anonymized" in e for e in result["errors"])

    def test_derived_not_approved(self):
        case = _derived_case({"approved_for_evaluation": False})
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("approved_for_evaluation" in e for e in result["errors"])

    def test_derived_contains_real_data(self):
        case = _derived_case({"contains_real_data": True})
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("contains_real_data" in e for e in result["errors"])

    def test_contains_real_data_without_anonymized(self):
        case = _case_with_provenance(
            {"origin": "imported", "created": "2026-03-22"},
            safety={"contains_real_data": True, "anonymized": False,
                    "approved_for_evaluation": True},
        )
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("contains_real_data" in e and "anonymized" in e
                    for e in result["errors"])

    def test_safety_not_dict(self):
        case = _case_with_provenance(
            {"origin": "derived", "created": "2026-03-22"}
        )
        case["safety"] = "not_a_dict"
        result = validate_provenance(case)
        assert result["valid"] is False
        assert any("safety must be a dict" in e for e in result["errors"])


# ── validate_case integration ─────────────────────────────────────


class TestValidateCaseIntegration:
    def test_case_without_provenance_still_valid(self):
        case = {
            "case_id": "test_01",
            "segments": [{
                "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
                "speaker_id": "spk_0",
                "normalized_text": "Headache.",
            }],
        }
        result = validate_case(case)
        assert result["valid"] is True
        assert any("missing provenance" in w for w in result["warnings"])

    def test_case_with_valid_provenance(self):
        case = _case_with_provenance({"origin": "synthetic", "created": "2026-03-22"})
        result = validate_case(case)
        assert result["valid"] is True
        assert not any("missing provenance" in w for w in result["warnings"])

    def test_case_with_invalid_provenance_fails(self):
        case = _case_with_provenance({"origin": "bad"})
        result = validate_case(case)
        assert result["valid"] is False

    def test_provenance_and_safety_are_known_fields(self):
        case = _case_with_provenance({"origin": "synthetic", "created": "2026-03-22"})
        result = validate_case(case)
        assert not any("unknown fields" in w for w in result["warnings"])


# ── synthea import integration ────────────────────────────────────


class TestSyntheaProvenance:
    def test_synthea_case_has_provenance(self):
        from app.synthea_import import synthea_patient_to_case

        patient = {
            "id": "p1", "age": 50, "gender": "male",
            "symptoms": ["cough"], "conditions": ["bronchitis"],
        }
        case = synthea_patient_to_case(patient)
        assert "provenance" in case
        prov = case["provenance"]
        assert prov["origin"] == "synthea"
        assert prov["source_ref"] == "synthea:p1"
        assert "created" in prov

    def test_synthea_case_validates(self):
        from app.synthea_import import synthea_patient_to_case

        patient = {
            "id": "p2", "age": 30, "symptoms": ["fever"],
            "conditions": ["flu"],
        }
        case = synthea_patient_to_case(patient)
        result = validate_case(case)
        assert result["valid"] is True
        assert not any("missing provenance" in w for w in result["warnings"])


# ── determinism ───────────────────────────────────────────────────


class TestDeterminism:
    def test_validate_deterministic(self):
        case = _derived_case()
        r1 = validate_provenance(case)
        r2 = validate_provenance(case)
        assert r1 == r2
