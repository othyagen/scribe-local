"""Tests for the Synthea import layer."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.synthea_import import (
    synthea_patient_to_case,
    build_segments,
    build_ground_truth,
    load_synthea_patients,
    convert_patients_to_cases,
)
from app.case_system import validate_case


# ── helpers ──────────────────────────────────────────────────────────


def _patient(**overrides) -> dict:
    base = {
        "id": "p1",
        "age": 65,
        "gender": "male",
        "conditions": ["pneumonia", "hypertension"],
        "symptoms": ["cough", "fever", "shortness of breath"],
        "duration_days": 3,
    }
    base.update(overrides)
    return base


def _minimal_patient() -> dict:
    return {"id": "min"}


# ── synthea_patient_to_case ──────────────────────────────────────────


class TestSyntheaPatientToCase:
    def test_case_id(self):
        case = synthea_patient_to_case(_patient())
        assert case["case_id"] == "synthea_p1"

    def test_title_includes_demographics(self):
        case = synthea_patient_to_case(_patient())
        assert "65-year-old" in case["title"]
        assert "male" in case["title"]

    def test_title_includes_symptoms(self):
        case = synthea_patient_to_case(_patient())
        assert "cough" in case["title"]

    def test_has_segments(self):
        case = synthea_patient_to_case(_patient())
        assert len(case["segments"]) >= 1

    def test_has_ground_truth(self):
        case = synthea_patient_to_case(_patient())
        gt = case["ground_truth"]
        assert "expected_hypotheses" in gt
        assert "key_findings" in gt
        assert "red_flags" in gt

    def test_has_config(self):
        case = synthea_patient_to_case(_patient())
        assert case["config"]["mode"] == "assist"

    def test_meta_source(self):
        case = synthea_patient_to_case(_patient())
        assert case["meta"]["source"] == "synthea"
        assert case["meta"]["original_id"] == "p1"

    def test_produces_valid_case(self):
        case = synthea_patient_to_case(_patient())
        validation = validate_case(case)
        assert validation["valid"] is True, validation["errors"]

    def test_does_not_mutate_input(self):
        patient = _patient()
        original = copy.deepcopy(patient)
        synthea_patient_to_case(patient)
        assert patient == original

    def test_deterministic(self):
        patient = _patient()
        c1 = synthea_patient_to_case(patient)
        c2 = synthea_patient_to_case(patient)
        assert c1 == c2


# ── missing fields tolerance ─────────────────────────────────────────


class TestMissingFields:
    def test_minimal_patient(self):
        case = synthea_patient_to_case(_minimal_patient())
        assert case["case_id"] == "synthea_min"
        validation = validate_case(case)
        assert validation["valid"] is True

    def test_no_symptoms(self):
        case = synthea_patient_to_case(_patient(symptoms=[]))
        assert len(case["segments"]) >= 1
        assert validate_case(case)["valid"] is True

    def test_no_conditions(self):
        case = synthea_patient_to_case(_patient(conditions=[]))
        gt = case["ground_truth"]
        assert gt["expected_hypotheses"] == []

    def test_no_age(self):
        p = _patient()
        del p["age"]
        case = synthea_patient_to_case(p)
        assert validate_case(case)["valid"] is True

    def test_no_gender(self):
        p = _patient()
        del p["gender"]
        case = synthea_patient_to_case(p)
        assert validate_case(case)["valid"] is True

    def test_no_duration(self):
        p = _patient()
        del p["duration_days"]
        case = synthea_patient_to_case(p)
        assert validate_case(case)["valid"] is True

    def test_empty_dict(self):
        case = synthea_patient_to_case({})
        assert case["case_id"] == "synthea_unknown"
        assert validate_case(case)["valid"] is True

    def test_no_id(self):
        p = _patient()
        del p["id"]
        case = synthea_patient_to_case(p)
        assert case["case_id"] == "synthea_unknown"


# ── build_segments ───────────────────────────────────────────────────


class TestBuildSegments:
    def test_opening_segment(self):
        segs = build_segments(_patient())
        assert "65-year-old male" in segs[0]["normalized_text"]

    def test_duration_in_opening(self):
        segs = build_segments(_patient())
        assert "3 days" in segs[0]["normalized_text"]

    def test_additional_symptoms(self):
        segs = build_segments(_patient())
        texts = [s["normalized_text"] for s in segs]
        assert any("shortness of breath" in t for t in texts)

    def test_history_conditions(self):
        segs = build_segments(_patient())
        texts = [s["normalized_text"] for s in segs]
        assert any("hypertension" in t for t in texts)

    def test_medications(self):
        segs = build_segments(_patient(medications=["aspirin"]))
        texts = [s["normalized_text"] for s in segs]
        assert any("aspirin" in t for t in texts)

    def test_allergies(self):
        segs = build_segments(_patient(allergies=["penicillin"]))
        texts = [s["normalized_text"] for s in segs]
        assert any("penicillin" in t for t in texts)

    def test_segment_structure(self):
        for seg in build_segments(_patient()):
            assert "seg_id" in seg
            assert "t0" in seg
            assert "t1" in seg
            assert "speaker_id" in seg
            assert "normalized_text" in seg
            assert seg["t1"] > seg["t0"]

    def test_sequential_timing(self):
        segs = build_segments(_patient())
        for i in range(1, len(segs)):
            assert segs[i]["t0"] >= segs[i - 1]["t0"]

    def test_empty_patient_gets_fallback(self):
        segs = build_segments({})
        assert len(segs) == 1
        assert "evaluation" in segs[0]["normalized_text"].lower()

    def test_week_duration(self):
        segs = build_segments(_patient(duration_days=7))
        assert "1 week" in segs[0]["normalized_text"]

    def test_weeks_duration(self):
        segs = build_segments(_patient(duration_days=14))
        assert "2 weeks" in segs[0]["normalized_text"]

    def test_single_day(self):
        segs = build_segments(_patient(duration_days=1))
        assert "1 day" in segs[0]["normalized_text"]


# ── build_ground_truth ──────────────────────────────────────────────


class TestBuildGroundTruth:
    def test_hypothesis_from_first_condition(self):
        gt = build_ground_truth(_patient())
        assert gt["expected_hypotheses"] == ["pneumonia"]

    def test_key_findings_from_symptoms(self):
        gt = build_ground_truth(_patient())
        assert gt["key_findings"] == ["cough", "fever", "shortness of breath"]

    def test_red_flags_detected(self):
        gt = build_ground_truth(_patient())
        assert "shortness of breath" in gt["red_flags"]

    def test_no_red_flags(self):
        gt = build_ground_truth(_patient(symptoms=["cough", "fever"]))
        assert gt["red_flags"] == []

    def test_chest_pain_red_flag(self):
        gt = build_ground_truth(_patient(symptoms=["chest pain"]))
        assert "chest pain" in gt["red_flags"]

    def test_empty_conditions(self):
        gt = build_ground_truth(_patient(conditions=[]))
        assert gt["expected_hypotheses"] == []

    def test_empty_symptoms(self):
        gt = build_ground_truth(_patient(symptoms=[]))
        assert gt["key_findings"] == []
        assert gt["red_flags"] == []


# ── load_synthea_patients ───────────────────────────────────────────


class TestLoadSyntheaPatients:
    def test_load_sample(self):
        sample = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"
        if sample.exists():
            patients = load_synthea_patients(sample)
            assert len(patients) >= 3
            for p in patients:
                assert "id" in p

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_synthea_patients("/nonexistent/path.json")

    def test_load_non_list(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="list"):
            load_synthea_patients(path)

    def test_load_from_tmp(self, tmp_path):
        data = [{"id": "t1", "symptoms": ["cough"]}]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        patients = load_synthea_patients(path)
        assert len(patients) == 1


# ── convert_patients_to_cases ───────────────────────────────────────


class TestConvertPatientsToCases:
    def test_converts_all(self):
        patients = [_patient(id="a"), _patient(id="b")]
        cases = convert_patients_to_cases(patients)
        assert len(cases) == 2
        assert cases[0]["case_id"] == "synthea_a"
        assert cases[1]["case_id"] == "synthea_b"

    def test_all_valid(self):
        patients = [_patient(id="a"), _minimal_patient()]
        for case in convert_patients_to_cases(patients):
            assert validate_case(case)["valid"] is True

    def test_empty_list(self):
        assert convert_patients_to_cases([]) == []

    def test_does_not_mutate_input(self):
        patients = [_patient()]
        original = copy.deepcopy(patients)
        convert_patients_to_cases(patients)
        assert patients == original


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_sample_file_round_trip(self):
        sample = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"
        if not sample.exists():
            pytest.skip("Sample file not found")
        patients = load_synthea_patients(sample)
        cases = convert_patients_to_cases(patients)
        for case in cases:
            validation = validate_case(case)
            assert validation["valid"] is True, (
                f"{case['case_id']}: {validation['errors']}"
            )
            assert case["meta"]["source"] == "synthea"
