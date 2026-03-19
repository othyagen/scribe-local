"""Tests for the Synthea suite runner integration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.synthea_import import load_synthea_patients, convert_patients_to_cases
from app.case_system import validate_case, run_case
from app.case_scoring import score_result_against_ground_truth
from app.case_analysis import analyze_case_results
from scripts.run_synthea_suite import run_synthea_suite, export_cases_to_yaml


_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"


# ── helpers ──────────────────────────────────────────────────────────


def _sample_patients() -> list[dict]:
    return [
        {
            "id": "t1",
            "age": 50,
            "gender": "female",
            "conditions": ["bronchitis"],
            "symptoms": ["cough", "fever"],
            "duration_days": 5,
        },
        {
            "id": "t2",
            "age": 30,
            "gender": "male",
            "conditions": [],
            "symptoms": ["headache"],
            "duration_days": 1,
        },
    ]


def _write_sample(tmp_path: Path, patients: list[dict]) -> Path:
    path = tmp_path / "patients.json"
    path.write_text(json.dumps(patients), encoding="utf-8")
    return path


# ── load and convert ─────────────────────────────────────────────────


class TestLoadAndConvert:
    def test_real_sample_loads(self):
        if not _SAMPLE_PATH.exists():
            pytest.skip("Sample file missing")
        patients = load_synthea_patients(_SAMPLE_PATH)
        assert len(patients) >= 3

    def test_converted_cases_valid(self):
        if not _SAMPLE_PATH.exists():
            pytest.skip("Sample file missing")
        patients = load_synthea_patients(_SAMPLE_PATH)
        cases = convert_patients_to_cases(patients)
        for case in cases:
            assert validate_case(case)["valid"] is True

    def test_custom_patients(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        patients = load_synthea_patients(path)
        cases = convert_patients_to_cases(patients)
        assert len(cases) == 2
        for case in cases:
            assert validate_case(case)["valid"] is True


# ── execution and scoring ────────────────────────────────────────────


class TestExecutionAndScoring:
    def test_cases_execute(self):
        if not _SAMPLE_PATH.exists():
            pytest.skip("Sample file missing")
        patients = load_synthea_patients(_SAMPLE_PATH)
        cases = convert_patients_to_cases(patients)
        for case in cases:
            result = run_case(case)
            assert result["validation"]["valid"] is True

    def test_scoring_produces_results(self):
        if not _SAMPLE_PATH.exists():
            pytest.skip("Sample file missing")
        patients = load_synthea_patients(_SAMPLE_PATH)
        cases = convert_patients_to_cases(patients)
        for case in cases:
            result = run_case(case)
            score = score_result_against_ground_truth(result)
            assert "hypotheses" in score
            assert "key_findings" in score
            assert "summary" in score

    def test_custom_patients_score(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        patients = load_synthea_patients(path)
        cases = convert_patients_to_cases(patients)
        for case in cases:
            result = run_case(case)
            score = score_result_against_ground_truth(result)
            assert score["has_ground_truth"] is True


# ── analysis ─────────────────────────────────────────────────────────


class TestAnalysis:
    def test_analysis_structure(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        analysis = suite["analysis"]
        assert "overall" in analysis
        assert "worst_cases" in analysis
        assert "hypothesis_failures" in analysis

    def test_overall_metrics(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        overall = suite["analysis"]["overall"]
        assert overall["total_cases"] == 2
        assert overall["scored_cases"] == 2
        assert isinstance(overall["avg_hypothesis_hit_rate"], float)
        assert isinstance(overall["avg_key_finding_hit_rate"], float)


# ── run_synthea_suite ────────────────────────────────────────────────


class TestRunSyntheaSuite:
    def test_suite_keys(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        assert set(suite.keys()) == {
            "patient_count", "case_count", "valid_case_count",
            "invalid_case_count", "cases", "scored_results", "analysis",
        }

    def test_counts(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        assert suite["patient_count"] == 2
        assert suite["case_count"] == 2
        assert suite["valid_case_count"] == 2
        assert suite["invalid_case_count"] == 0

    def test_scored_results_count(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        assert len(suite["scored_results"]) == 2

    def test_each_scored_result_has_score(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        for sr in suite["scored_results"]:
            assert "case_id" in sr
            assert "score" in sr
            assert "result_bundle" in sr

    def test_real_sample(self):
        if not _SAMPLE_PATH.exists():
            pytest.skip("Sample file missing")
        suite = run_synthea_suite(export_dir=None)
        assert suite["patient_count"] >= 3
        assert suite["valid_case_count"] >= 3
        assert len(suite["scored_results"]) >= 3


# ── export ───────────────────────────────────────────────────────────


class TestExport:
    def test_export_creates_files(self, tmp_path):
        patients = _sample_patients()
        cases = convert_patients_to_cases(patients)
        paths = export_cases_to_yaml(cases, tmp_path / "exported")
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".yaml"

    def test_exported_files_loadable(self, tmp_path):
        import yaml
        patients = _sample_patients()
        cases = convert_patients_to_cases(patients)
        paths = export_cases_to_yaml(cases, tmp_path / "exported")
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert validate_case(data)["valid"] is True

    def test_suite_with_export(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        export_dir = tmp_path / "cases" / "generated"
        suite = run_synthea_suite(sample_path=path, export_dir=export_dir)
        assert export_dir.exists()
        yaml_files = list(export_dir.glob("*.yaml"))
        assert len(yaml_files) == 2


# ── determinism and preservation ─────────────────────────────────────


class TestPreservation:
    def test_deterministic(self, tmp_path):
        path = _write_sample(tmp_path, _sample_patients())
        s1 = run_synthea_suite(sample_path=path, export_dir=None)
        s2 = run_synthea_suite(sample_path=path, export_dir=None)
        # Compare analysis (not full suite — result_bundle has session objects).
        assert s1["analysis"] == s2["analysis"]
        assert s1["patient_count"] == s2["patient_count"]

    def test_does_not_mutate_patients(self, tmp_path):
        patients = _sample_patients()
        original = copy.deepcopy(patients)
        cases = convert_patients_to_cases(patients)
        assert patients == original


# ── tolerance ────────────────────────────────────────────────────────


class TestTolerance:
    def test_empty_patients(self, tmp_path):
        path = _write_sample(tmp_path, [])
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        assert suite["patient_count"] == 0
        assert suite["case_count"] == 0
        assert suite["scored_results"] == []

    def test_minimal_patient(self, tmp_path):
        path = _write_sample(tmp_path, [{"id": "min"}])
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        assert suite["valid_case_count"] == 1
        assert len(suite["scored_results"]) == 1

    def test_patient_missing_fields(self, tmp_path):
        patients = [
            {"id": "a", "symptoms": ["cough"]},
            {"id": "b"},
        ]
        path = _write_sample(tmp_path, patients)
        suite = run_synthea_suite(sample_path=path, export_dir=None)
        assert suite["valid_case_count"] == 2
