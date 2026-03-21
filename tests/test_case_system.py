"""Tests for the case system — load, validate, and replay clinical cases."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from app.case_system import (
    load_case,
    load_all_cases,
    validate_case,
    run_case,
    run_case_script,
    extract_top_hypotheses,
    compare_result_to_ground_truth,
    _REQUIRED_FIELDS,
    _SEGMENT_REQUIRED_KEYS,
)


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
                "normalized_text": "Patient has headache and nausea.",
            },
        ],
    }
    base.update(overrides)
    return base


def _full_case() -> dict:
    return {
        "case_id": "full_01",
        "title": "Test case",
        "description": "A test case.",
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
        "config": {
            "mode": "assist",
            "update_strategy": "manual",
            "show_questions": True,
        },
        "ground_truth": {
            "expected_hypotheses": [],
            "red_flags": [],
            "key_findings": ["headache", "fever"],
        },
        "answer_script": [
            {"question_type": "duration", "value": "3 days", "related": "headache"},
        ],
        "meta": {
            "tags": ["test"],
            "difficulty": "easy",
            "source": "synthetic",
        },
    }


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


# ── validation tests ─────────────────────────────────────────────────


class TestValidateCase:
    def test_minimal_valid(self):
        result = validate_case(_minimal_case())
        assert result["valid"] is True
        assert result["errors"] == []

    def test_full_valid(self):
        result = validate_case(_full_case())
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_case_id(self):
        case = _minimal_case()
        del case["case_id"]
        result = validate_case(case)
        assert result["valid"] is False
        assert any("case_id" in e for e in result["errors"])

    def test_missing_segments(self):
        case = _minimal_case()
        del case["segments"]
        result = validate_case(case)
        assert result["valid"] is False
        assert any("segments" in e for e in result["errors"])

    def test_empty_segments(self):
        result = validate_case(_minimal_case(segments=[]))
        assert result["valid"] is False
        assert any("empty" in e for e in result["errors"])

    def test_segments_not_list(self):
        result = validate_case(_minimal_case(segments="bad"))
        assert result["valid"] is False
        assert any("list" in e for e in result["errors"])

    def test_segment_missing_keys(self):
        bad_seg = {"seg_id": "seg_0001"}
        result = validate_case(_minimal_case(segments=[bad_seg]))
        assert result["valid"] is False
        assert any("missing keys" in e for e in result["errors"])

    def test_segment_not_dict(self):
        result = validate_case(_minimal_case(segments=["bad"]))
        assert result["valid"] is False
        assert any("must be a dict" in e for e in result["errors"])

    def test_config_not_dict(self):
        result = validate_case(_minimal_case(config="bad"))
        assert result["valid"] is False

    def test_ground_truth_not_dict(self):
        result = validate_case(_minimal_case(ground_truth="bad"))
        assert result["valid"] is False

    def test_answer_script_not_list(self):
        result = validate_case(_minimal_case(answer_script="bad"))
        assert result["valid"] is False

    def test_answer_script_entry_not_dict(self):
        result = validate_case(_minimal_case(answer_script=["bad"]))
        assert result["valid"] is False

    def test_answer_script_missing_type_warns(self):
        result = validate_case(_minimal_case(answer_script=[{"value": "3 days"}]))
        assert result["valid"] is True
        assert len(result["warnings"]) >= 1

    def test_answer_script_missing_value_warns(self):
        result = validate_case(_minimal_case(
            answer_script=[{"question_type": "duration"}]
        ))
        assert result["valid"] is True
        assert len(result["warnings"]) >= 1

    def test_unknown_fields_warn(self):
        case = _minimal_case(extra_field="foo")
        result = validate_case(case)
        assert result["valid"] is True
        assert any("unknown" in w for w in result["warnings"])

    def test_empty_dict(self):
        result = validate_case({})
        assert result["valid"] is False
        assert len(result["errors"]) >= 2  # case_id + segments


# ── loading tests ────────────────────────────────────────────────────


class TestLoadCase:
    def test_load_valid_yaml(self, tmp_path):
        case = _minimal_case()
        path = tmp_path / "test.yaml"
        _write_yaml(path, case)
        loaded = load_case(path)
        assert loaded["case_id"] == "test_01"
        assert len(loaded["segments"]) == 1

    def test_load_yml_extension(self, tmp_path):
        path = tmp_path / "test.yml"
        _write_yaml(path, _minimal_case())
        loaded = load_case(path)
        assert loaded["case_id"] == "test_01"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_case("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid", encoding="utf-8")
        with pytest.raises(Exception):
            load_case(path)

    def test_load_non_dict_yaml(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_case(path)


class TestLoadAllCases:
    def test_load_directory(self, tmp_path):
        _write_yaml(tmp_path / "a.yaml", _minimal_case(case_id="a"))
        _write_yaml(tmp_path / "b.yml", _minimal_case(case_id="b"))
        cases = load_all_cases(tmp_path)
        assert len(cases) == 2
        assert cases[0]["case_id"] == "a"
        assert cases[1]["case_id"] == "b"

    def test_ignores_non_yaml(self, tmp_path):
        _write_yaml(tmp_path / "case.yaml", _minimal_case())
        (tmp_path / "readme.txt").write_text("not yaml", encoding="utf-8")
        cases = load_all_cases(tmp_path)
        assert len(cases) == 1

    def test_empty_directory(self, tmp_path):
        cases = load_all_cases(tmp_path)
        assert cases == []

    def test_nonexistent_directory(self):
        cases = load_all_cases("/nonexistent/dir")
        assert cases == []

    def test_sorted_by_filename(self, tmp_path):
        _write_yaml(tmp_path / "z.yaml", _minimal_case(case_id="z"))
        _write_yaml(tmp_path / "a.yaml", _minimal_case(case_id="a"))
        cases = load_all_cases(tmp_path)
        assert cases[0]["case_id"] == "a"
        assert cases[1]["case_id"] == "z"

    def test_load_real_cases(self):
        case_dir = Path(__file__).resolve().parent.parent / "resources" / "cases"
        if case_dir.is_dir():
            cases = load_all_cases(case_dir)
            assert len(cases) >= 3
            for case in cases:
                result = validate_case(case)
                assert result["valid"] is True, f"{case.get('case_id')}: {result['errors']}"


# ── run_case tests ───────────────────────────────────────────────────


class TestRunCase:
    def test_returns_expected_keys(self):
        result = run_case(_minimal_case())
        assert set(result.keys()) == {
            "case_id", "session", "app_view", "metrics",
            "ground_truth", "validation", "input_metadata",
        }

    def test_case_id_in_result(self):
        result = run_case(_minimal_case())
        assert result["case_id"] == "test_01"

    def test_metrics_have_expected_groups(self):
        result = run_case(_minimal_case())
        metrics = result["metrics"]
        assert "observation_metrics" in metrics
        assert "hypothesis_metrics" in metrics
        assert "risk_metrics" in metrics
        assert "interaction_metrics" in metrics

    def test_validation_passes(self):
        result = run_case(_minimal_case())
        assert result["validation"]["valid"] is True

    def test_invalid_case_returns_empty(self):
        result = run_case({"case_id": "bad"})
        assert result["validation"]["valid"] is False
        assert result["session"] == {}
        assert result["metrics"] == {}

    def test_session_has_clinical_state(self):
        result = run_case(_minimal_case())
        assert "clinical_state" in result["session"]

    def test_app_view_has_orchestrated(self):
        result = run_case(_minimal_case())
        assert "orchestrated" in result["app_view"]

    def test_default_config_applied(self):
        result = run_case(_minimal_case())
        config = result["session"]["config"]
        assert config["mode"] == "assist"
        assert config["show_questions"] is True

    def test_custom_config_merged(self):
        case = _minimal_case(config={"mode": "scribe"})
        result = run_case(case)
        assert result["session"]["config"]["mode"] == "scribe"

    def test_ground_truth_preserved(self):
        case = _minimal_case(ground_truth={"expected_hypotheses": ["Test"]})
        result = run_case(case)
        assert result["ground_truth"]["expected_hypotheses"] == ["Test"]

    def test_no_ground_truth_defaults_to_empty(self):
        result = run_case(_minimal_case())
        assert result["ground_truth"] == {}

    def test_does_not_mutate_input(self):
        case = _minimal_case()
        original = copy.deepcopy(case)
        run_case(case)
        assert case == original

    def test_deterministic(self):
        case = _full_case()
        r1 = run_case(case)
        r2 = run_case(case)
        assert r1["metrics"] == r2["metrics"]
        assert r1["case_id"] == r2["case_id"]


# ── run_case_script tests ────────────────────────────────────────────


class TestRunCaseScript:
    def test_returns_expected_keys(self):
        result = run_case_script(_full_case())
        assert set(result.keys()) == {
            "case_id", "session", "app_view", "metrics",
            "ground_truth", "validation", "input_metadata",
        }

    def test_script_applies_answers(self):
        result = run_case_script(_full_case())
        # After script + manual update, pending should be drained.
        pending = result["session"].get("pending_observations", [])
        assert len(pending) == 0

    def test_no_script_still_works(self):
        case = _minimal_case()
        result = run_case_script(case)
        assert result["validation"]["valid"] is True
        assert result["case_id"] == "test_01"

    def test_invalid_case(self):
        result = run_case_script({})
        assert result["validation"]["valid"] is False

    def test_does_not_mutate_input(self):
        case = _full_case()
        original = copy.deepcopy(case)
        run_case_script(case)
        assert case == original

    def test_deterministic(self):
        case = _full_case()
        r1 = run_case_script(case)
        r2 = run_case_script(case)
        assert r1["metrics"] == r2["metrics"]

    def test_script_with_related(self):
        case = _minimal_case(answer_script=[
            {"question_type": "duration", "value": "5 days", "related": "headache"},
        ])
        result = run_case_script(case)
        assert result["validation"]["valid"] is True


# ── extract_top_hypotheses tests ─────────────────────────────────────


class TestExtractTopHypotheses:
    def test_empty_result(self):
        result = {"session": {}}
        assert extract_top_hypotheses(result) == []

    def test_returns_list(self):
        result = run_case(_full_case())
        hyps = extract_top_hypotheses(result)
        assert isinstance(hyps, list)

    def test_respects_n(self):
        result = run_case(_full_case())
        hyps = extract_top_hypotheses(result, n=1)
        assert len(hyps) <= 1

    def test_no_session_key(self):
        assert extract_top_hypotheses({}) == []


# ── compare_result_to_ground_truth tests ─────────────────────────────


class TestCompareResultToGroundTruth:
    def test_empty_ground_truth(self):
        result = run_case(_minimal_case())
        comparison = compare_result_to_ground_truth(result)
        assert comparison["hypothesis_matches"] == []
        assert comparison["red_flag_matches"] == []
        assert comparison["finding_matches"] == []

    def test_finding_matches(self):
        case = _minimal_case(ground_truth={"key_findings": ["headache"]})
        result = run_case(case)
        comparison = compare_result_to_ground_truth(result)
        assert len(comparison["finding_matches"]) == 1
        # headache should be found.
        assert comparison["finding_matches"][0]["expected"] == "headache"
        assert comparison["finding_matches"][0]["found"] is True

    def test_missing_finding(self):
        case = _minimal_case(ground_truth={"key_findings": ["rash"]})
        result = run_case(case)
        comparison = compare_result_to_ground_truth(result)
        assert comparison["finding_matches"][0]["found"] is False

    def test_hypothesis_match_structure(self):
        case = _minimal_case(
            ground_truth={"expected_hypotheses": ["SomeCondition"]}
        )
        result = run_case(case)
        comparison = compare_result_to_ground_truth(result)
        assert len(comparison["hypothesis_matches"]) == 1
        assert "expected" in comparison["hypothesis_matches"][0]
        assert "found" in comparison["hypothesis_matches"][0]

    def test_red_flag_match_structure(self):
        case = _minimal_case(ground_truth={"red_flags": ["Sepsis"]})
        result = run_case(case)
        comparison = compare_result_to_ground_truth(result)
        assert len(comparison["red_flag_matches"]) == 1
        assert "expected" in comparison["red_flag_matches"][0]

    def test_no_mutation(self):
        case = _minimal_case(ground_truth={"key_findings": ["headache"]})
        result = run_case(case)
        original = copy.deepcopy(result)
        compare_result_to_ground_truth(result)
        assert result["ground_truth"] == original["ground_truth"]


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_load_validate_run_real_cases(self):
        case_dir = Path(__file__).resolve().parent.parent / "resources" / "cases"
        if not case_dir.is_dir():
            pytest.skip("No cases directory")
        cases = load_all_cases(case_dir)
        assert len(cases) >= 3
        for case in cases:
            validation = validate_case(case)
            assert validation["valid"] is True
            result = run_case(case)
            assert result["case_id"] == case["case_id"]
            assert result["validation"]["valid"] is True

    def test_scripted_real_cases(self):
        case_dir = Path(__file__).resolve().parent.parent / "resources" / "cases"
        if not case_dir.is_dir():
            pytest.skip("No cases directory")
        cases = load_all_cases(case_dir)
        scripted = [c for c in cases if c.get("answer_script")]
        assert len(scripted) >= 1
        for case in scripted:
            result = run_case_script(case)
            assert result["validation"]["valid"] is True
            assert result["session"].get("pending_observations") == []

    def test_batch_determinism(self):
        case_dir = Path(__file__).resolve().parent.parent / "resources" / "cases"
        if not case_dir.is_dir():
            pytest.skip("No cases directory")
        cases = load_all_cases(case_dir)
        for case in cases:
            r1 = run_case(case)
            r2 = run_case(case)
            assert r1["metrics"] == r2["metrics"]
