"""Tests for app.benchmark_runner — deterministic benchmark scoring."""

from __future__ import annotations

import json
import os

import pytest

from app.benchmark_runner import (
    build_segments_from_scenario,
    score_symptoms,
    score_negations,
    score_medications,
    score_durations,
    score_qualifiers,
    score_patterns,
    score_red_flags,
    score_sites,
    score_ice,
    run_benchmark,
    generate_summary_markdown,
    write_benchmark_outputs,
)


# ── build_segments_from_scenario ─────────────────────────────────


class TestBuildSegments:
    def test_correct_segment_count(self):
        scenario = {
            "dialogue": [
                {"speaker_id": "spk_0", "text": "Hello"},
                {"speaker_id": "spk_1", "text": "Hi there"},
                {"speaker_id": "spk_0", "text": "How are you?"},
            ],
        }
        segs = build_segments_from_scenario(scenario)
        assert len(segs) == 3

    def test_sequential_ids(self):
        scenario = {
            "dialogue": [
                {"speaker_id": "spk_0", "text": "A"},
                {"speaker_id": "spk_1", "text": "B"},
            ],
        }
        segs = build_segments_from_scenario(scenario)
        assert segs[0]["seg_id"] == 0
        assert segs[1]["seg_id"] == 1

    def test_sequential_timing(self):
        scenario = {
            "dialogue": [
                {"speaker_id": "spk_0", "text": "A"},
                {"speaker_id": "spk_1", "text": "B"},
            ],
        }
        segs = build_segments_from_scenario(scenario)
        assert segs[0]["t0"] == 0.0
        assert segs[0]["t1"] == 1.0
        assert segs[1]["t0"] == 1.0
        assert segs[1]["t1"] == 2.0

    def test_speaker_and_text_preserved(self):
        scenario = {
            "dialogue": [
                {"speaker_id": "spk_0", "text": "Good morning"},
            ],
        }
        segs = build_segments_from_scenario(scenario)
        assert segs[0]["speaker_id"] == "spk_0"
        assert segs[0]["normalized_text"] == "Good morning"

    def test_empty_dialogue(self):
        assert build_segments_from_scenario({"dialogue": []}) == []
        assert build_segments_from_scenario({}) == []


# ── score_symptoms ───────────────────────────────────────────────


class TestScoreSymptoms:
    def test_perfect_match(self):
        r = score_symptoms(["chest pain", "nausea"], ["chest pain", "nausea"])
        assert r["recall"] == 1.0
        assert r["precision"] == 1.0
        assert r["f1"] == 1.0
        assert r["false_negatives"] == []
        assert r["false_positives"] == []

    def test_partial_match(self):
        r = score_symptoms(["chest pain", "nausea"], ["chest pain"])
        assert r["recall"] == 0.5
        assert r["true_positives"] == ["chest pain"]
        assert r["false_negatives"] == ["nausea"]

    def test_no_overlap(self):
        r = score_symptoms(["chest pain"], ["headache"])
        assert r["recall"] == 0.0
        assert r["precision"] == 0.0
        assert r["false_negatives"] == ["chest pain"]
        assert r["false_positives"] == ["headache"]

    def test_empty_expected(self):
        r = score_symptoms([], ["headache"])
        assert r["recall"] == 1.0
        assert r["false_positives"] == ["headache"]

    def test_empty_both(self):
        r = score_symptoms([], [])
        assert r["recall"] == 1.0
        assert r["precision"] == 1.0

    def test_case_insensitive(self):
        r = score_symptoms(["Chest Pain"], ["chest pain"])
        assert r["recall"] == 1.0


# ── score_negations ──────────────────────────────────────────────


class TestScoreNegations:
    def test_substring_matching(self):
        r = score_negations(["sweating"], ["no sweating"])
        assert r["recall"] == 1.0
        assert r["true_positives"] == ["sweating"]

    def test_case_insensitive_substring(self):
        r = score_negations(["Sweating"], ["NO SWEATING"])
        assert r["recall"] == 1.0

    def test_no_match(self):
        r = score_negations(["diabetes"], ["no sweating"])
        assert r["recall"] == 0.0
        assert r["false_negatives"] == ["diabetes"]

    def test_empty_expected(self):
        r = score_negations([], ["no sweating"])
        assert r["recall"] == 1.0


# ── score_medications / score_durations ──────────────────────────


class TestScoreMedications:
    def test_exact_match(self):
        r = score_medications(["ibuprofen"], ["ibuprofen"])
        assert r["recall"] == 1.0

    def test_missing(self):
        r = score_medications(["ibuprofen"], [])
        assert r["recall"] == 0.0


class TestScoreDurations:
    def test_exact_match(self):
        r = score_durations(["three days"], ["three days"])
        assert r["recall"] == 1.0

    def test_partial(self):
        r = score_durations(["three days", "two weeks"], ["three days"])
        assert r["recall"] == 0.5


# ── score_qualifiers ─────────────────────────────────────────────


class TestScoreQualifiers:
    def test_empty_expected(self):
        r = score_qualifiers([], [])
        assert r["overall_accuracy"] == 1.0

    def test_scalar_field_match(self):
        expected = [{"symptom": "chest pain", "character": "dull"}]
        actual = [{"symptom": "chest pain", "qualifiers": {"character": "dull"}}]
        r = score_qualifiers(expected, actual)
        assert r["overall_accuracy"] == 1.0
        assert r["aggregate"]["character"] == 1.0

    def test_scalar_field_mismatch(self):
        expected = [{"symptom": "chest pain", "character": "sharp"}]
        actual = [{"symptom": "chest pain", "qualifiers": {"character": "dull"}}]
        r = score_qualifiers(expected, actual)
        assert r["overall_accuracy"] == 0.0
        assert r["aggregate"]["character"] == 0.0

    def test_missing_symptom_in_actual(self):
        expected = [{"symptom": "headache", "character": "throbbing"}]
        actual = []
        r = score_qualifiers(expected, actual)
        assert r["overall_accuracy"] == 0.0

    def test_list_field_recall(self):
        expected = [{
            "symptom": "chest pain",
            "aggravating_factors": ["exertion", "walking"],
        }]
        actual = [{
            "symptom": "chest pain",
            "qualifiers": {"aggravating_factors": ["exertion"]},
        }]
        r = score_qualifiers(expected, actual)
        assert r["aggregate"]["aggravating_factors"] == 0.5

    def test_case_insensitive_symptom_lookup(self):
        expected = [{"symptom": "Chest Pain", "character": "dull"}]
        actual = [{"symptom": "chest pain", "qualifiers": {"character": "dull"}}]
        r = score_qualifiers(expected, actual)
        assert r["overall_accuracy"] == 1.0

    def test_none_fields_skipped(self):
        expected = [{
            "symptom": "chest pain",
            "severity": None,
            "character": "dull",
        }]
        actual = [{"symptom": "chest pain", "qualifiers": {"character": "dull"}}]
        r = score_qualifiers(expected, actual)
        assert r["overall_accuracy"] == 1.0
        assert "severity" not in r["aggregate"]


# ── score_patterns / score_red_flags ─────────────────────────────


class TestScorePatterns:
    def test_match(self):
        r = score_patterns(
            ["angina_like"],
            [{"pattern": "angina_like", "label": "Angina", "evidence": []}],
        )
        assert r["recall"] == 1.0

    def test_no_match(self):
        r = score_patterns(["angina_like"], [])
        assert r["recall"] == 0.0

    def test_empty_expected(self):
        r = score_patterns([], [{"pattern": "angina_like"}])
        assert r["recall"] == 1.0


class TestScoreRedFlags:
    def test_match(self):
        r = score_red_flags(
            ["chest_pain_with_dyspnea"],
            [{"flag": "chest_pain_with_dyspnea", "severity": "high"}],
        )
        assert r["recall"] == 1.0

    def test_no_match(self):
        r = score_red_flags(["chest_pain_with_dyspnea"], [])
        assert r["recall"] == 0.0


# ── score_sites / score_ice (informational) ──────────────────────


class TestScoreSites:
    def test_count(self):
        r = score_sites([{"site": "chest"}, {"site": "arm"}])
        assert r["count"] == 2

    def test_empty(self):
        assert score_sites([])["count"] == 0


class TestScoreIce:
    def test_count(self):
        r = score_ice({
            "ideas": [{"text": "heart attack"}],
            "concerns": [],
            "expectations": [{"text": "ECG"}],
        })
        assert r["count"] == 2

    def test_empty(self):
        assert score_ice({})["count"] == 0


# ── run_benchmark ────────────────────────────────────────────────


class TestRunBenchmark:
    def test_runs_on_real_scenarios(self):
        from tools.synthetic_cases.scenarios import SCENARIOS
        results = run_benchmark(SCENARIOS)
        assert "timestamp" in results
        assert "cases" in results
        assert "aggregate" in results
        assert len(results["cases"]) == len(SCENARIOS)

    def test_deterministic(self):
        from tools.synthetic_cases.scenarios import SCENARIOS
        r1 = run_benchmark(SCENARIOS)
        r2 = run_benchmark(SCENARIOS)
        # Scores must be identical (timestamp will differ)
        for case_id in r1["cases"]:
            for metric in r1["cases"][case_id]:
                s1 = r1["cases"][case_id][metric]
                s2 = r2["cases"][case_id][metric]
                if "recall" in s1:
                    assert s1["recall"] == s2["recall"]
                if "overall_accuracy" in s1:
                    assert s1["overall_accuracy"] == s2["overall_accuracy"]

    def test_result_structure(self):
        from tools.synthetic_cases.scenarios import SCENARIOS
        results = run_benchmark(SCENARIOS)
        for case_id, case_result in results["cases"].items():
            assert "symptom_extraction" in case_result
            assert "negation_accuracy" in case_result
            assert "medication_accuracy" in case_result
            assert "duration_accuracy" in case_result
            assert "qualifier_accuracy" in case_result
            assert "pattern_detection" in case_result
            assert "red_flag_detection" in case_result
            assert "site_extraction" in case_result
            assert "ice_extraction" in case_result

    def test_aggregate_has_overall_score(self):
        from tools.synthetic_cases.scenarios import SCENARIOS
        results = run_benchmark(SCENARIOS)
        assert "overall_score" in results["aggregate"]
        assert 0.0 <= results["aggregate"]["overall_score"] <= 1.0


# ── generate_summary_markdown ────────────────────────────────────


class TestGenerateSummaryMarkdown:
    def test_produces_markdown(self):
        from tools.synthetic_cases.scenarios import SCENARIOS
        results = run_benchmark(SCENARIOS)
        md = generate_summary_markdown(results)
        assert "# Benchmark Summary" in md
        assert "## Per-Case Results" in md
        assert "## Aggregate Scores" in md

    def test_contains_case_names(self):
        from tools.synthetic_cases.scenarios import SCENARIOS
        results = run_benchmark(SCENARIOS)
        md = generate_summary_markdown(results)
        for case_id in SCENARIOS:
            assert case_id in md


# ── write_benchmark_outputs ──────────────────────────────────────


class TestWriteBenchmarkOutputs:
    def test_creates_files(self, tmp_path):
        results = {
            "timestamp": "2026-01-01T00:00:00",
            "cases": {},
            "aggregate": {"overall_score": 1.0},
        }
        output_dir = str(tmp_path / "benchmarks")
        write_benchmark_outputs(results, output_dir)

        assert os.path.exists(os.path.join(output_dir, "benchmark_results.json"))
        assert os.path.exists(os.path.join(output_dir, "benchmark_summary.md"))

    def test_json_is_valid(self, tmp_path):
        results = {
            "timestamp": "2026-01-01T00:00:00",
            "cases": {},
            "aggregate": {"overall_score": 0.95},
        }
        output_dir = str(tmp_path / "benchmarks")
        write_benchmark_outputs(results, output_dir)

        with open(os.path.join(output_dir, "benchmark_results.json")) as f:
            loaded = json.load(f)
        assert loaded["aggregate"]["overall_score"] == 0.95

    def test_creates_directory(self, tmp_path):
        output_dir = str(tmp_path / "nested" / "dir" / "benchmarks")
        results = {"timestamp": "", "cases": {}, "aggregate": {}}
        write_benchmark_outputs(results, output_dir)
        assert os.path.isdir(output_dir)
