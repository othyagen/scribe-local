"""Tests for the unified evaluation dashboard."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.run_evaluation_dashboard import (
    run_base_case_group,
    run_variation_group,
    run_adversarial_group,
    run_synthea_group,
    run_dashboard,
    summarize_group_analysis,
    build_global_analysis,
    build_mismatch_report,
    render_dashboard_report,
    _score_cases,
)
from app.case_system import validate_case
from app.case_analysis import analyze_case_results


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"
_SYNTHEA_PATH = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"


# ── helpers ──────────────────────────────────────────────────────────


def _minimal_case_dir(tmp_path: Path) -> Path:
    import yaml
    case = {
        "case_id": "tmp_01",
        "segments": [{
            "seg_id": "seg_0001",
            "t0": 0.0,
            "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient has headache and fever for 3 days.",
        }],
        "ground_truth": {"key_findings": ["headache", "fever"]},
    }
    d = tmp_path / "cases"
    d.mkdir()
    with open(d / "test.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(case, f)
    return d


def _minimal_synthea(tmp_path: Path) -> Path:
    patients = [{"id": "s1", "age": 50, "symptoms": ["cough"], "conditions": ["bronchitis"]}]
    path = tmp_path / "synthea.json"
    path.write_text(json.dumps(patients), encoding="utf-8")
    return path


# ── group structure ─────────────────────────────────────────────────


_GROUP_KEYS = {"label", "case_count", "scored_count", "scored_results", "analysis"}


class TestGroupStructure:
    def test_base_group_keys(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        assert set(group.keys()) == _GROUP_KEYS

    def test_variation_group_keys(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_variation_group(d)
        assert set(group.keys()) == _GROUP_KEYS

    def test_adversarial_group_keys(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_adversarial_group(d)
        assert set(group.keys()) == _GROUP_KEYS

    def test_synthea_group_keys(self, tmp_path):
        path = _minimal_synthea(tmp_path)
        group = run_synthea_group(path)
        assert set(group.keys()) == _GROUP_KEYS

    def test_labels(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        assert run_base_case_group(d)["label"] == "base"
        assert run_variation_group(d)["label"] == "variations"
        assert run_adversarial_group(d)["label"] == "adversarial"
        assert run_synthea_group(p)["label"] == "synthea"


# ── group counts ────────────────────────────────────────────────────


class TestGroupCounts:
    def test_base_count(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        assert group["case_count"] == 1
        assert group["scored_count"] == 1

    def test_variation_count(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_variation_group(d)
        assert group["case_count"] >= 6  # 1 base * 6 variations
        assert group["scored_count"] >= 6

    def test_adversarial_count(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_adversarial_group(d)
        assert group["case_count"] >= 6  # 1 base * 6 strategies
        assert group["scored_count"] >= 6

    def test_synthea_count(self, tmp_path):
        path = _minimal_synthea(tmp_path)
        group = run_synthea_group(path)
        assert group["case_count"] == 1
        assert group["scored_count"] == 1


# ── tolerance ────────────────────────────────────────────────────────


class TestTolerance:
    def test_empty_case_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        group = run_base_case_group(d)
        assert group["case_count"] == 0
        assert group["scored_count"] == 0

    def test_missing_synthea(self, tmp_path):
        group = run_synthea_group(tmp_path / "nonexistent.json")
        assert group["case_count"] == 0
        assert group["scored_count"] == 0

    def test_empty_synthea(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        group = run_synthea_group(path)
        assert group["case_count"] == 0


# ── summarize_group_analysis ─────────────────────────────────────────


class TestSummarizeGroupAnalysis:
    def test_returns_expected_keys(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        summary = summarize_group_analysis(group)
        assert "label" in summary
        assert "count" in summary
        assert "avg_hypothesis_hit_rate" in summary
        assert "avg_red_flag_hit_rate" in summary
        assert "avg_key_finding_hit_rate" in summary
        assert "worst_case" in summary
        assert "most_damaging_strategy" in summary

    def test_empty_group(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        group = run_base_case_group(d)
        summary = summarize_group_analysis(group)
        assert summary["count"] == 0
        assert summary["worst_case"] == "(none)"


# ── build_global_analysis ────────────────────────────────────────────


class TestBuildGlobalAnalysis:
    def test_combines_groups(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        groups = [run_base_case_group(d), run_synthea_group(p)]
        global_a = build_global_analysis(groups)
        assert global_a["overall"]["total_cases"] == 2

    def test_empty_groups(self):
        global_a = build_global_analysis([])
        assert global_a["overall"]["total_cases"] == 0


# ── render_dashboard_report ──────────────────────────────────────────


class TestRenderDashboardReport:
    def test_non_empty_output(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        groups = [run_base_case_group(d), run_synthea_group(p)]
        global_a = build_global_analysis(groups)
        report = render_dashboard_report(groups, global_a)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "EVALUATION DASHBOARD" in report

    def test_contains_sections(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        groups = [
            run_base_case_group(d),
            run_variation_group(d),
            run_adversarial_group(d),
            run_synthea_group(p),
        ]
        global_a = build_global_analysis(groups)
        report = render_dashboard_report(groups, global_a)
        assert "BASE" in report
        assert "VARIATIONS" in report
        assert "ADVERSARIAL" in report
        assert "SYNTHEA" in report
        assert "GLOBAL SUMMARY" in report

    def test_empty_groups(self):
        report = render_dashboard_report([], analyze_case_results([]))
        assert "EVALUATION DASHBOARD" in report
        assert "GLOBAL SUMMARY" in report


# ── run_dashboard ────────────────────────────────────────────────────


class TestRunDashboard:
    def test_returns_expected_keys(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        dashboard = run_dashboard(case_dir=d, synthea_path=p)
        assert "groups" in dashboard
        assert "global_analysis" in dashboard
        assert "mismatch_report" in dashboard
        assert len(dashboard["groups"]) == 4

    def test_group_labels(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        dashboard = run_dashboard(case_dir=d, synthea_path=p)
        labels = [g["label"] for g in dashboard["groups"]]
        assert labels == ["base", "variations", "adversarial", "synthea"]


# ── determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_deterministic(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        d1 = run_dashboard(case_dir=d, synthea_path=p)
        d2 = run_dashboard(case_dir=d, synthea_path=p)
        assert d1["global_analysis"] == d2["global_analysis"]
        for g1, g2 in zip(d1["groups"], d2["groups"]):
            assert g1["analysis"] == g2["analysis"]


# ── integration with real data ───────────────────────────────────────


class TestRealDataIntegration:
    def test_real_cases(self):
        if not _CASE_DIR.is_dir():
            pytest.skip("No cases directory")
        group = run_base_case_group(_CASE_DIR)
        assert group["case_count"] >= 3
        assert group["scored_count"] >= 3

    def test_real_synthea(self):
        if not _SYNTHEA_PATH.exists():
            pytest.skip("No synthea sample")
        group = run_synthea_group(_SYNTHEA_PATH)
        assert group["case_count"] >= 3
        assert group["scored_count"] >= 3

    def test_full_dashboard(self):
        if not _CASE_DIR.is_dir() or not _SYNTHEA_PATH.exists():
            pytest.skip("Missing data files")
        dashboard = run_dashboard()
        assert dashboard["global_analysis"]["overall"]["total_cases"] >= 42
        report = render_dashboard_report(
            dashboard["groups"],
            dashboard["global_analysis"],
            dashboard.get("mismatch_report"),
        )
        assert "GLOBAL SUMMARY" in report


# ── mismatch report integration ────────────────────────────────────


class TestBuildMismatchReport:
    def test_returns_expected_keys(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        report = build_mismatch_report([group])
        assert "mismatch_summary" in report
        assert "suggestions" in report
        assert "apply_preview" in report

    def test_mismatch_summary_structure(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        report = build_mismatch_report([group])
        summary = report["mismatch_summary"]
        assert "total_mismatches" in summary
        assert "cases_with_mismatches" in summary
        assert "cases_total" in summary

    def test_apply_preview_is_dry_run(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        report = build_mismatch_report([group])
        preview = report["apply_preview"]
        assert "proposed_changes" in preview
        assert "applied_changes" in preview
        assert "skipped_changes" in preview
        # Dry run: nothing applied.
        assert preview["applied_changes"] == []

    def test_empty_groups(self):
        report = build_mismatch_report([])
        assert report["mismatch_summary"]["total_mismatches"] == 0
        assert report["suggestions"] == []

    def test_cases_total_matches_scored(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        report = build_mismatch_report([group])
        assert report["mismatch_summary"]["cases_total"] == group["scored_count"]


class TestRenderMismatchSection:
    def test_mismatch_section_in_report(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        dashboard = run_dashboard(case_dir=d, synthea_path=p)
        report = render_dashboard_report(
            dashboard["groups"],
            dashboard["global_analysis"],
            dashboard.get("mismatch_report"),
        )
        assert "MISMATCH ANALYSIS" in report

    def test_no_mismatch_section_when_none(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        group = run_base_case_group(d)
        global_a = build_global_analysis([group])
        report = render_dashboard_report([group], global_a, None)
        assert "MISMATCH ANALYSIS" not in report

    def test_report_deterministic(self, tmp_path):
        d = _minimal_case_dir(tmp_path)
        p = _minimal_synthea(tmp_path)
        d1 = run_dashboard(case_dir=d, synthea_path=p)
        d2 = run_dashboard(case_dir=d, synthea_path=p)
        r1 = render_dashboard_report(
            d1["groups"], d1["global_analysis"], d1.get("mismatch_report"),
        )
        r2 = render_dashboard_report(
            d2["groups"], d2["global_analysis"], d2.get("mismatch_report"),
        )
        assert r1 == r2
