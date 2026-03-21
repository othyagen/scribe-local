"""Tests for the unified evaluation dashboard."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from collections import Counter

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
    _collect_encounter_data,
    _render_encounter_preview,
    _render_combined_hypotheses,
    _render_evidence_gaps,
    _render_suggested_questions,
    aggregate_compare_data,
    _render_compare_summary,
    _render_critical_changes,
    _render_findings_diff,
    _render_hypothesis_diff,
    _render_question_diff,
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


# ── encounter data collection ────────────────────────────────────


def _encounter_output(
    key_findings=None, red_flags=None, hypotheses=None, combined_hypotheses=None,
):
    return {
        "key_findings": key_findings or [],
        "red_flags": red_flags or [],
        "hypotheses": hypotheses or [],
        "combined_hypotheses": combined_hypotheses or {
            "must_not_miss": [], "most_likely": [], "less_likely": [],
        },
    }


def _scored_entry(encounter_output=None, evidence_gaps=None):
    """Build a minimal scored result entry with encounter_output embedded."""
    state = {}
    if encounter_output is not None:
        state["encounter_output"] = encounter_output
    if evidence_gaps is not None:
        state["hypothesis_evidence_gaps"] = evidence_gaps
    return {
        "case_id": "test_01",
        "score": {},
        "result_bundle": {"session": {"clinical_state": state}},
    }


def _group_with_entries(entries):
    return {
        "label": "test",
        "case_count": len(entries),
        "scored_count": len(entries),
        "scored_results": entries,
        "analysis": analyze_case_results([]),
    }


class TestCollectEncounterData:
    def test_empty_groups(self):
        data = _collect_encounter_data([])
        assert data["case_count"] == 0
        assert len(data["key_findings"]) == 0
        assert len(data["suggested_questions"]) == 0

    def test_extracts_from_scored_results(self):
        eo = _encounter_output(
            key_findings=["headache", "fever"],
            red_flags=[{"label": "Neck stiffness", "severity": "high"}],
            hypotheses=[{
                "title": "Meningitis", "rank": 1,
                "priority_class": "must_not_miss",
                "present_evidence": [], "conflicting_evidence": [],
                "findings": [
                    {"name": "fever", "status": "present", "reason": "Core triad"},
                    {"name": "photophobia", "status": "absent", "reason": "Classic sign"},
                ],
                "next_question": None,
            }],
        )
        group = _group_with_entries([_scored_entry(eo)])
        data = _collect_encounter_data([group])
        assert data["case_count"] == 1
        assert data["key_findings"]["headache"] == 1
        assert data["red_flags"]["Neck stiffness"] == 1
        assert "Meningitis" in data["hypotheses"]
        assert data["findings_by_hypothesis"]["Meningitis"]["photophobia"]["absent"] == 1

    def test_graceful_missing_encounter_output(self):
        entry = _scored_entry()  # no encounter_output
        group = _group_with_entries([entry])
        data = _collect_encounter_data([group])
        assert data["case_count"] == 0


class TestRenderEncounterPreview:
    def test_section_present_with_data(self):
        data = {
            "case_count": 2,
            "key_findings": Counter({"headache": 2, "fever": 1}),
            "red_flags": Counter({"Neck stiffness": 1}),
            "red_flag_severities": {"Neck stiffness": "high"},
            "hypotheses": {"Meningitis": {"count": 2, "total_rank": 2, "priority_classes": Counter({"must_not_miss": 2})}},
            "findings_by_hypothesis": {},
            "suggested_questions": [],
        }
        lines = _render_encounter_preview(data)
        text = "\n".join(lines)
        assert "ENCOUNTER OUTPUT PREVIEW" in text
        assert "headache" in text
        assert "Neck stiffness" in text
        assert "Meningitis" in text

    def test_empty_data(self):
        data = {
            "case_count": 0,
            "key_findings": Counter(),
            "red_flags": Counter(),
            "red_flag_severities": {},
            "hypotheses": {},
            "findings_by_hypothesis": {},
            "suggested_questions": [],
        }
        assert _render_encounter_preview(data) == []


class TestRenderCombinedHypotheses:
    def test_groups_in_correct_order(self):
        data = {
            "hypotheses": {
                "ACS": {"count": 1, "total_rank": 1, "priority_classes": Counter({"must_not_miss": 1})},
                "Migraine": {"count": 1, "total_rank": 2, "priority_classes": Counter({"less_likely": 1})},
                "Pneumonia": {"count": 1, "total_rank": 1, "priority_classes": Counter({"most_likely": 1})},
            },
        }
        lines = _render_combined_hypotheses(data)
        text = "\n".join(lines)
        assert "COMBINED HYPOTHESIS VIEW" in text
        # must_not_miss appears before most_likely, which appears before less_likely
        idx_mnm = text.index("must_not_miss")
        idx_ml = text.index("most_likely")
        idx_ll = text.index("less_likely")
        assert idx_mnm < idx_ml < idx_ll

    def test_hypothesis_in_correct_group(self):
        data = {
            "hypotheses": {
                "ACS": {"count": 1, "total_rank": 1, "priority_classes": Counter({"must_not_miss": 1})},
            },
        }
        lines = _render_combined_hypotheses(data)
        text = "\n".join(lines)
        assert "ACS" in text
        assert "must_not_miss" in text


class TestRenderEvidenceGaps:
    def test_shows_absent_findings(self):
        data = {
            "hypotheses": {
                "ACS": {"count": 1, "total_rank": 1, "priority_classes": Counter({"must_not_miss": 1})},
            },
            "findings_by_hypothesis": {
                "ACS": {
                    "troponin": {"absent": 3, "present": 0, "negated": 0, "reason": "Cardiac marker"},
                },
            },
        }
        lines = _render_evidence_gaps(data)
        text = "\n".join(lines)
        assert "EVIDENCE GAPS" in text
        assert "troponin" in text
        assert "Cardiac marker" in text

    def test_empty_findings(self):
        data = {"hypotheses": {}, "findings_by_hypothesis": {}}
        assert _render_evidence_gaps(data) == []


class TestRenderSuggestedQuestions:
    def test_renders_with_target_and_reason(self):
        data = {
            "suggested_questions": [{
                "question": "Have you coughed up blood?",
                "target_hypothesis": "PE",
                "reason": "Hemoptysis is a classic PE sign",
                "priority_class": "must_not_miss",
            }],
        }
        lines = _render_suggested_questions(data)
        text = "\n".join(lines)
        assert "SUGGESTED NEXT QUESTIONS" in text
        assert "Have you coughed up blood?" in text
        assert "PE" in text
        assert "Hemoptysis is a classic PE sign" in text

    def test_priority_ordering(self):
        data = {
            "suggested_questions": [
                {"question": "Q1", "target_hypothesis": "A", "reason": "", "priority_class": "less_likely"},
                {"question": "Q2", "target_hypothesis": "B", "reason": "", "priority_class": "must_not_miss"},
            ],
        }
        # _collect_encounter_data sorts by priority; test the render with pre-sorted
        sorted_qs = sorted(
            data["suggested_questions"],
            key=lambda q: {"must_not_miss": 0, "most_likely": 1, "less_likely": 2}.get(q["priority_class"], 2),
        )
        lines = _render_suggested_questions({"suggested_questions": sorted_qs})
        text = "\n".join(lines)
        assert text.index("must_not_miss") < text.index("less_likely")


class TestExistingDashboardUnchanged:
    def test_no_encounter_sections_when_none(self):
        report = render_dashboard_report([], analyze_case_results([]), None, None)
        assert "EVALUATION DASHBOARD" in report
        assert "GLOBAL SUMMARY" in report
        assert "ENCOUNTER OUTPUT PREVIEW" not in report
        assert "COMBINED HYPOTHESIS VIEW" not in report
        assert "EVIDENCE GAPS" not in report
        assert "SUGGESTED NEXT QUESTIONS" not in report

    def test_no_compare_sections_when_none(self):
        report = render_dashboard_report([], analyze_case_results([]), None, None, None)
        assert "TEXT VS TTS SUMMARY" not in report
        assert "CRITICAL CLINICAL CHANGES" not in report
        assert "FINDINGS DIFF OVERVIEW" not in report
        assert "HYPOTHESIS / PRIORITIZATION DIFF" not in report
        assert "QUESTION DIFF" not in report


# ── compare data helpers ───────────────────────────────────────────


def _compare_result(
    case_id="test_01",
    kf_shared=None, kf_text_only=None, kf_tts_only=None,
    rf_shared=None, rf_text_only=None, rf_tts_only=None,
    hyp_shared_titles=None, hyp_text_only_titles=None,
    hyp_tts_only_titles=None, hyp_rank_changes=None,
    prio_unchanged=None, prio_changed=None,
    prio_dropped=None, prio_added=None,
    q_shared=None, q_text_only=None, q_tts_only=None,
):
    """Build a minimal compare_case_modes result dict."""
    return {
        "text_result": {"case_id": case_id},
        "tts_result": {"case_id": case_id},
        "comparison": {
            "key_findings": {
                "shared": kf_shared or [],
                "text_only": kf_text_only or [],
                "tts_only": kf_tts_only or [],
            },
            "red_flags": {
                "shared": rf_shared or [],
                "text_only": rf_text_only or [],
                "tts_only": rf_tts_only or [],
            },
            "hypotheses": {
                "shared_titles": hyp_shared_titles or [],
                "text_only_titles": hyp_text_only_titles or [],
                "tts_only_titles": hyp_tts_only_titles or [],
                "rank_changes": hyp_rank_changes or [],
            },
            "prioritization": {
                "unchanged": prio_unchanged or [],
                "changed": prio_changed or [],
                "dropped": prio_dropped or [],
                "added": prio_added or [],
            },
            "questions": {
                "shared": q_shared or [],
                "text_only": q_text_only or [],
                "tts_only": q_tts_only or [],
            },
        },
    }


# ── aggregate_compare_data ─────────────────────────────────────────


class TestAggregateCompareData:
    def test_empty_list(self):
        data = aggregate_compare_data([])
        assert data["cases_compared"] == 0
        assert sum(data["findings_lost"].values()) == 0
        assert len(data["critical_changes"]) == 0

    def test_counts_findings(self):
        cr = _compare_result(
            kf_shared=["headache"],
            kf_text_only=["fever", "chills"],
            kf_tts_only=["nausea"],
        )
        data = aggregate_compare_data([cr])
        assert data["cases_compared"] == 1
        assert data["findings_lost"]["fever"] == 1
        assert data["findings_lost"]["chills"] == 1
        assert data["findings_gained"]["nausea"] == 1
        assert data["findings_shared"]["headache"] == 1

    def test_counts_red_flags(self):
        cr = _compare_result(
            rf_text_only=["Neck stiffness"],
            rf_tts_only=["Chest pain"],
        )
        data = aggregate_compare_data([cr])
        assert data["red_flags_lost"]["Neck stiffness"] == 1
        assert data["red_flags_gained"]["Chest pain"] == 1

    def test_tracks_rank_changes(self):
        cr = _compare_result(
            hyp_rank_changes=[{"title": "PE", "text_rank": 1, "tts_rank": 3}],
        )
        data = aggregate_compare_data([cr])
        assert len(data["hyp_rank_changes"]) == 1
        assert data["hyp_rank_changes"][0]["title"] == "PE"

    def test_tracks_prioritization(self):
        cr = _compare_result(
            prio_unchanged=[{"title": "A", "priority_class": "most_likely"}],
            prio_changed=[{"title": "B", "text_priority": "must_not_miss", "tts_priority": "most_likely"}],
            prio_dropped=[{"title": "C", "priority_class": "must_not_miss"}],
            prio_added=[{"title": "D", "priority_class": "less_likely"}],
        )
        data = aggregate_compare_data([cr])
        assert data["prio_unchanged"] == 1
        assert len(data["prio_changed"]) == 1
        assert len(data["prio_dropped"]) == 1
        assert len(data["prio_added"]) == 1

    def test_tracks_questions(self):
        cr = _compare_result(
            q_shared=["Q1"],
            q_text_only=["Q2"],
            q_tts_only=["Q3"],
        )
        data = aggregate_compare_data([cr])
        assert data["questions_shared"]["Q1"] == 1
        assert data["questions_text_only"]["Q2"] == 1
        assert data["questions_tts_only"]["Q3"] == 1

    def test_aggregates_across_cases(self):
        cr1 = _compare_result(kf_text_only=["fever"])
        cr2 = _compare_result(kf_text_only=["fever", "cough"])
        data = aggregate_compare_data([cr1, cr2])
        assert data["cases_compared"] == 2
        assert data["findings_lost"]["fever"] == 2
        assert data["findings_lost"]["cough"] == 1

    def test_critical_must_not_miss_dropped(self):
        cr = _compare_result(
            case_id="chest_pain",
            prio_dropped=[{"title": "ACS", "priority_class": "must_not_miss"}],
        )
        data = aggregate_compare_data([cr])
        assert len(data["critical_changes"]) == 1
        assert data["critical_changes"][0]["type"] == "must_not_miss_dropped"
        assert data["critical_changes"][0]["detail"] == "ACS"
        assert data["critical_changes"][0]["case_id"] == "chest_pain"

    def test_critical_red_flag_lost(self):
        cr = _compare_result(
            case_id="meningitis",
            rf_text_only=["Neck stiffness"],
        )
        data = aggregate_compare_data([cr])
        crit = [c for c in data["critical_changes"] if c["type"] == "red_flag_lost"]
        assert len(crit) == 1
        assert crit[0]["detail"] == "Neck stiffness"

    def test_critical_top_rank_changed(self):
        cr = _compare_result(
            hyp_rank_changes=[{"title": "PE", "text_rank": 1, "tts_rank": 3}],
        )
        data = aggregate_compare_data([cr])
        crit = [c for c in data["critical_changes"] if c["type"] == "top_rank_changed"]
        assert len(crit) == 1
        assert "PE" in crit[0]["detail"]

    def test_no_critical_for_non_top_rank_change(self):
        cr = _compare_result(
            hyp_rank_changes=[{"title": "PE", "text_rank": 2, "tts_rank": 3}],
        )
        data = aggregate_compare_data([cr])
        crit = [c for c in data["critical_changes"] if c["type"] == "top_rank_changed"]
        assert len(crit) == 0

    def test_no_critical_for_less_likely_dropped(self):
        cr = _compare_result(
            prio_dropped=[{"title": "X", "priority_class": "less_likely"}],
        )
        data = aggregate_compare_data([cr])
        crit = [c for c in data["critical_changes"] if c["type"] == "must_not_miss_dropped"]
        assert len(crit) == 0

    def test_skips_missing_comparison(self):
        data = aggregate_compare_data([{"text_result": {}, "tts_result": {}}])
        assert data["cases_compared"] == 0

    def test_deterministic(self):
        cr = _compare_result(
            kf_text_only=["a", "b"],
            rf_text_only=["X"],
            hyp_rank_changes=[{"title": "H", "text_rank": 1, "tts_rank": 2}],
        )
        d1 = aggregate_compare_data([cr])
        d2 = aggregate_compare_data([cr])
        assert d1 == d2


# ── render compare sections ───────────────────────────────────────


class TestRenderCompareSummary:
    def test_renders_with_data(self):
        data = aggregate_compare_data([_compare_result(
            kf_text_only=["fever"],
            kf_tts_only=["nausea"],
            rf_text_only=["X"],
        )])
        lines = _render_compare_summary(data)
        text = "\n".join(lines)
        assert "TEXT VS TTS SUMMARY" in text
        assert "findings_lost" in text
        assert "findings_gained" in text

    def test_empty_data(self):
        data = aggregate_compare_data([])
        assert _render_compare_summary(data) == []


class TestRenderCriticalChanges:
    def test_renders_critical(self):
        data = aggregate_compare_data([_compare_result(
            case_id="c1",
            prio_dropped=[{"title": "ACS", "priority_class": "must_not_miss"}],
        )])
        lines = _render_critical_changes(data)
        text = "\n".join(lines)
        assert "CRITICAL CLINICAL CHANGES" in text
        assert "must_not_miss_dropped" in text
        assert "ACS" in text
        assert "c1" in text

    def test_empty_when_no_critical(self):
        data = aggregate_compare_data([_compare_result(kf_shared=["headache"])])
        assert _render_critical_changes(data) == []


class TestRenderFindingsDiff:
    def test_renders_lost_and_gained(self):
        data = aggregate_compare_data([_compare_result(
            kf_text_only=["fever"],
            kf_tts_only=["nausea"],
            kf_shared=["headache"],
        )])
        lines = _render_findings_diff(data)
        text = "\n".join(lines)
        assert "FINDINGS DIFF OVERVIEW" in text
        assert "fever" in text
        assert "nausea" in text
        assert "shared_findings" in text

    def test_empty_when_no_diff(self):
        data = aggregate_compare_data([_compare_result(kf_shared=["headache"])])
        assert _render_findings_diff(data) == []


class TestRenderHypothesisDiff:
    def test_renders_dropped_and_rank_changes(self):
        data = aggregate_compare_data([_compare_result(
            hyp_text_only_titles=["PE"],
            hyp_rank_changes=[{"title": "ACS", "text_rank": 1, "tts_rank": 3}],
            prio_changed=[{"title": "ACS", "text_priority": "must_not_miss", "tts_priority": "most_likely"}],
        )])
        lines = _render_hypothesis_diff(data)
        text = "\n".join(lines)
        assert "HYPOTHESIS / PRIORITIZATION DIFF" in text
        assert "PE" in text
        assert "ACS" in text
        assert "#1->3" in text or "#1->" in text

    def test_empty_when_no_changes(self):
        data = aggregate_compare_data([_compare_result(kf_shared=["headache"])])
        assert _render_hypothesis_diff(data) == []


class TestRenderQuestionDiff:
    def test_renders_questions(self):
        data = aggregate_compare_data([_compare_result(
            q_text_only=["Any chest pain?"],
            q_tts_only=["Any cough?"],
            q_shared=["Any fever?"],
        )])
        lines = _render_question_diff(data)
        text = "\n".join(lines)
        assert "QUESTION DIFF" in text
        assert "Any chest pain?" in text
        assert "Any cough?" in text
        assert "shared_questions" in text

    def test_empty_when_no_diff(self):
        data = aggregate_compare_data([_compare_result(q_shared=["Q1"])])
        assert _render_question_diff(data) == []


# ── compare sections in full report ───────────────────────────────


class TestCompareInFullReport:
    def test_compare_sections_render(self):
        cr = _compare_result(
            kf_text_only=["fever"],
            rf_text_only=["Neck stiffness"],
            hyp_text_only_titles=["PE"],
            prio_dropped=[{"title": "ACS", "priority_class": "must_not_miss"}],
            q_text_only=["Any chest pain?"],
        )
        data = aggregate_compare_data([cr])
        report = render_dashboard_report(
            [], analyze_case_results([]), None, None, data,
        )
        assert "TEXT VS TTS SUMMARY" in report
        assert "CRITICAL CLINICAL CHANGES" in report
        assert "FINDINGS DIFF OVERVIEW" in report
        assert "HYPOTHESIS / PRIORITIZATION DIFF" in report
        assert "QUESTION DIFF" in report

    def test_existing_sections_preserved_with_compare(self):
        cr = _compare_result(kf_text_only=["fever"])
        data = aggregate_compare_data([cr])
        report = render_dashboard_report(
            [], analyze_case_results([]), None, None, data,
        )
        assert "EVALUATION DASHBOARD" in report
        assert "GLOBAL SUMMARY" in report

    def test_no_compare_sections_with_empty_data(self):
        data = aggregate_compare_data([])
        report = render_dashboard_report(
            [], analyze_case_results([]), None, None, data,
        )
        assert "TEXT VS TTS SUMMARY" not in report
