"""Tests for format-neutral clinical summary layer."""

from __future__ import annotations

import pytest

from app.clinical_summary import build_clinical_summary
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


_EXPECTED_KEYS = {
    "key_findings",
    "active_problems",
    "ranked_hypotheses",
    "red_flags",
    "timeline_summary",
    "problem_narrative",
    "symptom_groups",
    "medications",
}


# ── structure ──────────────────────────────────────────────────────


class TestStructure:
    def test_has_all_expected_keys(self):
        state = build_clinical_state([_seg("patient has headache.")])
        summary = state["clinical_summary"]
        assert set(summary.keys()) == _EXPECTED_KEYS

    def test_all_fields_correct_types(self):
        state = build_clinical_state([_seg("patient has headache.")])
        summary = state["clinical_summary"]
        assert isinstance(summary["key_findings"], list)
        assert isinstance(summary["active_problems"], list)
        assert isinstance(summary["ranked_hypotheses"], list)
        assert isinstance(summary["red_flags"], list)
        assert isinstance(summary["timeline_summary"], list)
        assert isinstance(summary["problem_narrative"], dict)
        assert isinstance(summary["symptom_groups"], list)
        assert isinstance(summary["medications"], list)

    def test_empty_segments(self):
        state = build_clinical_state([_seg("hello.")])
        summary = state["clinical_summary"]
        assert set(summary.keys()) == _EXPECTED_KEYS
        assert summary["key_findings"] == []
        assert summary["active_problems"] == []
        assert summary["medications"] == []


# ── key findings ───────────────────────────────────────────────────


class TestKeyFindings:
    def test_symptoms_included(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        findings = state["clinical_summary"]["key_findings"]
        symptom_values = [
            f["value"] for f in findings if f["type"] == "symptom"
        ]
        assert "headache" in symptom_values
        assert "nausea" in symptom_values

    def test_negations_included(self):
        state = build_clinical_state([
            _seg("denies fever.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        findings = state["clinical_summary"]["key_findings"]
        neg_findings = [f for f in findings if f["type"] == "negation"]
        assert len(neg_findings) >= 1

    def test_durations_included(self):
        state = build_clinical_state([
            _seg("headache for 3 days.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        findings = state["clinical_summary"]["key_findings"]
        dur_findings = [f for f in findings if f["type"] == "duration"]
        assert len(dur_findings) >= 1

    def test_finding_has_type_and_value(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for f in state["clinical_summary"]["key_findings"]:
            assert "type" in f
            assert "value" in f


# ── active problems ────────────────────────────────────────────────


class TestActiveProblems:
    def test_problems_extracted(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        problems = state["clinical_summary"]["active_problems"]
        assert len(problems) >= 1

    def test_problem_fields(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["clinical_summary"]["active_problems"]:
            assert "id" in prob
            assert "title" in prob
            assert "kind" in prob
            assert "priority" in prob
            assert "evidence_count" in prob

    def test_only_active_problems(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        # All problems from build_problem_list are active in v1
        for prob in state["clinical_summary"]["active_problems"]:
            assert prob["id"].startswith("prob_")


# ── ranked hypotheses ──────────────────────────────────────────────


class TestRankedHypotheses:
    def test_hypotheses_present_when_hints_match(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        hyps = state["clinical_summary"]["ranked_hypotheses"]
        assert len(hyps) >= 1

    def test_hypothesis_fields(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for hyp in state["clinical_summary"]["ranked_hypotheses"]:
            assert "id" in hyp
            assert "title" in hyp
            assert "rank" in hyp
            assert "score" in hyp
            assert "confidence" in hyp
            assert "supporting_count" in hyp
            assert "summary" in hyp

    def test_empty_when_no_hints(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["clinical_summary"]["ranked_hypotheses"] == []


# ── red flags ──────────────────────────────────────────────────────


class TestRedFlags:
    def test_red_flags_is_list(self):
        state = build_clinical_state([_seg("hello.")])
        assert isinstance(state["clinical_summary"]["red_flags"], list)

    def test_red_flag_fields_when_present(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for flag in state["clinical_summary"]["red_flags"]:
            assert "label" in flag
            assert "evidence" in flag


# ── timeline summary ───────────────────────────────────────────────


class TestTimelineSummary:
    def test_timeline_present(self):
        state = build_clinical_state([
            _seg("headache for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        timeline = state["clinical_summary"]["timeline_summary"]
        assert len(timeline) >= 1

    def test_timeline_entry_fields(self):
        state = build_clinical_state([
            _seg("headache for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for entry in state["clinical_summary"]["timeline_summary"]:
            assert "symptom" in entry
            assert "time_expression" in entry


# ── problem narrative ──────────────────────────────────────────────


class TestProblemNarrative:
    def test_narrative_structure(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        narrative = state["clinical_summary"]["problem_narrative"]
        assert "positive_features" in narrative
        assert "negative_features" in narrative
        assert "narrative" in narrative

    def test_narrative_types(self):
        state = build_clinical_state([_seg("hello.")])
        narrative = state["clinical_summary"]["problem_narrative"]
        assert isinstance(narrative["positive_features"], list)
        assert isinstance(narrative["negative_features"], list)
        assert isinstance(narrative["narrative"], str)


# ── symptom groups ─────────────────────────────────────────────────


class TestSymptomGroups:
    def test_groups_present(self):
        state = build_clinical_state([
            _seg("patient has cough and fever.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        groups = state["clinical_summary"]["symptom_groups"]
        assert len(groups) >= 1

    def test_group_fields(self):
        state = build_clinical_state([
            _seg("patient has cough.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for grp in state["clinical_summary"]["symptom_groups"]:
            assert "id" in grp
            assert "title" in grp
            assert "systems" in grp
            assert "temporal_bucket" in grp
            assert "observation_count" in grp


# ── medications ────────────────────────────────────────────────────


class TestMedications:
    def test_medications_extracted(self):
        state = build_clinical_state([
            _seg("prescribed ibuprofen 400 mg.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        meds = state["clinical_summary"]["medications"]
        assert any("ibuprofen" in m for m in meds)

    def test_empty_when_none(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["clinical_summary"]["medications"] == []


# ── preservation and determinism ────────────────────────────────────


class TestPreservation:
    def test_deterministic(self):
        segments = [
            _seg("patient has headache for 3 days. denies fever.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ]
        s1 = build_clinical_state(segments)["clinical_summary"]
        s2 = build_clinical_state(segments)["clinical_summary"]
        assert s1 == s2

    def test_does_not_mutate_state(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        original_keys = set(state.keys())
        build_clinical_summary(state)
        assert set(state.keys()) == original_keys


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_full_scenario(self):
        segments = [
            _seg("patient reports headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("denies fever. no chest pain.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
            _seg("prescribed ibuprofen 400 mg.",
                 seg_id="seg_0003", t0=5.0, t1=8.0),
        ]
        state = build_clinical_state(segments)
        summary = state["clinical_summary"]

        assert len(summary["key_findings"]) >= 3
        assert len(summary["active_problems"]) >= 1
        assert len(summary["timeline_summary"]) >= 1
        assert len(summary["medications"]) >= 1
        assert isinstance(summary["problem_narrative"]["narrative"], str)
