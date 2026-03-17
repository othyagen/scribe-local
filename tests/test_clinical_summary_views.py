"""Tests for clinical summary views."""

from __future__ import annotations

import pytest

from app.clinical_summary_views import (
    build_summary_views,
    build_overview_summary,
    build_reasoning_summary,
    build_risk_summary,
    build_symptom_summary,
)
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


def _summary(key_findings=None, active_problems=None,
             ranked_hypotheses=None, red_flags=None,
             timeline_summary=None, problem_narrative=None,
             symptom_groups=None, medications=None) -> dict:
    return {
        "key_findings": key_findings or [],
        "active_problems": active_problems or [],
        "ranked_hypotheses": ranked_hypotheses or [],
        "red_flags": red_flags or [],
        "timeline_summary": timeline_summary or [],
        "problem_narrative": problem_narrative or {
            "positive_features": [], "negative_features": [], "narrative": "",
        },
        "symptom_groups": symptom_groups or [],
        "medications": medications or [],
    }


_VIEW_KEYS = {"overview_summary", "reasoning_summary", "risk_summary", "symptom_summary"}


# ── build_summary_views ────────────────────────────────────────────


class TestBuildSummaryViews:
    def test_has_all_view_keys(self):
        views = build_summary_views(_summary())
        assert set(views.keys()) == _VIEW_KEYS

    def test_all_views_are_dicts(self):
        views = build_summary_views(_summary())
        for key in _VIEW_KEYS:
            assert isinstance(views[key], dict)


# ── overview summary ───────────────────────────────────────────────


class TestOverviewSummary:
    def test_counts_symptoms(self):
        s = _summary(key_findings=[
            {"type": "symptom", "value": "headache"},
            {"type": "symptom", "value": "nausea"},
            {"type": "negation", "value": "No fever"},
        ])
        overview = build_overview_summary(s)
        assert overview["symptom_count"] == 2
        assert overview["negation_count"] == 1

    def test_problem_count(self):
        s = _summary(active_problems=[
            {"id": "prob_0001", "title": "headache", "kind": "symptom_problem",
             "priority": "normal", "onset": None, "evidence_count": 1},
        ])
        overview = build_overview_summary(s)
        assert overview["problem_count"] == 1

    def test_medication_count(self):
        s = _summary(medications=["ibuprofen", "paracetamol"])
        overview = build_overview_summary(s)
        assert overview["medication_count"] == 2

    def test_top_hypothesis_present(self):
        s = _summary(ranked_hypotheses=[
            {"id": "hyp_0001", "title": "Pharyngitis", "rank": 1,
             "score": 3, "confidence": "moderate", "supporting_count": 2,
             "summary": "Hypothesis ranked #1 based on supporting clinical evidence."},
        ])
        overview = build_overview_summary(s)
        assert overview["top_hypothesis"]["title"] == "Pharyngitis"
        assert overview["top_hypothesis"]["rank"] == 1

    def test_top_hypothesis_none_when_empty(self):
        overview = build_overview_summary(_summary())
        assert overview["top_hypothesis"] is None

    def test_has_red_flags(self):
        s = _summary(red_flags=[{"label": "Warning", "evidence": ["fever"]}])
        assert build_overview_summary(s)["has_red_flags"] is True

    def test_no_red_flags(self):
        assert build_overview_summary(_summary())["has_red_flags"] is False

    def test_narrative_included(self):
        s = _summary(problem_narrative={
            "positive_features": [], "negative_features": [],
            "narrative": "Patient presents with headache.",
        })
        overview = build_overview_summary(s)
        assert overview["narrative"] == "Patient presents with headache."


# ── reasoning summary ──────────────────────────────────────────────


class TestReasoningSummary:
    def test_hypotheses_grouped_by_confidence(self):
        s = _summary(ranked_hypotheses=[
            {"id": "hyp_0001", "title": "A", "rank": 1, "score": 5,
             "confidence": "moderate", "supporting_count": 3, "summary": ""},
            {"id": "hyp_0002", "title": "B", "rank": 2, "score": 1,
             "confidence": "low", "supporting_count": 1, "summary": ""},
        ])
        reasoning = build_reasoning_summary(s)
        assert len(reasoning["hypotheses_by_confidence"]["moderate"]) == 1
        assert len(reasoning["hypotheses_by_confidence"]["low"]) == 1
        assert reasoning["hypothesis_count"] == 2

    def test_problems_grouped_by_kind(self):
        s = _summary(active_problems=[
            {"id": "prob_0001", "title": "headache", "kind": "symptom_problem",
             "priority": "normal", "onset": None, "evidence_count": 1},
            {"id": "prob_0002", "title": "ACS", "kind": "working_problem",
             "priority": "normal", "onset": None, "evidence_count": 2},
        ])
        reasoning = build_reasoning_summary(s)
        assert "symptom_problem" in reasoning["problems_by_kind"]
        assert "working_problem" in reasoning["problems_by_kind"]

    def test_empty_state(self):
        reasoning = build_reasoning_summary(_summary())
        assert reasoning["hypothesis_count"] == 0
        assert reasoning["problem_count"] == 0


# ── risk summary ───────────────────────────────────────────────────


class TestRiskSummary:
    def test_red_flags_included(self):
        s = _summary(red_flags=[
            {"label": "Meningitis risk", "evidence": ["headache", "fever"]},
        ])
        risk = build_risk_summary(s)
        assert risk["red_flag_count"] == 1
        assert risk["red_flags"][0]["label"] == "Meningitis risk"

    def test_urgent_problems_filtered(self):
        s = _summary(active_problems=[
            {"id": "prob_0001", "title": "headache", "kind": "symptom_problem",
             "priority": "normal", "onset": None, "evidence_count": 1},
            {"id": "prob_0002", "title": "risk", "kind": "risk_problem",
             "priority": "urgent", "onset": None, "evidence_count": 2},
        ])
        risk = build_risk_summary(s)
        assert risk["urgent_problem_count"] == 1
        assert risk["urgent_problems"][0]["title"] == "risk"

    def test_top_hypotheses_limited_to_3(self):
        hyps = [
            {"id": f"hyp_{i:04d}", "title": f"H{i}", "rank": i,
             "score": 5 - i, "confidence": "low", "supporting_count": 1,
             "summary": ""}
            for i in range(1, 6)
        ]
        s = _summary(ranked_hypotheses=hyps)
        risk = build_risk_summary(s)
        assert len(risk["top_hypotheses"]) == 3

    def test_empty_risk(self):
        risk = build_risk_summary(_summary())
        assert risk["red_flag_count"] == 0
        assert risk["urgent_problem_count"] == 0
        assert risk["top_hypotheses"] == []


# ── symptom summary ────────────────────────────────────────────────


class TestSymptomSummary:
    def test_symptoms_extracted(self):
        s = _summary(key_findings=[
            {"type": "symptom", "value": "headache"},
            {"type": "negation", "value": "No fever"},
            {"type": "duration", "value": "3 days"},
        ])
        sym = build_symptom_summary(s)
        assert sym["symptoms"] == ["headache"]
        assert sym["negations"] == ["No fever"]
        assert sym["durations"] == ["3 days"]

    def test_timeline_included(self):
        s = _summary(timeline_summary=[
            {"symptom": "headache", "time_expression": "3 days"},
        ])
        sym = build_symptom_summary(s)
        assert len(sym["timeline"]) == 1
        assert sym["timeline"][0]["symptom"] == "headache"

    def test_groups_by_system(self):
        s = _summary(symptom_groups=[
            {"id": "grp_0001", "title": "acute respiratory symptom group",
             "systems": ["respiratory"], "temporal_bucket": "acute",
             "observation_count": 2},
        ])
        sym = build_symptom_summary(s)
        assert "respiratory" in sym["groups_by_system"]
        assert len(sym["groups_by_system"]["respiratory"]) == 1

    def test_empty_symptom_summary(self):
        sym = build_symptom_summary(_summary())
        assert sym["symptoms"] == []
        assert sym["negations"] == []
        assert sym["durations"] == []
        assert sym["timeline"] == []
        assert sym["groups_by_system"] == {}


# ── preservation and determinism ────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_input(self):
        s = _summary(key_findings=[{"type": "symptom", "value": "headache"}])
        original_count = len(s["key_findings"])
        build_summary_views(s)
        assert len(s["key_findings"]) == original_count

    def test_deterministic(self):
        s = _summary(
            key_findings=[{"type": "symptom", "value": "headache"}],
            ranked_hypotheses=[
                {"id": "hyp_0001", "title": "Test", "rank": 1, "score": 2,
                 "confidence": "low", "supporting_count": 1, "summary": ""},
            ],
        )
        r1 = build_summary_views(s)
        r2 = build_summary_views(s)
        assert r1 == r2


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_views_in_clinical_state(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        assert "summary_views" in state
        assert set(state["summary_views"].keys()) == _VIEW_KEYS

    def test_overview_has_symptom_count(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        overview = state["summary_views"]["overview_summary"]
        assert overview["symptom_count"] >= 2

    def test_symptom_view_has_symptoms(self):
        state = build_clinical_state([
            _seg("patient has headache.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        sym = state["summary_views"]["symptom_summary"]
        assert "headache" in sym["symptoms"]

    def test_empty_state_views(self):
        state = build_clinical_state([_seg("hello.")])
        views = state["summary_views"]
        assert views["overview_summary"]["symptom_count"] == 0
        assert views["reasoning_summary"]["hypothesis_count"] == 0
        assert views["risk_summary"]["red_flag_count"] == 0
        assert views["symptom_summary"]["symptoms"] == []
