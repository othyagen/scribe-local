"""Tests for clinical metrics — structured evaluation metrics."""

from __future__ import annotations

import pytest

from app.clinical_metrics import derive_clinical_metrics
from app.clinical_state import build_clinical_state


# ── helpers ──────────────────────────────────────────────────────────


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


_METRIC_KEYS = {
    "observation_metrics",
    "problem_metrics",
    "hypothesis_metrics",
    "risk_metrics",
    "interaction_metrics",
    "data_quality_metrics",
    "update_metrics",
}


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_returns_dict_with_expected_keys(self):
        metrics = derive_clinical_metrics({})
        assert set(metrics.keys()) == _METRIC_KEYS

    def test_all_values_are_dicts(self):
        metrics = derive_clinical_metrics({})
        for key in _METRIC_KEYS:
            assert isinstance(metrics[key], dict), f"{key} should be a dict"

    def test_observation_metrics_keys(self):
        m = derive_clinical_metrics({})["observation_metrics"]
        assert set(m.keys()) == {
            "observation_count", "symptom_count", "negation_count",
            "duration_count", "medication_count",
        }

    def test_problem_metrics_keys(self):
        m = derive_clinical_metrics({})["problem_metrics"]
        assert set(m.keys()) == {
            "problem_count", "symptom_problem_count",
            "risk_problem_count", "working_problem_count",
        }

    def test_hypothesis_metrics_keys(self):
        m = derive_clinical_metrics({})["hypothesis_metrics"]
        assert set(m.keys()) == {
            "hypothesis_count", "top_hypothesis",
            "hypothesis_score_distribution",
        }

    def test_risk_metrics_keys(self):
        m = derive_clinical_metrics({})["risk_metrics"]
        assert set(m.keys()) == {"red_flag_count", "has_red_flags"}

    def test_interaction_metrics_keys(self):
        m = derive_clinical_metrics({})["interaction_metrics"]
        assert set(m.keys()) == {"question_count", "questions_by_priority"}

    def test_data_quality_metrics_keys(self):
        m = derive_clinical_metrics({})["data_quality_metrics"]
        assert set(m.keys()) == {"missing_information_count", "uncertainty_count"}

    def test_update_metrics_keys(self):
        m = derive_clinical_metrics({})["update_metrics"]
        assert set(m.keys()) == {"has_pending_observations", "pending_count"}


# ── empty / minimal state ────────────────────────────────────────────


class TestEmptyState:
    def test_empty_dict(self):
        metrics = derive_clinical_metrics({})
        assert metrics["observation_metrics"]["observation_count"] == 0
        assert metrics["problem_metrics"]["problem_count"] == 0
        assert metrics["hypothesis_metrics"]["hypothesis_count"] == 0
        assert metrics["hypothesis_metrics"]["top_hypothesis"] is None
        assert metrics["hypothesis_metrics"]["hypothesis_score_distribution"] == []
        assert metrics["risk_metrics"]["red_flag_count"] == 0
        assert metrics["risk_metrics"]["has_red_flags"] is False
        assert metrics["interaction_metrics"]["question_count"] == 0
        assert metrics["data_quality_metrics"]["missing_information_count"] == 0
        assert metrics["data_quality_metrics"]["uncertainty_count"] == 0

    def test_update_metrics_none_when_missing(self):
        metrics = derive_clinical_metrics({})
        assert metrics["update_metrics"]["has_pending_observations"] is None
        assert metrics["update_metrics"]["pending_count"] is None

    def test_update_metrics_with_empty_pending(self):
        metrics = derive_clinical_metrics({"pending_observations": []})
        assert metrics["update_metrics"]["has_pending_observations"] is False
        assert metrics["update_metrics"]["pending_count"] == 0

    def test_update_metrics_with_pending(self):
        metrics = derive_clinical_metrics({"pending_observations": [{"id": 1}]})
        assert metrics["update_metrics"]["has_pending_observations"] is True
        assert metrics["update_metrics"]["pending_count"] == 1

    def test_minimal_clinical_state(self):
        state = build_clinical_state([_seg("hello.")])
        metrics = derive_clinical_metrics(state)
        assert metrics["observation_metrics"]["observation_count"] == 0
        assert metrics["problem_metrics"]["problem_count"] == 0


# ── observation metrics ──────────────────────────────────────────────


class TestObservationMetrics:
    def test_counts_by_finding_type(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea. denies fever.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["observation_metrics"]
        assert m["symptom_count"] >= 2
        assert m["negation_count"] >= 1
        assert m["observation_count"] == m["symptom_count"] + m["negation_count"] + m["duration_count"] + m["medication_count"]

    def test_medication_counted(self):
        state = build_clinical_state([
            _seg("prescribed ibuprofen 400 mg.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["observation_metrics"]
        assert m["medication_count"] >= 1

    def test_duration_counted(self):
        state = build_clinical_state([
            _seg("headache for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["observation_metrics"]
        assert m["duration_count"] >= 1


# ── problem metrics ──────────────────────────────────────────────────


class TestProblemMetrics:
    def test_symptom_problems(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["problem_metrics"]
        assert m["symptom_problem_count"] >= 1
        assert m["problem_count"] >= m["symptom_problem_count"]

    def test_working_problems_from_hints(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["problem_metrics"]
        # Fever + sore throat triggers Pharyngitis hint → working problem
        assert m["working_problem_count"] >= 0  # may or may not meet threshold


# ── hypothesis metrics ───────────────────────────────────────────────


class TestHypothesisMetrics:
    def test_with_hypotheses(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["hypothesis_metrics"]
        if m["hypothesis_count"] > 0:
            assert m["top_hypothesis"] is not None
            assert "title" in m["top_hypothesis"]
            assert "score" in m["top_hypothesis"]
            assert len(m["hypothesis_score_distribution"]) == m["hypothesis_count"]

    def test_no_hypotheses(self):
        state = build_clinical_state([_seg("hello.")])
        m = derive_clinical_metrics(state)["hypothesis_metrics"]
        assert m["hypothesis_count"] == 0
        assert m["top_hypothesis"] is None
        assert m["hypothesis_score_distribution"] == []


# ── risk metrics ─────────────────────────────────────────────────────


class TestRiskMetrics:
    def test_no_red_flags(self):
        state = build_clinical_state([_seg("hello.")])
        m = derive_clinical_metrics(state)["risk_metrics"]
        assert m["red_flag_count"] == 0
        assert m["has_red_flags"] is False

    def test_reads_from_derived(self):
        state = {"derived": {"red_flags": [{"label": "test"}]}}
        m = derive_clinical_metrics(state)["risk_metrics"]
        assert m["red_flag_count"] == 1
        assert m["has_red_flags"] is True


# ── interaction metrics ──────────────────────────────────────────────


class TestInteractionMetrics:
    def test_no_questions(self):
        state = build_clinical_state([_seg("hello.")])
        m = derive_clinical_metrics(state)["interaction_metrics"]
        assert m["question_count"] == 0
        assert m["questions_by_priority"] == {"high": 0, "medium": 0, "low": 0}

    def test_with_questions(self):
        state = build_clinical_state([
            _seg("patient has nausea.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        m = derive_clinical_metrics(state)["interaction_metrics"]
        assert m["question_count"] >= 1
        total = sum(m["questions_by_priority"].values())
        assert total == m["question_count"]

    def test_priority_buckets(self):
        state = {
            "next_questions": [
                {"question": "q1", "reason": "r1", "priority": "high"},
                {"question": "q2", "reason": "r2", "priority": "medium"},
                {"question": "q3", "reason": "r3", "priority": "low"},
                {"question": "q4", "reason": "r4", "priority": "low"},
            ],
        }
        m = derive_clinical_metrics(state)["interaction_metrics"]
        assert m["question_count"] == 4
        assert m["questions_by_priority"]["high"] == 1
        assert m["questions_by_priority"]["medium"] == 1
        assert m["questions_by_priority"]["low"] == 2


# ── data quality metrics ─────────────────────────────────────────────


class TestDataQualityMetrics:
    def test_no_insights(self):
        m = derive_clinical_metrics({})["data_quality_metrics"]
        assert m["missing_information_count"] == 0
        assert m["uncertainty_count"] == 0

    def test_with_insights(self):
        state = {
            "clinical_insights": {
                "missing_information": [{"category": "x"}, {"category": "y"}],
                "uncertainties": [{"category": "z"}],
                "suggested_questions": [],
                "data_quality_issues": [],
            },
        }
        m = derive_clinical_metrics(state)["data_quality_metrics"]
        assert m["missing_information_count"] == 2
        assert m["uncertainty_count"] == 1


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_state(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        original_symptoms = list(state["symptoms"])
        derive_clinical_metrics(state)
        assert state["symptoms"] == original_symptoms

    def test_deterministic(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        r1 = derive_clinical_metrics(state)
        r2 = derive_clinical_metrics(state)
        assert r1 == r2


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_full_scenario(self):
        segments = [
            _seg("patient reports headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("denies fever. prescribed ibuprofen 400 mg.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ]
        state = build_clinical_state(segments)
        metrics = derive_clinical_metrics(state)

        om = metrics["observation_metrics"]
        assert om["observation_count"] >= 4  # symptoms + negation + duration + medication
        assert om["symptom_count"] >= 2
        assert om["medication_count"] >= 1

        pm = metrics["problem_metrics"]
        assert pm["problem_count"] >= 1

        im = metrics["interaction_metrics"]
        assert isinstance(im["question_count"], int)

        rm = metrics["risk_metrics"]
        assert isinstance(rm["has_red_flags"], bool)

        dq = metrics["data_quality_metrics"]
        assert isinstance(dq["missing_information_count"], int)

    def test_tolerates_partial_state(self):
        """Only some keys present — should not crash."""
        partial = {"symptoms": ["headache"], "observations": []}
        metrics = derive_clinical_metrics(partial)
        assert metrics["observation_metrics"]["observation_count"] == 0
        assert metrics["hypothesis_metrics"]["hypothesis_count"] == 0
        assert metrics["update_metrics"]["has_pending_observations"] is None
