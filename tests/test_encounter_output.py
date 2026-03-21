"""Tests for encounter output builder."""

from __future__ import annotations

import copy

import pytest

from app.encounter_output import build_encounter_output


# ── helpers ──────────────────────────────────────────────────────────


def _base_state(**overrides) -> dict:
    """Build a minimal clinical state dict."""
    state = {
        "symptoms": [],
        "negations": [],
        "hypotheses": [],
        "hypothesis_prioritization": [],
        "hypothesis_evidence_gaps": {
            "missing_evidence": [],
            "suggested_questions": [],
        },
        "derived": {"red_flags": []},
    }
    state.update(overrides)
    return state


def _hyp(
    title: str,
    rank: int = 1,
    hyp_id: str = "",
    supporting: list[str] | None = None,
    conflicting: list[str] | None = None,
) -> dict:
    return {
        "id": hyp_id or f"hyp_{rank:04d}",
        "title": title,
        "rank": rank,
        "score": 0,
        "confidence": "low",
        "status": "candidate",
        "supporting_observations": [],
        "conflicting_observations": [],
        "related_problems": [],
        "explanation": {
            "summary": f"Hypothesis ranked #{rank}.",
            "supporting_evidence": supporting or [],
            "conflicting_evidence": conflicting or [],
            "score": 0,
            "rank": rank,
        },
    }


def _prio(hyp_id: str, priority_class: str) -> dict:
    return {
        "hypothesis_id": hyp_id,
        "title": "",
        "rank": 0,
        "priority_class": priority_class,
        "reason": "",
        "evidence": [],
    }


def _me(hyp_id: str, findings: list[dict] | None = None) -> dict:
    return {
        "hypothesis_id": hyp_id,
        "title": "",
        "priority_class": "",
        "present": [],
        "absent": [],
        "negated": [],
        "findings": findings or [],
    }


def _sq(hypothesis: str, question: str, reason: str) -> dict:
    return {
        "question": question,
        "target_hypothesis": hypothesis,
        "target_finding": "",
        "priority_class": "",
        "reason": reason,
    }


def _rf(label: str, severity: str = "high") -> dict:
    return {"flag": label.replace(" ", "_"), "label": label, "severity": severity, "evidence": []}


# ── empty input ──────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_state(self):
        result = build_encounter_output(_base_state())
        assert result["key_findings"] == []
        assert result["red_flags"] == []
        assert result["hypotheses"] == []
        assert result["combined_hypotheses"] == {
            "must_not_miss": [],
            "most_likely": [],
            "less_likely": [],
        }

    def test_no_hypotheses_with_symptoms(self):
        result = build_encounter_output(_base_state(
            symptoms=["headache", "nausea"],
            negations=["No fever"],
        ))
        assert result["key_findings"] == ["headache", "nausea", "No fever"]
        assert result["hypotheses"] == []


# ── structure ────────────────────────────────────────────────────────


class TestStructure:
    def test_top_level_keys(self):
        result = build_encounter_output(_base_state())
        assert set(result.keys()) == {"key_findings", "red_flags", "hypotheses", "combined_hypotheses"}

    def test_hypothesis_entry_keys(self):
        state = _base_state(hypotheses=[_hyp("Pneumonia", rank=1)])
        result = build_encounter_output(state)
        entry = result["hypotheses"][0]
        expected_keys = {
            "title", "rank", "priority_class",
            "present_evidence", "conflicting_evidence",
            "findings", "next_question",
        }
        assert set(entry.keys()) == expected_keys

    def test_red_flag_entry_keys(self):
        state = _base_state()
        state["derived"]["red_flags"] = [_rf("Sudden severe headache")]
        result = build_encounter_output(state)
        assert len(result["red_flags"]) == 1
        rf = result["red_flags"][0]
        assert set(rf.keys()) == {"label", "severity"}


# ── hypothesis aggregation ───────────────────────────────────────────


class TestHypothesisAggregation:
    def test_priority_class_from_prioritization(self):
        state = _base_state(
            hypotheses=[_hyp("ACS", rank=1, hyp_id="h1")],
            hypothesis_prioritization=[_prio("h1", "must_not_miss")],
        )
        result = build_encounter_output(state)
        assert result["hypotheses"][0]["priority_class"] == "must_not_miss"

    def test_present_evidence_from_explanation(self):
        state = _base_state(
            hypotheses=[_hyp("ACS", rank=1, supporting=["chest pain", "dyspnea"])],
        )
        result = build_encounter_output(state)
        assert result["hypotheses"][0]["present_evidence"] == ["chest pain", "dyspnea"]

    def test_conflicting_evidence_from_explanation(self):
        state = _base_state(
            hypotheses=[_hyp("ACS", rank=1, conflicting=["normal ECG"])],
        )
        result = build_encounter_output(state)
        assert result["hypotheses"][0]["conflicting_evidence"] == ["normal ECG"]

    def test_findings_from_evidence_gaps(self):
        findings = [
            {"name": "chest pain", "status": "present", "reason": "Cardinal symptom"},
            {"name": "radiation", "status": "absent", "reason": "Classic ACS feature"},
        ]
        state = _base_state(
            hypotheses=[_hyp("ACS", rank=1, hyp_id="h1")],
            hypothesis_evidence_gaps={
                "missing_evidence": [_me("h1", findings)],
                "suggested_questions": [],
            },
        )
        result = build_encounter_output(state)
        assert result["hypotheses"][0]["findings"] == findings

    def test_next_question_from_suggested_questions(self):
        state = _base_state(
            hypotheses=[_hyp("ACS", rank=1, hyp_id="h1")],
            hypothesis_evidence_gaps={
                "missing_evidence": [],
                "suggested_questions": [
                    _sq("ACS", "Does it spread to the arm?", "Radiation is a classic ACS feature"),
                ],
            },
        )
        result = build_encounter_output(state)
        nq = result["hypotheses"][0]["next_question"]
        assert nq is not None
        assert nq["question"] == "Does it spread to the arm?"
        assert nq["reason"] == "Radiation is a classic ACS feature"

    def test_next_question_none_when_no_match(self):
        state = _base_state(hypotheses=[_hyp("Pneumonia", rank=1)])
        result = build_encounter_output(state)
        assert result["hypotheses"][0]["next_question"] is None


# ── ordering ─────────────────────────────────────────────────────────


class TestOrdering:
    def test_hypotheses_sorted_by_rank(self):
        state = _base_state(hypotheses=[
            _hyp("B", rank=3, hyp_id="h3"),
            _hyp("A", rank=1, hyp_id="h1"),
            _hyp("C", rank=2, hyp_id="h2"),
        ])
        result = build_encounter_output(state)
        ranks = [h["rank"] for h in result["hypotheses"]]
        assert ranks == [1, 2, 3]

    def test_titles_follow_rank_order(self):
        state = _base_state(hypotheses=[
            _hyp("Second", rank=2, hyp_id="h2"),
            _hyp("First", rank=1, hyp_id="h1"),
        ])
        result = build_encounter_output(state)
        titles = [h["title"] for h in result["hypotheses"]]
        assert titles == ["First", "Second"]


# ── no mutation ──────────────────────────────────────────────────────


class TestNoMutation:
    def test_state_not_mutated(self):
        state = _base_state(
            symptoms=["headache"],
            hypotheses=[_hyp("Meningitis", rank=1, hyp_id="h1",
                             supporting=["headache"])],
            hypothesis_prioritization=[_prio("h1", "must_not_miss")],
            hypothesis_evidence_gaps={
                "missing_evidence": [_me("h1", [
                    {"name": "fever", "status": "absent", "reason": "Core triad"},
                ])],
                "suggested_questions": [
                    _sq("Meningitis", "Have you had a fever?", "Core triad"),
                ],
            },
        )
        state["derived"] = {"red_flags": [_rf("Neck stiffness")]}
        original = copy.deepcopy(state)
        build_encounter_output(state)
        assert state == original


# ── determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _base_state(
            symptoms=["headache", "fever"],
            hypotheses=[
                _hyp("Meningitis", rank=1, hyp_id="h1", supporting=["headache"]),
                _hyp("Migraine", rank=2, hyp_id="h2"),
            ],
            hypothesis_prioritization=[
                _prio("h1", "must_not_miss"),
                _prio("h2", "less_likely"),
            ],
        )
        state["derived"] = {"red_flags": [_rf("Neck stiffness")]}
        r1 = build_encounter_output(state)
        r2 = build_encounter_output(state)
        assert r1 == r2


# ── key findings ─────────────────────────────────────────────────────


class TestKeyFindings:
    def test_includes_symptoms_and_negations(self):
        state = _base_state(
            symptoms=["headache", "nausea"],
            negations=["No fever", "Denies chest pain"],
        )
        result = build_encounter_output(state)
        assert result["key_findings"] == [
            "headache", "nausea", "No fever", "Denies chest pain",
        ]

    def test_empty_when_no_findings(self):
        result = build_encounter_output(_base_state())
        assert result["key_findings"] == []


# ── red flags ────────────────────────────────────────────────────────


class TestRedFlags:
    def test_red_flags_extracted(self):
        state = _base_state()
        state["derived"]["red_flags"] = [
            _rf("Sudden severe headache", "high"),
            _rf("Chest pain with dyspnea", "moderate"),
        ]
        result = build_encounter_output(state)
        assert len(result["red_flags"]) == 2
        assert result["red_flags"][0]["label"] == "Sudden severe headache"
        assert result["red_flags"][1]["severity"] == "moderate"

    def test_empty_label_skipped(self):
        state = _base_state()
        state["derived"]["red_flags"] = [{"flag": "x", "label": "", "severity": "high", "evidence": []}]
        result = build_encounter_output(state)
        assert result["red_flags"] == []


# ── combined hypotheses ──────────────────────────────────────────────


class TestCombinedHypotheses:
    def test_grouping_correct(self):
        state = _base_state(
            hypotheses=[
                _hyp("Pneumonia", rank=1, hyp_id="h1"),
                _hyp("ACS", rank=2, hyp_id="h2"),
                _hyp("Migraine", rank=3, hyp_id="h3"),
            ],
            hypothesis_prioritization=[
                _prio("h1", "most_likely"),
                _prio("h2", "must_not_miss"),
                _prio("h3", "less_likely"),
            ],
        )
        result = build_encounter_output(state)
        combined = result["combined_hypotheses"]
        assert len(combined["must_not_miss"]) == 1
        assert combined["must_not_miss"][0]["title"] == "ACS"
        assert len(combined["most_likely"]) == 1
        assert combined["most_likely"][0]["title"] == "Pneumonia"
        assert len(combined["less_likely"]) == 1
        assert combined["less_likely"][0]["title"] == "Migraine"

    def test_rank_order_within_group(self):
        state = _base_state(
            hypotheses=[
                _hyp("PE", rank=3, hyp_id="h3"),
                _hyp("ACS", rank=1, hyp_id="h1"),
                _hyp("Sepsis", rank=2, hyp_id="h2"),
            ],
            hypothesis_prioritization=[
                _prio("h1", "must_not_miss"),
                _prio("h2", "must_not_miss"),
                _prio("h3", "must_not_miss"),
            ],
        )
        result = build_encounter_output(state)
        mnm = result["combined_hypotheses"]["must_not_miss"]
        ranks = [h["rank"] for h in mnm]
        assert ranks == [1, 2, 3]

    def test_empty_groups_when_no_hypotheses(self):
        result = build_encounter_output(_base_state())
        combined = result["combined_hypotheses"]
        assert combined == {
            "must_not_miss": [],
            "most_likely": [],
            "less_likely": [],
        }

    def test_original_hypotheses_list_unchanged(self):
        """combined_hypotheses is additive — hypotheses list still present."""
        state = _base_state(
            hypotheses=[_hyp("Pneumonia", rank=1, hyp_id="h1")],
            hypothesis_prioritization=[_prio("h1", "most_likely")],
        )
        result = build_encounter_output(state)
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["title"] == "Pneumonia"
        assert len(result["combined_hypotheses"]["most_likely"]) == 1

    def test_entries_contain_full_data(self):
        """Each entry in combined_hypotheses has all hypothesis fields."""
        state = _base_state(
            hypotheses=[_hyp("ACS", rank=1, hyp_id="h1", supporting=["chest pain"])],
            hypothesis_prioritization=[_prio("h1", "must_not_miss")],
        )
        result = build_encounter_output(state)
        entry = result["combined_hypotheses"]["must_not_miss"][0]
        expected_keys = {
            "title", "rank", "priority_class",
            "present_evidence", "conflicting_evidence",
            "findings", "next_question",
        }
        assert set(entry.keys()) == expected_keys
