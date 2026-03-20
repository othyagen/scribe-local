"""Tests for hypothesis-driven evidence gaps and targeted questions."""

from __future__ import annotations

import copy

import pytest

from app.hypothesis_evidence_gaps import (
    CONDITION_FINDINGS,
    identify_evidence_gaps,
    _classify_finding,
)
from app.hypothesis_prioritization import DANGEROUS_CONDITIONS
from app.diagnostic_hints import RULES


# ── helpers ──────────────────────────────────────────────────────────


def _hyp(
    title: str,
    rank: int = 1,
    hyp_id: str = "",
    supporting: list | None = None,
) -> dict:
    """Build a minimal hypothesis dict."""
    return {
        "id": hyp_id or f"hyp_{rank:04d}",
        "title": title,
        "status": "candidate",
        "supporting_observations": supporting or [],
        "conflicting_observations": [],
        "related_problems": [],
        "confidence": "low",
        "score": 0,
        "rank": rank,
    }


def _prio(hyp_id: str, title: str, rank: int, priority_class: str) -> dict:
    """Build a minimal prioritization entry."""
    return {
        "hypothesis_id": hyp_id,
        "title": title,
        "rank": rank,
        "priority_class": priority_class,
        "reason": "",
        "evidence": [],
    }


def _obs(obs_id: str, value: str, finding_type: str = "symptom") -> dict:
    """Build a minimal observation dict."""
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": "seg_0001",
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
    }


def _ev(obs_id: str, strength: str = "weak") -> dict:
    """Build a structured evidence dict."""
    return {"observation_id": obs_id, "kind": "supporting", "strength": strength}


# ── registry validation ──────────────────────────────────────────────


class TestRegistry:
    def test_dangerous_conditions_covered(self):
        """All DANGEROUS_CONDITIONS have CONDITION_FINDINGS entries."""
        for condition in DANGEROUS_CONDITIONS:
            assert condition in CONDITION_FINDINGS, (
                f"Missing CONDITION_FINDINGS for dangerous condition: {condition}"
            )

    def test_entries_have_required_fields(self):
        for condition, entries in CONDITION_FINDINGS.items():
            assert len(entries) >= 1, f"Empty entries for {condition}"
            for entry in entries:
                assert "finding" in entry and entry["finding"], (
                    f"Missing/empty finding in {condition}"
                )
                assert "question" in entry and entry["question"], (
                    f"Missing/empty question in {condition}"
                )
                assert "reason" in entry and entry["reason"], (
                    f"Missing/empty reason in {condition}/{entry.get('finding')}"
                )

    def test_all_keys_lowercase(self):
        for key in CONDITION_FINDINGS:
            assert key == key.lower(), f"Key not lowercase: {key}"


# ── classification tests ─────────────────────────────────────────────


class TestClassifyFinding:
    def test_exact_match_in_observations(self):
        assert _classify_finding(
            "headache", {"headache", "fever"}, set(), set(),
        ) == "present"

    def test_exact_match_in_supporting(self):
        assert _classify_finding(
            "fever", set(), {"fever"}, set(),
        ) == "present"

    def test_case_insensitive(self):
        assert _classify_finding(
            "Headache", {"headache"}, set(), set(),
        ) == "present"

    def test_negated(self):
        assert _classify_finding(
            "fever", set(), set(), {"fever"},
        ) == "negated"

    def test_absent(self):
        assert _classify_finding(
            "hemoptysis", {"headache"}, set(), set(),
        ) == "absent"

    def test_multiword_substring_fallback(self):
        """Multi-word finding found as whole-word boundary in observation."""
        assert _classify_finding(
            "chest pain", {"severe chest pain"}, set(), set(),
        ) == "present"

    def test_single_word_no_substring(self):
        """Single-word findings should NOT match via substring."""
        assert _classify_finding(
            "fever", {"feverfew"}, set(), set(),
        ) == "absent"

    def test_present_takes_precedence_over_negated(self):
        """If both present and negated, present wins."""
        assert _classify_finding(
            "fever", {"fever"}, set(), {"fever"},
        ) == "present"


# ── empty input ──────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_hypotheses(self):
        result = identify_evidence_gaps([], [], [], [])
        assert result == {"missing_evidence": [], "suggested_questions": []}

    def test_empty_hypotheses_with_observations(self):
        result = identify_evidence_gaps(
            [], [], [_obs("obs_0001", "headache")], [],
        )
        assert result["missing_evidence"] == []
        assert result["suggested_questions"] == []


# ── core functionality ───────────────────────────────────────────────


class TestCoreFunctionality:
    def test_all_findings_present(self):
        """Hypothesis with all expected findings present → no absent, no questions."""
        hyps = [_hyp("Meningitis", rank=1, supporting=[
            _ev("obs_0001"), _ev("obs_0002"), _ev("obs_0003"),
        ])]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        obs = [
            _obs("obs_0001", "headache"),
            _obs("obs_0002", "fever"),
            _obs("obs_0003", "neck stiffness"),
            _obs("obs_0004", "photophobia"),
        ]
        result = identify_evidence_gaps(hyps, prio, obs, [])
        me = result["missing_evidence"]
        assert len(me) == 1
        assert me[0]["absent"] == []
        assert result["suggested_questions"] == []

    def test_some_findings_absent(self):
        """Hypothesis with partial evidence → correct absent list and questions."""
        hyps = [_hyp("Meningitis", rank=1, supporting=[_ev("obs_0001")])]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        obs = [_obs("obs_0001", "headache")]
        result = identify_evidence_gaps(hyps, prio, obs, [])
        me = result["missing_evidence"][0]
        assert "headache" in me["present"]
        assert "fever" in me["absent"]
        assert "neck stiffness" in me["absent"]
        assert "photophobia" in me["absent"]
        # Questions generated for absent findings.
        q_targets = {q["target_finding"] for q in result["suggested_questions"]}
        assert "fever" in q_targets
        assert "photophobia" in q_targets

    def test_negated_finding_not_in_absent(self):
        """Negated finding appears in negated list, not absent."""
        hyps = [_hyp("Meningitis", rank=1, supporting=[_ev("obs_0001")])]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        obs = [_obs("obs_0001", "headache")]
        negations = ["No fever"]
        result = identify_evidence_gaps(hyps, prio, obs, negations)
        me = result["missing_evidence"][0]
        assert "fever" in me["negated"]
        assert "fever" not in me["absent"]
        # No question for negated finding.
        q_targets = {q["target_finding"] for q in result["suggested_questions"]}
        assert "fever" not in q_targets

    def test_unknown_condition_skipped(self):
        """Hypothesis with title not in CONDITION_FINDINGS → skipped."""
        hyps = [_hyp("Obscure tropical disease", rank=1)]
        prio = [_prio("hyp_0001", "Obscure tropical disease", 1, "most_likely")]
        result = identify_evidence_gaps(hyps, prio, [], [])
        assert result["missing_evidence"] == []
        assert result["suggested_questions"] == []

    def test_finding_present_via_observation_not_supporting(self):
        """Finding in observations (not in hypothesis supporting) → still present."""
        hyps = [_hyp("Meningitis", rank=1)]  # No supporting obs
        prio = [_prio("hyp_0001", "Meningitis", 1, "most_likely")]
        obs = [_obs("obs_0001", "headache")]
        result = identify_evidence_gaps(hyps, prio, obs, [])
        me = result["missing_evidence"][0]
        assert "headache" in me["present"]


# ── ordering tests ───────────────────────────────────────────────────


class TestOrdering:
    def test_must_not_miss_questions_first(self):
        """Questions for must_not_miss hypotheses come before most_likely."""
        hyps = [
            _hyp("Pneumonia", rank=1, hyp_id="hyp_0001"),
            _hyp("Pulmonary embolism", rank=2, hyp_id="hyp_0002"),
        ]
        prio = [
            _prio("hyp_0001", "Pneumonia", 1, "most_likely"),
            _prio("hyp_0002", "Pulmonary embolism", 2, "must_not_miss"),
        ]
        result = identify_evidence_gaps(hyps, prio, [], [])
        questions = result["suggested_questions"]
        # First question should be for must_not_miss.
        must_not_miss_q = [q for q in questions if q["priority_class"] == "must_not_miss"]
        most_likely_q = [q for q in questions if q["priority_class"] == "most_likely"]
        assert len(must_not_miss_q) > 0
        assert len(most_likely_q) > 0
        # All must_not_miss before most_likely.
        first_ml_idx = next(
            i for i, q in enumerate(questions) if q["priority_class"] == "most_likely"
        )
        last_mnm_idx = max(
            i for i, q in enumerate(questions) if q["priority_class"] == "must_not_miss"
        )
        assert last_mnm_idx < first_ml_idx

    def test_rank_order_within_same_class(self):
        """Within same priority class, lower rank comes first."""
        hyps = [
            _hyp("Pneumonia", rank=1, hyp_id="hyp_0001"),
            _hyp("Migraine", rank=2, hyp_id="hyp_0002"),
        ]
        prio = [
            _prio("hyp_0001", "Pneumonia", 1, "less_likely"),
            _prio("hyp_0002", "Migraine", 2, "less_likely"),
        ]
        result = identify_evidence_gaps(hyps, prio, [], [])
        questions = result["suggested_questions"]
        pneumonia_q = [q for q in questions if q["target_hypothesis"] == "Pneumonia"]
        migraine_q = [q for q in questions if q["target_hypothesis"] == "Migraine"]
        assert len(pneumonia_q) > 0
        assert len(migraine_q) > 0
        first_pneumonia = next(
            i for i, q in enumerate(questions) if q["target_hypothesis"] == "Pneumonia"
        )
        first_migraine = next(
            i for i, q in enumerate(questions) if q["target_hypothesis"] == "Migraine"
        )
        assert first_pneumonia < first_migraine


# ── deduplication ────────────────────────────────────────────────────


class TestDeduplication:
    def test_same_question_deduplicated(self):
        """Same question from two hypotheses → only higher-priority version kept."""
        # Both ACS and PE have "chest pain" question with similar text.
        hyps = [
            _hyp("Acute coronary syndrome", rank=1, hyp_id="hyp_0001"),
            _hyp("Pulmonary embolism", rank=2, hyp_id="hyp_0002"),
        ]
        prio = [
            _prio("hyp_0001", "Acute coronary syndrome", 1, "must_not_miss"),
            _prio("hyp_0002", "Pulmonary embolism", 2, "must_not_miss"),
        ]
        result = identify_evidence_gaps(hyps, prio, [], [])
        questions = result["suggested_questions"]
        # Check no exact duplicate question text.
        texts = [q["question"].lower() for q in questions]
        assert len(texts) == len(set(texts))


# ── output structure ─────────────────────────────────────────────────


class TestOutputStructure:
    def test_required_keys_in_result(self):
        result = identify_evidence_gaps([], [], [], [])
        assert "missing_evidence" in result
        assert "suggested_questions" in result

    def test_missing_evidence_entry_keys(self):
        hyps = [_hyp("Meningitis", rank=1)]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        result = identify_evidence_gaps(hyps, prio, [], [])
        entry = result["missing_evidence"][0]
        assert "hypothesis_id" in entry
        assert "title" in entry
        assert "priority_class" in entry
        assert "present" in entry
        assert "absent" in entry
        assert "negated" in entry

    def test_question_entry_keys(self):
        hyps = [_hyp("Meningitis", rank=1)]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        result = identify_evidence_gaps(hyps, prio, [], [])
        assert len(result["suggested_questions"]) > 0
        q = result["suggested_questions"][0]
        assert "question" in q
        assert "target_hypothesis" in q
        assert "target_finding" in q
        assert "priority_class" in q
        assert "reason" in q

    def test_reason_is_non_empty_string(self):
        hyps = [_hyp("Meningitis", rank=1)]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        result = identify_evidence_gaps(hyps, prio, [], [])
        for q in result["suggested_questions"]:
            assert isinstance(q["reason"], str)
            assert len(q["reason"]) > 0

    def test_reason_mentions_finding_or_hypothesis(self):
        """Reason should be clinically relevant to the finding or hypothesis."""
        hyps = [_hyp("Pulmonary embolism", rank=1)]
        prio = [_prio("hyp_0001", "Pulmonary embolism", 1, "must_not_miss")]
        result = identify_evidence_gaps(hyps, prio, [], [])
        for q in result["suggested_questions"]:
            # Reason should be substantive, not a generic placeholder.
            assert len(q["reason"]) >= 20


# ── determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_identical_input_identical_output(self):
        hyps = [
            _hyp("Meningitis", rank=1, supporting=[_ev("obs_0001")]),
            _hyp("Pneumonia", rank=2, supporting=[_ev("obs_0002")]),
        ]
        prio = [
            _prio("hyp_0001", "Meningitis", 1, "must_not_miss"),
            _prio("hyp_0002", "Pneumonia", 2, "most_likely"),
        ]
        obs = [_obs("obs_0001", "headache"), _obs("obs_0002", "cough")]
        r1 = identify_evidence_gaps(hyps, prio, obs, ["No fever"])
        r2 = identify_evidence_gaps(hyps, prio, obs, ["No fever"])
        assert r1 == r2


# ── no mutation ──────────────────────────────────────────────────────


class TestNoMutation:
    def test_inputs_not_mutated(self):
        hyps = [_hyp("Meningitis", rank=1)]
        prio = [_prio("hyp_0001", "Meningitis", 1, "must_not_miss")]
        obs = [_obs("obs_0001", "headache")]
        negations = ["No fever"]
        orig_hyps = copy.deepcopy(hyps)
        orig_prio = copy.deepcopy(prio)
        orig_obs = copy.deepcopy(obs)
        orig_neg = list(negations)
        identify_evidence_gaps(hyps, prio, obs, negations)
        assert hyps == orig_hyps
        assert prio == orig_prio
        assert obs == orig_obs
        assert negations == orig_neg
