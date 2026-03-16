"""Tests for evidence strength annotation."""

from __future__ import annotations

import pytest

from app.evidence_strength import (
    CATEGORY_STRENGTH,
    get_evidence_strength,
    annotate_evidence_strength,
)
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────

def _obs(obs_id: str, category: str | None = "symptom") -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": "symptom",
        "value": "headache",
        "seg_id": "seg_0001",
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": "test.",
        "category": category,
        "attributes": {},
        "confidence": None,
    }


def _prob(obs_ids: list[str], conflicting: list[str] | None = None) -> dict:
    return {
        "id": "prob_0001",
        "title": "headache",
        "kind": "symptom_problem",
        "status": "active",
        "observations": obs_ids,
        "encounters": [],
        "supporting_observations": obs_ids,
        "conflicting_observations": conflicting or [],
        "actions": [],
        "documents": [],
        "priority": "normal",
    }


def _hyp(obs_ids: list[str], conflicting: list[str] | None = None) -> dict:
    return {
        "id": "hyp_0001",
        "title": "Pneumonia",
        "status": "candidate",
        "supporting_observations": obs_ids,
        "conflicting_observations": conflicting or [],
        "related_problems": [],
        "confidence": "low",
    }


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── category to strength mapping ───────────────────────────────────


class TestCategoryStrengthMapping:
    def test_symptom_is_weak(self):
        assert get_evidence_strength({"category": "symptom"}) == "weak"

    def test_clinical_sign_is_moderate(self):
        assert get_evidence_strength({"category": "clinical_sign"}) == "moderate"

    def test_vital_is_strong(self):
        assert get_evidence_strength({"category": "vital"}) == "strong"

    def test_laboratory_is_strong(self):
        assert get_evidence_strength({"category": "laboratory"}) == "strong"

    def test_medication_is_weak(self):
        assert get_evidence_strength({"category": "medication"}) == "weak"

    def test_allergy_is_moderate(self):
        assert get_evidence_strength({"category": "allergy"}) == "moderate"

    def test_device_is_moderate(self):
        assert get_evidence_strength({"category": "device"}) == "moderate"

    def test_none_category_is_weak(self):
        assert get_evidence_strength({"category": None}) == "weak"

    def test_missing_category_is_weak(self):
        assert get_evidence_strength({}) == "weak"

    def test_all_categories_mapped(self):
        from app.observation_taxonomy import OBSERVATION_CATEGORIES
        for cat in OBSERVATION_CATEGORIES:
            assert cat in CATEGORY_STRENGTH

    def test_only_valid_strengths(self):
        valid = {"weak", "moderate", "strong"}
        for strength in CATEGORY_STRENGTH.values():
            assert strength in valid


# ── supporting evidence conversion ─────────────────────────────────


class TestSupportingEvidence:
    def test_converted_to_structured_objects(self):
        problems = [_prob(["obs_0001"])]
        observations = [_obs("obs_0001", "symptom")]
        new_probs, _ = annotate_evidence_strength(problems, observations, [])
        evidence = new_probs[0]["supporting_observations"]
        assert len(evidence) == 1
        assert evidence[0] == {
            "observation_id": "obs_0001",
            "kind": "supporting",
            "strength": "weak",
        }

    def test_strength_from_category(self):
        problems = [_prob(["obs_0001"])]
        observations = [_obs("obs_0001", "laboratory")]
        new_probs, _ = annotate_evidence_strength(problems, observations, [])
        assert new_probs[0]["supporting_observations"][0]["strength"] == "strong"

    def test_multiple_observations(self):
        problems = [_prob(["obs_0001", "obs_0002"])]
        observations = [
            _obs("obs_0001", "symptom"),
            _obs("obs_0002", "vital"),
        ]
        new_probs, _ = annotate_evidence_strength(problems, observations, [])
        evidence = new_probs[0]["supporting_observations"]
        assert evidence[0]["strength"] == "weak"
        assert evidence[1]["strength"] == "strong"

    def test_unknown_obs_id_defaults_to_weak(self):
        problems = [_prob(["obs_9999"])]
        new_probs, _ = annotate_evidence_strength(problems, [], [])
        assert new_probs[0]["supporting_observations"][0]["strength"] == "weak"


# ── conflicting evidence conversion ────────────────────────────────


class TestConflictingEvidence:
    def test_empty_remains_empty(self):
        problems = [_prob(["obs_0001"], conflicting=[])]
        observations = [_obs("obs_0001")]
        new_probs, _ = annotate_evidence_strength(problems, observations, [])
        assert new_probs[0]["conflicting_observations"] == []

    def test_converted_with_kind_conflicting(self):
        problems = [_prob(["obs_0001"], conflicting=["obs_0002"])]
        observations = [_obs("obs_0001"), _obs("obs_0002", "clinical_sign")]
        new_probs, _ = annotate_evidence_strength(problems, observations, [])
        conflict = new_probs[0]["conflicting_observations"]
        assert len(conflict) == 1
        assert conflict[0]["kind"] == "conflicting"
        assert conflict[0]["strength"] == "moderate"


# ── hypothesis evidence ────────────────────────────────────────────


class TestHypothesisEvidence:
    def test_hypothesis_supporting_converted(self):
        hypotheses = [_hyp(["obs_0001"])]
        observations = [_obs("obs_0001", "symptom")]
        _, new_hyps = annotate_evidence_strength([], observations, hypotheses)
        evidence = new_hyps[0]["supporting_observations"]
        assert len(evidence) == 1
        assert evidence[0]["kind"] == "supporting"
        assert evidence[0]["strength"] == "weak"

    def test_hypothesis_conflicting_empty(self):
        hypotheses = [_hyp(["obs_0001"])]
        observations = [_obs("obs_0001")]
        _, new_hyps = annotate_evidence_strength([], observations, hypotheses)
        assert new_hyps[0]["conflicting_observations"] == []


# ── preservation and determinism ────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_problems(self):
        prob = _prob(["obs_0001"])
        original_supporting = list(prob["supporting_observations"])
        annotate_evidence_strength([prob], [_obs("obs_0001")], [])
        assert prob["supporting_observations"] == original_supporting
        assert isinstance(prob["supporting_observations"][0], str)

    def test_does_not_mutate_hypotheses(self):
        hyp = _hyp(["obs_0001"])
        original_supporting = list(hyp["supporting_observations"])
        annotate_evidence_strength([], [_obs("obs_0001")], [hyp])
        assert hyp["supporting_observations"] == original_supporting
        assert isinstance(hyp["supporting_observations"][0], str)

    def test_preserves_other_fields(self):
        prob = _prob(["obs_0001"])
        new_probs, _ = annotate_evidence_strength(
            [prob], [_obs("obs_0001")], [],
        )
        assert new_probs[0]["id"] == prob["id"]
        assert new_probs[0]["title"] == prob["title"]
        assert new_probs[0]["kind"] == prob["kind"]

    def test_deterministic(self):
        problems = [_prob(["obs_0001", "obs_0002"])]
        observations = [_obs("obs_0001"), _obs("obs_0002", "vital")]
        hypotheses = [_hyp(["obs_0001"])]
        r1 = annotate_evidence_strength(problems, observations, hypotheses)
        r2 = annotate_evidence_strength(problems, observations, hypotheses)
        assert r1 == r2

    def test_empty_inputs(self):
        new_probs, new_hyps = annotate_evidence_strength([], [], [])
        assert new_probs == []
        assert new_hyps == []


# ── integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_problems_have_structured_evidence(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["problems"]:
            for ev in prob["supporting_observations"]:
                assert isinstance(ev, dict)
                assert "observation_id" in ev
                assert "kind" in ev
                assert "strength" in ev
                assert ev["kind"] == "supporting"

    def test_hypotheses_have_structured_evidence(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for hyp in state["hypotheses"]:
            for ev in hyp["supporting_observations"]:
                assert isinstance(ev, dict)
                assert ev["kind"] == "supporting"
                assert ev["strength"] in ("weak", "moderate", "strong")

    def test_conflicting_empty_in_v1(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        for prob in state["problems"]:
            assert prob["conflicting_observations"] == []
