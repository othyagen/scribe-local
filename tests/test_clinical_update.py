"""Tests for clinical update — full state rebuild with new observations."""

from __future__ import annotations

import pytest

from app.clinical_update import apply_update
from app.clinical_state import build_clinical_state
from app.clinical_input import ingest_structured_answers


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


def _make_obs(observation_id: str = "obs_9001",
              finding_type: str = "allergy",
              value: str = "penicillin") -> dict:
    return {
        "observation_id": observation_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": None,
        "speaker_id": None,
        "t_start": None,
        "t_end": None,
        "source_text": None,
        "source": "structured_answer",
        "category": "allergy" if finding_type == "allergy" else finding_type,
        "attributes": {},
        "confidence": None,
    }


_SEGMENTS = [
    _seg("patient has headache and nausea.", seg_id="seg_0001", t0=0.0, t1=3.0),
    _seg("prescribed ibuprofen.", seg_id="seg_0002", t0=3.0, t1=5.0),
]


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_returns_dict(self):
        state = build_clinical_state(_SEGMENTS)
        result = apply_update(state, [], _SEGMENTS)
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        state = build_clinical_state(_SEGMENTS)
        result = apply_update(state, [], _SEGMENTS)
        assert set(result.keys()) == set(state.keys())

    def test_returns_new_dict(self):
        state = build_clinical_state(_SEGMENTS)
        result = apply_update(state, [], _SEGMENTS)
        assert result is not state


# ── no new observations ──────────────────────────────────────────────


class TestNoNewObservations:
    def test_same_observations_when_empty(self):
        state = build_clinical_state(_SEGMENTS)
        result = apply_update(state, [], _SEGMENTS)
        # With no new observations, base pipeline output is returned.
        assert len(result["observations"]) == len(state["observations"])

    def test_same_symptoms(self):
        state = build_clinical_state(_SEGMENTS)
        result = apply_update(state, [], _SEGMENTS)
        assert result["symptoms"] == state["symptoms"]


# ── with new observations ────────────────────────────────────────────


class TestWithNewObservations:
    def test_observation_count_increases(self):
        state = build_clinical_state(_SEGMENTS)
        base_count = len(state["observations"])
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert len(result["observations"]) == base_count + 1

    def test_new_observation_present(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs(observation_id="obs_9001", value="penicillin")]
        result = apply_update(state, new_obs, _SEGMENTS)
        obs_ids = [o["observation_id"] for o in result["observations"]]
        assert "obs_9001" in obs_ids

    def test_base_observations_preserved(self):
        state = build_clinical_state(_SEGMENTS)
        base_ids = {o["observation_id"] for o in state["observations"]}
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        result_ids = {o["observation_id"] for o in result["observations"]}
        assert base_ids.issubset(result_ids)

    def test_extraction_level_unchanged(self):
        """Symptoms, negations, etc. come from segments, not observations."""
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert result["symptoms"] == state["symptoms"]
        assert result["negations"] == state["negations"]
        assert result["durations"] == state["durations"]
        assert result["medications"] == state["medications"]


# ── downstream recomputation ─────────────────────────────────────────


class TestDownstreamRecomputation:
    def test_encounters_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["encounters"], list)

    def test_symptom_groups_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["symptom_groups"], list)

    def test_problems_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["problems"], list)

    def test_hypotheses_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["hypotheses"], list)

    def test_clinical_summary_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["clinical_summary"], dict)

    def test_summary_views_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["summary_views"], dict)

    def test_clinical_insights_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["clinical_insights"], dict)

    def test_next_questions_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["next_questions"], list)

    def test_clinical_graph_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["clinical_graph"], dict)
        assert "nodes" in result["clinical_graph"]
        assert "edges" in result["clinical_graph"]

    def test_derived_recomputed(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert isinstance(result["derived"], dict)
        assert "problem_representation" in result["derived"]
        assert "structured_symptoms" in result["derived"]
        assert "problem_narrative" in result["derived"]


# ── segment-derived layers preserved ─────────────────────────────────


class TestSegmentDerived:
    def test_ice_preserved(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert result["ice"] == state["ice"]

    def test_intensities_preserved(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert result["intensities"] == state["intensities"]

    def test_sites_preserved(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        result = apply_update(state, new_obs, _SEGMENTS)
        assert result["sites"] == state["sites"]


# ── preservation and determinism ─────────────────────────────────────


class TestPreservation:
    def test_does_not_mutate_input_state(self):
        state = build_clinical_state(_SEGMENTS)
        original_obs_count = len(state["observations"])
        original_symptoms = list(state["symptoms"])
        new_obs = [_make_obs()]
        apply_update(state, new_obs, _SEGMENTS)
        assert len(state["observations"]) == original_obs_count
        assert state["symptoms"] == original_symptoms

    def test_does_not_mutate_new_observations(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        original = dict(new_obs[0])
        apply_update(state, new_obs, _SEGMENTS)
        assert new_obs[0] == original

    def test_does_not_mutate_segments(self):
        state = build_clinical_state(_SEGMENTS)
        segments = [dict(s) for s in _SEGMENTS]
        original_count = len(segments)
        apply_update(state, [_make_obs()], segments)
        assert len(segments) == original_count

    def test_deterministic(self):
        state = build_clinical_state(_SEGMENTS)
        new_obs = [_make_obs()]
        r1 = apply_update(state, new_obs, _SEGMENTS)
        r2 = apply_update(state, new_obs, _SEGMENTS)
        # Compare key structural properties (datetime.now() in normalized_timeline
        # means exact equality may differ, so compare stable fields).
        assert r1["observations"] == r2["observations"]
        assert r1["symptoms"] == r2["symptoms"]
        assert r1["problems"] == r2["problems"]
        assert r1["hypotheses"] == r2["hypotheses"]


# ── end-to-end with clinical_input ───────────────────────────────────


class TestEndToEnd:
    def test_input_to_update_loop(self):
        """Full loop: state → input → update → new state."""
        segments = _SEGMENTS
        state = build_clinical_state(segments)

        # Ingest structured answers.
        ingestion = ingest_structured_answers(state, [
            {"type": "duration", "value": "3 days", "related": "headache"},
            {"type": "allergy", "value": "penicillin"},
        ])
        assert len(ingestion["new_observations"]) == 2
        assert ingestion["unparsed_answers"] == []

        # Apply update.
        updated = apply_update(
            state, ingestion["new_observations"], segments,
        )

        # Updated state has more observations.
        assert len(updated["observations"]) > len(state["observations"])

        # All pipeline outputs present.
        assert "clinical_summary" in updated
        assert "clinical_insights" in updated
        assert "next_questions" in updated
        assert "clinical_graph" in updated

    def test_severity_answer_affects_state(self):
        segments = [
            _seg("patient has headache.", seg_id="seg_0001", t0=0.0, t1=3.0),
        ]
        state = build_clinical_state(segments)

        ingestion = ingest_structured_answers(state, [
            {"type": "severity", "value": "severe", "related": "headache"},
        ])

        updated = apply_update(
            state, ingestion["new_observations"], segments,
        )

        # The new severity observation should be in the updated state.
        obs_values = [
            (o["finding_type"], o.get("attributes", {}).get("severity"))
            for o in updated["observations"]
        ]
        assert ("symptom", "severe") in obs_values

    def test_speaker_roles_forwarded(self):
        segments = _SEGMENTS
        roles = {"spk_0": {"role": "patient", "confidence": 0.8, "evidence": []}}
        state = build_clinical_state(segments, speaker_roles=roles)

        updated = apply_update(
            state, [_make_obs()], segments, speaker_roles=roles,
        )
        assert updated["speaker_roles"] == roles
