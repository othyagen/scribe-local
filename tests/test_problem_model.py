"""Tests for the problem model layer."""

from __future__ import annotations

import pytest

from app.problem_model import build_problem_list
from app.graph.problem_builder import build_problem_graph
from app.graph.graph import ClinicalGraph
from app.graph.types import NodeType
from app.clinical_state import build_clinical_state


# ── helpers ──────────────────────────────────────────────────────────


def _obs(obs_id: str, finding_type: str, value: str,
         seg_id: str = "seg_0001", t_start: float = 0.0) -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": seg_id,
        "speaker_id": "spk_0",
        "t_start": t_start,
        "t_end": t_start + 1.0,
        "source_text": f"patient has {value}.",
    }


def _minimal_state(**overrides) -> dict:
    """Build a minimal clinical state dict for testing."""
    state: dict = {
        "observations": [],
        "encounters": [],
        "derived": {
            "structured_symptoms": [],
            "red_flags": [],
        },
        "diagnostic_hints": [],
    }
    state.update(overrides)
    return state


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── TestBuildProblemList ─────────────────────────────────────────────


class TestBuildProblemList:
    def test_empty_state(self):
        problems = build_problem_list(_minimal_state())
        assert problems == []

    def test_single_symptom_observation(self):
        state = _minimal_state(observations=[
            _obs("obs_0001", "symptom", "headache"),
        ])
        problems = build_problem_list(state)
        assert len(problems) == 1
        p = problems[0]
        assert p["title"] == "headache"
        assert p["kind"] == "symptom_problem"
        assert p["status"] == "active"
        assert p["priority"] == "normal"
        assert p["observations"] == ["obs_0001"]
        assert p["actions"] == []
        assert p["documents"] == []

    def test_multiple_symptom_observations(self):
        state = _minimal_state(observations=[
            _obs("obs_0001", "symptom", "headache", seg_id="seg_0001"),
            _obs("obs_0002", "symptom", "nausea", seg_id="seg_0001"),
            _obs("obs_0003", "symptom", "headache", seg_id="seg_0002"),
        ])
        problems = build_problem_list(state)
        symptom_problems = [p for p in problems if p["kind"] == "symptom_problem"]
        assert len(symptom_problems) == 2
        titles = [p["title"] for p in symptom_problems]
        assert "headache" in titles
        assert "nausea" in titles

        # headache should collect both observations
        hp = next(p for p in symptom_problems if p["title"] == "headache")
        assert hp["observations"] == ["obs_0001", "obs_0003"]

    def test_non_symptom_observations_ignored(self):
        state = _minimal_state(observations=[
            _obs("obs_0001", "medication", "ibuprofen"),
            _obs("obs_0002", "duration", "3 days"),
        ])
        problems = build_problem_list(state)
        assert problems == []

    def test_onset_enriched_from_structured_symptoms(self):
        state = _minimal_state(
            observations=[_obs("obs_0001", "symptom", "headache")],
        )
        state["derived"]["structured_symptoms"] = [
            {"symptom": "headache", "temporal": {"onset": "sudden"}},
        ]
        problems = build_problem_list(state)
        assert problems[0]["onset"] == "sudden"

    def test_onset_none_when_no_structured_data(self):
        state = _minimal_state(
            observations=[_obs("obs_0001", "symptom", "headache")],
        )
        problems = build_problem_list(state)
        assert problems[0]["onset"] is None

    def test_encounter_ids_included(self):
        state = _minimal_state(
            observations=[_obs("obs_0001", "symptom", "headache")],
            encounters=[{"id": "enc_0001", "type": "consultation"}],
        )
        problems = build_problem_list(state)
        assert problems[0]["encounters"] == ["enc_0001"]

    def test_deterministic_ids(self):
        state = _minimal_state(observations=[
            _obs("obs_0001", "symptom", "headache"),
            _obs("obs_0002", "symptom", "nausea"),
        ])
        p1 = build_problem_list(state)
        p2 = build_problem_list(state)
        assert [p["id"] for p in p1] == [p["id"] for p in p2]
        assert p1[0]["id"] == "prob_0001"
        assert p1[1]["id"] == "prob_0002"


class TestRiskProblem:
    def test_red_flag_creates_risk_problem(self):
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "headache"),
            ],
        )
        state["derived"]["red_flags"] = [
            {
                "flag": "sudden_severe_headache",
                "label": "Sudden severe headache",
                "severity": "high",
                "evidence": ["headache"],
            },
        ]
        problems = build_problem_list(state)
        risk = [p for p in problems if p["kind"] == "risk_problem"]
        assert len(risk) == 1
        assert risk[0]["title"] == "Sudden severe headache"
        assert risk[0]["priority"] == "urgent"
        assert risk[0]["observations"] == ["obs_0001"]

    def test_red_flag_skipped_if_no_observations(self):
        state = _minimal_state()
        state["derived"]["red_flags"] = [
            {
                "flag": "test_flag",
                "label": "Test flag",
                "severity": "high",
                "evidence": ["nonexistent symptom"],
            },
        ]
        problems = build_problem_list(state)
        assert [p for p in problems if p["kind"] == "risk_problem"] == []


class TestWorkingProblem:
    def test_hint_with_two_obs_creates_working_problem(self):
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "fever"),
                _obs("obs_0002", "symptom", "sore throat"),
            ],
            diagnostic_hints=[
                {
                    "condition": "Pharyngitis",
                    "snomed_code": "363746003",
                    "evidence": ["fever", "sore throat"],
                },
            ],
        )
        problems = build_problem_list(state)
        working = [p for p in problems if p["kind"] == "working_problem"]
        assert len(working) == 1
        assert working[0]["title"] == "Pharyngitis"
        assert working[0]["priority"] == "normal"
        assert set(working[0]["observations"]) == {"obs_0001", "obs_0002"}

    def test_hint_with_one_obs_skipped(self):
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "fever"),
            ],
            diagnostic_hints=[
                {
                    "condition": "Pharyngitis",
                    "snomed_code": "363746003",
                    "evidence": ["fever", "sore throat"],
                },
            ],
        )
        problems = build_problem_list(state)
        working = [p for p in problems if p["kind"] == "working_problem"]
        assert working == []

    def test_hint_deduped_with_symptom_problem(self):
        """If a hint title matches a symptom, skip the working_problem."""
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "headache"),
                _obs("obs_0002", "symptom", "nausea"),
            ],
            diagnostic_hints=[
                {
                    "condition": "headache",
                    "snomed_code": "25064002",
                    "evidence": ["headache", "nausea"],
                },
            ],
        )
        problems = build_problem_list(state)
        working = [p for p in problems if p["kind"] == "working_problem"]
        assert working == []

    def test_hint_dedup_case_insensitive(self):
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "Headache"),
                _obs("obs_0002", "symptom", "nausea"),
            ],
            diagnostic_hints=[
                {
                    "condition": "HEADACHE",
                    "snomed_code": "25064002",
                    "evidence": ["headache", "nausea"],
                },
            ],
        )
        problems = build_problem_list(state)
        working = [p for p in problems if p["kind"] == "working_problem"]
        assert working == []


# ── TestProblemGraph ─────────────────────────────────────────────────


class TestProblemGraph:
    def test_problem_nodes_created(self):
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "headache"),
                _obs("obs_0002", "symptom", "nausea"),
            ],
        )
        state["problems"] = build_problem_list(state)

        graph = ClinicalGraph()
        build_problem_graph(graph, state)

        nodes = [n for n in graph.to_dict()["nodes"]
                 if n["node_type"] == NodeType.PROBLEM]
        assert len(nodes) == 2
        titles = [n["attributes"]["title"] for n in nodes]
        assert "headache" in titles
        assert "nausea" in titles

    def test_no_edges(self):
        state = _minimal_state(
            observations=[_obs("obs_0001", "symptom", "headache")],
        )
        state["problems"] = build_problem_list(state)

        graph = ClinicalGraph()
        build_problem_graph(graph, state)
        assert graph.to_dict()["edges"] == []

    def test_empty_problems_no_nodes(self):
        state = _minimal_state()
        state["problems"] = []

        graph = ClinicalGraph()
        build_problem_graph(graph, state)
        assert graph.to_dict()["nodes"] == []

    def test_deterministic_node_ids(self):
        state = _minimal_state(
            observations=[
                _obs("obs_0001", "symptom", "headache"),
                _obs("obs_0002", "symptom", "nausea"),
            ],
        )
        state["problems"] = build_problem_list(state)

        g1 = ClinicalGraph()
        build_problem_graph(g1, state)
        g2 = ClinicalGraph()
        build_problem_graph(g2, state)

        ids1 = [n["node_id"] for n in g1.to_dict()["nodes"]]
        ids2 = [n["node_id"] for n in g2.to_dict()["nodes"]]
        assert ids1 == ids2


# ── TestProblemIntegration ───────────────────────────────────────────


class TestProblemIntegration:
    def test_problems_key_in_clinical_state(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "problems" in state
        assert isinstance(state["problems"], list)

    def test_problem_obs_ids_subset_of_state_obs(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
        ])
        all_obs_ids = {o["observation_id"] for o in state["observations"]}
        for prob in state["problems"]:
            for obs_id in prob["observations"]:
                assert obs_id in all_obs_ids, (
                    f"Problem {prob['id']} references {obs_id} "
                    f"not in observations"
                )

    def test_graph_contains_problem_nodes(self):
        state = build_clinical_state([
            _seg("patient has headache.", seg_id="seg_0001"),
        ])
        nodes = state["clinical_graph"]["nodes"]
        problem_nodes = [n for n in nodes if n["node_type"] == "problem"]
        assert len(problem_nodes) >= 1

    def test_empty_segments_no_problems(self):
        state = build_clinical_state([])
        assert state["problems"] == []

    def test_full_scenario(self):
        """Realistic scenario: symptoms + diagnostic hint + red flag."""
        segments = [
            _seg("patient reports fever and sore throat for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("denies chest pain.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
        ]
        state = build_clinical_state(segments)
        problems = state["problems"]

        # Should have at least symptom_problems
        kinds = {p["kind"] for p in problems}
        assert "symptom_problem" in kinds

        # All problems have required fields
        for prob in problems:
            assert "id" in prob
            assert "title" in prob
            assert "kind" in prob
            assert "status" in prob
            assert "observations" in prob
            assert "encounters" in prob
            assert "actions" in prob
            assert "documents" in prob
            assert "priority" in prob
            assert prob["status"] == "active"
            assert isinstance(prob["observations"], list)
