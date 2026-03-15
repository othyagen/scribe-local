"""Tests for encounter timeline layer."""

from __future__ import annotations

import pytest

from app.encounter import build_encounters
from app.graph.encounter_builder import build_encounter_graph
from app.graph.graph import ClinicalGraph
from app.graph.types import NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(t0: float, t1: float, text: str = "test") -> dict:
    return {"t0": t0, "t1": t1, "normalized_text": text, "seg_id": 0}


def _obs(obs_id: str, finding_type: str = "symptom", value: str = "x") -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": 0,
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 1.0,
        "source_text": value,
    }


# ===========================================================================
# TestBuildEncounters
# ===========================================================================

class TestBuildEncounters:
    """Tests for build_encounters()."""

    def test_empty_segments_returns_empty(self):
        result = build_encounters([], [])
        assert result == []

    def test_single_encounter(self):
        segs = [_seg(1.0, 2.0), _seg(3.0, 4.0)]
        obs = [_obs("obs_0001"), _obs("obs_0002")]
        result = build_encounters(segs, obs)
        assert len(result) == 1
        enc = result[0]
        assert enc["id"] == "enc_0001"
        assert enc["type"] == "consultation"
        assert enc["modality"] == "in_person"
        assert enc["actions"] == []
        assert enc["documents"] == []

    def test_observation_ids_collected(self):
        segs = [_seg(0.0, 1.0)]
        obs = [_obs("obs_0001"), _obs("obs_0002"), _obs("obs_0003")]
        result = build_encounters(segs, obs)
        assert result[0]["observations"] == ["obs_0001", "obs_0002", "obs_0003"]

    def test_timestamp_from_earliest_segment(self):
        segs = [_seg(5.0, 6.0), _seg(2.0, 3.0), _seg(10.0, 11.0)]
        result = build_encounters(segs, [])
        # Earliest t0 is 2.0
        assert "1970-01-01T00:00:02" in result[0]["timestamp"]

    def test_custom_type_and_modality(self):
        segs = [_seg(0.0, 1.0)]
        result = build_encounters(
            segs, [], encounter_type="emergency", modality="telehealth",
        )
        assert result[0]["type"] == "emergency"
        assert result[0]["modality"] == "telehealth"

    def test_invalid_type_defaults(self):
        segs = [_seg(0.0, 1.0)]
        result = build_encounters(segs, [], encounter_type="invalid_type")
        assert result[0]["type"] == "consultation"

    def test_invalid_modality_defaults(self):
        segs = [_seg(0.0, 1.0)]
        result = build_encounters(segs, [], modality="carrier_pigeon")
        assert result[0]["modality"] == "in_person"

    def test_observations_without_id_skipped(self):
        segs = [_seg(0.0, 1.0)]
        obs = [_obs("obs_0001"), {"finding_type": "symptom", "value": "x"}]
        result = build_encounters(segs, obs)
        assert result[0]["observations"] == ["obs_0001"]

    def test_no_observations(self):
        segs = [_seg(0.0, 1.0)]
        result = build_encounters(segs, [])
        assert result[0]["observations"] == []


# ===========================================================================
# TestEncounterGraph
# ===========================================================================

class TestEncounterGraph:
    """Tests for build_encounter_graph()."""

    def test_encounter_node_created(self):
        graph = ClinicalGraph()
        state = {
            "encounters": [{
                "id": "enc_0001",
                "type": "consultation",
                "modality": "in_person",
                "timestamp": "1970-01-01T00:00:00",
                "observations": ["obs_0001", "obs_0002"],
                "actions": [],
                "documents": [],
            }],
        }
        build_encounter_graph(graph, state)
        d = graph.to_dict()
        enc_nodes = [n for n in d["nodes"] if n["node_type"] == NodeType.ENCOUNTER]
        assert len(enc_nodes) == 1
        node = enc_nodes[0]
        assert node["value"] == "enc_0001"
        assert node["attributes"]["type"] == "consultation"
        assert node["attributes"]["modality"] == "in_person"
        assert node["attributes"]["observation_count"] == 2

    def test_no_edges_created(self):
        graph = ClinicalGraph()
        state = {
            "encounters": [{
                "id": "enc_0001",
                "type": "consultation",
                "modality": "in_person",
                "timestamp": "1970-01-01T00:00:00",
                "observations": ["obs_0001"],
                "actions": [],
                "documents": [],
            }],
        }
        build_encounter_graph(graph, state)
        d = graph.to_dict()
        assert d["edges"] == []

    def test_empty_encounters_no_nodes(self):
        graph = ClinicalGraph()
        build_encounter_graph(graph, {"encounters": []})
        assert graph.to_dict()["nodes"] == []

    def test_missing_encounters_key(self):
        graph = ClinicalGraph()
        build_encounter_graph(graph, {})
        assert graph.to_dict()["nodes"] == []

    def test_deterministic_node_ids(self):
        state = {
            "encounters": [{
                "id": "enc_0001",
                "type": "consultation",
                "modality": "in_person",
                "timestamp": "t",
                "observations": [],
                "actions": [],
                "documents": [],
            }],
        }
        g1 = ClinicalGraph()
        g2 = ClinicalGraph()
        build_encounter_graph(g1, state)
        build_encounter_graph(g2, state)
        assert g1.to_dict() == g2.to_dict()


# ===========================================================================
# TestEncounterIntegration
# ===========================================================================

class TestEncounterIntegration:
    """Integration tests with build_clinical_state()."""

    def test_clinical_state_has_encounters(self):
        from app.clinical_state import build_clinical_state

        segments = [
            {"normalized_text": "I have a headache", "t0": 0.0, "t1": 2.0, "seg_id": 0},
        ]
        state = build_clinical_state(segments)
        assert "encounters" in state
        assert isinstance(state["encounters"], list)

    def test_clinical_state_encounter_has_observations(self):
        from app.clinical_state import build_clinical_state

        segments = [
            {"normalized_text": "I have a headache", "t0": 0.0, "t1": 2.0, "seg_id": 0},
        ]
        state = build_clinical_state(segments)
        if state["encounters"]:
            enc = state["encounters"][0]
            assert "observations" in enc
            # All observation IDs from state should be referenced
            obs_ids = [o["observation_id"] for o in state["observations"]]
            assert enc["observations"] == obs_ids

    def test_graph_contains_encounter_node(self):
        from app.clinical_state import build_clinical_state

        segments = [
            {"normalized_text": "I have a headache", "t0": 0.0, "t1": 2.0, "seg_id": 0},
        ]
        state = build_clinical_state(segments)
        graph = state.get("clinical_graph", {})
        enc_nodes = [
            n for n in graph.get("nodes", [])
            if n.get("node_type") == NodeType.ENCOUNTER
        ]
        assert len(enc_nodes) >= 1
