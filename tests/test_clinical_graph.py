"""Tests for the clinical graph layer."""

from __future__ import annotations

import pytest

from app.graph.models import Node, Edge
from app.graph.graph import ClinicalGraph
from app.graph.types import NodeType, EdgeType
from app.graph.symptom_builder import build_symptom_graph
from app.clinical_graph import build_clinical_graph


# ══════════════════════════════════════════════════════════════════
# Graph core (models + container)
# ══════════════════════════════════════════════════════════════════


class TestNodeModel:
    def test_create_node(self):
        n = Node(node_id="n1", node_type="symptom", value="headache")
        assert n.node_id == "n1"
        assert n.node_type == "symptom"
        assert n.value == "headache"
        assert n.attributes == {}
        assert n.evidence_obs_ids == []

    def test_node_with_attributes(self):
        n = Node(
            node_id="n1", node_type="modifier", value="exertion",
            attributes={"modifier_type": "aggravating"},
            evidence_obs_ids=["obs_0001"],
        )
        assert n.attributes["modifier_type"] == "aggravating"
        assert n.evidence_obs_ids == ["obs_0001"]

    def test_node_to_dict(self):
        n = Node(node_id="n1", node_type="symptom", value="headache",
                 evidence_obs_ids=["obs_0001"])
        d = n.to_dict()
        assert d == {
            "node_id": "n1",
            "node_type": "symptom",
            "value": "headache",
            "attributes": {},
            "evidence_obs_ids": ["obs_0001"],
        }


class TestEdgeModel:
    def test_create_edge(self):
        e = Edge(edge_id="e1", source_id="n1", target_id="n2",
                 edge_type="HAS_SITE")
        assert e.edge_id == "e1"
        assert e.source_id == "n1"
        assert e.target_id == "n2"
        assert e.edge_type == "HAS_SITE"

    def test_edge_to_dict(self):
        e = Edge(edge_id="e1", source_id="n1", target_id="n2",
                 edge_type="HAS_SITE", evidence_obs_ids=["obs_0001"])
        d = e.to_dict()
        assert d == {
            "edge_id": "e1",
            "source_id": "n1",
            "target_id": "n2",
            "edge_type": "HAS_SITE",
            "attributes": {},
            "evidence_obs_ids": ["obs_0001"],
        }


class TestClinicalGraphContainer:
    def test_empty_graph(self):
        g = ClinicalGraph()
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_add_node(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        assert g.node_count == 1
        assert g.get_node("n1") is not None
        assert g.get_node("n1").value == "headache"

    def test_add_duplicate_node_raises(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        with pytest.raises(ValueError, match="Duplicate node_id"):
            g.add_node(Node(node_id="n1", node_type="symptom", value="nausea"))

    def test_add_edge(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                        edge_type="HAS_SITE"))
        assert g.edge_count == 1
        assert g.get_edge("e1") is not None

    def test_add_edge_missing_source(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        with pytest.raises(ValueError, match="Source node not found"):
            g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                            edge_type="HAS_SITE"))

    def test_add_edge_missing_target(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        with pytest.raises(ValueError, match="Target node not found"):
            g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                            edge_type="HAS_SITE"))

    def test_add_duplicate_edge_raises(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                        edge_type="HAS_SITE"))
        with pytest.raises(ValueError, match="Duplicate edge_id"):
            g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                            edge_type="HAS_SITE"))

    def test_get_nodes_by_type(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_node(Node(node_id="n3", node_type="symptom", value="nausea"))

        symptoms = g.get_nodes_by_type("symptom")
        assert len(symptoms) == 2
        assert symptoms[0].value == "headache"
        assert symptoms[1].value == "nausea"

        sites = g.get_nodes_by_type("site")
        assert len(sites) == 1

        assert g.get_nodes_by_type("nonexistent") == []

    def test_get_edges_from(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_node(Node(node_id="n3", node_type="character", value="sharp"))
        g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                        edge_type="HAS_SITE"))
        g.add_edge(Edge(edge_id="e2", source_id="n1", target_id="n3",
                        edge_type="HAS_CHARACTER"))

        edges = g.get_edges_from("n1")
        assert len(edges) == 2
        assert g.get_edges_from("n2") == []

    def test_get_edges_to(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                        edge_type="HAS_SITE"))

        assert len(g.get_edges_to("n2")) == 1
        assert g.get_edges_to("n1") == []

    def test_get_nonexistent_node(self):
        g = ClinicalGraph()
        assert g.get_node("missing") is None

    def test_get_nonexistent_edge(self):
        g = ClinicalGraph()
        assert g.get_edge("missing") is None


class TestGraphSerialization:
    def test_empty_graph_serialization(self):
        g = ClinicalGraph()
        d = g.to_dict()
        assert d == {"nodes": [], "edges": []}

    def test_serialization_preserves_insertion_order(self):
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache"))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                        edge_type="HAS_SITE"))
        d = g.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["nodes"][0]["node_id"] == "n1"
        assert d["nodes"][1]["node_id"] == "n2"
        assert d["edges"][0]["edge_id"] == "e1"

    def test_serialization_is_plain_dict(self):
        """Ensure to_dict produces JSON-serializable output."""
        import json
        g = ClinicalGraph()
        g.add_node(Node(node_id="n1", node_type="symptom", value="headache",
                        evidence_obs_ids=["obs_0001"]))
        g.add_node(Node(node_id="n2", node_type="site", value="frontal"))
        g.add_edge(Edge(edge_id="e1", source_id="n1", target_id="n2",
                        edge_type="HAS_SITE", evidence_obs_ids=["obs_0001"]))
        d = g.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ══════════════════════════════════════════════════════════════════
# Type constants
# ══════════════════════════════════════════════════════════════════


class TestTypes:
    def test_node_types_are_strings(self):
        assert isinstance(NodeType.SYMPTOM, str)
        assert isinstance(NodeType.SITE, str)
        assert isinstance(NodeType.ICE_ITEM, str)

    def test_edge_types_are_strings(self):
        assert isinstance(EdgeType.HAS_SITE, str)
        assert isinstance(EdgeType.ASSOCIATED_WITH, str)
        assert isinstance(EdgeType.EXPRESSED_AS, str)


# ══════════════════════════════════════════════════════════════════
# Symptom builder
# ══════════════════════════════════════════════════════════════════


def _obs(finding_type: str, value: str, seg_id: str,
         obs_id: str = "obs_0001") -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": seg_id,
        "speaker_id": "spk_0",
        "t_start": 0.0,
        "t_end": 2.0,
        "source_text": f"text with {value}",
    }


def _minimal_state(**overrides) -> dict:
    base: dict = {
        "symptoms": [],
        "qualifiers": [],
        "timeline": [],
        "durations": [],
        "negations": [],
        "medications": [],
        "observations": [],
        "sites": [],
        "intensities": [],
        "ice": {"ideas": [], "concerns": [], "expectations": []},
        "derived": {
            "red_flags": [],
            "symptom_representations": [],
            "structured_symptoms": [],
        },
    }
    base.update(overrides)
    return base


class TestSymptomBuilderEmpty:
    def test_empty_state(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state())
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_no_symptoms(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(symptoms=[]))
        assert g.node_count == 0


class TestSymptomBuilderNodes:
    def test_symptom_nodes_created(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache", "nausea"],
        ))
        syms = g.get_nodes_by_type(NodeType.SYMPTOM)
        assert len(syms) == 2
        assert syms[0].value == "headache"
        assert syms[1].value == "nausea"

    def test_evidence_obs_ids_on_symptom_node(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("symptom", "headache", "seg_0002", obs_id="obs_0003"),
            ],
        ))
        syms = g.get_nodes_by_type(NodeType.SYMPTOM)
        assert len(syms) == 1
        assert "obs_0001" in syms[0].evidence_obs_ids
        assert "obs_0003" in syms[0].evidence_obs_ids


class TestSymptomBuilderQualifierEdges:
    def test_character_edge(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["chest pain"],
            qualifiers=[{
                "symptom": "chest pain",
                "qualifiers": {"character": "sharp"},
            }],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [{
                    "symptom": "chest pain",
                    "spatial": {"site": None, "laterality": None, "radiation": None},
                    "qualitative": {"character": "sharp", "intensity": None,
                                    "intensity_raw": None, "severity": None},
                    "temporal": {"onset": None, "onset_type": None,
                                 "duration": None, "pattern": None,
                                 "progression": None},
                    "modifiers": {"aggravating_factors": [], "relieving_factors": []},
                    "context": {"associated_present": [], "associated_absent": [],
                                "prior_episodes": []},
                    "safety": {"red_flags_present": [], "red_flags_absent": []},
                    "patient_perspective": {"ideas": [], "concerns": [],
                                            "expectations": []},
                    "observation_ids": [],
                }],
            },
        ))
        chars = g.get_nodes_by_type(NodeType.CHARACTER)
        assert len(chars) == 1
        assert chars[0].value == "sharp"

        # Edge from instance to character (not from concept)
        inst_nodes = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)
        assert len(inst_nodes) == 1
        edges = g.get_edges_from(inst_nodes[0].node_id)
        char_edges = [e for e in edges if e.edge_type == EdgeType.HAS_CHARACTER]
        assert len(char_edges) == 1
        assert char_edges[0].target_id == chars[0].node_id

    def test_onset_edge(self):
        g = ClinicalGraph()
        ss = {
            "symptom": "headache",
            "spatial": {"site": None, "laterality": None, "radiation": None},
            "qualitative": {"character": None, "intensity": None,
                            "intensity_raw": None, "severity": None},
            "temporal": {"onset": "sudden", "onset_type": None,
                         "duration": None, "pattern": None, "progression": None},
            "modifiers": {"aggravating_factors": [], "relieving_factors": []},
            "context": {"associated_present": [], "associated_absent": [],
                        "prior_episodes": []},
            "safety": {"red_flags_present": [], "red_flags_absent": []},
            "patient_perspective": {"ideas": [], "concerns": [],
                                    "expectations": []},
            "observation_ids": [],
        }
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [ss],
            },
        ))
        onsets = g.get_nodes_by_type(NodeType.ONSET)
        assert len(onsets) == 1
        assert onsets[0].value == "sudden"

    def test_aggravating_modifier_edge(self):
        g = ClinicalGraph()
        ss = {
            "symptom": "headache",
            "spatial": {"site": None, "laterality": None, "radiation": None},
            "qualitative": {"character": None, "intensity": None,
                            "intensity_raw": None, "severity": None},
            "temporal": {"onset": None, "onset_type": None,
                         "duration": None, "pattern": None, "progression": None},
            "modifiers": {
                "aggravating_factors": ["light", "noise"],
                "relieving_factors": ["rest"],
            },
            "context": {"associated_present": [], "associated_absent": [],
                        "prior_episodes": []},
            "safety": {"red_flags_present": [], "red_flags_absent": []},
            "patient_perspective": {"ideas": [], "concerns": [],
                                    "expectations": []},
            "observation_ids": [],
        }
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [ss],
            },
        ))
        mods = g.get_nodes_by_type(NodeType.MODIFIER)
        assert len(mods) == 3  # light, noise, rest
        agg = [m for m in mods if m.attributes.get("modifier_type") == "aggravating"]
        rel = [m for m in mods if m.attributes.get("modifier_type") == "relieving"]
        assert len(agg) == 2
        assert len(rel) == 1

        inst_node = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        edges = g.get_edges_from(inst_node.node_id)
        agg_edges = [e for e in edges if e.edge_type == EdgeType.AGGRAVATED_BY]
        rel_edges = [e for e in edges if e.edge_type == EdgeType.RELIEVED_BY]
        assert len(agg_edges) == 2
        assert len(rel_edges) == 1


class TestSymptomBuilderAssociations:
    def test_associated_with_edge(self):
        """Two symptoms co-occurring produce ASSOCIATED_WITH edges."""
        g = ClinicalGraph()
        ss_head = {
            "symptom": "headache",
            "spatial": {"site": None, "laterality": None, "radiation": None},
            "qualitative": {"character": None, "intensity": None,
                            "intensity_raw": None, "severity": None},
            "temporal": {"onset": None, "onset_type": None,
                         "duration": None, "pattern": None, "progression": None},
            "modifiers": {"aggravating_factors": [], "relieving_factors": []},
            "context": {"associated_present": ["nausea"], "associated_absent": [],
                        "prior_episodes": []},
            "safety": {"red_flags_present": [], "red_flags_absent": []},
            "patient_perspective": {"ideas": [], "concerns": [],
                                    "expectations": []},
            "observation_ids": [],
        }
        ss_naus = {
            "symptom": "nausea",
            "spatial": {"site": None, "laterality": None, "radiation": None},
            "qualitative": {"character": None, "intensity": None,
                            "intensity_raw": None, "severity": None},
            "temporal": {"onset": None, "onset_type": None,
                         "duration": None, "pattern": None, "progression": None},
            "modifiers": {"aggravating_factors": [], "relieving_factors": []},
            "context": {"associated_present": ["headache"], "associated_absent": [],
                        "prior_episodes": []},
            "safety": {"red_flags_present": [], "red_flags_absent": []},
            "patient_perspective": {"ideas": [], "concerns": [],
                                    "expectations": []},
            "observation_ids": [],
        }
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache", "nausea"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [ss_head, ss_naus],
            },
        ))
        sym_nodes = g.get_nodes_by_type(NodeType.SYMPTOM)
        assert len(sym_nodes) == 2

        # headache -> ASSOCIATED_WITH -> nausea
        head_node = [n for n in sym_nodes if n.value == "headache"][0]
        assoc_edges = [
            e for e in g.get_edges_from(head_node.node_id)
            if e.edge_type == EdgeType.ASSOCIATED_WITH
        ]
        assert len(assoc_edges) == 1


class TestSymptomBuilderNegations:
    def test_negation_node_and_edge(self):
        g = ClinicalGraph()
        ss = {
            "symptom": "headache",
            "spatial": {"site": None, "laterality": None, "radiation": None},
            "qualitative": {"character": None, "intensity": None,
                            "intensity_raw": None, "severity": None},
            "temporal": {"onset": None, "onset_type": None,
                         "duration": None, "pattern": None, "progression": None},
            "modifiers": {"aggravating_factors": [], "relieving_factors": []},
            "context": {"associated_present": [],
                        "associated_absent": ["No fever"],
                        "prior_episodes": []},
            "safety": {"red_flags_present": [], "red_flags_absent": []},
            "patient_perspective": {"ideas": [], "concerns": [],
                                    "expectations": []},
            "observation_ids": [],
        }
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [ss],
            },
        ))
        negs = g.get_nodes_by_type(NodeType.NEGATED_SYMPTOM)
        assert len(negs) == 1
        assert negs[0].value == "No fever"

        sym_node = g.get_nodes_by_type(NodeType.SYMPTOM)[0]
        neg_edges = [
            e for e in g.get_edges_from(sym_node.node_id)
            if e.edge_type == EdgeType.NEGATED_BY
        ]
        assert len(neg_edges) == 1


class TestSymptomBuilderICE:
    def test_ice_node_and_edge(self):
        g = ClinicalGraph()
        ss = {
            "symptom": "headache",
            "spatial": {"site": None, "laterality": None, "radiation": None},
            "qualitative": {"character": None, "intensity": None,
                            "intensity_raw": None, "severity": None},
            "temporal": {"onset": None, "onset_type": None,
                         "duration": None, "pattern": None, "progression": None},
            "modifiers": {"aggravating_factors": [], "relieving_factors": []},
            "context": {"associated_present": [], "associated_absent": [],
                        "prior_episodes": []},
            "safety": {"red_flags_present": [], "red_flags_absent": []},
            "patient_perspective": {
                "ideas": ["I think it might be a migraine"],
                "concerns": [],
                "expectations": [],
            },
            "observation_ids": [],
        }
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [ss],
            },
        ))
        ice_nodes = g.get_nodes_by_type(NodeType.ICE_ITEM)
        assert len(ice_nodes) == 1
        assert ice_nodes[0].value == "I think it might be a migraine"
        assert ice_nodes[0].attributes["category"] == "ideas"

        sym_node = g.get_nodes_by_type(NodeType.SYMPTOM)[0]
        ice_edges = [
            e for e in g.get_edges_from(sym_node.node_id)
            if e.edge_type == EdgeType.EXPRESSED_AS
        ]
        assert len(ice_edges) == 1


# ══════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════


class TestBuildClinicalGraph:
    def test_returns_clinical_graph(self):
        graph = build_clinical_graph(_minimal_state())
        assert isinstance(graph, ClinicalGraph)

    def test_empty_state_produces_empty_graph(self):
        graph = build_clinical_graph(_minimal_state())
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_populated_state_produces_nodes(self):
        graph = build_clinical_graph(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001"),
            ],
        ))
        assert graph.node_count >= 1

    def test_serializable(self):
        import json
        graph = build_clinical_graph(_minimal_state(symptoms=["headache"]))
        d = graph.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _minimal_state(
            symptoms=["headache", "nausea"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("symptom", "nausea", "seg_0001", obs_id="obs_0002"),
            ],
        )
        g1 = build_clinical_graph(state)
        g2 = build_clinical_graph(state)
        assert g1.to_dict() == g2.to_dict()


# ══════════════════════════════════════════════════════════════════
# Symptom instance nodes
# ══════════════════════════════════════════════════════════════════


def _ss(symptom: str, **overrides) -> dict:
    """Helper to build a structured_symptom entry with sane defaults."""
    base = {
        "symptom": symptom,
        "spatial": {"site": None, "laterality": None, "radiation": None},
        "qualitative": {"character": None, "intensity": None,
                        "intensity_raw": None, "severity": None},
        "temporal": {"onset": None, "onset_type": None,
                     "duration": None, "pattern": None, "progression": None},
        "modifiers": {"aggravating_factors": [], "relieving_factors": []},
        "context": {"associated_present": [], "associated_absent": [],
                    "prior_episodes": []},
        "safety": {"red_flags_present": [], "red_flags_absent": []},
        "patient_perspective": {"ideas": [], "concerns": [],
                                "expectations": []},
        "observation_ids": [],
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


class TestSymptomInstanceNodes:
    def test_instance_node_created(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [_ss("headache")],
            },
        ))
        instances = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)
        assert len(instances) == 1
        assert instances[0].value == "headache"

    def test_instance_of_edge(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [_ss("headache")],
            },
        ))
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        concept = g.get_nodes_by_type(NodeType.SYMPTOM)[0]
        edges = g.get_edges_from(inst.node_id)
        iof = [e for e in edges if e.edge_type == EdgeType.INSTANCE_OF]
        assert len(iof) == 1
        assert iof[0].target_id == concept.node_id

    def test_instance_evidence_obs_ids(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache", observation_ids=["obs_0001"]),
                ],
            },
        ))
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        assert "obs_0001" in inst.evidence_obs_ids

    def test_qualifier_edges_from_instance_not_concept(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache", qualitative={"character": "throbbing"}),
                ],
            },
        ))
        concept = g.get_nodes_by_type(NodeType.SYMPTOM)[0]
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        # No qualifier edges from concept node
        concept_edges = g.get_edges_from(concept.node_id)
        assert not any(e.edge_type == EdgeType.HAS_CHARACTER for e in concept_edges)
        # Qualifier edge from instance
        inst_edges = g.get_edges_from(inst.node_id)
        char_edges = [e for e in inst_edges if e.edge_type == EdgeType.HAS_CHARACTER]
        assert len(char_edges) == 1

    def test_no_instance_without_structured_symptoms(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [],
            },
        ))
        assert g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE) == []
        assert g.get_nodes_by_type(NodeType.SYMPTOM) != []

    def test_severity_node_and_edge(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache", qualitative={"severity": "severe"}),
                ],
            },
        ))
        sevs = g.get_nodes_by_type(NodeType.SEVERITY)
        assert len(sevs) == 1
        assert sevs[0].value == "severe"
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        sev_edges = [e for e in g.get_edges_from(inst.node_id)
                     if e.edge_type == EdgeType.HAS_SEVERITY]
        assert len(sev_edges) == 1
        assert sev_edges[0].target_id == sevs[0].node_id

    def test_intensity_node_and_edge(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache", qualitative={
                        "intensity": 7, "intensity_raw": "7 out of 10",
                    }),
                ],
            },
        ))
        ints = g.get_nodes_by_type(NodeType.INTENSITY)
        assert len(ints) == 1
        assert ints[0].value == "7"
        assert ints[0].attributes.get("raw_text") == "7 out of 10"
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        int_edges = [e for e in g.get_edges_from(inst.node_id)
                     if e.edge_type == EdgeType.HAS_INTENSITY]
        assert len(int_edges) == 1

    def test_duration_node_and_edge(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache", temporal={"duration": "3 days"}),
                ],
            },
        ))
        durs = g.get_nodes_by_type(NodeType.DURATION)
        assert len(durs) == 1
        assert durs[0].value == "3 days"
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        dur_edges = [e for e in g.get_edges_from(inst.node_id)
                     if e.edge_type == EdgeType.HAS_DURATION]
        assert len(dur_edges) == 1

    def test_radiation_node_and_edge(self):
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["chest pain"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("chest pain", spatial={"radiation": "left arm"}),
                ],
            },
        ))
        rads = g.get_nodes_by_type(NodeType.RADIATION)
        assert len(rads) == 1
        assert rads[0].value == "left arm"
        inst = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)[0]
        rad_edges = [e for e in g.get_edges_from(inst.node_id)
                     if e.edge_type == EdgeType.HAS_RADIATION]
        assert len(rad_edges) == 1

    def test_determinism_with_instances(self):
        state = _minimal_state(
            symptoms=["headache", "nausea"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("symptom", "nausea", "seg_0001", obs_id="obs_0002"),
            ],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache", qualitative={"character": "throbbing"}),
                    _ss("nausea"),
                ],
            },
        )
        g1 = build_clinical_graph(state)
        g2 = build_clinical_graph(state)
        assert g1.to_dict() == g2.to_dict()

    def test_instance_ids_follow_structured_order(self):
        """Instance IDs are deterministic from structured_symptoms ordering."""
        g = ClinicalGraph()
        build_symptom_graph(g, _minimal_state(
            symptoms=["headache", "nausea"],
            derived={
                "red_flags": [],
                "symptom_representations": [],
                "structured_symptoms": [
                    _ss("headache"),
                    _ss("nausea"),
                ],
            },
        ))
        instances = g.get_nodes_by_type(NodeType.SYMPTOM_INSTANCE)
        assert len(instances) == 2
        assert instances[0].node_id == "inst_0001"
        assert instances[0].value == "headache"
        assert instances[1].node_id == "inst_0002"
        assert instances[1].value == "nausea"
