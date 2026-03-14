"""Symptom-domain graph builder.

Reads structured outputs already produced by the pipeline and populates
the clinical graph with symptom nodes, qualifier nodes, and the edges
that link them.

Conservative linking only — same-segment evidence required.  Never
fabricates relationships.  Always attaches evidence_obs_ids when possible.

Pure function — no extraction logic, no I/O.
"""

from __future__ import annotations

from app.graph.models import Node, Edge
from app.graph.graph import ClinicalGraph
from app.graph.types import NodeType, EdgeType


def build_symptom_graph(graph: ClinicalGraph, clinical_state: dict) -> None:
    """Populate *graph* with symptom-domain nodes and edges.

    Reads from:
        - ``state["symptoms"]``
        - ``state["observations"]``
        - ``state["qualifiers"]``
        - ``state["sites"]``
        - ``state["ice"]``
        - ``state["negations"]``
        - ``state["derived"]["structured_symptoms"]``

    Nodes created:
        - One ``symptom`` node per extracted symptom
        - Qualifier nodes (site, character, onset, pattern, progression)
        - Modifier nodes (aggravating/relieving factors)
        - Negated symptom nodes
        - ICE item nodes (same-segment only)
        - Associated symptom nodes (same-segment only)

    Edges link symptoms to their qualifiers and modifiers using
    conservative same-segment evidence rules.

    Args:
        graph: :class:`ClinicalGraph` to populate (mutated in place).
        clinical_state: dict produced by :func:`build_clinical_state`.
    """
    symptoms: list[str] = clinical_state.get("symptoms", [])
    if not symptoms:
        return

    observations: list[dict] = clinical_state.get("observations", [])
    structured: list[dict] = (
        clinical_state.get("derived", {}).get("structured_symptoms", [])
    )
    qualifiers: list[dict] = clinical_state.get("qualifiers", [])

    # Build observation-id index: symptom (lower) -> list of obs_ids
    obs_index: dict[str, list[str]] = {}
    for obs in observations:
        if obs.get("finding_type") == "symptom":
            key = obs.get("value", "").lower()
            obs_index.setdefault(key, []).append(obs["observation_id"])

    # Build qualifier index: symptom (lower) -> qualifiers dict
    qual_index: dict[str, dict] = {}
    for entry in qualifiers:
        key = entry.get("symptom", "").lower()
        if key and key not in qual_index:
            qual_index[key] = entry.get("qualifiers", {})

    # ID counters
    _counters: dict[str, int] = {}

    def _next_id(prefix: str) -> str:
        _counters[prefix] = _counters.get(prefix, 0) + 1
        return f"{prefix}_{_counters[prefix]:04d}"

    # 1. Create symptom nodes
    sym_node_ids: dict[str, str] = {}
    for symptom in symptoms:
        node_id = _next_id("sym")
        sym_node_ids[symptom.lower()] = node_id
        graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.SYMPTOM,
            value=symptom,
            evidence_obs_ids=list(obs_index.get(symptom.lower(), [])),
        ))

    # 2. Create qualifier/modifier nodes and edges from structured symptoms
    for ss in structured:
        symptom = ss.get("symptom", "")
        sym_key = symptom.lower()
        sym_nid = sym_node_ids.get(sym_key)
        if sym_nid is None:
            continue

        sym_obs = obs_index.get(sym_key, [])

        # Spatial: site
        site = ss.get("spatial", {}).get("site")
        if site:
            nid = _next_id("site")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.SITE,
                value=site,
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_SITE,
                evidence_obs_ids=list(sym_obs),
            ))

        # Qualitative: character
        character = ss.get("qualitative", {}).get("character")
        if character:
            nid = _next_id("char")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.CHARACTER,
                value=character,
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_CHARACTER,
                evidence_obs_ids=list(sym_obs),
            ))

        # Temporal: onset
        onset = ss.get("temporal", {}).get("onset")
        if onset:
            nid = _next_id("onset")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.ONSET,
                value=onset,
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_ONSET,
                evidence_obs_ids=list(sym_obs),
            ))

        # Temporal: pattern
        pattern = ss.get("temporal", {}).get("pattern")
        if pattern:
            nid = _next_id("pat")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.PATTERN,
                value=pattern,
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_PATTERN,
                evidence_obs_ids=list(sym_obs),
            ))

        # Temporal: progression
        progression = ss.get("temporal", {}).get("progression")
        if progression:
            nid = _next_id("prog")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.PROGRESSION,
                value=progression,
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_PROGRESSION,
                evidence_obs_ids=list(sym_obs),
            ))

        # Modifiers: aggravating factors
        for factor in ss.get("modifiers", {}).get("aggravating_factors", []):
            nid = _next_id("mod")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.MODIFIER,
                value=factor,
                attributes={"modifier_type": "aggravating"},
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.AGGRAVATED_BY,
                evidence_obs_ids=list(sym_obs),
            ))

        # Modifiers: relieving factors
        for factor in ss.get("modifiers", {}).get("relieving_factors", []):
            nid = _next_id("mod")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.MODIFIER,
                value=factor,
                attributes={"modifier_type": "relieving"},
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.RELIEVED_BY,
                evidence_obs_ids=list(sym_obs),
            ))

        # Context: associated_present (same-segment symptoms)
        for assoc in ss.get("context", {}).get("associated_present", []):
            assoc_nid = sym_node_ids.get(assoc.lower())
            if assoc_nid and assoc_nid != sym_nid:
                # Create edge directly between existing symptom nodes
                graph.add_edge(Edge(
                    edge_id=_next_id("e"),
                    source_id=sym_nid,
                    target_id=assoc_nid,
                    edge_type=EdgeType.ASSOCIATED_WITH,
                    evidence_obs_ids=list(sym_obs),
                ))

        # Context: associated_absent (same-segment negations)
        for neg_text in ss.get("context", {}).get("associated_absent", []):
            nid = _next_id("neg")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.NEGATED_SYMPTOM,
                value=neg_text,
                evidence_obs_ids=list(sym_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=sym_nid,
                target_id=nid,
                edge_type=EdgeType.NEGATED_BY,
                evidence_obs_ids=list(sym_obs),
            ))

        # Patient perspective: ICE items (same-segment only)
        pp = ss.get("patient_perspective", {})
        for category in ("ideas", "concerns", "expectations"):
            for text in pp.get(category, []):
                nid = _next_id("ice")
                graph.add_node(Node(
                    node_id=nid,
                    node_type=NodeType.ICE_ITEM,
                    value=text,
                    attributes={"category": category},
                    evidence_obs_ids=list(sym_obs),
                ))
                graph.add_edge(Edge(
                    edge_id=_next_id("e"),
                    source_id=sym_nid,
                    target_id=nid,
                    edge_type=EdgeType.EXPRESSED_AS,
                    evidence_obs_ids=list(sym_obs),
                ))
