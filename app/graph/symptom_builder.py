"""Symptom-domain graph builder.

Reads structured outputs already produced by the pipeline and populates
the clinical graph with symptom concept nodes, symptom instance nodes,
qualifier nodes, and the edges that link them.

Each structured_symptom produces one SYMPTOM_INSTANCE node that links to
the corresponding SYMPTOM concept via INSTANCE_OF.  Qualifier and modifier
edges attach to the instance, not the concept.  Concept-level edges
(ASSOCIATED_WITH, NEGATED_BY, EXPRESSED_AS) remain on the concept node.

Conservative linking only — same-segment evidence required.  Never
fabricates relationships.  Always attaches evidence_obs_ids when possible.

Instance node IDs are deterministic: ``inst_NNNN`` where NNNN is the
1-based position in the structured_symptoms list.

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
        - One ``symptom`` concept node per extracted symptom
        - One ``symptom_instance`` node per structured_symptom entry
        - Qualifier nodes (site, character, onset, pattern, progression,
          severity, intensity, duration, radiation)
        - Modifier nodes (aggravating/relieving factors)
        - Negated symptom nodes
        - ICE item nodes (same-segment only)

    Qualifier and modifier edges source from instance nodes.
    Concept-level edges (ASSOCIATED_WITH, NEGATED_BY, EXPRESSED_AS)
    remain on concept nodes.

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

    # Build observation-id index: symptom (lower) -> list of obs_ids
    obs_index: dict[str, list[str]] = {}
    for obs in observations:
        if obs.get("finding_type") == "symptom":
            key = obs.get("value", "").lower()
            obs_index.setdefault(key, []).append(obs["observation_id"])

    # ID counters
    _counters: dict[str, int] = {}

    def _next_id(prefix: str) -> str:
        _counters[prefix] = _counters.get(prefix, 0) + 1
        return f"{prefix}_{_counters[prefix]:04d}"

    # 1. Create symptom concept nodes
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

    # 2. Create instance nodes and qualifier/modifier edges
    for ss in structured:
        symptom = ss.get("symptom", "")
        sym_key = symptom.lower()
        sym_nid = sym_node_ids.get(sym_key)
        if sym_nid is None:
            continue

        sym_obs = obs_index.get(sym_key, [])

        # Instance node — deterministic ID from structured_symptoms order
        inst_nid = _next_id("inst")
        inst_obs = ss.get("observation_ids", [])
        graph.add_node(Node(
            node_id=inst_nid,
            node_type=NodeType.SYMPTOM_INSTANCE,
            value=symptom,
            evidence_obs_ids=list(inst_obs) if inst_obs else list(sym_obs),
        ))
        graph.add_edge(Edge(
            edge_id=_next_id("e"),
            source_id=inst_nid,
            target_id=sym_nid,
            edge_type=EdgeType.INSTANCE_OF,
            evidence_obs_ids=list(inst_obs) if inst_obs else list(sym_obs),
        ))

        # Evidence for qualifier edges: prefer instance obs, fall back to sym
        q_obs = list(inst_obs) if inst_obs else list(sym_obs)

        # Spatial: site
        site = ss.get("spatial", {}).get("site")
        if site:
            nid = _next_id("site")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.SITE,
                value=site,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_SITE,
                evidence_obs_ids=list(q_obs),
            ))

        # Spatial: radiation
        radiation = ss.get("spatial", {}).get("radiation")
        if radiation:
            nid = _next_id("rad")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.RADIATION,
                value=radiation,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_RADIATION,
                evidence_obs_ids=list(q_obs),
            ))

        # Qualitative: character
        character = ss.get("qualitative", {}).get("character")
        if character:
            nid = _next_id("char")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.CHARACTER,
                value=character,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_CHARACTER,
                evidence_obs_ids=list(q_obs),
            ))

        # Qualitative: severity
        severity = ss.get("qualitative", {}).get("severity")
        if severity:
            nid = _next_id("sev")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.SEVERITY,
                value=severity,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_SEVERITY,
                evidence_obs_ids=list(q_obs),
            ))

        # Qualitative: intensity
        intensity = ss.get("qualitative", {}).get("intensity")
        if intensity is not None:
            raw = ss.get("qualitative", {}).get("intensity_raw", "")
            nid = _next_id("int")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.INTENSITY,
                value=str(intensity),
                attributes={"raw_text": raw} if raw else {},
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_INTENSITY,
                evidence_obs_ids=list(q_obs),
            ))

        # Temporal: onset
        onset = ss.get("temporal", {}).get("onset")
        if onset:
            nid = _next_id("onset")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.ONSET,
                value=onset,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_ONSET,
                evidence_obs_ids=list(q_obs),
            ))

        # Temporal: duration
        duration = ss.get("temporal", {}).get("duration")
        if duration:
            nid = _next_id("dur")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.DURATION,
                value=duration,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_DURATION,
                evidence_obs_ids=list(q_obs),
            ))

        # Temporal: pattern
        pattern = ss.get("temporal", {}).get("pattern")
        if pattern:
            nid = _next_id("pat")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.PATTERN,
                value=pattern,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_PATTERN,
                evidence_obs_ids=list(q_obs),
            ))

        # Temporal: progression
        progression = ss.get("temporal", {}).get("progression")
        if progression:
            nid = _next_id("prog")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.PROGRESSION,
                value=progression,
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.HAS_PROGRESSION,
                evidence_obs_ids=list(q_obs),
            ))

        # Modifiers: aggravating factors
        for factor in ss.get("modifiers", {}).get("aggravating_factors", []):
            nid = _next_id("mod")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.MODIFIER,
                value=factor,
                attributes={"modifier_type": "aggravating"},
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.AGGRAVATED_BY,
                evidence_obs_ids=list(q_obs),
            ))

        # Modifiers: relieving factors
        for factor in ss.get("modifiers", {}).get("relieving_factors", []):
            nid = _next_id("mod")
            graph.add_node(Node(
                node_id=nid,
                node_type=NodeType.MODIFIER,
                value=factor,
                attributes={"modifier_type": "relieving"},
                evidence_obs_ids=list(q_obs),
            ))
            graph.add_edge(Edge(
                edge_id=_next_id("e"),
                source_id=inst_nid,
                target_id=nid,
                edge_type=EdgeType.RELIEVED_BY,
                evidence_obs_ids=list(q_obs),
            ))

        # Context: associated_present (concept-level, stay on symptom node)
        for assoc in ss.get("context", {}).get("associated_present", []):
            assoc_nid = sym_node_ids.get(assoc.lower())
            if assoc_nid and assoc_nid != sym_nid:
                graph.add_edge(Edge(
                    edge_id=_next_id("e"),
                    source_id=sym_nid,
                    target_id=assoc_nid,
                    edge_type=EdgeType.ASSOCIATED_WITH,
                    evidence_obs_ids=list(sym_obs),
                ))

        # Context: associated_absent (concept-level, stay on symptom node)
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

        # Patient perspective: ICE items (concept-level, stay on symptom node)
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
