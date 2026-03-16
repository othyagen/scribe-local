"""Problem-domain graph builder.

Creates PROBLEM nodes from ``state["problems"]``.  No edges in v1 —
observation ID references stay in the problem dict in clinical_state.

Pure function — no I/O.
"""

from __future__ import annotations

from app.graph.models import Node
from app.graph.graph import ClinicalGraph
from app.graph.types import NodeType


def build_problem_graph(graph: ClinicalGraph, clinical_state: dict) -> None:
    """Populate *graph* with problem nodes.

    Reads from ``state["problems"]`` and creates one PROBLEM node
    per problem.

    Args:
        graph: :class:`ClinicalGraph` to populate (mutated in place).
        clinical_state: dict produced by :func:`build_clinical_state`.
    """
    problems = clinical_state.get("problems", [])
    if not problems:
        return

    _counters: dict[str, int] = {}

    def _next_id(prefix: str) -> str:
        _counters[prefix] = _counters.get(prefix, 0) + 1
        return f"{prefix}_{_counters[prefix]:04d}"

    for prob in problems:
        node_id = _next_id("prob")
        obs_ids = prob.get("observations", [])
        graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.PROBLEM,
            value=prob.get("id", ""),
            attributes={
                "title": prob.get("title", ""),
                "kind": prob.get("kind", ""),
                "status": prob.get("status", ""),
                "priority": prob.get("priority", ""),
                "observation_count": len(obs_ids),
            },
        ))
