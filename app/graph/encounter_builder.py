"""Encounter-domain graph builder.

Creates ENCOUNTER nodes from ``state["encounters"]``.  No edges in v1 —
observation ID references stay in the encounter dict in clinical_state.

Pure function — no I/O.
"""

from __future__ import annotations

from app.graph.models import Node
from app.graph.graph import ClinicalGraph
from app.graph.types import NodeType


def build_encounter_graph(graph: ClinicalGraph, clinical_state: dict) -> None:
    """Populate *graph* with encounter nodes.

    Reads from ``state["encounters"]`` and creates one ENCOUNTER node
    per encounter.

    Args:
        graph: :class:`ClinicalGraph` to populate (mutated in place).
        clinical_state: dict produced by :func:`build_clinical_state`.
    """
    encounters = clinical_state.get("encounters", [])
    if not encounters:
        return

    _counters: dict[str, int] = {}

    def _next_id(prefix: str) -> str:
        _counters[prefix] = _counters.get(prefix, 0) + 1
        return f"{prefix}_{_counters[prefix]:04d}"

    for enc in encounters:
        node_id = _next_id("enc")
        obs_ids = enc.get("observations", [])
        graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.ENCOUNTER,
            value=enc.get("id", ""),
            attributes={
                "type": enc.get("type", ""),
                "modality": enc.get("modality", ""),
                "timestamp": enc.get("timestamp", ""),
                "observation_count": len(obs_ids),
            },
        ))
