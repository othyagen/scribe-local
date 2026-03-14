"""Minimal clinical graph container.

Deterministic, serializable graph with typed nodes and edges.
Designed for additive construction — nodes and edges are appended,
never removed.
"""

from __future__ import annotations

from app.graph.models import Node, Edge


class ClinicalGraph:
    """Lightweight directed graph for clinical evidence.

    Nodes and edges are stored in insertion order.  Lookup indexes
    are maintained for efficient querying by type and adjacency.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        # Indexes
        self._nodes_by_type: dict[str, list[str]] = {}
        self._edges_from: dict[str, list[str]] = {}
        self._edges_to: dict[str, list[str]] = {}

    # ── mutation ──────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Raises ``ValueError`` if a node with the same ``node_id``
        already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self._nodes[node.node_id] = node
        self._nodes_by_type.setdefault(node.node_type, []).append(node.node_id)

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge to the graph.

        Raises ``ValueError`` if an edge with the same ``edge_id``
        already exists, or if source/target nodes are missing.
        """
        if edge.edge_id in self._edges:
            raise ValueError(f"Duplicate edge_id: {edge.edge_id}")
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node not found: {edge.source_id}")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node not found: {edge.target_id}")
        self._edges[edge.edge_id] = edge
        self._edges_from.setdefault(edge.source_id, []).append(edge.edge_id)
        self._edges_to.setdefault(edge.target_id, []).append(edge.edge_id)

    # ── queries ──────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Node | None:
        """Return a node by ID, or ``None``."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Edge | None:
        """Return an edge by ID, or ``None``."""
        return self._edges.get(edge_id)

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Return all nodes of a given type, in insertion order."""
        return [
            self._nodes[nid]
            for nid in self._nodes_by_type.get(node_type, [])
        ]

    def get_edges_from(self, node_id: str) -> list[Edge]:
        """Return all outgoing edges from a node."""
        return [
            self._edges[eid]
            for eid in self._edges_from.get(node_id, [])
        ]

    def get_edges_to(self, node_id: str) -> list[Edge]:
        """Return all incoming edges to a node."""
        return [
            self._edges[eid]
            for eid in self._edges_to.get(node_id, [])
        ]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ── serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the entire graph to a plain dict.

        Returns a dict with ``nodes`` and ``edges`` lists, each
        containing serialized node/edge dicts in insertion order.
        """
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }
