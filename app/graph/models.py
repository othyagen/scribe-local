"""Dataclass definitions for clinical graph nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """A node in the clinical graph.

    Attributes:
        node_id: unique identifier (e.g. ``sym_0001``, ``site_0001``).
        node_type: type constant from :class:`NodeType`.
        value: the clinical value this node represents (e.g. ``"headache"``).
        attributes: arbitrary key-value pairs for domain-specific data.
        evidence_obs_ids: observation IDs that support this node's existence.
    """

    node_id: str
    node_type: str
    value: str
    attributes: dict = field(default_factory=dict)
    evidence_obs_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "value": self.value,
            "attributes": dict(self.attributes),
            "evidence_obs_ids": list(self.evidence_obs_ids),
        }


@dataclass
class Edge:
    """A directed edge in the clinical graph.

    Attributes:
        edge_id: unique identifier (e.g. ``e_0001``).
        source_id: node_id of the source node.
        target_id: node_id of the target node.
        edge_type: type constant from :class:`EdgeType`.
        attributes: arbitrary key-value pairs for domain-specific data.
        evidence_obs_ids: observation IDs that support this relationship.
    """

    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    attributes: dict = field(default_factory=dict)
    evidence_obs_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "attributes": dict(self.attributes),
            "evidence_obs_ids": list(self.evidence_obs_ids),
        }
