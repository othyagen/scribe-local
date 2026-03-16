"""Observation taxonomy — category enrichment for observation records.

Adds standardised ``category``, ``attributes``, and ``confidence`` fields
to raw observations produced by ``observation_layer.py``.  Pure post-
extraction enrichment — extraction logic is untouched.

Invariants:
  - All original observation fields are preserved unchanged.
  - ``category`` is always present on every observation (``None`` for
    finding_types that are modifiers, not primary categories).
  - Input list is never mutated; returns new list of new dicts.
  - Deterministic — same input always produces same output.
"""

from __future__ import annotations

# ── taxonomy constants ──────────────────────────────────────────────

OBSERVATION_CATEGORIES: frozenset[str] = frozenset({
    "symptom",
    "clinical_sign",
    "vital",
    "laboratory",
    "microbiology",
    "imaging",
    "waveform",
    "functional_test",
    "diagnosis",
    "risk_factor",
    "medication",
    "allergy",
    "family_history",
    "social_history",
    "device",
    "pregnancy_status",
    "administrative",
})

FINDING_TYPE_TO_CATEGORY: dict[str, str | None] = {
    "symptom": "symptom",
    "medication": "medication",
    "negation": None,
    "duration": None,
}


# ── enrichment ──────────────────────────────────────────────────────

def enrich_observations(observations: list[dict]) -> list[dict]:
    """Add taxonomy fields to each observation.

    Args:
        observations: raw observation dicts from ``build_observation_layer()``.

    Returns:
        New list of new dicts, each with added ``category``,
        ``attributes``, and ``confidence`` fields.
    """
    enriched: list[dict] = []
    for obs in observations:
        new = dict(obs)
        new["category"] = FINDING_TYPE_TO_CATEGORY.get(
            obs.get("finding_type"), None,
        )
        new["attributes"] = {}
        new["confidence"] = None
        enriched.append(new)
    return enriched
