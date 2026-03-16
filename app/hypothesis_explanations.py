"""Hypothesis explanations — transparent reasoning for each hypothesis.

Builds a deterministic explanation object for each hypothesis from its
existing evidence, score, and rank.  No new inference — only assembles
existing data into a human-readable structure.

Pure function — no I/O, no ML, no input mutation.
"""

from __future__ import annotations


def build_hypothesis_explanations(
    hypotheses: list[dict],
    observations: list[dict],
) -> list[dict]:
    """Add explanation objects to each hypothesis.

    Args:
        hypotheses: ranked hypothesis dicts with structured evidence.
        observations: enriched observation dicts.

    Returns:
        New list of new dicts with added ``explanation`` field.
    """
    obs_index: dict[str, str] = {
        obs["observation_id"]: obs.get("value", "")
        for obs in observations
        if obs.get("observation_id")
    }

    result: list[dict] = []
    for hyp in hypotheses:
        new = dict(hyp)
        rank = hyp.get("rank", 0)
        score = hyp.get("score", 0)

        supporting_values = _extract_values(
            hyp.get("supporting_observations", []), obs_index,
        )
        conflicting_values = _extract_values(
            hyp.get("conflicting_observations", []), obs_index,
        )

        new["explanation"] = {
            "summary": f"Hypothesis ranked #{rank} based on supporting clinical evidence.",
            "supporting_evidence": supporting_values,
            "conflicting_evidence": conflicting_values,
            "score": score,
            "rank": rank,
        }
        result.append(new)

    return result


def _extract_values(
    evidence_list: list,
    obs_index: dict[str, str],
) -> list[str]:
    """Extract observation values from structured evidence objects."""
    values: list[str] = []
    for ev in evidence_list:
        if isinstance(ev, dict):
            obs_id = ev.get("observation_id", "")
        else:
            obs_id = ev
        value = obs_index.get(obs_id, "")
        if value:
            values.append(value)
    return values
