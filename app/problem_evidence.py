"""Problem evidence — annotate problems with evidence references.

Adds ``supporting_observations`` and ``conflicting_observations`` to each
problem dict.  v1: supporting mirrors existing ``observations`` field;
conflicting is always empty.

Pure function — no I/O, no side effects, no input mutation.
"""

from __future__ import annotations


def annotate_problem_evidence(
    problems: list[dict],
    observations: list[dict],
) -> list[dict]:
    """Add evidence references to each problem.

    Args:
        problems: problem dicts from ``build_problem_list()``.
        observations: enriched observation dicts.

    Returns:
        New list of new dicts with added ``supporting_observations``
        and ``conflicting_observations`` fields.  All original fields
        are preserved unchanged.
    """
    # Build set of valid observation IDs for validation
    valid_ids: set[str] = {
        obs.get("observation_id", "")
        for obs in observations
        if obs.get("observation_id")
    }

    result: list[dict] = []
    for prob in problems:
        new = dict(prob)
        # Supporting = existing linked observations, validated against layer
        linked = prob.get("observations", [])
        new["supporting_observations"] = [
            oid for oid in linked if oid in valid_ids
        ]
        new["conflicting_observations"] = []
        result.append(new)
    return result
