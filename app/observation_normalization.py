"""Observation normalization — link qualifier observations to symptoms.

Runs after taxonomy enrichment.  Links negation and duration observations
to symptom observations in the same segment via ``attributes``.  Original
observations are preserved unchanged — normalization is additive only.

Pure function — no I/O, no side effects, no input mutation.
"""

from __future__ import annotations


def normalize_observations(observations: list[dict]) -> list[dict]:
    """Add qualifier attributes to symptom observations by segment.

    For each segment, if a symptom observation co-occurs with a negation
    or duration observation, the symptom's ``attributes`` dict is enriched:

    - ``negated = True`` if a negation observation exists in the segment.
    - ``duration = <value>`` if a duration observation exists in the segment.

    Args:
        observations: enriched observation dicts (post-taxonomy).

    Returns:
        New list of new dicts.  Qualifier observations are unchanged;
        symptom observations may have enriched ``attributes``.
    """
    if not observations:
        return []

    # Index qualifiers by seg_id
    negation_segs: set[str | None] = set()
    duration_by_seg: dict[str | None, str] = {}

    for obs in observations:
        ft = obs.get("finding_type")
        seg = obs.get("seg_id")
        if ft == "negation":
            negation_segs.add(seg)
        elif ft == "duration":
            # First duration per segment wins
            if seg not in duration_by_seg:
                duration_by_seg[seg] = obs.get("value", "")

    result: list[dict] = []
    for obs in observations:
        new = dict(obs)
        # Deep-copy attributes so we never mutate the input dict's attrs
        new["attributes"] = dict(obs.get("attributes") or {})

        if obs.get("finding_type") == "symptom":
            seg = obs.get("seg_id")
            if seg in negation_segs:
                new["attributes"]["negated"] = True
            if seg in duration_by_seg:
                new["attributes"]["duration"] = duration_by_seg[seg]

        result.append(new)
    return result
