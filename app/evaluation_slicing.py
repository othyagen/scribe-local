"""Evaluation slicing — group scored results by case metadata.

Groups scored case results by metadata fields and computes average
hypothesis hit rates per group.

Pure functions — no I/O, no mutation, deterministic.
"""

from __future__ import annotations

from collections import defaultdict

from app.case_schema import extract_case_metadata

_VALID_KEYS = frozenset({
    "organ_system", "presenting_complaint", "difficulty", "origin", "tag",
})

_LIST_FIELDS = {
    "organ_system": "organ_systems",
    "presenting_complaint": "presenting_complaints",
    "tag": "tags",
}

_SCALAR_FIELDS = {
    "difficulty": "difficulty",
    "origin": "origin",
}


def slice_evaluation(results: list[dict], key: str) -> dict[str, float]:
    """Group scored results by a metadata key and average hypothesis_hit_rate.

    Args:
        results: list of dicts, each with ``"case"`` (case dict) and
            ``"score"`` (from :func:`score_result_against_ground_truth`).
        key: one of ``"organ_system"``, ``"presenting_complaint"``,
            ``"difficulty"``, ``"origin"``, ``"tag"``.

    Returns:
        ``{group_value: avg_hypothesis_hit_rate}``.

    Raises:
        ValueError: if *key* is not a recognised slicing key.
    """
    if key not in _VALID_KEYS:
        raise ValueError(f"unknown slicing key {key!r}, expected one of {sorted(_VALID_KEYS)}")

    if not results:
        return {}

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for entry in results:
        case = entry.get("case") or {}
        score = entry.get("score") or {}
        hit_rate = (score.get("summary") or {}).get("hypothesis_hit_rate", 0.0)
        meta = extract_case_metadata(case)

        if key in _LIST_FIELDS:
            values = meta.get(_LIST_FIELDS[key]) or []
            if not values:
                values = ["unknown"]
            for v in values:
                totals[v] += hit_rate
                counts[v] += 1
        else:
            value = meta.get(_SCALAR_FIELDS[key]) or "unknown"
            totals[value] += hit_rate
            counts[value] += 1

    return {group: totals[group] / counts[group] for group in totals}
