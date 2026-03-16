"""Hypothesis ranking — deterministic scoring and ranking of hypotheses.

Scores each hypothesis based on supporting and conflicting evidence
strength weights, then assigns ranks by descending score.

Pure function — no I/O, no ML, no input mutation.
"""

from __future__ import annotations

_STRENGTH_WEIGHTS: dict[str, int] = {
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


def get_strength_weight(strength: str) -> int:
    """Return numeric weight for a strength level.

    Args:
        strength: ``"weak"``, ``"moderate"``, or ``"strong"``.

    Returns:
        Integer weight (1, 2, or 3).  Defaults to 0 for unknown values.
    """
    return _STRENGTH_WEIGHTS.get(strength, 0)


def rank_hypotheses(hypotheses: list[dict]) -> list[dict]:
    """Score and rank hypotheses by evidence strength.

    Score = sum(supporting weights) - sum(conflicting weights).
    Higher score ranks first.  Equal scores preserve input order.
    Rank starts at 1.

    Args:
        hypotheses: hypothesis dicts with structured evidence objects.

    Returns:
        New list of new dicts with added ``score`` and ``rank`` fields,
        sorted by descending score (stable).
    """
    if not hypotheses:
        return []

    scored: list[tuple[int, int, dict]] = []
    for i, hyp in enumerate(hypotheses):
        score = _compute_score(hyp)
        scored.append((score, i, hyp))

    # Sort by descending score; equal scores preserve original order (stable)
    scored.sort(key=lambda x: (-x[0], x[1]))

    result: list[dict] = []
    for rank, (score, _orig_idx, hyp) in enumerate(scored, start=1):
        new = dict(hyp)
        new["score"] = score
        new["rank"] = rank
        result.append(new)

    return result


def _compute_score(hyp: dict) -> int:
    """Compute evidence score for a hypothesis."""
    supporting = sum(
        get_strength_weight(ev.get("strength", ""))
        for ev in hyp.get("supporting_observations", [])
        if isinstance(ev, dict)
    )
    conflicting = sum(
        get_strength_weight(ev.get("strength", ""))
        for ev in hyp.get("conflicting_observations", [])
        if isinstance(ev, dict)
    )
    return supporting - conflicting
