"""Diagnostic hypotheses — candidate conditions from diagnostic hints.

Builds hypothesis records from existing ``diagnostic_hints``, linking
each to supporting observations and related problems.  v1: conflicting
observations are always empty.

Pure function — no I/O, no ML, no LLM, no input mutation.
"""

from __future__ import annotations


def build_diagnostic_hypotheses(clinical_state: dict) -> list[dict]:
    """Build diagnostic hypotheses from clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of hypothesis dicts, deterministic order matching
        ``diagnostic_hints`` order.
    """
    hints: list[dict] = clinical_state.get("diagnostic_hints", [])
    observations: list[dict] = clinical_state.get("observations", [])
    problems: list[dict] = clinical_state.get("problems", [])

    # Index: symptom value (lowercase) → list of observation IDs
    obs_index: dict[str, list[str]] = {}
    for obs in observations:
        if obs.get("finding_type") == "symptom":
            key = obs.get("value", "").lower()
            obs_id = obs.get("observation_id", "")
            if key and obs_id:
                obs_index.setdefault(key, []).append(obs_id)

    # Index: observation ID → set of problem IDs that reference it
    obs_to_problems: dict[str, list[str]] = {}
    for prob in problems:
        prob_id = prob.get("id", "")
        for obs_id in prob.get("observations", []):
            obs_to_problems.setdefault(obs_id, []).append(prob_id)

    hypotheses: list[dict] = []
    for i, hint in enumerate(hints):
        condition = hint.get("condition", "")
        if not condition:
            continue

        evidence: list[str] = hint.get("evidence", [])
        supporting = _collect_obs_ids(evidence, obs_index)
        related = _collect_related_problems(supporting, obs_to_problems)
        confidence = _compute_confidence(len(supporting))

        hypotheses.append({
            "id": f"hyp_{i + 1:04d}",
            "title": condition,
            "status": "candidate",
            "supporting_observations": supporting,
            "conflicting_observations": [],
            "related_problems": related,
            "confidence": confidence,
        })

    return hypotheses


def _collect_obs_ids(
    evidence: list[str],
    obs_index: dict[str, list[str]],
) -> list[str]:
    """Collect unique observation IDs for evidence symptom names."""
    seen: set[str] = set()
    result: list[str] = []
    for item in evidence:
        for obs_id in obs_index.get(item.lower(), []):
            if obs_id not in seen:
                seen.add(obs_id)
                result.append(obs_id)
    return result


def _collect_related_problems(
    obs_ids: list[str],
    obs_to_problems: dict[str, list[str]],
) -> list[str]:
    """Collect unique problem IDs related to the given observations."""
    seen: set[str] = set()
    result: list[str] = []
    for obs_id in obs_ids:
        for prob_id in obs_to_problems.get(obs_id, []):
            if prob_id not in seen:
                seen.add(prob_id)
                result.append(prob_id)
    return result


def _compute_confidence(supporting_count: int) -> str:
    """Determine confidence level from supporting observation count."""
    if supporting_count >= 5:
        return "strong"
    if supporting_count >= 3:
        return "moderate"
    return "low"
