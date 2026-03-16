"""Evidence strength — deterministic strength annotation for evidence items.

Converts flat observation ID lists in ``supporting_observations`` and
``conflicting_observations`` into structured evidence objects carrying
``observation_id``, ``kind``, and ``strength``.

Pure function — no I/O, no ML, no input mutation.
"""

from __future__ import annotations

# ── category → strength mapping ─────────────────────────────────────

CATEGORY_STRENGTH: dict[str | None, str] = {
    "symptom": "weak",
    "clinical_sign": "moderate",
    "vital": "strong",
    "laboratory": "strong",
    "microbiology": "strong",
    "imaging": "strong",
    "waveform": "strong",
    "functional_test": "strong",
    "diagnosis": "strong",
    "risk_factor": "weak",
    "medication": "weak",
    "allergy": "moderate",
    "family_history": "weak",
    "social_history": "weak",
    "device": "moderate",
    "pregnancy_status": "moderate",
    "administrative": "weak",
    None: "weak",
}


def get_evidence_strength(observation: dict) -> str:
    """Return strength level for an observation based on its category.

    Args:
        observation: enriched observation dict with ``category`` field.

    Returns:
        ``"weak"``, ``"moderate"``, or ``"strong"``.
    """
    return CATEGORY_STRENGTH.get(observation.get("category"), "weak")


def annotate_evidence_strength(
    problems: list[dict],
    observations: list[dict],
    hypotheses: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Convert evidence ID lists into structured evidence objects.

    Args:
        problems: problem dicts with ``supporting_observations`` and
            ``conflicting_observations`` as lists of observation IDs.
        observations: enriched observation dicts.
        hypotheses: hypothesis dicts with same evidence fields.

    Returns:
        Tuple of (new_problems, new_hypotheses) with evidence fields
        converted to structured objects.
    """
    obs_index: dict[str, dict] = {
        obs["observation_id"]: obs
        for obs in observations
        if obs.get("observation_id")
    }

    new_problems = [
        _annotate_item(prob, obs_index) for prob in problems
    ]
    new_hypotheses = [
        _annotate_item(hyp, obs_index) for hyp in hypotheses
    ]
    return new_problems, new_hypotheses


def _annotate_item(item: dict, obs_index: dict[str, dict]) -> dict:
    """Annotate a single problem or hypothesis with evidence strength."""
    new = dict(item)
    new["supporting_observations"] = _convert_evidence_list(
        item.get("supporting_observations", []), "supporting", obs_index,
    )
    new["conflicting_observations"] = _convert_evidence_list(
        item.get("conflicting_observations", []), "conflicting", obs_index,
    )
    return new


def _convert_evidence_list(
    obs_ids: list,
    kind: str,
    obs_index: dict[str, dict],
) -> list[dict]:
    """Convert a list of observation IDs into structured evidence objects."""
    result: list[dict] = []
    for obs_id in obs_ids:
        if isinstance(obs_id, dict):
            # Already converted — preserve as-is
            result.append(obs_id)
            continue
        obs = obs_index.get(obs_id)
        strength = get_evidence_strength(obs) if obs else "weak"
        result.append({
            "observation_id": obs_id,
            "kind": kind,
            "strength": strength,
        })
    return result
