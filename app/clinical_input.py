"""Clinical input — structured answer ingestion.

Converts structured answers (from follow-up questions) into observation-
compatible dicts.  Does not merge into state, does not recompute any
downstream layer, does not mutate anything.

Supported answer types (v1): duration, severity, allergy, dosage.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

# Answer types we can convert to observations.
_SUPPORTED_TYPES = frozenset({"duration", "severity", "allergy", "dosage"})


def ingest_structured_answers(
    clinical_state: dict,
    answers: list[dict],
) -> dict:
    """Convert structured answers into new observation-compatible objects.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
            Used read-only to determine the next observation ID.
        answers: list of answer dicts, each with at least ``type`` and
            ``value``.  Optional keys: ``related`` (symptom/medication
            name the answer refers to).

    Returns:
        dict with:
        - ``new_observations``: list of observation-compatible dicts.
        - ``unparsed_answers``: list of answer dicts that could not be
          converted (unsupported type or missing required fields).
    """
    existing_obs = clinical_state.get("observations", [])
    next_id = len(existing_obs) + 1

    new_observations: list[dict] = []
    unparsed_answers: list[dict] = []

    for answer in answers:
        answer_type = answer.get("type", "")
        value = answer.get("value", "")
        related = answer.get("related")

        if not answer_type or not value:
            unparsed_answers.append(answer)
            continue

        if answer_type not in _SUPPORTED_TYPES:
            unparsed_answers.append(answer)
            continue

        obs = _convert_answer(answer_type, value, related, next_id)
        if obs is None:
            unparsed_answers.append(answer)
            continue

        new_observations.append(obs)
        next_id += 1

    return {
        "new_observations": new_observations,
        "unparsed_answers": unparsed_answers,
    }


def _convert_answer(
    answer_type: str,
    value: str,
    related: str | None,
    obs_id: int,
) -> dict | None:
    """Convert a single answer to an observation-compatible dict."""
    observation_id = f"obs_{obs_id:04d}"

    if answer_type == "duration":
        return _make_observation(
            observation_id=observation_id,
            finding_type="duration",
            category=None,
            value=value.strip(),
            source="structured_answer",
            attributes={"related_symptom": related} if related else {},
        )

    if answer_type == "severity":
        return _make_observation(
            observation_id=observation_id,
            finding_type="symptom",
            category="symptom",
            value=related or "",
            source="structured_answer",
            attributes={"severity": value.strip()},
        )

    if answer_type == "allergy":
        return _make_observation(
            observation_id=observation_id,
            finding_type="allergy",
            category="allergy",
            value=value.strip(),
            source="structured_answer",
            attributes={},
        )

    if answer_type == "dosage":
        return _make_observation(
            observation_id=observation_id,
            finding_type="medication",
            category="medication",
            value=related or "",
            source="structured_answer",
            attributes={"dosage": value.strip()},
        )

    return None  # pragma: no cover — all types handled above


def _make_observation(
    *,
    observation_id: str,
    finding_type: str,
    category: str | None,
    value: str,
    source: str,
    attributes: dict,
) -> dict:
    """Build an observation-compatible dict."""
    return {
        "observation_id": observation_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": None,
        "speaker_id": None,
        "t_start": None,
        "t_end": None,
        "source_text": None,
        "source": source,
        "category": category,
        "attributes": attributes,
        "confidence": None,
    }
