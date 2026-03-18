"""Clinical flow — manual update flow orchestration.

Composes existing modules (clinical_input, clinical_orchestration,
clinical_update) into a simple flow for handling structured answers,
gating updates, and building application-level state.

Does not introduce new clinical logic.  Pure orchestration only.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

from typing import Optional

from app.clinical_input import ingest_structured_answers
from app.clinical_orchestration import orchestrate_outputs, should_apply_update
from app.clinical_update import apply_update


def handle_answers(
    clinical_state: dict,
    answers: list[dict],
    config: dict | None = None,
) -> dict:
    """Process structured answers and determine update behaviour.

    Calls :func:`ingest_structured_answers` to convert answers into
    observations, then consults :func:`should_apply_update` to decide
    whether a state rebuild should happen now or be deferred.

    Args:
        clinical_state: current clinical state (read-only).
        answers: list of structured answer dicts.
        config: orchestration config (same format as
            :func:`orchestrate_outputs`).

    Returns:
        dict with:
        - ``new_observations``: observation-compatible dicts from answers.
        - ``unparsed_answers``: answers that could not be converted.
        - ``should_update``: bool — whether an automatic update is gated.
        - ``pending_observations``: same as *new_observations* when
          ``should_update`` is False; empty list when True (consumed).
    """
    ingestion = ingest_structured_answers(clinical_state, answers)

    new_obs = ingestion["new_observations"]
    unparsed = ingestion["unparsed_answers"]

    update_now = should_apply_update(config, "new_answers") if new_obs else False

    return {
        "new_observations": new_obs,
        "unparsed_answers": unparsed,
        "should_update": update_now,
        "pending_observations": [] if update_now else list(new_obs),
    }


def apply_pending_update(
    clinical_state: dict,
    pending_observations: list[dict],
    segments: list[dict],
    speaker_roles: Optional[dict[str, dict]] = None,
    confidence_entries: Optional[list[dict]] = None,
    config: object | None = None,
) -> dict:
    """Apply pending observations and rebuild the clinical state.

    Delegates to :func:`apply_update` for a full pipeline rebuild
    with the combined observations.

    Args:
        clinical_state: current clinical state (read-only).
        pending_observations: observations waiting to be integrated.
        segments: original normalised segment dicts.
        speaker_roles: optional speaker role mapping.
        confidence_entries: optional ASR quality entries.
        config: optional pipeline config object.

    Returns:
        A new, complete clinical state dict.  If *pending_observations*
        is empty, returns a fresh rebuild of the base state (no-op).
    """
    return apply_update(
        clinical_state,
        pending_observations,
        segments,
        speaker_roles=speaker_roles,
        confidence_entries=confidence_entries,
        config=config,
    )


def build_app_state(
    clinical_state: dict,
    config: dict | None = None,
    pending_observations: list[dict] | None = None,
) -> dict:
    """Build application-level state for downstream consumers.

    Combines :func:`orchestrate_outputs` with pending-update metadata.

    Args:
        clinical_state: current clinical state (read-only).
        config: orchestration config.
        pending_observations: observations awaiting manual apply.

    Returns:
        dict with:
        - ``orchestrated``: output from :func:`orchestrate_outputs`.
        - ``has_pending``: bool — whether observations are waiting.
        - ``pending_count``: number of pending observations.
    """
    orchestrated = orchestrate_outputs(clinical_state, config)

    pending = pending_observations or []

    return {
        "orchestrated": orchestrated,
        "has_pending": len(pending) > 0,
        "pending_count": len(pending),
    }
