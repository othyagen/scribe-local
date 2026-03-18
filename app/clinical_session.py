"""Clinical session — app-facing facade for the clinical engine.

Provides simple entry points that compose existing modules into a
stable integration surface.  Does not introduce new clinical logic,
mutation, or real-time behaviour.

All functions are pure — they accept state and return new state.
The caller is responsible for holding and threading state between calls.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

from typing import Optional

from app.clinical_state import build_clinical_state
from app.clinical_flow import handle_answers, apply_pending_update, build_app_state
from app.clinical_orchestration import orchestrate_outputs


def initialize_session(
    segments: list[dict],
    speaker_roles: Optional[dict[str, dict]] = None,
    confidence_entries: Optional[list[dict]] = None,
    config: dict | None = None,
) -> dict:
    """Initialize a clinical session from transcript segments.

    Builds the full clinical state and wraps it in a session envelope
    with orchestration config and empty pending queue.

    Args:
        segments: normalised segment dicts.
        speaker_roles: optional speaker role mapping.
        confidence_entries: optional ASR quality entries.
        config: orchestration config (mode, visibility, strategy).

    Returns:
        Session dict with ``clinical_state``, ``segments``,
        ``config``, ``pending_observations``, ``speaker_roles``,
        ``confidence_entries``.
    """
    clinical_state = build_clinical_state(
        segments,
        speaker_roles=speaker_roles,
        confidence_entries=confidence_entries,
    )

    return {
        "clinical_state": clinical_state,
        "segments": list(segments),
        "config": dict(config) if config else {},
        "pending_observations": [],
        "speaker_roles": speaker_roles,
        "confidence_entries": confidence_entries,
    }


def get_app_view(session: dict) -> dict:
    """Get the current application view for the session.

    Combines orchestrated outputs with pending-update status.

    Args:
        session: session dict from :func:`initialize_session` or
            a subsequent update.

    Returns:
        App-level view dict from :func:`build_app_state`.
    """
    return build_app_state(
        session["clinical_state"],
        config=session.get("config"),
        pending_observations=session.get("pending_observations"),
    )


def submit_answers(session: dict, answers: list[dict]) -> dict:
    """Submit structured answers to the session.

    Ingests answers, applies update gating, and returns a new session
    with updated pending queue.  Does not mutate the input session.

    Args:
        session: current session dict.
        answers: list of structured answer dicts.

    Returns:
        New session dict with updated ``pending_observations`` and
        an ``answer_result`` key containing ingestion details.
    """
    result = handle_answers(
        session["clinical_state"],
        answers,
        session.get("config"),
    )

    if result["should_update"]:
        # Automatic mode: apply immediately.
        updated_state = apply_pending_update(
            session["clinical_state"],
            result["new_observations"],
            session["segments"],
            speaker_roles=session.get("speaker_roles"),
            confidence_entries=session.get("confidence_entries"),
        )
        return {
            "clinical_state": updated_state,
            "segments": session["segments"],
            "config": session.get("config", {}),
            "pending_observations": [],
            "speaker_roles": session.get("speaker_roles"),
            "confidence_entries": session.get("confidence_entries"),
            "answer_result": result,
        }

    # Manual mode: accumulate pending.
    new_pending = list(session.get("pending_observations", []))
    new_pending.extend(result["pending_observations"])

    return {
        "clinical_state": session["clinical_state"],
        "segments": session["segments"],
        "config": session.get("config", {}),
        "pending_observations": new_pending,
        "speaker_roles": session.get("speaker_roles"),
        "confidence_entries": session.get("confidence_entries"),
        "answer_result": result,
    }


def apply_manual_update(session: dict) -> dict:
    """Apply all pending observations and rebuild the clinical state.

    Drains the pending queue and produces a new session with an
    updated clinical state.  No-op if the queue is empty.

    Args:
        session: current session dict.

    Returns:
        New session dict with updated ``clinical_state`` and
        empty ``pending_observations``.
    """
    pending = session.get("pending_observations", [])

    if not pending:
        return {
            "clinical_state": session["clinical_state"],
            "segments": session["segments"],
            "config": session.get("config", {}),
            "pending_observations": [],
            "speaker_roles": session.get("speaker_roles"),
            "confidence_entries": session.get("confidence_entries"),
        }

    updated_state = apply_pending_update(
        session["clinical_state"],
        pending,
        session["segments"],
        speaker_roles=session.get("speaker_roles"),
        confidence_entries=session.get("confidence_entries"),
    )

    return {
        "clinical_state": updated_state,
        "segments": session["segments"],
        "config": session.get("config", {}),
        "pending_observations": [],
        "speaker_roles": session.get("speaker_roles"),
        "confidence_entries": session.get("confidence_entries"),
    }
