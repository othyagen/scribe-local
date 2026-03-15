"""Encounter timeline — groups observations into clinical encounters.

One session = one encounter (v1).  The encounter references observation
IDs without copying them, keeping the data model observation-generic.
Actions and documents are empty lists, ready for future extractors.

Pure function — no I/O, deterministic.
"""

from __future__ import annotations

from datetime import datetime

_VALID_TYPES = {"consultation", "follow_up", "emergency", "procedure"}
_VALID_MODALITIES = {"in_person", "telehealth", "phone"}

_DEFAULT_TYPE = "consultation"
_DEFAULT_MODALITY = "in_person"


def build_encounters(
    segments: list[dict],
    observations: list[dict],
    encounter_type: str = _DEFAULT_TYPE,
    modality: str = _DEFAULT_MODALITY,
) -> list[dict]:
    """Build encounter records from segments and observations.

    Groups all segments into a single encounter (one session = one
    encounter).  Timestamp derived from the earliest segment ``t0``.

    Args:
        segments: list of normalized segment dicts (with ``t0``).
        observations: list of observation dicts (with ``observation_id``).
        encounter_type: encounter classification; defaults to
            ``"consultation"``.  Invalid values fall back to default.
        modality: encounter modality; defaults to ``"in_person"``.
            Invalid values fall back to default.

    Returns:
        list containing a single encounter dict, or empty list if no
        segments are provided.
    """
    if not segments:
        return []

    # Validate type/modality
    if encounter_type not in _VALID_TYPES:
        encounter_type = _DEFAULT_TYPE
    if modality not in _VALID_MODALITIES:
        modality = _DEFAULT_MODALITY

    # Timestamp from earliest segment
    t0_values = [seg.get("t0", 0.0) for seg in segments]
    earliest_t0 = min(t0_values)

    # Build ISO timestamp from session start (use epoch + offset)
    timestamp = datetime.utcfromtimestamp(earliest_t0).isoformat()

    # Collect observation IDs
    obs_ids = [
        obs["observation_id"]
        for obs in observations
        if "observation_id" in obs
    ]

    return [
        {
            "id": "enc_0001",
            "type": encounter_type,
            "modality": modality,
            "timestamp": timestamp,
            "observations": obs_ids,
            "actions": [],
            "documents": [],
        }
    ]
