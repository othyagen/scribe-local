"""Symptom groups — cluster related symptom observations by encounter.

Groups symptom observations by encounter, body system, and temporal
bucket to form lightweight clinical syndrome clusters.  Organisational
only — does not change observations, problems, or hypotheses.

Pure function — no I/O beyond config loading, no ML, no input mutation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
_SYSTEMS_FILE = _RESOURCES_DIR / "symptom_systems.json"

# ── configuration loading ───────────────────────────────────────────


def _load_symptom_systems(path: Path | None = None) -> dict[str, list[str]]:
    """Load symptom-to-system mapping from JSON file."""
    p = path or _SYSTEMS_FILE
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {
                k: [s.lower() for s in v]
                for k, v in data.items()
                if isinstance(v, list)
            }
    except (OSError, json.JSONDecodeError):
        pass
    return {}


# Module-level cache
_SYMPTOM_SYSTEMS: dict[str, list[str]] | None = None


def _get_symptom_systems() -> dict[str, list[str]]:
    global _SYMPTOM_SYSTEMS
    if _SYMPTOM_SYSTEMS is None:
        _SYMPTOM_SYSTEMS = _load_symptom_systems()
    return _SYMPTOM_SYSTEMS


# ── temporal bucket ─────────────────────────────────────────────────

_ACUTE_KEYWORDS = {"day", "days", "yesterday", "today", "hour", "hours"}
_SUBACUTE_KEYWORDS = {"week", "weeks"}
_CHRONIC_KEYWORDS = {"month", "months", "year", "years"}

# Pattern: number followed by unit
_DURATION_NUM_PATTERN = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)", re.IGNORECASE)


def _classify_temporal_bucket(duration: str | None) -> str:
    """Classify a duration string into a temporal bucket."""
    if not duration:
        return "unknown"

    d = duration.lower().strip()

    # Try numeric parsing first
    m = _DURATION_NUM_PATTERN.search(d)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "day":
            return "acute" if num < 7 else ("subacute" if num <= 30 else "chronic")
        if unit == "week":
            days = num * 7
            return "acute" if days < 7 else ("subacute" if days <= 30 else "chronic")
        if unit == "month":
            return "chronic" if num > 1 else "subacute"
        if unit == "year":
            return "chronic"

    # Keyword fallback
    words = set(d.split())
    if words & _ACUTE_KEYWORDS:
        return "acute"
    if words & _SUBACUTE_KEYWORDS:
        return "subacute"
    if words & _CHRONIC_KEYWORDS:
        return "chronic"

    return "unknown"


# ── system assignment ───────────────────────────────────────────────


def _assign_systems(value: str, systems_map: dict[str, list[str]]) -> list[str]:
    """Assign body systems to a symptom value."""
    v = value.lower()
    matched: list[str] = []
    for system, keywords in systems_map.items():
        if system == "general":
            continue  # check general last as fallback
        for kw in keywords:
            if kw in v:
                matched.append(system)
                break

    if not matched:
        # Check general explicitly
        general_kws = systems_map.get("general", [])
        for kw in general_kws:
            if kw in v:
                matched.append("general")
                break

    if not matched:
        matched.append("general")

    return matched


# ── grouping ────────────────────────────────────────────────────────


def build_symptom_groups(clinical_state: dict) -> list[dict]:
    """Build symptom groups from clinical state.

    Groups symptom observations by encounter, body system, and temporal
    bucket.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of symptom group dicts, deterministic order.
    """
    observations: list[dict] = clinical_state.get("observations", [])
    encounters: list[dict] = clinical_state.get("encounters", [])

    systems_map = _get_symptom_systems()

    # Filter symptom observations only
    symptom_obs = [
        obs for obs in observations
        if obs.get("category") == "symptom"
    ]

    if not symptom_obs:
        return []

    # Build encounter → observation ID set
    enc_obs_sets: dict[str, set[str]] = {}
    for enc in encounters:
        enc_id = enc.get("id", "")
        enc_obs_sets[enc_id] = set(enc.get("observations", []))

    # Assign each symptom obs to encounters
    # If no encounters, use a synthetic one
    if not encounters:
        enc_obs_sets["enc_0000"] = {
            obs.get("observation_id", "") for obs in symptom_obs
        }

    # Group key: (encounter_id, primary_system, temporal_bucket)
    groups: dict[tuple[str, str, str], list[str]] = {}
    group_order: list[tuple[str, str, str]] = []

    for obs in symptom_obs:
        obs_id = obs.get("observation_id", "")
        value = obs.get("value", "")
        attrs = obs.get("attributes") or {}
        duration = attrs.get("duration")
        bucket = _classify_temporal_bucket(duration)

        systems = _assign_systems(value, systems_map)
        primary_system = systems[0]

        # Find which encounter(s) this observation belongs to
        matched_encs = [
            enc_id for enc_id, obs_set in enc_obs_sets.items()
            if obs_id in obs_set
        ]
        if not matched_encs:
            matched_encs = list(enc_obs_sets.keys())[:1] or ["enc_0000"]

        for enc_id in matched_encs:
            key = (enc_id, primary_system, bucket)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            if obs_id not in groups[key]:
                groups[key].append(obs_id)

    # Build result
    result: list[dict] = []
    for i, key in enumerate(group_order):
        enc_id, system, bucket = key
        obs_ids = groups[key]
        result.append({
            "id": f"grp_{i + 1:04d}",
            "title": f"{bucket} {system} symptom group",
            "kind": "symptom_group",
            "systems": [system],
            "temporal_bucket": bucket,
            "observations": obs_ids,
            "encounters": [enc_id],
        })

    return result
