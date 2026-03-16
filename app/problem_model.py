"""Problem model — organizes observations into clinically meaningful problems.

Observation-first design: problems are derived from observation records
and reference observation IDs only.  Structured symptoms, diagnostic hints,
and red flags are used as helpers for onset/priority enrichment, but the
source of truth is always the observation layer.

Compatible with future observation types (labs, vitals, imaging, etc.).

Pure function — no I/O, no ML, no LLM.
"""

from __future__ import annotations


def build_problem_list(clinical_state: dict) -> list[dict]:
    """Build a problem list from clinical state observations.

    Problem generation rules (v1):

    1. **symptom_problem** — one per distinct symptom value found in
       observations (finding_type == "symptom").  Enriched with onset
       from structured symptoms if available.

    2. **risk_problem** — one per red flag whose evidence symptoms all
       appear in the observation layer.  Priority "urgent".

    3. **working_problem** — one per diagnostic hint, only if:
       - at least 2 supporting observations exist, AND
       - no existing symptom_problem has the same title (case-insensitive).

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of problem dicts, deterministic order.
    """
    observations: list[dict] = clinical_state.get("observations", [])
    encounters: list[dict] = clinical_state.get("encounters", [])
    derived = clinical_state.get("derived", {})
    structured_symptoms: list[dict] = derived.get("structured_symptoms", [])
    red_flags: list[dict] = derived.get("red_flags", [])
    diagnostic_hints: list[dict] = clinical_state.get("diagnostic_hints", [])

    encounter_ids = [enc.get("id", "") for enc in encounters if enc.get("id")]

    # Index: symptom value (lowercase) → list of observation IDs
    obs_index = _build_obs_index(observations)

    # Index: symptom (lowercase) → onset from structured symptoms
    onset_index = _build_onset_index(structured_symptoms)

    counter = 0
    problems: list[dict] = []
    seen_titles: set[str] = set()

    # 1. Symptom problems — one per distinct observed symptom
    for symptom_key in _unique_symptom_values(observations):
        obs_ids = obs_index.get(symptom_key, [])
        if not obs_ids:
            continue

        counter += 1
        # Use original-case value from first observation
        title = _original_value(observations, symptom_key)
        onset = onset_index.get(symptom_key)

        problems.append({
            "id": f"prob_{counter:04d}",
            "title": title,
            "kind": "symptom_problem",
            "status": "active",
            "onset": onset,
            "observations": list(obs_ids),
            "encounters": list(encounter_ids),
            "actions": [],
            "documents": [],
            "priority": "normal",
        })
        seen_titles.add(symptom_key)

    # 2. Risk problems from red flags
    for flag in red_flags:
        label = flag.get("label", "")
        if not label:
            continue

        evidence: list[str] = flag.get("evidence", [])
        obs_ids = _collect_obs_ids_for_evidence(evidence, obs_index)
        # Only create if at least one observation supports the flag
        if not obs_ids:
            continue

        counter += 1
        problems.append({
            "id": f"prob_{counter:04d}",
            "title": label,
            "kind": "risk_problem",
            "status": "active",
            "onset": None,
            "observations": obs_ids,
            "encounters": list(encounter_ids),
            "actions": [],
            "documents": [],
            "priority": "urgent",
        })

    # 3. Working problems from diagnostic hints (conservative)
    for hint in diagnostic_hints:
        condition = hint.get("condition", "")
        if not condition:
            continue
        if condition.lower() in seen_titles:
            continue

        evidence = hint.get("evidence", [])
        obs_ids = _collect_obs_ids_for_evidence(evidence, obs_index)
        if len(obs_ids) < 2:
            continue

        counter += 1
        problems.append({
            "id": f"prob_{counter:04d}",
            "title": condition,
            "kind": "working_problem",
            "status": "active",
            "onset": None,
            "observations": obs_ids,
            "encounters": list(encounter_ids),
            "actions": [],
            "documents": [],
            "priority": "normal",
        })
        seen_titles.add(condition.lower())

    return problems


# ── helpers ──────────────────────────────────────────────────────────


def _build_obs_index(observations: list[dict]) -> dict[str, list[str]]:
    """Map symptom value (lowercase) → list of observation IDs."""
    index: dict[str, list[str]] = {}
    for obs in observations:
        if obs.get("finding_type") == "symptom":
            key = obs.get("value", "").lower()
            obs_id = obs.get("observation_id", "")
            if key and obs_id:
                index.setdefault(key, []).append(obs_id)
    return index


def _build_onset_index(structured_symptoms: list[dict]) -> dict[str, str | None]:
    """Map symptom (lowercase) → onset string from structured symptoms."""
    index: dict[str, str | None] = {}
    for ss in structured_symptoms:
        symptom = ss.get("symptom", "")
        if not symptom:
            continue
        key = symptom.lower()
        if key not in index:
            onset = (ss.get("temporal") or {}).get("onset")
            index[key] = onset
    return index


def _unique_symptom_values(observations: list[dict]) -> list[str]:
    """Return unique symptom values (lowercase) preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for obs in observations:
        if obs.get("finding_type") == "symptom":
            key = obs.get("value", "").lower()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _original_value(observations: list[dict], symptom_key: str) -> str:
    """Return original-case value for a symptom from first matching obs."""
    for obs in observations:
        if (obs.get("finding_type") == "symptom"
                and obs.get("value", "").lower() == symptom_key):
            return obs.get("value", symptom_key)
    return symptom_key


def _collect_obs_ids_for_evidence(
    evidence: list[str],
    obs_index: dict[str, list[str]],
) -> list[str]:
    """Collect unique observation IDs for evidence symptom names."""
    seen: set[str] = set()
    result: list[str] = []
    for item in evidence:
        key = item.lower()
        for obs_id in obs_index.get(key, []):
            if obs_id not in seen:
                seen.add(obs_id)
                result.append(obs_id)
    return result
