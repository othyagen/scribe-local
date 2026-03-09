"""Lightweight deterministic FHIR export layer.

Converts the existing structured clinical_state into a small set of
FHIR-compatible resource dicts wrapped in a Bundle.  Export only —
never modifies pipeline logic or extraction results.

Supports four initial resource types:
  - Encounter (session metadata)
  - Observation (symptoms / findings)
  - Condition (problem representation + classification)
  - Composition (narrative summary)

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


# ── deterministic ID generation ──────────────────────────────────


def _stable_id(resource_type: str, *parts: str) -> str:
    """Generate a deterministic resource ID from content parts.

    Uses a truncated SHA-256 hex digest so the same input always
    produces the same ID.
    """
    content = f"{resource_type}:{':'.join(parts)}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ── resource builders ────────────────────────────────────────────


def _build_encounter(clinical_state: dict) -> dict:
    """Build a FHIR Encounter resource from session metadata."""
    derived = clinical_state.get("derived", {})
    core_problem = derived.get("problem_summary", "")

    encounter: dict = {
        "resourceType": "Encounter",
        "id": _stable_id("Encounter", "scribe-session"),
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "AMB",
            "display": "ambulatory",
        },
    }

    if core_problem:
        encounter["reasonCode"] = [{
            "text": core_problem,
        }]

    return encounter


def _build_observations(clinical_state: dict) -> list[dict]:
    """Build FHIR Observation resources from symptoms / findings."""
    derived = clinical_state.get("derived", {})
    ontology_concepts: list[dict] = derived.get("ontology_concepts", [])
    symptom_reps: list[dict] = derived.get("symptom_representations", [])

    # Index symptom representations by lowercase name for enrichment
    rep_index: dict[str, dict] = {}
    for rep in symptom_reps:
        key = rep.get("symptom", "").lower()
        if key:
            rep_index[key] = rep

    observations: list[dict] = []

    # Prefer ontology-mapped concepts (they have codes)
    seen_symptoms: set[str] = set()

    for concept in ontology_concepts:
        symptom_text = concept.get("text", "")
        code = concept.get("code", "")
        label = concept.get("label", symptom_text)
        system = concept.get("system", "")

        obs: dict = {
            "resourceType": "Observation",
            "id": _stable_id("Observation", symptom_text.lower()),
            "status": "preliminary",
            "code": {
                "text": label,
            },
        }

        if code and system:
            obs["code"]["coding"] = [{
                "system": system,
                "code": code,
                "display": label,
            }]

        # Enrich with qualifier data from symptom representations
        rep = rep_index.get(symptom_text.lower())
        if rep:
            components = _build_observation_components(rep)
            if components:
                obs["component"] = components

        observations.append(obs)
        seen_symptoms.add(symptom_text.lower())

    # Add any symptoms not covered by ontology concepts (text-only)
    symptoms: list[str] = clinical_state.get("symptoms", [])
    for symptom in symptoms:
        if symptom.lower() in seen_symptoms:
            continue
        seen_symptoms.add(symptom.lower())

        obs = {
            "resourceType": "Observation",
            "id": _stable_id("Observation", symptom.lower()),
            "status": "preliminary",
            "code": {"text": symptom},
        }

        rep = rep_index.get(symptom.lower())
        if rep:
            components = _build_observation_components(rep)
            if components:
                obs["component"] = components

        observations.append(obs)

    return observations


def _build_observation_components(rep: dict) -> list[dict]:
    """Build FHIR Observation.component entries from a symptom representation."""
    components: list[dict] = []

    _FIELDS = [
        ("severity", "Severity"),
        ("onset", "Onset"),
        ("duration", "Duration"),
        ("pattern", "Pattern"),
        ("progression", "Progression"),
        ("laterality", "Laterality"),
        ("radiation", "Radiation"),
    ]

    for field, display in _FIELDS:
        value = rep.get(field)
        if value:
            components.append({
                "code": {"text": display},
                "valueString": value,
            })

    # Aggravating / relieving factors
    for field, display in [
        ("aggravating_factors", "Aggravating factor"),
        ("relieving_factors", "Relieving factor"),
    ]:
        factors = rep.get(field, [])
        for factor in factors:
            if factor:
                components.append({
                    "code": {"text": display},
                    "valueString": factor,
                })

    return components


def _build_conditions(clinical_state: dict) -> list[dict]:
    """Build FHIR Condition resources from problem representation + classification."""
    derived = clinical_state.get("derived", {})
    pr = derived.get("problem_representation", {})
    classification = derived.get("classification", {})

    conditions: list[dict] = []

    # Primary condition from problem representation
    core_symptom = pr.get("core_symptom")
    if core_symptom:
        condition: dict = {
            "resourceType": "Condition",
            "id": _stable_id("Condition", core_symptom.lower()),
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }],
            },
            "verificationStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "provisional",
                }],
            },
            "code": {"text": core_symptom},
        }

        # Add severity if available
        severity = pr.get("severity")
        if severity:
            condition["severity"] = {"text": severity}

        # Add onset if available
        onset = pr.get("onset")
        if onset:
            condition["onsetString"] = onset

        # Add classification suggestions as note (conservative — not coding)
        suggestions = classification.get("suggestions", [])
        if suggestions:
            system_name = classification.get("system", "")
            notes = []
            for s in suggestions:
                code = s.get("code", "")
                label = s.get("label", "")
                kind = s.get("kind", "")
                notes.append(
                    f"{system_name} suggestion: {code} {label} ({kind})"
                )
            condition["note"] = [{"text": "; ".join(notes)}]

        conditions.append(condition)

    return conditions


def _build_composition(clinical_state: dict) -> dict:
    """Build a FHIR Composition resource as a narrative summary."""
    derived = clinical_state.get("derived", {})

    problem_summary = derived.get("problem_summary", "")
    running_summary = derived.get("running_summary", {})
    red_flags: list[dict] = derived.get("red_flags", [])
    classification = derived.get("classification", {})

    sections: list[dict] = []

    # Problem summary section
    if problem_summary:
        sections.append({
            "title": "Chief Complaint",
            "text": {
                "status": "generated",
                "div": problem_summary,
            },
        })

    # Running summary section
    core_problem = running_summary.get("core_problem", "")
    additional = running_summary.get("additional_symptoms", [])
    patterns = running_summary.get("patterns_detected", [])

    summary_parts: list[str] = []
    if core_problem:
        summary_parts.append(core_problem)
    if additional:
        summary_parts.append(
            f"Additional symptoms: {', '.join(additional)}"
        )
    if patterns:
        summary_parts.append(
            f"Patterns: {', '.join(patterns)}"
        )

    if summary_parts:
        sections.append({
            "title": "Clinical Summary",
            "text": {
                "status": "generated",
                "div": " | ".join(summary_parts),
            },
        })

    # Red flags section
    if red_flags:
        flag_lines = [
            f"{rf['label']} (severity: {rf['severity']})"
            for rf in red_flags
            if "label" in rf and "severity" in rf
        ]
        if flag_lines:
            sections.append({
                "title": "Red Flags",
                "text": {
                    "status": "generated",
                    "div": "; ".join(flag_lines),
                },
            })

    # Classification section
    suggestions = classification.get("suggestions", [])
    if suggestions:
        system_name = classification.get("system", "")
        cls_lines = [
            f"{s.get('code', '')} {s.get('label', '')}"
            for s in suggestions
        ]
        sections.append({
            "title": f"Classification Suggestions ({system_name})",
            "text": {
                "status": "generated",
                "div": "; ".join(cls_lines),
            },
        })

    composition: dict = {
        "resourceType": "Composition",
        "id": _stable_id("Composition", "scribe-narrative"),
        "status": "preliminary",
        "type": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "11488-4",
                "display": "Consult note",
            }],
        },
        "title": "SCRIBE Clinical Summary",
    }

    if sections:
        composition["section"] = sections

    return composition


# ── public API ───────────────────────────────────────────────────


def build_fhir_bundle(clinical_state: dict) -> dict:
    """Convert structured clinical state into a lightweight FHIR Bundle.

    Produces a ``collection`` Bundle containing Encounter, Observation,
    Condition, and Composition resources derived deterministically from
    existing clinical_state data.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        FHIR Bundle dict with ``resourceType``, ``type``, and ``entry``
        keys.  Each entry is a dict with a ``resource`` key.
    """
    entries: list[dict] = []

    # Encounter
    encounter = _build_encounter(clinical_state)
    entries.append({"resource": encounter})

    # Observations
    for obs in _build_observations(clinical_state):
        entries.append({"resource": obs})

    # Conditions
    for cond in _build_conditions(clinical_state):
        entries.append({"resource": cond})

    # Composition
    composition = _build_composition(clinical_state)
    entries.append({"resource": composition})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }
