"""Structured clinical state — assembles all deterministic pipeline outputs.

Orchestrates existing extractor, review-flag, timeline, and diagnostic-hint
modules into a single dictionary.  No new extraction logic — this module
only calls and collects.

The returned structure is designed to be easily extensible with future
fields (problem_representation, objective_findings, labs, etc.) without
breaking existing consumers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.extractors import (
    extract_symptoms,
    extract_negations,
    extract_durations,
    extract_medications,
)
from app.symptom_timeline import extract_symptom_timeline
from app.review_flags import generate_review_flags
from app.diagnostic_hints import generate_diagnostic_hints
from app.history_extraction import extract_history_context
from app.qualifier_extraction import extract_qualifiers
from app.problem_representation import (
    build_problem_representation,
    build_symptom_representations,
)
from app.problem_summary import summarize_problem
from app.ontology_mapper import map_symptoms_to_concepts
from app.pattern_matcher import match_clinical_patterns
from app.live_summary import build_running_summary
from app.temporal_normalizer import normalize_timeline
from app.temporal_reasoner import derive_temporal_context
from app.red_flag_detector import detect_red_flags
from app.output_selector import apply_optional_outputs


def build_clinical_state(
    segments: list[dict],
    speaker_roles: Optional[dict[str, dict]] = None,
    confidence_entries: Optional[list[dict]] = None,
    config: object | None = None,
) -> dict:
    """Build a structured clinical state from normalized segments.

    Assembles outputs from all deterministic extraction modules into
    a single dictionary.

    Args:
        segments: list of normalized segment dicts (with ``normalized_text``).
        speaker_roles: optional ``{speaker_id: {role, confidence, evidence}}``.
        confidence_entries: optional list of per-segment ASR quality dicts.

    Returns:
        dict with keys: ``symptoms``, ``durations``, ``negations``,
        ``medications``, ``timeline``, ``review_flags``,
        ``diagnostic_hints``, ``speaker_roles``, ``history``.
    """
    full_text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).strip()

    symptoms = extract_symptoms(full_text)
    negations = extract_negations(full_text)
    durations = extract_durations(full_text)
    medications = extract_medications(full_text)

    timeline = extract_symptom_timeline(segments)

    review_flags = generate_review_flags(
        segments, confidence_entries=confidence_entries,
    )

    diagnostic_hints = generate_diagnostic_hints(symptoms, negations)

    history = extract_history_context(segments)

    qualifiers = extract_qualifiers(segments, extracted_findings=symptoms)

    state = {
        "symptoms": symptoms,
        "durations": durations,
        "negations": negations,
        "medications": medications,
        "timeline": timeline,
        "review_flags": review_flags,
        "diagnostic_hints": diagnostic_hints,
        "speaker_roles": speaker_roles,
        "history": history,
        "qualifiers": qualifiers,
    }

    pr = build_problem_representation(state)
    symptom_reps = build_symptom_representations(state)
    state["derived"] = {
        "problem_representation": pr,
        "problem_focus": pr.get("core_symptom"),
        "symptom_representations": symptom_reps,
    }
    state["derived"]["problem_summary"] = summarize_problem(state)
    state["derived"]["ontology_concepts"] = map_symptoms_to_concepts(state)
    state["derived"]["clinical_patterns"] = match_clinical_patterns(state)
    state["derived"]["running_summary"] = build_running_summary(state)
    state["derived"]["normalized_timeline"] = normalize_timeline(
        timeline, reference_date=datetime.now(),
    )
    state["derived"]["temporal_context"] = derive_temporal_context(state)
    state["derived"]["red_flags"] = detect_red_flags(state)

    apply_optional_outputs(state, config)

    return state
