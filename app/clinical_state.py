"""Structured clinical state — pipeline orchestrator (S4-S8).

Calls extraction, observation, structuring, reasoning, and graph modules
in a fixed forward-only order.  No new extraction logic — this module
only calls and collects.

Invariants (see docs/ARCHITECTURE.md):
  - Later stages never mutate earlier-stage outputs.
  - The returned dict is the source of truth; exports are projections.
  - Same input produces same output (deterministic).
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
    build_problem_narrative,
)
from app.observation_layer import build_observation_layer
from app.observation_taxonomy import enrich_observations
from app.observation_normalization import normalize_observations
from app.ice_extraction import extract_ice
from app.intensity_extraction import extract_intensities
from app.site_extraction import extract_sites
from app.structured_symptom_model import build_structured_symptoms
from app.problem_summary import summarize_problem
from app.ontology_mapper import map_symptoms_to_concepts
from app.pattern_matcher import match_clinical_patterns
from app.live_summary import build_running_summary
from app.temporal_normalizer import normalize_timeline
from app.temporal_reasoner import derive_temporal_context
from app.red_flag_detector import detect_red_flags
from app.output_selector import apply_optional_outputs
from app.encounter import build_encounters
from app.problem_model import build_problem_list
from app.clinical_graph import build_clinical_graph
from app.problem_evidence import annotate_problem_evidence
from app.diagnostic_hypotheses import build_diagnostic_hypotheses
from app.evidence_strength import annotate_evidence_strength
from app.hypothesis_ranking import rank_hypotheses


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

    # Layer 1: Observations (all occurrences, not just first)
    raw_observations = build_observation_layer(
        segments, symptoms, negations, durations, medications,
    )
    state["observations"] = normalize_observations(
        enrich_observations(raw_observations),
    )

    # Encounter timeline
    state["encounters"] = build_encounters(segments, state["observations"])

    # Layer 2 additions
    state["ice"] = extract_ice(segments)
    state["intensities"] = extract_intensities(segments)
    state["sites"] = extract_sites(segments)

    # Existing derived computations (unchanged order)
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

    # Layer 3: Structured symptoms (after red_flags populated)
    state["derived"]["structured_symptoms"] = build_structured_symptoms(state)

    # Layer 4: Problem narrative
    state["derived"]["problem_narrative"] = build_problem_narrative(state)

    # Problem list — observation-first, after all derived computations
    state["problems"] = build_problem_list(state)

    # Problem evidence — annotate with supporting/conflicting references
    state["problems"] = annotate_problem_evidence(
        state["problems"], state["observations"],
    )

    # Diagnostic hypotheses — candidate conditions from hints
    state["hypotheses"] = build_diagnostic_hypotheses(state)

    # Evidence strength — annotate evidence with strength levels
    state["problems"], state["hypotheses"] = annotate_evidence_strength(
        state["problems"], state["observations"], state["hypotheses"],
    )

    # Hypothesis ranking — score and rank by evidence strength
    state["hypotheses"] = rank_hypotheses(state["hypotheses"])

    # Clinical graph — additive evidence-aware representation
    state["clinical_graph"] = build_clinical_graph(state).to_dict()

    apply_optional_outputs(state, config)

    return state
