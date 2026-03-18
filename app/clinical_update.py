"""Clinical update — full state rebuild with additional observations.

Accepts an existing clinical state, new observations, and the original
segments, then produces a complete updated state by re-running the
pipeline with combined observations.  Does not mutate any inputs.

Strategy: re-run ``build_clinical_state`` for extraction-level outputs,
then replace the observation list with (base + new) and re-run all
observation-dependent stages in the same order as the original pipeline.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.clinical_state import build_clinical_state
from app.encounter import build_encounters
from app.symptom_groups import build_symptom_groups
from app.problem_representation import (
    build_problem_representation,
    build_symptom_representations,
    build_problem_narrative,
)
from app.structured_symptom_model import build_structured_symptoms
from app.problem_summary import summarize_problem
from app.ontology_mapper import map_symptoms_to_concepts
from app.pattern_matcher import match_clinical_patterns
from app.live_summary import build_running_summary
from app.temporal_normalizer import normalize_timeline
from app.temporal_reasoner import derive_temporal_context
from app.red_flag_detector import detect_red_flags
from app.output_selector import apply_optional_outputs
from app.problem_model import build_problem_list
from app.problem_evidence import annotate_problem_evidence
from app.diagnostic_hypotheses import build_diagnostic_hypotheses
from app.evidence_strength import annotate_evidence_strength
from app.hypothesis_ranking import rank_hypotheses
from app.hypothesis_explanations import build_hypothesis_explanations
from app.clinical_summary import build_clinical_summary
from app.clinical_summary_views import build_summary_views
from app.clinical_insights import derive_clinical_insights
from app.clinical_interaction import derive_next_questions
from app.clinical_graph import build_clinical_graph


def apply_update(
    clinical_state: dict,
    new_observations: list[dict],
    segments: list[dict],
    speaker_roles: Optional[dict[str, dict]] = None,
    confidence_entries: Optional[list[dict]] = None,
    config: object | None = None,
) -> dict:
    """Rebuild clinical state with additional observations.

    Runs the full pipeline on the given segments to produce a fresh
    extraction base, then combines the base observations with
    *new_observations* and re-runs all downstream stages.

    Args:
        clinical_state: existing state dict (read-only, used for
            context only — not mutated).
        new_observations: list of observation-compatible dicts to add
            (e.g. from :func:`ingest_structured_answers`).
        segments: original normalised segment dicts.
        speaker_roles: optional speaker role mapping.
        confidence_entries: optional ASR quality entries.
        config: optional pipeline config object.

    Returns:
        A new, complete clinical state dict with the combined
        observations and all downstream layers recomputed.
    """
    # Phase 1: fresh extraction base from segments.
    base = build_clinical_state(
        segments,
        speaker_roles=speaker_roles,
        confidence_entries=confidence_entries,
        config=config,
    )

    if not new_observations:
        return base

    # Phase 2: combine observations (base + new, no mutation).
    combined_observations = list(base["observations"]) + list(new_observations)

    # Phase 3: rebuild state from extraction outputs + combined observations.
    state = {
        "symptoms": base["symptoms"],
        "durations": base["durations"],
        "negations": base["negations"],
        "medications": base["medications"],
        "timeline": base["timeline"],
        "review_flags": base["review_flags"],
        "diagnostic_hints": base["diagnostic_hints"],
        "speaker_roles": base["speaker_roles"],
        "history": base["history"],
        "qualifiers": base["qualifiers"],
        "observations": combined_observations,
    }

    # Encounters — needs segments + observations.
    state["encounters"] = build_encounters(segments, combined_observations)

    # Symptom groups.
    state["symptom_groups"] = build_symptom_groups(state)

    # Segment-derived layers (unchanged from base — not observation-dependent).
    state["ice"] = base["ice"]
    state["intensities"] = base["intensities"]
    state["sites"] = base["sites"]

    # Derived computations.
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
        state["timeline"], reference_date=datetime.now(),
    )
    state["derived"]["temporal_context"] = derive_temporal_context(state)
    state["derived"]["red_flags"] = detect_red_flags(state)

    # Structured symptoms (after red_flags).
    state["derived"]["structured_symptoms"] = build_structured_symptoms(state)

    # Problem narrative.
    state["derived"]["problem_narrative"] = build_problem_narrative(state)

    # Problem list.
    state["problems"] = build_problem_list(state)

    # Problem evidence.
    state["problems"] = annotate_problem_evidence(
        state["problems"], state["observations"],
    )

    # Diagnostic hypotheses.
    state["hypotheses"] = build_diagnostic_hypotheses(state)

    # Evidence strength.
    state["problems"], state["hypotheses"] = annotate_evidence_strength(
        state["problems"], state["observations"], state["hypotheses"],
    )

    # Hypothesis ranking.
    state["hypotheses"] = rank_hypotheses(state["hypotheses"])

    # Hypothesis explanations.
    state["hypotheses"] = build_hypothesis_explanations(
        state["hypotheses"], state["observations"],
    )

    # Clinical summary.
    state["clinical_summary"] = build_clinical_summary(state)

    # Summary views.
    state["summary_views"] = build_summary_views(state["clinical_summary"])

    # Clinical insights.
    state["clinical_insights"] = derive_clinical_insights(state)

    # Next questions.
    state["next_questions"] = derive_next_questions(state)

    # Clinical graph.
    state["clinical_graph"] = build_clinical_graph(state).to_dict()

    apply_optional_outputs(state, config)

    return state
