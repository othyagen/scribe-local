"""Clinical metrics — structured evaluation metrics from clinical state.

Provides format-neutral, read-only metrics for evaluating and inspecting
clinical state outputs.  Does not influence any clinical reasoning.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


def derive_clinical_metrics(clinical_state: dict) -> dict:
    """Extract structured metrics from a clinical state.

    Tolerates missing keys gracefully — returns zero/empty/None for
    any field not present in the state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`
            or any partial state dict.

    Returns:
        dict with stable schema containing observation, problem,
        hypothesis, risk, interaction, data quality, and update metrics.
    """
    return {
        "observation_metrics": _observation_metrics(clinical_state),
        "problem_metrics": _problem_metrics(clinical_state),
        "hypothesis_metrics": _hypothesis_metrics(clinical_state),
        "risk_metrics": _risk_metrics(clinical_state),
        "interaction_metrics": _interaction_metrics(clinical_state),
        "data_quality_metrics": _data_quality_metrics(clinical_state),
        "update_metrics": _update_metrics(clinical_state),
    }


# ── observation metrics ──────────────────────────────────────────────


def _observation_metrics(state: dict) -> dict:
    observations = state.get("observations", [])

    counts: dict[str, int] = {}
    for obs in observations:
        ft = obs.get("finding_type", "other")
        counts[ft] = counts.get(ft, 0) + 1

    return {
        "observation_count": len(observations),
        "symptom_count": counts.get("symptom", 0),
        "negation_count": counts.get("negation", 0),
        "duration_count": counts.get("duration", 0),
        "medication_count": counts.get("medication", 0),
    }


# ── problem metrics ──────────────────────────────────────────────────


def _problem_metrics(state: dict) -> dict:
    problems = state.get("problems", [])

    kind_counts: dict[str, int] = {}
    for prob in problems:
        kind = prob.get("kind", "other")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    return {
        "problem_count": len(problems),
        "symptom_problem_count": kind_counts.get("symptom_problem", 0),
        "risk_problem_count": kind_counts.get("risk_problem", 0),
        "working_problem_count": kind_counts.get("working_problem", 0),
    }


# ── hypothesis metrics ───────────────────────────────────────────────


def _hypothesis_metrics(state: dict) -> dict:
    hypotheses = state.get("hypotheses", [])

    scores = [h.get("score", 0) for h in hypotheses]

    top = None
    if hypotheses:
        first = hypotheses[0]
        top = {
            "title": first.get("title", ""),
            "score": first.get("score", 0),
        }

    return {
        "hypothesis_count": len(hypotheses),
        "top_hypothesis": top,
        "hypothesis_score_distribution": scores,
    }


# ── risk metrics ─────────────────────────────────────────────────────


def _risk_metrics(state: dict) -> dict:
    derived = state.get("derived", {})
    red_flags = derived.get("red_flags", [])

    return {
        "red_flag_count": len(red_flags),
        "has_red_flags": len(red_flags) > 0,
    }


# ── interaction metrics ──────────────────────────────────────────────


def _interaction_metrics(state: dict) -> dict:
    questions = state.get("next_questions", [])

    by_priority: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for q in questions:
        p = q.get("priority", "low")
        if p in by_priority:
            by_priority[p] += 1
        else:
            by_priority[p] = by_priority.get(p, 0) + 1

    return {
        "question_count": len(questions),
        "questions_by_priority": by_priority,
    }


# ── data quality metrics ─────────────────────────────────────────────


def _data_quality_metrics(state: dict) -> dict:
    insights = state.get("clinical_insights", {})

    missing = insights.get("missing_information", [])
    uncertainties = insights.get("uncertainties", [])

    return {
        "missing_information_count": len(missing),
        "uncertainty_count": len(uncertainties),
    }


# ── update metrics ───────────────────────────────────────────────────


def _update_metrics(state: dict) -> dict:
    pending = state.get("pending_observations")

    if pending is None:
        return {
            "has_pending_observations": None,
            "pending_count": None,
        }

    return {
        "has_pending_observations": len(pending) > 0,
        "pending_count": len(pending),
    }
