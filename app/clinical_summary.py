"""Clinical summary — format-neutral structured summary from clinical state.

Assembles a reusable intermediate summary object from the full clinical
state.  Designed to support multiple downstream renderers, UI views,
and export formats without coupling to any specific note structure.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


def build_clinical_summary(clinical_state: dict) -> dict:
    """Build a structured clinical summary from clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with format-neutral summary fields.
    """
    derived = clinical_state.get("derived", {})

    return {
        "key_findings": _extract_key_findings(clinical_state),
        "active_problems": _extract_active_problems(clinical_state),
        "ranked_hypotheses": _extract_ranked_hypotheses(clinical_state),
        "red_flags": _extract_red_flags(derived),
        "timeline_summary": _extract_timeline_summary(clinical_state),
        "problem_narrative": _extract_problem_narrative(derived),
        "symptom_groups": _extract_symptom_groups(clinical_state),
        "medications": _extract_medications(clinical_state),
    }


def _extract_key_findings(state: dict) -> list[dict]:
    """Extract key findings from symptoms, negations, and durations."""
    findings: list[dict] = []

    for symptom in state.get("symptoms", []):
        findings.append({"type": "symptom", "value": symptom})

    for negation in state.get("negations", []):
        findings.append({"type": "negation", "value": negation})

    for duration in state.get("durations", []):
        findings.append({"type": "duration", "value": duration})

    return findings


def _extract_active_problems(state: dict) -> list[dict]:
    """Extract active problems with their evidence summary."""
    result: list[dict] = []
    for prob in state.get("problems", []):
        if prob.get("status") != "active":
            continue
        supporting = prob.get("supporting_observations", [])
        result.append({
            "id": prob.get("id", ""),
            "title": prob.get("title", ""),
            "kind": prob.get("kind", ""),
            "priority": prob.get("priority", "normal"),
            "onset": prob.get("onset"),
            "evidence_count": len(supporting),
        })
    return result


def _extract_ranked_hypotheses(state: dict) -> list[dict]:
    """Extract ranked hypotheses with explanation summaries."""
    result: list[dict] = []
    for hyp in state.get("hypotheses", []):
        explanation = hyp.get("explanation", {})
        result.append({
            "id": hyp.get("id", ""),
            "title": hyp.get("title", ""),
            "rank": hyp.get("rank", 0),
            "score": hyp.get("score", 0),
            "confidence": hyp.get("confidence", "low"),
            "supporting_count": len(hyp.get("supporting_observations", [])),
            "summary": explanation.get("summary", ""),
        })
    return result


def _extract_red_flags(derived: dict) -> list[dict]:
    """Extract red flags with labels and evidence."""
    result: list[dict] = []
    for flag in derived.get("red_flags", []):
        result.append({
            "label": flag.get("label", ""),
            "evidence": flag.get("evidence", []),
        })
    return result


def _extract_timeline_summary(state: dict) -> list[dict]:
    """Extract symptom timeline entries."""
    result: list[dict] = []
    for entry in state.get("timeline", []):
        result.append({
            "symptom": entry.get("symptom", ""),
            "time_expression": entry.get("time_expression"),
        })
    return result


def _extract_problem_narrative(derived: dict) -> dict:
    """Extract the problem narrative."""
    narrative = derived.get("problem_narrative", {})
    return {
        "positive_features": narrative.get("positive_features", []),
        "negative_features": narrative.get("negative_features", []),
        "narrative": narrative.get("narrative", ""),
    }


def _extract_symptom_groups(state: dict) -> list[dict]:
    """Extract symptom group summaries."""
    result: list[dict] = []
    for grp in state.get("symptom_groups", []):
        result.append({
            "id": grp.get("id", ""),
            "title": grp.get("title", ""),
            "systems": grp.get("systems", []),
            "temporal_bucket": grp.get("temporal_bucket", "unknown"),
            "observation_count": len(grp.get("observations", [])),
        })
    return result


def _extract_medications(state: dict) -> list[str]:
    """Extract medication list."""
    return list(state.get("medications", []))
