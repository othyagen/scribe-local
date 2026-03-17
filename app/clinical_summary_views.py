"""Clinical summary views — derived selectors on the clinical summary.

Reads the existing ``clinical_summary`` dict and returns format-neutral
views that select, group, and reshape data for different downstream use
cases.  Does not modify the summary or any upstream state.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


def build_summary_views(clinical_summary: dict) -> dict:
    """Build all summary views from a clinical summary.

    Args:
        clinical_summary: dict produced by :func:`build_clinical_summary`.

    Returns:
        dict with view names as keys.
    """
    return {
        "overview_summary": build_overview_summary(clinical_summary),
        "reasoning_summary": build_reasoning_summary(clinical_summary),
        "risk_summary": build_risk_summary(clinical_summary),
        "symptom_summary": build_symptom_summary(clinical_summary),
    }


def build_overview_summary(clinical_summary: dict) -> dict:
    """High-level clinical overview for quick consumption.

    Selects: symptom count, problem count, top hypothesis, medication
    count, and whether red flags are present.
    """
    findings = clinical_summary.get("key_findings", [])
    problems = clinical_summary.get("active_problems", [])
    hypotheses = clinical_summary.get("ranked_hypotheses", [])
    red_flags = clinical_summary.get("red_flags", [])
    medications = clinical_summary.get("medications", [])
    narrative = clinical_summary.get("problem_narrative", {})

    symptom_count = sum(
        1 for f in findings if f.get("type") == "symptom"
    )
    negation_count = sum(
        1 for f in findings if f.get("type") == "negation"
    )

    top_hypothesis = None
    if hypotheses:
        h = hypotheses[0]
        top_hypothesis = {
            "title": h.get("title", ""),
            "rank": h.get("rank", 0),
            "confidence": h.get("confidence", "low"),
        }

    return {
        "symptom_count": symptom_count,
        "negation_count": negation_count,
        "problem_count": len(problems),
        "medication_count": len(medications),
        "has_red_flags": len(red_flags) > 0,
        "top_hypothesis": top_hypothesis,
        "narrative": narrative.get("narrative", ""),
    }


def build_reasoning_summary(clinical_summary: dict) -> dict:
    """Diagnostic reasoning view: hypotheses, evidence, and problems.

    Groups hypotheses by confidence level and links to active problems.
    """
    hypotheses = clinical_summary.get("ranked_hypotheses", [])
    problems = clinical_summary.get("active_problems", [])

    by_confidence: dict[str, list[dict]] = {
        "strong": [],
        "moderate": [],
        "low": [],
    }
    for hyp in hypotheses:
        conf = hyp.get("confidence", "low")
        bucket = by_confidence.get(conf, by_confidence["low"])
        bucket.append({
            "title": hyp.get("title", ""),
            "rank": hyp.get("rank", 0),
            "score": hyp.get("score", 0),
            "supporting_count": hyp.get("supporting_count", 0),
            "summary": hyp.get("summary", ""),
        })

    # Problems by kind
    by_kind: dict[str, list[dict]] = {}
    for prob in problems:
        kind = prob.get("kind", "other")
        by_kind.setdefault(kind, []).append({
            "title": prob.get("title", ""),
            "priority": prob.get("priority", "normal"),
            "evidence_count": prob.get("evidence_count", 0),
        })

    return {
        "hypothesis_count": len(hypotheses),
        "hypotheses_by_confidence": by_confidence,
        "problem_count": len(problems),
        "problems_by_kind": by_kind,
    }


def build_risk_summary(clinical_summary: dict) -> dict:
    """Risk-focused view: red flags, urgent problems, and high-rank hypotheses."""
    red_flags = clinical_summary.get("red_flags", [])
    problems = clinical_summary.get("active_problems", [])
    hypotheses = clinical_summary.get("ranked_hypotheses", [])

    urgent_problems = [
        {
            "title": p.get("title", ""),
            "kind": p.get("kind", ""),
            "evidence_count": p.get("evidence_count", 0),
        }
        for p in problems
        if p.get("priority") == "urgent"
    ]

    top_hypotheses = [
        {
            "title": h.get("title", ""),
            "rank": h.get("rank", 0),
            "score": h.get("score", 0),
            "confidence": h.get("confidence", "low"),
        }
        for h in hypotheses[:3]
    ]

    return {
        "red_flag_count": len(red_flags),
        "red_flags": [
            {"label": f.get("label", ""), "evidence": f.get("evidence", [])}
            for f in red_flags
        ],
        "urgent_problem_count": len(urgent_problems),
        "urgent_problems": urgent_problems,
        "top_hypotheses": top_hypotheses,
    }


def build_symptom_summary(clinical_summary: dict) -> dict:
    """Symptom-focused view: findings, timeline, and groups by system."""
    findings = clinical_summary.get("key_findings", [])
    timeline = clinical_summary.get("timeline_summary", [])
    groups = clinical_summary.get("symptom_groups", [])

    symptoms = [
        f.get("value", "") for f in findings if f.get("type") == "symptom"
    ]
    negations = [
        f.get("value", "") for f in findings if f.get("type") == "negation"
    ]
    durations = [
        f.get("value", "") for f in findings if f.get("type") == "duration"
    ]

    # Groups indexed by system
    by_system: dict[str, list[dict]] = {}
    for grp in groups:
        for system in grp.get("systems", ["general"]):
            by_system.setdefault(system, []).append({
                "title": grp.get("title", ""),
                "temporal_bucket": grp.get("temporal_bucket", "unknown"),
                "observation_count": grp.get("observation_count", 0),
            })

    return {
        "symptoms": symptoms,
        "negations": negations,
        "durations": durations,
        "timeline": [
            {
                "symptom": e.get("symptom", ""),
                "time_expression": e.get("time_expression"),
            }
            for e in timeline
        ],
        "groups_by_system": by_system,
    }
