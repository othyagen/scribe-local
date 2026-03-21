"""Encounter output — aggregated clinical reasoning view.

Combines hypotheses, prioritization, evidence gaps, and red flags
into a single clinician-friendly structure.  Pure projection of
existing state fields — no new clinical logic.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


def build_encounter_output(state: dict) -> dict:
    """Build a structured encounter output from clinical state.

    Args:
        state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with ``key_findings``, ``red_flags``, and ``hypotheses``.
    """
    hypotheses = state.get("hypotheses", [])
    prioritization = state.get("hypothesis_prioritization", [])
    evidence_gaps = state.get("hypothesis_evidence_gaps", {})
    red_flags_raw = state.get("derived", {}).get("red_flags", [])
    symptoms = state.get("symptoms", [])
    negations = state.get("negations", [])

    # Key findings: symptoms + negations.
    key_findings = list(symptoms) + list(negations)

    # Red flags: label + severity.
    red_flags = [
        {"label": rf.get("label", ""), "severity": rf.get("severity", "")}
        for rf in red_flags_raw
        if rf.get("label")
    ]

    # Build lookup indices.
    prio_index = {
        p.get("hypothesis_id", ""): p.get("priority_class", "less_likely")
        for p in prioritization
    }

    me_index: dict[str, dict] = {}
    for me in evidence_gaps.get("missing_evidence", []):
        me_index[me.get("hypothesis_id", "")] = me

    sq_index: dict[str, dict] = {}
    for sq in evidence_gaps.get("suggested_questions", []):
        title = sq.get("target_hypothesis", "")
        if title and title not in sq_index:
            sq_index[title] = sq

    # Build hypothesis entries sorted by rank.
    hyp_entries: list[dict] = []
    for hyp in sorted(hypotheses, key=lambda h: h.get("rank", 0)):
        hyp_id = hyp.get("id", "")
        title = hyp.get("title", "")
        rank = hyp.get("rank", 0)
        explanation = hyp.get("explanation", {})

        # Evidence from explanation layer.
        present_evidence = list(explanation.get("supporting_evidence", []))
        conflicting_evidence = list(explanation.get("conflicting_evidence", []))

        # Priority class from prioritization layer.
        priority_class = prio_index.get(hyp_id, "less_likely")

        # Findings from evidence gaps layer.
        me_entry = me_index.get(hyp_id, {})
        findings = [
            {
                "name": f.get("name", ""),
                "status": f.get("status", "absent"),
                "reason": f.get("reason", ""),
            }
            for f in me_entry.get("findings", [])
        ]

        # Next question: first matching suggested question for this hypothesis.
        sq = sq_index.get(title)
        next_question = None
        if sq:
            next_question = {
                "question": sq.get("question", ""),
                "reason": sq.get("reason", ""),
            }

        hyp_entries.append({
            "title": title,
            "rank": rank,
            "priority_class": priority_class,
            "present_evidence": present_evidence,
            "conflicting_evidence": conflicting_evidence,
            "findings": findings,
            "next_question": next_question,
        })

    # Combined view: group by priority class, sorted by rank within each.
    combined: dict[str, list[dict]] = {
        "must_not_miss": [],
        "most_likely": [],
        "less_likely": [],
    }
    for entry in hyp_entries:
        pc = entry["priority_class"]
        if pc in combined:
            combined[pc].append(entry)

    return {
        "key_findings": key_findings,
        "red_flags": red_flags,
        "hypotheses": hyp_entries,
        "combined_hypotheses": combined,
    }
