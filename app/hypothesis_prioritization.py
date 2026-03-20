"""Optional hypothesis prioritization — must-not-miss clinical safety layer.

Classifies hypotheses into priority classes without modifying the
existing ranking, scoring, or explanation outputs.  Purely additive
post-processing layer.

Priority classes:
  - ``most_likely``: top-ranked hypothesis (rank 1), not dangerous.
  - ``must_not_miss``: dangerous condition with supporting evidence
    or a relevant red flag.
  - ``less_likely``: everything else.

A dangerous condition is NOT elevated to ``must_not_miss`` by name
alone — it requires at least one piece of supporting evidence or a
matching red flag.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


# ── dangerous conditions registry ───────────────────────────────────

DANGEROUS_CONDITIONS: dict[str, str] = {
    "acute coronary syndrome": "life-threatening cardiac event",
    "meningitis": "rapidly progressive CNS infection",
    "pulmonary embolism": "potentially fatal thromboembolic event",
    "sepsis": "systemic infection with organ dysfunction risk",
    "ectopic pregnancy": "surgical emergency if ruptured",
    "gastrointestinal bleeding": "risk of hemodynamic compromise",
    "gi bleed": "risk of hemodynamic compromise",
    "aortic dissection": "rapidly fatal without intervention",
    "stroke": "time-critical neurological emergency",
    "tension pneumothorax": "immediately life-threatening",
}


# ── public API ──────────────────────────────────────────────────────


def prioritize_hypotheses(
    hypotheses: list[dict],
    red_flags: list[dict] | None = None,
) -> list[dict]:
    """Classify hypotheses into priority classes.

    This is an optional view layer — it never modifies the input
    hypotheses.  Consumers can use standard ranking, prioritization,
    or both.

    Args:
        hypotheses: ranked hypothesis list from ``state["hypotheses"]``.
        red_flags: red flag list from ``state["derived"]["red_flags"]``.

    Returns:
        List of prioritization entries (one per hypothesis), each with
        ``hypothesis_id``, ``title``, ``rank``, ``priority_class``,
        ``reason``, and ``evidence``.
    """
    if not hypotheses:
        return []

    red_flag_labels = _extract_red_flag_labels(red_flags or [])

    result: list[dict] = []
    for hyp in hypotheses:
        entry = _classify(hyp, red_flag_labels)
        result.append(entry)

    return result


# ── internal helpers ────────────────────────────────────────────────


def _extract_red_flag_labels(red_flags: list[dict]) -> set[str]:
    """Collect lowercased red flag labels for matching."""
    labels: set[str] = set()
    for rf in red_flags:
        label = rf.get("label", "")
        if label:
            labels.add(label.strip().lower())
        for ev in rf.get("evidence", []):
            if isinstance(ev, str):
                labels.add(ev.strip().lower())
    return labels


def _get_supporting_values(hyp: dict) -> list[str]:
    """Extract human-readable supporting evidence values."""
    values: list[str] = []
    for ev in hyp.get("supporting_observations", []):
        if isinstance(ev, dict):
            obs_id = ev.get("observation_id", "")
            if obs_id:
                values.append(obs_id)
        elif isinstance(ev, str):
            values.append(ev)
    return values


def _has_supporting_evidence(hyp: dict) -> bool:
    """Return True if the hypothesis has any supporting observations."""
    obs = hyp.get("supporting_observations", [])
    return len(obs) > 0


def _has_matching_red_flag(hyp: dict, red_flag_labels: set[str]) -> bool:
    """Return True if any red flag label overlaps with hypothesis evidence."""
    if not red_flag_labels:
        return False
    title_lower = hyp.get("title", "").strip().lower()
    # Check if hypothesis title keywords appear in red flag evidence.
    for word in title_lower.split():
        if len(word) >= 4 and word in red_flag_labels:
            return True
    # Check if any supporting observation ID overlaps with red flag labels.
    for ev in hyp.get("supporting_observations", []):
        if isinstance(ev, dict):
            obs_id = ev.get("observation_id", "").strip().lower()
            if obs_id and obs_id in red_flag_labels:
                return True
    return False


def _classify(hyp: dict, red_flag_labels: set[str]) -> dict:
    """Classify a single hypothesis into a priority class."""
    title = hyp.get("title", "")
    title_lower = title.strip().lower()
    rank = hyp.get("rank", 0)
    evidence = _get_supporting_values(hyp)

    is_dangerous = title_lower in DANGEROUS_CONDITIONS
    has_evidence = _has_supporting_evidence(hyp)
    has_red_flag = _has_matching_red_flag(hyp, red_flag_labels)

    if is_dangerous and (has_evidence or has_red_flag):
        danger_reason = DANGEROUS_CONDITIONS[title_lower]
        basis = []
        if has_evidence:
            basis.append("supporting evidence present")
        if has_red_flag:
            basis.append("relevant red flag present")
        reason = f"{danger_reason} — {', '.join(basis)}"
        priority_class = "must_not_miss"
    elif rank == 1:
        reason = "top-ranked hypothesis by evidence strength"
        priority_class = "most_likely"
    else:
        reason = "lower-ranked hypothesis without danger signals"
        priority_class = "less_likely"

    return {
        "hypothesis_id": hyp.get("id", ""),
        "title": title,
        "rank": rank,
        "priority_class": priority_class,
        "reason": reason,
        "evidence": evidence,
    }
