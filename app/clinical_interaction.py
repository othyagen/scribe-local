"""Clinical interaction — next-step question derivation from insights.

Reads ``clinical_insights`` from state and produces a prioritised,
filtered list of follow-up questions.  Does not perform any new
diagnostic reasoning — only reshapes and ranks existing suggestions.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

# Maximum questions returned.
_MAX_QUESTIONS = 5

# Priority order for question categories (lower index = higher priority).
_CATEGORY_PRIORITY: dict[str, int] = {
    "allergy": 0,
    "duration": 1,
    "severity": 2,
}

_DEFAULT_PRIORITY = 99


def derive_next_questions(clinical_state: dict) -> list[dict]:
    """Derive prioritised next-step questions from clinical insights.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of dicts, each with ``question``, ``reason``, ``priority``.
        Sorted by priority (highest first), capped at :data:`_MAX_QUESTIONS`.
    """
    insights = clinical_state.get("clinical_insights")
    if not insights:
        return []

    suggested = insights.get("suggested_questions", [])
    missing = insights.get("missing_information", [])

    # Build a set of "important" gaps — symptoms that have missing info
    # flagged as missing_duration or missing_dosage (safety-relevant).
    important_gaps: set[str] = set()
    for gap in missing:
        cat = gap.get("category", "")
        if cat in ("missing_duration", "missing_dosage"):
            related = gap.get("related")
            if related:
                important_gaps.add(related.lower())

    questions: list[dict] = []
    seen: set[str] = set()

    for sq in suggested:
        q_text = sq.get("question", "")
        category = sq.get("category", "")
        related = sq.get("related")

        # Deduplicate by question text.
        key = q_text.lower()
        if key in seen:
            continue
        seen.add(key)

        # Determine priority.
        base = _CATEGORY_PRIORITY.get(category, _DEFAULT_PRIORITY)

        # Boost if the related finding also appears in important gaps.
        if related and related.lower() in important_gaps:
            base = max(0, base - 1)

        # Build reason from category.
        reason = _build_reason(category, related)

        priority_label = _priority_label(base)

        questions.append({
            "question": q_text,
            "reason": reason,
            "priority": priority_label,
            "_sort_key": base,
        })

    # Sort: lower _sort_key = higher priority, then alphabetical for stability.
    questions.sort(key=lambda q: (q["_sort_key"], q["question"]))

    # Strip internal sort key and cap.
    return [
        {"question": q["question"], "reason": q["reason"], "priority": q["priority"]}
        for q in questions[:_MAX_QUESTIONS]
    ]


def _build_reason(category: str, related: str | None) -> str:
    """Build a human-readable reason string."""
    if category == "allergy":
        return "Medications prescribed without documented allergy status"
    if category == "duration" and related:
        return f"No duration recorded for {related}"
    if category == "severity" and related:
        return f"No severity recorded for {related}"
    if category == "duration":
        return "Missing symptom duration"
    if category == "severity":
        return "Missing symptom severity"
    return "Additional information needed"


def _priority_label(sort_key: int) -> str:
    """Map numeric sort key to a priority label."""
    if sort_key <= 0:
        return "high"
    if sort_key <= 1:
        return "medium"
    return "low"
