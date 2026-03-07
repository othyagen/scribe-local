"""Extract symptom–time-expression pairs from normalized segments.

Pairs each detected symptom with the nearest time expression in the same
segment.  Deduplicates by symptom (case-insensitive, first occurrence wins).

No ML, no LLM — regex only.  Reuses patterns from ``app.extractors``.
"""

from __future__ import annotations

import re

from app.extractors import _SYMPTOM_PATTERNS, _DURATION_PATTERN

# Relative time expressions not covered by _DURATION_PATTERN
_TIME_EXPR_PATTERN = re.compile(
    r"\b("
    r"since yesterday|since last night|since last week|since this morning"
    r"|started today|started yesterday"
    r"|for (?:two|three|four|five|six|seven|eight|nine|ten)"
    r" (?:days?|weeks?|months?|years?|hours?|minutes?)"
    r")\b",
    re.IGNORECASE,
)


def _extract_time_expression(text: str) -> str | None:
    """Return the first time expression found in *text*, or None."""
    # Try numeric duration first ("3 days")
    m = _DURATION_PATTERN.search(text)
    if m:
        return f"{m.group(1)} {m.group(2).lower()}"
    # Try relative phrases ("since yesterday")
    m = _TIME_EXPR_PATTERN.search(text)
    if m:
        return m.group(1).lower()
    return None


def extract_symptom_timeline(segments: list[dict]) -> list[dict]:
    """Extract symptom–time pairs from normalized segments.

    Returns a list of dicts, each with:
      - ``symptom``: str
      - ``time_expression``: str | None
      - ``seg_id``: str | None
      - ``speaker_id``: str | None
      - ``t_start``: float | None

    Deduplicated by symptom (case-insensitive); first occurrence wins.
    Ordered by segment order (which reflects ``t_start``).
    """
    seen: set[str] = set()
    results: list[dict] = []

    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        time_expr = _extract_time_expression(text)

        for pattern, keyword in _SYMPTOM_PATTERNS:
            if keyword.lower() in seen:
                continue
            if pattern.search(text):
                seen.add(keyword.lower())
                results.append({
                    "symptom": keyword,
                    "time_expression": time_expr,
                    "seg_id": seg.get("seg_id"),
                    "speaker_id": seg.get("speaker_id"),
                    "t_start": seg.get("t0"),
                })

    return results
