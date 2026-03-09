"""Deterministic temporal normalization for time expressions.

Converts natural-language time expressions into standardised ISO date
strings (``YYYY-MM-DD``) or ISO 8601 durations (``P…``).

Only normalizes expressions with explicit temporal evidence.
Does NOT infer clinical onset order from mention order — mention order
and clinical time are separate concepts.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta


# ── weekday helpers ──────────────────────────────────────────────

_WEEKDAY_MAP: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# ── duration patterns ────────────────────────────────────────────

# Matches "3 days", "for 3 days", "2 weeks", "5 hours", "10 minutes"
_DURATION_RE = re.compile(
    r"^(?:for\s+)?(\d+)\s+"
    r"(second|seconds|minute|minutes|hour|hours"
    r"|day|days|week|weeks|month|months|year|years)$",
    re.IGNORECASE,
)

_UNIT_TO_ISO: dict[str, str] = {
    "second": "S",
    "seconds": "S",
    "minute": "M",
    "minutes": "M",
    "hour": "H",
    "hours": "H",
    "day": "D",
    "days": "D",
    "week": "W",
    "weeks": "W",
    "month": "M",
    "months": "M",
    "year": "Y",
    "years": "Y",
}

# Time-based units need PT prefix; date-based use P prefix
_TIME_UNITS = {"S", "H"}  # M is ambiguous — handled explicitly


def _duration_to_iso(count: int, unit_key: str) -> str:
    """Convert count + unit to ISO 8601 duration string."""
    unit_lower = unit_key.lower()
    iso_letter = _UNIT_TO_ISO.get(unit_lower)
    if iso_letter is None:
        return ""

    # "M" is ambiguous in ISO 8601 (months vs minutes).
    # Under P it means months; under PT it means minutes.
    if unit_lower in ("minute", "minutes"):
        return f"PT{count}M"
    if unit_lower in ("month", "months"):
        return f"P{count}M"

    if iso_letter in _TIME_UNITS:
        return f"PT{count}{iso_letter}"
    return f"P{count}{iso_letter}"


# ── relative-day patterns ────────────────────────────────────────

_RELATIVE_DAY_MAP: dict[str, int] = {
    "today": 0,
    "yesterday": -1,
    "day before yesterday": -2,
    "tomorrow": 1,
}

# ── "since <weekday>" ───────────────────────────────────────────

_SINCE_WEEKDAY_RE = re.compile(
    r"^since\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
    re.IGNORECASE,
)


def _most_recent_weekday(target: int, ref: datetime) -> datetime:
    """Return the most recent occurrence of *target* weekday before *ref*.

    If *ref* is on *target* weekday, returns the previous week's
    occurrence (never returns the reference day itself).
    """
    current = ref.weekday()
    days_back = (current - target) % 7
    if days_back == 0:
        days_back = 7
    return ref - timedelta(days=days_back)


# ── public API ───────────────────────────────────────────────────


def normalize_time_expression(
    expr: str,
    reference_date: datetime,
) -> str | None:
    """Normalize a single time expression.

    Only normalizes expressions with explicit temporal evidence.
    Returns ``None`` for anything ambiguous or unrecognized.

    Args:
        expr: raw time expression string.
        reference_date: anchor date for relative calculations.

    Returns:
        ISO date string (``YYYY-MM-DD``) or ISO 8601 duration (``P…``),
        or ``None`` if the expression cannot be safely normalized.
    """
    if not expr or not expr.strip():
        return None

    cleaned = expr.strip().lower()

    # 1. Relative days (today, yesterday, day before yesterday, tomorrow)
    if cleaned in _RELATIVE_DAY_MAP:
        delta = _RELATIVE_DAY_MAP[cleaned]
        dt = reference_date + timedelta(days=delta)
        return dt.strftime("%Y-%m-%d")

    # 2. "since <weekday>"
    m = _SINCE_WEEKDAY_RE.match(cleaned)
    if m:
        target = _WEEKDAY_MAP[m.group(1).lower()]
        dt = _most_recent_weekday(target, reference_date)
        return dt.strftime("%Y-%m-%d")

    # 3. "last week"
    if cleaned == "last week":
        dt = reference_date - timedelta(weeks=1)
        return dt.strftime("%Y-%m-%d")

    # 4. Bare weekday name (monday, tuesday, …)
    if cleaned in _WEEKDAY_MAP:
        target = _WEEKDAY_MAP[cleaned]
        dt = _most_recent_weekday(target, reference_date)
        return dt.strftime("%Y-%m-%d")

    # 5. Duration patterns (3 days, for 2 weeks, 5 hours, …)
    m = _DURATION_RE.match(cleaned)
    if m:
        count = int(m.group(1))
        unit = m.group(2)
        iso = _duration_to_iso(count, unit)
        if iso:
            return iso

    return None


def normalize_timeline(
    timeline: list[dict],
    reference_date: datetime,
) -> list[dict]:
    """Normalize all time expressions in a symptom timeline.

    Returns a **new** list of dicts — each original entry is shallow-copied
    with an added ``normalized_time`` field.  The original entries and the
    original ``time_expression`` values are preserved unchanged.

    Does NOT infer temporal order from mention position.

    Args:
        timeline: list of timeline entry dicts.
        reference_date: anchor date for relative calculations.

    Returns:
        list of dicts with ``normalized_time`` added to each entry.
    """
    result: list[dict] = []
    for entry in timeline:
        new_entry = dict(entry)
        expr = entry.get("time_expression")
        if expr:
            new_entry["normalized_time"] = normalize_time_expression(
                expr, reference_date,
            )
        else:
            new_entry["normalized_time"] = None
        result.append(new_entry)
    return result
