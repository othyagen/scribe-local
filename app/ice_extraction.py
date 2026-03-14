"""Conservative ICE (Ideas, Concerns, Expectations) extraction.

High-precision extraction only — triggers on specific multi-word phrases
that clearly indicate patient ideas, concerns, or expectations.  Avoids
vague single-word triggers.

Each trigger must be followed by at least 2 words of content.  Captures
trigger + up to 50 chars trailing text.  Deduplicates by text
(case-insensitive).

Pure function — no ML, no LLM, no I/O.
"""

from __future__ import annotations

import re


# ── trigger patterns ──────────────────────────────────────────────

_IDEA_TRIGGERS = [
    r"I think it might be",
    r"I think it could be",
    r"I wonder if it'?s",
    r"could this be",
    r"maybe it'?s",
    r"I read that it could be",
]

_CONCERN_TRIGGERS = [
    r"I'?m worried about",
    r"I'?m worried that",
    r"I'?m afraid it might be",
    r"I'?m concerned about",
    r"my concern is",
    r"I'?m scared it could be",
]

_EXPECTATION_TRIGGERS = [
    r"I was hoping for",
    r"I was hoping you could",
    r"I'?d like to get",
    r"I would like",
    r"I was expecting",
    r"could I get",
]


def _build_pattern(triggers: list[str]) -> re.Pattern[str]:
    """Build a compiled pattern from trigger phrases.

    Captures the trigger (group 1) and trailing content (group 2).
    """
    alternatives = "|".join(triggers)
    return re.compile(
        r"\b(" + alternatives + r")\s+(.{1,50})",
        re.IGNORECASE,
    )


_IDEA_PAT = _build_pattern(_IDEA_TRIGGERS)
_CONCERN_PAT = _build_pattern(_CONCERN_TRIGGERS)
_EXPECTATION_PAT = _build_pattern(_EXPECTATION_TRIGGERS)

_MIN_CONTENT_WORDS = 2


def extract_ice(segments: list[dict]) -> dict:
    """Extract ICE (Ideas, Concerns, Expectations) from segments.

    Args:
        segments: list of normalized segment dicts.

    Returns:
        dict with keys ``ideas``, ``concerns``, ``expectations``,
        each a list of ``{text, seg_id, speaker_id, t_start}`` dicts.
    """
    ideas = _extract_category(segments, _IDEA_PAT)
    concerns = _extract_category(segments, _CONCERN_PAT)
    expectations = _extract_category(segments, _EXPECTATION_PAT)

    return {
        "ideas": ideas,
        "concerns": concerns,
        "expectations": expectations,
    }


def _extract_category(
    segments: list[dict],
    pattern: re.Pattern[str],
) -> list[dict]:
    """Extract items for one ICE category."""
    results: list[dict] = []
    seen: set[str] = set()

    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        for m in pattern.finditer(text):
            trigger = m.group(1)
            content = m.group(2).strip().rstrip(".,;!?")

            # Require at least 2 words of content
            if len(content.split()) < _MIN_CONTENT_WORDS:
                continue

            full_text = f"{trigger} {content}"
            key = full_text.lower()
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "text": full_text,
                "seg_id": seg.get("seg_id"),
                "speaker_id": seg.get("speaker_id"),
                "t_start": seg.get("t0"),
            })

    return results
