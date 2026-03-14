"""Numeric pain intensity extraction (X/10 scale).

Extracts numeric pain intensity scores from clinical transcript
segments.  Supports formats: X/10, X out of 10, pain level X,
pain score X, VAS X.  Value clamped 0-10.

Deduplicates by (value, seg_id).

Pure function — no ML, no LLM, no I/O.
"""

from __future__ import annotations

import re


# Pattern requires pain/severity context or bare X/10 format
_INTENSITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # X/10 — bare format, common enough to not require context
    (re.compile(
        r"\b(\d{1,2})\s*/\s*10\b",
        re.IGNORECASE,
    ), "X/10"),
    # X out of 10
    (re.compile(
        r"\b(\d{1,2})\s+out\s+of\s+10\b",
        re.IGNORECASE,
    ), "X out of 10"),
    # pain level X
    (re.compile(
        r"\bpain\s+level\s+(\d{1,2})\b",
        re.IGNORECASE,
    ), "pain level X"),
    # pain score X
    (re.compile(
        r"\bpain\s+score\s+(?:of\s+)?(\d{1,2})\b",
        re.IGNORECASE,
    ), "pain score X"),
    # VAS X
    (re.compile(
        r"\bVAS\s+(\d{1,2})\b",
        re.IGNORECASE,
    ), "VAS X"),
]

# Context words that indicate the X/10 is NOT about pain (pages, etc.)
_FALSE_POSITIVE_PREFIX = re.compile(
    r"\b(?:page|chapter|section|question|item|step|round)\s+$",
    re.IGNORECASE,
)


def extract_intensities(segments: list[dict]) -> list[dict]:
    """Extract numeric pain intensity scores from segments.

    Args:
        segments: list of normalized segment dicts.

    Returns:
        list of ``{value, raw_text, scale, seg_id, speaker_id, t_start}``
        dicts.  Value clamped 0-10.  Deduplicates by (value, seg_id).
    """
    results: list[dict] = []
    seen: set[tuple[int, str | None]] = set()

    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        seg_id = seg.get("seg_id")

        for pattern, scale in _INTENSITY_PATTERNS:
            for m in pattern.finditer(text):
                raw_value = int(m.group(1))

                # Skip false positives for bare X/10 format
                if scale == "X/10":
                    prefix = text[:m.start()]
                    if _FALSE_POSITIVE_PREFIX.search(prefix):
                        continue

                value = max(0, min(10, raw_value))
                raw_text = m.group(0)
                key = (value, seg_id)

                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "value": value,
                    "raw_text": raw_text,
                    "scale": "numeric",
                    "seg_id": seg_id,
                    "speaker_id": seg.get("speaker_id"),
                    "t_start": seg.get("t0"),
                })

    return results
