"""Anatomical site mention extraction.

Extracts anatomical site mentions per-segment.  Does NOT link sites
to symptoms — that happens in the structured symptom model with
conservative linking.

Returns all occurrences (same site in different segments produces
separate entries), consistent with the observation philosophy.

Pure function — no ML, no LLM, no I/O.
"""

from __future__ import annotations

import re

from app.extractor_vocab import load_vocab


_DEFAULT_SITES: list[str] = [
    "axillary",
    "cervical",
    "costovertebral",
    "epigastric",
    "flank",
    "frontal",
    "inguinal",
    "interscapular",
    "left lower quadrant",
    "left upper quadrant",
    "lumbar",
    "occipital",
    "orbital",
    "periorbital",
    "periumbilical",
    "plantar",
    "precordial",
    "retrosternal",
    "right lower quadrant",
    "right upper quadrant",
    "sacral",
    "submandibular",
    "substernal",
    "suprapubic",
    "temporal",
    "thoracic",
]

SITE_KEYWORDS: list[str] = load_vocab("anatomical_sites", _DEFAULT_SITES)

# Pre-compile site patterns (longest first for multi-word matches)
_SITE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), kw)
    for kw in sorted(SITE_KEYWORDS, key=len, reverse=True)
]


def extract_sites(segments: list[dict]) -> list[dict]:
    """Extract anatomical site mentions from segments.

    Args:
        segments: list of normalized segment dicts.

    Returns:
        list of ``{site, seg_id, speaker_id, t_start}`` dicts.
        Does NOT deduplicate globally — returns all occurrences.
    """
    results: list[dict] = []

    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        seg_id = seg.get("seg_id")
        speaker_id = seg.get("speaker_id")
        t_start = seg.get("t0")

        # Track sites found in this segment to avoid duplicates within same segment
        seen_in_seg: set[str] = set()

        for pattern, keyword in _SITE_PATTERNS:
            if keyword.lower() in seen_in_seg:
                continue
            if pattern.search(text):
                seen_in_seg.add(keyword.lower())
                results.append({
                    "site": keyword,
                    "seg_id": seg_id,
                    "speaker_id": speaker_id,
                    "t_start": t_start,
                })

    return results
