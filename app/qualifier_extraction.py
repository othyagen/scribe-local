"""Semantic qualifier extraction — deterministic clinical qualifier detection.

Extracts clinical qualifiers (severity, onset, character, pattern,
progression, laterality, radiation, aggravating/relieving factors)
from transcript segments and links them to nearby symptom mentions.

No ML, no LLM — regex + keyword matching only.
"""

from __future__ import annotations

import re

from app.extractors import SYMPTOM_KEYWORDS


# ── qualifier vocabulary ────────────────────────────────────────────

_SEVERITY_TERMS: dict[str, str] = {
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "intense": "intense",
    "slight": "mild",
    "terrible": "severe",
    "excruciating": "severe",
    "unbearable": "severe",
    "worst": "severe",
}

_ONSET_TERMS: dict[str, str] = {
    "sudden": "sudden",
    "suddenly": "sudden",
    "gradual": "gradual",
    "gradually": "gradual",
    "acute": "acute",
    "abrupt": "sudden",
    "abruptly": "sudden",
    "insidious": "gradual",
}

_CHARACTER_TERMS: dict[str, str] = {
    "cramping": "cramping",
    "burning": "burning",
    "stabbing": "stabbing",
    "dull": "dull",
    "pressure-like": "pressure-like",
    "pressure": "pressure-like",
    "sharp": "sharp",
    "aching": "aching",
    "throbbing": "throbbing",
    "squeezing": "squeezing",
    "tearing": "tearing",
    "colicky": "colicky",
    "gnawing": "gnawing",
    "productive": "productive",
    "dry": "dry",
    "shooting": "shooting",
    "tingling": "tingling",
    "pounding": "throbbing",
    "stinging": "stinging",
}

_PATTERN_TERMS: dict[str, str] = {
    "intermittent": "intermittent",
    "constant": "constant",
    "episodic": "episodic",
    "continuous": "constant",
    "on and off": "intermittent",
    "comes and goes": "intermittent",
    "persistent": "constant",
    "recurrent": "episodic",
}

_PROGRESSION_TERMS: dict[str, str] = {
    "worsening": "worsening",
    "getting worse": "worsening",
    "worse": "worsening",
    "improving": "improving",
    "getting better": "improving",
    "better": "improving",
    "stable": "stable",
    "unchanged": "stable",
    "progressing": "worsening",
    "resolving": "improving",
}

_LATERALITY_TERMS: dict[str, str] = {
    "left": "left",
    "right": "right",
    "bilateral": "bilateral",
    "unilateral": "unilateral",
    "both sides": "bilateral",
    "left-sided": "left",
    "right-sided": "right",
    "left side": "left",
    "right side": "right",
}


# ── compiled patterns ──────────────────────────────────────────────

def _build_vocab_pattern(terms: dict[str, str]) -> re.Pattern[str]:
    """Compile a case-insensitive pattern from vocabulary keys."""
    escaped = [re.escape(k) for k in sorted(terms, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


_SEVERITY_PAT = _build_vocab_pattern(_SEVERITY_TERMS)
_ONSET_PAT = _build_vocab_pattern(_ONSET_TERMS)
_CHARACTER_PAT = _build_vocab_pattern(_CHARACTER_TERMS)
_PATTERN_PAT = _build_vocab_pattern(_PATTERN_TERMS)
_PROGRESSION_PAT = _build_vocab_pattern(_PROGRESSION_TERMS)
_LATERALITY_PAT = _build_vocab_pattern(_LATERALITY_TERMS)

_BOUNDARY_WORDS = (
    r"(?=\s+(?:worse|better|relieved|aggravated|exacerbated|worsened|"
    r"improved|eased|helped|triggered|provoked|and\s+(?:worse|better|"
    r"relieved|aggravated|exacerbated))|[.,;!?]|$)"
)

_RADIATION_PAT = re.compile(
    r"\b(?:radiat(?:es?|ing)|spread(?:s|ing)?|extending|refers?)\s+to\s+"
    r"(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)" + _BOUNDARY_WORDS,
    re.IGNORECASE,
)

_FACTOR_STOP = r"(?:\b(?:and|or|but|worse|better|relieved|aggravated|exacerbated|worsened|improved|eased|helped|triggered|provoked)\b|[.,;!?])"

_AGGRAVATING_PAT = re.compile(
    r"\b(?:worse\s+with|worsened\s+by|aggravated\s+by|triggered\s+by|"
    r"exacerbated\s+by|increases?\s+with|provoked\s+by|"
    r"gets?\s+worse\s+when|worse\s+when|worse\s+if|"
    r"worse\s+after|worsened\s+after)\s+"
    r"([a-z]+(?:\s+(?!" + _FACTOR_STOP + r")[a-z]+){0,4})",
    re.IGNORECASE,
)

_RELIEVING_PAT = re.compile(
    r"\b(?:relieved\s+by|better\s+with|improved\s+by|eased\s+by|"
    r"helped\s+by|decreases?\s+with|alleviated\s+by|"
    r"gets?\s+better\s+when|better\s+when|better\s+if)\s+"
    r"([a-z]+(?:\s+(?!" + _FACTOR_STOP + r")[a-z]+){0,4})",
    re.IGNORECASE,
)

# Leading subject pronouns to strip from captured factor text
_LEADING_PRONOUN_PAT = re.compile(
    r"^(?:I|you|he|she|it|we|they|my|your|his|her|its|our|their)\s+",
    re.IGNORECASE,
)

# Negation triggers — skip qualifier if preceded by these
_NEGATION_PREFIX_PAT = re.compile(
    r"\b(?:no|not|denies|denied|without|absent)\s+$",
    re.IGNORECASE,
)

# Pre-compiled symptom patterns (longest first)
_SYMPTOM_PATS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), kw)
    for kw in sorted(SYMPTOM_KEYWORDS, key=len, reverse=True)
]


# ── token proximity ────────────────────────────────────────────────

_TOKEN_WINDOW = 15


def _char_distance(symptom_match: re.Match, qualifier_match: re.Match) -> int:
    """Character distance between two matches (0 if overlapping)."""
    s_start, s_end = symptom_match.span()
    q_start, q_end = qualifier_match.span()

    if q_end <= s_start:
        return s_start - q_end
    elif s_end <= q_start:
        return q_start - s_end
    return 0


def _token_distance(text: str, symptom_match: re.Match,
                    qualifier_match: re.Match) -> int:
    """Token count between two matches (-1 if overlapping)."""
    s_start, s_end = symptom_match.span()
    q_start, q_end = qualifier_match.span()

    if q_end <= s_start:
        between = text[q_end:s_start]
    elif s_end <= q_start:
        between = text[s_end:q_start]
    else:
        return -1  # overlapping

    return len(between.split()) if between.strip() else 0


def _within_token_window(text: str, symptom_match: re.Match,
                         qualifier_match: re.Match) -> bool:
    """Check if qualifier is within ±TOKEN_WINDOW tokens of symptom."""
    dist = _token_distance(text, symptom_match, qualifier_match)
    return dist < 0 or dist <= _TOKEN_WINDOW


def _is_negated(text: str, match: re.Match) -> bool:
    """Check if a match is preceded by a negation trigger."""
    prefix = text[:match.start()]
    # Check the last few words before the match
    return bool(_NEGATION_PREFIX_PAT.search(prefix))


# ── per-segment extraction ─────────────────────────────────────────

def _extract_list_matches(
    text: str,
    pattern: re.Pattern[str],
    symptom_match: re.Match,
) -> list[str]:
    """Extract all factor matches near a symptom, return cleaned list."""
    results: list[str] = []
    for m in pattern.finditer(text):
        if _within_token_window(text, symptom_match, m):
            value = m.group(1).strip().rstrip(".,;!?")
            if value and value.lower() not in {r.lower() for r in results}:
                results.append(value)
    return results


def _extract_segment_qualifiers(
    text: str,
    known_symptoms: set[str],
) -> list[dict]:
    """Extract qualifier entries for one segment's text."""
    results: list[dict] = []

    # Find symptom mentions in this segment
    symptom_matches: list[tuple[re.Match, str]] = []
    for pat, kw in _SYMPTOM_PATS:
        for m in pat.finditer(text):
            if kw in known_symptoms:
                symptom_matches.append((m, kw))

    # Deduplicate by keyword (first match only)
    seen_symptoms: set[str] = set()
    unique_matches: list[tuple[re.Match, str]] = []
    for m, kw in symptom_matches:
        if kw not in seen_symptoms:
            seen_symptoms.add(kw)
            unique_matches.append((m, kw))

    def _nearest_symptom(q_match: re.Match) -> tuple[re.Match, str] | None:
        """Find the nearest symptom to a qualifier match."""
        best = None
        best_dist = float("inf")
        for sm, kw in unique_matches:
            if _within_token_window(text, sm, q_match):
                d = _char_distance(sm, q_match)
                if d < best_dist:
                    best_dist = d
                    best = (sm, kw)
        return best

    # Build qualifier dict per symptom
    qual_map: dict[str, dict] = {kw: {} for _, kw in unique_matches}
    match_map: dict[str, re.Match] = {kw: sm for sm, kw in unique_matches}

    # Single-value qualifiers: assign to nearest symptom
    _SINGLE_QUALS: list[tuple[re.Pattern, dict[str, str], str, bool]] = [
        (_SEVERITY_PAT, _SEVERITY_TERMS, "severity", True),
        (_ONSET_PAT, _ONSET_TERMS, "onset", True),
        (_CHARACTER_PAT, _CHARACTER_TERMS, "character", True),
        (_PATTERN_PAT, _PATTERN_TERMS, "pattern", True),
        (_PROGRESSION_PAT, _PROGRESSION_TERMS, "progression", True),
        (_LATERALITY_PAT, _LATERALITY_TERMS, "laterality", True),
    ]

    for pat, term_map, key, check_neg in _SINGLE_QUALS:
        for m in pat.finditer(text):
            if check_neg and _is_negated(text, m):
                continue
            nearest = _nearest_symptom(m)
            if nearest is None:
                continue
            _, kw = nearest
            if key not in qual_map[kw]:
                qual_map[kw][key] = term_map[m.group(1).lower()]

    # Radiation
    for m in _RADIATION_PAT.finditer(text):
        nearest = _nearest_symptom(m)
        if nearest is None:
            continue
        _, kw = nearest
        if "radiation" not in qual_map[kw]:
            qual_map[kw]["radiation"] = "to " + m.group(1).strip().rstrip(".,;!?")

    # Aggravating factors
    for m in _AGGRAVATING_PAT.finditer(text):
        nearest = _nearest_symptom(m)
        if nearest is None:
            continue
        _, kw = nearest
        value = m.group(1).strip().rstrip(".,;!?")
        value = _LEADING_PRONOUN_PAT.sub("", value).strip()
        if value:
            qual_map[kw].setdefault("aggravating_factors", [])
            if value.lower() not in {v.lower() for v in qual_map[kw]["aggravating_factors"]}:
                qual_map[kw]["aggravating_factors"].append(value)

    # Relieving factors
    for m in _RELIEVING_PAT.finditer(text):
        nearest = _nearest_symptom(m)
        if nearest is None:
            continue
        _, kw = nearest
        value = m.group(1).strip().rstrip(".,;!?")
        value = _LEADING_PRONOUN_PAT.sub("", value).strip()
        if value:
            qual_map[kw].setdefault("relieving_factors", [])
            if value.lower() not in {v.lower() for v in qual_map[kw]["relieving_factors"]}:
                qual_map[kw]["relieving_factors"].append(value)

    for _, symptom in unique_matches:
        if qual_map[symptom]:
            results.append({
                "symptom": symptom,
                "qualifiers": qual_map[symptom],
            })

    return results


# ── question detection ─────────────────────────────────────────────

def _is_question(text: str) -> bool:
    """Return True if the segment looks like a question (doctor asking)."""
    return text.rstrip().endswith("?")


# ── context-aware segment merging ─────────────────────────────────

_CONTEXT_WINDOW = 3  # look back this many segments for symptom context


def _build_context_segments(
    segments: list[dict],
    known: set[str],
) -> list[dict]:
    """Build context-enriched segments for cross-segment qualifier linking.

    For each segment that contains qualifier terms but no known symptom,
    prepend the most recently mentioned symptom from prior segments
    (within a window) as a synthetic prefix.  This lets the per-segment
    extractor link qualifiers to symptoms mentioned earlier.

    Returns a new list — original segments are not mutated.
    """
    # Track which symptom was last seen per segment index
    recent_symptom: str | None = None
    recent_symptom_idx: int = -1
    result: list[dict] = []

    for i, seg in enumerate(segments):
        text = seg.get("normalized_text", "")
        if not text:
            result.append(seg)
            continue

        text_lower = text.lower()

        # Check if this segment mentions a known symptom
        found_symptom: str | None = None
        for pat, kw in _SYMPTOM_PATS:
            if kw in known and pat.search(text):
                # Prefer multi-word symptoms over single-word
                if found_symptom is None or len(kw) > len(found_symptom):
                    found_symptom = kw

        if found_symptom is not None:
            recent_symptom = found_symptom
            recent_symptom_idx = i
            result.append(seg)
            continue

        # No symptom in this segment — check if it has qualifier terms
        has_qualifier = any(
            p.search(text)
            for p in (
                _SEVERITY_PAT, _ONSET_PAT, _CHARACTER_PAT,
                _PATTERN_PAT, _PROGRESSION_PAT, _LATERALITY_PAT,
                _RADIATION_PAT, _AGGRAVATING_PAT, _RELIEVING_PAT,
            )
        )

        if has_qualifier and recent_symptom is not None \
                and (i - recent_symptom_idx) <= _CONTEXT_WINDOW:
            # Prepend the recent symptom so the extractor can link
            enriched = dict(seg)
            enriched["normalized_text"] = recent_symptom + ". " + text
            result.append(enriched)
        else:
            result.append(seg)

    return result


# ── public API ─────────────────────────────────────────────────────

def extract_qualifiers(
    segments: list[dict],
    extracted_findings: list[str] | None = None,
) -> list[dict]:
    """Extract semantic qualifiers linked to symptoms from segments.

    Args:
        segments: list of normalized segment dicts (with ``normalized_text``).
        extracted_findings: optional list of known symptom strings from
            prior extraction.  If ``None``, symptoms are detected from
            the full text.

    Returns:
        list of ``{"symptom": str, "qualifiers": {…}}`` dicts.
        Only symptoms with at least one detected qualifier are included.
        Deduplicated by symptom — first segment wins.
    """
    if not segments:
        return []

    # Determine the known-symptom set
    if extracted_findings is not None:
        known: set[str] = {s.lower() for s in extracted_findings}
    else:
        # Fall back to detecting from full text
        from app.extractors import extract_symptoms
        full_text = " ".join(
            seg.get("normalized_text", "") for seg in segments
        ).strip()
        known = {s.lower() for s in extract_symptoms(full_text)}

    if not known:
        return []

    # Filter out question segments (e.g. doctor asking about qualifiers)
    # before context building so they don't pollute symptom tracking or
    # inflate segment distance
    non_question = [
        seg for seg in segments
        if not _is_question(seg.get("normalized_text", ""))
    ]

    # Build context-enriched segments for cross-segment linking
    enriched = _build_context_segments(non_question, known)

    # Collect qualifiers per segment, merging across segments per symptom
    merged: dict[str, dict] = {}  # symptom_key → qualifiers dict
    order: list[str] = []  # preserve first-seen order

    for seg in enriched:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        for entry in _extract_segment_qualifiers(text, known):
            symptom_key = entry["symptom"].lower()
            if symptom_key not in merged:
                merged[symptom_key] = {}
                order.append(symptom_key)

            existing = merged[symptom_key]
            for key, value in entry["qualifiers"].items():
                if key in ("aggravating_factors", "relieving_factors"):
                    existing.setdefault(key, [])
                    for v in value:
                        if v.lower() not in {e.lower() for e in existing[key]}:
                            existing[key].append(v)
                elif key not in existing:
                    # First-segment wins for scalar qualifiers
                    existing[key] = value

    # Consolidate sub-symptom qualifiers into multi-word parent symptoms.
    # E.g. if "abdominal pain" and "pain" both have qualifiers, merge
    # "pain"'s qualifiers into "abdominal pain" (since the patient is
    # talking about the same thing) and drop the standalone "pain" entry.
    consumed: set[str] = set()
    for parent_key in order:
        for child_key in order:
            if child_key == parent_key or child_key in consumed:
                continue
            # Check if child is a sub-word of parent (e.g. "pain" in "abdominal pain")
            if child_key in parent_key and len(parent_key) > len(child_key):
                # Merge child qualifiers into parent
                parent_quals = merged[parent_key]
                child_quals = merged[child_key]
                for key, value in child_quals.items():
                    if key in ("aggravating_factors", "relieving_factors"):
                        parent_quals.setdefault(key, [])
                        for v in value:
                            if v.lower() not in {e.lower() for e in parent_quals[key]}:
                                parent_quals[key].append(v)
                    elif key not in parent_quals:
                        parent_quals[key] = value
                consumed.add(child_key)

    results: list[dict] = []
    for symptom_key in order:
        if symptom_key in consumed:
            continue
        if merged[symptom_key]:
            # Find the original-cased symptom name from SYMPTOM_PATS
            symptom_name = symptom_key
            for _, kw in _SYMPTOM_PATS:
                if kw.lower() == symptom_key:
                    symptom_name = kw
                    break
            results.append({
                "symptom": symptom_name,
                "qualifiers": merged[symptom_key],
            })

    return results
