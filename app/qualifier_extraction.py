"""Semantic qualifier extraction — deterministic clinical qualifier detection.

Extracts clinical qualifiers (severity, onset, pattern, progression,
laterality, radiation, aggravating/relieving factors) from transcript
segments and links them to nearby symptom mentions.

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
    r"exacerbated\s+by|increases?\s+with|provoked\s+by)\s+"
    r"([a-z]+(?:\s+(?!" + _FACTOR_STOP + r")[a-z]+){0,2})",
    re.IGNORECASE,
)

_RELIEVING_PAT = re.compile(
    r"\b(?:relieved\s+by|better\s+with|improved\s+by|eased\s+by|"
    r"helped\s+by|decreases?\s+with|alleviated\s+by)\s+"
    r"([a-z]+(?:\s+(?!" + _FACTOR_STOP + r")[a-z]+){0,2})",
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

_TOKEN_WINDOW = 8


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

    # Collect qualifiers per segment, first occurrence wins
    seen: set[str] = set()
    results: list[dict] = []

    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue

        for entry in _extract_segment_qualifiers(text, known):
            symptom_key = entry["symptom"].lower()
            if symptom_key not in seen:
                seen.add(symptom_key)
                results.append(entry)

    return results
