"""Layer 3: Structured symptom model — domain-grouped per-symptom structures.

Conservative linking throughout — prefers null/empty over speculative
filling.  Only populates structured fields when there is local,
traceable evidence.

Pure function — no ML, no LLM, no I/O.
"""

from __future__ import annotations

import re

from app.extractors import _SYMPTOM_PATTERNS


# ── prior episode phrases ─────────────────────────────────────────

_PRIOR_EPISODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bthis has happened before\b",
        r"\bhad this before\b",
        r"\bsimilar episode\b",
        r"\bhappened before\b",
        r"\bhad this last (?:year|month|week)\b",
        r"\brecurring\b",
        r"\bprevious episode\b",
    ]
]


def build_structured_symptoms(clinical_state: dict) -> list[dict]:
    """Build domain-grouped per-symptom structures from clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        list of structured symptom dicts, one per symptom, preserving
        symptom list order.
    """
    symptoms: list[str] = clinical_state.get("symptoms", [])
    if not symptoms:
        return []

    # Gather inputs
    qualifiers: list[dict] = clinical_state.get("qualifiers", [])
    timeline: list[dict] = clinical_state.get("timeline", [])
    observations: list[dict] = clinical_state.get("observations", [])
    sites: list[dict] = clinical_state.get("sites", [])
    intensities: list[dict] = clinical_state.get("intensities", [])
    ice: dict = clinical_state.get("ice", {})
    derived = clinical_state.get("derived", {})
    red_flags: list[dict] = derived.get("red_flags", [])
    segments: list[dict] = _gather_segments_from_observations(observations)

    # Build indexes
    qual_index = _index_qualifiers(qualifiers)
    time_index = _index_timeline(timeline)
    obs_by_seg = _index_observations_by_seg(observations)
    site_by_seg = _index_by_seg(sites, "site")
    intensity_by_seg = _index_by_seg(intensities, "value")
    ice_by_seg = _index_ice_by_seg(ice)
    symptom_seg_ids = _symptom_to_seg_ids(observations)
    red_flag_index = _index_red_flags(red_flags)

    # Segment text index for proximity-based site linking
    seg_texts = _index_segment_texts(observations)

    result: list[dict] = []
    for symptom in symptoms:
        key = symptom.lower()
        quals = qual_index.get(key, {})
        seg_ids = symptom_seg_ids.get(key, set())

        # Spatial
        site = _link_site(symptom, seg_ids, site_by_seg, seg_texts)
        laterality = quals.get("laterality")
        radiation = quals.get("radiation")

        # Qualitative
        intensity_val, intensity_raw = _link_intensity(
            symptom, seg_ids, intensity_by_seg, seg_texts,
        )
        severity = quals.get("severity")
        character = quals.get("character")

        # Temporal
        onset = quals.get("onset")
        duration = time_index.get(key)
        pattern = quals.get("pattern")
        progression = quals.get("progression")

        # Modifiers
        aggravating = list(quals.get("aggravating_factors", []))
        relieving = list(quals.get("relieving_factors", []))

        # Context — same-segment only
        associated_present = _same_seg_symptoms(
            symptom, seg_ids, obs_by_seg,
        )
        associated_absent = _same_seg_negations(
            symptom, seg_ids, obs_by_seg,
        )
        prior_episodes = _detect_prior_episodes(symptom, seg_ids, seg_texts)

        # Safety
        red_flags_present = red_flag_index.get(key, [])

        # Patient perspective — same-segment ICE
        ideas, concerns, expectations = _same_seg_ice(
            seg_ids, ice_by_seg,
        )

        # Observation IDs
        obs_ids = _collect_observation_ids(key, observations)

        result.append({
            "symptom": symptom,
            "spatial": {
                "site": site,
                "laterality": laterality,
                "radiation": radiation,
            },
            "qualitative": {
                "character": character,
                "intensity": intensity_val,
                "intensity_raw": intensity_raw,
                "severity": severity,
            },
            "temporal": {
                "onset": onset,
                "onset_type": None,  # DEFERRED
                "duration": duration,
                "pattern": pattern,
                "progression": progression,
            },
            "modifiers": {
                "aggravating_factors": aggravating,
                "relieving_factors": relieving,
            },
            "context": {
                "associated_present": associated_present,
                "associated_absent": associated_absent,
                "prior_episodes": prior_episodes,
            },
            "safety": {
                "red_flags_present": red_flags_present,
                "red_flags_absent": [],  # V1: always empty
            },
            "patient_perspective": {
                "ideas": ideas,
                "concerns": concerns,
                "expectations": expectations,
            },
            "observation_ids": obs_ids,
        })

    return result


# ── index builders ────────────────────────────────────────────────


def _index_qualifiers(qualifiers: list[dict]) -> dict[str, dict]:
    """Index qualifiers by symptom (case-insensitive, first wins)."""
    index: dict[str, dict] = {}
    for entry in qualifiers:
        key = entry.get("symptom", "").lower()
        if key and key not in index:
            index[key] = entry.get("qualifiers", {})
    return index


def _index_timeline(timeline: list[dict]) -> dict[str, str]:
    """Index timeline time_expression by symptom (first match)."""
    index: dict[str, str] = {}
    for entry in timeline:
        sym = entry.get("symptom", "").lower()
        expr = entry.get("time_expression")
        if sym and expr and sym not in index:
            index[sym] = expr
    return index


def _index_observations_by_seg(
    observations: list[dict],
) -> dict[str | None, list[dict]]:
    """Group observations by seg_id."""
    index: dict[str | None, list[dict]] = {}
    for obs in observations:
        seg_id = obs.get("seg_id")
        index.setdefault(seg_id, []).append(obs)
    return index


def _index_by_seg(
    items: list[dict],
    value_key: str,
) -> dict[str | None, list[dict]]:
    """Group items by seg_id."""
    index: dict[str | None, list[dict]] = {}
    for item in items:
        seg_id = item.get("seg_id")
        index.setdefault(seg_id, []).append(item)
    return index


def _index_ice_by_seg(ice: dict) -> dict[str | None, dict[str, list[str]]]:
    """Group ICE items by seg_id."""
    index: dict[str | None, dict[str, list[str]]] = {}
    for category in ("ideas", "concerns", "expectations"):
        for item in ice.get(category, []):
            seg_id = item.get("seg_id")
            index.setdefault(seg_id, {"ideas": [], "concerns": [], "expectations": []})
            index[seg_id][category].append(item.get("text", ""))
    return index


def _symptom_to_seg_ids(observations: list[dict]) -> dict[str, set[str]]:
    """Map symptom (lowercase) to set of seg_ids from observations."""
    index: dict[str, set[str]] = {}
    for obs in observations:
        if obs.get("finding_type") == "symptom":
            key = obs.get("value", "").lower()
            seg_id = obs.get("seg_id")
            if key and seg_id:
                index.setdefault(key, set()).add(seg_id)
    return index


def _index_red_flags(red_flags: list[dict]) -> dict[str, list[str]]:
    """Map symptom (lowercase) to red flag labels from evidence."""
    index: dict[str, list[str]] = {}
    for flag in red_flags:
        label = flag.get("label", "")
        for evidence_item in flag.get("evidence", []):
            key = evidence_item.lower()
            # Only index actual symptom names (not qualifier descriptions)
            if ":" not in evidence_item:
                index.setdefault(key, [])
                if label not in index[key]:
                    index[key].append(label)
    return index


def _index_segment_texts(
    observations: list[dict],
) -> dict[str | None, str]:
    """Map seg_id to source_text from first observation in that segment."""
    index: dict[str | None, str] = {}
    for obs in observations:
        seg_id = obs.get("seg_id")
        if seg_id and seg_id not in index:
            index[seg_id] = obs.get("source_text", "")
    return index


def _gather_segments_from_observations(
    observations: list[dict],
) -> list[dict]:
    """Reconstruct minimal segment list from observations."""
    seen: set[str] = set()
    segments: list[dict] = []
    for obs in observations:
        seg_id = obs.get("seg_id")
        if seg_id and seg_id not in seen:
            seen.add(seg_id)
            segments.append({
                "seg_id": seg_id,
                "normalized_text": obs.get("source_text", ""),
            })
    return segments


# ── conservative linking ──────────────────────────────────────────


def _link_site(
    symptom: str,
    seg_ids: set[str],
    site_by_seg: dict[str | None, list[dict]],
    seg_texts: dict[str | None, str],
) -> str | None:
    """Link a site to a symptom only when in same segment.

    If multiple symptoms in one segment, link site to syntactically
    closest symptom (by character position).
    """
    best_site: str | None = None
    best_distance = float("inf")

    for seg_id in seg_ids:
        sites_in_seg = site_by_seg.get(seg_id, [])
        if not sites_in_seg:
            continue

        text = seg_texts.get(seg_id, "")
        if not text:
            continue

        # Find symptom position in text
        sym_pos = _find_position(text, symptom)
        if sym_pos < 0:
            continue

        for site_entry in sites_in_seg:
            site_name = site_entry.get("site", "")
            site_pos = _find_position(text, site_name)
            if site_pos < 0:
                continue
            dist = abs(sym_pos - site_pos)
            if dist < best_distance:
                best_distance = dist
                best_site = site_name

    return best_site


def _link_intensity(
    symptom: str,
    seg_ids: set[str],
    intensity_by_seg: dict[str | None, list[dict]],
    seg_texts: dict[str | None, str],
) -> tuple[int | None, str | None]:
    """Link intensity to symptom only when in same segment."""
    for seg_id in seg_ids:
        intensities_in_seg = intensity_by_seg.get(seg_id, [])
        if intensities_in_seg:
            entry = intensities_in_seg[0]
            return entry.get("value"), entry.get("raw_text")
    return None, None


def _find_position(text: str, term: str) -> int:
    """Find case-insensitive position of term in text."""
    return text.lower().find(term.lower())


def _same_seg_symptoms(
    symptom: str,
    seg_ids: set[str],
    obs_by_seg: dict[str | None, list[dict]],
) -> list[str]:
    """Find other symptoms that co-occur in the same segments."""
    key = symptom.lower()
    associated: list[str] = []
    seen: set[str] = set()

    for seg_id in seg_ids:
        for obs in obs_by_seg.get(seg_id, []):
            if obs.get("finding_type") != "symptom":
                continue
            other = obs.get("value", "")
            other_key = other.lower()
            if other_key != key and other_key not in seen:
                seen.add(other_key)
                associated.append(other)

    return associated


def _same_seg_negations(
    symptom: str,
    seg_ids: set[str],
    obs_by_seg: dict[str | None, list[dict]],
) -> list[str]:
    """Find negations that co-occur in the same segments as this symptom."""
    negations: list[str] = []
    seen: set[str] = set()

    for seg_id in seg_ids:
        for obs in obs_by_seg.get(seg_id, []):
            if obs.get("finding_type") != "negation":
                continue
            value = obs.get("value", "")
            key = value.lower()
            if key not in seen:
                seen.add(key)
                negations.append(value)

    return negations


def _detect_prior_episodes(
    symptom: str,
    seg_ids: set[str],
    seg_texts: dict[str | None, str],
) -> list[str]:
    """Detect prior episode mentions in same segments as symptom."""
    episodes: list[str] = []
    seen: set[str] = set()

    for seg_id in seg_ids:
        text = seg_texts.get(seg_id, "")
        if not text:
            continue

        for pattern in _PRIOR_EPISODE_PATTERNS:
            m = pattern.search(text)
            if m:
                phrase = m.group(0)
                key = phrase.lower()
                if key not in seen:
                    seen.add(key)
                    episodes.append(phrase)

    return episodes


def _same_seg_ice(
    seg_ids: set[str],
    ice_by_seg: dict[str | None, dict[str, list[str]]],
) -> tuple[list[str], list[str], list[str]]:
    """Collect ICE items from segments containing this symptom."""
    ideas: list[str] = []
    concerns: list[str] = []
    expectations: list[str] = []
    seen_i: set[str] = set()
    seen_c: set[str] = set()
    seen_e: set[str] = set()

    for seg_id in seg_ids:
        ice_in_seg = ice_by_seg.get(seg_id)
        if not ice_in_seg:
            continue
        for text in ice_in_seg.get("ideas", []):
            if text.lower() not in seen_i:
                seen_i.add(text.lower())
                ideas.append(text)
        for text in ice_in_seg.get("concerns", []):
            if text.lower() not in seen_c:
                seen_c.add(text.lower())
                concerns.append(text)
        for text in ice_in_seg.get("expectations", []):
            if text.lower() not in seen_e:
                seen_e.add(text.lower())
                expectations.append(text)

    return ideas, concerns, expectations


def _collect_observation_ids(
    symptom_key: str,
    observations: list[dict],
) -> list[str]:
    """Collect observation IDs for this symptom."""
    return [
        obs["observation_id"]
        for obs in observations
        if obs.get("finding_type") == "symptom"
        and obs.get("value", "").lower() == symptom_key
    ]
