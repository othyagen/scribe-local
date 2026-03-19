"""Case variation generator — deterministic case transforms for replay.

Produces clinically meaningful variants of existing cases by applying
lightweight, text-level transformations.  Each variation returns a new
valid case dict with updated ``case_id`` and variation metadata.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

import copy
import re


# ── variation registry ──────────────────────────────────────────────


_VARIATION_REGISTRY: dict[str, callable] = {}


def _register(name: str):
    """Decorator to register a variation function."""
    def decorator(fn):
        _VARIATION_REGISTRY[name] = fn
        return fn
    return decorator


def list_supported_variations() -> list[str]:
    """Return the names of all supported variation types."""
    return sorted(_VARIATION_REGISTRY.keys())


# ── public API ──────────────────────────────────────────────────────


def apply_variation(case: dict, variation_name: str) -> dict:
    """Apply a named variation to a case.

    Args:
        case: parsed case dict (not mutated).
        variation_name: one of :func:`list_supported_variations`.

    Returns:
        New case dict with variation applied, updated ``case_id``,
        and variation metadata in ``meta``.

    Raises:
        ValueError: if the variation name is not supported.
    """
    if variation_name not in _VARIATION_REGISTRY:
        raise ValueError(
            f"Unknown variation: {variation_name!r}. "
            f"Supported: {list_supported_variations()}"
        )

    base = copy.deepcopy(case)
    fn = _VARIATION_REGISTRY[variation_name]
    result = fn(base)

    # Stamp metadata.
    original_id = case.get("case_id", "unknown")
    result["case_id"] = f"{original_id}__{variation_name}"

    meta = result.setdefault("meta", {})
    meta["base_case_id"] = original_id
    meta["applied_variation"] = variation_name

    return result


def generate_case_variations(case: dict) -> list[dict]:
    """Generate all supported variations of a case.

    Args:
        case: parsed case dict (not mutated).

    Returns:
        List of new case dicts, one per supported variation.
    """
    results: list[dict] = []
    for name in list_supported_variations():
        results.append(apply_variation(case, name))
    return results


def summarize_case_variation(case: dict) -> dict:
    """Summarize a case variation's key properties.

    Args:
        case: a case dict (original or variation).

    Returns:
        Summary dict with case_id, base_case_id, applied_variation,
        segment_count, and tags.
    """
    meta = case.get("meta") or {}
    return {
        "case_id": case.get("case_id", ""),
        "base_case_id": meta.get("base_case_id", ""),
        "applied_variation": meta.get("applied_variation", ""),
        "segment_count": len(case.get("segments") or []),
        "tags": list(meta.get("tags") or []),
    }


# ── text helpers ────────────────────────────────────────────────────


_FEVER_PATTERN = re.compile(
    r"\b(?:fever|febrile|fevers)\b",
    re.IGNORECASE,
)

_DURATION_SHORT = re.compile(
    r"\b(\d+)\s*(days?|hours?|hrs?)\b",
    re.IGNORECASE,
)

_AGE_PATTERN = re.compile(
    r"\b(\d+)[- ]year[- ]old\b",
    re.IGNORECASE,
)

# Common core symptoms likely to appear.
_CORE_SYMPTOMS = [
    "fever", "cough", "shortness of breath", "chest pain",
    "headache", "nausea", "abdominal pain", "dysuria",
    "vomiting", "diarrhea",
]


def _remove_fever_from_text(text: str) -> str:
    """Remove fever mentions from a text string."""
    # Remove "and fever", "fever and", ", fever,", standalone "fever"
    result = re.sub(r",?\s*\band\b\s+fever\b", "", text, flags=re.IGNORECASE)
    result = re.sub(r"\bfever\b\s+\band\b\s*,?", "", result, flags=re.IGNORECASE)
    result = re.sub(r",?\s*\bfever\b\s*,?", " ", result, flags=re.IGNORECASE)
    # Clean up double spaces and leading/trailing whitespace.
    result = re.sub(r"\s{2,}", " ", result).strip()
    # Fix trailing period if missing.
    if result and result[-1] not in ".!?":
        result += "."
    return result


def _replace_short_duration(text: str, replacement: str = "2 weeks") -> tuple[str, bool]:
    """Replace short duration phrases with a longer one."""
    found = False

    def _replacer(m):
        nonlocal found
        found = True
        return replacement

    result = _DURATION_SHORT.sub(_replacer, text)
    return result, found


def _find_core_symptom(segments: list[dict]) -> str | None:
    """Find the first core symptom mentioned in segments."""
    full_text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).lower()
    for symptom in _CORE_SYMPTOMS:
        if symptom in full_text:
            return symptom
    return None


def _next_seg_id(segments: list[dict]) -> str:
    """Generate the next sequential segment ID."""
    max_n = 0
    for seg in segments:
        sid = seg.get("seg_id", "")
        m = re.search(r"(\d+)$", sid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"seg_{max_n + 1:04d}"


def _next_t(segments: list[dict]) -> float:
    """Get the end time of the last segment."""
    if not segments:
        return 0.0
    return max(seg.get("t1", 0.0) for seg in segments)


def _append_segment(segments: list[dict], text: str) -> list[dict]:
    """Append a new segment with the given text."""
    t_start = _next_t(segments)
    new_seg = {
        "seg_id": _next_seg_id(segments),
        "t0": t_start,
        "t1": t_start + 3.0,
        "speaker_id": "spk_0",
        "normalized_text": text,
    }
    return segments + [new_seg]


def _add_warning(case: dict, warning: str) -> None:
    """Add a variation warning to meta."""
    meta = case.setdefault("meta", {})
    meta["variation_warning"] = warning


# ── variation implementations ───────────────────────────────────────


@_register("remove_fever")
def _var_remove_fever(case: dict) -> dict:
    """Remove fever mentions from segments and ground truth."""
    segments = case.get("segments") or []
    modified = False

    new_segments: list[dict] = []
    for seg in segments:
        text = seg.get("normalized_text", "")
        if _FEVER_PATTERN.search(text):
            new_text = _remove_fever_from_text(text)
            if new_text != text:
                modified = True
                seg = dict(seg)
                seg["normalized_text"] = new_text
        new_segments.append(seg)

    case["segments"] = new_segments

    # Remove fever from ground truth key findings.
    gt = case.get("ground_truth")
    if isinstance(gt, dict):
        findings = gt.get("key_findings")
        if isinstance(findings, list):
            gt["key_findings"] = [
                f for f in findings if f.lower() != "fever"
            ]

    if not modified:
        _add_warning(case, "No fever mentions found in segments")

    return case


@_register("add_duration_longer")
def _var_add_duration_longer(case: dict) -> dict:
    """Replace short durations with longer durations."""
    segments = case.get("segments") or []
    any_replaced = False

    new_segments: list[dict] = []
    for seg in segments:
        text = seg.get("normalized_text", "")
        new_text, replaced = _replace_short_duration(text)
        if replaced:
            any_replaced = True
            seg = dict(seg)
            seg["normalized_text"] = new_text
        new_segments.append(seg)

    if not any_replaced:
        # Append a duration segment.
        new_segments = _append_segment(
            new_segments,
            "Symptoms have been present for approximately 2 weeks.",
        )

    case["segments"] = new_segments

    # Update answer script durations if present.
    script = case.get("answer_script")
    if isinstance(script, list):
        new_script: list[dict] = []
        for entry in script:
            entry = dict(entry)
            if entry.get("question_type") == "duration":
                entry["value"] = "2 weeks"
            new_script.append(entry)
        case["answer_script"] = new_script

    return case


@_register("add_elderly_context")
def _var_add_elderly_context(case: dict) -> dict:
    """Add elderly patient context."""
    segments = case.get("segments") or []

    # Check if an age is already mentioned and replace it.
    age_found = False
    new_segments: list[dict] = []
    for seg in segments:
        text = seg.get("normalized_text", "")
        if _AGE_PATTERN.search(text) and not age_found:
            age_found = True
            new_text = _AGE_PATTERN.sub("82-year-old", text)
            seg = dict(seg)
            seg["normalized_text"] = new_text
        new_segments.append(seg)

    if not age_found:
        # Prepend an age/context segment.
        t_start = 0.0
        # Shift existing segments.
        shifted: list[dict] = []
        for seg in new_segments:
            seg = dict(seg)
            seg["t0"] = seg.get("t0", 0.0) + 3.0
            seg["t1"] = seg.get("t1", 0.0) + 3.0
            shifted.append(seg)
        context_seg = {
            "seg_id": "seg_0000",
            "t0": t_start,
            "t1": t_start + 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "82-year-old patient with multiple comorbidities.",
        }
        new_segments = [context_seg] + shifted

    case["segments"] = new_segments

    # Add elderly tag.
    meta = case.setdefault("meta", {})
    tags = list(meta.get("tags") or [])
    if "elderly" not in tags:
        tags.append("elderly")
    meta["tags"] = tags

    return case


@_register("add_negation_of_core_symptom")
def _var_add_negation(case: dict) -> dict:
    """Add a segment negating a core symptom."""
    segments = case.get("segments") or []
    symptom = _find_core_symptom(segments)

    if symptom is None:
        _add_warning(case, "No core symptom found to negate")
        case["segments"] = list(segments)
        return case

    # Pick a symptom to negate that is NOT the primary one.
    # Find all mentioned symptoms and negate the second one,
    # or pick a plausible related symptom to negate.
    full_text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).lower()

    mentioned = [s for s in _CORE_SYMPTOMS if s in full_text]
    # Pick the last mentioned symptom to negate, or a common one.
    if len(mentioned) >= 2:
        to_negate = mentioned[-1]
    else:
        # Negate a plausible related symptom not already mentioned.
        not_mentioned = [s for s in _CORE_SYMPTOMS if s not in full_text]
        to_negate = not_mentioned[0] if not_mentioned else "fever"

    negation_text = f"Denies {to_negate}."
    case["segments"] = _append_segment(segments, negation_text)

    return case


@_register("add_missing_information")
def _var_add_missing_info(case: dict) -> dict:
    """Remove one useful detail from segments."""
    segments = case.get("segments") or []

    if len(segments) <= 1:
        _add_warning(case, "Too few segments to remove information")
        case["segments"] = list(segments)
        return case

    # Remove the last non-negation segment (one that doesn't start
    # with "Denies" or "No ").
    removable_idx = None
    for i in range(len(segments) - 1, 0, -1):
        text = segments[i].get("normalized_text", "")
        if not text.lower().startswith(("denies", "no ")):
            removable_idx = i
            break

    if removable_idx is not None:
        new_segments = [s for j, s in enumerate(segments) if j != removable_idx]
        case["segments"] = new_segments
    else:
        _add_warning(case, "No suitable segment found to remove")
        case["segments"] = list(segments)

    return case


@_register("add_conflicting_information")
def _var_add_conflicting(case: dict) -> dict:
    """Append a segment introducing a mild contradiction."""
    segments = case.get("segments") or []
    full_text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).lower()

    # Find something to contradict.
    contradiction = None
    if "fever" in full_text:
        contradiction = "No measured fever at home."
    elif "pain" in full_text:
        contradiction = "Reports the pain comes and goes, sometimes feels fine."
    elif "cough" in full_text:
        contradiction = "Earlier today the cough seemed to have resolved."
    elif "nausea" in full_text:
        contradiction = "Ate a full meal without any nausea."
    else:
        contradiction = "Patient later reports feeling mostly fine."

    case["segments"] = _append_segment(segments, contradiction)

    return case
