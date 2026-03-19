"""Adversarial case generation — stress-test variants for pipeline robustness.

Produces adversarially modified case variants designed to probe edge
cases and failure modes in the clinical reasoning pipeline.  For
evaluation and testing only — never affects clinical reasoning or
scoring logic.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

import copy
import re


# ── strategy registry ───────────────────────────────────────────────


_STRATEGY_REGISTRY: dict[str, callable] = {}


def _register(name: str):
    """Decorator to register an adversarial strategy."""
    def decorator(fn):
        _STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


def list_adversarial_strategies() -> list[str]:
    """Return the names of all supported adversarial strategies."""
    return sorted(_STRATEGY_REGISTRY.keys())


# ── public API ──────────────────────────────────────────────────────


def apply_adversarial(case: dict, strategy_name: str) -> dict:
    """Apply a named adversarial strategy to a case.

    Args:
        case: parsed case dict (not mutated).
        strategy_name: one of :func:`list_adversarial_strategies`.

    Returns:
        New case dict with adversarial modification applied.

    Raises:
        ValueError: if the strategy name is not supported.
    """
    if strategy_name not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown adversarial strategy: {strategy_name!r}. "
            f"Supported: {list_adversarial_strategies()}"
        )

    base = copy.deepcopy(case)
    fn = _STRATEGY_REGISTRY[strategy_name]
    result = fn(base)

    # Stamp metadata.
    original_id = case.get("case_id", "unknown")
    result["case_id"] = f"{original_id}__adv_{strategy_name}"

    meta = result.setdefault("meta", {})
    meta["base_case_id"] = original_id
    meta["applied_variation"] = f"adv_{strategy_name}"

    result.setdefault("adversarial", {})
    result["adversarial"]["strategy"] = strategy_name

    return result


def generate_adversarial_cases(case: dict) -> list[dict]:
    """Generate all adversarial variants of a case.

    Args:
        case: parsed case dict (not mutated).

    Returns:
        List of adversarially modified case dicts.
    """
    results: list[dict] = []
    for name in list_adversarial_strategies():
        results.append(apply_adversarial(case, name))
    return results


# ── segment helpers ─────────────────────────────────────────────────


def _next_seg_id(segments: list[dict]) -> str:
    """Generate the next sequential segment ID."""
    max_n = 0
    for seg in segments:
        m = re.search(r"(\d+)$", seg.get("seg_id", ""))
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


_CORE_SYMPTOMS = [
    "fever", "cough", "shortness of breath", "chest pain",
    "headache", "nausea", "abdominal pain", "dysuria",
    "vomiting", "diarrhea",
]


def _find_mentioned_symptoms(segments: list[dict]) -> list[str]:
    """Find core symptoms mentioned in segments."""
    full_text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).lower()
    return [s for s in _CORE_SYMPTOMS if s in full_text]


# ── strategy implementations ───────────────────────────────────────


@_register("noise_injection")
def _strat_noise_injection(case: dict) -> dict:
    """Add irrelevant low-specificity symptoms."""
    segments = case.get("segments") or []
    mentioned = set(_find_mentioned_symptoms(segments))

    noise_symptoms = [
        "Patient also reports mild fatigue.",
        "Occasional mild headache noted.",
        "Some general malaise reported.",
    ]

    # Only add symptoms not already in the case.
    for text in noise_symptoms:
        # Check if key word is already mentioned.
        key = text.split()[-2].lower().rstrip(".")  # e.g. "fatigue", "headache", "malaise"
        if key not in " ".join(seg.get("normalized_text", "").lower() for seg in segments):
            segments = _append_segment(segments, text)

    case["segments"] = segments
    return case


@_register("contradiction_injection")
def _strat_contradiction(case: dict) -> dict:
    """Add directly contradicting information."""
    segments = case.get("segments") or []
    mentioned = _find_mentioned_symptoms(segments)

    contradictions_added = 0
    if "fever" in mentioned:
        segments = _append_segment(segments, "Patient states no fever was measured at home.")
        contradictions_added += 1
    if "chest pain" in mentioned:
        segments = _append_segment(segments, "Patient later denies any chest discomfort.")
        contradictions_added += 1
    if "cough" in mentioned:
        segments = _append_segment(segments, "Reports the cough resolved yesterday.")
        contradictions_added += 1

    if contradictions_added == 0:
        # Generic contradiction.
        segments = _append_segment(
            segments,
            "Patient later states symptoms have completely resolved.",
        )

    case["segments"] = segments
    return case


@_register("negation_flip")
def _strat_negation_flip(case: dict) -> dict:
    """Flip 1-2 key symptoms into negated versions."""
    segments = case.get("segments") or []
    mentioned = _find_mentioned_symptoms(segments)

    # Negate up to 2 found symptoms.
    to_negate = mentioned[:2]
    if not to_negate:
        case["segments"] = list(segments)
        meta = case.setdefault("meta", {})
        meta["variation_warning"] = "No symptoms found to negate"
        return case

    for symptom in to_negate:
        segments = _append_segment(segments, f"Denies {symptom}.")

    case["segments"] = segments
    return case


@_register("temporal_confusion")
def _strat_temporal_confusion(case: dict) -> dict:
    """Introduce conflicting temporal information."""
    segments = case.get("segments") or []

    conflicting_times = [
        "Symptoms started just this morning.",
        "Reports this has been going on for about 2 weeks.",
    ]

    for text in conflicting_times:
        segments = _append_segment(segments, text)

    case["segments"] = segments
    return case


@_register("symptom_dilution")
def _strat_symptom_dilution(case: dict) -> dict:
    """Add many low-specificity symptoms to dilute signal."""
    segments = case.get("segments") or []

    dilution_symptoms = [
        "Patient also feels tired.",
        "Some dizziness when standing up.",
        "Mild generalized weakness.",
        "Occasional lightheadedness.",
        "Reports poor appetite.",
        "Mild body aches.",
    ]

    for text in dilution_symptoms:
        segments = _append_segment(segments, text)

    case["segments"] = segments
    return case


@_register("incomplete_case")
def _strat_incomplete_case(case: dict) -> dict:
    """Remove one important segment."""
    segments = case.get("segments") or []

    if len(segments) <= 1:
        meta = case.setdefault("meta", {})
        meta["variation_warning"] = "Too few segments to remove"
        case["segments"] = list(segments)
        return case

    # Remove the first non-negation segment after the first one.
    remove_idx = None
    for i in range(1, len(segments)):
        text = segments[i].get("normalized_text", "").lower()
        if not text.startswith(("denies", "no ")):
            remove_idx = i
            break

    if remove_idx is not None:
        case["segments"] = [s for j, s in enumerate(segments) if j != remove_idx]
    else:
        # Remove the last segment as fallback.
        case["segments"] = segments[:-1]

    return case
