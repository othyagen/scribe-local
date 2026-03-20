"""Mismatch explanation and aggregation for evaluation debugging.

Produces structured reasons for why expected ground-truth labels
did not match detected outputs.  Compares raw, canonicalized, and
normalized forms to pinpoint the failure mode.

:func:`summarize_mismatches` aggregates across multiple cases to
surface the most frequent failure patterns.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

from app.canonicalization import canonicalize_label
from app.clinical_terminology import get_canonical_label, get_term


# ── public API ───────────────────────────────────────────────────────


def explain_mismatches(result_bundle: dict, score: dict) -> list[dict]:
    """Explain all mismatches between ground truth and detected outputs.

    Args:
        result_bundle: result from :func:`run_case` or :func:`run_case_script`.
        score: score dict from :func:`score_result_against_ground_truth`.

    Returns:
        List of mismatch dicts, each with ``field``, ``label``,
        ``canonical``, ``reason``, and ``detail`` keys.
        Empty list if everything matches.
    """
    gt_raw = result_bundle.get("ground_truth") or {}
    state = result_bundle.get("session", {}).get("clinical_state", {})

    mismatches: list[dict] = []

    mismatches.extend(_explain_hypothesis_mismatches(gt_raw, state, score))
    mismatches.extend(_explain_key_finding_mismatches(gt_raw, state, score))
    mismatches.extend(_explain_red_flag_mismatches(gt_raw, state, score))

    return mismatches


# ── internal helpers ─────────────────────────────────────────────────


def _normalize(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""


def _explain_hypothesis_mismatches(
    gt_raw: dict,
    state: dict,
    score: dict,
) -> list[dict]:
    expected_raw = gt_raw.get("expected_hypotheses") or []
    if not expected_raw:
        return []

    missing = score.get("hypotheses", {}).get("missing", [])
    if not missing:
        return []

    actual_hyps = state.get("hypotheses", [])
    actual_titles = [_normalize(h.get("title", "")) for h in actual_hyps]
    actual_set = set(actual_titles)

    results: list[dict] = []
    for label in missing:
        canonical = get_canonical_label(label)
        norm = _normalize(canonical)
        reason, detail = _diagnose_missing(
            label, canonical, norm, actual_set, actual_titles, "hypothesis",
        )
        results.append({
            "field": "expected_hypotheses",
            "label": label,
            "canonical": canonical,
            "reason": reason,
            "detail": detail,
        })
    return results


def _explain_key_finding_mismatches(
    gt_raw: dict,
    state: dict,
    score: dict,
) -> list[dict]:
    expected_raw = gt_raw.get("key_findings") or []
    if not expected_raw:
        return []

    missing = score.get("key_findings", {}).get("missing", [])
    if not missing:
        return []

    actual_symptoms = {_normalize(s) for s in state.get("symptoms", [])}
    actual_list = sorted(actual_symptoms)

    results: list[dict] = []
    for label in missing:
        canonical = get_canonical_label(label)
        norm = _normalize(canonical)
        reason, detail = _diagnose_missing(
            label, canonical, norm, actual_symptoms, actual_list, "key_finding",
        )
        results.append({
            "field": "key_findings",
            "label": label,
            "canonical": canonical,
            "reason": reason,
            "detail": detail,
        })
    return results


def _explain_red_flag_mismatches(
    gt_raw: dict,
    state: dict,
    score: dict,
) -> list[dict]:
    expected_raw = gt_raw.get("red_flags") or []
    if not expected_raw:
        return []

    missing = score.get("red_flags", {}).get("missing", [])
    if not missing:
        return []

    derived = state.get("derived", {})
    actual_flags = derived.get("red_flags", [])
    actual_labels = {_normalize(rf.get("label", "")) for rf in actual_flags}
    actual_list = sorted(actual_labels)

    results: list[dict] = []
    for label in missing:
        canonical = get_canonical_label(label)
        norm = _normalize(canonical)
        reason, detail = _diagnose_missing(
            label, canonical, norm, actual_labels, actual_list, "red_flag",
        )
        results.append({
            "field": "red_flags",
            "label": label,
            "canonical": canonical,
            "reason": reason,
            "detail": detail,
        })
    return results


def _diagnose_missing(
    raw_label: str,
    canonical: str,
    normalized: str,
    actual_set: set[str],
    actual_list: list[str],
    field_type: str,
) -> tuple[str, str]:
    """Determine why a label is missing and return (reason, detail)."""

    # Check if the raw label itself would match (synonym mismatch).
    raw_norm = _normalize(raw_label)
    if raw_norm != normalized and raw_norm in actual_set:
        return (
            "synonym_mismatch",
            f"raw '{raw_label}' matches but canonical '{canonical}' does not",
        )

    # Check if canonical differs from raw but neither matches.
    if raw_norm != normalized:
        return (
            "canonical_mismatch",
            f"'{raw_label}' canonicalized to '{canonical}' — "
            f"neither found in detected: {actual_list}",
        )

    # Check for partial overlap (substring match).
    partial = [a for a in actual_list if normalized in a or a in normalized]
    if partial:
        return (
            "partial_overlap",
            f"'{raw_label}' partially overlaps with detected: {partial}",
        )

    # Not detected at all.
    if not actual_list:
        detail = f"no {field_type}s detected"
    else:
        detail = f"'{raw_label}' not found in detected: {actual_list}"
    return ("not_detected", detail)


# ── aggregation ──────────────────────────────────────────────────────


def summarize_mismatches(
    all_mismatches: list[list[dict]],
    top_n: int = 5,
) -> dict:
    """Aggregate mismatch explanations across multiple cases.

    Args:
        all_mismatches: list of per-case mismatch lists (each from
            :func:`explain_mismatches`).
        top_n: number of top items to include in ranked lists.

    Returns:
        Summary dict with counts and ranked lists.
    """
    reason_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    synonym_labels: dict[str, int] = {}
    total = 0

    for case_mismatches in all_mismatches:
        for entry in case_mismatches:
            total += 1
            reason = entry.get("reason", "unknown")
            label = entry.get("label", "")
            field = entry.get("field", "")

            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
            field_counts[field] = field_counts.get(field, 0) + 1

            if reason in ("synonym_mismatch", "canonical_mismatch"):
                synonym_labels[label] = synonym_labels.get(label, 0) + 1

    return {
        "total_mismatches": total,
        "cases_with_mismatches": sum(1 for m in all_mismatches if m),
        "cases_total": len(all_mismatches),
        "by_reason": reason_counts,
        "by_field": field_counts,
        "top_missed_labels": _top_n(label_counts, top_n),
        "top_synonym_issues": _top_n(synonym_labels, top_n),
        "top_reasons": _top_n(reason_counts, top_n),
    }


def _top_n(counts: dict[str, int], n: int) -> list[dict]:
    """Return top N entries sorted by count desc, then alphabetically."""
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [{"label": k, "count": v} for k, v in sorted_items[:n]]
