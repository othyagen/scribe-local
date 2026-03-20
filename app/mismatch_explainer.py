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
from app.clinical_terminology import (
    CLINICAL_TERMS,
    add_synonym,
    get_canonical_label,
    get_term,
)


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


# ── improvement suggestions ──────────────────────────────────────────


def suggest_improvements(summary: dict) -> list[dict]:
    """Generate actionable suggestions from a mismatch summary.

    Maps common failure patterns to concrete fixes.  Rule-based,
    deterministic, no LLM.

    Args:
        summary: dict from :func:`summarize_mismatches`.

    Returns:
        List of suggestion dicts, each with ``issue``,
        ``suggested_fix``, and ``affected_labels`` keys.
        Sorted by number of affected labels desc.
    """
    suggestions: list[dict] = []

    by_reason = summary.get("by_reason", {})
    top_missed = summary.get("top_missed_labels", [])
    top_synonym = summary.get("top_synonym_issues", [])

    # Not-detected labels → missing extraction rules or vocabulary.
    not_detected_count = by_reason.get("not_detected", 0)
    if not_detected_count > 0:
        # Collect labels whose primary failure is not_detected.
        # top_missed_labels includes all reasons; filter by checking
        # that the label is not primarily a synonym issue.
        synonym_set = {e["label"] for e in top_synonym}
        nd_labels = [
            e["label"] for e in top_missed
            if e["label"] not in synonym_set
        ]
        if nd_labels:
            suggestions.append({
                "issue": "not_detected",
                "suggested_fix": (
                    "Add missing terms to resources/extractors/symptoms.json "
                    "or add extraction rules — these labels were expected "
                    "but never extracted from transcript text"
                ),
                "affected_labels": nd_labels,
            })

    # Synonym/canonical mismatches → add synonyms to terminology.
    synonym_count = by_reason.get("synonym_mismatch", 0)
    canonical_count = by_reason.get("canonical_mismatch", 0)
    if synonym_count > 0 or canonical_count > 0:
        syn_labels = [e["label"] for e in top_synonym]
        if syn_labels:
            suggestions.append({
                "issue": "synonym_or_canonical_mismatch",
                "suggested_fix": (
                    "Add these labels as synonyms in "
                    "app/clinical_terminology.py CLINICAL_TERMS — "
                    "ground truth uses non-canonical forms that fail "
                    "to match after canonicalization"
                ),
                "affected_labels": syn_labels,
            })

    # Partial overlap → labels too broad or too narrow.
    partial_count = by_reason.get("partial_overlap", 0)
    if partial_count > 0:
        # Partial-overlap labels are in top_missed but not in synonym issues.
        synonym_set = {e["label"] for e in top_synonym}
        partial_labels = [
            e["label"] for e in top_missed
            if e["label"] not in synonym_set
        ]
        if partial_labels:
            suggestions.append({
                "issue": "partial_overlap",
                "suggested_fix": (
                    "Ground truth labels partially match detected terms — "
                    "use more specific canonical labels in case YAML "
                    "or add exact synonyms to clinical terminology"
                ),
                "affected_labels": partial_labels,
            })

    # Sort by number of affected labels desc.
    suggestions.sort(key=lambda s: -len(s["affected_labels"]))

    return suggestions


# ── apply suggestions ───────────────────────────────────────────────


def apply_suggestions(
    suggestions: list[dict],
    *,
    dry_run: bool = True,
) -> dict:
    """Apply safe, automatic updates from improvement suggestions.

    Currently the only safe auto-apply action is adding a synonym to
    :mod:`app.clinical_terminology` when a label matches an existing
    canonical term with high confidence.  Extractor and core reasoning
    changes are never applied automatically.

    Args:
        suggestions: list from :func:`suggest_improvements`.
        dry_run: if ``True`` (default), return proposed changes only.
            If ``False``, apply safe updates and populate
            ``applied_changes``.

    Returns:
        Dict with ``proposed_changes``, ``applied_changes``, and
        ``skipped_changes`` (with reasons).
    """
    proposed: list[dict] = []
    applied: list[dict] = []
    skipped: list[dict] = []

    for suggestion in suggestions:
        for label in suggestion.get("affected_labels", []):
            change = _classify_label(label)

            if change["action"] == "add_synonym":
                proposed.append(change)
                if not dry_run:
                    ok = add_synonym(change["canonical_target"], label)
                    if ok:
                        applied.append(change)
            else:
                skipped.append(change)

    return {
        "proposed_changes": proposed,
        "applied_changes": applied,
        "skipped_changes": skipped,
    }


def _classify_label(label: str) -> dict:
    """Classify a single label into an action category."""
    norm = label.strip().lower()

    # Already known — synonym or canonical term.
    canonical = get_canonical_label(label)
    if canonical != label.strip():
        return {
            "label": label,
            "action": "skip",
            "reason": "already_registered",
            "detail": f"'{label}' already maps to '{canonical}'",
        }

    if norm in CLINICAL_TERMS:
        return {
            "label": label,
            "action": "skip",
            "reason": "already_registered",
            "detail": f"'{label}' is a canonical term",
        }

    # Try to find a strong synonym target.
    target = _find_synonym_target(norm)
    if target is not None:
        return {
            "label": label,
            "action": "add_synonym",
            "canonical_target": target,
            "detail": f"Add '{label}' as synonym of '{target}'",
        }

    # Unknown label with no strong match — needs manual work.
    return {
        "label": label,
        "action": "skip",
        "reason": "no_safe_mapping",
        "detail": f"No confident canonical target for '{label}'",
    }


def _find_synonym_target(norm_label: str) -> str | None:
    """Find an existing canonical term that strongly matches *norm_label*.

    Matching rule — exactly ONE of these must hold for exactly ONE
    canonical term:

    * The label equals ``<modifier> <canonical>`` (e.g.
      "severe chest pain" where "chest pain" is canonical).
    * The label equals ``<canonical> <modifier>`` (e.g.
      "chest pain severe").

    The canonical term must appear as a complete word-boundary-aligned
    suffix or prefix of the label, not merely a substring anywhere.
    Short labels (< 4 chars) and short canonical terms (< 4 chars)
    are excluded to prevent spurious matches.

    Returns:
        The canonical term key if exactly one strong match is found,
        ``None`` otherwise (zero matches or ambiguous).
    """
    if len(norm_label) < 4:
        return None

    matches: list[str] = []
    for term_key in CLINICAL_TERMS:
        if len(term_key) < 4:
            continue
        if term_key == norm_label:
            # Exact match handled by caller (already_registered).
            continue
        if _is_modifier_match(norm_label, term_key):
            matches.append(term_key)

    if len(matches) == 1:
        return matches[0]
    return None


def _is_modifier_match(label: str, canonical: str) -> bool:
    """Return True if *label* is *canonical* with a modifier word prepended or appended.

    Requires the canonical part to be word-boundary-aligned and the
    modifier to be at least 2 characters.
    """
    # "<modifier> <canonical>"
    if label.endswith(" " + canonical):
        modifier = label[: len(label) - len(canonical) - 1]
        if len(modifier) >= 2 and " " not in modifier:
            return True

    # "<canonical> <modifier>"
    if label.startswith(canonical + " "):
        modifier = label[len(canonical) + 1:]
        if len(modifier) >= 2 and " " not in modifier:
            return True

    return False
