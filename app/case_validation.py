"""Case ground-truth validation against clinical terminology.

Catches invalid, incomplete, or inconsistent ground truth early —
before evaluation runs produce silent scoring failures.

Uses :mod:`app.clinical_terminology` as the source of truth for
canonical labels and red flag status.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

from app.clinical_terminology import get_canonical_label, is_red_flag, get_term


# ── public API ───────────────────────────────────────────────────────


def validate_ground_truth(gt: dict) -> dict:
    """Validate a ground-truth dict against clinical terminology.

    Args:
        gt: ground-truth dict from a case definition.

    Returns:
        ``{"valid": bool, "errors": [...], "warnings": [...]}``
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(gt, dict):
        return {"valid": False, "errors": ["ground_truth must be a dict"], "warnings": []}

    # Required structure: at least one scoring field should be present.
    scoring_fields = ("expected_hypotheses", "red_flags", "key_findings")
    has_any = any(gt.get(f) for f in scoring_fields)
    if not has_any:
        warnings.append("ground_truth has no scoring fields (expected_hypotheses, red_flags, key_findings)")

    # Validate each label list.
    for field in scoring_fields:
        raw = gt.get(field)
        if raw is None:
            continue
        if not isinstance(raw, list):
            errors.append(f"{field} must be a list")
            continue
        _validate_label_list(field, raw, errors, warnings)

    # Cross-field: red_flags consistency.
    _check_red_flag_consistency(gt, errors, warnings)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_case_ground_truth(case: dict) -> dict:
    """Validate the ground_truth block within a full case dict.

    Convenience wrapper — extracts ground_truth and delegates to
    :func:`validate_ground_truth`.

    Args:
        case: parsed case dict.

    Returns:
        ``{"valid": bool, "errors": [...], "warnings": [...]}``
    """
    gt = case.get("ground_truth")
    if gt is None:
        return {"valid": True, "errors": [], "warnings": ["no ground_truth field"]}
    return validate_ground_truth(gt)


# ── internal helpers ─────────────────────────────────────────────────


def _validate_label_list(
    field: str,
    labels: list,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check labels for unknown terms and canonicalization duplicates."""
    seen_canonical: dict[str, str] = {}

    for i, label in enumerate(labels):
        if not isinstance(label, str):
            errors.append(f"{field}[{i}] must be a string, got {type(label).__name__}")
            continue
        if not label.strip():
            errors.append(f"{field}[{i}] is empty")
            continue

        canonical = get_canonical_label(label)

        # Warn if label is not canonical (will be mapped during scoring).
        if canonical != label:
            warnings.append(
                f"{field}[{i}] '{label}' is a synonym — canonical form is '{canonical}'"
            )

        # Warn if label is unknown to the terminology registry.
        if get_term(canonical) is None:
            # Hypotheses are condition names, not symptoms — skip unknown check.
            if field != "expected_hypotheses":
                warnings.append(f"{field}[{i}] '{label}' is not in clinical terminology")

        # Error if canonicalization creates duplicates.
        canon_lower = canonical.lower()
        if canon_lower in seen_canonical:
            errors.append(
                f"{field}[{i}] '{label}' duplicates '{seen_canonical[canon_lower]}' "
                f"after canonicalization (both → '{canonical}')"
            )
        else:
            seen_canonical[canon_lower] = label


def _check_red_flag_consistency(
    gt: dict,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Warn if key_findings contain red-flag terms but red_flags is empty."""
    red_flags = gt.get("red_flags")
    key_findings = gt.get("key_findings")

    if key_findings is None or not isinstance(key_findings, list):
        return

    # Find key_findings that are red flags.
    red_flag_findings: list[str] = []
    for label in key_findings:
        if isinstance(label, str) and is_red_flag(label):
            red_flag_findings.append(label)

    if not red_flag_findings:
        return

    # red_flags is missing or empty but key_findings has red-flag terms.
    if not red_flags:
        warnings.append(
            f"key_findings contains red-flag terms ({', '.join(red_flag_findings)}) "
            f"but red_flags is empty — consider adding them"
        )
