"""Case provenance and safety metadata — unified case model.

Defines the provenance and safety schema for SCRIBE cases.  Every case
should declare its origin (synthetic, synthea, derived, imported) and
cases derived from real encounters must pass safety gates before use.

Pure validation — no I/O, no mutation, deterministic.
"""

from __future__ import annotations

import datetime


VALID_ORIGINS = frozenset({"synthetic", "synthea", "derived", "imported"})


def build_provenance(origin: str, **kwargs) -> dict:
    """Build a provenance dict with validated origin.

    Raises:
        ValueError: If *origin* is not in :data:`VALID_ORIGINS`.
    """
    if origin not in VALID_ORIGINS:
        raise ValueError(
            f"invalid origin {origin!r}; must be one of {sorted(VALID_ORIGINS)}"
        )
    prov: dict = {"origin": origin}
    prov.setdefault("created", kwargs.pop("created", datetime.date.today().isoformat()))
    prov.update(kwargs)
    return prov


def default_safety() -> dict:
    """Return a safe default safety block for non-derived cases."""
    return {
        "contains_real_data": False,
        "approved_for_evaluation": True,
    }


def validate_provenance(case: dict) -> dict:
    """Validate provenance and safety metadata on a case dict.

    Returns:
        ``{"valid": bool, "errors": [...], "warnings": [...]}``
    """
    errors: list[str] = []
    warnings: list[str] = []

    prov = case.get("provenance")
    if prov is None:
        warnings.append("missing provenance metadata")
        return {"valid": True, "errors": errors, "warnings": warnings}

    if not isinstance(prov, dict):
        errors.append("provenance must be a dict")
        return {"valid": False, "errors": errors, "warnings": warnings}

    origin = prov.get("origin")
    if origin is None:
        errors.append("provenance.origin is required")
    elif origin not in VALID_ORIGINS:
        errors.append(
            f"provenance.origin {origin!r} invalid; "
            f"must be one of {sorted(VALID_ORIGINS)}"
        )

    if not prov.get("created"):
        errors.append("provenance.created is required")

    # Safety validation.
    safety = case.get("safety")

    if origin == "derived":
        if safety is None:
            errors.append("safety block required for derived cases")
        elif not isinstance(safety, dict):
            errors.append("safety must be a dict")
        else:
            if not safety.get("anonymized"):
                errors.append("derived cases must have safety.anonymized=true")
            if not safety.get("approved_for_evaluation"):
                errors.append(
                    "derived cases must have safety.approved_for_evaluation=true"
                )
            if safety.get("contains_real_data"):
                errors.append(
                    "derived cases must have safety.contains_real_data=false"
                )

    if isinstance(safety, dict):
        if safety.get("contains_real_data") and not safety.get("anonymized"):
            errors.append(
                "cases with contains_real_data=true must have anonymized=true"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
