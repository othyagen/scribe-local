"""Extended case schema validation and metadata extraction.

Additive layer on top of :func:`app.case_system.validate_case`.  Validates
optional ``classification`` and ``patient`` fields introduced by the case
operating layer.  Does not replace or duplicate existing validation.

Metadata precedence rules (canonical, tested):
  - origin:     provenance.origin  →  meta.source  →  "unknown"
  - difficulty:  meta.difficulty   →  "unspecified"
  - tags:        meta.tags         →  []
  - title:       title             →  ""
  - description: description       →  ""

Pure functions — no I/O, no mutation, deterministic.
"""

from __future__ import annotations


# ── constants ──────────────────────────────────────────────────────


_CLASSIFICATION_FIELDS = frozenset({"organ_systems", "presenting_complaints", "diagnosis_targets"})
_DIAGNOSIS_SYSTEMS = frozenset({"icd10", "icpc", "snomed"})
_PATIENT_FIELDS = frozenset({"age", "sex"})


# ── validation ─────────────────────────────────────────────────────


def validate_extended_schema(case: dict) -> dict:
    """Validate optional classification and patient fields.

    Returns ``{"errors": [...], "warnings": [...]}``.  Callers merge
    these into the result of :func:`app.case_system.validate_case`.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Classification.
    classification = case.get("classification")
    if classification is None:
        warnings.append("missing classification metadata")
    elif not isinstance(classification, dict):
        errors.append("classification must be a dict")
    else:
        for field in ("organ_systems", "presenting_complaints"):
            val = classification.get(field)
            if val is not None:
                if not isinstance(val, list):
                    errors.append(f"classification.{field} must be a list")
                elif any(not isinstance(v, str) or not v for v in val):
                    errors.append(f"classification.{field} entries must be non-empty strings")

        targets = classification.get("diagnosis_targets")
        if targets is not None:
            if not isinstance(targets, dict):
                errors.append("classification.diagnosis_targets must be a dict")
            else:
                for system in targets:
                    if system not in _DIAGNOSIS_SYSTEMS:
                        warnings.append(
                            f"classification.diagnosis_targets: unknown system {system!r}"
                        )
                    entries = targets[system]
                    if not isinstance(entries, list):
                        errors.append(
                            f"classification.diagnosis_targets.{system} must be a list"
                        )
                        continue
                    for i, entry in enumerate(entries):
                        if not isinstance(entry, dict):
                            errors.append(
                                f"diagnosis_targets.{system}[{i}] must be a dict"
                            )
                        elif "code" not in entry or "display" not in entry:
                            errors.append(
                                f"diagnosis_targets.{system}[{i}] requires code and display"
                            )

    # Patient.
    patient = case.get("patient")
    if patient is None:
        warnings.append("missing patient metadata")
    elif not isinstance(patient, dict):
        errors.append("patient must be a dict")
    else:
        age = patient.get("age")
        if age is not None and not isinstance(age, (int, float)):
            errors.append("patient.age must be a number")
        sex = patient.get("sex")
        if sex is not None and not isinstance(sex, str):
            errors.append("patient.sex must be a string")

    return {"errors": errors, "warnings": warnings}


# ── metadata extraction ────────────────────────────────────────────


def extract_case_metadata(case: dict) -> dict:
    """Extract lightweight metadata for registry indexing.

    Precedence rules:
      - origin:     provenance.origin  →  meta.source  →  "unknown"
      - difficulty:  meta.difficulty   →  "unspecified"
      - tags:        meta.tags         →  []
    """
    meta = case.get("meta") or {}
    prov = case.get("provenance") or {}
    classification = case.get("classification") or {}
    patient = case.get("patient") or {}
    targets = classification.get("diagnosis_targets") or {}

    def _codes(system: str) -> list[str]:
        return [
            e["code"] for e in targets.get(system, [])
            if isinstance(e, dict) and "code" in e
        ]

    return {
        "case_id": case.get("case_id", ""),
        "title": case.get("title", ""),
        "origin": prov.get("origin") or meta.get("source") or "unknown",
        "difficulty": meta.get("difficulty") or "unspecified",
        "organ_systems": classification.get("organ_systems") or [],
        "presenting_complaints": classification.get("presenting_complaints") or [],
        "tags": list(meta.get("tags") or []),
        "patient_age": patient.get("age"),
        "patient_sex": patient.get("sex"),
        "icd10_codes": _codes("icd10"),
        "icpc_codes": _codes("icpc"),
        "snomed_codes": _codes("snomed"),
        "has_ground_truth": bool(case.get("ground_truth")),
        "segment_count": len(case.get("segments") or []),
    }
