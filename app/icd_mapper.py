"""Deterministic ICD-10 code suggestion from structured clinical state.

Maps extracted symptoms and clinical patterns to ICD-10 codes using curated
file-based mappings.  Produces suggestions only — never auto-selects codes.

Two levels of suggestion:

1. **Symptom-level** — direct symptom-to-code mapping (always safe).
2. **Pattern-level** — conservative pattern-to-code mapping, only when a
   recognised clinical pattern has strong structured evidence.

When in doubt, falls back to symptom-level coding.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_DEFAULT_ICD10_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "resources", "classification", "icd10_symptoms.json",
)

_DEFAULT_ICD10_PATTERN_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "resources", "classification", "icd10_patterns.json",
)


# ── file loading ─────────────────────────────────────────────────


def _load_json_map(path: str, label: str) -> dict[str, dict]:
    """Load a JSON object file, lowercase keys, skip _-prefixed keys."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        print(
            f"WARNING: invalid JSON in {label} file: {path}",
            file=sys.stderr,
        )
        return {}

    if not isinstance(raw, dict):
        print(
            f"WARNING: {label} file is not a JSON object: {path}",
            file=sys.stderr,
        )
        return {}

    return {
        k.lower(): v for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, dict)
    }


def load_icd_map(path: str | None = None) -> dict[str, dict]:
    """Load a symptom → ICD-10 code mapping from a JSON file.

    Args:
        path: path to the JSON mapping file.  If ``None``, the shipped
            ``resources/classification/icd10_symptoms.json`` is used.

    Returns:
        dict mapping lowercase symptom strings to ICD-10 entry dicts.
        Returns an empty dict if the file is missing or invalid.
    """
    return _load_json_map(path or _DEFAULT_ICD10_PATH, "ICD-10")


def load_icd_pattern_map(path: str | None = None) -> dict[str, dict]:
    """Load a pattern → ICD-10 code mapping from a JSON file.

    Args:
        path: path to the JSON mapping file.  If ``None``, the shipped
            ``resources/classification/icd10_patterns.json`` is used.

    Returns:
        dict mapping lowercase pattern IDs to ICD-10 entry dicts.
    """
    return _load_json_map(path or _DEFAULT_ICD10_PATTERN_PATH, "ICD-10 pattern")


# ── suggestion logic ─────────────────────────────────────────────


def suggest_icd10_codes(
    clinical_state: dict,
    icd_map: dict[str, dict] | None = None,
    pattern_map: dict[str, dict] | None = None,
) -> list[dict]:
    """Suggest ICD-10 codes for extracted symptoms and clinical patterns.

    Symptom-level suggestions are always emitted first.  Pattern-level
    suggestions are added conservatively — only for patterns that have
    an explicit mapping in the pattern file and whose evidence symptoms
    are already covered by symptom-level suggestions.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
        icd_map: pre-loaded symptom mapping from :func:`load_icd_map`.
        pattern_map: pre-loaded pattern mapping from :func:`load_icd_pattern_map`.

    Returns:
        list of suggestion dicts, each with ``code``, ``label``, ``kind``,
        and ``evidence`` keys.  No duplicate codes.
    """
    if icd_map is None:
        icd_map = load_icd_map()
    if pattern_map is None:
        pattern_map = load_icd_pattern_map()

    seen_codes: set[str] = set()
    result: list[dict] = []

    # ── 1. Symptom-level suggestions ─────────────────────────
    symptoms: list[str] = clinical_state.get("symptoms", [])

    for symptom in symptoms:
        entry = icd_map.get(symptom.lower())
        if entry is None:
            continue
        code = entry.get("code", "")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        result.append({
            "code": code,
            "label": entry.get("label", ""),
            "kind": "symptom",
            "evidence": [symptom],
        })

    # ── 2. Pattern-level suggestions (conservative) ──────────
    derived = clinical_state.get("derived", {})
    patterns: list[dict] = derived.get("clinical_patterns", [])

    for pattern in patterns:
        pattern_id = pattern.get("pattern", "").lower()
        entry = pattern_map.get(pattern_id)
        if entry is None:
            continue
        code = entry.get("code", "")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        result.append({
            "code": code,
            "label": entry.get("label", ""),
            "kind": "pattern",
            "evidence": list(pattern.get("evidence", [])),
        })

    return result
