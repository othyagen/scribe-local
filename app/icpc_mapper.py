"""Deterministic ICPC-2 code suggestion from structured clinical state.

Maps extracted symptoms to ICPC-2 symptom/complaint codes using a curated
file-based mapping.  Produces suggestions only — never auto-selects codes.

No ML, no LLM, no external API calls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_DEFAULT_ICPC_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "resources", "classification", "icpc_symptoms.json",
)


def load_icpc_map(path: str | None = None) -> dict[str, dict]:
    """Load a symptom → ICPC code mapping from a JSON file.

    Args:
        path: path to the JSON mapping file.  If ``None``, the shipped
            ``resources/classification/icpc_symptoms.json`` is used.

    Returns:
        dict mapping lowercase symptom strings to ICPC entry dicts.
        Returns an empty dict if the file is missing or invalid.
    """
    resolved = path or _DEFAULT_ICPC_PATH
    try:
        text = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        print(
            f"WARNING: invalid JSON in ICPC file: {resolved}",
            file=sys.stderr,
        )
        return {}

    if not isinstance(raw, dict):
        print(
            f"WARNING: ICPC file is not a JSON object: {resolved}",
            file=sys.stderr,
        )
        return {}

    # Normalise keys to lowercase, skip comment keys
    return {
        k.lower(): v for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, dict)
    }


def suggest_icpc_codes(
    clinical_state: dict,
    icpc_map: dict[str, dict] | None = None,
) -> list[dict]:
    """Suggest ICPC codes for extracted symptoms.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
        icpc_map: pre-loaded mapping from :func:`load_icpc_map`.
            If ``None``, the default shipped mapping is loaded.

    Returns:
        list of suggestion dicts, one per mapped symptom, preserving
        symptom order.  Duplicate codes (same code) are not repeated.
    """
    if icpc_map is None:
        icpc_map = load_icpc_map()

    symptoms: list[str] = clinical_state.get("symptoms", [])

    seen_codes: set[str] = set()
    result: list[dict] = []

    for symptom in symptoms:
        entry = icpc_map.get(symptom.lower())
        if entry is None:
            continue
        code = entry.get("code", "")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        result.append({
            "code": code,
            "label": entry.get("label", ""),
            "kind": entry.get("kind", "symptom"),
            "evidence": [symptom],
        })

    return result
