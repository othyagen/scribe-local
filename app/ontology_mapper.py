"""Lightweight deterministic ontology mapper.

Maps extracted symptoms to standardised clinical concept entries
using a curated file-based mapping (initially SNOMED CT).

No ML, no LLM, no external API calls.  Exact case-insensitive
lookup only — no fuzzy matching.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_DEFAULT_ONTOLOGY_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "resources", "ontology", "symptoms_snomed.json",
)


def load_ontology_map(path: str | None = None) -> dict[str, dict]:
    """Load a symptom → concept mapping from a JSON file.

    Args:
        path: path to the JSON mapping file.  If ``None``, the
            shipped ``resources/ontology/symptoms_snomed.json`` is used.

    Returns:
        dict mapping lowercase symptom strings to concept dicts.
        Returns an empty dict if the file is missing or invalid.
    """
    resolved = path or _DEFAULT_ONTOLOGY_PATH
    try:
        text = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        print(
            f"WARNING: invalid JSON in ontology file: {resolved}",
            file=sys.stderr,
        )
        return {}

    if not isinstance(raw, dict):
        print(
            f"WARNING: ontology file is not a JSON object: {resolved}",
            file=sys.stderr,
        )
        return {}

    # Normalise keys to lowercase
    return {k.lower(): v for k, v in raw.items()}


def map_symptoms_to_concepts(
    clinical_state: dict,
    ontology_map: dict[str, dict] | None = None,
) -> list[dict]:
    """Map extracted symptoms to ontology concepts.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.
        ontology_map: pre-loaded mapping from :func:`load_ontology_map`.
            If ``None``, the default shipped mapping is loaded.

    Returns:
        list of concept dicts, one per mapped symptom, preserving
        symptom order.  Symptoms not found in the map are skipped.
        Duplicate concepts (same code) are not repeated.
    """
    if ontology_map is None:
        ontology_map = load_ontology_map()

    symptoms: list[str] = clinical_state.get("symptoms", [])

    seen_codes: set[str] = set()
    result: list[dict] = []

    for symptom in symptoms:
        entry = ontology_map.get(symptom.lower())
        if entry is None:
            continue
        code = entry.get("code", "")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        result.append({
            "text": symptom,
            **entry,
        })

    return result
