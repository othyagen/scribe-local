"""Canonical label alignment for evaluation boundaries.

Maps synonym / variant labels to a single canonical form so that
ground-truth expectations match system outputs during scoring.

Applied at evaluation boundaries only — never inside the reasoning
pipeline.  Pure functions, deterministic, no I/O, no input mutation.
"""

from __future__ import annotations

import copy

# ── synonym map ─────────────────────────────────────────────────────

# Explicit mapping: variant → canonical label.
# Keys must be lowercase.  Values are the canonical form.
# Extend this dict to cover new synonym families.
LABEL_SYNONYMS: dict[str, str] = {
    # Respiratory
    "shortness of breath": "dyspnea",
    "short of breath": "dyspnea",
    "breathlessness": "dyspnea",
    "difficulty breathing": "dyspnea",
    "sob": "dyspnea",
    # Urinary
    "painful urination": "dysuria",
    "burning urination": "dysuria",
    "frequent urination": "urinary frequency",
    # Cardiac
    "chest discomfort": "chest pain",
    "chest tightness": "chest pain",
}


# ── public API ──────────────────────────────────────────────────────


def canonicalize_label(label: str) -> str:
    """Canonicalize a single label string.

    Strips whitespace and applies synonym mapping (case-insensitive
    lookup).  Unknown labels are returned stripped but with original
    casing preserved — downstream ``_normalize()`` in scoring handles
    case folding.

    Args:
        label: raw label string.

    Returns:
        Canonical label if a synonym match is found, otherwise the
        original label stripped of whitespace.
    """
    stripped = label.strip()
    key = stripped.lower()
    return LABEL_SYNONYMS.get(key, stripped)


def canonicalize_labels(labels: list[str]) -> list[str]:
    """Canonicalize a list of labels, preserving order.

    Args:
        labels: list of raw label strings.

    Returns:
        New list with each label canonicalized.
    """
    return [canonicalize_label(l) for l in labels]


def canonicalize_ground_truth(gt: dict) -> dict:
    """Canonicalize all label lists in a ground-truth dict.

    Returns a new dict — does NOT mutate the input.

    Canonicalizes:
        - ``expected_hypotheses``
        - ``red_flags``
        - ``key_findings``

    Args:
        gt: ground-truth dict from a case definition.

    Returns:
        New dict with canonicalized label lists.
    """
    result = dict(gt)

    for key in ("expected_hypotheses", "red_flags", "key_findings"):
        raw = gt.get(key)
        if raw is not None:
            result[key] = canonicalize_labels(raw)

    return result
