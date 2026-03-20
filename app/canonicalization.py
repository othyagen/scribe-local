"""Canonical label alignment for evaluation boundaries.

Maps synonym / variant labels to a single canonical form so that
ground-truth expectations match system outputs during scoring.

Delegates to :mod:`app.clinical_terminology` as the single source of
truth for synonym resolution.  A small supplementary map covers terms
not yet promoted to the terminology registry.

Applied at evaluation boundaries only — never inside the reasoning
pipeline.  Pure functions, deterministic, no I/O, no input mutation.
"""

from __future__ import annotations

from app.clinical_terminology import (
    CLINICAL_TERMS,
    get_canonical_label as _terminology_lookup,
)

# ── supplementary synonyms ───────────────────────────────────────────

# Terms not yet in the terminology registry.  Kept minimal — migrate
# to CLINICAL_TERMS as terms are promoted.
_EXTRA_SYNONYMS: dict[str, str] = {
    "frequent urination": "urinary frequency",
}

# ── derived synonym map (public, for tests / inspection) ─────────────

# Built from the terminology registry + extras.  Preserves the
# LABEL_SYNONYMS export that existing tests rely on.
LABEL_SYNONYMS: dict[str, str] = {}
for _label, _term in CLINICAL_TERMS.items():
    for _syn in _term["synonyms"]:
        LABEL_SYNONYMS[_syn.lower()] = _label
LABEL_SYNONYMS.update(_EXTRA_SYNONYMS)


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

    # Primary lookup via terminology registry
    result = _terminology_lookup(stripped)
    if result != stripped:
        return result

    # Supplementary lookup for terms not yet in the registry
    key = stripped.lower()
    return _EXTRA_SYNONYMS.get(key, stripped)


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
