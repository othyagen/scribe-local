"""Lexicon management — add, remove, and list normalization terms.

Targets the custom.json lexicon file (highest priority in normalization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _lexicon_path(lexicon_dir: str, language: str) -> Path:
    """Return the path to the custom lexicon file for *language*."""
    return Path(lexicon_dir) / language / "custom.json"


def _load_lexicon(path: Path) -> Dict[str, str]:
    """Load replacements dict from a lexicon JSON file."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("replacements", {})


def _save_lexicon(path: Path, replacements: Dict[str, str]) -> None:
    """Write replacements dict back to a lexicon JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"replacements": replacements}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_term(value: str) -> str | None:
    """Validate a term string.  Returns error message or None if valid."""
    if not value:
        return "term must not be empty"
    if not value.strip():
        return "term must not be only whitespace"
    if value != value.strip():
        return "term must not have leading or trailing whitespace"
    return None


def add_term(
    lexicon_dir: str, language: str, from_text: str, to_text: str,
) -> Tuple[str, bool]:
    """Add or update a term mapping in the custom lexicon.

    Returns (action, success) where action is 'added' or 'updated'.
    """
    path = _lexicon_path(lexicon_dir, language)
    replacements = _load_lexicon(path)
    action = "updated" if from_text in replacements else "added"
    replacements[from_text] = to_text
    _save_lexicon(path, replacements)
    return action, True


def remove_term(
    lexicon_dir: str, language: str, from_text: str,
) -> bool:
    """Remove a term mapping from the custom lexicon.

    Returns True if the term was found and removed, False if not found.
    """
    path = _lexicon_path(lexicon_dir, language)
    replacements = _load_lexicon(path)
    if from_text not in replacements:
        return False
    del replacements[from_text]
    _save_lexicon(path, replacements)
    return True


def list_terms(
    lexicon_dir: str, language: str,
) -> List[Tuple[str, str]]:
    """List all term mappings in the custom lexicon, sorted alphabetically.

    Returns list of (from_text, to_text) tuples.
    """
    path = _lexicon_path(lexicon_dir, language)
    replacements = _load_lexicon(path)
    return sorted(replacements.items(), key=lambda x: x[0].lower())


def format_term_list(terms: List[Tuple[str, str]]) -> str:
    """Format a list of term mappings for display."""
    if not terms:
        return "No custom terms defined."
    lines = []
    for from_text, to_text in terms:
        lines.append(f"  {from_text} → {to_text}")
    return "\n".join(lines)
