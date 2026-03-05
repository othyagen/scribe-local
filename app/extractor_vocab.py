"""Load extractor vocabularies from external JSON files.

Vocabulary files live under resources/extractors/<name>.json.
Falls back to hardcoded defaults if the file is missing or contains
invalid JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources" / "extractors"


def load_vocab(name: str, defaults: list[str]) -> list[str]:
    """Load vocabulary list from resources/extractors/<name>.json.

    Returns the loaded list on success.  Falls back to *defaults* if the
    file is missing or contains invalid JSON.  A missing file is silent;
    invalid JSON prints a warning to stderr.
    """
    path = _RESOURCES_DIR / f"{name}.json"

    if not path.exists():
        return list(defaults)

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: invalid vocabulary file {path}: {exc}", file=sys.stderr)
        return list(defaults)

    if not isinstance(data, list):
        print(
            f"WARNING: vocabulary file {path} does not contain a JSON array",
            file=sys.stderr,
        )
        return list(defaults)

    return data
