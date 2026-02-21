"""Speaker tagging — assign free-text tags/labels to diarized speakers.

Pure derived layer.  Reads diarized outputs, writes tag_labeled files.
Never modifies RAW, normalized, or diarized outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_or_create_tags(output_dir: str, timestamp: str) -> dict:
    """Load existing speaker tags or create an empty mapping.

    File: ``speaker_tags_<timestamp>.json``
    """
    path = Path(output_dir) / f"speaker_tags_{timestamp}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def apply_auto_tags(tags: dict, mode: str, speakers: list[str]) -> dict:
    """Fill missing speaker entries with auto-generated tags.

    Modes:
        ``alphabetical`` — spk_0 → "Speaker A", spk_1 → "Speaker B", ...
        ``index``        — spk_0 → "Speaker 1", spk_1 → "Speaker 2", ...

    Existing entries are NEVER overwritten.
    """
    if mode == "none":
        return tags

    for spk in speakers:
        if spk in tags:
            continue

        # Extract numeric index from speaker id (e.g. "spk_2" → 2)
        m = re.match(r"spk_(\d+)", spk)
        idx = int(m.group(1)) if m else speakers.index(spk)

        if mode == "alphabetical":
            tag = f"Speaker {chr(65 + idx)}"
        elif mode == "index":
            tag = f"Speaker {idx + 1}"
        else:
            continue

        tags[spk] = {"tag": tag, "label": None}

    return tags


def set_tag(tags: dict, speaker: str, tag: str) -> dict:
    """Set or update the tag for a speaker."""
    if speaker not in tags:
        tags[speaker] = {"tag": None, "label": None}
    tags[speaker]["tag"] = tag
    return tags


def set_label(tags: dict, speaker: str, label: str) -> dict:
    """Set or update the label for a speaker."""
    if speaker not in tags:
        tags[speaker] = {"tag": None, "label": None}
    tags[speaker]["label"] = label
    return tags


def save_tags(tags: dict, output_dir: str, timestamp: str) -> Path:
    """Write speaker tags to ``speaker_tags_<timestamp>.json``."""
    path = Path(output_dir) / f"speaker_tags_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    return path


def _format_speaker_token(entry: dict) -> str | None:
    """Build display token from a tag entry, or None if no tag."""
    tag = entry.get("tag")
    if not tag:
        return None
    label = entry.get("label")
    if label:
        return f"{tag}: {label}"
    return tag


def generate_tag_labeled_txt(
    diarized_txt_path: Path,
    tags: dict,
    output_dir: str,
    timestamp: str,
) -> Path:
    """Generate a transcript with speaker tags replacing speaker_ids.

    Reads ``diarized_<timestamp>.txt`` and writes
    ``tag_labeled_<timestamp>.txt``.
    """
    content = diarized_txt_path.read_text(encoding="utf-8")

    def _replace(m: re.Match) -> str:
        spk = m.group(1)
        entry = tags.get(spk)
        if entry:
            token = _format_speaker_token(entry)
            if token:
                return f"[{token}]"
        return m.group(0)

    result = re.sub(r"\[(spk_\d+)\]", _replace, content)

    out = Path(output_dir) / f"tag_labeled_{timestamp}.txt"
    out.write_text(result, encoding="utf-8")
    return out
