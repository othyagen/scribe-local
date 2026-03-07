"""Export clinical notes from normalized transcripts using YAML templates.

Templates define sections (each with a title, list of extractors, and
optional scope).  Extractors are deterministic regex+keyword functions
registered in ``app.extractors``.

This module never modifies RAW or normalized outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from app.extractors import get_extractor
from app.commit import _fmt_ts
from app.role_detection import get_role_label


# ── template loading ──────────────────────────────────────────────────

TEMPLATE_DIR = Path("templates")

_REQUIRED_KEYS = {"name", "format", "sections"}
_VALID_FORMATS = {"markdown", "text"}
_VALID_SCOPES = {"all", "patient_only", "clinician_only"}

# Soft scoping: if scoped extraction yields fewer items than this,
# supplement with all-segment extraction.
_SOFT_SCOPE_MIN_ITEMS = 1


def load_template(template_id: str, template_dir: Optional[Path] = None) -> dict:
    """Load a YAML template by ID.

    Looks for ``<template_dir>/<template_id>.yaml``.
    Raises FileNotFoundError if the template does not exist.
    """
    base = template_dir or TEMPLATE_DIR
    path = base / f"{template_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"template not found: {path}"
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    validate_template(data)
    return data


def validate_template(template: dict) -> None:
    """Check that a template dict has the required structure.

    Raises ValueError on invalid templates.
    """
    missing = _REQUIRED_KEYS - set(template.keys())
    if missing:
        raise ValueError(f"template missing required keys: {missing}")

    fmt = template.get("format", "markdown")
    if fmt not in _VALID_FORMATS:
        raise ValueError(
            f"invalid template format '{fmt}', must be one of {_VALID_FORMATS}"
        )

    sections = template.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("template 'sections' must be a list")

    for i, section in enumerate(sections):
        if not isinstance(section, dict) or "title" not in section:
            raise ValueError(f"section {i} must be a dict with a 'title' key")
        scope = section.get("scope", "all")
        if scope not in _VALID_SCOPES:
            raise ValueError(
                f"section {i} has invalid scope '{scope}', "
                f"must be one of {_VALID_SCOPES}"
            )


# ── scope helpers ─────────────────────────────────────────────────────

def _template_needs_roles(template: dict) -> bool:
    """Return True if any section scope != 'all' or transcript_section is on."""
    if template.get("transcript_section", False):
        return True
    for section in template.get("sections", []):
        if section.get("scope", "all") != "all":
            return True
    return False


def _filter_segments_by_scope(
    segments: list[dict],
    scope: str,
    speaker_roles: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """Filter segments by speaker scope using detected roles.

    - scope "all" → return all segments
    - scope "patient_only" → segments from speakers with role==patient
    - scope "clinician_only" → segments from speakers with role==clinician

    Falls back to all segments when roles are unavailable or no speaker
    matches the requested role.
    """
    if scope == "all":
        return segments

    if not speaker_roles:
        return segments

    target_role = "patient" if scope == "patient_only" else "clinician"

    filtered = [
        seg for seg in segments
        if speaker_roles.get(seg.get("speaker_id", ""), {}).get("role") == target_role
    ]

    # Fallback: if no segments match the target role, return all
    if not filtered:
        return segments

    return filtered


def _run_extractors_on_text(
    text: str,
    extractor_names: list[str],
    seen: set[str],
) -> list[str]:
    """Run named extractors on text, deduplicating case-insensitively.

    Updates ``seen`` in-place and returns newly found items.
    """
    items: list[str] = []
    for ext_name in extractor_names:
        try:
            extractor = get_extractor(ext_name)
        except KeyError:
            continue
        for item in extractor(text):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                items.append(item)
    return items


def _run_extractors_on_segments(
    segments: list[dict],
    extractor_names: list[str],
    seen: set[str],
) -> list[dict]:
    """Run extractors per-segment, returning items with evidence metadata.

    Each result dict has:
      - ``item``: the extracted string
      - ``evidence``: ``{segment_id, speaker_id, t_start}``

    Deduplicates case-insensitively via *seen*; first occurrence wins.
    Updates *seen* in-place.
    """
    results: list[dict] = []
    for seg in segments:
        text = seg.get("normalized_text", "")
        if not text:
            continue
        evidence = {
            "segment_id": seg.get("seg_id", ""),
            "speaker_id": seg.get("speaker_id", ""),
            "t_start": seg.get("t0", 0.0),
        }
        for ext_name in extractor_names:
            try:
                extractor = get_extractor(ext_name)
            except KeyError:
                continue
            for item in extractor(text):
                key = item.lower()
                if key not in seen:
                    seen.add(key)
                    results.append({"item": item, "evidence": evidence})
    return results


def _fmt_evidence(ev: dict) -> str:
    """Format evidence as a compact bracket reference.

    Returns e.g. ``[seg_0003, spk_1, 01:42]``.
    Gracefully handles missing keys.
    """
    seg = ev.get("segment_id", "")
    spk = ev.get("speaker_id", "")
    t = ev.get("t_start")
    parts = [p for p in (seg, spk) if p]
    if t is not None:
        parts.append(_fmt_ts(t))
    if not parts:
        return ""
    return "[" + ", ".join(parts) + "]"


# ── note building ─────────────────────────────────────────────────────

def build_clinical_note(
    segments: list[dict],
    template: dict,
    speaker_roles: Optional[dict[str, dict]] = None,
    review_flags: Optional[list[dict]] = None,
    symptom_timeline: Optional[list[dict]] = None,
) -> str:
    """Build a clinical note string from normalized segments and a template.

    Pure function — no side effects, no file I/O.

    Args:
        segments: list of normalized segment dicts
        template: parsed YAML template dict
        speaker_roles: optional {speaker_id: {role, confidence, evidence}}
        review_flags: optional list of review flag dicts to append
        symptom_timeline: optional list of symptom–time dicts
    """
    fmt = template.get("format", "markdown")
    is_md = fmt == "markdown"
    show_evidence = template.get("show_evidence", False)

    lines: list[str] = []

    # Title
    name = template.get("name", "Clinical Note")
    if is_md:
        lines.append(f"# {name}")
    else:
        lines.append(name.upper())
        lines.append("=" * len(name))
    lines.append("")

    # Sections
    for section in template.get("sections", []):
        title = section["title"]
        extractors = section.get("extractors", [])
        scope = section.get("scope", "all")

        if is_md:
            lines.append(f"## {title}")
        else:
            lines.append(title)
            lines.append("-" * len(title))
        lines.append("")

        # Filter segments by scope
        scoped_segments = _filter_segments_by_scope(
            segments, scope, speaker_roles,
        )

        if not extractors:
            # No extractors — include raw transcript lines for this section
            for seg in scoped_segments:
                text = seg.get("normalized_text", "").strip()
                if text:
                    if is_md:
                        lines.append(f"- {text}")
                    else:
                        lines.append(f"  {text}")
            lines.append("")
            continue

        # Run extractors with soft scoping:
        # First try scoped text; if too sparse, supplement with all segments.
        seen: set[str] = set()

        if show_evidence:
            all_ev_items = _run_extractors_on_segments(
                scoped_segments, extractors, seen,
            )
            if scope != "all" and len(all_ev_items) < _SOFT_SCOPE_MIN_ITEMS:
                all_ev_items.extend(
                    _run_extractors_on_segments(segments, extractors, seen)
                )
            if all_ev_items:
                for entry in all_ev_items:
                    ref = _fmt_evidence(entry.get("evidence", {}))
                    suffix = f"  {ref}" if ref else ""
                    if is_md:
                        lines.append(f"- {entry['item']}{suffix}")
                    else:
                        lines.append(f"  {entry['item']}{suffix}")
            else:
                if is_md:
                    lines.append("_No items detected._")
                else:
                    lines.append("  (No items detected.)")
        else:
            scoped_text = " ".join(
                seg.get("normalized_text", "") for seg in scoped_segments
            ).strip()
            all_items = _run_extractors_on_text(scoped_text, extractors, seen)

            # Soft scoping: if scoped results are sparse and scope != "all",
            # supplement with all-segment extraction (deduplicated).
            if scope != "all" and len(all_items) < _SOFT_SCOPE_MIN_ITEMS:
                full_text = " ".join(
                    seg.get("normalized_text", "") for seg in segments
                ).strip()
                all_items.extend(
                    _run_extractors_on_text(full_text, extractors, seen)
                )

            if all_items:
                for item in all_items:
                    if is_md:
                        lines.append(f"- {item}")
                    else:
                        lines.append(f"  {item}")
            else:
                if is_md:
                    lines.append("_No items detected._")
                else:
                    lines.append("  (No items detected.)")
        lines.append("")

    # Transcript section (optional)
    if template.get("transcript_section", False):
        if is_md:
            lines.append("## Transcript")
        else:
            lines.append("Transcript")
            lines.append("-" * len("Transcript"))
        lines.append("")
        for seg in segments:
            text = seg.get("normalized_text", "").strip()
            if not text:
                continue
            t0 = seg.get("t0", 0.0)
            t1 = seg.get("t1", 0.0)
            speaker_id = seg.get("speaker_id", "")

            # Use role label when available, else raw speaker_id
            if speaker_roles and speaker_id in speaker_roles:
                label = get_role_label(
                    speaker_roles[speaker_id]["role"], speaker_id,
                )
            else:
                label = speaker_id

            ts0 = _fmt_ts(t0)
            ts1 = _fmt_ts(t1)
            if is_md:
                lines.append(f"- [{ts0} - {ts1}] [{label}] {text}")
            else:
                lines.append(f"  [{ts0} - {ts1}] [{label}] {text}")
        lines.append("")

    # Review flags section (optional)
    if template.get("show_review_flags", False) and review_flags:
        if is_md:
            lines.append("## Review Flags")
        else:
            lines.append("Review Flags")
            lines.append("-" * len("Review Flags"))
        lines.append("")
        for flag in review_flags:
            severity = flag.get("severity", "info")
            message = flag.get("message", "")
            prefix = severity.upper()
            if is_md:
                lines.append(f"- **{prefix}**: {message}")
            else:
                lines.append(f"  [{prefix}] {message}")
        lines.append("")

    # Symptom timeline section (optional)
    if template.get("show_symptom_timeline", False) and symptom_timeline:
        if is_md:
            lines.append("## Symptom Timeline")
        else:
            lines.append("Symptom Timeline")
            lines.append("-" * len("Symptom Timeline"))
        lines.append("")
        for entry in symptom_timeline:
            symptom = entry.get("symptom", "")
            time_expr = entry.get("time_expression")
            if time_expr:
                bullet = f"{symptom} \u2014 {time_expr}"
            else:
                bullet = symptom
            if is_md:
                lines.append(f"- {bullet}")
            else:
                lines.append(f"  {bullet}")
        lines.append("")

    return "\n".join(lines)


# ── file writer ───────────────────────────────────────────────────────

def write_clinical_note(
    segments: list[dict],
    template: dict,
    output_dir: str,
    session_ts: str,
    template_id: str,
    speaker_roles: Optional[dict[str, dict]] = None,
    review_flags: Optional[list[dict]] = None,
    symptom_timeline: Optional[list[dict]] = None,
) -> Path:
    """Write clinical note to ``clinical_note_<ts>_<template_id>.<ext>``.

    Returns the output path.
    """
    fmt = template.get("format", "markdown")
    ext = "md" if fmt == "markdown" else "txt"

    content = build_clinical_note(
        segments, template, speaker_roles, review_flags,
        symptom_timeline=symptom_timeline,
    )
    p = Path(output_dir) / f"clinical_note_{session_ts}_{template_id}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
