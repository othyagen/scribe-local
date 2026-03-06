"""Tests for multi-template clinical note export (--export-notes)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.main import _parse_export_notes, _export_multi_notes


# ── _parse_export_notes tests ────────────────────────────────────────

def test_parse_export_notes_basic():
    args = SimpleNamespace(export_notes="soap,isbar")
    assert _parse_export_notes(args) == ["soap", "isbar"]


def test_parse_export_notes_whitespace():
    args = SimpleNamespace(export_notes=" soap , isbar ")
    assert _parse_export_notes(args) == ["soap", "isbar"]


def test_parse_export_notes_dedupe():
    args = SimpleNamespace(export_notes="soap,soap,isbar")
    assert _parse_export_notes(args) == ["soap", "isbar"]


def test_parse_export_notes_empty_entries():
    args = SimpleNamespace(export_notes="soap,,isbar,")
    assert _parse_export_notes(args) == ["soap", "isbar"]


def test_parse_export_notes_none():
    args = SimpleNamespace(export_notes=None)
    assert _parse_export_notes(args) == []


def test_parse_export_notes_missing_attr():
    args = SimpleNamespace()
    assert _parse_export_notes(args) == []


# ── _export_multi_notes tests ───────────────────────────────────────

TS = "2026-03-01_10-00-00"


def _make_segments(texts: list[str]) -> list[dict]:
    return [
        {
            "seg_id": f"seg_{i+1:04d}",
            "t0": float(i),
            "t1": float(i + 1),
            "speaker_id": "spk_0",
            "normalized_text": t,
        }
        for i, t in enumerate(texts)
    ]


def test_export_multi_notes_writes_files(tmp_path):
    segments = _make_segments(["I have a headache.", "Take ibuprofen."])
    paths = _export_multi_notes(
        ["soap", "summary"], segments, str(tmp_path), TS, "en",
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    assert paths[0].name == f"clinical_note_{TS}_soap.md"
    assert paths[1].name == f"clinical_note_{TS}_summary.md"


def test_export_multi_notes_roles_computed_once(tmp_path):
    segments = _make_segments(["I have a headache.", "Take ibuprofen."])
    # soap has scoped sections → needs roles; summary may not
    with patch(
        "app.role_detection.detect_speaker_roles", return_value={}
    ) as mock_roles:
        paths = _export_multi_notes(
            ["soap", "summary"], segments, str(tmp_path), TS, "en",
        )
    assert len(paths) == 2
    assert mock_roles.call_count <= 1


def test_export_multi_notes_missing_template_raises(tmp_path):
    segments = _make_segments(["hello"])
    with pytest.raises(FileNotFoundError):
        _export_multi_notes(
            ["nonexistent_template"], segments, str(tmp_path), TS, "en",
        )
