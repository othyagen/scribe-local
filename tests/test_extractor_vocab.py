"""Tests for app.extractor_vocab — loading vocabularies from JSON files."""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import pytest

from app import extractor_vocab
from app.extractor_vocab import load_vocab


# ── helpers ───────────────────────────────────────────────────────────

DEFAULTS = ["alpha", "bravo", "charlie"]


@pytest.fixture()
def vocab_dir(tmp_path):
    """Patch _RESOURCES_DIR to a temporary directory."""
    with mock.patch.object(extractor_vocab, "_RESOURCES_DIR", tmp_path):
        yield tmp_path


# ── tests ─────────────────────────────────────────────────────────────


def test_load_from_json(vocab_dir):
    """Valid JSON file loads correctly."""
    terms = ["xray", "yoga", "zulu"]
    (vocab_dir / "test.json").write_text(json.dumps(terms), encoding="utf-8")

    result = load_vocab("test", DEFAULTS)
    assert result == terms


def test_missing_file_returns_defaults(vocab_dir):
    """Missing file returns defaults without error."""
    result = load_vocab("nonexistent", DEFAULTS)
    assert result == DEFAULTS


def test_invalid_json_returns_defaults_with_warning(vocab_dir, capsys):
    """Broken JSON returns defaults and prints a warning to stderr."""
    (vocab_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    result = load_vocab("bad", DEFAULTS)
    assert result == DEFAULTS

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "bad.json" in captured.err


def test_empty_list_returns_empty(vocab_dir):
    """Valid empty array returns empty list, not defaults."""
    (vocab_dir / "empty.json").write_text("[]", encoding="utf-8")

    result = load_vocab("empty", DEFAULTS)
    assert result == []


def test_non_list_returns_defaults(vocab_dir, capsys):
    """JSON object (not array) returns defaults with warning."""
    (vocab_dir / "obj.json").write_text('{"a": 1}', encoding="utf-8")

    result = load_vocab("obj", DEFAULTS)
    assert result == DEFAULTS

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "does not contain a JSON array" in captured.err


def test_default_symptoms_file_loads():
    """The shipped symptoms.json loads and contains 'headache'."""
    from app.extractors import _DEFAULT_SYMPTOMS

    result = load_vocab("symptoms", _DEFAULT_SYMPTOMS)
    assert "headache" in result


def test_default_medications_file_loads():
    """The shipped medications.json loads and contains 'ibuprofen'."""
    from app.extractors import _DEFAULT_MEDICATIONS

    result = load_vocab("medications", _DEFAULT_MEDICATIONS)
    assert "ibuprofen" in result


def test_custom_term_available(vocab_dir):
    """A term added to the JSON file is present in the loaded list."""
    terms = ["alpha", "bravo", "custom_new_term"]
    (vocab_dir / "test.json").write_text(json.dumps(terms), encoding="utf-8")

    result = load_vocab("test", DEFAULTS)
    assert "custom_new_term" in result
