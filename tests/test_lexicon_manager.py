"""Tests for lexicon management CLI functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.lexicon_manager import (
    add_term,
    remove_term,
    list_terms,
    format_term_list,
    validate_term,
    _lexicon_path,
    _load_lexicon,
    _save_lexicon,
)


# ── helpers ──────────────────────────────────────────────────────────

def _setup_lexicon(tmp_path: Path, lang: str = "en", replacements: dict | None = None):
    """Create a custom lexicon file for testing."""
    lexdir = tmp_path / "lexicons"
    lang_dir = lexdir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    data = {"replacements": replacements or {}}
    (lang_dir / "custom.json").write_text(json.dumps(data), encoding="utf-8")
    return str(lexdir)


# ── add_term ─────────────────────────────────────────────────────────

class TestAddTerm:
    def test_add_new_term(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path)
        action, ok = add_term(lexdir, "en", "bp", "blood pressure")
        assert action == "added"
        assert ok is True
        # Verify persisted
        path = _lexicon_path(lexdir, "en")
        data = json.loads(path.read_text("utf-8"))
        assert data["replacements"]["bp"] == "blood pressure"

    def test_update_existing_term(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path, replacements={"bp": "old value"})
        action, ok = add_term(lexdir, "en", "bp", "blood pressure")
        assert action == "updated"
        assert ok is True
        path = _lexicon_path(lexdir, "en")
        data = json.loads(path.read_text("utf-8"))
        assert data["replacements"]["bp"] == "blood pressure"

    def test_add_preserves_existing_terms(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path, replacements={"pt": "patient"})
        add_term(lexdir, "en", "bp", "blood pressure")
        path = _lexicon_path(lexdir, "en")
        data = json.loads(path.read_text("utf-8"))
        assert data["replacements"]["pt"] == "patient"
        assert data["replacements"]["bp"] == "blood pressure"

    def test_add_creates_file_if_missing(self, tmp_path):
        lexdir = str(tmp_path / "lexicons")
        add_term(lexdir, "da", "afd", "afdeling")
        path = _lexicon_path(lexdir, "da")
        assert path.exists()
        data = json.loads(path.read_text("utf-8"))
        assert data["replacements"]["afd"] == "afdeling"

    def test_add_multi_word_term(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path)
        action, _ = add_term(lexdir, "en", "blood presure", "blood pressure")
        assert action == "added"
        terms = list_terms(lexdir, "en")
        assert ("blood presure", "blood pressure") in terms


# ── remove_term ──────────────────────────────────────────────────────

class TestRemoveTerm:
    def test_remove_existing_term(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path, replacements={"pt": "patient", "bp": "blood pressure"})
        removed = remove_term(lexdir, "en", "pt")
        assert removed is True
        path = _lexicon_path(lexdir, "en")
        data = json.loads(path.read_text("utf-8"))
        assert "pt" not in data["replacements"]
        assert data["replacements"]["bp"] == "blood pressure"

    def test_remove_nonexistent_term(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path, replacements={"pt": "patient"})
        removed = remove_term(lexdir, "en", "xyz")
        assert removed is False

    def test_remove_from_empty_lexicon(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path)
        removed = remove_term(lexdir, "en", "pt")
        assert removed is False

    def test_remove_from_missing_file(self, tmp_path):
        lexdir = str(tmp_path / "lexicons")
        removed = remove_term(lexdir, "en", "pt")
        assert removed is False


# ── list_terms ───────────────────────────────────────────────────────

class TestListTerms:
    def test_list_sorted_alphabetically(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path, replacements={
            "pt": "patient",
            "bp": "blood pressure",
            "dept": "department",
        })
        terms = list_terms(lexdir, "en")
        assert terms == [
            ("bp", "blood pressure"),
            ("dept", "department"),
            ("pt", "patient"),
        ]

    def test_list_empty_lexicon(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path)
        terms = list_terms(lexdir, "en")
        assert terms == []

    def test_list_missing_file(self, tmp_path):
        lexdir = str(tmp_path / "lexicons")
        terms = list_terms(lexdir, "en")
        assert terms == []

    def test_list_case_insensitive_sort(self, tmp_path):
        lexdir = _setup_lexicon(tmp_path, replacements={
            "Zebra": "zebra",
            "apple": "apple fruit",
        })
        terms = list_terms(lexdir, "en")
        assert terms[0][0] == "apple"
        assert terms[1][0] == "Zebra"


# ── format_term_list ─────────────────────────────────────────────────

class TestFormatTermList:
    def test_format_empty(self):
        assert format_term_list([]) == "No custom terms defined."

    def test_format_with_terms(self):
        terms = [("bp", "blood pressure"), ("pt", "patient")]
        output = format_term_list(terms)
        assert "bp → blood pressure" in output
        assert "pt → patient" in output

    def test_format_lines(self):
        terms = [("a", "alpha"), ("b", "beta")]
        lines = format_term_list(terms).split("\n")
        assert len(lines) == 2


# ── validate_term ────────────────────────────────────────────────────

class TestValidateTerm:
    def test_valid_term(self):
        assert validate_term("patient") is None

    def test_valid_multi_word(self):
        assert validate_term("blood pressure") is None

    def test_empty_string(self):
        err = validate_term("")
        assert err is not None
        assert "empty" in err

    def test_whitespace_only(self):
        err = validate_term("   ")
        assert err is not None
        assert "whitespace" in err

    def test_leading_whitespace(self):
        err = validate_term(" pt")
        assert err is not None
        assert "leading" in err or "whitespace" in err

    def test_trailing_whitespace(self):
        err = validate_term("pt ")
        assert err is not None
        assert "trailing" in err or "whitespace" in err


# ── internal helpers ─────────────────────────────────────────────────

class TestInternals:
    def test_lexicon_path(self, tmp_path):
        path = _lexicon_path(str(tmp_path), "da")
        assert path == tmp_path / "da" / "custom.json"

    def test_load_missing_file(self, tmp_path):
        result = _load_lexicon(tmp_path / "nonexistent.json")
        assert result == {}

    def test_save_creates_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "custom.json"
        _save_lexicon(path, {"hello": "world"})
        assert path.exists()
        data = json.loads(path.read_text("utf-8"))
        assert data["replacements"]["hello"] == "world"

    def test_round_trip(self, tmp_path):
        path = tmp_path / "test.json"
        original = {"pt": "patient", "bp": "blood pressure"}
        _save_lexicon(path, original)
        loaded = _load_lexicon(path)
        assert loaded == original
