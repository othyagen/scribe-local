"""Tests for configuration validation (--validate-config)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.config import AppConfig, NormalizationConfig
from app.config_validation import (
    validate_config,
    format_validation_report,
    _check_output_dir,
    _check_lexicons,
    _check_extractor_vocabs,
    _check_templates,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_config(
    tmp_path: Path,
    output_dir: str | None = None,
    lexicon_dir: str | None = None,
    language: str = "en",
) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = output_dir or str(tmp_path / "outputs")
    cfg.language = language
    if lexicon_dir:
        cfg.normalization.lexicon_dir = lexicon_dir
    else:
        cfg.normalization.lexicon_dir = str(tmp_path / "lexicons")
    return cfg


def _write_lexicon(base: Path, lang: str, domain: str, data: dict) -> None:
    d = base / lang
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{domain}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8",
    )


def _write_template(template_dir: Path, name: str, data: dict) -> None:
    template_dir.mkdir(parents=True, exist_ok=True)
    with open(template_dir / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def _write_vocab(vocab_dir: Path, name: str, data: list | dict | str) -> None:
    vocab_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        (vocab_dir / f"{name}.json").write_text(data, encoding="utf-8")
    else:
        (vocab_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )


# ── output directory ─────────────────────────────────────────────────


class TestCheckOutputDir:
    def test_existing_dir_ok(self, tmp_path):
        out = tmp_path / "outputs"
        out.mkdir()
        issues: list[dict] = []
        _check_output_dir(str(out), issues)
        assert issues == []

    def test_missing_dir_with_existing_parent_ok(self, tmp_path):
        out = tmp_path / "outputs"  # does not exist, but parent does
        issues: list[dict] = []
        _check_output_dir(str(out), issues)
        assert issues == []

    def test_missing_dir_and_parent(self, tmp_path):
        out = tmp_path / "nonexistent" / "deeply" / "nested"
        issues: list[dict] = []
        _check_output_dir(str(out), issues)
        assert len(issues) == 1
        assert issues[0]["level"] == "warning"
        assert "parent" in issues[0]["message"]

    def test_path_is_file_not_dir(self, tmp_path):
        f = tmp_path / "outputs"
        f.write_text("not a dir")
        issues: list[dict] = []
        _check_output_dir(str(f), issues)
        assert len(issues) == 1
        assert issues[0]["level"] == "error"
        assert "not a directory" in issues[0]["message"]


# ── lexicons ─────────────────────────────────────────────────────────


class TestCheckLexicons:
    def test_valid_lexicons(self, tmp_path):
        base = tmp_path / "lexicons"
        _write_lexicon(base, "en", "custom", {"replacements": {"a": "b"}})
        _write_lexicon(base, "en", "medical", {"replacements": {}})
        issues: list[dict] = []
        _check_lexicons(str(base), "en", issues)
        assert issues == []

    def test_missing_lexicon_dir(self, tmp_path):
        issues: list[dict] = []
        _check_lexicons(str(tmp_path / "nonexistent"), "en", issues)
        assert len(issues) == 1
        assert issues[0]["level"] == "warning"

    def test_malformed_json(self, tmp_path):
        base = tmp_path / "lexicons"
        d = base / "en"
        d.mkdir(parents=True)
        (d / "custom.json").write_text("{bad json", encoding="utf-8")
        issues: list[dict] = []
        _check_lexicons(str(base), "en", issues)
        assert len(issues) == 1
        assert issues[0]["level"] == "error"
        assert "invalid JSON" in issues[0]["message"]

    def test_missing_replacements_key(self, tmp_path):
        base = tmp_path / "lexicons"
        _write_lexicon(base, "en", "custom", {"wrong_key": {}})
        issues: list[dict] = []
        _check_lexicons(str(base), "en", issues)
        assert len(issues) == 1
        assert issues[0]["level"] == "warning"
        assert "replacements" in issues[0]["message"]

    def test_missing_individual_file_ok(self, tmp_path):
        """Missing individual lexicon file is fine (optional)."""
        base = tmp_path / "lexicons" / "en"
        base.mkdir(parents=True)
        issues: list[dict] = []
        _check_lexicons(str(tmp_path / "lexicons"), "en", issues)
        assert issues == []


# ── extractor vocabularies ───────────────────────────────────────────


class TestCheckExtractorVocabs:
    def test_valid_vocabs(self, tmp_path, monkeypatch):
        vocab_dir = tmp_path / "resources" / "extractors"
        _write_vocab(vocab_dir, "symptoms", ["headache", "nausea"])
        _write_vocab(vocab_dir, "medications", ["aspirin"])
        monkeypatch.chdir(tmp_path)
        issues: list[dict] = []
        _check_extractor_vocabs(issues)
        assert issues == []

    def test_missing_vocab_file(self, tmp_path, monkeypatch):
        vocab_dir = tmp_path / "resources" / "extractors"
        vocab_dir.mkdir(parents=True)
        # only symptoms, no medications
        _write_vocab(vocab_dir, "symptoms", ["headache"])
        monkeypatch.chdir(tmp_path)
        issues: list[dict] = []
        _check_extractor_vocabs(issues)
        assert len(issues) == 1
        assert "medications" in issues[0]["message"]

    def test_malformed_json(self, tmp_path, monkeypatch):
        vocab_dir = tmp_path / "resources" / "extractors"
        _write_vocab(vocab_dir, "symptoms", "{not an array}")
        _write_vocab(vocab_dir, "medications", ["aspirin"])
        monkeypatch.chdir(tmp_path)
        issues: list[dict] = []
        _check_extractor_vocabs(issues)
        assert len(issues) == 1
        assert issues[0]["level"] == "error"

    def test_not_array(self, tmp_path, monkeypatch):
        vocab_dir = tmp_path / "resources" / "extractors"
        _write_vocab(vocab_dir, "symptoms", {"not": "array"})
        _write_vocab(vocab_dir, "medications", ["aspirin"])
        monkeypatch.chdir(tmp_path)
        issues: list[dict] = []
        _check_extractor_vocabs(issues)
        assert len(issues) == 1
        assert "not a JSON array" in issues[0]["message"]


# ── templates ────────────────────────────────────────────────────────


class TestCheckTemplates:
    def test_valid_templates(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        _write_template(tpl_dir, "soap", {
            "name": "SOAP Note",
            "format": "markdown",
            "sections": [{"title": "Subjective"}],
        })
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        assert issues == []

    def test_missing_template_dir(self, tmp_path):
        issues: list[dict] = []
        _check_templates(issues, template_dir=tmp_path / "nonexistent")
        assert len(issues) == 1
        assert issues[0]["level"] == "warning"

    def test_no_yaml_files(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        assert len(issues) == 1
        assert "no template files" in issues[0]["message"]

    def test_missing_required_fields(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        _write_template(tpl_dir, "bad", {"name": "Bad Note"})
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        assert len(issues) == 1
        assert issues[0]["level"] == "error"
        assert "missing required fields" in issues[0]["message"]

    def test_invalid_format(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        _write_template(tpl_dir, "bad", {
            "name": "Note",
            "format": "html",
            "sections": [],
        })
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        assert len(issues) == 1
        assert "invalid format" in issues[0]["message"]

    def test_malformed_yaml(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "bad.yaml").write_text(
            ":\n  - : :\n  [invalid", encoding="utf-8",
        )
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        assert len(issues) == 1
        assert issues[0]["level"] == "error"
        assert "invalid YAML" in issues[0]["message"]

    def test_sections_not_list(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        _write_template(tpl_dir, "bad", {
            "name": "Note",
            "format": "markdown",
            "sections": "not a list",
        })
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        assert len(issues) == 1
        assert "must be a list" in issues[0]["message"]

    def test_multiple_templates_mixed(self, tmp_path):
        tpl_dir = tmp_path / "templates"
        _write_template(tpl_dir, "good", {
            "name": "Good",
            "format": "markdown",
            "sections": [],
        })
        _write_template(tpl_dir, "bad", {"name": "Bad"})
        issues: list[dict] = []
        _check_templates(issues, template_dir=tpl_dir)
        # Only the bad one produces an issue
        assert len(issues) == 1
        assert "bad.yaml" in issues[0]["message"]


# ── format_validation_report ─────────────────────────────────────────


class TestFormatReport:
    def test_all_pass(self, tmp_path):
        cfg = _make_config(tmp_path)
        (tmp_path / "outputs").mkdir()
        report = format_validation_report([], cfg)
        assert "[PASS]" in report
        assert "config loaded" in report
        assert "[FAIL]" not in report

    def test_errors_shown(self, tmp_path):
        cfg = _make_config(tmp_path)
        issues = [
            {"level": "error", "message": "template bad.yaml missing required fields: format, sections"},
        ]
        report = format_validation_report(issues, cfg)
        assert "ERROR" in report
        assert "template" in report.lower()

    def test_warnings_shown(self, tmp_path):
        cfg = _make_config(tmp_path)
        issues = [
            {"level": "warning", "message": "lexicon directory not found: /fake"},
        ]
        report = format_validation_report(issues, cfg)
        assert "WARNING" in report


# ── full validate_config ─────────────────────────────────────────────


class TestValidateConfig:
    def test_valid_config_no_issues(self, tmp_path, monkeypatch):
        """Valid config with all resources in place."""
        # Set up lexicons
        lex_dir = tmp_path / "lexicons"
        _write_lexicon(lex_dir, "en", "custom", {"replacements": {}})

        # Set up extractor vocabs
        vocab_dir = tmp_path / "resources" / "extractors"
        _write_vocab(vocab_dir, "symptoms", ["headache"])
        _write_vocab(vocab_dir, "medications", ["aspirin"])

        # Set up templates
        tpl_dir = tmp_path / "templates"
        _write_template(tpl_dir, "soap", {
            "name": "SOAP",
            "format": "markdown",
            "sections": [{"title": "S"}],
        })

        # Set up output dir
        (tmp_path / "outputs").mkdir()

        monkeypatch.chdir(tmp_path)
        cfg = _make_config(tmp_path, lexicon_dir=str(lex_dir))
        issues = validate_config(cfg)
        assert issues == []

    def test_multiple_issues_collected(self, tmp_path, monkeypatch):
        """Missing resources produce multiple issues."""
        monkeypatch.chdir(tmp_path)
        # No lexicons, no vocabs, no templates, no output dir parent
        cfg = _make_config(
            tmp_path,
            output_dir=str(tmp_path / "nonexistent" / "deep" / "out"),
            lexicon_dir=str(tmp_path / "missing_lexicons"),
        )
        issues = validate_config(cfg)
        # Should have at least: lexicon dir, vocab files, template dir, output dir
        assert len(issues) >= 3
