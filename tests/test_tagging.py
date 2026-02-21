"""Tests for the speaker tagging layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tagging import (
    load_or_create_tags,
    apply_auto_tags,
    set_tag,
    set_label,
    save_tags,
    generate_tag_labeled_txt,
)
from app.config import build_arg_parser


# ── load / create ─────────────────────────────────────────────────────

class TestLoadOrCreateTags:
    def test_creates_empty_when_missing(self, tmp_path):
        tags = load_or_create_tags(str(tmp_path), "2026-01-01_12-00-00")
        assert tags == {}

    def test_loads_existing_file(self, tmp_path):
        p = tmp_path / "speaker_tags_2026-01-01_12-00-00.json"
        existing = {"spk_0": {"tag": "Host", "label": None}}
        p.write_text(json.dumps(existing), encoding="utf-8")
        tags = load_or_create_tags(str(tmp_path), "2026-01-01_12-00-00")
        assert tags == existing

    def test_does_not_overwrite_existing(self, tmp_path):
        p = tmp_path / "speaker_tags_2026-01-01_12-00-00.json"
        existing = {"spk_0": {"tag": "Host", "label": "Alice"}}
        p.write_text(json.dumps(existing), encoding="utf-8")
        load_or_create_tags(str(tmp_path), "2026-01-01_12-00-00")
        reloaded = json.loads(p.read_text(encoding="utf-8"))
        assert reloaded == existing


# ── auto tags ─────────────────────────────────────────────────────────

class TestAutoTags:
    def test_alphabetical_fills_missing(self):
        tags = {}
        apply_auto_tags(tags, "alphabetical", ["spk_0", "spk_1", "spk_2"])
        assert tags["spk_0"]["tag"] == "Speaker A"
        assert tags["spk_1"]["tag"] == "Speaker B"
        assert tags["spk_2"]["tag"] == "Speaker C"

    def test_index_fills_missing(self):
        tags = {}
        apply_auto_tags(tags, "index", ["spk_0", "spk_1"])
        assert tags["spk_0"]["tag"] == "Speaker 1"
        assert tags["spk_1"]["tag"] == "Speaker 2"

    def test_does_not_overwrite_existing(self):
        tags = {"spk_0": {"tag": "Host", "label": "Alice"}}
        apply_auto_tags(tags, "alphabetical", ["spk_0", "spk_1"])
        assert tags["spk_0"]["tag"] == "Host"
        assert tags["spk_0"]["label"] == "Alice"
        assert tags["spk_1"]["tag"] == "Speaker B"

    def test_none_is_noop(self):
        tags = {}
        apply_auto_tags(tags, "none", ["spk_0"])
        assert tags == {}


# ── set tag / label ───────────────────────────────────────────────────

class TestSetTagLabel:
    def test_set_tag_creates_entry(self):
        tags = {}
        set_tag(tags, "spk_0", "Interviewer")
        assert tags["spk_0"]["tag"] == "Interviewer"
        assert tags["spk_0"]["label"] is None

    def test_set_tag_updates_existing(self):
        tags = {"spk_0": {"tag": "Host", "label": "Alice"}}
        set_tag(tags, "spk_0", "Guest")
        assert tags["spk_0"]["tag"] == "Guest"
        assert tags["spk_0"]["label"] == "Alice"

    def test_set_label_creates_entry(self):
        tags = {}
        set_label(tags, "spk_0", "Mette")
        assert tags["spk_0"]["label"] == "Mette"
        assert tags["spk_0"]["tag"] is None

    def test_set_label_updates_existing(self):
        tags = {"spk_0": {"tag": "Host", "label": "Alice"}}
        set_label(tags, "spk_0", "Bob")
        assert tags["spk_0"]["label"] == "Bob"
        assert tags["spk_0"]["tag"] == "Host"

    def test_unknown_speaker_allowed(self):
        tags = {}
        set_tag(tags, "spk_99", "Mystery")
        assert tags["spk_99"]["tag"] == "Mystery"


# ── save tags ─────────────────────────────────────────────────────────

class TestSaveTags:
    def test_writes_json(self, tmp_path):
        tags = {"spk_0": {"tag": "Host", "label": "Alice"}}
        path = save_tags(tags, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.name == "speaker_tags_2026-01-01_12-00-00.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == tags


# ── generate tag-labeled txt ──────────────────────────────────────────

def _write_diarized_txt(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "diarized_2026-01-01_12-00-00.txt"
    p.write_text(content, encoding="utf-8")
    return p


class TestGenerateTagLabeledTxt:
    def test_replaces_speaker_with_tag(self, tmp_path):
        txt = _write_diarized_txt(tmp_path,
            "[00:00:00.000 - 00:00:03.000] [spk_0] hello world\n")
        tags = {"spk_0": {"tag": "Host", "label": None}}
        out = generate_tag_labeled_txt(txt, tags, str(tmp_path), "2026-01-01_12-00-00")
        content = out.read_text(encoding="utf-8")
        assert "[Host]" in content
        assert "[spk_0]" not in content
        assert "hello world" in content

    def test_uses_tag_and_label(self, tmp_path):
        txt = _write_diarized_txt(tmp_path,
            "[00:00:00.000 - 00:00:03.000] [spk_0] hello\n")
        tags = {"spk_0": {"tag": "Host", "label": "Alice"}}
        out = generate_tag_labeled_txt(txt, tags, str(tmp_path), "2026-01-01_12-00-00")
        content = out.read_text(encoding="utf-8")
        assert "[Host: Alice]" in content

    def test_keeps_original_when_no_tag(self, tmp_path):
        txt = _write_diarized_txt(tmp_path,
            "[00:00:00.000 - 00:00:03.000] [spk_0] hello\n")
        tags = {}
        out = generate_tag_labeled_txt(txt, tags, str(tmp_path), "2026-01-01_12-00-00")
        content = out.read_text(encoding="utf-8")
        assert "[spk_0]" in content

    def test_keeps_original_when_tag_is_null(self, tmp_path):
        txt = _write_diarized_txt(tmp_path,
            "[00:00:00.000 - 00:00:03.000] [spk_0] hello\n")
        tags = {"spk_0": {"tag": None, "label": "Alice"}}
        out = generate_tag_labeled_txt(txt, tags, str(tmp_path), "2026-01-01_12-00-00")
        content = out.read_text(encoding="utf-8")
        assert "[spk_0]" in content

    def test_preserves_paragraph_breaks(self, tmp_path):
        txt = _write_diarized_txt(tmp_path,
            "[00:00:00.000 - 00:00:03.000] [spk_0] hello\n"
            "\n"
            "[00:00:05.000 - 00:00:08.000] [spk_1] world\n")
        tags = {"spk_0": {"tag": "A", "label": None},
                "spk_1": {"tag": "B", "label": None}}
        out = generate_tag_labeled_txt(txt, tags, str(tmp_path), "2026-01-01_12-00-00")
        content = out.read_text(encoding="utf-8")
        lines = content.split("\n")
        assert lines[1] == ""

    def test_multiple_speakers_in_file(self, tmp_path):
        txt = _write_diarized_txt(tmp_path,
            "[00:00:00.000 - 00:00:03.000] [spk_0] hello\n"
            "[00:00:03.000 - 00:00:06.000] [spk_1] world\n")
        tags = {"spk_0": {"tag": "Host", "label": None},
                "spk_1": {"tag": "Guest", "label": "Bob"}}
        out = generate_tag_labeled_txt(txt, tags, str(tmp_path), "2026-01-01_12-00-00")
        content = out.read_text(encoding="utf-8")
        assert "[Host]" in content
        assert "[Guest: Bob]" in content

    def test_output_filename(self, tmp_path):
        txt = _write_diarized_txt(tmp_path, "")
        out = generate_tag_labeled_txt(txt, {}, str(tmp_path), "2026-01-01_12-00-00")
        assert out.name == "tag_labeled_2026-01-01_12-00-00.txt"

    def test_empty_input(self, tmp_path):
        txt = _write_diarized_txt(tmp_path, "")
        out = generate_tag_labeled_txt(txt, {}, str(tmp_path), "2026-01-01_12-00-00")
        assert out.read_text(encoding="utf-8") == ""


# ── CLI parsing ───────────────────────────────────────────────────────

class TestCLIParsing:
    def test_auto_tags_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.auto_tags == "none"

    def test_auto_tags_choices(self):
        parser = build_arg_parser()
        for choice in ("none", "alphabetical", "index"):
            args = parser.parse_args(["--auto-tags", choice])
            assert args.auto_tags == choice

    def test_set_tag_repeatable(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--set-tag", "spk_0=Host",
            "--set-tag", "spk_1=Guest",
        ])
        assert args.set_tag == ["spk_0=Host", "spk_1=Guest"]

    def test_set_label_repeatable(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--set-label", "spk_0=Alice",
            "--set-label", "spk_1=Bob",
        ])
        assert args.set_label == ["spk_0=Alice", "spk_1=Bob"]

    def test_session_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--session", "2026-01-01_12-00-00"])
        assert args.session == "2026-01-01_12-00-00"

    def test_session_default_none(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.session is None
