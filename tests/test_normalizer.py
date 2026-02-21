"""Tests for the Normalizer."""

from __future__ import annotations

import json

import pytest

from app.commit import RawSegment
from app.config import AppConfig, NormalizationConfig
from app.normalize import Normalizer, NormalizationChange, _split_punct


# ── helpers ──────────────────────────────────────────────────────────

def _make_segment(text: str) -> RawSegment:
    """Create a minimal RawSegment for testing."""
    return RawSegment(
        seg_id="seg_0001",
        t0=0.0,
        t1=1.0,
        speaker_id="spk_0",
        raw_text=text,
        model_name="test-model",
        language="en",
        paragraph_id="para_0000",
    )


def _make_normalizer(
    tmp_path,
    replacements: dict[str, dict[str, str]] | None = None,
    language: str = "en",
    fuzzy_threshold: float = 0.92,
    enabled: bool = True,
) -> Normalizer:
    """Create a Normalizer with custom lexicons written to tmp_path."""
    lexicon_dir = tmp_path / "lexicons"
    lang_dir = lexicon_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    if replacements is None:
        replacements = {}

    for domain in ("custom", "medical", "general"):
        data = {"replacements": replacements.get(domain, {})}
        (lang_dir / f"{domain}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    config = AppConfig(
        language=language,
        normalization=NormalizationConfig(
            enabled=enabled,
            fuzzy_threshold=fuzzy_threshold,
            lexicon_dir=str(lexicon_dir),
        ),
    )
    return Normalizer(config)


# ── _split_punct ─────────────────────────────────────────────────────

class TestSplitPunct:
    def test_no_punct(self):
        assert _split_punct("hello") == ("", "hello", "")

    def test_trailing_punct(self):
        assert _split_punct("hello,") == ("", "hello", ",")

    def test_leading_punct(self):
        assert _split_punct("(hello") == ("(", "hello", "")

    def test_both_punct(self):
        assert _split_punct("(hello!)") == ("(", "hello", "!)")

    def test_only_punct(self):
        assert _split_punct("...") == ("...", "", "")

    def test_empty(self):
        assert _split_punct("") == ("", "", "")


# ── disabled normalization ───────────────────────────────────────────

class TestDisabledNormalization:
    def test_pass_through_returns_raw_text(self, tmp_path):
        norm = _make_normalizer(tmp_path, enabled=False)
        seg = _make_segment("gonna do it")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "gonna do it"
        assert changes == []


# ── exact word replacement ───────────────────────────────────────────

class TestExactWordReplacement:
    def test_single_word_exact_match(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("I gonna leave")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "I going to leave"
        assert len(changes) == 1
        assert changes[0].from_text == "gonna"
        assert changes[0].to_text == "going to"
        assert changes[0].method == "exact"
        assert changes[0].confidence == 1.0

    def test_case_insensitive_match(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("GONNA do it")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "going to do it"

    def test_no_match_leaves_text_unchanged(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("hello world")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "hello world"
        assert changes == []

    def test_word_with_trailing_punct(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("I gonna, leave")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "I going to, leave"

    def test_multiple_replacements(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to", "wanna": "want to"},
        })
        seg = _make_segment("I gonna wanna leave")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "I going to want to leave"
        assert len(changes) == 2


# ── exact phrase replacement ─────────────────────────────────────────

class TestExactPhraseReplacement:
    def test_multi_word_phrase(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "medical": {"blood pressure": "blood pressure"},
        })
        seg = _make_segment("check blood pressure now")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "check blood pressure now"
        # identity replacement still logs a change
        assert len(changes) == 1
        assert changes[0].method == "exact"

    def test_phrase_case_insensitive(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "medical": {"blood pressure": "BP"},
        })
        seg = _make_segment("check Blood Pressure now")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "check BP now"

    def test_longest_phrase_matched_first(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "custom": {
                "high blood": "HB",
                "high blood pressure": "HBP",
            },
        })
        seg = _make_segment("has high blood pressure")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "has HBP"


# ── fuzzy matching ───────────────────────────────────────────────────

class TestFuzzyMatching:
    def test_fuzzy_match_above_threshold(self, tmp_path):
        # "paracetamol" vs "paracetamoll" (typo) — ratio ~0.96
        norm = _make_normalizer(tmp_path, replacements={
            "medical": {"paracetamol": "paracetamol"},
        }, fuzzy_threshold=0.90)
        seg = _make_segment("take paracetamoll")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "take paracetamol"
        assert len(changes) == 1
        assert changes[0].method == "fuzzy"
        assert changes[0].confidence >= 0.90

    def test_fuzzy_match_below_threshold_no_change(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "medical": {"paracetamol": "paracetamol"},
        }, fuzzy_threshold=0.99)
        seg = _make_segment("take paracetamoll")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "take paracetamoll"
        assert changes == []

    def test_exact_preferred_over_fuzzy(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        }, fuzzy_threshold=0.80)
        seg = _make_segment("gonna leave")
        result, changes = norm.normalize(seg)
        assert changes[0].method == "exact"


# ── domain priority ─────────────────────────────────────────────────

class TestDomainPriority:
    def test_custom_wins_over_general(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "custom": {"pt": "patient"},
            "general": {"pt": "point"},
        })
        seg = _make_segment("the pt arrived")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "the patient arrived"
        assert changes[0].domain == "custom"

    def test_custom_wins_over_medical(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "custom": {"bp": "business plan"},
            "medical": {"bp": "blood pressure"},
        })
        seg = _make_segment("review bp")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "review business plan"
        assert changes[0].domain == "custom"


# ── change audit metadata ───────────────────────────────────────────

class TestChangeAudit:
    def test_change_has_correct_seg_id(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna")
        _, changes = norm.normalize(seg)
        assert changes[0].seg_id == "seg_0001"

    def test_change_has_correct_speaker(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna")
        _, changes = norm.normalize(seg)
        assert changes[0].speaker_id == "spk_0"

    def test_change_has_correct_language(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna")
        _, changes = norm.normalize(seg)
        assert changes[0].language == "en"

    def test_change_to_dict_roundtrip(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna")
        _, changes = norm.normalize(seg)
        d = changes[0].to_dict()
        assert d["from_text"] == "gonna"
        assert d["to_text"] == "going to"
        assert d["method"] == "exact"
        assert d["domain"] == "general"


# ── normalized segment output ────────────────────────────────────────

class TestNormalizedSegmentOutput:
    def test_preserves_raw_text(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna leave")
        result, _ = norm.normalize(seg)
        assert result.raw_text == "gonna leave"
        assert result.normalized_text == "going to leave"

    def test_preserves_segment_metadata(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna")
        result, _ = norm.normalize(seg)
        assert result.seg_id == seg.seg_id
        assert result.t0 == seg.t0
        assert result.t1 == seg.t1
        assert result.speaker_id == seg.speaker_id
        assert result.model_name == seg.model_name
        assert result.language == seg.language
        assert result.paragraph_id == seg.paragraph_id

    def test_to_txt_line_format(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("gonna")
        result, _ = norm.normalize(seg)
        line = result.to_txt_line()
        assert "[spk_0]" in line
        assert "going to" in line


# ── empty / edge cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_text(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == ""
        assert changes == []

    def test_empty_lexicons(self, tmp_path):
        norm = _make_normalizer(tmp_path)
        seg = _make_segment("hello world")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "hello world"
        assert changes == []

    def test_only_punctuation(self, tmp_path):
        norm = _make_normalizer(tmp_path, replacements={
            "general": {"gonna": "going to"},
        })
        seg = _make_segment("...")
        result, changes = norm.normalize(seg)
        assert result.normalized_text == "..."
        assert changes == []
