"""Tests for ASR confidence surfacing and review flag pipeline integration."""

import pytest

from app.confidence import detect_low_confidence_segments, AVG_LOGPROB_THRESHOLD
from app.export_clinical_note import build_clinical_note
from app.review_flags import generate_review_flags


# ── detect_low_confidence_segments tests ─────────────────────────────


def _make_entry(seg_id, avg_logprob, no_speech_prob=0.1, compression_ratio=1.5):
    return {
        "seg_id": seg_id,
        "t0": 0.0,
        "t1": 1.0,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
    }


def test_detect_low_confidence_segments_filters():
    entries = [
        _make_entry("seg_0", -0.5),   # clean
        _make_entry("seg_1", -1.5),   # low confidence
        _make_entry("seg_2", -0.8),   # clean
    ]
    result = detect_low_confidence_segments(entries)
    assert len(result) == 1
    assert result[0]["seg_id"] == "seg_1"


def test_detect_low_confidence_segments_custom_threshold():
    entries = [
        _make_entry("seg_0", -0.5),
        _make_entry("seg_1", -0.7),
        _make_entry("seg_2", -1.5),
    ]
    result = detect_low_confidence_segments(entries, threshold=-0.6)
    assert len(result) == 2
    seg_ids = {e["seg_id"] for e in result}
    assert seg_ids == {"seg_1", "seg_2"}


def test_detect_low_confidence_segments_none_logprob_excluded():
    entries = [
        _make_entry("seg_0", None),
        _make_entry("seg_1", -1.5),
    ]
    result = detect_low_confidence_segments(entries)
    assert len(result) == 1
    assert result[0]["seg_id"] == "seg_1"


def test_detect_low_confidence_segments_empty_input():
    assert detect_low_confidence_segments([]) == []


def test_detect_low_confidence_segments_includes_flags():
    entries = [_make_entry("seg_0", -1.5)]
    result = detect_low_confidence_segments(entries)
    assert len(result) == 1
    assert "low_confidence" in result[0]["flags"]


def test_detect_low_confidence_segments_all_clean():
    entries = [
        _make_entry("seg_0", -0.3),
        _make_entry("seg_1", -0.5),
    ]
    result = detect_low_confidence_segments(entries)
    assert result == []


# ── review flags in clinical notes integration ───────────────────────


def test_review_flags_with_confidence_in_note():
    """Template with show_review_flags: true + low-confidence → section appears."""
    segments = [
        {"seg_id": "seg_0", "normalized_text": "Hello world", "t0": 0.0, "t1": 1.0},
    ]
    confidence_entries = [_make_entry("seg_0", -1.5)]
    review_flags = generate_review_flags(segments, confidence_entries=confidence_entries)

    template = {
        "name": "Test Note",
        "format": "markdown",
        "sections": [{"title": "Notes"}],
        "show_review_flags": True,
    }
    note = build_clinical_note(segments, template, review_flags=review_flags)
    assert "## Review Flags" in note
    assert "low_confidence_segment" in note.lower().replace(" ", "_") or "Low ASR confidence" in note


def test_review_flags_without_confidence_no_crash():
    """Missing confidence data still produces a note without errors."""
    segments = [
        {"seg_id": "seg_0", "normalized_text": "Hello world", "t0": 0.0, "t1": 1.0},
    ]
    review_flags = generate_review_flags(segments, confidence_entries=None)

    template = {
        "name": "Test Note",
        "format": "markdown",
        "sections": [{"title": "Notes"}],
        "show_review_flags": True,
    }
    # Should not raise — review flags section only appears if there are flags
    note = build_clinical_note(segments, template, review_flags=review_flags)
    assert "# Test Note" in note
