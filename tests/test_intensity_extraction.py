"""Tests for numeric pain intensity extraction."""

from __future__ import annotations

import pytest

from app.intensity_extraction import extract_intensities


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── empty / basic ────────────────────────────────────────────────


class TestBasic:
    def test_empty_segments(self):
        assert extract_intensities([]) == []

    def test_no_intensity(self):
        assert extract_intensities([_seg("patient has headache.")]) == []


# ── format detection ─────────────────────────────────────────────


class TestFormats:
    def test_x_over_10(self):
        result = extract_intensities([_seg("pain is 7/10.")])
        assert len(result) == 1
        assert result[0]["value"] == 7
        assert result[0]["scale"] == "numeric"

    def test_x_out_of_10(self):
        result = extract_intensities([_seg("pain is 8 out of 10.")])
        assert len(result) == 1
        assert result[0]["value"] == 8

    def test_pain_level(self):
        result = extract_intensities([_seg("pain level 6.")])
        assert len(result) == 1
        assert result[0]["value"] == 6

    def test_pain_score(self):
        result = extract_intensities([_seg("pain score of 5.")])
        assert len(result) == 1
        assert result[0]["value"] == 5

    def test_vas(self):
        result = extract_intensities([_seg("VAS 4.")])
        assert len(result) == 1
        assert result[0]["value"] == 4

    def test_raw_text_preserved(self):
        result = extract_intensities([_seg("pain is 7/10.")])
        assert result[0]["raw_text"] == "7/10"


# ── clamping ─────────────────────────────────────────────────────


class TestClamping:
    def test_value_clamped_high(self):
        result = extract_intensities([_seg("pain is 11/10.")])
        assert len(result) == 1
        assert result[0]["value"] == 10

    def test_value_zero(self):
        result = extract_intensities([_seg("pain is 0/10.")])
        assert len(result) == 1
        assert result[0]["value"] == 0


# ── seg_id preserved ─────────────────────────────────────────────


class TestMetadata:
    def test_seg_id_preserved(self):
        result = extract_intensities([
            _seg("pain is 7/10.", seg_id="seg_0005", speaker_id="spk_1", t0=5.0),
        ])
        assert result[0]["seg_id"] == "seg_0005"
        assert result[0]["speaker_id"] == "spk_1"
        assert result[0]["t_start"] == 5.0


# ── deduplication ────────────────────────────────────────────────


class TestDeduplication:
    def test_same_value_same_segment(self):
        result = extract_intensities([
            _seg("pain is 7/10, I said 7 out of 10."),
        ])
        assert len(result) == 1

    def test_same_value_different_segments(self):
        result = extract_intensities([
            _seg("pain is 7/10.", seg_id="seg_0001"),
            _seg("still 7/10.", seg_id="seg_0002"),
        ])
        assert len(result) == 2


# ── no false positives ───────────────────────────────────────────


class TestNoFalsePositives:
    def test_page_reference_not_matched(self):
        result = extract_intensities([
            _seg("see page 7/10 for details."),
        ])
        # The "page" context should suppress false positive
        assert len(result) == 0
