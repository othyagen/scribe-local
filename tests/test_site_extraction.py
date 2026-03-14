"""Tests for anatomical site extraction."""

from __future__ import annotations

import pytest

from app.site_extraction import extract_sites, SITE_KEYWORDS


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
        assert extract_sites([]) == []

    def test_no_sites(self):
        assert extract_sites([_seg("patient has headache.")]) == []


# ── site detection ───────────────────────────────────────────────


class TestSiteDetection:
    def test_single_site(self):
        result = extract_sites([_seg("pain in the epigastric region.")])
        assert len(result) == 1
        assert result[0]["site"] == "epigastric"

    def test_multi_word_site(self):
        result = extract_sites([
            _seg("tenderness in the right upper quadrant."),
        ])
        assert len(result) == 1
        assert result[0]["site"] == "right upper quadrant"

    def test_seg_id_preserved(self):
        result = extract_sites([
            _seg("epigastric pain.", seg_id="seg_0005", speaker_id="spk_1", t0=5.0),
        ])
        assert result[0]["seg_id"] == "seg_0005"
        assert result[0]["speaker_id"] == "spk_1"
        assert result[0]["t_start"] == 5.0


# ── multiple occurrences ─────────────────────────────────────────


class TestMultipleOccurrences:
    def test_same_site_multiple_segments(self):
        result = extract_sites([
            _seg("epigastric pain noted.", seg_id="seg_0001"),
            _seg("epigastric tenderness.", seg_id="seg_0002"),
        ])
        assert len(result) == 2

    def test_different_sites_same_segment(self):
        result = extract_sites([
            _seg("pain in epigastric and lumbar regions."),
        ])
        assert len(result) == 2
        sites = {r["site"] for r in result}
        assert sites == {"epigastric", "lumbar"}


# ── case insensitive ─────────────────────────────────────────────


class TestCaseInsensitive:
    def test_uppercase_match(self):
        result = extract_sites([_seg("EPIGASTRIC pain.")])
        assert len(result) == 1
        assert result[0]["site"] == "epigastric"

    def test_mixed_case_match(self):
        result = extract_sites([_seg("Right Upper Quadrant tenderness.")])
        assert len(result) == 1
        assert result[0]["site"] == "right upper quadrant"


# ── vocabulary ───────────────────────────────────────────────────


class TestVocabulary:
    def test_vocabulary_loaded(self):
        assert len(SITE_KEYWORDS) >= 20

    def test_known_sites_in_vocab(self):
        lower_sites = {s.lower() for s in SITE_KEYWORDS}
        assert "epigastric" in lower_sites
        assert "right upper quadrant" in lower_sites
        assert "temporal" in lower_sites

    def test_various_sites(self):
        for site in ["substernal", "frontal", "occipital", "suprapubic"]:
            result = extract_sites([_seg(f"noted {site} pain.")])
            assert len(result) >= 1, f"Failed to detect {site}"
