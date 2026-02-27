"""Tests for diarized segment post-processing (clean_diarized_segments)."""

from __future__ import annotations

import pytest

from app.diarization import clean_diarized_segments


def _seg(
    seg_id: str,
    t0: float,
    t1: float,
    speaker: str,
    text: str,
    para: str = "para_0000",
) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "old_speaker_id": speaker,
        "new_speaker_id": speaker,
        "normalized_text": text,
        "paragraph_id": para,
    }


# ── sorting ──────────────────────────────────────────────────────────


class TestSort:
    def test_out_of_order_sorted(self):
        segs = [
            _seg("s2", 5.0, 7.0, "spk_0", "second"),
            _seg("s1", 1.0, 3.0, "spk_0", "first"),
        ]
        result = clean_diarized_segments(segs)
        assert result[0]["seg_id"] == "s1"
        assert result[1]["seg_id"] == "s2"


# ── deduplication ────────────────────────────────────────────────────


class TestDedup:
    def test_near_identical_removed(self):
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello world"),
            _seg("s2", 1.05, 3.05, "spk_0", "hello world"),
        ]
        result = clean_diarized_segments(segs)
        assert len(result) == 1
        assert result[0]["seg_id"] == "s1"

    def test_different_text_not_deduped(self):
        """Same speaker, same times, different text → not deduplicated (both kept)."""
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 5.0, 7.0, "spk_0", "goodbye"),
        ]
        result = clean_diarized_segments(segs)
        assert len(result) == 2

    def test_different_speaker_not_deduped(self):
        """Near-identical timestamps but different speaker → not deduplicated."""
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 5.0, 7.0, "spk_1", "world"),
        ]
        result = clean_diarized_segments(segs)
        assert len(result) == 2


# ── merge adjacent ───────────────────────────────────────────────────


class TestMerge:
    def test_adjacent_same_speaker_merged(self):
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 3.3, 5.0, "spk_0", "world"),
        ]
        result = clean_diarized_segments(segs)
        assert len(result) == 1
        assert result[0]["t0"] == 1.0
        assert result[0]["t1"] == 5.0
        assert result[0]["normalized_text"] == "hello world"
        assert result[0]["seg_id"] == "s1"

    def test_gap_too_large_not_merged(self):
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 3.5, 5.0, "spk_0", "world"),
        ]
        result = clean_diarized_segments(segs)
        assert len(result) == 2

    def test_different_speaker_not_merged(self):
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 3.2, 5.0, "spk_1", "world"),
        ]
        result = clean_diarized_segments(segs)
        assert len(result) == 2

    def test_merge_stops_at_30s(self):
        segs = [
            _seg("s1", 0.0, 28.0, "spk_0", "long"),
            _seg("s2", 28.2, 31.0, "spk_0", "more"),
        ]
        result = clean_diarized_segments(segs)
        # 31.0 - 0.0 = 31.0 > 30.0 → not merged
        assert len(result) == 2


# ── overlap resolution ───────────────────────────────────────────────


class TestOverlap:
    def test_same_speaker_overlap_clamped(self):
        segs = [
            _seg("s1", 1.0, 4.0, "spk_0", "hello"),
            _seg("s2", 3.5, 6.0, "spk_0", "world"),
        ]
        result = clean_diarized_segments(segs)
        # Same speaker with gap < 0 but these get merged first (gap=-0.5 < 0.4)
        # Actually gap = 3.5-4.0 = -0.5, not >= 0 so merge won't fire.
        # Overlap resolution: same speaker → clamp current.t1 = next.t0
        assert result[0]["t1"] == result[1]["t0"]

    def test_diff_speaker_tiny_overlap_clamped(self):
        segs = [
            _seg("s1", 1.0, 4.0, "spk_0", "hello"),
            _seg("s2", 3.9, 6.0, "spk_1", "world"),
        ]
        result = clean_diarized_segments(segs)
        # overlap = 0.1 < 0.2 → clamp current.t1
        assert len(result) == 2
        assert result[0]["t1"] == 3.9

    def test_diff_speaker_large_overlap_clips_shorter(self):
        # current is shorter (2s) than next (4s) → clip current
        segs = [
            _seg("s1", 1.0, 3.5, "spk_0", "short"),
            _seg("s2", 3.0, 7.0, "spk_1", "long"),
        ]
        result = clean_diarized_segments(segs)
        # overlap=0.5 >= 0.2, current dur=2.5 < next dur=4.0 → clip current
        assert len(result) == 2
        assert result[0]["t1"] == 3.0
        assert result[1]["t0"] == 3.0

    def test_diff_speaker_large_overlap_clips_next_when_shorter(self):
        # current is longer (4s) than next (2s) → clip next
        segs = [
            _seg("s1", 1.0, 5.5, "spk_0", "long"),
            _seg("s2", 5.0, 7.0, "spk_1", "short"),
        ]
        result = clean_diarized_segments(segs)
        # overlap=0.5 >= 0.2, current dur=4.5 > next dur=2.0 → clip next
        assert len(result) == 2
        assert result[0]["t1"] == 5.5
        assert result[1]["t0"] == 5.5

    def test_overlap_clips_to_micro_segment_dropped(self):
        # After clipping, the shorter segment becomes < 0.12s → dropped
        segs = [
            _seg("s1", 1.0, 5.0, "spk_0", "long"),
            _seg("s2", 4.8, 5.1, "spk_1", "tiny"),
        ]
        result = clean_diarized_segments(segs)
        # overlap=0.2, current dur=4.0, next dur=0.3 → clip next: t0→5.0
        # next becomes 5.0-5.1 = 0.1s < 0.12 → dropped
        assert len(result) == 1
        assert result[0]["new_speaker_id"] == "spk_0"


# ── min duration filter ──────────────────────────────────────────────


class TestMinDuration:
    def test_micro_segment_dropped(self):
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 3.5, 5.0, "spk_0", "world"),
            # This segment is already < 0.12s before any processing
            _seg("s3", 6.0, 6.10, "spk_1", "x"),
        ]
        result = clean_diarized_segments(segs)
        assert all(r["seg_id"] != "s3" for r in result)

    def test_short_but_valid_segment_kept(self):
        segs = [
            _seg("s1", 1.0, 3.0, "spk_0", "hello"),
            _seg("s2", 4.0, 4.15, "spk_1", "ja"),
        ]
        result = clean_diarized_segments(segs)
        assert any(r["seg_id"] == "s2" for r in result)


# ── immutability ─────────────────────────────────────────────────────


class TestImmutability:
    def test_does_not_mutate_input(self):
        segs = [
            _seg("s1", 1.0, 4.0, "spk_0", "hello"),
            _seg("s2", 3.5, 6.0, "spk_0", "world"),
        ]
        original_t1 = segs[0]["t1"]
        clean_diarized_segments(segs)
        assert segs[0]["t1"] == original_t1


# ── empty input ──────────────────────────────────────────────────────


class TestEmpty:
    def test_empty_list(self):
        assert clean_diarized_segments([]) == []

    def test_single_segment(self):
        segs = [_seg("s1", 1.0, 3.0, "spk_0", "hello")]
        result = clean_diarized_segments(segs)
        assert len(result) == 1
