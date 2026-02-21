"""Tests for diarization turn smoothing."""

from __future__ import annotations

import pytest

from app.config import DiarizationConfig, AppConfig, _build_diarization
from app.diarization import smooth_turns


def _turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


# ── merge short turns ────────────────────────────────────────────────

class TestMergeShortTurns:
    def test_short_turn_merged_into_prev_same_speaker(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 3.5, "spk_0"),  # 0.5s < 0.7s, same as prev
            _turn(3.5, 6.0, "spk_1"),
        ]
        result = smooth_turns(turns)
        assert len(result) == 2
        assert result[0]["speaker"] == "spk_0"
        assert result[0]["end"] == 3.5
        assert result[1]["speaker"] == "spk_1"

    def test_short_turn_merged_into_next_same_speaker(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 3.5, "spk_1"),  # 0.5s < 0.7s, same as next
            _turn(3.5, 6.0, "spk_1"),
        ]
        result = smooth_turns(turns)
        assert len(result) == 2
        assert result[0]["speaker"] == "spk_0"
        assert result[0]["end"] == 3.0
        assert result[1]["speaker"] == "spk_1"
        assert result[1]["start"] == 3.0

    def test_short_turn_merged_into_longer_neighbor(self):
        turns = [
            _turn(0.0, 4.0, "spk_0"),   # 4s (longer)
            _turn(4.0, 4.5, "spk_2"),   # 0.5s short, different from both
            _turn(4.5, 6.0, "spk_1"),   # 1.5s
        ]
        result = smooth_turns(turns)
        assert len(result) == 2
        assert result[0]["speaker"] == "spk_0"
        assert result[0]["end"] == 4.5  # absorbed the short turn
        assert result[1]["speaker"] == "spk_1"

    def test_short_turn_merged_into_shorter_when_longer_not_available(self):
        # Short turn between two, next is longer
        turns = [
            _turn(0.0, 1.0, "spk_0"),   # 1.0s
            _turn(1.0, 1.5, "spk_2"),   # 0.5s short
            _turn(1.5, 5.0, "spk_1"),   # 3.5s (longer)
        ]
        result = smooth_turns(turns)
        assert len(result) == 2
        assert result[1]["speaker"] == "spk_1"
        assert result[1]["start"] == 1.0  # absorbed short turn

    def test_short_turn_at_start(self):
        turns = [
            _turn(0.0, 0.3, "spk_0"),   # short, no prev
            _turn(0.3, 3.0, "spk_1"),
        ]
        result = smooth_turns(turns)
        assert len(result) == 1
        assert result[0]["speaker"] == "spk_1"
        assert result[0]["start"] == 0.0

    def test_short_turn_at_end(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 3.4, "spk_1"),   # short, no next
        ]
        result = smooth_turns(turns)
        assert len(result) == 1
        assert result[0]["speaker"] == "spk_0"
        assert result[0]["end"] == 3.4

    def test_all_turns_long_no_change(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 6.0, "spk_1"),
            _turn(6.0, 9.0, "spk_0"),
        ]
        result = smooth_turns(turns)
        assert len(result) == 3
        assert result == turns

    def test_cascade_merge(self):
        # After merging the first short turn, adjacent turns may become
        # same-speaker and the gap merge should handle it
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 3.3, "spk_1"),   # short → merge into longer prev
            _turn(3.3, 6.0, "spk_0"),
        ]
        result = smooth_turns(turns)
        # Short spk_1 merged into spk_0 prev, then two spk_0 turns
        # gap-merged (gap = 0.0)
        assert len(result) == 1
        assert result[0]["speaker"] == "spk_0"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 6.0


# ── gap merge ────────────────────────────────────────────────────────

class TestGapMerge:
    def test_same_speaker_gap_merged(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.2, 6.0, "spk_0"),   # gap 0.2s <= 0.3s
        ]
        result = smooth_turns(turns)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 6.0

    def test_different_speaker_gap_not_merged(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.1, 6.0, "spk_1"),   # different speaker
        ]
        result = smooth_turns(turns)
        assert len(result) == 2

    def test_gap_above_threshold_not_merged(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.5, 6.0, "spk_0"),   # gap 0.5s > 0.3s
        ]
        result = smooth_turns(turns)
        assert len(result) == 2


# ── preservation ─────────────────────────────────────────────────────

class TestPreservation:
    def test_timestamps_monotonic(self):
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(2.0, 2.4, "spk_1"),   # short
            _turn(2.4, 2.8, "spk_0"),   # short
            _turn(2.8, 5.0, "spk_1"),
            _turn(5.0, 8.0, "spk_0"),
        ]
        result = smooth_turns(turns)
        for t in result:
            assert t["start"] < t["end"], f"start >= end: {t}"
        for i in range(len(result) - 1):
            assert result[i]["end"] <= result[i + 1]["start"], \
                f"non-monotonic: {result[i]} -> {result[i+1]}"

    def test_speaker_labels_preserved(self):
        original_speakers = {"spk_0", "spk_1", "spk_2"}
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 3.5, "spk_2"),
            _turn(3.5, 6.0, "spk_1"),
        ]
        result = smooth_turns(turns)
        result_speakers = {t["speaker"] for t in result}
        assert result_speakers.issubset(original_speakers)

    def test_empty_input(self):
        assert smooth_turns([]) == []

    def test_single_turn(self):
        turns = [_turn(0.0, 5.0, "spk_0")]
        result = smooth_turns(turns)
        assert len(result) == 1
        assert result[0] == turns[0]

    def test_does_not_mutate_input(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 3.5, "spk_1"),
            _turn(3.5, 6.0, "spk_0"),
        ]
        original = [dict(t) for t in turns]
        smooth_turns(turns)
        assert turns == original


# ── config parsing ───────────────────────────────────────────────────

class TestConfigParsing:
    def test_smoothing_defaults(self):
        cfg = DiarizationConfig()
        assert cfg.smoothing is True
        assert cfg.min_turn_sec == 0.7
        assert cfg.gap_merge_sec == 0.3

    def test_smoothing_from_yaml_dict(self):
        d = {
            "enabled": True,
            "backend": "pyannote",
            "smoothing": False,
            "min_turn_sec": 1.0,
            "gap_merge_sec": 0.5,
        }
        cfg = _build_diarization(d)
        assert cfg.smoothing is False
        assert cfg.min_turn_sec == 1.0
        assert cfg.gap_merge_sec == 0.5
