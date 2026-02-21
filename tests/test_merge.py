"""Tests for speaker merge functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import build_arg_parser
from app.diarization import (
    apply_merge_map,
    load_merge_map,
    resolve_merge_chains,
    save_merge_map,
)


def _turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


# ── resolve chains ───────────────────────────────────────────────────

class TestResolveChains:
    def test_direct_mapping_unchanged(self):
        result = resolve_merge_chains({"spk_2": "spk_0"})
        assert result == {"spk_2": "spk_0"}

    def test_chain_resolved(self):
        result = resolve_merge_chains({"spk_3": "spk_2", "spk_2": "spk_0"})
        assert result["spk_3"] == "spk_0"
        assert result["spk_2"] == "spk_0"

    def test_long_chain(self):
        result = resolve_merge_chains({
            "spk_4": "spk_3",
            "spk_3": "spk_2",
            "spk_2": "spk_1",
            "spk_1": "spk_0",
        })
        for key in ("spk_1", "spk_2", "spk_3", "spk_4"):
            assert result[key] == "spk_0"

    def test_cycle_raises(self):
        with pytest.raises(ValueError, match="Cycle detected"):
            resolve_merge_chains({"spk_0": "spk_1", "spk_1": "spk_0"})

    def test_self_reference_raises(self):
        with pytest.raises(ValueError, match="Cycle detected"):
            resolve_merge_chains({"spk_0": "spk_0"})

    def test_empty_map(self):
        assert resolve_merge_chains({}) == {}

    def test_multiple_independent_merges(self):
        result = resolve_merge_chains({
            "spk_2": "spk_0",
            "spk_3": "spk_1",
        })
        assert result == {"spk_2": "spk_0", "spk_3": "spk_1"}


# ── apply merge map ─────────────────────────────────────────────────

class TestApplyMergeMap:
    def test_replaces_speaker_ids(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 6.0, "spk_1"),
            _turn(6.0, 9.0, "spk_2"),  # becomes spk_0, not adjacent to first spk_0
        ]
        result = apply_merge_map(turns, {"spk_2": "spk_0"})
        assert len(result) == 3
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"
        assert result[2]["speaker"] == "spk_0"

    def test_merges_adjacent_same_speaker(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 6.0, "spk_2"),  # becomes spk_0
            _turn(6.0, 9.0, "spk_1"),
        ]
        result = apply_merge_map(turns, {"spk_2": "spk_0"})
        assert len(result) == 2
        assert result[0]["speaker"] == "spk_0"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 6.0
        assert result[1]["speaker"] == "spk_1"

    def test_non_adjacent_same_speaker_kept(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 6.0, "spk_1"),
            _turn(6.0, 9.0, "spk_2"),  # becomes spk_0
        ]
        result = apply_merge_map(turns, {"spk_2": "spk_0"})
        assert len(result) == 3
        assert result[0]["speaker"] == "spk_0"
        assert result[2]["speaker"] == "spk_0"

    def test_empty_merge_map_no_change(self):
        turns = [_turn(0.0, 3.0, "spk_0"), _turn(3.0, 6.0, "spk_1")]
        result = apply_merge_map(turns, {})
        assert len(result) == 2
        assert result[0] == turns[0]
        assert result[1] == turns[1]

    def test_does_not_mutate_input(self):
        turns = [
            _turn(0.0, 3.0, "spk_0"),
            _turn(3.0, 6.0, "spk_2"),
        ]
        original = [dict(t) for t in turns]
        apply_merge_map(turns, {"spk_2": "spk_0"})
        assert turns == original

    def test_overlapping_turns_takes_max_end(self):
        turns = [
            _turn(0.0, 4.0, "spk_0"),
            _turn(3.0, 6.0, "spk_2"),  # overlaps, becomes spk_0
        ]
        result = apply_merge_map(turns, {"spk_2": "spk_0"})
        assert len(result) == 1
        assert result[0]["end"] == 6.0

    def test_empty_turns(self):
        assert apply_merge_map([], {"spk_2": "spk_0"}) == []


# ── load / save merge map ────────────────────────────────────────────

class TestLoadSaveMergeMap:
    def test_load_missing_returns_empty(self, tmp_path):
        result = load_merge_map(str(tmp_path), "2026-01-01_12-00-00")
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        merge_map = {"spk_2": "spk_0", "spk_3": "spk_1"}
        path = save_merge_map(merge_map, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.name == "speaker_merge_2026-01-01_12-00-00.json"
        loaded = load_merge_map(str(tmp_path), "2026-01-01_12-00-00")
        assert loaded == merge_map


# ── CLI parsing ──────────────────────────────────────────────────────

class TestCLIParsing:
    def test_merge_repeatable(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--merge", "spk_2=spk_0",
            "--merge", "spk_3=spk_1",
        ])
        assert args.merge == ["spk_2=spk_0", "spk_3=spk_1"]

    def test_merge_default_empty(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.merge == []
