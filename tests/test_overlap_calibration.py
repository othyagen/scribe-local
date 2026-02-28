"""Tests for overlap detection, prototype-based matching, and UNKNOWN policy."""

from __future__ import annotations

import copy
import math

import pytest

from app.calibration import (
    MIN_PROTOTYPE_DURATION_SEC,
    assign_clusters_to_profile,
    apply_cluster_override,
    build_cluster_embeddings_with_stats,
    detect_and_mark_overlap,
)


# ── helpers ──────────────────────────────────────────────────────────

def _turn(start: float, end: float, speaker: str, **extra) -> dict:
    d = {"start": start, "end": end, "speaker": speaker}
    d.update(extra)
    return d


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


# ── detect_and_mark_overlap ─────────────────────────────────────────

class TestDetectAndMarkOverlap:
    def test_no_overlap(self):
        turns = [
            _turn(0.0, 1.0, "spk_0"),
            _turn(1.0, 2.0, "spk_1"),
            _turn(3.0, 4.0, "spk_0"),
        ]
        result = detect_and_mark_overlap(turns)
        assert len(result) == 3
        assert all(t["overlap"] is False for t in result)
        assert all("overlap_with" not in t for t in result)

    def test_basic_overlap(self):
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(1.5, 3.0, "spk_1"),
        ]
        result = detect_and_mark_overlap(turns)
        assert result[0]["overlap"] is True
        assert result[0]["overlap_with"] == "spk_1"
        assert result[1]["overlap"] is True
        assert result[1]["overlap_with"] == "spk_0"

    def test_chain_overlap(self):
        """A overlaps B, B overlaps C — all three marked."""
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(1.5, 3.5, "spk_1"),
            _turn(3.0, 5.0, "spk_2"),
        ]
        result = detect_and_mark_overlap(turns)
        assert all(t["overlap"] is True for t in result)

    def test_non_mutating(self):
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(1.5, 3.0, "spk_1"),
        ]
        original = copy.deepcopy(turns)
        detect_and_mark_overlap(turns)
        assert turns == original


# ── min duration prototype filtering ────────────────────────────────

class TestMinDurationPrototype:
    def test_short_turns_excluded(self):
        """Turn < 1.2s with embedding is excluded from prototype."""
        emb = _l2([1.0, 0.0, 0.0])
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=emb),   # 1.0s < 1.2s
            _turn(2.0, 4.0, "spk_0", embedding=emb),   # 2.0s >= 1.2s
        ]
        cluster_embs, stats = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=MIN_PROTOTYPE_DURATION_SEC,
        )
        assert "spk_0" in cluster_embs
        assert stats["spk_0"]["embedded_turn_count"] == 1
        assert stats["spk_0"]["embedded_total_sec"] == 2.0

    def test_long_turns_included(self):
        """Turns >= 1.2s are included in prototype."""
        emb_a = _l2([1.0, 0.0, 0.0])
        emb_b = _l2([0.0, 1.0, 0.0])
        turns = [
            _turn(0.0, 2.0, "spk_0", embedding=emb_a),
            _turn(3.0, 5.0, "spk_0", embedding=emb_b),
        ]
        cluster_embs, stats = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=MIN_PROTOTYPE_DURATION_SEC,
        )
        assert stats["spk_0"]["embedded_turn_count"] == 2


# ── prototype with overlap filtering ────────────────────────────────

class TestPrototypeWithOverlap:
    def test_overlap_excluded_from_prototype(self):
        """Overlap turns with embeddings are excluded from prototype."""
        emb_good = _l2([1.0, 0.0, 0.0])
        emb_bad = _l2([0.0, 0.0, 1.0])
        turns = [
            _turn(0.0, 3.0, "spk_0", embedding=emb_good, overlap=False),
            _turn(5.0, 8.0, "spk_0", embedding=emb_bad, overlap=True,
                  overlap_with="spk_1"),
        ]
        cluster_embs, stats = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=0.0,
        )
        assert stats["spk_0"]["embedded_turn_count"] == 1
        # Prototype should match the good embedding, not the bad one
        assert cluster_embs["spk_0"][0] == pytest.approx(emb_good[0], abs=1e-6)

    def test_mean_and_normalize(self):
        """Two valid embeddings produce L2-normalized mean prototype."""
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]
        turns = [
            _turn(0.0, 3.0, "spk_0", embedding=emb_a, overlap=False),
            _turn(4.0, 7.0, "spk_0", embedding=emb_b, overlap=False),
        ]
        cluster_embs, _ = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=0.0,
        )
        proto = cluster_embs["spk_0"]
        # Mean of [1,0,0] and [0,1,0] = [0.5, 0.5, 0], L2-normalized
        norm = math.sqrt(0.5**2 + 0.5**2)
        assert proto[0] == pytest.approx(0.5 / norm, abs=1e-6)
        assert proto[1] == pytest.approx(0.5 / norm, abs=1e-6)
        assert proto[2] == pytest.approx(0.0, abs=1e-6)


# ── cluster mapping & many-to-one safeguard ─────────────────────────

class TestClusterMapping:
    def test_many_to_one_safeguard(self):
        """Two clusters closest to same profile speaker: only higher wins."""
        profile = {
            "speakers": {
                "Alice": {"embedding": _l2([1.0, 0.0, 0.0])},
            },
            "speaker_id_map": {"Alice": "spk_0"},
        }
        # spk_0 closer to Alice than spk_1
        cluster_embs = {
            "spk_0": _l2([0.95, 0.05, 0.0]),
            "spk_1": _l2([0.80, 0.20, 0.0]),
        }
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.5, margin=0.0,
        )
        # Only one cluster can be assigned (greedy 1:1)
        assert len(mapping) == 1
        assert "spk_0" in mapping

        # Apply override — spk_1 should become UNKNOWN
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(3.0, 5.0, "spk_1"),
        ]
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0", "spk_1"},
        )
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "UNKNOWN"

    def test_threshold_below_gets_unknown(self):
        """Cluster below threshold gets UNKNOWN."""
        profile = {
            "speakers": {
                "Alice": {"embedding": _l2([1.0, 0.0, 0.0])},
            },
            "speaker_id_map": {"Alice": "spk_0"},
        }
        # Very dissimilar
        cluster_embs = {"spk_0": _l2([0.0, 0.0, 1.0])}
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.9, margin=0.0,
        )
        assert mapping == {}

        turns = [_turn(0.0, 2.0, "spk_0")]
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0"},
        )
        assert result[0]["speaker"] == "UNKNOWN"


# ── UNKNOWN fallback ────────────────────────────────────────────────

class TestUnknownFallback:
    def test_no_valid_embeddings_gets_unknown(self):
        """Cluster with only overlap/short turns has no prototype → UNKNOWN."""
        turns = [
            _turn(0.0, 0.5, "spk_0", overlap=True, overlap_with="spk_1"),
            _turn(0.3, 0.8, "spk_1", overlap=True, overlap_with="spk_0"),
        ]
        cluster_embs, stats = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=MIN_PROTOTYPE_DURATION_SEC,
        )
        # No prototypes — both turns are overlap + short
        assert cluster_embs == {}

        # If we still try to override with empty mapping and eligible IDs
        # containing the original speakers, non-overlap turns would get UNKNOWN.
        # But these are overlap turns, so they keep original speaker_id.
        result = apply_cluster_override(
            turns, {},
            eligible_cluster_ids={"spk_0", "spk_1"},
        )
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"

    def test_overlap_turns_keep_original_speaker(self):
        """Overlap turns never get UNKNOWN even when cluster is unassigned."""
        turns = [
            _turn(0.0, 2.0, "spk_0", overlap=True, overlap_with="spk_1"),
            _turn(3.0, 5.0, "spk_0", overlap=False),
        ]
        # spk_0 is eligible but not in mapping
        result = apply_cluster_override(
            turns, {},
            eligible_cluster_ids={"spk_0"},
        )
        # Overlap turn keeps spk_0, non-overlap gets UNKNOWN
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "UNKNOWN"


# ── freeze rule ─────────────────────────────────────────────────────

class TestFreezeRule:
    def test_overlap_speaker_not_overridden(self):
        """Overlap turn keeps original speaker_id even when mapping exists."""
        turns = [
            _turn(0.0, 2.0, "spk_0", overlap=True, overlap_with="spk_1"),
        ]
        mapping = {"spk_0": "cal_0"}
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0"},
        )
        assert result[0]["speaker"] == "spk_0"

    def test_non_overlap_speaker_overridden(self):
        """Non-overlap turn in same cluster is overridden normally."""
        turns = [
            _turn(0.0, 2.0, "spk_0", overlap=True, overlap_with="spk_1"),
            _turn(3.0, 5.0, "spk_0", overlap=False),
        ]
        mapping = {"spk_0": "cal_0"}
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0"},
        )
        assert result[0]["speaker"] == "spk_0"   # frozen
        assert result[1]["speaker"] == "cal_0"    # overridden
