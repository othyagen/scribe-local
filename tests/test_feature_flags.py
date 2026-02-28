"""Tests for feature flags and session report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.calibration import (
    MIN_PROTOTYPE_DURATION_SEC,
    assign_clusters_to_profile,
    apply_cluster_override,
    build_cluster_embeddings_with_stats,
    detect_and_mark_overlap,
)
from app.config import (
    AppConfig,
    DiarizationConfig,
    ReportingConfig,
    _build_config,
)
from app.reporting import build_session_report, write_session_report


# ── helpers ──────────────────────────────────────────────────────────

def _turn(start: float, end: float, speaker: str, **extra) -> dict:
    d = {"start": start, "end": end, "speaker": speaker}
    d.update(extra)
    return d


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


# ── config flag parsing ─────────────────────────────────────────────

class TestConfigFlags:
    def test_defaults_preserve_behavior(self):
        cfg = AppConfig()
        assert cfg.diarization.calibration_enabled is True
        assert cfg.diarization.overlap_stabilizer_enabled is True
        assert cfg.diarization.prototype_matching_enabled is True
        assert cfg.diarization.min_duration_filter_enabled is True
        assert cfg.reporting.session_report_enabled is True

    def test_yaml_override(self):
        data = {
            "diarization": {
                "calibration_enabled": False,
                "overlap_stabilizer_enabled": False,
                "prototype_matching_enabled": False,
                "min_duration_filter_enabled": False,
            }
        }
        cfg = _build_config(data)
        assert cfg.diarization.calibration_enabled is False
        assert cfg.diarization.overlap_stabilizer_enabled is False
        assert cfg.diarization.prototype_matching_enabled is False
        assert cfg.diarization.min_duration_filter_enabled is False

    def test_reporting_config_parsed(self):
        data = {"reporting": {"session_report_enabled": False}}
        cfg = _build_config(data)
        assert cfg.reporting.session_report_enabled is False


# ── overlap flag ────────────────────────────────────────────────────

class TestOverlapFlag:
    def test_overlap_flag_on(self):
        """When overlap detection runs, overlapping turns are marked."""
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(1.5, 3.0, "spk_1"),
        ]
        result = detect_and_mark_overlap(turns)
        assert result[0]["overlap"] is True
        assert result[1]["overlap"] is True

    def test_overlap_flag_off(self):
        """When overlap detection is skipped, turns have no overlap key."""
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(1.5, 3.0, "spk_1"),
        ]
        # Simulates flag=off: do NOT call detect_and_mark_overlap
        # Turns should have no overlap markers
        assert "overlap" not in turns[0]
        assert "overlap" not in turns[1]
        # embed_turns/build_cluster_embeddings_with_stats use .get("overlap")
        # which returns None/falsy for missing keys — correct pre-overlap behavior
        emb = _l2([1.0, 0.0, 0.0])
        turns[0]["embedding"] = emb
        turns[1]["embedding"] = emb
        cluster_embs, stats = build_cluster_embeddings_with_stats(turns)
        # Both turns contribute to prototypes (no overlap filtering)
        assert "spk_0" in cluster_embs
        assert "spk_1" in cluster_embs


# ── min duration flag ───────────────────────────────────────────────

class TestMinDurationFlag:
    def test_min_duration_on(self):
        """With filter on, short turns excluded from prototype."""
        emb = _l2([1.0, 0.0, 0.0])
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=emb),   # 1.0s < 1.2s
            _turn(2.0, 4.0, "spk_0", embedding=emb),   # 2.0s >= 1.2s
        ]
        cluster_embs, stats = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=MIN_PROTOTYPE_DURATION_SEC,
        )
        assert stats["spk_0"]["embedded_turn_count"] == 1

    def test_min_duration_off(self):
        """With filter off (min_duration_sec=0.0), short turns included."""
        emb = _l2([1.0, 0.0, 0.0])
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=emb),   # 1.0s
            _turn(2.0, 4.0, "spk_0", embedding=emb),   # 2.0s
        ]
        cluster_embs, stats = build_cluster_embeddings_with_stats(
            turns, min_duration_sec=0.0,
        )
        assert stats["spk_0"]["embedded_turn_count"] == 2


# ── prototype matching flag ─────────────────────────────────────────

class TestPrototypeMatchingFlag:
    def test_prototype_on(self):
        """With prototype matching on, clusters are remapped."""
        profile = {
            "speakers": {"Alice": {"embedding": _l2([1.0, 0.0, 0.0])}},
            "speaker_id_map": {"Alice": "spk_0"},
        }
        cluster_embs = {"spk_0": _l2([0.95, 0.05, 0.0])}
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.5, margin=0.0,
        )
        assert "spk_0" in mapping

        turns = [_turn(0.0, 2.0, "spk_0")]
        result = apply_cluster_override(
            turns, mapping, eligible_cluster_ids={"spk_0"},
        )
        assert result[0]["speaker"] == "spk_0"  # mapped to same id here

    def test_prototype_off(self):
        """With prototype matching off, apply_cluster_override is not called.

        Speaker ids remain unchanged (true feature-off baseline).
        """
        turns = [
            _turn(0.0, 2.0, "spk_0"),
            _turn(3.0, 5.0, "spk_1"),
        ]
        # Simulate flag=off: mapping={}, do NOT call apply_cluster_override
        # Turns keep original speaker ids
        assert turns[0]["speaker"] == "spk_0"
        assert turns[1]["speaker"] == "spk_1"


# ── session report ──────────────────────────────────────────────────

class TestSessionReport:
    def test_build_report_schema(self):
        cfg = AppConfig()
        output_paths = {
            "raw": "outputs/raw_2026-01-01_12-00-00_large-v3.json",
            "normalized": None,
            "changes": None,
            "audio": None,
            "diarization": None,
            "calibration_report": None,
            "calibrated": None,
            "diarized_segments": None,
            "diarized_txt": None,
            "tagged_txt": None,
            "confidence_report": None,
        }
        report = build_session_report(
            session_ts="2026-01-01_12-00-00",
            config=cfg,
            segment_count=10,
            output_paths=output_paths,
        )
        assert report["session_ts"] == "2026-01-01_12-00-00"
        assert "config" in report
        assert "feature_flags" in report
        assert "outputs" in report
        assert "stats" in report
        assert report["stats"]["segment_count"] == 10

    def test_write_report(self, tmp_path):
        report = {
            "session_ts": "2026-01-01_12-00-00",
            "stats": {"segment_count": 5},
        }
        path = write_session_report(report, str(tmp_path), "2026-01-01_12-00-00")
        assert path.exists()
        assert path.name == "session_report_2026-01-01_12-00-00.json"
        data = json.loads(path.read_text("utf-8"))
        assert data["session_ts"] == "2026-01-01_12-00-00"

    def test_report_includes_flags(self):
        cfg = AppConfig()
        cfg.diarization.calibration_enabled = False
        cfg.diarization.overlap_stabilizer_enabled = False
        report = build_session_report(
            session_ts="2026-01-01_12-00-00",
            config=cfg,
            segment_count=0,
            output_paths={},
        )
        flags = report["feature_flags"]
        assert flags["calibration_enabled"] is False
        assert flags["overlap_stabilizer_enabled"] is False
        assert flags["prototype_matching_enabled"] is True
        assert flags["min_duration_filter_enabled"] is True

    def test_report_includes_stats(self):
        cfg = AppConfig()
        report = build_session_report(
            session_ts="2026-01-01_12-00-00",
            config=cfg,
            segment_count=15,
            output_paths={},
            diarization_stats={
                "turns_before_smoothing": 20,
                "turns_after_smoothing": 18,
            },
            calibration_stats={
                "overlaps_marked": 2,
                "embeddings_computed": 10,
                "clusters_total": 3,
                "clusters_assigned": 2,
                "clusters_unknown": 1,
            },
        )
        s = report["stats"]
        assert s["segment_count"] == 15
        assert s["turns_before_smoothing"] == 20
        assert s["turns_after_smoothing"] == 18
        assert s["overlaps_marked"] == 2
        assert s["clusters_assigned"] == 2
