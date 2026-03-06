"""Tests for --reprocess-all batch reprocessing."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from app.config import AppConfig
from app.main import (
    ReprocessError,
    _discover_all_sessions,
    _reprocess_all,
    _reprocess_session,
)


def _seg_dict(
    seg_id: str = "seg_0001",
    t0: float = 0.0,
    t1: float = 1.5,
    speaker: str = "spk_0",
    raw_text: str = "hello world",
    model_name: str = "large-v3",
    lang: str = "en",
    paragraph_id: str = "para_0001",
) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker,
        "raw_text": raw_text,
        "model_name": model_name,
        "language": lang,
        "paragraph_id": paragraph_id,
    }


def _write_raw(path: Path, segments: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def _make_config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    cfg.reporting.session_report_enabled = False
    return cfg


def _make_args(**overrides) -> types.SimpleNamespace:
    defaults = dict(
        export_srt=False,
        export_vtt=False,
        export_summary=False,
        export_clinical_note=False,
        export_notes=None,
        auto_tags="none",
        note_template="soap",
        template_alias=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestDiscoverAllSessions:
    def test_finds_raw_files_sorted_oldest_first(self, tmp_path):
        _write_raw(tmp_path / "raw_2026-03-01_10-00-00_large-v3.json",
                    [_seg_dict()])
        _write_raw(tmp_path / "raw_2026-03-03_10-00-00_large-v3.json",
                    [_seg_dict()])
        _write_raw(tmp_path / "raw_2026-03-02_10-00-00_large-v3.json",
                    [_seg_dict()])

        result = _discover_all_sessions(str(tmp_path))
        timestamps = [ts for ts, _ in result]
        assert timestamps == [
            "2026-03-01_10-00-00",
            "2026-03-02_10-00-00",
            "2026-03-03_10-00-00",
        ]

    def test_empty_dir(self, tmp_path):
        assert _discover_all_sessions(str(tmp_path)) == []

    def test_ignores_non_raw_files(self, tmp_path):
        (tmp_path / "normalized_2026-03-01_10-00-00_large-v3.json").write_text("{}")
        assert _discover_all_sessions(str(tmp_path)) == []


class TestReprocessAll:
    def test_discovers_and_reprocesses_sessions(self, tmp_path, capsys):
        for day in ("01", "02", "03"):
            ts = f"2026-03-{day}_10-00-00"
            _write_raw(
                tmp_path / f"raw_{ts}_large-v3.json",
                [_seg_dict(t0=0.0, t1=1.0), _seg_dict(seg_id="seg_0002", t0=1.5, t1=3.0)],
            )

        cfg = _make_config(tmp_path)
        args = _make_args()
        _reprocess_all(cfg, args)

        captured = capsys.readouterr().out
        assert "Found 3 sessions." in captured
        assert "OK" in captured
        assert "Succeeded: 3" in captured
        assert "Failed:    0" in captured

        # Normalized files created for each session
        for day in ("01", "02", "03"):
            ts = f"2026-03-{day}_10-00-00"
            norm = tmp_path / f"normalized_{ts}_large-v3.json"
            assert norm.exists()
            data = json.loads(norm.read_text("utf-8"))
            assert len(data) == 2

    def test_continues_after_failure(self, tmp_path, capsys):
        # Session 1: corrupt RAW (invalid JSON)
        ts1 = "2026-03-01_10-00-00"
        (tmp_path / f"raw_{ts1}_large-v3.json").write_text("NOT VALID JSON\n")

        # Session 2: valid RAW
        ts2 = "2026-03-02_10-00-00"
        _write_raw(
            tmp_path / f"raw_{ts2}_large-v3.json",
            [_seg_dict()],
        )

        cfg = _make_config(tmp_path)
        args = _make_args()
        _reprocess_all(cfg, args)

        captured = capsys.readouterr().out
        assert "FAILED" in captured
        assert "OK" in captured
        assert "Succeeded: 1" in captured
        assert "Failed:    1" in captured

        # Valid session was still processed
        norm = tmp_path / f"normalized_{ts2}_large-v3.json"
        assert norm.exists()

    def test_corrupt_raw_counted_as_failed(self, tmp_path, capsys):
        ts = "2026-03-01_10-00-00"
        (tmp_path / f"raw_{ts}_large-v3.json").write_text("{bad json\n")

        cfg = _make_config(tmp_path)
        args = _make_args()
        _reprocess_all(cfg, args)

        captured = capsys.readouterr().out
        assert "Found 1 sessions." in captured
        assert "FAILED" in captured
        assert "Failed:    1" in captured
        assert "Succeeded: 0" in captured

    def test_empty_raw_is_failed(self, tmp_path, capsys):
        ts = "2026-03-01_10-00-00"
        # Empty file — no valid segments
        (tmp_path / f"raw_{ts}_large-v3.json").write_text("")

        cfg = _make_config(tmp_path)
        args = _make_args()
        _reprocess_all(cfg, args)

        captured = capsys.readouterr().out
        assert "FAILED" in captured
        assert "Failed:    1" in captured

    def test_summary_counts(self, tmp_path, capsys):
        # 2 valid, 1 corrupt
        for day, valid in [("01", True), ("02", False), ("03", True)]:
            ts = f"2026-03-{day}_10-00-00"
            path = tmp_path / f"raw_{ts}_large-v3.json"
            if valid:
                _write_raw(path, [_seg_dict()])
            else:
                path.write_text("CORRUPT\n")

        cfg = _make_config(tmp_path)
        args = _make_args()
        _reprocess_all(cfg, args)

        captured = capsys.readouterr().out
        assert "Found:     3" in captured
        assert "Succeeded: 2" in captured
        assert "Failed:    1" in captured

    def test_no_sessions_found(self, tmp_path, capsys):
        cfg = _make_config(tmp_path)
        args = _make_args()
        _reprocess_all(cfg, args)

        captured = capsys.readouterr().out
        assert "Found 0 sessions." in captured
        assert "Found:     0" in captured
        assert "Succeeded: 0" in captured
        assert "Failed:    0" in captured


class TestReprocessErrorHandling:
    def test_reprocess_session_raises_on_no_raw(self, tmp_path):
        cfg = _make_config(tmp_path)
        args = _make_args()
        with pytest.raises(ReprocessError, match="no RAW file found"):
            _reprocess_session(cfg, args, "2026-03-01_10-00-00")

    def test_reprocess_session_raises_on_empty_raw(self, tmp_path):
        ts = "2026-03-01_10-00-00"
        (tmp_path / f"raw_{ts}_large-v3.json").write_text("")

        cfg = _make_config(tmp_path)
        args = _make_args()
        with pytest.raises(ReprocessError, match="no valid segments"):
            _reprocess_session(cfg, args, ts)
