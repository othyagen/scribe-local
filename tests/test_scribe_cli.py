"""Tests for the scribe CLI entry point."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.scribe import build_parser, main


# ── helpers ────────────────────────────────────────────────────────


def _write_case(tmp_path: Path, case_id: str, **kwargs) -> Path:
    import yaml

    case = {
        "case_id": case_id,
        "segments": [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
             "speaker_id": "spk_0", "normalized_text": "Test."},
        ],
        **kwargs,
    }
    p = tmp_path / f"{case_id}.yaml"
    p.write_text(yaml.dump(case, default_flow_style=False), encoding="utf-8")
    return p


# ── parser ─────────────────────────────────────────────────────────


class TestParser:
    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_no_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1


# ── cases list ─────────────────────────────────────────────────────


class TestCasesList:
    def test_list_prints_output(self, tmp_path, capsys):
        _write_case(tmp_path, "test_case_01", title="Test Case")
        main(["--case-dir", str(tmp_path), "cases", "list"])
        out = capsys.readouterr().out
        assert "test_case_01" in out
        assert "1 case(s) found" in out

    def test_list_no_cases(self, tmp_path, capsys):
        main(["--case-dir", str(tmp_path), "cases", "list"])
        out = capsys.readouterr().out
        assert "No cases found" in out

    def test_list_with_tag_filter(self, tmp_path, capsys):
        _write_case(tmp_path, "cardiac_01", meta={"tags": ["cardiac"]})
        _write_case(tmp_path, "resp_01", meta={"tags": ["respiratory"]})
        main(["--case-dir", str(tmp_path), "cases", "list", "--tag", "cardiac"])
        out = capsys.readouterr().out
        assert "cardiac_01" in out
        assert "resp_01" not in out

    def test_list_with_origin_filter(self, tmp_path, capsys):
        _write_case(
            tmp_path, "syn_01",
            provenance={"origin": "synthetic", "created": "2026-01-01"},
        )
        _write_case(
            tmp_path, "imp_01",
            provenance={"origin": "imported", "created": "2026-01-01"},
        )
        main(["--case-dir", str(tmp_path), "cases", "list", "--origin", "synthetic"])
        out = capsys.readouterr().out
        assert "syn_01" in out
        assert "imp_01" not in out


# ── cases show ─────────────────────────────────────────────────────


class TestCasesShow:
    def test_show_by_path(self, tmp_path, capsys):
        p = _write_case(tmp_path, "show_me", title="Show Me")
        main(["cases", "show", str(p)])
        out = capsys.readouterr().out
        assert "show_me" in out
        assert "Show Me" in out

    def test_show_by_case_id(self, tmp_path, capsys):
        _write_case(tmp_path, "by_id", title="By ID")
        main(["--case-dir", str(tmp_path), "cases", "show", "by_id"])
        out = capsys.readouterr().out
        assert "by_id" in out

    def test_show_unknown_case_id(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main(["--case-dir", str(tmp_path), "cases", "show", "nonexistent"])
        assert exc_info.value.code == 1


# ── cases validate ─────────────────────────────────────────────────


class TestCasesValidate:
    def test_validate_valid_case(self, tmp_path, capsys):
        p = _write_case(
            tmp_path, "valid_01",
            provenance={"origin": "synthetic", "created": "2026-01-01"},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["cases", "validate", str(p)])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "VALID" in out

    def test_validate_prints_warnings(self, tmp_path, capsys):
        p = _write_case(tmp_path, "warn_01")
        with pytest.raises(SystemExit) as exc_info:
            main(["cases", "validate", str(p)])
        # Still valid (warnings only), but shows warnings.
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Warning" in out

    def test_validate_composed_output(self, tmp_path, capsys):
        """Validation composes core + extended schema results."""
        p = _write_case(
            tmp_path, "composed_01",
            provenance={"origin": "synthetic", "created": "2026-01-01"},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["cases", "validate", str(p)])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        # Should include extended schema warnings (missing classification/patient).
        assert "classification" in out or "patient" in out


# ── cases create ───────────────────────────────────────────────────


class TestCasesCreate:
    def test_create_prints_template(self, capsys):
        main(["cases", "create"])
        out = capsys.readouterr().out
        assert "case_id:" in out
        assert "segments:" in out
        assert "ground_truth:" in out


# ── run ────────────────────────────────────────────────────────────


class TestRun:
    def test_run_case(self, tmp_path, capsys):
        p = _write_case(
            tmp_path, "run_01",
            provenance={"origin": "synthetic", "created": "2026-01-01"},
        )
        main(["cases", "show", str(p)])  # Sanity check path works.
        capsys.readouterr()  # Clear.
        main(["run", str(p)])
        out = capsys.readouterr().out
        assert "run_01" in out

    def test_run_unknown_case(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main(["--case-dir", str(tmp_path), "run", "nonexistent"])
        assert exc_info.value.code == 1
