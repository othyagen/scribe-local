"""Tests for session export bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.export_bundle import export_session_bundle, _discover_session_files


TS = "2026-03-01_10-00-00"


def _create_session_files(output_dir: Path, ts: str) -> dict[str, Path]:
    """Create a set of typical session output files. Returns {name: path}."""
    files = {}

    # RAW transcript
    p = output_dir / f"raw_{ts}_large-v3.json"
    p.write_text('{"seg_id":"seg_0001"}\n', encoding="utf-8")
    files["raw_json"] = p

    p = output_dir / f"raw_{ts}_large-v3.txt"
    p.write_text("hello world\n", encoding="utf-8")
    files["raw_txt"] = p

    # Normalized transcript
    p = output_dir / f"normalized_{ts}_large-v3.json"
    p.write_text(json.dumps([{"seg_id": "seg_0001"}]), encoding="utf-8")
    files["norm_json"] = p

    p = output_dir / f"normalized_{ts}_large-v3.txt"
    p.write_text("hello world\n", encoding="utf-8")
    files["norm_txt"] = p

    # Confidence report
    p = output_dir / f"confidence_report_{ts}.json"
    p.write_text(json.dumps({"segments": []}), encoding="utf-8")
    files["confidence"] = p

    # Session report
    p = output_dir / f"session_report_{ts}.json"
    p.write_text(json.dumps({"config": {}}), encoding="utf-8")
    files["report"] = p

    # Clinical notes (multiple templates)
    p = output_dir / f"clinical_note_{ts}_soap.md"
    p.write_text("# SOAP Note\n", encoding="utf-8")
    files["note_soap"] = p

    p = output_dir / f"clinical_note_{ts}_summary.md"
    p.write_text("# Summary Note\n", encoding="utf-8")
    files["note_summary"] = p

    return files


class TestExportDirectoryBundle:
    def test_creates_bundle_directory(self, tmp_path):
        _create_session_files(tmp_path, TS)
        result = export_session_bundle(TS, tmp_path)
        assert result.is_dir()
        assert result.name == f"session_{TS}"

    def test_bundle_contains_expected_files(self, tmp_path):
        _create_session_files(tmp_path, TS)
        result = export_session_bundle(TS, tmp_path)
        names = {f.name for f in result.iterdir()}
        assert "raw_transcript.json" in names
        assert "raw_transcript.txt" in names
        assert "normalized_transcript.json" in names
        assert "normalized_transcript.txt" in names
        assert "confidence_report.json" in names
        assert "session_report.json" in names
        assert "clinical_note_soap.md" in names
        assert "clinical_note_summary.md" in names

    def test_file_contents_preserved(self, tmp_path):
        _create_session_files(tmp_path, TS)
        result = export_session_bundle(TS, tmp_path)
        content = (result / "raw_transcript.json").read_text("utf-8")
        assert "seg_0001" in content

    def test_original_files_unchanged(self, tmp_path):
        files = _create_session_files(tmp_path, TS)
        original_content = files["raw_json"].read_text("utf-8")
        export_session_bundle(TS, tmp_path)
        assert files["raw_json"].read_text("utf-8") == original_content


class TestExportZipBundle:
    def test_creates_zip_file(self, tmp_path):
        _create_session_files(tmp_path, TS)
        result = export_session_bundle(TS, tmp_path, zip_output=True)
        assert result.suffix == ".zip"
        assert result.exists()

    def test_zip_contains_expected_files(self, tmp_path):
        _create_session_files(tmp_path, TS)
        result = export_session_bundle(TS, tmp_path, zip_output=True)
        with zipfile.ZipFile(result) as zf:
            names = set(zf.namelist())
        assert "raw_transcript.json" in names
        assert "clinical_note_soap.md" in names
        assert "session_report.json" in names

    def test_zip_cleans_up_temp_directory(self, tmp_path):
        _create_session_files(tmp_path, TS)
        export_session_bundle(TS, tmp_path, zip_output=True)
        bundle_dir = tmp_path / f"session_{TS}"
        assert not bundle_dir.exists()

    def test_zip_file_contents_preserved(self, tmp_path):
        _create_session_files(tmp_path, TS)
        result = export_session_bundle(TS, tmp_path, zip_output=True)
        with zipfile.ZipFile(result) as zf:
            content = zf.read("raw_transcript.json").decode("utf-8")
        assert "seg_0001" in content


class TestMissingFiles:
    def test_missing_optional_files_not_included(self, tmp_path):
        """Bundle with only raw + normalized (no confidence, no notes)."""
        p = tmp_path / f"raw_{TS}_large-v3.json"
        p.write_text('{"seg_id":"seg_0001"}\n', encoding="utf-8")
        result = export_session_bundle(TS, tmp_path)
        names = {f.name for f in result.iterdir()}
        assert "raw_transcript.json" in names
        assert "confidence_report.json" not in names
        assert "session_report.json" not in names

    def test_session_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no session files found"):
            export_session_bundle("2099-01-01_00-00-00", tmp_path)


class TestFileRenaming:
    def test_clinical_note_keeps_template_id(self, tmp_path):
        p = tmp_path / f"clinical_note_{TS}_soap.md"
        p.write_text("# SOAP\n", encoding="utf-8")
        result = export_session_bundle(TS, tmp_path)
        assert (result / "clinical_note_soap.md").exists()

    def test_raw_transcript_simplified(self, tmp_path):
        p = tmp_path / f"raw_{TS}_large-v3.json"
        p.write_text("{}\n", encoding="utf-8")
        result = export_session_bundle(TS, tmp_path)
        assert (result / "raw_transcript.json").exists()

    def test_audio_wav_simplified(self, tmp_path):
        p = tmp_path / f"audio_{TS}.wav"
        p.write_bytes(b"\x00" * 100)
        result = export_session_bundle(TS, tmp_path)
        assert (result / "audio.wav").exists()


class TestDiscoverSessionFiles:
    def test_empty_directory(self, tmp_path):
        files = _discover_session_files(tmp_path, TS)
        assert files == []

    def test_discovers_all_patterns(self, tmp_path):
        _create_session_files(tmp_path, TS)
        files = _discover_session_files(tmp_path, TS)
        simple_names = [name for _, name in files]
        assert "raw_transcript.json" in simple_names
        assert "normalized_transcript.json" in simple_names
        assert "confidence_report.json" in simple_names
        assert "clinical_note_soap.md" in simple_names
        assert "clinical_note_summary.md" in simple_names
