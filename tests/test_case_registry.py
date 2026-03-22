"""Tests for case registry — building, filtering, and resolution."""

from __future__ import annotations

import pytest
from pathlib import Path

from app.case_registry import (
    CaseEntry,
    build_registry,
    filter_registry,
    resolve_case,
)


# ── helpers ────────────────────────────────────────────────────────


def _write_case(tmp_path: Path, case_id: str, **kwargs) -> Path:
    """Write a minimal YAML case file and return its path."""
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


def _entry(**overrides) -> CaseEntry:
    """Build a CaseEntry with sensible defaults."""
    defaults = {
        "case_id": "test_01",
        "path": Path("fake.yaml"),
        "title": "",
        "origin": "unknown",
        "difficulty": "unspecified",
        "organ_systems": [],
        "presenting_complaints": [],
        "tags": [],
        "patient_age": None,
        "patient_sex": None,
        "icd10_codes": [],
        "icpc_codes": [],
        "snomed_codes": [],
        "has_ground_truth": False,
        "segment_count": 0,
    }
    defaults.update(overrides)
    return CaseEntry(**defaults)


# ── build_registry ────────────────────────────────────────────────


class TestBuildRegistry:
    def test_scans_yaml_files(self, tmp_path):
        _write_case(tmp_path, "case_a")
        _write_case(tmp_path, "case_b")
        entries = build_registry([tmp_path])
        assert len(entries) == 2
        ids = [e.case_id for e in entries]
        assert "case_a" in ids
        assert "case_b" in ids

    def test_sorted_by_case_id(self, tmp_path):
        _write_case(tmp_path, "zebra")
        _write_case(tmp_path, "alpha")
        entries = build_registry([tmp_path])
        assert entries[0].case_id == "alpha"
        assert entries[1].case_id == "zebra"

    def test_ignores_non_yaml(self, tmp_path):
        _write_case(tmp_path, "good")
        (tmp_path / "readme.txt").write_text("not a case", encoding="utf-8")
        entries = build_registry([tmp_path])
        assert len(entries) == 1

    def test_skips_invalid_yaml(self, tmp_path):
        _write_case(tmp_path, "good")
        (tmp_path / "bad.yaml").write_text("{{invalid", encoding="utf-8")
        entries = build_registry([tmp_path])
        assert len(entries) == 1

    def test_empty_dir(self, tmp_path):
        entries = build_registry([tmp_path])
        assert entries == []

    def test_nonexistent_dir(self):
        entries = build_registry([Path("/nonexistent/dir")])
        assert entries == []

    def test_multiple_dirs(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        _write_case(d1, "from_d1")
        _write_case(d2, "from_d2")
        entries = build_registry([d1, d2])
        assert len(entries) == 2

    def test_metadata_extracted(self, tmp_path):
        _write_case(
            tmp_path, "rich_case",
            title="Rich",
            meta={"tags": ["cardiac"], "difficulty": "hard"},
            provenance={"origin": "synthetic", "created": "2026-01-01"},
        )
        entries = build_registry([tmp_path])
        assert entries[0].title == "Rich"
        assert entries[0].origin == "synthetic"
        assert entries[0].difficulty == "hard"
        assert entries[0].tags == ["cardiac"]

    def test_seed_cases(self):
        """Build registry from actual seed cases directory."""
        seed_dir = Path(__file__).resolve().parent.parent / "resources" / "cases"
        if not seed_dir.is_dir():
            pytest.skip("seed cases not found")
        entries = build_registry([seed_dir])
        assert len(entries) >= 1
        ids = [e.case_id for e in entries]
        assert "chest_pain_01" in ids or "pneumonia_01" in ids


# ── filter_registry ───────────────────────────────────────────────


class TestFilterRegistry:
    def test_filter_by_tag_exact(self):
        entries = [
            _entry(case_id="a", tags=["cardiac"]),
            _entry(case_id="b", tags=["respiratory"]),
        ]
        result = filter_registry(entries, tag="cardiac")
        assert len(result) == 1
        assert result[0].case_id == "a"

    def test_filter_by_tag_case_insensitive(self):
        entries = [_entry(case_id="a", tags=["Cardiac"])]
        result = filter_registry(entries, tag="cardiac")
        assert len(result) == 1

    def test_filter_by_tag_no_substring(self):
        """'card' should NOT match 'cardiac' — exact only."""
        entries = [_entry(case_id="a", tags=["cardiac"])]
        result = filter_registry(entries, tag="card")
        assert len(result) == 0

    def test_filter_by_organ_system_exact(self):
        entries = [
            _entry(case_id="a", organ_systems=["cardiovascular"]),
            _entry(case_id="b", organ_systems=["respiratory"]),
        ]
        result = filter_registry(entries, organ_system="cardiovascular")
        assert len(result) == 1

    def test_filter_by_organ_system_no_substring(self):
        entries = [_entry(case_id="a", organ_systems=["cardiovascular"])]
        result = filter_registry(entries, organ_system="cardio")
        assert len(result) == 0

    def test_filter_by_complaint_exact(self):
        entries = [_entry(case_id="a", presenting_complaints=["chest pain"])]
        result = filter_registry(entries, complaint="chest pain")
        assert len(result) == 1

    def test_filter_by_complaint_case_insensitive(self):
        entries = [_entry(case_id="a", presenting_complaints=["Chest Pain"])]
        result = filter_registry(entries, complaint="chest pain")
        assert len(result) == 1

    def test_filter_by_origin_exact(self):
        entries = [
            _entry(case_id="a", origin="synthetic"),
            _entry(case_id="b", origin="synthea"),
        ]
        result = filter_registry(entries, origin="synthetic")
        assert len(result) == 1
        assert result[0].case_id == "a"

    def test_filter_by_difficulty(self):
        entries = [
            _entry(case_id="a", difficulty="easy"),
            _entry(case_id="b", difficulty="hard"),
        ]
        result = filter_registry(entries, difficulty="easy")
        assert len(result) == 1

    def test_filter_icd_exact_code(self):
        entries = [
            _entry(case_id="a", icd10_codes=["I20.9"]),
            _entry(case_id="b", icd10_codes=["J18.9"]),
        ]
        result = filter_registry(entries, icd="I20.9")
        assert len(result) == 1
        assert result[0].case_id == "a"

    def test_filter_icd_normalized(self):
        """Code match strips whitespace and casefolds."""
        entries = [_entry(case_id="a", icd10_codes=["I20.9"])]
        result = filter_registry(entries, icd=" i20.9 ")
        assert len(result) == 1

    def test_filter_icd_no_substring(self):
        """'I20' should NOT match 'I20.9' — exact only."""
        entries = [_entry(case_id="a", icd10_codes=["I20.9"])]
        result = filter_registry(entries, icd="I20")
        assert len(result) == 0

    def test_filter_icpc_exact(self):
        entries = [_entry(case_id="a", icpc_codes=["K74"])]
        result = filter_registry(entries, icpc="K74")
        assert len(result) == 1

    def test_filter_snomed_exact(self):
        entries = [_entry(case_id="a", snomed_codes=["194828000"])]
        result = filter_registry(entries, snomed="194828000")
        assert len(result) == 1

    def test_filter_and_combination(self):
        entries = [
            _entry(case_id="a", tags=["cardiac"], origin="synthetic"),
            _entry(case_id="b", tags=["cardiac"], origin="synthea"),
            _entry(case_id="c", tags=["respiratory"], origin="synthetic"),
        ]
        result = filter_registry(entries, tag="cardiac", origin="synthetic")
        assert len(result) == 1
        assert result[0].case_id == "a"

    def test_filter_no_matches(self):
        entries = [_entry(case_id="a", tags=["cardiac"])]
        result = filter_registry(entries, tag="neuro")
        assert len(result) == 0

    def test_filter_no_criteria_returns_all(self):
        entries = [_entry(case_id="a"), _entry(case_id="b")]
        result = filter_registry(entries)
        assert len(result) == 2


# ── resolve_case ──────────────────────────────────────────────────


class TestResolveCase:
    def test_resolve_by_path(self, tmp_path):
        p = _write_case(tmp_path, "test_01")
        resolved = resolve_case(str(p), [tmp_path])
        assert resolved == p

    def test_resolve_by_case_id(self, tmp_path):
        _write_case(tmp_path, "target_case")
        resolved = resolve_case("target_case", [tmp_path])
        assert resolved.name == "target_case.yaml"

    def test_resolve_path_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_case("nonexistent.yaml", [])

    def test_resolve_case_id_not_found(self, tmp_path):
        _write_case(tmp_path, "other")
        with pytest.raises(ValueError, match="no case found"):
            resolve_case("missing", [tmp_path])

    def test_resolve_duplicate_case_id(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        _write_case(d1, "dup")
        _write_case(d2, "dup")
        with pytest.raises(ValueError, match="multiple cases"):
            resolve_case("dup", [d1, d2])

    def test_resolve_path_with_slash(self, tmp_path):
        p = _write_case(tmp_path, "slashed")
        # Use forward-slash path to trigger path mode.
        resolved = resolve_case(str(p).replace("\\", "/"), [])
        assert resolved.exists()

    def test_resolve_exact_id_not_substring(self, tmp_path):
        _write_case(tmp_path, "chest_pain_01")
        _write_case(tmp_path, "chest_pain_02")
        resolved = resolve_case("chest_pain_01", [tmp_path])
        assert "chest_pain_01" in resolved.name


# ── determinism ───────────────────────────────────────────────────


class TestDeterminism:
    def test_build_registry_deterministic(self, tmp_path):
        _write_case(tmp_path, "a")
        _write_case(tmp_path, "b")
        r1 = build_registry([tmp_path])
        r2 = build_registry([tmp_path])
        assert [e.case_id for e in r1] == [e.case_id for e in r2]

    def test_filter_deterministic(self):
        entries = [
            _entry(case_id="a", tags=["x"]),
            _entry(case_id="b", tags=["y"]),
        ]
        r1 = filter_registry(entries, tag="x")
        r2 = filter_registry(entries, tag="x")
        assert [e.case_id for e in r1] == [e.case_id for e in r2]
