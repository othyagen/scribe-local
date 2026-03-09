"""Tests for deterministic ICD-10 mapper and router integration."""

from __future__ import annotations

import json

import pytest

from app.icd_mapper import load_icd_map, load_icd_pattern_map, suggest_icd10_codes
from app.classification_router import apply_classification
from app.clinical_state import build_clinical_state
from app.config import ClassificationConfig


# ── helpers ─────────────────────────────────────────────────────────


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _state(**overrides) -> dict:
    base: dict = {
        "symptoms": [],
        "qualifiers": [],
        "timeline": [],
        "durations": [],
        "negations": [],
        "medications": [],
        "diagnostic_hints": [],
        "derived": {
            "clinical_patterns": [],
            "symptom_representations": [],
        },
    }
    base.update(overrides)
    return base


def _state_with_patterns(symptoms: list[str], patterns: list[dict]) -> dict:
    s = _state(symptoms=symptoms)
    s["derived"]["clinical_patterns"] = patterns
    return s


def _dict_config(enabled: bool = True, system: str = "icd10") -> dict:
    return {"classification": {"enabled": enabled, "system": system}}


def _dataclass_config(enabled: bool = True, system: str = "icd10"):
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.classification = ClassificationConfig(enabled=enabled, system=system)
    return cfg


# ══════════════════════════════════════════════════════════════════
# load_icd_map
# ══════════════════════════════════════════════════════════════════


class TestLoadIcdMap:
    def test_load_default_file(self):
        icd = load_icd_map()
        assert isinstance(icd, dict)
        assert len(icd) > 0
        assert "headache" in icd
        assert icd["headache"]["code"] == "R51"

    def test_load_custom_file(self, tmp_path):
        data = {"fever": {"code": "R50.9", "label": "Fever", "kind": "symptom"}}
        f = tmp_path / "custom.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        icd = load_icd_map(str(f))
        assert "fever" in icd
        assert len(icd) == 1

    def test_missing_file_returns_empty(self):
        assert load_icd_map("/nonexistent/path.json") == {}

    def test_invalid_json_returns_empty(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not valid json{{{", encoding="utf-8")
        icd = load_icd_map(str(f))
        assert icd == {}
        assert "WARNING" in capsys.readouterr().err

    def test_comment_keys_excluded(self):
        icd = load_icd_map()
        assert not any(k.startswith("_") for k in icd)

    def test_keys_lowercased(self, tmp_path):
        data = {"Headache": {"code": "R51", "label": "Headache", "kind": "symptom"}}
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        icd = load_icd_map(str(f))
        assert "headache" in icd
        assert "Headache" not in icd


class TestLoadIcdPatternMap:
    def test_load_default_file(self):
        pmap = load_icd_pattern_map()
        assert isinstance(pmap, dict)
        assert "migraine_like" in pmap
        assert pmap["migraine_like"]["code"] == "G43.909"

    def test_missing_file_returns_empty(self):
        assert load_icd_pattern_map("/nonexistent/path.json") == {}


# ══════════════════════════════════════════════════════════════════
# Symptom-level mapping
# ══════════════════════════════════════════════════════════════════


class TestSymptomLevel:
    def test_single_symptom(self):
        codes = suggest_icd10_codes(_state(symptoms=["headache"]))
        assert len(codes) == 1
        assert codes[0]["code"] == "R51"
        assert codes[0]["label"] == "Headache"
        assert codes[0]["kind"] == "symptom"
        assert codes[0]["evidence"] == ["headache"]

    def test_multiple_symptoms(self):
        codes = suggest_icd10_codes(
            _state(symptoms=["headache", "nausea", "cough"]),
        )
        assert len(codes) == 3
        code_list = [c["code"] for c in codes]
        assert "R51" in code_list
        assert "R11.0" in code_list
        assert "R05" in code_list

    def test_preserves_order(self):
        codes = suggest_icd10_codes(
            _state(symptoms=["cough", "headache"]),
        )
        assert codes[0]["code"] == "R05"
        assert codes[1]["code"] == "R51"

    def test_case_insensitive(self):
        codes = suggest_icd10_codes(_state(symptoms=["Headache"]))
        assert len(codes) == 1
        assert codes[0]["code"] == "R51"

    def test_unknown_symptom_skipped(self):
        codes = suggest_icd10_codes(
            _state(symptoms=["headache", "alien syndrome"]),
        )
        assert len(codes) == 1

    def test_duplicate_codes_deduplicated(self):
        # dyspnea and shortness of breath both map to R06.00
        codes = suggest_icd10_codes(
            _state(symptoms=["dyspnea", "shortness of breath"]),
        )
        assert len(codes) == 1
        assert codes[0]["code"] == "R06.00"

    def test_empty_symptoms(self):
        assert suggest_icd10_codes(_state()) == []

    def test_empty_map(self):
        assert suggest_icd10_codes(_state(symptoms=["headache"]), {}, {}) == []

    def test_key_symptom_codes(self):
        """Verify the core ICD-10 codes from the spec."""
        icd = load_icd_map()
        assert icd["headache"]["code"] == "R51"
        assert icd["nausea"]["code"] == "R11.0"
        assert icd["cough"]["code"] == "R05"
        assert icd["chest pain"]["code"] == "R07.9"
        assert icd["dyspnea"]["code"] == "R06.00"
        assert icd["diarrhea"]["code"] == "R19.7"
        assert icd["dysuria"]["code"] == "R30.0"
        assert icd["hematuria"]["code"] == "R31.9"
        assert icd["hemoptysis"]["code"] == "R04.2"


# ══════════════════════════════════════════════════════════════════
# Pattern-level mapping (conservative)
# ══════════════════════════════════════════════════════════════════


class TestPatternLevel:
    def test_migraine_pattern_suggested(self):
        state = _state_with_patterns(
            symptoms=["headache", "nausea"],
            patterns=[{
                "pattern": "migraine_like",
                "label": "Migraine-like pattern",
                "evidence": ["headache", "nausea"],
            }],
        )
        codes = suggest_icd10_codes(state)
        code_list = [c["code"] for c in codes]
        # Symptom-level codes first
        assert "R51" in code_list
        assert "R11.0" in code_list
        # Pattern-level migraine code added
        assert "G43.909" in code_list
        pattern_entry = [c for c in codes if c["code"] == "G43.909"][0]
        assert pattern_entry["kind"] == "pattern"
        assert "headache" in pattern_entry["evidence"]
        assert "nausea" in pattern_entry["evidence"]

    def test_unmapped_pattern_uses_symptom_fallback(self):
        """Patterns without a mapping file entry fall back to symptom codes."""
        state = _state_with_patterns(
            symptoms=["chest pain", "dyspnea"],
            patterns=[{
                "pattern": "lower_respiratory_pattern",
                "label": "Lower respiratory pattern",
                "evidence": ["cough", "fever", "dyspnea"],
            }],
        )
        codes = suggest_icd10_codes(state)
        code_list = [c["code"] for c in codes]
        # Symptom codes present
        assert "R07.9" in code_list  # chest pain
        assert "R06.00" in code_list  # dyspnea
        # No pattern-level code for lower_respiratory_pattern
        kinds = [c["kind"] for c in codes]
        assert "pattern" not in kinds

    def test_angina_like_no_pattern_mapping(self):
        """Angina-like uses symptom fallback — no pattern mapping."""
        state = _state_with_patterns(
            symptoms=["chest pain"],
            patterns=[{
                "pattern": "angina_like",
                "label": "Angina-like pattern",
                "evidence": ["chest pain", "aggravating factor: exertion",
                             "relieving factor: rest"],
            }],
        )
        codes = suggest_icd10_codes(state)
        code_list = [c["code"] for c in codes]
        assert "R07.9" in code_list  # symptom-level chest pain
        kinds = [c["kind"] for c in codes]
        assert "pattern" not in kinds

    def test_pattern_code_not_duplicated_with_symptom(self):
        """If a pattern code were the same as a symptom code, no duplicate."""
        # Create a custom pattern map where pattern maps to same code as a symptom
        custom_pmap = {"migraine_like": {"code": "R51", "label": "Headache", "kind": "pattern"}}
        state = _state_with_patterns(
            symptoms=["headache"],
            patterns=[{
                "pattern": "migraine_like",
                "label": "Migraine-like",
                "evidence": ["headache", "nausea"],
            }],
        )
        codes = suggest_icd10_codes(state, pattern_map=custom_pmap)
        r51_entries = [c for c in codes if c["code"] == "R51"]
        assert len(r51_entries) == 1

    def test_no_patterns_no_pattern_codes(self):
        state = _state(symptoms=["headache"])
        codes = suggest_icd10_codes(state)
        kinds = [c["kind"] for c in codes]
        assert "pattern" not in kinds


# ══════════════════════════════════════════════════════════════════
# No duplicates
# ══════════════════════════════════════════════════════════════════


class TestNoDuplicates:
    def test_no_duplicate_codes(self):
        codes = suggest_icd10_codes(
            _state(symptoms=["headache", "nausea", "fever", "cough"]),
        )
        code_list = [c["code"] for c in codes]
        assert len(code_list) == len(set(code_list))


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state_with_patterns(
            symptoms=["headache", "nausea"],
            patterns=[{
                "pattern": "migraine_like",
                "label": "Migraine-like",
                "evidence": ["headache", "nausea"],
            }],
        )
        r1 = suggest_icd10_codes(state)
        r2 = suggest_icd10_codes(state)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Classification router integration
# ══════════════════════════════════════════════════════════════════


class TestRouterIntegration:
    def test_icd10_via_dict_config(self):
        cfg = _dict_config(enabled=True, system="icd10")
        result = apply_classification(
            _state(symptoms=["headache"]), cfg,
        )
        assert result["system"] == "ICD-10"
        assert len(result["suggestions"]) >= 1
        assert result["suggestions"][0]["code"] == "R51"

    def test_icd10_empty_when_no_symptoms(self):
        cfg = _dict_config(enabled=True, system="icd10")
        result = apply_classification(_state(), cfg)
        assert result["system"] == "ICD-10"
        assert result["suggestions"] == []

    def test_icd10_not_placeholder_anymore(self):
        """ICD-10 should now produce real suggestions, not empty placeholder."""
        cfg = _dict_config(enabled=True, system="icd10")
        result = apply_classification(
            _state(symptoms=["headache"]), cfg,
        )
        assert len(result["suggestions"]) > 0


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestClinicalStateIntegration:
    def test_icd10_classification_in_derived(self):
        cfg = _dataclass_config(enabled=True, system="icd10")
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        assert "classification" in state["derived"]
        cls = state["derived"]["classification"]
        assert cls["system"] == "ICD-10"
        assert len(cls["suggestions"]) >= 1
        assert cls["suggestions"][0]["code"] == "R51"

    def test_migraine_pattern_in_clinical_state(self):
        cfg = _dataclass_config(enabled=True, system="icd10")
        state = build_clinical_state(
            [_seg("patient has headache and nausea.")], config=cfg,
        )
        cls = state["derived"]["classification"]
        code_list = [c["code"] for c in cls["suggestions"]]
        assert "R51" in code_list       # headache symptom
        assert "R11.0" in code_list     # nausea symptom
        assert "G43.909" in code_list   # migraine pattern

    def test_structured_data_unchanged(self):
        cfg = _dataclass_config(enabled=True, system="icd10")
        state = build_clinical_state(
            [_seg("patient has headache and nausea.")], config=cfg,
        )
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert isinstance(state["derived"]["problem_representation"], dict)
        assert isinstance(state["derived"]["symptom_representations"], list)
        assert isinstance(state["derived"]["ontology_concepts"], list)
        assert isinstance(state["derived"]["clinical_patterns"], list)
        assert isinstance(state["derived"]["red_flags"], list)
        assert isinstance(state["derived"]["temporal_context"], dict)

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        cfg = _dataclass_config(enabled=True, system="icd10")
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        original_cls = dict(state["derived"]["classification"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["classification"] == original_cls
        assert "ai_overlay" in state["derived"]

    def test_no_classification_by_default(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "classification" not in state["derived"]

    def test_empty_input(self):
        cfg = _dataclass_config(enabled=True, system="icd10")
        state = build_clinical_state([_seg("hello.")], config=cfg)
        cls = state["derived"]["classification"]
        assert cls["system"] == "ICD-10"
        assert cls["suggestions"] == []
