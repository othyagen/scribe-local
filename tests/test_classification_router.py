"""Tests for classification router and ICPC mapper."""

from __future__ import annotations

import json

import pytest

from app.classification_router import apply_classification
from app.icpc_mapper import load_icpc_map, suggest_icpc_codes
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
    }
    base.update(overrides)
    return base


def _dict_config(enabled: bool = True, system: str = "icpc") -> dict:
    """Build a plain dict config for the router."""
    return {"classification": {"enabled": enabled, "system": system}}


def _dataclass_config(enabled: bool = True, system: str = "icpc"):
    """Build a config object with ClassificationConfig attribute."""
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.classification = ClassificationConfig(enabled=enabled, system=system)
    return cfg


# ══════════════════════════════════════════════════════════════════
# load_icpc_map
# ══════════════════════════════════════════════════════════════════


class TestLoadIcpcMap:
    def test_load_default_file(self):
        icpc = load_icpc_map()
        assert isinstance(icpc, dict)
        assert len(icpc) > 0
        assert "headache" in icpc
        assert icpc["headache"]["code"] == "N01"

    def test_load_custom_file(self, tmp_path):
        data = {"fever": {"code": "A03", "label": "Fever", "kind": "symptom"}}
        f = tmp_path / "custom.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        icpc = load_icpc_map(str(f))
        assert "fever" in icpc
        assert len(icpc) == 1

    def test_missing_file_returns_empty(self):
        assert load_icpc_map("/nonexistent/path.json") == {}

    def test_invalid_json_returns_empty(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not valid json{{{", encoding="utf-8")
        icpc = load_icpc_map(str(f))
        assert icpc == {}
        assert "WARNING" in capsys.readouterr().err

    def test_non_object_json_returns_empty(self, tmp_path, capsys):
        f = tmp_path / "arr.json"
        f.write_text('["a", "b"]', encoding="utf-8")
        icpc = load_icpc_map(str(f))
        assert icpc == {}
        assert "WARNING" in capsys.readouterr().err

    def test_keys_lowercased(self, tmp_path):
        data = {"Headache": {"code": "N01", "label": "Headache", "kind": "symptom"}}
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        icpc = load_icpc_map(str(f))
        assert "headache" in icpc
        assert "Headache" not in icpc

    def test_comment_keys_excluded(self):
        icpc = load_icpc_map()
        assert not any(k.startswith("_") for k in icpc)


# ══════════════════════════════════════════════════════════════════
# suggest_icpc_codes
# ══════════════════════════════════════════════════════════════════


class TestSuggestIcpcCodes:
    def test_known_symptom(self):
        codes = suggest_icpc_codes(_state(symptoms=["headache"]))
        assert len(codes) == 1
        assert codes[0]["code"] == "N01"
        assert codes[0]["label"] == "Headache"
        assert codes[0]["kind"] == "symptom"
        assert codes[0]["evidence"] == ["headache"]

    def test_multiple_symptoms(self):
        codes = suggest_icpc_codes(
            _state(symptoms=["headache", "nausea", "fever"]),
        )
        assert len(codes) == 3
        code_list = [c["code"] for c in codes]
        assert "N01" in code_list
        assert "D09" in code_list
        assert "A03" in code_list

    def test_preserves_order(self):
        codes = suggest_icpc_codes(
            _state(symptoms=["fever", "headache"]),
        )
        assert codes[0]["code"] == "A03"
        assert codes[1]["code"] == "N01"

    def test_case_insensitive(self):
        codes = suggest_icpc_codes(_state(symptoms=["Headache"]))
        assert len(codes) == 1
        assert codes[0]["code"] == "N01"

    def test_unknown_symptom_skipped(self):
        codes = suggest_icpc_codes(
            _state(symptoms=["headache", "alien syndrome"]),
        )
        assert len(codes) == 1

    def test_duplicate_codes_deduplicated(self):
        # dyspnea and shortness of breath both map to R02
        codes = suggest_icpc_codes(
            _state(symptoms=["dyspnea", "shortness of breath"]),
        )
        assert len(codes) == 1
        assert codes[0]["code"] == "R02"

    def test_empty_symptoms(self):
        assert suggest_icpc_codes(_state()) == []

    def test_empty_map(self):
        assert suggest_icpc_codes(_state(symptoms=["headache"]), {}) == []


# ══════════════════════════════════════════════════════════════════
# Classification disabled
# ══════════════════════════════════════════════════════════════════


class TestClassificationDisabled:
    def test_none_config(self):
        result = apply_classification(_state(symptoms=["headache"]), None)
        assert result == {}

    def test_enabled_false(self):
        cfg = _dict_config(enabled=False, system="icpc")
        result = apply_classification(_state(symptoms=["headache"]), cfg)
        assert result == {}

    def test_system_none(self):
        cfg = _dict_config(enabled=True, system="none")
        result = apply_classification(_state(symptoms=["headache"]), cfg)
        assert result == {}


# ══════════════════════════════════════════════════════════════════
# ICPC selected
# ══════════════════════════════════════════════════════════════════


class TestIcpcSelected:
    def test_icpc_via_dict_config(self):
        cfg = _dict_config(enabled=True, system="icpc")
        result = apply_classification(
            _state(symptoms=["headache"]), cfg,
        )
        assert result["system"] == "ICPC"
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["code"] == "N01"

    def test_icpc_via_dataclass_config(self):
        cfg = _dataclass_config(enabled=True, system="icpc")
        result = apply_classification(
            _state(symptoms=["headache"]), cfg,
        )
        assert result["system"] == "ICPC"
        assert len(result["suggestions"]) >= 1

    def test_icpc_empty_when_no_symptoms(self):
        cfg = _dict_config(enabled=True, system="icpc")
        result = apply_classification(_state(), cfg)
        assert result["system"] == "ICPC"
        assert result["suggestions"] == []


# ══════════════════════════════════════════════════════════════════
# Unknown / future systems
# ══════════════════════════════════════════════════════════════════


class TestUnknownSystems:
    def test_unknown_system_returns_empty(self):
        cfg = _dict_config(enabled=True, system="foobar")
        result = apply_classification(_state(symptoms=["headache"]), cfg)
        assert result == {}

    def test_icd10_produces_suggestions(self):
        cfg = _dict_config(enabled=True, system="icd10")
        result = apply_classification(_state(symptoms=["headache"]), cfg)
        assert result["system"] == "ICD-10"
        assert len(result["suggestions"]) >= 1

    def test_icd11_placeholder(self):
        cfg = _dict_config(enabled=True, system="icd11")
        result = apply_classification(_state(symptoms=["headache"]), cfg)
        assert result["system"] == "ICD-11"
        assert result["suggestions"] == []


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state(symptoms=["headache", "nausea", "fever"])
        cfg = _dict_config(enabled=True, system="icpc")
        r1 = apply_classification(state, cfg)
        r2 = apply_classification(state, cfg)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_no_classification_by_default(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "classification" not in state["derived"]

    def test_classification_with_config(self):
        cfg = _dataclass_config(enabled=True, system="icpc")
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        assert "classification" in state["derived"]
        cls = state["derived"]["classification"]
        assert cls["system"] == "ICPC"
        assert len(cls["suggestions"]) >= 1
        assert cls["suggestions"][0]["code"] == "N01"

    def test_classification_disabled_no_key(self):
        cfg = _dataclass_config(enabled=False)
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        assert "classification" not in state["derived"]

    def test_structured_data_unchanged(self):
        cfg = _dataclass_config(enabled=True, system="icpc")
        state = build_clinical_state(
            [_seg("patient has headache and nausea.")], config=cfg,
        )
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert isinstance(state["derived"]["problem_representation"], dict)
        assert isinstance(state["derived"]["symptom_representations"], list)
        assert isinstance(state["derived"]["problem_summary"], str)
        assert isinstance(state["derived"]["ontology_concepts"], list)
        assert isinstance(state["derived"]["clinical_patterns"], list)
        assert isinstance(state["derived"]["running_summary"], dict)
        assert isinstance(state["derived"]["normalized_timeline"], list)
        assert isinstance(state["derived"]["temporal_context"], dict)
        assert isinstance(state["derived"]["red_flags"], list)

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        cfg = _dataclass_config(enabled=True, system="icpc")
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

    def test_empty_input_no_classification(self):
        state = build_clinical_state([_seg("hello.")])
        assert "classification" not in state["derived"]
