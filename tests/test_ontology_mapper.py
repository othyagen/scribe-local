"""Tests for lightweight deterministic ontology mapper."""

from __future__ import annotations

import json
import os

import pytest

from app.ontology_mapper import load_ontology_map, map_symptoms_to_concepts
from app.clinical_state import build_clinical_state


# ── helpers ─────────────────────────────────────────────────────────


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


def _sample_map() -> dict[str, dict]:
    return {
        "headache": {
            "system": "SNOMED",
            "code": "25064002",
            "display": "Headache",
            "type": "symptom",
        },
        "nausea": {
            "system": "SNOMED",
            "code": "422587007",
            "display": "Nausea",
            "type": "symptom",
        },
        "chest pain": {
            "system": "SNOMED",
            "code": "29857009",
            "display": "Chest pain",
            "type": "symptom",
        },
    }


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ══════════════════════════════════════════════════════════════════
# load_ontology_map
# ══════════════════════════════════════════════════════════════════


class TestLoadOntologyMap:
    def test_load_default_file(self):
        omap = load_ontology_map()
        assert isinstance(omap, dict)
        assert len(omap) > 0
        assert "headache" in omap
        assert omap["headache"]["code"] == "25064002"

    def test_load_custom_file(self, tmp_path):
        data = {"fever": {"system": "SNOMED", "code": "386661006",
                          "display": "Fever", "type": "symptom"}}
        f = tmp_path / "custom.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        omap = load_ontology_map(str(f))
        assert "fever" in omap
        assert len(omap) == 1

    def test_missing_file_returns_empty(self):
        omap = load_ontology_map("/nonexistent/path.json")
        assert omap == {}

    def test_invalid_json_returns_empty(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not valid json{{{", encoding="utf-8")
        omap = load_ontology_map(str(f))
        assert omap == {}
        assert "WARNING" in capsys.readouterr().err

    def test_non_object_json_returns_empty(self, tmp_path, capsys):
        f = tmp_path / "array.json"
        f.write_text('["a", "b"]', encoding="utf-8")
        omap = load_ontology_map(str(f))
        assert omap == {}
        assert "WARNING" in capsys.readouterr().err

    def test_keys_lowercased(self, tmp_path):
        data = {"Headache": {"system": "SNOMED", "code": "25064002",
                             "display": "Headache", "type": "symptom"}}
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        omap = load_ontology_map(str(f))
        assert "headache" in omap
        assert "Headache" not in omap

    def test_shipped_file_covers_all_symptoms(self):
        omap = load_ontology_map()
        symptoms_file = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "resources", "extractors", "symptoms.json",
        )
        with open(symptoms_file, encoding="utf-8") as f:
            symptoms = json.load(f)
        for sym in symptoms:
            assert sym.lower() in omap, f"Missing ontology entry for: {sym}"


# ══════════════════════════════════════════════════════════════════
# map_symptoms_to_concepts
# ══════════════════════════════════════════════════════════════════


class TestMapSymptomsToConcepts:
    def test_known_symptom_maps(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["headache"]), _sample_map(),
        )
        assert len(concepts) == 1
        assert concepts[0]["text"] == "headache"
        assert concepts[0]["code"] == "25064002"
        assert concepts[0]["system"] == "SNOMED"
        assert concepts[0]["display"] == "Headache"
        assert concepts[0]["type"] == "symptom"

    def test_multiple_symptoms_preserve_order(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["nausea", "headache", "chest pain"]),
            _sample_map(),
        )
        assert [c["text"] for c in concepts] == ["nausea", "headache", "chest pain"]

    def test_case_insensitive(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["Headache"]), _sample_map(),
        )
        assert len(concepts) == 1
        assert concepts[0]["code"] == "25064002"

    def test_unknown_symptom_skipped(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["headache", "alien syndrome"]),
            _sample_map(),
        )
        assert len(concepts) == 1
        assert concepts[0]["text"] == "headache"

    def test_duplicate_symptoms_no_duplicate_concepts(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["headache", "headache"]),
            _sample_map(),
        )
        assert len(concepts) == 1

    def test_empty_symptoms(self):
        concepts = map_symptoms_to_concepts(_state(), _sample_map())
        assert concepts == []

    def test_empty_map(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["headache"]), {},
        )
        assert concepts == []

    def test_default_map_loads_when_none(self):
        concepts = map_symptoms_to_concepts(
            _state(symptoms=["headache"]),
        )
        assert len(concepts) == 1
        assert concepts[0]["code"] == "25064002"


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state(symptoms=["headache", "nausea"])
        omap = _sample_map()
        r1 = map_symptoms_to_concepts(state, omap)
        r2 = map_symptoms_to_concepts(state, omap)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_ontology_concepts_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "ontology_concepts" in state["derived"]
        concepts = state["derived"]["ontology_concepts"]
        assert isinstance(concepts, list)
        assert len(concepts) >= 1
        assert concepts[0]["code"] == "25064002"

    def test_ontology_concepts_empty_when_no_symptoms(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["derived"]["ontology_concepts"] == []

    def test_multiple_symptoms_mapped(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        concepts = state["derived"]["ontology_concepts"]
        codes = [c["code"] for c in concepts]
        assert "25064002" in codes  # headache
        assert "422587007" in codes  # nausea

    def test_structured_data_unchanged(self):
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ])
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert isinstance(state["derived"]["problem_representation"], dict)
        assert isinstance(state["derived"]["symptom_representations"], list)
        assert isinstance(state["derived"]["problem_summary"], str)

    def test_compatible_with_ai_overlay(self):
        from app.llm_overlay import apply_ai_overlay

        state = build_clinical_state([_seg("patient has headache.")])
        original_concepts = list(state["derived"]["ontology_concepts"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["ontology_concepts"] == original_concepts
        assert "ai_overlay" in state["derived"]
