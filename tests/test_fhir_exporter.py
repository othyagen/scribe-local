"""Tests for lightweight deterministic FHIR export layer."""

from __future__ import annotations

import pytest

from app.fhir_exporter import build_fhir_bundle, _stable_id
from app.clinical_state import build_clinical_state
from app.config import ClassificationConfig, ExportConfig


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


def _fhir_config(**overrides):
    """Build a config object with FHIR export enabled."""
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.export = ExportConfig(fhir_enabled=True)
    cfg.classification = ClassificationConfig(
        enabled=overrides.get("cls_enabled", False),
        system=overrides.get("cls_system", "none"),
    )
    cfg.ai = type("_AiCfg", (), {"enabled": False})()
    return cfg


def _minimal_state(**overrides) -> dict:
    """Build a minimal clinical_state dict for direct exporter tests."""
    base: dict = {
        "symptoms": [],
        "qualifiers": [],
        "timeline": [],
        "durations": [],
        "negations": [],
        "medications": [],
        "diagnostic_hints": [],
        "derived": {
            "problem_representation": {},
            "problem_focus": None,
            "symptom_representations": [],
            "problem_summary": "",
            "ontology_concepts": [],
            "clinical_patterns": [],
            "running_summary": {},
            "normalized_timeline": [],
            "temporal_context": {},
            "red_flags": [],
        },
    }
    base.update(overrides)
    return base


def _state_with_symptoms(symptoms: list[str],
                         concepts: list[dict] | None = None,
                         reps: list[dict] | None = None,
                         pr: dict | None = None,
                         classification: dict | None = None,
                         summary: str = "",
                         running: dict | None = None,
                         red_flags: list[dict] | None = None) -> dict:
    """Build a clinical_state with populated derived fields."""
    state = _minimal_state(symptoms=symptoms)
    d = state["derived"]
    if concepts is not None:
        d["ontology_concepts"] = concepts
    if reps is not None:
        d["symptom_representations"] = reps
    if pr is not None:
        d["problem_representation"] = pr
    if classification is not None:
        d["classification"] = classification
    if summary:
        d["problem_summary"] = summary
    if running is not None:
        d["running_summary"] = running
    if red_flags is not None:
        d["red_flags"] = red_flags
    return state


# ══════════════════════════════════════════════════════════════════
# Stable ID generation
# ══════════════════════════════════════════════════════════════════


class TestStableId:
    def test_deterministic(self):
        assert _stable_id("Obs", "a") == _stable_id("Obs", "a")

    def test_different_input_different_id(self):
        assert _stable_id("Obs", "a") != _stable_id("Obs", "b")

    def test_different_resource_type_different_id(self):
        assert _stable_id("Obs", "a") != _stable_id("Cond", "a")

    def test_length(self):
        assert len(_stable_id("Encounter", "session")) == 16


# ══════════════════════════════════════════════════════════════════
# Bundle structure
# ══════════════════════════════════════════════════════════════════


class TestBundleStructure:
    def test_resource_type(self):
        bundle = build_fhir_bundle(_minimal_state())
        assert bundle["resourceType"] == "Bundle"

    def test_bundle_type(self):
        bundle = build_fhir_bundle(_minimal_state())
        assert bundle["type"] == "collection"

    def test_entries_is_list(self):
        bundle = build_fhir_bundle(_minimal_state())
        assert isinstance(bundle["entry"], list)

    def test_every_entry_has_resource(self):
        bundle = build_fhir_bundle(_minimal_state())
        for entry in bundle["entry"]:
            assert "resource" in entry

    def test_minimal_state_has_encounter_and_composition(self):
        """Even an empty state produces Encounter + Composition."""
        bundle = build_fhir_bundle(_minimal_state())
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Encounter" in types
        assert "Composition" in types

    def test_no_observations_for_empty_symptoms(self):
        bundle = build_fhir_bundle(_minimal_state())
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Observation" not in types

    def test_no_conditions_for_empty_state(self):
        bundle = build_fhir_bundle(_minimal_state())
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Condition" not in types


# ══════════════════════════════════════════════════════════════════
# Encounter
# ══════════════════════════════════════════════════════════════════


class TestEncounter:
    def test_encounter_resource_type(self):
        bundle = build_fhir_bundle(_minimal_state())
        enc = _find_resource(bundle, "Encounter")
        assert enc["resourceType"] == "Encounter"

    def test_encounter_status(self):
        enc = _find_resource(build_fhir_bundle(_minimal_state()), "Encounter")
        assert enc["status"] == "finished"

    def test_encounter_class(self):
        enc = _find_resource(build_fhir_bundle(_minimal_state()), "Encounter")
        assert enc["class"]["code"] == "AMB"

    def test_encounter_reason_when_summary(self):
        state = _state_with_symptoms(
            ["headache"],
            summary="Patient presents with headache.",
        )
        enc = _find_resource(build_fhir_bundle(state), "Encounter")
        assert "reasonCode" in enc
        assert enc["reasonCode"][0]["text"] == "Patient presents with headache."

    def test_encounter_no_reason_when_empty(self):
        enc = _find_resource(build_fhir_bundle(_minimal_state()), "Encounter")
        assert "reasonCode" not in enc

    def test_encounter_has_id(self):
        enc = _find_resource(build_fhir_bundle(_minimal_state()), "Encounter")
        assert isinstance(enc["id"], str)
        assert len(enc["id"]) > 0


# ══════════════════════════════════════════════════════════════════
# Observation
# ══════════════════════════════════════════════════════════════════


class TestObservation:
    def test_observation_from_ontology_concept(self):
        state = _state_with_symptoms(
            ["headache"],
            concepts=[{
                "text": "headache",
                "code": "25064002",
                "label": "Headache",
                "system": "http://snomed.info/sct",
            }],
        )
        bundle = build_fhir_bundle(state)
        obs = _find_resource(bundle, "Observation")
        assert obs is not None
        assert obs["code"]["text"] == "Headache"
        assert obs["code"]["coding"][0]["code"] == "25064002"

    def test_observation_text_only_symptom(self):
        """Symptoms not in ontology get text-only observations."""
        state = _state_with_symptoms(["alien syndrome"])
        bundle = build_fhir_bundle(state)
        obs = _find_resource(bundle, "Observation")
        assert obs is not None
        assert obs["code"]["text"] == "alien syndrome"
        assert "coding" not in obs["code"]

    def test_observation_status_preliminary(self):
        state = _state_with_symptoms(["headache"])
        obs = _find_resource(build_fhir_bundle(state), "Observation")
        assert obs["status"] == "preliminary"

    def test_multiple_observations(self):
        state = _state_with_symptoms(
            ["headache", "nausea"],
            concepts=[
                {"text": "headache", "code": "25064002",
                 "label": "Headache", "system": "http://snomed.info/sct"},
                {"text": "nausea", "code": "422587007",
                 "label": "Nausea", "system": "http://snomed.info/sct"},
            ],
        )
        bundle = build_fhir_bundle(state)
        obs_list = _find_all_resources(bundle, "Observation")
        assert len(obs_list) == 2

    def test_observation_no_duplicate_symptoms(self):
        """Symptom covered by ontology should not appear again as text-only."""
        state = _state_with_symptoms(
            ["headache"],
            concepts=[{
                "text": "headache",
                "code": "25064002",
                "label": "Headache",
                "system": "http://snomed.info/sct",
            }],
        )
        bundle = build_fhir_bundle(state)
        obs_list = _find_all_resources(bundle, "Observation")
        assert len(obs_list) == 1

    def test_observation_with_qualifiers(self):
        state = _state_with_symptoms(
            ["headache"],
            reps=[{
                "symptom": "headache",
                "severity": "moderate",
                "onset": "gradual",
                "duration": "3 days",
                "pattern": "throbbing",
                "progression": "worsening",
                "laterality": None,
                "radiation": None,
                "aggravating_factors": ["exertion"],
                "relieving_factors": ["rest"],
            }],
        )
        bundle = build_fhir_bundle(state)
        obs = _find_resource(bundle, "Observation")
        assert "component" in obs
        codes = [c["code"]["text"] for c in obs["component"]]
        assert "Severity" in codes
        assert "Onset" in codes
        assert "Duration" in codes
        assert "Pattern" in codes
        assert "Progression" in codes
        assert "Aggravating factor" in codes
        assert "Relieving factor" in codes

    def test_observation_no_components_when_no_qualifiers(self):
        state = _state_with_symptoms(["headache"])
        obs = _find_resource(build_fhir_bundle(state), "Observation")
        assert "component" not in obs

    def test_observation_has_stable_id(self):
        state = _state_with_symptoms(["headache"])
        obs = _find_resource(build_fhir_bundle(state), "Observation")
        assert isinstance(obs["id"], str)
        assert len(obs["id"]) == 16


# ══════════════════════════════════════════════════════════════════
# Condition
# ══════════════════════════════════════════════════════════════════


class TestCondition:
    def test_condition_from_core_symptom(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache", "severity": None, "onset": None},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert cond is not None
        assert cond["code"]["text"] == "headache"

    def test_condition_verification_provisional(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache"},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        vs = cond["verificationStatus"]["coding"][0]["code"]
        assert vs == "provisional"

    def test_condition_clinical_status_active(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache"},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        cs = cond["clinicalStatus"]["coding"][0]["code"]
        assert cs == "active"

    def test_condition_severity(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache", "severity": "severe"},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert cond["severity"]["text"] == "severe"

    def test_condition_no_severity_when_none(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache", "severity": None},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert "severity" not in cond

    def test_condition_onset(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache", "onset": "gradual"},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert cond["onsetString"] == "gradual"

    def test_condition_classification_as_note(self):
        """Classification suggestions appear as notes, not as coding."""
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache"},
            classification={
                "system": "ICPC",
                "suggestions": [
                    {"code": "N01", "label": "Headache", "kind": "symptom"},
                ],
            },
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert "note" in cond
        assert "ICPC suggestion: N01 Headache (symptom)" in cond["note"][0]["text"]

    def test_condition_no_note_without_classification(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache"},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert "note" not in cond

    def test_no_condition_without_core_symptom(self):
        bundle = build_fhir_bundle(_minimal_state())
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Condition" not in types

    def test_condition_has_stable_id(self):
        state = _state_with_symptoms(
            ["headache"],
            pr={"core_symptom": "headache"},
        )
        cond = _find_resource(build_fhir_bundle(state), "Condition")
        assert isinstance(cond["id"], str)
        assert len(cond["id"]) == 16


# ══════════════════════════════════════════════════════════════════
# Composition
# ══════════════════════════════════════════════════════════════════


class TestComposition:
    def test_composition_resource_type(self):
        comp = _find_resource(build_fhir_bundle(_minimal_state()), "Composition")
        assert comp["resourceType"] == "Composition"

    def test_composition_status(self):
        comp = _find_resource(build_fhir_bundle(_minimal_state()), "Composition")
        assert comp["status"] == "preliminary"

    def test_composition_type_loinc(self):
        comp = _find_resource(build_fhir_bundle(_minimal_state()), "Composition")
        assert comp["type"]["coding"][0]["code"] == "11488-4"

    def test_composition_title(self):
        comp = _find_resource(build_fhir_bundle(_minimal_state()), "Composition")
        assert comp["title"] == "SCRIBE Clinical Summary"

    def test_composition_chief_complaint_section(self):
        state = _state_with_symptoms(
            ["headache"],
            summary="Patient presents with headache.",
        )
        comp = _find_resource(build_fhir_bundle(state), "Composition")
        titles = [s["title"] for s in comp.get("section", [])]
        assert "Chief Complaint" in titles

    def test_composition_clinical_summary_section(self):
        state = _state_with_symptoms(
            ["headache"],
            running={
                "core_problem": "Headache noted.",
                "additional_symptoms": ["nausea"],
                "patterns_detected": [],
                "alerts": [],
            },
        )
        comp = _find_resource(build_fhir_bundle(state), "Composition")
        titles = [s["title"] for s in comp.get("section", [])]
        assert "Clinical Summary" in titles
        summary_sec = [s for s in comp["section"]
                       if s["title"] == "Clinical Summary"][0]
        assert "nausea" in summary_sec["text"]["div"]

    def test_composition_red_flags_section(self):
        state = _state_with_symptoms(
            ["headache"],
            red_flags=[{
                "flag": "sudden_severe_headache",
                "label": "Sudden severe headache",
                "severity": "high",
                "evidence": ["headache"],
            }],
        )
        comp = _find_resource(build_fhir_bundle(state), "Composition")
        titles = [s["title"] for s in comp.get("section", [])]
        assert "Red Flags" in titles

    def test_composition_classification_section(self):
        state = _state_with_symptoms(
            ["headache"],
            classification={
                "system": "ICPC",
                "suggestions": [
                    {"code": "N01", "label": "Headache"},
                ],
            },
        )
        comp = _find_resource(build_fhir_bundle(state), "Composition")
        titles = [s["title"] for s in comp.get("section", [])]
        assert any("Classification" in t for t in titles)

    def test_composition_no_sections_when_empty(self):
        comp = _find_resource(build_fhir_bundle(_minimal_state()), "Composition")
        assert "section" not in comp

    def test_composition_has_stable_id(self):
        comp = _find_resource(build_fhir_bundle(_minimal_state()), "Composition")
        assert isinstance(comp["id"], str)
        assert len(comp["id"]) == 16


# ══════════════════════════════════════════════════════════════════
# Empty / edge cases
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_completely_empty_state(self):
        bundle = build_fhir_bundle({})
        assert bundle["resourceType"] == "Bundle"
        assert isinstance(bundle["entry"], list)

    def test_missing_derived(self):
        bundle = build_fhir_bundle({"symptoms": ["headache"]})
        assert bundle["resourceType"] == "Bundle"

    def test_empty_symptoms_list(self):
        bundle = build_fhir_bundle(_minimal_state())
        obs_list = _find_all_resources(bundle, "Observation")
        assert obs_list == []

    def test_empty_ontology_concepts(self):
        state = _state_with_symptoms(["headache"])
        state["derived"]["ontology_concepts"] = []
        bundle = build_fhir_bundle(state)
        obs_list = _find_all_resources(bundle, "Observation")
        # Should still have text-only observation
        assert len(obs_list) == 1
        assert "coding" not in obs_list[0]["code"]


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _state_with_symptoms(
            ["headache", "nausea"],
            concepts=[
                {"text": "headache", "code": "25064002",
                 "label": "Headache", "system": "http://snomed.info/sct"},
            ],
            pr={"core_symptom": "headache"},
            summary="Patient presents with headache.",
        )
        b1 = build_fhir_bundle(state)
        b2 = build_fhir_bundle(state)
        assert b1 == b2

    def test_stable_ids_across_calls(self):
        state = _state_with_symptoms(["headache"])
        b1 = build_fhir_bundle(state)
        b2 = build_fhir_bundle(state)
        ids1 = [e["resource"]["id"] for e in b1["entry"]]
        ids2 = [e["resource"]["id"] for e in b2["entry"]]
        assert ids1 == ids2


# ══════════════════════════════════════════════════════════════════
# AI overlay compatibility
# ══════════════════════════════════════════════════════════════════


class TestAiOverlayCompatibility:
    def test_overlay_does_not_overwrite_fhir_bundle(self):
        from app.llm_overlay import apply_ai_overlay

        cfg = _fhir_config()
        state = build_clinical_state([_seg("patient has headache.")], config=cfg)
        original_bundle = dict(state["derived"]["fhir_bundle"])

        overlay = {
            "ai_overlay": {"soap_draft": "Draft."},
            "ai_overlay_meta": {"model": "test"},
        }
        apply_ai_overlay(state, overlay)
        assert state["derived"]["fhir_bundle"] == original_bundle
        assert "ai_overlay" in state["derived"]

    def test_fhir_bundle_present_when_enabled(self):
        cfg = _fhir_config()
        state = build_clinical_state([_seg("hello.")], config=cfg)
        assert "fhir_bundle" in state["derived"]
        assert state["derived"]["fhir_bundle"]["resourceType"] == "Bundle"

    def test_fhir_bundle_absent_when_disabled(self):
        state = build_clinical_state([_seg("hello.")])
        assert "fhir_bundle" not in state["derived"]


# ══════════════════════════════════════════════════════════════════
# Integration with clinical_state
# ══════════════════════════════════════════════════════════════════


class TestClinicalStateIntegration:
    def test_fhir_bundle_in_derived(self):
        cfg = _fhir_config()
        state = build_clinical_state([_seg("patient has headache.")], config=cfg)
        assert "fhir_bundle" in state["derived"]
        bundle = state["derived"]["fhir_bundle"]
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"

    def test_observations_from_real_symptoms(self):
        cfg = _fhir_config()
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ], config=cfg)
        bundle = state["derived"]["fhir_bundle"]
        obs_list = _find_all_resources(bundle, "Observation")
        obs_texts = [o["code"]["text"] for o in obs_list]
        assert any("headache" in t.lower() for t in obs_texts)
        assert any("nausea" in t.lower() for t in obs_texts)

    def test_condition_from_real_problem(self):
        cfg = _fhir_config()
        state = build_clinical_state([
            _seg("patient has headache for 3 days."),
        ], config=cfg)
        bundle = state["derived"]["fhir_bundle"]
        cond = _find_resource(bundle, "Condition")
        assert cond is not None
        assert "headache" in cond["code"]["text"].lower()

    def test_composition_has_sections_for_real_input(self):
        cfg = _fhir_config()
        state = build_clinical_state([
            _seg("patient has headache and nausea for 3 days."),
        ], config=cfg)
        bundle = state["derived"]["fhir_bundle"]
        comp = _find_resource(bundle, "Composition")
        assert "section" in comp
        assert len(comp["section"]) >= 1

    def test_empty_segments_still_produces_bundle(self):
        cfg = _fhir_config()
        state = build_clinical_state([], config=cfg)
        bundle = state["derived"]["fhir_bundle"]
        assert bundle["resourceType"] == "Bundle"
        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Encounter" in types
        assert "Composition" in types

    def test_fhir_does_not_modify_extraction_fields(self):
        cfg = _fhir_config()
        state = build_clinical_state([
            _seg("patient has headache and nausea."),
        ], config=cfg)
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert isinstance(state["derived"]["problem_representation"], dict)
        assert isinstance(state["derived"]["ontology_concepts"], list)
        assert isinstance(state["derived"]["red_flags"], list)

    def test_classification_in_fhir_when_configured(self):
        cfg = _fhir_config(cls_enabled=True, cls_system="icpc")

        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        bundle = state["derived"]["fhir_bundle"]
        cond = _find_resource(bundle, "Condition")
        if cond:
            assert "note" in cond
            assert "ICPC" in cond["note"][0]["text"]


# ── test utilities ──────────────────────────────────────────────


def _find_resource(bundle: dict, resource_type: str) -> dict | None:
    """Find the first resource of a given type in a bundle."""
    for entry in bundle.get("entry", []):
        if entry.get("resource", {}).get("resourceType") == resource_type:
            return entry["resource"]
    return None


def _find_all_resources(bundle: dict, resource_type: str) -> list[dict]:
    """Find all resources of a given type in a bundle."""
    return [
        entry["resource"]
        for entry in bundle.get("entry", [])
        if entry.get("resource", {}).get("resourceType") == resource_type
    ]
