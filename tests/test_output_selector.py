"""Tests for the output selector layer."""

from __future__ import annotations

import pytest

from app.output_selector import (
    apply_optional_outputs,
    should_run_ai_overlay,
)
from app.clinical_state import build_clinical_state
from app.config import (
    AiConfig,
    AppConfig,
    ClassificationConfig,
    ExportConfig,
)


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


def _core_state(text: str = "patient has headache.") -> dict:
    """Build a clinical_state with only the deterministic core (no config)."""
    return build_clinical_state([_seg(text)])


def _dict_config(
    ai_enabled: bool = False,
    cls_enabled: bool = False,
    cls_system: str = "none",
    fhir_enabled: bool = False,
) -> dict:
    """Build a plain dict config for the output selector."""
    return {
        "ai": {"enabled": ai_enabled},
        "classification": {"enabled": cls_enabled, "system": cls_system},
        "export": {"fhir_enabled": fhir_enabled},
    }


def _dataclass_config(
    ai_enabled: bool = False,
    cls_enabled: bool = False,
    cls_system: str = "none",
    fhir_enabled: bool = False,
):
    """Build a config object with dataclass attributes."""
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.ai = AiConfig(enabled=ai_enabled)
    cfg.classification = ClassificationConfig(enabled=cls_enabled, system=cls_system)
    cfg.export = ExportConfig(fhir_enabled=fhir_enabled)
    return cfg


# ══════════════════════════════════════════════════════════════════
# All optional outputs disabled
# ══════════════════════════════════════════════════════════════════


class TestAllDisabled:
    def test_no_classification_key(self):
        state = _core_state()
        apply_optional_outputs(state, _dict_config())
        assert "classification" not in state["derived"]

    def test_no_fhir_bundle_key(self):
        state = _core_state()
        apply_optional_outputs(state, _dict_config())
        assert "fhir_bundle" not in state["derived"]

    def test_ai_overlay_not_requested(self):
        assert should_run_ai_overlay(_dict_config()) is False

    def test_none_config_disables_all(self):
        state = _core_state()
        apply_optional_outputs(state, None)
        assert "classification" not in state["derived"]
        assert "fhir_bundle" not in state["derived"]

    def test_core_derived_preserved(self):
        state = _core_state()
        apply_optional_outputs(state, _dict_config())
        assert "problem_representation" in state["derived"]
        assert "problem_summary" in state["derived"]
        assert "ontology_concepts" in state["derived"]
        assert "clinical_patterns" in state["derived"]
        assert "red_flags" in state["derived"]
        assert "running_summary" in state["derived"]
        assert "temporal_context" in state["derived"]


# ══════════════════════════════════════════════════════════════════
# Classification only enabled
# ══════════════════════════════════════════════════════════════════


class TestClassificationOnly:
    def test_icpc_classification_present(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="icpc")
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert state["derived"]["classification"]["system"] == "ICPC"

    def test_icd10_classification_present(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="icd10")
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert state["derived"]["classification"]["system"] == "ICD-10"

    def test_no_fhir_when_only_classification(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="icpc")
        apply_optional_outputs(state, cfg)
        assert "fhir_bundle" not in state["derived"]

    def test_classification_disabled_when_system_none(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="none")
        apply_optional_outputs(state, cfg)
        assert "classification" not in state["derived"]

    def test_classification_disabled_when_enabled_false(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=False, cls_system="icpc")
        apply_optional_outputs(state, cfg)
        assert "classification" not in state["derived"]

    def test_dataclass_config_classification(self):
        state = _core_state()
        cfg = _dataclass_config(cls_enabled=True, cls_system="icpc")
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert state["derived"]["classification"]["system"] == "ICPC"


# ══════════════════════════════════════════════════════════════════
# FHIR only enabled
# ══════════════════════════════════════════════════════════════════


class TestFhirOnly:
    def test_fhir_bundle_present(self):
        state = _core_state()
        cfg = _dict_config(fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert "fhir_bundle" in state["derived"]
        assert state["derived"]["fhir_bundle"]["resourceType"] == "Bundle"

    def test_no_classification_when_only_fhir(self):
        state = _core_state()
        cfg = _dict_config(fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert "classification" not in state["derived"]

    def test_fhir_disabled_by_default(self):
        state = _core_state()
        cfg = _dict_config(fhir_enabled=False)
        apply_optional_outputs(state, cfg)
        assert "fhir_bundle" not in state["derived"]

    def test_dataclass_config_fhir(self):
        state = _core_state()
        cfg = _dataclass_config(fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert "fhir_bundle" in state["derived"]


# ══════════════════════════════════════════════════════════════════
# AI overlay check
# ══════════════════════════════════════════════════════════════════


class TestAiOverlay:
    def test_ai_enabled_detected(self):
        cfg = _dict_config(ai_enabled=True)
        assert should_run_ai_overlay(cfg) is True

    def test_ai_disabled_detected(self):
        cfg = _dict_config(ai_enabled=False)
        assert should_run_ai_overlay(cfg) is False

    def test_ai_none_config(self):
        assert should_run_ai_overlay(None) is False

    def test_ai_dataclass_config_enabled(self):
        cfg = _dataclass_config(ai_enabled=True)
        assert should_run_ai_overlay(cfg) is True

    def test_ai_dataclass_config_disabled(self):
        cfg = _dataclass_config(ai_enabled=False)
        assert should_run_ai_overlay(cfg) is False


# ══════════════════════════════════════════════════════════════════
# Multiple optional outputs enabled together
# ══════════════════════════════════════════════════════════════════


class TestMultipleEnabled:
    def test_classification_and_fhir(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" in state["derived"]

    def test_fhir_sees_classification(self):
        """FHIR runs after classification, so it can include classification data."""
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        bundle = state["derived"]["fhir_bundle"]
        # Composition should mention classification
        comp = _find_resource(bundle, "Composition")
        section_titles = [s["title"] for s in comp.get("section", [])]
        assert any("Classification" in t for t in section_titles)

    def test_all_three_flags(self):
        cfg = _dict_config(
            ai_enabled=True, cls_enabled=True,
            cls_system="icd10", fhir_enabled=True,
        )
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" in state["derived"]
        assert should_run_ai_overlay(cfg) is True

    def test_classification_and_ai_no_fhir(self):
        cfg = _dict_config(
            ai_enabled=True, cls_enabled=True,
            cls_system="icpc", fhir_enabled=False,
        )
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" not in state["derived"]
        assert should_run_ai_overlay(cfg) is True

    def test_dataclass_config_all_enabled(self):
        cfg = _dataclass_config(
            ai_enabled=True, cls_enabled=True,
            cls_system="icpc", fhir_enabled=True,
        )
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" in state["derived"]
        assert should_run_ai_overlay(cfg) is True


# ══════════════════════════════════════════════════════════════════
# Deterministic core unchanged
# ══════════════════════════════════════════════════════════════════


class TestCoreUnchanged:
    def test_symptoms_preserved(self):
        state = _core_state("patient has headache and nausea.")
        cfg = _dict_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]

    def test_derived_core_fields_intact(self):
        state = _core_state()
        original_pr = dict(state["derived"]["problem_representation"])
        original_flags = list(state["derived"]["red_flags"])
        cfg = _dict_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert state["derived"]["problem_representation"] == original_pr
        assert state["derived"]["red_flags"] == original_flags

    def test_extraction_fields_not_mutated(self):
        state = _core_state("patient has headache for 3 days.")
        original_symptoms = list(state["symptoms"])
        original_durations = list(state["durations"])
        cfg = _dict_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        apply_optional_outputs(state, cfg)
        assert state["symptoms"] == original_symptoms
        assert state["durations"] == original_durations


# ══════════════════════════════════════════════════════════════════
# Unknown classification system
# ══════════════════════════════════════════════════════════════════


class TestUnknownSystems:
    def test_unknown_system_no_classification(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="foobar")
        apply_optional_outputs(state, cfg)
        assert "classification" not in state["derived"]

    def test_icd11_placeholder(self):
        state = _core_state()
        cfg = _dict_config(cls_enabled=True, cls_system="icd11")
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert state["derived"]["classification"]["system"] == "ICD-11"
        assert state["derived"]["classification"]["suggestions"] == []


# ══════════════════════════════════════════════════════════════════
# Config format compatibility
# ══════════════════════════════════════════════════════════════════


class TestConfigCompatibility:
    def test_dict_config_format(self):
        cfg = _dict_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" in state["derived"]

    def test_dataclass_config_format(self):
        cfg = _dataclass_config(cls_enabled=True, cls_system="icpc", fhir_enabled=True)
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" in state["derived"]

    def test_missing_export_key_defaults_disabled(self):
        cfg = {"classification": {"enabled": True, "system": "icpc"}}
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" not in state["derived"]

    def test_missing_classification_key_defaults_disabled(self):
        cfg = {"export": {"fhir_enabled": True}}
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" not in state["derived"]
        assert "fhir_bundle" in state["derived"]

    def test_missing_ai_key_defaults_disabled(self):
        assert should_run_ai_overlay({}) is False

    def test_partial_dataclass_config(self):
        """Config with only classification attr (no export/ai)."""
        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.classification = ClassificationConfig(enabled=True, system="icpc")
        state = _core_state()
        apply_optional_outputs(state, cfg)
        assert "classification" in state["derived"]
        assert "fhir_bundle" not in state["derived"]


# ══════════════════════════════════════════════════════════════════
# Integration via build_clinical_state
# ══════════════════════════════════════════════════════════════════


class TestBuildClinicalStateIntegration:
    def test_no_config_no_optional_outputs(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "classification" not in state["derived"]
        assert "fhir_bundle" not in state["derived"]

    def test_config_enables_classification(self):
        cfg = _dataclass_config(cls_enabled=True, cls_system="icpc")
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        assert "classification" in state["derived"]
        assert state["derived"]["classification"]["system"] == "ICPC"

    def test_config_enables_fhir(self):
        cfg = _dataclass_config(fhir_enabled=True)
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        assert "fhir_bundle" in state["derived"]

    def test_config_enables_all(self):
        cfg = _dataclass_config(
            cls_enabled=True, cls_system="icpc", fhir_enabled=True,
        )
        state = build_clinical_state(
            [_seg("patient has headache.")], config=cfg,
        )
        assert "classification" in state["derived"]
        assert "fhir_bundle" in state["derived"]


# ── test utilities ──────────────────────────────────────────────


def _find_resource(bundle: dict, resource_type: str) -> dict | None:
    for entry in bundle.get("entry", []):
        if entry.get("resource", {}).get("resourceType") == resource_type:
            return entry["resource"]
    return None
