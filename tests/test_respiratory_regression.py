"""Regression tests for respiratory symptom extraction and pneumonia scoring.

Validates that the pneumonia seed case and Synthea respiratory cases score
correctly after adding dyspnea canonicalization, pneumonia 2-of-3 rule,
and red flag alignment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extractors import extract_symptoms
from app.diagnostic_hints import generate_diagnostic_hints
from app.red_flag_detector import detect_red_flags
from app.case_system import load_case, validate_case, run_case
from app.case_scoring import score_result_against_ground_truth
from app.case_variations import apply_variation

_PNEUMONIA_PATH = Path(__file__).resolve().parent.parent / "resources" / "cases" / "pneumonia.yaml"


# ── extraction ──────────────────────────────────────────────────────


class TestRespiratoryExtraction:
    def test_shortness_of_breath_extracts_as_dyspnea(self):
        result = extract_symptoms("patient has shortness of breath")
        assert "dyspnea" in result

    def test_short_of_breath_extracts_as_dyspnea(self):
        result = extract_symptoms("Feels more short of breath today.")
        assert "dyspnea" in result

    def test_difficulty_breathing_extracts_as_dyspnea(self):
        result = extract_symptoms("reports difficulty breathing")
        assert "dyspnea" in result

    def test_breathlessness_extracts_as_dyspnea(self):
        result = extract_symptoms("reports breathlessness")
        assert "dyspnea" in result

    def test_dyspnea_literal(self):
        result = extract_symptoms("presenting with dyspnea")
        assert "dyspnea" in result

    def test_full_pneumonia_transcript(self):
        text = (
            "65-year-old male with cough and fever for 3 days. "
            "Feels more short of breath today. "
            "Denies chest pain. No nausea or vomiting."
        )
        symptoms = extract_symptoms(text)
        assert "cough" in symptoms
        assert "fever" in symptoms
        assert "dyspnea" in symptoms


# ── diagnostic hint ─────────────────────────────────────────────────


class TestPneumoniaDiagnosticHint:
    def test_all_three(self):
        hints = generate_diagnostic_hints(["cough", "fever", "dyspnea"])
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions

    def test_cough_and_fever(self):
        hints = generate_diagnostic_hints(["cough", "fever"])
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions

    def test_cough_and_dyspnea(self):
        hints = generate_diagnostic_hints(["cough", "dyspnea"])
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions

    def test_fever_and_dyspnea(self):
        hints = generate_diagnostic_hints(["fever", "dyspnea"])
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions

    def test_single_symptom_does_not_trigger(self):
        for sym in ["cough", "fever", "dyspnea"]:
            hints = generate_diagnostic_hints([sym])
            conditions = [h["condition"] for h in hints]
            assert "Pneumonia" not in conditions, f"should not fire for {sym} alone"

    def test_suppressed_by_negation(self):
        hints = generate_diagnostic_hints(
            ["cough", "fever", "dyspnea"],
            ["No fever", "No dyspnea"],
        )
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" not in conditions


# ── red flag detection ──────────────────────────────────────────────


class TestRespiratoryRedFlags:
    def _state_with_symptoms(self, symptoms: list[str]) -> dict:
        reps = [{"symptom": s} for s in symptoms]
        return {"derived": {"symptom_representations": reps}}

    def test_dyspnea_red_flag(self):
        state = self._state_with_symptoms(["dyspnea"])
        flags = detect_red_flags(state)
        labels = [f["label"] for f in flags]
        assert "Shortness of breath" in labels

    def test_chest_pain_red_flag(self):
        state = self._state_with_symptoms(["chest pain"])
        flags = detect_red_flags(state)
        labels = [f["label"] for f in flags]
        assert "Chest pain" in labels

    def test_no_false_positive(self):
        state = self._state_with_symptoms(["headache"])
        flags = detect_red_flags(state)
        labels = [f["label"] for f in flags]
        assert "Shortness of breath" not in labels
        assert "Chest pain" not in labels


# ── seed case replay ────────────────────────────────────────────────


class TestPneumoniaSeedCase:
    @pytest.fixture()
    def pneumonia_case(self):
        if not _PNEUMONIA_PATH.exists():
            pytest.skip("Pneumonia case file missing")
        return load_case(_PNEUMONIA_PATH)

    def test_valid(self, pneumonia_case):
        assert validate_case(pneumonia_case)["valid"]

    def test_key_findings_present(self, pneumonia_case):
        result = run_case(pneumonia_case)
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert kf["hit_rate"] >= 0.66, f"missing: {kf['missing']}"

    def test_hypothesis_present(self, pneumonia_case):
        result = run_case(pneumonia_case)
        score = score_result_against_ground_truth(result)
        hyp = score["hypotheses"]
        assert "Pneumonia" in hyp["present"]

    def test_overall_score_improvement(self, pneumonia_case):
        result = run_case(pneumonia_case)
        score = score_result_against_ground_truth(result)
        summary = score["summary"]
        composite = (
            summary["hypothesis_hit_rate"]
            + summary["red_flag_hit_rate"]
            + summary["key_finding_hit_rate"]
        ) / 3
        assert composite >= 0.5


# ── variant replay ──────────────────────────────────────────────────


class TestPneumoniaVariant:
    @pytest.fixture()
    def pneumonia_case(self):
        if not _PNEUMONIA_PATH.exists():
            pytest.skip("Pneumonia case file missing")
        return load_case(_PNEUMONIA_PATH)

    def test_add_duration_longer_still_scores(self, pneumonia_case):
        variant = apply_variation(pneumonia_case, "add_duration_longer")
        assert validate_case(variant)["valid"]
        result = run_case(variant)
        score = score_result_against_ground_truth(result)
        assert score["has_ground_truth"]
        hyp = score["hypotheses"]
        assert "Pneumonia" in hyp["present"]
